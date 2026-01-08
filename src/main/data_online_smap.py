# data_online_era5_swe.py
# -*- coding: utf-8 -*-
"""
在线从 1 km 栅格构建 ERA5-Land SWE 反演训练样本
- 动态变量: chelsa_sfxwind (2015+2016)
- 标签: era5land_swe (2016年，每月一个文件，每个波段代表一天)
- 静态变量: 暂时无
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import rasterio
from datetime import datetime, timedelta
from pyproj import Transformer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import calendar

# ============= 配置区域 =============
REGION = "XINGJIANG"  # 根据你的数据调整
YEAR_TARGET = 2016
LAG_DAYS = 36
T = LAG_DAYS + 1  # 365天窗口
P = 5  # patch大小
R = P // 2

MIN_VALID_PIXELS = 100
SAMPLES_PER_DAY = 2000

# 动态变量 - 暂时只有一个
VAR_ORDER = ["chelsa_sfxwind", "pr"]
DYN_YEARS = [2015, 2016]
DYN_VAR_PATHS = {
    "chelsa_sfxwind": "chelsa_sfxwind",
    "pr": "pr_xinjiang"  # 新的降水数据路径
}

# 数据路径
FEATURE_ROOT = Path(r"G:\王扬")
ERA5_ROOT = Path(r"G:\王扬\era5_swe\xinjiang")


# 动态变量路径模板
def dyn_path(var: str, year: int) -> Path:
    """动态变量路径 - 根据变量类型返回不同路径"""
    if var == "chelsa_sfxwind":
        # 原有路径：G:\王扬\chelsa_sfxwind\XINJIANG\2015\
        return FEATURE_ROOT / "chelsa_sfxwind" / "XINJIANG" / str(year)
    elif var == "pr":
        # 新的降水路径：G:\王扬\pr_xinjiang\2015\
        return FEATURE_ROOT / "pr_xinjiang" / str(year)
    else:
        raise ValueError(f"未知的动态变量: {var}")


def load_raster_as_array(path: Path) -> Tuple[np.ndarray, rasterio.Affine, str]:
    """读取栅格为 numpy 数组 (C,H,W)"""
    ds = rasterio.open(path)
    arr = ds.read().astype(np.float32)
    transform = ds.transform
    crs = ds.crs.to_string()
    ds.close()
    return arr, transform, crs


def check_data_files():
    """检查数据文件是否存在"""
    print("=" * 50)
    print("检查数据文件...")

    # 检查动态变量
    print("\n动态变量 (chelsa_sfxwind):")
    for year in DYN_YEARS:
        # 使用glob匹配文件模式
        pattern = f"*_CHELSA_sfcWind_*_{year}_V.2.1.tif"
        search_path = FEATURE_ROOT / "chelsa_sfxwind" / REGION / str(year) / pattern
        files = list(search_path.parent.glob(search_path.name))
        if files:
            print(f"  ✓ {year}: 找到 {len(files)} 个文件")
            for f in files[:3]:  # 显示前3个文件
                print(f"    - {f.name}")
        else:
            print(f"  ✗ {year}: 没有找到匹配 {pattern} 的文件")

    # 检查ERA5 SWE文件
    print("\nERA5 SWE文件:")
    era5_files = list(ERA5_ROOT.glob(f"{REGION}_era5_swe_*.tif"))
    if era5_files:
        print(f"  ✓ 找到 {len(era5_files)} 个文件")
        for f in era5_files[:5]:  # 显示前5个文件
            print(f"    - {f.name}")
    else:
        print(f"  ✗ 在 {ERA5_ROOT} 下没有找到ERA5 SWE文件")

    print("=" * 50)


# 在开始前先检查文件
check_data_files()


class OnlineERASWEDataset(Dataset):
    def __init__(
            self,
            region: str = REGION,
            year_target: int = YEAR_TARGET,
            feature_root: Path = FEATURE_ROOT,
            era5_root: Path = ERA5_ROOT,

            lag_days: int = LAG_DAYS,
            patch_size: int = P,
            min_valid_pixels: int = MIN_VALID_PIXELS,
            samples_per_day: int = SAMPLES_PER_DAY,
    ):
        super().__init__()
        self.region = region
        self.year_target = year_target
        self.feature_root = feature_root
        self.era5_root = era5_root
        self.lag_days = lag_days
        self.patch_size = patch_size
        self.P = patch_size  # 兼容旧代码
        self.R = patch_size // 2
        self.min_valid_pixels = min_valid_pixels
        self.samples_per_day = samples_per_day

        print(f"\n初始化数据集:")
        print(f"  区域: {region}")
        print(f"  目标年份: {year_target}")
        print(f"  动态变量: {VAR_ORDER}")
        print(f"  动态年份: {DYN_YEARS}")

        # 加载数据
        self._load_dynamic_all_years()
        self._load_static_rasters()  # 暂时没有静态变量，创建占位符
        self._build_meta_from_era5()
        self._compute_minmax_sampling(
            sample_fraction=0.01,  # 1%的采样率
            max_files=50  # 最多处理50个SWE文件
        )

        print(f"\n初始化完成!")
        print(f"  总样本数: {len(self.meta_index)}")
        print(f"  时间轴长度: {len(self.all_dates_dt)}")

    def _parse_date_from_filename(self, filename: str) -> datetime:
        """
        从文件名解析日期
        支持格式:
        1. XINGJANG_CHELSA_sfcWind_05_03_2015_V.2.1 (日_月_年)
        2. NDVI_NAQU_2018_1km_daily_EASE2.tif (假设desc是日期)
        3. 其他包含日期的格式
        """
        import re

        # 尝试格式1: XINGJANG_CHELSA_sfcWind_05_03_2015_V.2.1
        # 查找 "数字_数字_数字" 模式 (日_月_年)
        match = re.search(r'(\d{2})_(\d{2})_(\d{4})', filename)
        if match:
            day, month, year = match.groups()
            return datetime(int(year), int(month), int(day))

        # 尝试格式2: 8位连续数字 (YYYYMMDD)
        match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
        if match:
            year, month, day = match.groups()
            return datetime(int(year), int(month), int(day))

        # 尝试格式3: 带分隔符的日期 (YYYY-MM-DD)
        match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', filename)
        if match:
            year, month, day = match.groups()
            return datetime(int(year), int(month), int(day))

        # 如果都失败，可能这个"文件名"实际上已经是日期字符串了
        # 例如: "2018-01-01" 或 "NDVI-2018-01-01"
        try:
            # 尝试直接解析为日期
            return datetime.strptime(filename, "%Y-%m-%d")
        except ValueError:
            pass

        # 尝试去掉变量名前缀
        for var in ["NDVI", "LST", "Alb", "ET", "Pre", "sfcWind", "pr", "tas", "hurs", "rlds", "rsds"]:
            if filename.startswith(f"{var}-"):
                date_str = filename.replace(f"{var}-", "")
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass

        raise ValueError(f"无法从文件名找到日期: {filename}")

    def _build_meta_from_swe(self):
        """
        构建样本索引 - 正确调用方式
        """
        swe_files = sorted(self.swe_root.glob("*.tif"))

        for f in swe_files:  # f 是 Path 对象
            name = f.name  # 获取文件名字符串

            try:
                # 正确：只传入文件名
                date_dt = self._parse_date_from_filename(name)

                # 继续处理...

            except ValueError as e:
                print(f"跳过 {name}: {e}")
                continue

    def _load_dynamic_all_years(self):
        """加载所有动态变量"""
        print(f"\n加载动态变量...")

        self.dyn_data = {}  # var -> (T_all, H, W)
        first_var_processed = False
        first_H, first_W = None, None

        for var_idx, var in enumerate(VAR_ORDER):
            print(f"\n[{var_idx + 1}/{len(VAR_ORDER)}] 加载变量: {var}")

            # 收集所有文件
            all_files = []
            for year in DYN_YEARS:
                var_dir = dyn_path(var, year)
                if not var_dir.exists():
                    raise FileNotFoundError(f"{var} {year} 目录不存在: {var_dir}")

                # 获取所有.tif文件
                files = sorted(list(var_dir.glob("*.tif")))
                if not files:
                    raise FileNotFoundError(f"在 {var_dir} 下没有找到.tif文件")

                print(f"  {year}: 找到 {len(files)} 个文件")
                all_files.extend(files)

            print(f"  总共找到 {len(all_files)} 个{var}文件")

            # 读取第一个文件获取尺寸信息
            with rasterio.open(all_files[0]) as ds:
                first_data = ds.read(1).astype(np.float32)
                H, W = first_data.shape

                # 如果是第一个变量，保存transform和CRS
                if not first_var_processed:
                    self.transform = ds.transform
                    self.crs_proj = ds.crs.to_string()
                    first_H, first_W = H, W
                    first_var_processed = True
                    print(f"  参考尺寸: H={H}, W={W}")
                else:
                    # 检查后续变量尺寸是否一致
                    if H != first_H or W != first_W:
                        print(f"  警告: {var} 尺寸不匹配: ({H},{W}) vs 参考({first_H},{first_W})")
                        # 可能需要调整尺寸，这里先记录

            # 解析日期并排序文件（复用原有逻辑）
            dated_files = []
            for f in all_files:
                # 解析文件名中的日期，例如: XINGJIANG_CHELSA_pr_01_01_2016_V.2.1.tif
                name = f.stem
                dt = self._parse_date_from_filename(name)
                if dt:
                    dated_files.append((dt, f))

            if not dated_files:
                raise ValueError(f"没有成功解析{var}文件的日期信息")

            # 按日期排序
            dated_files.sort(key=lambda x: x[0])

            # 创建该变量的时间轴
            var_dates_dt = [dt for dt, _ in dated_files]

            # 检查日期连续性（只对第一个变量详细检查）
            if var == VAR_ORDER[0]:
                self.all_dates_dt = var_dates_dt
                self.T_all = len(self.all_dates_dt)
                self.H = H
                self.W = W
            else:
                # 检查其他变量是否与第一个变量日期一致
                if len(var_dates_dt) != len(self.all_dates_dt):
                    print(f"  警告: {var} 天数 {len(var_dates_dt)} 与参考 {len(self.all_dates_dt)} 不一致")

            # 创建该变量的数据数组 (T_all, H, W)
            var_arr = np.zeros((len(var_dates_dt), H, W), dtype=np.float32)

            print(f"  读取数据...")
            for i, (dt, f) in enumerate(dated_files):
                try:
                    with rasterio.open(f) as ds:
                        data = ds.read(1).astype(np.float32)

                    # 检查并调整尺寸
                    if data.shape != (H, W):
                        print(f"  警告: 文件 {f.name} 尺寸不匹配: {data.shape}")
                        # 简单调整
                        if data.shape[0] >= H and data.shape[1] >= W:
                            data = data[:H, :W]
                        else:
                            temp = np.zeros((H, W), dtype=np.float32)
                            h_min = min(H, data.shape[0])
                            w_min = min(W, data.shape[1])
                            temp[:h_min, :w_min] = data[:h_min, :w_min]
                            data = temp

                    var_arr[i] = data

                    if (i + 1) % 100 == 0:
                        print(f"    已读取 {i + 1}/{len(dated_files)} 个文件")

                except Exception as e:
                    print(f"  读取 {f.name} 失败: {e}")
                    var_arr[i] = np.zeros((H, W), dtype=np.float32)

            # 过滤掉每年的最后4天
            if var == VAR_ORDER[0]:  # 只对第一个变量应用过滤
                valid_idx = []
                for i, d in enumerate(var_dates_dt):
                    if d.month == 12 and d.day >= 28:
                        continue
                    valid_idx.append(i)

                # 应用过滤
                self.all_dates_dt = [var_dates_dt[i] for i in valid_idx]
                self.T_all = len(self.all_dates_dt)

                # 过滤第一个变量
                self.dyn_data[var] = var_arr[valid_idx]
            else:
                # 其他变量使用相同的过滤索引
                self.dyn_data[var] = var_arr

            print(f"  {var} 数据形状: {self.dyn_data[var].shape}")

        # 构建日期到索引的映射
        self.date_to_index = {d: i for i, d in enumerate(self.all_dates_dt)}

        print(f"\n所有动态变量加载完成!")
        print(f"  过滤后时间轴: {self.T_all} 天")
        print(f"  起始: {self.all_dates_dt[0].strftime('%Y-%m-%d')}")
        print(f"  结束: {self.all_dates_dt[-1].strftime('%Y-%m-%d')}")
        print(f"  变量数量: {len(self.dyn_data)}")
        for var, arr in self.dyn_data.items():
            print(f"    {var}: {arr.shape}")

    def _load_static_rasters(self):
        """加载静态变量 - 暂时没有，创建占位符"""
        print(f"\n加载静态变量...")
        print(f"  注意: 当前没有静态变量，使用全零数组作为占位符")

        # 创建全零的静态变量数组
        self.s2_arr = np.zeros((6, self.H, self.W), dtype=np.float32)
        self.soil_arr = np.zeros((7, self.H, self.W), dtype=np.float32)
        self.terrain_arr = np.zeros((8, self.H, self.W), dtype=np.float32)
        self.lc_arr = np.zeros((1, self.H, self.W), dtype=np.float32)
        self.kop_arr = np.zeros((1, self.H, self.W), dtype=np.float32)

        # 坐标系转换器
        self.transformer = Transformer.from_crs(self.crs_proj, "EPSG:4326", always_xy=True)

    def _build_meta_from_era5(self):
        """从ERA5 SWE数据构建样本索引"""
        print(f"\n从ERA5 SWE数据构建样本索引...")
        print(f"  ERA5目录: {self.era5_root}")

        self.meta_index: List[Tuple[int, int, int]] = []
        self.era5_grids: Dict[int, np.ndarray] = {}

        # 确保有cv2
        try:
            import cv2
            HAS_CV2 = True
        except ImportError:
            HAS_CV2 = False
            print("  警告: 没有安装OpenCV，将使用简单裁剪/填充")

        # 查找ERA5 SWE文件
        pattern = "XINGJIANG_era5_swe_*.tif"
        era5_files = sorted(list(self.era5_root.glob(pattern)))
        print(f"  找到 {len(era5_files)} 个ERA5 SWE文件")

        if not era5_files:
            raise FileNotFoundError(f"在 {self.era5_root} 下没有找到匹配 {pattern} 的文件")

        # 处理每个月的文件
        for month_file in era5_files:
            print(f"\n  处理文件: {month_file.name}")

            # 从文件名解析月份: XINGJIANG_era5_swe_201501.tif
            name = month_file.stem  # 去掉.tif
            parts = name.split('_')
            year_month = None
            for part in parts:
                if part.isdigit() and len(part) == 6:  # YYYYMM格式
                    year_month = part
                    break

            if year_month is None:
                print(f"    跳过: 无法解析文件名 {name}")
                continue

            year = int(year_month[:4])
            month = int(year_month[4:])

            if year != self.year_target:
                print(f"    跳过: 不是目标年份 {self.year_target}")
                continue

            try:
                with rasterio.open(month_file) as ds:
                    # 读取所有波段
                    swe_data = ds.read().astype(np.float32)  # (n_bands, H, W)
                    swe_nodata = ds.nodata

                    # ========= 在这里调整尺寸 =========
                    if swe_data.shape[1:] != (self.H, self.W):
                        print(f"    调整ERA5尺寸: {swe_data.shape[1:]} -> ({self.H}, {self.W})")

                        if HAS_CV2:
                            # 使用OpenCV调整尺寸
                            resized_data = []
                            for i in range(swe_data.shape[0]):
                                band = swe_data[i]
                                resized_band = cv2.resize(band, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                                resized_data.append(resized_band)
                            swe_data = np.stack(resized_data, axis=0)
                        else:
                            # 简单的裁剪或填充
                            if swe_data.shape[1] >= self.H and swe_data.shape[2] >= self.W:
                                swe_data = swe_data[:, :self.H, :self.W]
                            else:
                                # 填充
                                temp = np.zeros((swe_data.shape[0], self.H, self.W), dtype=np.float32)
                                h_min = min(self.H, swe_data.shape[1])
                                w_min = min(self.W, swe_data.shape[2])
                                temp[:, :h_min, :w_min] = swe_data[:, :h_min, :w_min]
                                swe_data = temp
                        print(f"    调整后尺寸: {swe_data.shape}")
                    else:
                        print(f"    尺寸匹配: {swe_data.shape[1:]} == ({self.H}, {self.W})")

                # 获取该月的天数
                month_days = calendar.monthrange(year, month)[1]

                # 检查波段数是否匹配
                if swe_data.shape[0] != month_days:
                    print(f"    警告: 波段数 {swe_data.shape[0]} 不等于该月天数 {month_days}")
                    # 只使用前month_days个波段
                    swe_data = swe_data[:month_days]

                # 处理每个波段（每天）
                for day in range(1, swe_data.shape[0] + 1):
                    try:
                        date_dt = datetime(year, month, day)

                        # 简化：直接跳过不在动态时间轴中的日期
                        if date_dt not in self.date_to_index:
                            # 不再尝试找替代日期，直接跳过
                            continue

                        d_idx = self.date_to_index[date_dt]

                        # 检查是否有足够的历史数据
                        if d_idx < self.lag_days:
                            # 不打印，减少输出
                            continue

                        # 获取当天的SWE数据
                        swe_day = swe_data[day - 1]  # 0-based索引

                        # 检查尺寸（现在应该已经调整过了）
                        if swe_day.shape != (self.H, self.W):
                            print(f"    警告: 尺寸仍然不匹配 {swe_day.shape} vs ({self.H},{self.W})")
                            continue

                        # 有效像元掩码
                        if swe_nodata is not None:
                            valid_mask = (swe_day != swe_nodata) & np.isfinite(swe_day)
                        else:
                            valid_mask = np.isfinite(swe_day)

                        valid_pixels = np.count_nonzero(valid_mask)

                        if valid_pixels < self.min_valid_pixels:
                            continue

                        # 边缘裁剪后的候选像元
                        candidate_indices = np.argwhere(valid_mask)
                        candidate_indices = [
                            (r, c) for (r, c) in candidate_indices
                            if (self.R <= r < self.H - self.R) and (self.R <= c < self.W - self.R)
                        ]

                        if not candidate_indices:
                            continue

                        # 随机采样
                        np.random.shuffle(candidate_indices)
                        if self.samples_per_day is not None:
                            candidate_indices = candidate_indices[:self.samples_per_day]

                        # 保存SWE网格和索引
                        self.era5_grids[d_idx] = swe_day
                        for (r, c) in candidate_indices:
                            self.meta_index.append((d_idx, int(r), int(c)))

                        print(f"    {date_dt.strftime('%Y-%m-%d')}: 添加 {len(candidate_indices)} 个样本")

                    except Exception as e:
                        # 不打印每个日期的错误，减少输出
                        continue

            except Exception as e:
                print(f"    读取文件 {month_file.name} 失败: {e}")
                continue

        print(f"\n总样本数: {len(self.meta_index)}")
        print(f"动态变量参考尺寸: H={self.H}, W={self.W}")

        # 如果样本数为0，至少创建一个虚拟样本以便调试
        if len(self.meta_index) == 0:
            print("警告: 没有构建任何样本，检查日期匹配和尺寸问题")
            # 可以添加一些虚拟样本用于测试
            # self._add_debug_samples()

    def _compute_minmax_sampling(self, sample_fraction=0.01, max_files=50):
        """
        通过采样计算所有特征的min/max
        参数:
            sample_fraction: 采样比例 (0.01 = 1%)
            max_files: 最多处理的文件数
        """
        print(f"\n[Norm] 采样计算Min-Max (采样率: {sample_fraction * 100:.1f}%, 最多{max_files}个文件)...")

        # ========== 1. 动态变量 (从内存数据采样) ==========
        dyn_min_list = []
        dyn_max_list = []

        for var in VAR_ORDER:
            arr = self.dyn_data[var]  # (T_all_filt, H, W)

            # 采样策略：时间维度全取，空间维度采样
            T, H, W = arr.shape

            # 如果数据太大，进行采样
            if H * W > 1000000:  # 超过100万像素
                # 随机采样空间位置
                n_samples = min(100000, H * W)  # 最多10万个像素
                flat_indices = np.random.choice(H * W, n_samples, replace=False)
                rows = flat_indices // W
                cols = flat_indices % W

                # 提取采样数据
                sampled_data = arr[:, rows, cols].flatten()
            else:
                sampled_data = arr.flatten()

            # 移除NaN
            valid_data = sampled_data[np.isfinite(sampled_data)]

            if len(valid_data) > 0:
                dyn_min_list.append(float(np.min(valid_data)))
                dyn_max_list.append(float(np.max(valid_data)))
            else:
                print(f"  警告: {var} 无有效数据，使用默认值")
                dyn_min_list.append(0.0)
                dyn_max_list.append(1.0)

            print(f"  {var}: 基于 {len(valid_data):,} 个样本")

        dyn_min = np.array(dyn_min_list, dtype=np.float32)
        dyn_max = np.array(dyn_max_list, dtype=np.float32)
        self.C_dyn = len(VAR_ORDER)

        # ========== 2. 静态特征 (从内存数据采样) ==========
        print("\n  计算静态特征统计量...")

        def compute_static_stats(arr, name):
            """计算静态特征的统计量"""
            C, H, W = arr.shape

            # 如果数据太大，采样
            if H * W > 500000:
                # 每通道采样一定数量的像素
                n_samples_per_channel = min(50000, H * W)
                sampled_mins = []
                sampled_maxs = []

                for c in range(C):
                    # 随机采样像素
                    flat_indices = np.random.choice(H * W, n_samples_per_channel, replace=False)
                    rows = flat_indices // W
                    cols = flat_indices % W

                    channel_data = arr[c, rows, cols]
                    valid_data = channel_data[np.isfinite(channel_data)]

                    if len(valid_data) > 0:
                        sampled_mins.append(np.min(valid_data))
                        sampled_maxs.append(np.max(valid_data))
                    else:
                        sampled_mins.append(0.0)
                        sampled_maxs.append(1.0)

                return np.array(sampled_mins, dtype=np.float32), np.array(sampled_maxs, dtype=np.float32)
            else:
                # 小数据直接计算
                mins = np.zeros(C, dtype=np.float32)
                maxs = np.zeros(C, dtype=np.float32)

                for c in range(C):
                    channel_data = arr[c].flatten()
                    valid_data = channel_data[np.isfinite(channel_data)]

                    if len(valid_data) > 0:
                        mins[c] = np.min(valid_data)
                        maxs[c] = np.max(valid_data)
                    else:
                        mins[c] = 0.0
                        maxs[c] = 1.0

                return mins, maxs

        # 计算各静态特征
        s2_min, s2_max = compute_static_stats(self.s2_arr, "S2")
        soil_min, soil_max = compute_static_stats(self.soil_arr, "Soil")
        terrain_min, terrain_max = compute_static_stats(self.terrain_arr, "Terrain")

        # 单波段特征
        lc_min = np.array([np.nanmin(self.lc_arr)], dtype=np.float32)
        lc_max = np.array([np.nanmax(self.lc_arr)], dtype=np.float32)

        kop_min = np.array([np.nanmin(self.kop_arr)], dtype=np.float32)
        kop_max = np.array([np.nanmax(self.kop_arr)], dtype=np.float32)

        # ========== 3. SWE数据 (从文件采样) ==========
        print("\n  采样计算SWE统计量...")

        # 获取SWE文件列表
        swe_files = sorted(self.era5_root.glob("*.tif"))

        if not swe_files:
            print("  警告: 未找到SWE文件，使用默认值")
            swe_min, swe_max = 0.0, 100.0
        else:
            # 限制处理的文件数量
            n_files = min(len(swe_files), max_files)
            sample_files = list(swe_files)

            if n_files < len(swe_files):
                # 随机选择文件
                import random
                sample_files = random.sample(swe_files, n_files)

            swe_samples = []
            processed_files = 0

            for f in sample_files:
                try:
                    with rasterio.open(f) as ds:
                        swe_data = ds.read(1).astype(np.float32)
                        swe_nodata = ds.nodata

                    H, W = swe_data.shape

                    # 处理无效值
                    if swe_nodata is not None:
                        valid_mask = (swe_data != swe_nodata) & np.isfinite(swe_data)
                    else:
                        valid_mask = np.isfinite(swe_data)

                    valid_data = swe_data[valid_mask]

                    if len(valid_data) > 0:
                        # 从有效数据中采样
                        n_samples = max(1, int(len(valid_data) * sample_fraction))

                        if n_samples < len(valid_data):
                            idx = np.random.choice(len(valid_data), n_samples, replace=False)
                            swe_samples.extend(valid_data[idx])
                        else:
                            swe_samples.extend(valid_data)

                    processed_files += 1

                    # 每处理10个文件打印进度
                    if processed_files % 10 == 0:
                        print(f"    已处理 {processed_files}/{n_files} 个文件，采集 {len(swe_samples):,} 个样本")

                        # 如果样本足够多，可以提前停止
                        if len(swe_samples) > 100000:
                            print(f"    已采集足够样本 ({len(swe_samples):,})，提前停止")
                            break

                except Exception as e:
                    print(f"    警告: 处理 {f.name} 时出错: {e}")
                    continue

            swe_samples = np.array(swe_samples)

            if len(swe_samples) > 0:
                swe_min = float(np.min(swe_samples))
                swe_max = float(np.max(swe_samples))
                swe_mean = float(np.mean(swe_samples))
                swe_std = float(np.std(swe_samples))

                print(f"  SWE统计: 样本数={len(swe_samples):,}, 范围=[{swe_min:.3f}, {swe_max:.3f}]")
                print(f"           均值={swe_mean:.3f}, 标准差={swe_std:.3f}")

                # 检查异常值
                q1 = np.percentile(swe_samples, 1)
                q99 = np.percentile(swe_samples, 99)
                print(f"           1%分位数={q1:.3f}, 99%分位数={q99:.3f}")

                # 可选：使用分位数范围避免异常值
                # swe_min, swe_max = q1, q99
            else:
                print("  警告: 未采集到有效SWE样本，使用默认值")
                swe_min, swe_max = 0.0, 100.0

        # ========== 4. 组合空间特征 ==========
        # spatial: S2(6)+Soil(7)+Terrain(8)+LC(1)+Kop(1)+dyn(5)
        spatial_min = np.concatenate(
            [s2_min, soil_min, terrain_min, lc_min, kop_min, dyn_min], axis=0
        ).astype(np.float32)
        spatial_max = np.concatenate(
            [s2_max, soil_max, terrain_max, lc_max, kop_max, dyn_max], axis=0
        ).astype(np.float32)

        # ========== 5. 组合点特征 ==========
        # point: Soil(7)+Terrain(8)+S2(6)+LC(1)+Kop(1)+dyn(5)+time(6)
        time_min = np.full(6, -1.0, dtype=np.float32)
        time_max = np.full(6, 1.0, dtype=np.float32)

        point_min = np.concatenate(
            [soil_min, terrain_min, s2_min, lc_min, kop_min, dyn_min, time_min],
            axis=0
        ).astype(np.float32)
        point_max = np.concatenate(
            [soil_max, terrain_max, s2_max, lc_max, kop_max, dyn_max, time_max],
            axis=0
        ).astype(np.float32)

        # ========== 6. 保存结果 ==========
        eps = 1e-6
        self.eps = eps

        # SWE单独保存（如果你需要）
        self.swe_min = swe_min
        self.swe_max = swe_max + eps

        # 动态特征
        self.min_dyn_t = torch.from_numpy(dyn_min).view(-1, 1, 1, 1)
        self.max_dyn_t = torch.from_numpy(dyn_max + eps).view(-1, 1, 1, 1)

        # 空间特征
        self.min_spatial_t = torch.from_numpy(spatial_min).view(-1, 1, 1)
        self.max_spatial_t = torch.from_numpy(spatial_max + eps).view(-1, 1, 1)

        # 点特征
        self.min_point_t = torch.from_numpy(point_min)
        self.max_point_t = torch.from_numpy(point_max + eps)

        self.C_spatial = spatial_min.shape[0]
        self.C_point = point_min.shape[0]

        print(f"\n  统计量计算完成:")
        print(f"    dyn_min shape: {dyn_min.shape}, 范围: [{dyn_min.min():.3f}, {dyn_max.max():.3f}]")
        print(f"    spatial_min shape: {spatial_min.shape}")
        print(f"    point_min shape: {point_min.shape}")
        print(f"    SWE范围: [{swe_min:.3f}, {swe_max:.3f}]")

        # 打印各特征范围（用于验证）
        print("\n  各特征范围概览:")
        feature_names = ["S2", "Soil", "Terrain", "LC", "Kop", "Dynamic"]
        starts = [0, 6, 13, 21, 22, 23]
        ends = [6, 13, 21, 22, 23, 28]

        for name, start, end in zip(feature_names, starts, ends):
            if start < len(spatial_min):
                actual_end = min(end, len(spatial_min))
                min_vals = spatial_min[start:actual_end]
                max_vals = spatial_max[start:actual_end]
                print(f"    {name}: [{min_vals.min():.3f}, {max_vals.max():.3f}]")

    def _pixel_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        """将像素坐标转换为经纬度"""
        x, y = self.transform * (col + 0.5, row + 0.5)
        lon, lat = self.transformer.transform(x, y)
        return lon, lat

    def _build_sample_numpy(self, d_idx: int, r: int, c: int):
        """用numpy构建一个样本"""
        # 检查索引范围
        if d_idx < self.lag_days or d_idx >= self.T_all:
            return None

        # --- A. 动态时序 cube ---
        dyn_cube = np.zeros((1, T, self.P, self.P), dtype=np.float32)  # (1, T, P, P)
        start_idx = d_idx - self.lag_days
        end_idx = d_idx

        dyn_arr = self.dyn_data["chelsa_sfxwind"]  # (T_all, H, W)

        # 提取时间窗口
        time_slice = dyn_arr[start_idx:end_idx + 1]  # (T, H, W)

        # 提取空间窗口
        r0, r1 = r - self.R, r + self.R + 1
        c0, c1 = c - self.R, c + self.R + 1

        # 检查边界
        if r0 < 0 or r1 > self.H or c0 < 0 or c1 > self.W:
            return None

        patch = time_slice[:, r0:r1, c0:c1]  # (T, P, P)
        dyn_cube[0] = patch  # 放入(1, T, P, P)

        # --- B. 空间 patch (静态变量为占位符) ---
        # 静态变量通道数
        static_channels = 6 + 7 + 8 + 1 + 1  # S2+Soil+Terrain+LC+Kop
        dyn_channels = 1  # chelsa_sfxwind

        spatial_patch = np.zeros((static_channels + dyn_channels, self.P, self.P), dtype=np.float32)

        # 将动态变量的最后一天放入空间patch
        spatial_patch[-dyn_channels:] = dyn_cube[:, -1, :, :]  # (1, P, P)

        # --- C. 点特征 ---
        # 静态部分 (6+7+8+1+1=23维)
        static_point = np.zeros(23, dtype=np.float32)

        # 动态部分 (中心点最后一天的值)
        center_dyn = dyn_cube[:, -1, self.R, self.R]  # (1,)

        # 时间特征
        date_dt = self.all_dates_dt[d_idx]
        doy = date_dt.timetuple().tm_yday
        doy_sin = np.sin(2 * np.pi * doy / 365.0)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)

        lon, lat = self._pixel_to_lonlat(r, c)
        lon_sin = np.sin(np.deg2rad(lon))
        lon_cos = np.cos(np.deg2rad(lon))
        lat_sin = np.sin(np.deg2rad(lat))
        lat_cos = np.cos(np.deg2rad(lat))

        time_feats = np.array([doy_sin, doy_cos, lon_sin, lon_cos, lat_sin, lat_cos], dtype=np.float32)

        # 合并点特征
        point_feats = np.concatenate([static_point, center_dyn, time_feats], axis=0).astype(np.float32)

        # --- D. SWE 标签 ---
        if d_idx not in self.era5_grids:
            return None

        swe_grid = self.era5_grids[d_idx]
        y = float(swe_grid[r, c])

        # 检查NaN/Inf
        if (not np.isfinite(dyn_cube).all() or
                not np.isfinite(spatial_patch).all() or
                not np.isfinite(point_feats).all() or
                not np.isfinite(y)):
            return None

        return dyn_cube, spatial_patch, point_feats, y

    def __len__(self):
        return len(self.meta_index)

    def __getitem__(self, idx: int):
        """获取一个样本"""
        max_retry = 10
        cur_idx = idx

        for _ in range(max_retry):
            d_idx, r, c = self.meta_index[cur_idx]
            sample = self._build_sample_numpy(d_idx, r, c)

            if sample is not None:
                dyn_cube, spatial_patch, point_feats, y = sample

                # 转换为torch张量
                dyn_t = torch.from_numpy(dyn_cube)
                spatial_t = torch.from_numpy(spatial_patch)
                point_t = torch.from_numpy(point_feats)
                y_t = torch.tensor(y, dtype=torch.float32)

                # Min-Max标准化
                dyn_t = (dyn_t - self.min_dyn_t) / (self.max_dyn_t - self.min_dyn_t)
                spatial_t = (spatial_t - self.min_spatial_t) / (self.max_spatial_t - self.min_spatial_t)
                point_t = (point_t - self.min_point_t) / (self.max_point_t - self.min_point_t)
                y_t = (y_t - self.swe_min) / (self.swe_max - self.swe_min)

                return dyn_t, spatial_t, point_t, y_t

            # 当前样本无效，尝试下一个
            cur_idx = (cur_idx + 1) % len(self.meta_index)

        raise IndexError(f"在idx={idx}附近连续{max_retry}个样本均无效")




def build_dataloaders(
        batch_size: int = 32,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
):
    """构建数据加载器"""
    try:
        dataset = OnlineERASWEDataset()

        n_total = len(dataset)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        print(f"\n[DataLoader] 样本总数: {n_total}, train={n_train}, val={n_val}")

        if n_total == 0:
            raise ValueError("数据集为空，无法创建数据加载器")

        # 划分训练集和验证集
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # 获取一个样本来检查维度
        if n_train > 0:
            try:
                sample_dyn, sample_spatial, sample_point, _ = next(iter(train_loader))
                C_dyn, T_real, P_real, _ = sample_dyn.shape[1:]
                C_spatial = sample_spatial.shape[1]
                C_point = sample_point.shape[1]

                print(f"\n[DataLoader] 推断出来的维度:")
                print(f"  C_dyn={C_dyn}, T={T_real}, P={P_real}")
                print(f"  C_spatial={C_spatial}, C_point={C_point}")

                return train_loader, val_loader, (C_dyn, T_real, P_real, C_spatial, C_point)
            except Exception as e:
                print(f"获取样本维度失败: {e}")
                # 返回默认维度
                return train_loader, val_loader, (1, T, P, 24, 30)  # 估计的维度

        return train_loader, val_loader, (1, T, P, 24, 30)

    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("=" * 50)
    print("开始测试数据加载...")
    print("=" * 50)

    try:
        # 创建数据集
        print("\n1. 创建数据集...")
        dataset = OnlineERASWEDataset()

        # 测试获取样本
        print(f"\n2. 测试获取样本...")
        if len(dataset) > 0:
            for i in range(min(3, len(dataset))):
                print(f"\n  样本 {i}:")
                try:
                    dyn, spatial, point, y = dataset[i]
                    print(f"    dyn shape: {dyn.shape}")
                    print(f"    spatial shape: {spatial.shape}")
                    print(f"    point shape: {point.shape}")
                    print(f"    y value: {y.item():.4f}")
                except Exception as e:
                    print(f"    获取样本{i}失败: {e}")
        else:
            print("  数据集为空!")

        # 测试数据加载器
        print(f"\n3. 测试数据加载器...")
        train_loader, val_loader, shapes = build_dataloaders(batch_size=4, val_ratio=0.2)

        print(f"\n4. 测试批次加载...")
        if train_loader:
            batch = next(iter(train_loader))
            print(f"   批次大小: {len(batch)}")
            for i, data in enumerate(batch):
                print(f"   数据{i}: {type(data)}, shape: {data.shape}")

        print(f"\n✓ 数据加载测试完成!")

    except Exception as e:
        print(f"\n✗ 数据加载测试失败: {e}")
        import traceback

        traceback.print_exc()