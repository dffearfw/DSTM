# data_online_era5_swe.py
# -*- coding: utf-8 -*-
"""
在线从栅格构建 SWE 反演训练样本
- 卷积特征: chelsa_sfxwind, lst, rh, clamday, dem
- 点特征: ls, 经纬度, doy
- 标签: fusedSWE
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
import re

# ============= 配置区域 =============
REGION = "XINJIANG"
YEAR_TARGET = 2016
PATCH_SIZE = 5  # patch大小
R = PATCH_SIZE // 2

MIN_VALID_PIXELS = 100
SAMPLES_PER_DAY = 2000

# 卷积特征（参与卷积的变量）
CONV_VARS = ["chelsa_sfxwind", "lst", "rh"]
CONV_STATIC_VARS = ["clamday", "dem"]  # 静态卷积特征

# 点特征（不参与卷积）
POINT_VARS = ["ls"]  # 点特征变量

# 数据路径
FEATURE_ROOT = Path(r"G:\王扬")
LABEL_ROOT = Path(r"G:\王扬\fusedSWE\XINJIANG")  # 标签路径

# 卷积变量路径模板
def conv_var_path(var: str, year: int) -> Path:
    """卷积变量路径"""
    if var == "chelsa_sfxwind":
        return FEATURE_ROOT / "chelsa_sfxwind" / "XINJIANG" / "resap25km"
    elif var == "lst":
        return FEATURE_ROOT / "lst" / f"ERA5_Xinjiang_{year}_025deg"
    elif var == "rh":
        return FEATURE_ROOT / "rh" / f"ERA5_RH_Xinjiang_{year}_025deg"
    else:
        raise ValueError(f"未知的卷积变量: {var}")

# 静态卷积变量路径
def conv_static_path(var: str, year: int, threshold: float = 0.5) -> List[Path]:
    """静态卷积变量路径"""
    if var == "clamday":
        clamday_path = FEATURE_ROOT / "clamday" / "XINJIANG"
        pattern = f"*{year}*threshold{threshold}*.tif"
        return list(clamday_path.glob(pattern))
    elif var == "dem":
        dem_path = FEATURE_ROOT / "dem" / "XINJIANG"
        return [
            list(dem_path.glob("*_Terrain_*.tif"))[0],
            list(dem_path.glob("*_Terrain_StdDev_*.tif"))[0]
        ]
    else:
        raise ValueError(f"未知的静态卷积变量: {var}")

# 点变量路径
def point_var_path(var: str, year: int) -> Path:
    """点变量路径"""
    if var == "ls":
        ls_path = FEATURE_ROOT / "ls" / "XINJIANG"
        pattern = f"*{year}*Median*.tif"
        files = list(ls_path.glob(pattern))
        return files[0] if files else None
    else:
        raise ValueError(f"未知的点变量: {var}")

class SWEDataset(Dataset):
    def __init__(
            self,
            region: str = REGION,
            year_target: int = YEAR_TARGET,
            feature_root: Path = FEATURE_ROOT,
            label_root: Path = LABEL_ROOT,
            patch_size: int = PATCH_SIZE,
            min_valid_pixels: int = MIN_VALID_PIXELS,
            samples_per_day: int = SAMPLES_PER_DAY,
            clamday_threshold: float = 0.5,
    ):
        super().__init__()
        self.region = region
        self.year_target = year_target
        self.feature_root = feature_root
        self.label_root = label_root
        self.patch_size = patch_size
        self.P = patch_size
        self.R = patch_size // 2
        self.min_valid_pixels = min_valid_pixels
        self.samples_per_day = samples_per_day
        self.clamday_threshold = clamday_threshold

        print(f"\n初始化数据集:")
        print(f"  区域: {region}")
        print(f"  目标年份: {year_target}")
        print(f"  卷积特征: {CONV_VARS + CONV_STATIC_VARS}")
        print(f"  点特征: {POINT_VARS}")
        print(f"  Clamday阈值: {clamday_threshold}")

        self.clamday_data = None
        self.dem_data = None

        # 加载所有数据
        self._setup_unified_grid()
        self._load_data_unified()
        self._build_sample_index()
        self._compute_minmax_sampling()

        print(f"\n初始化完成!")
        print(f"  总样本数: {len(self.meta_index)}")
        print(f"  卷积特征维度: {self.C_conv}")
        print(f"  点特征维度: {self.C_point}")

    def _setup_unified_grid(self):
        """设置统一网格"""
        print(f"\n设置统一网格...")

        # 使用标签数据的网格作为参考
        label_files = list(self.label_root.glob("*.tif"))
        if not label_files:
            raise FileNotFoundError("找不到标签文件")

        with rasterio.open(label_files[0]) as ds:
            self.common_bounds = ds.bounds
            self.transform = ds.transform
            self.crs_proj = ds.crs.to_string()
            self.H, self.W = ds.shape

        print(f"参考网格（使用标签数据）:")
        print(f"  范围: {self.common_bounds}")
        print(f"  尺寸: {self.H}行 × {self.W}列")
        print(f"  分辨率: {abs(self.transform.a):.3f}° × {abs(self.transform.e):.3f}°")

        # 创建坐标系转换器
        self.transformer = Transformer.from_crs(self.crs_proj, "EPSG:4326", always_xy=True)

    def _load_data_unified(self):
        """加载所有数据（统一到公共区域）"""
        # 加载卷积特征
        self._load_conv_data_unified()
        # 加载点特征
        self._load_point_data_unified()
        # 加载标签
        self._load_labels_unified()

    def _load_conv_data_unified(self):
        """加载卷积特征数据（统一到公共区域）"""
        print(f"\n加载卷积特征数据...")

        self.conv_dyn_data = {}
        self.all_dates = []

        for var_idx, var in enumerate(CONV_VARS):
            print(f"\n[{var_idx + 1}/{len(CONV_VARS)}] 处理变量: {var}")

            # 加载原始数据（现在返回4个值）
            var_data, var_dates, src_bounds, src_transform = self._load_single_variable(var)
            if var_data is None:
                continue

            # 统一到公共区域（现在传递bounds和transform）
            var_data_unified = self._unify_to_common_grid(
                var_data, var, src_bounds, src_transform
            )

            # 如果是第一个变量，设置时间轴
            if var == CONV_VARS[0]:
                self.all_dates = var_dates
                self.date_to_index = {d: i for i, d in enumerate(var_dates)}

            self.conv_dyn_data[var] = var_data_unified
            print(f"  {var} 数据形状: {var_data_unified.shape}")

    def _load_single_variable(self, var: str):
        """加载单个变量的数据，返回数据和元数据"""
        print(f"  加载 {var} 数据...")

        # 收集两年数据
        all_files = []
        for year in [self.year_target - 1, self.year_target]:
            var_dir = conv_var_path(var, year)
            if not var_dir.exists():
                print(f"    {year}: 目录不存在，跳过")
                continue

            if var == "chelsa_sfxwind":
                files = list(var_dir.glob(f"*{year}*.tif"))
            else:
                files = list(var_dir.glob("*.tif"))

            if not files:
                print(f"    {year}: 没有.tif文件，跳过")
                continue

            print(f"    {year}: 找到 {len(files)} 个文件")
            all_files.extend(files)

        if not all_files:
            print(f"  未找到 {var} 文件")
            return None, [], None, None  # 返回None

        # 获取第一个文件的bounds和transform
        with rasterio.open(all_files[0]) as ds:
            src_bounds = ds.bounds
            src_transform = ds.transform

        # 处理月份文件（lst和rh）
        if var in ["lst", "rh"]:
            monthly_data = {}
            for f in all_files:
                try:
                    name = f.stem
                    match = re.search(r'(\d{4})(\d{2})', name)
                    if not match:
                        continue

                    year = int(match.group(1))
                    month = int(match.group(2))

                    with rasterio.open(f) as ds:
                        data = ds.read()  # (n_bands, H, W)
                        n_bands = data.shape[0]

                        month_days = calendar.monthrange(year, month)[1]
                        n_bands = min(n_bands, month_days)

                        for day in range(1, n_bands + 1):
                            date_dt = datetime(year, month, day)
                            band_data = data[day - 1].astype(np.float32)
                            monthly_data[date_dt] = band_data

                except Exception as e:
                    print(f"    处理文件 {f.name} 失败: {e}")
                    continue

            sorted_dates = sorted(monthly_data.keys())
            var_arr = np.stack([monthly_data[dt] for dt in sorted_dates], axis=0)
            return var_arr, sorted_dates, src_bounds, src_transform

        else:  # chelsa_sfxwind
            dated_files = []
            for f in all_files:
                try:
                    name = f.stem
                    dt = self._parse_date_from_filename(name)
                    dated_files.append((dt, f))
                except Exception as e:
                    print(f"    跳过无法解析日期的文件 {f.name}: {e}")
                    continue

            dated_files.sort(key=lambda x: x[0])
            var_dates = [dt for dt, _ in dated_files]

            with rasterio.open(dated_files[0][1]) as ds:
                H, W = ds.shape

            var_arr = np.zeros((len(dated_files), H, W), dtype=np.float32)

            for i, (dt, f) in enumerate(dated_files):
                try:
                    with rasterio.open(f) as ds:
                        data = ds.read(1).astype(np.float32)
                    var_arr[i] = np.nan_to_num(data, nan=0.0)
                except Exception as e:
                    print(f"    读取 {f.name} 失败: {e}")
                    var_arr[i] = np.zeros((H, W), dtype=np.float32)

            return var_arr, var_dates, src_bounds, src_transform

    def _unify_to_common_grid(self, data: np.ndarray, var_name: str,
                                    src_bounds, src_transform) -> np.ndarray:
        """
        基于地理坐标对齐到公共网格
        """
        print(f"  基于地理坐标对齐 {var_name}...")

        # 获取目标网格参数
        target_h, target_w = self.H, self.W
        target_transform = self.transform

        # 检查数据维度
        if len(data.shape) == 3:
            # 时间序列数据
            T, src_h, src_w = data.shape
            aligned = np.zeros((T, target_h, target_w), dtype=data.dtype)

            for t in range(T):
                aligned[t] = self._align_single_layer(
                    data[t], src_transform, target_transform, target_h, target_w
                )
        else:
            # 单层数据
            aligned = self._align_single_layer(
                data, src_transform, target_transform, target_h, target_w
            )

        return aligned

    def _simple_resize(self, data, target_h, target_w):
        """简单的最近邻缩放"""
        from scipy.ndimage import zoom

        h, w = data.shape
        zoom_factors = (target_h / h, target_w / w)

        return zoom(data, zoom_factors, order=0)  # 最近邻插值

    def _align_single_layer(self, src_data, src_transform,
                            target_transform, target_h, target_w):
        """对齐单个图层"""
        aligned = np.zeros((target_h, target_w), dtype=src_data.dtype)

        # 获取源数据形状
        src_h, src_w = src_data.shape

        for row in range(target_h):
            for col in range(target_w):
                # 目标像素中心的地理坐标
                target_x, target_y = target_transform * (col + 0.5, row + 0.5)

                # 在源数据中的行列号
                src_row, src_col = rasterio.transform.rowcol(
                    src_transform, target_x, target_y, op=round
                )

                # 检查边界
                if 0 <= src_row < src_h and 0 <= src_col < src_w:
                    aligned[row, col] = src_data[src_row, src_col]
                else:
                    aligned[row, col] = np.nan  # 或0

        return aligned

    def _align_to_grid(self, data: np.ndarray, src_bounds, src_transform) -> np.ndarray:
        """将数据对齐到公共网格"""
        # 计算偏移（以像素为单位）
        res_x = abs(self.transform.a)
        res_y = abs(self.transform.e)

        # 计算在公共网格中的起始位置
        col_offset = int((src_bounds.left - self.common_bounds.left) / res_x)
        row_offset = int((self.common_bounds.top - src_bounds.top) / res_y)

        # 创建目标数组
        if len(data.shape) == 2:
            target_data = np.zeros((self.H, self.W), dtype=data.dtype)

            # 计算复制区域
            src_h, src_w = data.shape
            dest_h = min(src_h, self.H - row_offset)
            dest_w = min(src_w, self.W - col_offset)

            if dest_h > 0 and dest_w > 0:
                target_data[row_offset:row_offset + dest_h, col_offset:col_offset + dest_w] = \
                    data[:dest_h, :dest_w]

        elif len(data.shape) == 3:
            T, src_h, src_w = data.shape
            target_data = np.zeros((T, self.H, self.W), dtype=data.dtype)

            dest_h = min(src_h, self.H - row_offset)
            dest_w = min(src_w, self.W - col_offset)

            if dest_h > 0 and dest_w > 0:
                for t in range(T):
                    target_data[t, row_offset:row_offset + dest_h, col_offset:col_offset + dest_w] = \
                        data[t, :dest_h, :dest_w]

        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")

        return target_data

    def _load_static_conv_features_unified(self):
        """加载静态卷积特征（统一到公共区域）"""
        print(f"\n加载静态卷积特征...")

        # clamday
        clamday_files = conv_static_path("clamday", self.year_target, self.clamday_threshold)
        if clamday_files:
            with rasterio.open(clamday_files[0]) as ds:
                clamday_data = ds.read(1).astype(np.float32)
                src_bounds = ds.bounds

            self.clamday_data = self._align_to_grid(clamday_data, src_bounds, ds.transform)
            print(f"  Clamday形状: {self.clamday_data.shape}")
        else:
            self.clamday_data = np.zeros((self.H, self.W), dtype=np.float32)
            print(f"  警告: 未找到Clamday文件")

        # dem
        dem_files = conv_static_path("dem", self.year_target)
        dem_data = []
        for i, dem_file in enumerate(dem_files):
            with rasterio.open(dem_file) as ds:
                dem_band = ds.read(1).astype(np.float32)
                src_bounds = ds.bounds

            aligned_dem = self._align_to_grid(dem_band, src_bounds, ds.transform)
            dem_data.append(aligned_dem)
            print(f"  DEM{i + 1}形状: {aligned_dem.shape}")

        self.dem_data = dem_data  # [mean, std]

    def _load_point_data_unified(self):
        """加载点特征数据（统一到公共区域）"""
        print(f"\n加载点特征数据...")

        # 加载ls
        ls_file = point_var_path("ls", self.year_target)
        if ls_file and ls_file.exists():
            with rasterio.open(ls_file) as ds:
                ls_data_raw = ds.read()  # (C_ls, H, W)
                src_bounds = ds.bounds

            # 对齐每个波段
            aligned_bands = []
            for i in range(ls_data_raw.shape[0]):
                band_aligned = self._align_to_grid(ls_data_raw[i], src_bounds, ds.transform)
                aligned_bands.append(band_aligned)

            self.ls_data = np.stack(aligned_bands, axis=0)
            print(f"  LS数据形状: {self.ls_data.shape}")
        else:
            # 如果没有LS数据，使用一个全零的单波段
            self.ls_data = np.zeros((1, self.H, self.W), dtype=np.float32)
            print(f"  警告: 未找到LS文件")

    def _load_labels_unified(self):
        """加载标签数据"""
        print(f"\n加载标签数据...")

        self.label_data = {}  # date -> (H, W) 标签数组

        # 查找所有标签文件
        label_files = sorted(list(self.label_root.glob("*.tif")))
        print(f"  找到 {len(label_files)} 个标签文件")

        for label_file in label_files:
            try:
                # 解析日期
                name = label_file.name
                dt = self._parse_date_from_filename(name)

                if dt.year != self.year_target:
                    continue

                with rasterio.open(label_file) as ds:
                    label_arr = ds.read(1).astype(np.float32)  # (H, W)
                    label_nodata = ds.nodata

                # 标签数据应该已经在统一网格上，但检查一下
                if label_arr.shape != (self.H, self.W):
                    print(f"  警告: 标签形状不匹配: {label_arr.shape} vs ({self.H}, {self.W})")
                    # 尝试调整
                    label_arr = self._resize_to_standard(label_arr, self.H, self.W)

                # 存储标签
                self.label_data[dt] = (label_arr, label_nodata)

                print(f"  {dt.strftime('%Y-%m-%d')}: 加载成功")

            except Exception as e:
                print(f"  加载标签文件 {label_file.name} 失败: {e}")
                continue

        if not self.label_data:
            raise ValueError("没有加载到任何标签数据")

    def _parse_date_from_filename(self, filename: str) -> datetime:
        """从文件名解析日期"""
        # 尝试格式1: XINGJANG_CHELSA_sfcWind_05_03_2015_V.2.1_resampled.tif
        match = re.search(r'(\d{2})_(\d{2})_(\d{4})', filename)
        if match:
            day, month, year = match.groups()
            return datetime(int(year), int(month), int(day))

        # 尝试格式2: XINGJIANG_XGB_SWE_DAILY_025_20150101.tif
        match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
        if match:
            year, month, day = match.groups()
            return datetime(int(year), int(month), int(day))

        # 尝试格式3: ERA5_ST_201501_UTC8_27830m.tif (月份文件)
        match = re.search(r'(\d{4})(\d{2})', filename)
        if match and ("ERA5_ST" in filename or "ERA5_RH" in filename):
            year, month = match.groups()
            return datetime(int(year), int(month), 1)

        raise ValueError(f"无法从文件名找到日期: {filename}")

    def _build_sample_index(self):
        """构建样本索引"""
        print(f"\n构建样本索引...")

        self.meta_index: List[Tuple[datetime, int, int]] = []

        processed_dates = set()
        samples_per_date = {}

        for date_dt, (label_arr, label_nodata) in self.label_data.items():
            # 检查日期是否在卷积特征的时间轴中
            if date_dt not in self.date_to_index:
                # 查找最近的有效日期
                valid_date = None
                min_diff = float('inf')
                for conv_date in self.all_dates:
                    if conv_date.year == date_dt.year and conv_date.month == date_dt.month:
                        diff = abs((conv_date - date_dt).days)
                        if diff < min_diff:
                            min_diff = diff
                            valid_date = conv_date

                if valid_date is None or min_diff > 3:  # 允许最多3天的差异
                    continue

                date_dt = valid_date

            # 避免重复处理同一日期
            if date_dt in processed_dates:
                continue
            processed_dates.add(date_dt)

            # 有效像元掩码
            if label_nodata is not None:
                valid_mask = (label_arr != label_nodata) & np.isfinite(label_arr)
            else:
                valid_mask = np.isfinite(label_arr)

            valid_pixels = np.count_nonzero(valid_mask)

            if valid_pixels < self.min_valid_pixels:
                continue

            # 确保只选择能够提取完整patch的像素
            candidate_indices = []
            for (r, c) in np.argwhere(valid_mask):
                # 检查边界条件
                r0, r1 = r - self.R, r + self.R + 1
                c0, c1 = c - self.R, c + self.R + 1

                # 如果像素太靠近边界，跳过
                if r0 < 0 or r1 > self.H or c0 < 0 or c1 > self.W:
                    continue

                candidate_indices.append((r, c))

            if not candidate_indices:
                continue

            # 随机采样
            np.random.shuffle(candidate_indices)
            if self.samples_per_day is not None:
                n_samples = min(self.samples_per_day, len(candidate_indices))
                candidate_indices = candidate_indices[:n_samples]

            # 验证每个候选像素都能提取特征
            valid_samples = []
            for (r, c) in candidate_indices:
                # 测试是否能构建特征
                conv_patch = self._build_spatial_features(date_dt, r, c)
                point_feats = self._build_point_features(date_dt, r, c)

                if conv_patch is not None and point_feats is not None:
                    valid_samples.append((r, c))

            # 添加到索引
            for (r, c) in valid_samples:
                self.meta_index.append((date_dt, int(r), int(c)))

            samples_per_date[date_dt] = len(valid_samples)

        print(f"\n总样本数: {len(self.meta_index)}")

    def _compute_minmax_sampling(self):
        """计算特征的min/max用于标准化"""
        print(f"\n计算特征统计量...")

        # 1. 卷积特征的统计量
        conv_mins = []
        conv_maxs = []

        # 动态卷积特征
        for var in CONV_VARS:
            arr = self.conv_dyn_data[var]  # (T, H, W)
            valid_data = arr[np.isfinite(arr)]
            if len(valid_data) > 0:
                conv_mins.append(float(np.min(valid_data)))
                conv_maxs.append(float(np.max(valid_data)))
            else:
                conv_mins.append(0.0)
                conv_maxs.append(1.0)

        # 静态卷积特征
        static_conv_features = [
            ("clamday", self.clamday_data),
            ("dem_mean", self.dem_data[0]),
            ("dem_std", self.dem_data[1])
        ]

        for name, arr in static_conv_features:
            valid_data = arr[np.isfinite(arr)]
            if len(valid_data) > 0:
                conv_mins.append(float(np.min(valid_data)))
                conv_maxs.append(float(np.max(valid_data)))
            else:
                conv_mins.append(0.0)
                conv_maxs.append(1.0)

        self.conv_min = np.array(conv_mins, dtype=np.float32)
        self.conv_max = np.array(conv_maxs, dtype=np.float32)
        self.C_conv = len(self.conv_min)

        # 2. 点特征的统计量
        point_mins = []
        point_maxs = []

        # LS特征
        for i in range(self.ls_data.shape[0]):
            band_data = self.ls_data[i]
            valid_data = band_data[np.isfinite(band_data)]
            if len(valid_data) > 0:
                point_mins.append(float(np.min(valid_data)))
                point_maxs.append(float(np.max(valid_data)))
            else:
                point_mins.append(0.0)
                point_maxs.append(1.0)

        # 添加经纬度范围（归一化到0-1）
        point_mins.extend([0.0, 0.0])  # 经纬度最小值
        point_maxs.extend([1.0, 1.0])  # 经纬度最大值

        # 添加doy范围
        point_mins.append(0.0)  # doy最小值 (1月1日归一化后为0)
        point_maxs.append(1.0)  # doy最大值 (12月31日归一化后为1)

        self.point_min = np.array(point_mins, dtype=np.float32)
        self.point_max = np.array(point_maxs, dtype=np.float32)
        self.C_point = len(self.point_min)

        # 3. 标签的统计量
        all_labels = []
        for label_arr, label_nodata in self.label_data.values():
            if label_nodata is not None:
                valid_labels = label_arr[label_arr != label_nodata]
            else:
                valid_labels = label_arr[np.isfinite(label_arr)]
            all_labels.extend(valid_labels.flatten())

        all_labels = np.array(all_labels)
        valid_labels = all_labels[np.isfinite(all_labels)]

        if len(valid_labels) > 0:
            self.label_min = float(np.min(valid_labels))
            self.label_max = float(np.max(valid_labels))
        else:
            self.label_min = 0.0
            self.label_max = 1.0

        print(f"  卷积特征: {self.C_conv} 个通道")
        print(f"  点特征: {self.C_point} 个维度")
        print(f"  标签范围: [{self.label_min:.4f}, {self.label_max:.4f}]")

    def _build_time_features(self, date_dt: datetime) -> np.ndarray:
        """构建时间特征"""
        # 年日 (一年中的第几天)
        day_of_year = date_dt.timetuple().tm_yday
        # 归一化到0-1
        doy_norm = (day_of_year - 1) / 365.0
        return np.array([doy_norm], dtype=np.float32)

    def _build_spatial_features(self, date_dt: datetime, r: int, c: int) -> np.ndarray:
        """构建卷积特征"""
        # 获取卷积特征索引
        date_idx = self.date_to_index.get(date_dt)
        if date_idx is None:
            return None

        # 提取patch区域
        r0 = max(0, r - self.R)
        r1 = min(self.H, r + self.R + 1)
        c0 = max(0, c - self.R)
        c1 = min(self.W, c + self.R + 1)

        actual_h = r1 - r0
        actual_w = c1 - c0

        # 如果patch太小，返回None
        if actual_h < 3 or actual_w < 3:
            return None

        # 收集所有卷积特征
        conv_features = []

        # 1. 动态卷积特征 (取当天的)
        for var in CONV_VARS:
            var_arr = self.conv_dyn_data[var]  # (T, H, W)
            if date_idx >= var_arr.shape[0]:
                patch = var_arr[-1, r0:r1, c0:c1]
            else:
                patch = var_arr[date_idx, r0:r1, c0:c1]

            if patch.shape != (self.P, self.P):
                patch = self._resize_to_standard(patch, self.P, self.P)
            conv_features.append(patch)

        # 2. 静态卷积特征 (clamday, dem)
        # clamday
        clamday_patch = self.clamday_data[r0:r1, c0:c1]
        clamday_patch = self._resize_to_standard(clamday_patch, self.P, self.P)
        conv_features.append(clamday_patch)

        # dem_mean
        dem_mean_patch = self.dem_data[0][r0:r1, c0:c1]
        dem_mean_patch = self._resize_to_standard(dem_mean_patch, self.P, self.P)
        conv_features.append(dem_mean_patch)

        # dem_std
        dem_std_patch = self.dem_data[1][r0:r1, c0:c1]
        dem_std_patch = self._resize_to_standard(dem_std_patch, self.P, self.P)
        conv_features.append(dem_std_patch)

        try:
            conv_patch = np.stack(conv_features, axis=0)  # (C_conv, P, P)
            return conv_patch
        except Exception as e:
            return None

    def _resize_to_standard(self, patch: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """调整patch到标准尺寸"""
        h, w = patch.shape

        if h == target_h and w == target_w:
            return patch

        # 创建目标数组
        result = np.zeros((target_h, target_w), dtype=patch.dtype)

        # 计算复制区域
        h_start = (target_h - h) // 2 if h < target_h else 0
        w_start = (target_w - w) // 2 if w < target_w else 0

        h_end = h_start + min(h, target_h)
        w_end = w_start + min(w, target_w)

        src_h_start = max(0, -h_start)
        src_w_start = max(0, -w_start)

        # 复制数据
        result[h_start:h_end, w_start:w_end] = patch[src_h_start:src_h_start + (h_end - h_start),
                                               src_w_start:src_w_start + (w_end - w_start)]

        return result

    def _build_point_features(self, date_dt: datetime, r: int, c: int) -> np.ndarray:
        """构建点特征"""
        point_features = []

        # 1. LS特征
        for i in range(self.ls_data.shape[0]):
            point_features.append(float(self.ls_data[i, r, c]))

        # 2. 经纬度特征
        lon, lat = self._pixel_to_lonlat(r, c)
        # 归一化经纬度
        lon_norm = (lon + 180) / 360  # 假设经度范围-180到180
        lat_norm = (lat + 90) / 180  # 假设纬度范围-90到90
        point_features.extend([lon_norm, lat_norm])

        # 3. 时间特征
        time_feats = self._build_time_features(date_dt)
        point_features.extend(time_feats)

        point_feats_array = np.array(point_features, dtype=np.float32)

        # 检查NaN
        if np.any(np.isnan(point_feats_array)):
            point_feats_array = np.nan_to_num(point_feats_array, nan=0.0)

        return point_feats_array

    def _pixel_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        """将像素坐标转换为经纬度"""
        x, y = self.transform * (col + 0.5, row + 0.5)
        lon, lat = self.transformer.transform(x, y)
        return lon, lat

    def __len__(self):
        return len(self.meta_index)

    def __getitem__(self, idx: int):
        """获取一个样本"""
        max_retry = 10
        cur_idx = idx

        for _ in range(max_retry):
            date_dt, r, c = self.meta_index[cur_idx]

            # 构建卷积特征
            conv_patch = self._build_spatial_features(date_dt, r, c)
            if conv_patch is None:
                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            # 构建点特征
            point_feats = self._build_point_features(date_dt, r, c)
            if point_feats is None:
                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            # 获取标签
            label_arr, label_nodata = self.label_data[date_dt]
            y = label_arr[r, c]

            # 检查标签是否有效
            if (label_nodata is not None and y == label_nodata) or np.isnan(y):
                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            # 转换为torch张量
            conv_t = torch.from_numpy(conv_patch)
            point_t = torch.from_numpy(point_feats)
            y_t = torch.tensor(y, dtype=torch.float32)

            # Min-Max标准化
            eps = 1e-6

            conv_t = (conv_t - torch.from_numpy(self.conv_min).view(-1, 1, 1)) / \
                     (torch.from_numpy(self.conv_max + eps).view(-1, 1, 1) - torch.from_numpy(self.conv_min).view(-1, 1, 1))

            point_t = (point_t - torch.from_numpy(self.point_min)) / \
                      (torch.from_numpy(self.point_max + eps) - torch.from_numpy(self.point_min))

            y_t = (y_t - self.label_min) / (self.label_max - self.label_min)

            return conv_t, point_t, y_t

        raise IndexError(f"在idx={idx}附近连续{max_retry}个样本均无效")

def build_dataloaders(
        batch_size: int = 32,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
        **dataset_kwargs
):
    """构建数据加载器"""
    try:
        dataset = SWEDataset(**dataset_kwargs)

        n_total = len(dataset)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        print(f"\n[DataLoader] 样本总数: {n_total}, train={n_train}, val={n_val}")

        if n_total == 0:
            raise ValueError("数据集为空")

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

        return train_loader, val_loader, (dataset.C_conv, dataset.C_point)

    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("=" * 50)
    print("测试数据加载...")
    print("=" * 50)

    try:
        # 创建数据集
        print("\n1. 创建数据集...")
        dataset = SWEDataset()

        # 测试获取样本
        print(f"\n2. 测试获取样本...")
        if len(dataset) > 0:
            for i in range(min(3, len(dataset))):
                print(f"\n  样本 {i}:")
                try:
                    conv, point, y = dataset[i]
                    print(f"    conv shape: {conv.shape}")
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
                print(f"   数据{i}: shape = {data.shape}")

        print(f"\n✓ 数据加载测试完成!")

    except Exception as e:
        print(f"\n✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()