# data_online_era5_swe.py
# -*- coding: utf-8 -*-
"""
在线从栅格构建 SWE 反演训练样本
- 卷积特征: chelsa_sfxwind, lst, rh, clamday, dem
- 点特征: ls, S1_VV, S1_VH, 经纬度, doy
- 标签: fusedSWE
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import rasterio
from datetime import datetime, timedelta
from pyproj import Transformer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import calendar
import re
from collections import defaultdict

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
POINT_VARS = ["ls", "S1_VV", "S1_VH", "SMAP_TBV", "SMAP_TBH"]  # 添加哨兵1和SMAP亮温

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


# 点变量路径 - 修改添加哨兵1
def point_var_path(var: str, year: int) -> Union[Path, List[Path]]:
    """点变量路径"""
    if var == "ls":
        ls_path = FEATURE_ROOT / "ls" / "XINJIANG"
        pattern = f"*{year}*Median*.tif"
        files = list(ls_path.glob(pattern))
        return files[0] if files else None
    elif var == "S1_VV" or var == "S1_VH":
        # 哨兵1数据路径
        s1_path = FEATURE_ROOT / "S1_Yearly_Results" / "XINJIANG"
        if not s1_path.exists():
            return []

        # 获取两年数据
        all_files = []
        for y in [year - 1, year]:
            pattern = f"*{y}*.tif"
            all_files.extend(s1_path.glob(pattern))

        return sorted(all_files) if all_files else []
    elif var == "SMAP_TBV" or var == "SMAP_TBH":
        # SMAP亮温数据路径 - 在函数内部定义
        smap_root = Path(r"G:\王扬\smap_data\xinjiang")
        if not smap_root.exists():
            print(f"  SMAP路径不存在: {smap_root}")
            return []

        # 获取所有SMAP文件
        pattern = f"*{year}*.tif"
        smap_files = list(smap_root.glob(pattern))
        if not smap_files:
            print(f"  未找到{year}年的SMAP文件")
        return sorted(smap_files) if smap_files else []
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
            s1_interp_method: str = "nearest",  # nearest, linear, time_weighted
            s1_max_gap_days: int = 7,  # 最大插值间隔
            s1_nodata_value: float = -9999.0,
            smap_interp_method: str = "nearest",  # SMAP插值方法
            smap_max_gap_days: int = 7,  # SMAP最大插值间隔
            smap_nodata_value: float = -9999.0,  # SMAP nodata值
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

        # 哨兵1参数
        self.s1_interp_method = s1_interp_method
        self.s1_max_gap_days = s1_max_gap_days
        self.s1_nodata_value = s1_nodata_value

        # SMAP参数
        self.smap_interp_method = smap_interp_method
        self.smap_max_gap_days = smap_max_gap_days
        self.smap_nodata_value = smap_nodata_value

        print(f"\n初始化数据集:")
        print(f"  区域: {region}")
        print(f"  目标年份: {year_target}")
        print(f"  卷积特征: {CONV_VARS + CONV_STATIC_VARS}")
        print(f"  点特征: {POINT_VARS}")
        print(f"  Clamday阈值: {clamday_threshold}")
        print(f"  哨兵1插值方法: {s1_interp_method}")
        print(f"  SMAP插值方法: {smap_interp_method}")

        # 哨兵1数据存储
        self.s1_data = {}  # date -> {"VV": array, "VH": array}
        self.all_s1_dates = []  # 所有有哨兵1数据的日期

        # SMAP数据存储
        self.smap_data = {}  # date -> {"TBV": array, "TBH": array}
        self.all_smap_dates = []  # 所有有SMAP数据的日期

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

    def _check_alignment_quality(self, var_name: str, src_data: np.ndarray, aligned_data: np.ndarray):
        """检查对齐质量"""
        print(f"    对齐质量检查 ({var_name}):")

        # 统计有效像素
        src_valid = np.sum(~np.isnan(src_data))
        aligned_valid = np.sum(~np.isnan(aligned_data))

        print(f"      源数据有效像素: {src_valid} ({src_valid / src_data.size * 100:.1f}%)")
        print(f"      对齐后有效像素: {aligned_valid} ({aligned_valid / aligned_data.size * 100:.1f}%)")

        # 检查数值范围
        if src_valid > 0 and aligned_valid > 0:
            src_min, src_max = np.nanmin(src_data), np.nanmax(src_data)
            aligned_min, aligned_max = np.nanmin(aligned_data), np.nanmax(aligned_data)

            print(f"      源数据范围: [{src_min:.4f}, {src_max:.4f}]")
            print(f"      对齐后范围: [{aligned_min:.4f}, {aligned_max:.4f}]")

            # 检查是否有很多NaN
            nan_ratio = np.sum(np.isnan(aligned_data)) / aligned_data.size
            if nan_ratio > 0.5:
                print(f"      ⚠ 警告: 对齐后有 {nan_ratio * 100:.1f}% 的NaN值")

    def _load_data_unified(self):
        """加载所有数据（统一到公共区域）"""
        # 加载卷积特征
        self._load_conv_data_unified()
        # 加载静态卷积特征
        self._load_static_conv_features_unified()
        # 加载点特征
        self._load_point_data_unified()
        # 加载标签
        self._load_labels_unified()

        # 验证所有数据的对齐
        self._validate_all_alignment()

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

    def _align_single_layer(self, src_data, src_transform,
                            target_transform, target_h, target_w):
        """对齐单个图层（基于地理坐标的精确对齐）"""
        print(f"      对齐: {src_data.shape} -> ({target_h}, {target_w})")

        aligned = np.full((target_h, target_w), np.nan, dtype=src_data.dtype)

        # 获取源数据形状
        src_h, src_w = src_data.shape

        # 获取目标网格的所有坐标
        rows, cols = np.meshgrid(range(target_h), range(target_w), indexing='ij')

        # 批量计算目标像素中心的地理坐标
        target_xs = target_transform.a * cols.ravel() + target_transform.b * rows.ravel() + target_transform.c + target_transform.a * 0.5 + target_transform.b * 0.5
        target_ys = target_transform.d * cols.ravel() + target_transform.e * rows.ravel() + target_transform.f + target_transform.d * 0.5 + target_transform.e * 0.5

        # 批量转换为源数据中的行列号
        src_rows, src_cols = rasterio.transform.rowcol(
            src_transform,
            target_xs,
            target_ys,
            op=round
        )

        # 一次性赋值
        for i in range(len(src_rows)):
            r, c = src_rows[i], src_cols[i]
            row_idx, col_idx = rows.ravel()[i], cols.ravel()[i]

            if 0 <= r < src_h and 0 <= c < src_w:
                aligned[row_idx, col_idx] = src_data[r, c]

        # 如果没有有效数据，填充0
        if np.all(np.isnan(aligned)):
            print(f"      ⚠ 警告: 对齐后全部为NaN，填充0")
            aligned = np.zeros((target_h, target_w), dtype=src_data.dtype)
        else:
            # 填充NaN为0
            nan_mask = np.isnan(aligned)
            if np.any(nan_mask):
                print(f"      填充 {np.sum(nan_mask)} 个NaN值为0")
                aligned[nan_mask] = 0.0

        return aligned

    def _load_static_conv_features_unified(self):
        """加载静态卷积特征（统一到公共区域）"""
        print(f"\n加载静态卷积特征...")

        # 初始化属性
        self.clamday_data = None
        self.dem_data = None

        # clamday
        clamday_files = conv_static_path("clamday", self.year_target, self.clamday_threshold)
        if clamday_files:
            with rasterio.open(clamday_files[0]) as ds:
                clamday_data_raw = ds.read(1).astype(np.float32)
                src_bounds = ds.bounds
                src_transform = ds.transform

            # 对齐到公共网格
            self.clamday_data = self._align_single_layer(
                clamday_data_raw, src_transform, self.transform, self.H, self.W
            )
            print(f"  Clamday形状: {self.clamday_data.shape}")
        else:
            self.clamday_data = np.zeros((self.H, self.W), dtype=np.float32)
            print(f"  警告: 未找到Clamday文件")

        # dem
        dem_files = conv_static_path("dem", self.year_target)
        self.dem_data = []

        if dem_files:
            for i, dem_file in enumerate(dem_files):
                with rasterio.open(dem_file) as ds:
                    dem_band_raw = ds.read(1).astype(np.float32)
                    src_bounds = ds.bounds
                    src_transform = ds.transform

                aligned_dem = self._align_single_layer(
                    dem_band_raw, src_transform, self.transform, self.H, self.W
                )
                self.dem_data.append(aligned_dem)
                print(f"  DEM{i + 1}形状: {aligned_dem.shape}")
        else:
            # 如果没有DEM文件，创建两个全零的波段
            self.dem_data = [
                np.zeros((self.H, self.W), dtype=np.float32),
                np.zeros((self.H, self.W), dtype=np.float32)
            ]
            print(f"  警告: 未找到DEM文件")

        # 确保dem_data有至少两个波段
        if len(self.dem_data) < 2:
            # 如果只有一个波段，复制一份作为第二个
            if len(self.dem_data) == 1:
                self.dem_data.append(self.dem_data[0].copy())
            else:
                # 如果没有波段，创建两个
                self.dem_data = [
                    np.zeros((self.H, self.W), dtype=np.float32),
                    np.zeros((self.H, self.W), dtype=np.float32)
                ]

    def _load_sentinel1_data(self):
        """加载哨兵1数据 - 按照点特征的逻辑"""
        print(f"  加载哨兵1数据...")

        # 获取哨兵1文件列表
        s1_files = point_var_path("S1_VV", self.year_target)
        if not s1_files:
            print(f"  警告: 未找到哨兵1数据")
            return

        print(f"  找到 {len(s1_files)} 个哨兵1文件")

        # 处理每个文件
        for s1_file in s1_files:
            try:
                # 读取文件
                with rasterio.open(s1_file) as ds:
                    # 解析文件名获取月份
                    filename = s1_file.stem
                    match = re.search(r'(\d{4})_(\d{2})', filename)
                    if match:
                        year = int(match.group(1))
                        month = int(match.group(2))
                    else:
                        continue

                    # 获取波段数量
                    n_bands = ds.count
                    band_descriptions = ds.descriptions if ds.descriptions else []

                    # 处理每个波段
                    for band_idx in range(1, n_bands + 1):
                        band_name = band_descriptions[band_idx - 1] if band_idx - 1 < len(
                            band_descriptions) else f"Band_{band_idx}"

                        # 解析波段名称获取日期和极化方式
                        pol, band_date = self._parse_s1_band_info(band_name, year, month)
                        if pol is None or band_date is None:
                            continue

                        # 读取波段数据
                        band_data = ds.read(band_idx).astype(np.float32)
                        src_bounds = ds.bounds
                        src_transform = ds.transform

                        # 对齐到公共网格
                        aligned_data = self._align_single_layer(
                            band_data, src_transform, self.transform, self.H, self.W
                        )

                        # 处理nodata
                        if ds.nodata is not None:
                            aligned_data[aligned_data == ds.nodata] = self.s1_nodata_value

                        # 存储数据
                        if band_date not in self.s1_data:
                            self.s1_data[band_date] = {}

                        self.s1_data[band_date][pol] = aligned_data

                        # 添加到日期列表
                        if band_date not in self.all_s1_dates:
                            self.all_s1_dates.append(band_date)

                        print(f"    处理波段: {band_name}, 日期: {band_date.strftime('%Y-%m-%d')}, 极化: {pol}")

            except Exception as e:
                print(f"    处理文件 {s1_file.name} 失败: {e}")
                continue

        # 按日期排序
        self.all_s1_dates.sort()

        print(f"  哨兵1数据: {len(self.all_s1_dates)} 个日期")
        print(
            f"  日期范围: {self.all_s1_dates[0].strftime('%Y-%m-%d')} 到 {self.all_s1_dates[-1].strftime('%Y-%m-%d')}")

    def _load_smap_data(self):
        """加载SMAP亮温数据"""
        print(f"  加载SMAP亮温数据...")

        # 使用point_var_path函数获取SMAP文件列表
        smap_files = point_var_path("SMAP_TBV", self.year_target)
        if not smap_files:
            print(f"  警告: 未找到SMAP数据")
            return

        print(f"  找到 {len(smap_files)} 个SMAP文件")

        # 处理每个SMAP文件
        for smap_file in smap_files:
            try:
                # 读取文件
                with rasterio.open(smap_file) as ds:
                    # 解析文件名获取日期
                    filename = smap_file.name
                    # 解析格式: SMAP_L1C_TB_E_00914_D_20150404T014324_R19240_001_resampled.tif
                    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
                    if not match:
                        print(f"    无法解析文件名: {filename}")
                        continue

                    year, month, day = match.groups()
                    band_date = datetime(int(year), int(month), int(day))

                    # SMAP文件有两个波段: 垂直极化(TBV)和水平极化(TBH)
                    # 假设波段1是TBV，波段2是TBH
                    if ds.count >= 2:
                        # 读取TBV (垂直极化)
                        tbv_data = ds.read(1).astype(np.float32)
                        # 读取TBH (水平极化)
                        tbh_data = ds.read(2).astype(np.float32)

                        src_bounds = ds.bounds
                        src_transform = ds.transform

                        # 对齐到公共网格
                        tbv_aligned = self._align_single_layer(
                            tbv_data, src_transform, self.transform, self.H, self.W
                        )
                        tbh_aligned = self._align_single_layer(
                            tbh_data, src_transform, self.transform, self.H, self.W
                        )

                        # 处理nodata
                        if ds.nodata is not None:
                            tbv_aligned[tbv_aligned == ds.nodata] = self.smap_nodata_value
                            tbh_aligned[tbh_aligned == ds.nodata] = self.smap_nodata_value

                        # 存储数据
                        if band_date not in self.smap_data:
                            self.smap_data[band_date] = {}

                        self.smap_data[band_date]["TBV"] = tbv_aligned
                        self.smap_data[band_date]["TBH"] = tbh_aligned

                        # 添加到日期列表
                        if band_date not in self.all_smap_dates:
                            self.all_smap_dates.append(band_date)

                        print(f"    处理SMAP文件: {filename}, 日期: {band_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                print(f"    处理SMAP文件 {smap_file.name} 失败: {e}")
                continue

        # 按日期排序
        self.all_smap_dates.sort()

        if self.all_smap_dates:
            print(f"  SMAP数据: {len(self.all_smap_dates)} 个日期")
            print(
                f"  日期范围: {self.all_smap_dates[0].strftime('%Y-%m-%d')} 到 {self.all_smap_dates[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"  警告: 没有有效的SMAP数据")

    def _parse_s1_band_info(self, band_name: str, year: int, month: int):
        """解析哨兵1波段信息"""
        pol = None
        day = 1

        # 查找极化方式
        if 'VV' in band_name.upper():
            pol = 'VV'
        elif 'VH' in band_name.upper():
            pol = 'VH'

        # 查找日期
        date_match = re.search(r'(\d{4})(\d{2})(\d{2})', band_name)
        if date_match:
            y, m, d = date_match.groups()
            try:
                band_date = datetime(int(y), int(m), int(d))
                return pol, band_date
            except:
                pass

        # 如果无法解析具体日期，使用月份第一天
        try:
            band_date = datetime(year, month, day)
        except:
            band_date = None

        return pol, band_date

    def _get_sentinel1_value(self, date_dt: datetime, r: int, c: int) -> Tuple[float, float]:
        """获取指定日期和位置的哨兵1值"""
        vv_value = self.s1_nodata_value
        vh_value = self.s1_nodata_value

        # 如果正好有这个日期的数据
        if date_dt in self.s1_data:
            if 'VV' in self.s1_data[date_dt]:
                vv_value = float(self.s1_data[date_dt]['VV'][r, c])
            if 'VH' in self.s1_data[date_dt]:
                vh_value = float(self.s1_data[date_dt]['VH'][r, c])
            return vv_value, vh_value

        # 否则进行时间插值
        if not self.all_s1_dates:
            return vv_value, vh_value

        # 最近邻插值
        if self.s1_interp_method == "nearest":
            nearest_date = min(self.all_s1_dates, key=lambda d: abs((d - date_dt).days))
            if abs((nearest_date - date_dt).days) <= self.s1_max_gap_days:
                if nearest_date in self.s1_data:
                    if 'VV' in self.s1_data[nearest_date]:
                        vv_value = float(self.s1_data[nearest_date]['VV'][r, c])
                    if 'VH' in self.s1_data[nearest_date]:
                        vh_value = float(self.s1_data[nearest_date]['VH'][r, c])

        return vv_value, vh_value

    def _load_point_data_unified(self):
        """加载点特征数据（统一到公共区域）"""
        print(f"\n加载点特征数据...")

        # 加载ls
        ls_file = point_var_path("ls", self.year_target)
        if ls_file and ls_file.exists():
            print(f"  处理LS文件: {ls_file.name}")

            with rasterio.open(ls_file) as ds:
                ls_data_raw = ds.read()  # (C_ls, H, W)
                src_bounds = ds.bounds
                src_transform = ds.transform

            # 对齐每个波段
            aligned_bands = []
            for i in range(ls_data_raw.shape[0]):
                print(f"    对齐波段 {i + 1}/{ls_data_raw.shape[0]}")

                # 使用 _align_single_layer 进行地理坐标对齐
                band_aligned = self._align_single_layer(
                    ls_data_raw[i],
                    src_transform,
                    self.transform,
                    self.H,
                    self.W
                )
                aligned_bands.append(band_aligned)

            self.ls_data = np.stack(aligned_bands, axis=0)
            print(f"  LS数据形状: {self.ls_data.shape}")

            # 检查对齐质量
            self._check_alignment_quality("LS", ls_data_raw[0], self.ls_data[0])

        else:
            # 如果没有LS数据，使用一个全零的单波段
            self.ls_data = np.zeros((1, self.H, self.W), dtype=np.float32)
            print(f"  警告: 未找到LS文件，使用零数组")

        # 加载哨兵1数据
        self._load_sentinel1_data()

        # 加载SMAP数据
        self._load_smap_data()

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

    def _validate_all_alignment(self):
        """验证所有数据的对齐情况"""
        print(f"\n验证所有数据的对齐...")

        # 检查所有数据的形状是否一致
        expected_shape = (self.H, self.W)

        print("  1. 检查卷积特征对齐:")
        for var in CONV_VARS:
            if var in self.conv_dyn_data:
                data = self.conv_dyn_data[var]
                if len(data.shape) == 3:
                    T, H, W = data.shape
                    if (H, W) == expected_shape:
                        print(f"    ✓ {var}: {H}x{W} (对齐正确)")
                    else:
                        print(f"    ✗ {var}: {H}x{W} != {expected_shape}")

        print("  2. 检查点特征对齐:")
        if hasattr(self, 'ls_data'):
            C, H, W = self.ls_data.shape
            if (H, W) == expected_shape:
                print(f"    ✓ LS数据: {H}x{W} (对齐正确)")
            else:
                print(f"    ✗ LS数据: {H}x{W} != {expected_shape}")

        # 检查哨兵1数据
        if self.s1_data:
            for date_dt in list(self.s1_data.keys())[:1]:  # 只检查第一个日期
                for pol in ['VV', 'VH']:
                    if pol in self.s1_data[date_dt]:
                        arr = self.s1_data[date_dt][pol]
                        H, W = arr.shape
                        if (H, W) == expected_shape:
                            print(f"    ✓ S1_{pol}: {H}x{W} (对齐正确)")
                            break
                        else:
                            print(f"    ✗ S1_{pol}: {H}x{W} != {expected_shape}")
                            break
                break

        # 检查SMAP数据
        if self.smap_data:
            for date_dt in list(self.smap_data.keys())[:1]:  # 只检查第一个日期
                for pol in ['TBV', 'TBH']:
                    if pol in self.smap_data[date_dt]:
                        arr = self.smap_data[date_dt][pol]
                        H, W = arr.shape
                        if (H, W) == expected_shape:
                            print(f"    ✓ SMAP_{pol}: {H}x{W} (对齐正确)")
                            break
                        else:
                            print(f"    ✗ SMAP_{pol}: {H}x{W} != {expected_shape}")
                            break
                break

        print("  3. 检查标签数据对齐:")
        for date_dt, (label_arr, _) in self.label_data.items():
            H, W = label_arr.shape
            if (H, W) == expected_shape:
                print(f"    ✓ {date_dt}: {H}x{W} (对齐正确)")
                break
            else:
                print(f"    ✗ {date_dt}: {H}x{W} != {expected_shape}")
                break

        print("  4. 检查静态特征对齐:")
        if hasattr(self, 'clamday_data'):
            H, W = self.clamday_data.shape
            if (H, W) == expected_shape:
                print(f"    ✓ Clamday: {H}x{W} (对齐正确)")
            else:
                print(f"    ✗ Clamday: {H}x{W} != {expected_shape}")

        if hasattr(self, 'dem_data') and len(self.dem_data) > 0:
            for i, dem_layer in enumerate(self.dem_data):
                H, W = dem_layer.shape
                if (H, W) == expected_shape:
                    print(f"    ✓ DEM{i}: {H}x{W} (对齐正确)")
                else:
                    print(f"    ✗ DEM{i}: {H}x{W} != {expected_shape}")

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
        """构建样本索引 - 支持所有年份"""
        print(f"\n构建样本索引...")

        self.meta_index: List[Tuple[datetime, int, int]] = []
        samples_per_date = {}
        samples_per_year = {}

        for date_dt, (label_arr, label_nodata) in self.label_data.items():
            # 检查日期是否在卷积特征的时间轴中
            if date_dt not in self.date_to_index:
                # 查找最近的有效日期（放宽限制）
                valid_date = None
                min_diff = float('inf')
                for conv_date in self.all_dates:
                    diff = abs((conv_date - date_dt).days)
                    if diff < min_diff:
                        min_diff = diff
                        valid_date = conv_date

                if valid_date is None or min_diff > 7:  # 放宽到最多7天的差异
                    print(f"跳过日期 {date_dt}，找不到对应的卷积特征（最小差异: {min_diff}天）")
                    continue

                date_dt = valid_date

            # 统计年份信息
            year = date_dt.year
            if year not in samples_per_year:
                samples_per_year[year] = 0

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
                try:
                    # 测试是否能构建特征
                    conv_patch = self._build_spatial_features(date_dt, r, c)
                    point_feats = self._build_point_features(date_dt, r, c)

                    # 检查静态特征是否存在
                    if self.clamday_data is None:
                        print(f"警告: clamday_data 为 None，跳过样本构建")
                        continue

                    if conv_patch is not None and point_feats is not None:
                        # 检查是否有NaN值
                        if not np.any(np.isnan(conv_patch)) and not np.any(np.isnan(point_feats)):
                            valid_samples.append((r, c))
                except Exception as e:
                    print(f"构建特征失败 ({r}, {c}): {e}")
                    continue

            # 添加到索引
            for (r, c) in valid_samples:
                self.meta_index.append((date_dt, int(r), int(c)))

            samples_per_date[date_dt] = len(valid_samples)
            samples_per_year[year] += len(valid_samples)

        print(f"\n总样本数: {len(self.meta_index)}")
        print(f"按年份统计:")
        for year in sorted(samples_per_year.keys()):
            print(f"  {year}年: {samples_per_year[year]} 个样本")

        # 显示每天的样本数（前10天）
        if samples_per_date:
            print(f"\n按日期统计（前10天）:")
            for date, count in sorted(samples_per_date.items())[:10]:
                print(f"  {date.strftime('%Y-%m-%d')}: {count} 个样本")
        else:
            print("警告: 没有找到有效的样本")

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

        # 哨兵1特征 (VV和VH)
        s1_vv_all = []
        s1_vh_all = []

        for date_dt, pol_data in self.s1_data.items():
            if 'VV' in pol_data:
                data_vv = pol_data['VV']
                valid_vv = data_vv[data_vv != self.s1_nodata_value]
                s1_vv_all.extend(valid_vv.flatten())
            if 'VH' in pol_data:
                data_vh = pol_data['VH']
                valid_vh = data_vh[data_vh != self.s1_nodata_value]
                s1_vh_all.extend(valid_vh.flatten())

        # VV统计
        if s1_vv_all:
            s1_vv_all = np.array(s1_vv_all)
            valid_s1_vv = s1_vv_all[np.isfinite(s1_vv_all)]
            if len(valid_s1_vv) > 0:
                point_mins.append(float(np.min(valid_s1_vv)))
                point_maxs.append(float(np.max(valid_s1_vv)))
            else:
                point_mins.append(-30.0)
                point_maxs.append(5.0)
        else:
            point_mins.append(-30.0)
            point_maxs.append(5.0)

        # VH统计
        if s1_vh_all:
            s1_vh_all = np.array(s1_vh_all)
            valid_s1_vh = s1_vh_all[np.isfinite(s1_vh_all)]
            if len(valid_s1_vh) > 0:
                point_mins.append(float(np.min(valid_s1_vh)))
                point_maxs.append(float(np.max(valid_s1_vh)))
            else:
                point_mins.append(-35.0)
                point_maxs.append(0.0)
        else:
            point_mins.append(-35.0)
            point_maxs.append(0.0)

        # SMAP亮温特征 (TBV和TBH)
        smap_tbv_all = []
        smap_tbh_all = []

        for date_dt, pol_data in self.smap_data.items():
            if 'TBV' in pol_data:
                data_tbv = pol_data['TBV']
                valid_tbv = data_tbv[data_tbv != self.smap_nodata_value]
                smap_tbv_all.extend(valid_tbv.flatten())
            if 'TBH' in pol_data:
                data_tbh = pol_data['TBH']
                valid_tbh = data_tbh[data_tbh != self.smap_nodata_value]
                smap_tbh_all.extend(valid_tbh.flatten())

        # TBV统计
        if smap_tbv_all:
            smap_tbv_all = np.array(smap_tbv_all)
            valid_smap_tbv = smap_tbv_all[np.isfinite(smap_tbv_all)]
            if len(valid_smap_tbv) > 0:
                point_mins.append(float(np.min(valid_smap_tbv)))
                point_maxs.append(float(np.max(valid_smap_tbv)))
            else:
                point_mins.append(100.0)  # SMAP亮温典型范围
                point_maxs.append(300.0)
        else:
            point_mins.append(100.0)
            point_maxs.append(300.0)

        # TBH统计
        if smap_tbh_all:
            smap_tbh_all = np.array(smap_tbh_all)
            valid_smap_tbh = smap_tbh_all[np.isfinite(smap_tbh_all)]
            if len(valid_smap_tbh) > 0:
                point_mins.append(float(np.min(valid_smap_tbh)))
                point_maxs.append(float(np.max(valid_smap_tbh)))
            else:
                point_mins.append(100.0)
                point_maxs.append(300.0)
        else:
            point_mins.append(100.0)
            point_maxs.append(300.0)

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
        print(
            f"  点特征: {self.C_point} 个维度 (LS: {self.ls_data.shape[0]}, S1_VV, S1_VH, SMAP_TBV, SMAP_TBH, lon, lat, doy)")
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

        # 2. 哨兵1特征 (VV和VH)
        s1_vv, s1_vh = self._get_sentinel1_value(date_dt, r, c)
        point_features.append(float(s1_vv))
        point_features.append(float(s1_vh))

        # 3. SMAP亮温特征 (TBV和TBH)
        smap_tbv, smap_tbh = self._get_smap_value(date_dt, r, c)
        point_features.append(float(smap_tbv))
        point_features.append(float(smap_tbh))

        # 4. 经纬度特征
        lon, lat = self._pixel_to_lonlat(r, c)
        # 归一化经纬度
        lon_norm = (lon + 180) / 360  # 假设经度范围-180到180
        lat_norm = (lat + 90) / 180  # 假设纬度范围-90到90
        point_features.extend([lon_norm, lat_norm])

        # 5. 时间特征
        time_feats = self._build_time_features(date_dt)
        point_features.extend(time_feats)

        point_feats_array = np.array(point_features, dtype=np.float32)

        # 检查NaN和nodata值
        # 哨兵1
        if point_feats_array[self.ls_data.shape[0]] == self.s1_nodata_value:  # VV
            point_feats_array[self.ls_data.shape[0]] = 0.0
        if point_feats_array[self.ls_data.shape[0] + 1] == self.s1_nodata_value:  # VH
            point_feats_array[self.ls_data.shape[0] + 1] = 0.0

        # SMAP
        smap_vv_index = self.ls_data.shape[0] + 2  # TBV位置
        smap_vh_index = self.ls_data.shape[0] + 3  # TBH位置
        if point_feats_array[smap_vv_index] == self.smap_nodata_value:
            point_feats_array[smap_vv_index] = 0.0
        if point_feats_array[smap_vh_index] == self.smap_nodata_value:
            point_feats_array[smap_vh_index] = 0.0

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

    def _get_microwave_features(self, date_dt: datetime, r: int, c: int) -> Dict[str, float]:
        """获取所有微波特征（哨兵1和SMAP）"""
        features = {}

        # 获取哨兵1特征
        s1_vv, s1_vh = self._get_sentinel1_value(date_dt, r, c)
        features['S1_VV'] = s1_vv
        features['S1_VH'] = s1_vh

        # 获取SMAP特征
        smap_tbv, smap_tbh = self._get_smap_value(date_dt, r, c)
        features['SMAP_TBV'] = smap_tbv
        features['SMAP_TBH'] = smap_tbh

        return features

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
                     (torch.from_numpy(self.conv_max + eps).view(-1, 1, 1) - torch.from_numpy(self.conv_min).view(-1, 1,
                                                                                                                  1))

            point_t = (point_t - torch.from_numpy(self.point_min)) / \
                      (torch.from_numpy(self.point_max + eps) - torch.from_numpy(self.point_min))

            y_t = (y_t - self.label_min) / (self.label_max - self.label_min)

            return conv_t, point_t, y_t

        raise IndexError(f"在idx={idx}附近连续{max_retry}个样本均无效")


# 以下代码保持不变...
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


def build_temporal_split_dataloaders(
        train_years: List[int] = [2015],  # 改为列表形式，支持多年份
        val_years: List[int] = [2016],
        batch_size: int = 32,
        num_workers: int = 0,
        **dataset_kwargs
):
    """按年份划分数据加载器 - 支持多年份"""
    print(f"\n[按年份划分] 训练年份: {train_years}, 验证年份: {val_years}")

    try:
        # 只保留 SWEDataset 需要的参数
        swedataset_params = [
            'region', 'year_target', 'feature_root', 'label_root',
            'patch_size', 'min_valid_pixels', 'samples_per_day', 'clamday_threshold'
        ]

        dataset_kwargs_filtered = {}
        for param in swedataset_params:
            if param in dataset_kwargs:
                dataset_kwargs_filtered[param] = dataset_kwargs[param]

        # 确保必须有 year_target
        if 'year_target' not in dataset_kwargs_filtered:
            dataset_kwargs_filtered['year_target'] = max(val_years)  # 使用验证集中最大的年份

        print(f"SWEDataset 参数: {list(dataset_kwargs_filtered.keys())}")

        # 创建数据集
        dataset = SWEDataset(**dataset_kwargs_filtered)

        # 收集训练集和验证集索引
        train_indices = []
        val_indices = []

        print("筛选按年份划分的样本...")

        # 统计各年份的样本
        year_counts = {}
        for idx in range(len(dataset)):
            date_dt, r, c = dataset.meta_index[idx]
            year = date_dt.year
            year_counts[year] = year_counts.get(year, 0) + 1

        print(f"数据集中各年份样本数:")
        for year, count in sorted(year_counts.items()):
            print(f"  {year}年: {count} 个样本")

        # 如果没有2015年数据但有2014年，可以调整
        if 2015 not in year_counts and 2014 in year_counts and train_years == [2015]:
            print(f"警告: 没有2015年数据，自动使用2014年作为训练集")
            train_years = [2014]

        for idx in range(len(dataset)):
            date_dt, r, c = dataset.meta_index[idx]
            year = date_dt.year

            if year in train_years:
                train_indices.append(idx)
            elif year in val_years:
                val_indices.append(idx)

        print(f"  训练集样本数: {len(train_indices)}")
        print(f"  验证集样本数: {len(val_indices)}")

        if len(train_indices) == 0:
            print(f"警告: 没有找到训练年份 {train_years} 的样本")
            # 如果训练集为空，使用验证集的一部分作为训练集
            if len(val_indices) > 0:
                print("使用验证集的一部分作为训练集")
                split_point = len(val_indices) // 2
                train_indices = val_indices[:split_point]
                val_indices = val_indices[split_point:]
                print(f"  重新划分: train={len(train_indices)}, val={len(val_indices)}")
            else:
                raise ValueError(f"没有找到任何有效样本")

        if len(val_indices) == 0:
            print(f"警告: 没有找到验证年份 {val_years} 的样本")
            # 如果验证集为空，使用训练集的一部分作为验证集
            if len(train_indices) > 0:
                print("使用训练集的一部分作为验证集")
                split_point = len(train_indices) // 5  # 20%作为验证集
                val_indices = train_indices[:split_point]
                train_indices = train_indices[split_point:]
                print(f"  重新划分: train={len(train_indices)}, val={len(val_indices)}")

        # 打乱顺序
        np.random.seed(dataset_kwargs.get('seed', 42))
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        # 创建子集
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # 获取特征维度
        C_conv = dataset.C_conv
        C_point = dataset.C_point

        return train_loader, val_loader, (C_conv, C_point)

    except Exception as e:
        print(f"创建按年份划分的数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def build_spatial_split_dataloaders(
        spatial_split_ratio: float = 0.2,
        split_by: str = 'rows',  # 'rows', 'cols', 'blocks'
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        **dataset_kwargs
):
    """按空间区域划分数据加载器"""
    print(f"\n[按空间划分] 方式: {split_by}, 验证比例: {spatial_split_ratio}")

    try:
        dataset = SWEDataset(**dataset_kwargs)

        # 获取所有样本的空间位置
        print("收集空间位置信息...")
        locations = {}
        for idx in range(len(dataset)):
            date_dt, r, c = dataset.meta_index[idx]
            loc_key = (r, c)
            if loc_key not in locations:
                locations[loc_key] = []
            locations[loc_key].append(idx)

        unique_locations = list(locations.keys())
        print(f"  唯一空间位置数: {len(unique_locations)}")

        # 按不同方式划分
        np.random.seed(seed)

        if split_by == 'rows':
            # 按行划分
            all_rows = sorted(set(r for r, c in unique_locations))
            n_val_rows = int(len(all_rows) * spatial_split_ratio)
            val_rows = set(np.random.choice(all_rows, n_val_rows, replace=False))

            train_indices = []
            val_indices = []

            for (r, c), idx_list in locations.items():
                if r in val_rows:
                    val_indices.extend(idx_list)
                else:
                    train_indices.extend(idx_list)

        elif split_by == 'cols':
            # 按列划分
            all_cols = sorted(set(c for r, c in unique_locations))
            n_val_cols = int(len(all_cols) * spatial_split_ratio)
            val_cols = set(np.random.choice(all_cols, n_val_cols, replace=False))

            train_indices = []
            val_indices = []

            for (r, c), idx_list in locations.items():
                if c in val_cols:
                    val_indices.extend(idx_list)
                else:
                    train_indices.extend(idx_list)

        elif split_by == 'blocks':
            # 按块划分
            block_size = 50  # 50x50像素块
            H, W = dataset.H, dataset.W

            blocks = []
            for i in range(0, H, block_size):
                for j in range(0, W, block_size):
                    block_indices = []
                    for (r, c), idx_list in locations.items():
                        if i <= r < i + block_size and j <= c < j + block_size:
                            block_indices.extend(idx_list)
                    if block_indices:
                        blocks.append(block_indices)

            np.random.shuffle(blocks)
            n_val_blocks = int(len(blocks) * spatial_split_ratio)

            val_indices = []
            for i in range(n_val_blocks):
                val_indices.extend(blocks[i])

            train_indices = []
            for i in range(n_val_blocks, len(blocks)):
                train_indices.extend(blocks[i])

        else:
            raise ValueError(f"不支持的划分方式: {split_by}")

        print(f"  训练集样本数: {len(train_indices)}")
        print(f"  验证集样本数: {len(val_indices)}")

        # 创建子集和数据加载器...
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, (dataset.C_conv, dataset.C_point)

    except Exception as e:
        print(f"创建空间划分数据加载器失败: {e}")
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
        dataset = SWEDataset(s1_interp_method="nearest", s1_max_gap_days=7)

        # 测试获取样本
        print(f"\n2. 测试获取样本...")
        if len(dataset) > 0:
            for i in range(min(3, len(dataset))):
                print(f"\n  样本 {i}:")
                try:
                    conv, point, y = dataset[i]
                    print(f"    conv shape: {conv.shape}")
                    print(f"    point shape: {point.shape}")
                    print(
                        f"    哨兵1特征位置: [{point[dataset.ls_data.shape[0]]:.4f}, {point[dataset.ls_data.shape[0] + 1]:.4f}]")
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