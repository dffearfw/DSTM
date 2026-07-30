# data_online_era5_swe.py
# -*- coding: utf-8 -*-
"""
在线从栅格构建 SWE 反演训练样本
- 卷积特征: chelsa_sfxwind, lst, rh, clamday, dem
- 点特征: ls, S1_VV, S1_VH, 经纬度, doy
- 标签: fusedSWE
"""
import os
from sklearn.model_selection import KFold
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
import rasterio
from datetime import datetime, timedelta, timedelta
from pyproj import Transformer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import calendar
from bisect import bisect_left
import re
from collections import defaultdict
import pickle
import hashlib
import json
import glob
import pandas as pd
from scipy.interpolate import griddata
import time
import psutil
import gc
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
# ============= 配置区域 =============
REGION = "CHINA"
YEAR_TARGET = 2016
PATCH_SIZE = 5  # patch大小
R = PATCH_SIZE // 2

# ============================================================
# ERA5-Land glacier / extreme SWE artifact filter
# 冰川格点在 HTESSEL/ERA5-Land 中可能被赋予约 10 m SWE。
# QGIS 中可能显示为 9900+ mm，不一定严格等于 10000。
# 所以这里用阈值过滤，而不是 ==10000。
# ============================================================
FILTER_GLACIER_SWE_ARTIFACTS = os.environ.get(
    "FILTER_GLACIER_SWE_ARTIFACTS", "1"
) == "1"

GLACIER_SWE_THRESHOLD_MM = float(
    os.environ.get("GLACIER_SWE_THRESHOLD_MM", "2000.0")
)

MIN_VALID_PIXELS = 50
SAMPLES_PER_DAY = 50000

# 卷积特征（参与卷积的变量）
CONV_VARS = ["chelsa_sfxwind", "lst", "rh", "pr"]
CONV_STATIC_VARS = ["clamday", "dem"]  # 静态卷积特征

# 点特征（不参与卷积）
POINT_VARS = ["ls", "S1_VV", "S1_VH", "SMAP_TBV", "SMAP_TBH"]  # 添加哨兵1和SMAP亮温

# 数据路径
FEATURE_ROOT = Path(r"/root/ablation")
LABEL_ROOT = Path(r"/root/ablation/era5landswe")  # 标签路径


def conv_var_path(var: str, year: int) -> Path:
    """卷积变量路径"""
    if var == "chelsa_sfxwind":
        return FEATURE_ROOT / "sfxwind" / "cn"  # 改为cn文件夹
    elif var == "lst":
        return FEATURE_ROOT / "lst" / "cn"  # 改为cn文件夹
    elif var == "rh":
        return FEATURE_ROOT / "rh" / "cn"  # 改为cn文件夹
    elif var == "pr":
        return FEATURE_ROOT / "pr" / "cn"  # 新增pr路径
    else:
        raise ValueError(f"未知的卷积变量: {var}")

def _as_year_list(year):
    """把 year / [years] 统一成 list[int]。"""
    if isinstance(year, (list, tuple, set)):
        return [int(y) for y in year]
    return [int(year)]


def _threshold_tokens(threshold: float) -> List[str]:
    """同时兼容 threshold0.5 和 threshold0p5 这两类命名。"""
    if threshold is None:
        return []
    raw = f"{float(threshold):g}"
    return [raw, raw.replace('.', 'p')]


def _glob_unique(root: Path, patterns: List[str]) -> List[Path]:
    """按多个 pattern 查找并去重排序。"""
    files = []
    for pat in patterns:
        files.extend(root.glob(pat))
    return sorted(set(files), key=lambda p: p.name)


def conv_static_path(var: str, year, threshold: float = 0.5) -> List[Path]:
    """静态卷积变量路径：适配新 ablation/clamday 与 DEM_terrain_8band 命名。"""
    if var == "clamday":
        clamday_path = FEATURE_ROOT / "clamday" / "cn"
        if not clamday_path.exists():
            print(f"    ⚠ Clamday目录不存在: {clamday_path}")
            return []

        years = _as_year_list(year)
        pats = []
        for y in years:
            for token in _threshold_tokens(threshold):
                pats += [
                    f"China_CalmDays_{y}_threshold{token}.tif",
                    f"*{y}*threshold{token}*.tif",
                ]
            pats.append(f"*{y}*CalmDays*.tif")
            pats.append(f"*{y}*.tif")

        files = _glob_unique(clamday_path, pats)
        if not files:
            files = _glob_unique(clamday_path, ["*CalmDays*.tif", "*.tif"])

        print(f"    Clamday文件: 找到 {len(files)} 个")
        if files:
            print(f"      例如: {files[0].name}")
        return files

    elif var == "dem":
        dem_path = FEATURE_ROOT / "DEM"
        if not dem_path.exists():
            print(f"    ⚠ DEM目录不存在: {dem_path}")
            return []

        def _not_std(p: Path) -> bool:
            name = p.name.lower()
            return ("std" not in name) and ("stddev" not in name)

        # Terrain raw 8 band：必须排除 STD/StdDev
        terrain_candidates = _glob_unique(dem_path, [
            "DEM_terrain_8band_aligned_fixedDomain.tif",
            "DEM_terrain_8band_aligned.tif",
            "*_Terrain_China_27830m.tif",
            "*Terrain*.tif",
            "*terrain*8band*aligned*.tif",
        ])
        terrain_files = [p for p in terrain_candidates if _not_std(p)]

        # STD 8 band：只找 STD/StdDev
        stddev_files = _glob_unique(dem_path, [
            "DEM_terrain_8band_STD_aligned_fixedDomain.tif",
            "DEM_terrain_8band_STD_aligned.tif",
            "*terrain*8band*STD*aligned*.tif",
            "*_Terrain_StdDev_China_27830m.tif",
            "*StdDev*.tif",
        ])

        dem_files = []

        if terrain_files:
            dem_files.append(terrain_files[0])
            print(f"    找到DEM Terrain文件: {terrain_files[0].name}")
        else:
            print("    ⚠ 未找到DEM Terrain raw文件")

        if stddev_files:
            # 避免同一个文件重复加入
            if not dem_files or stddev_files[0] != dem_files[0]:
                dem_files.append(stddev_files[0])
                print(f"    找到DEM StdDev文件: {stddev_files[0].name}")
            else:
                print(f"    ⚠ StdDev文件与Terrain文件重复，已跳过: {stddev_files[0].name}")
        else:
            print("    ⚠ 未找到DEM StdDev文件")

        print(f"    DEM文件: 找到 {len(dem_files)} 个有效文件 (共 {len(dem_files) * 8} band)")
        return dem_files

    else:
        raise ValueError(f"未知的静态卷积变量: {var}")

def point_var_path(var: str, year) -> Union[Path, List[Path]]:
    """点变量路径：适配 Landsat/S1/SMAP 新命名。"""

    if var == "ls":
        ls_path = FEATURE_ROOT / "ls" / "cn"
        if not ls_path.exists():
            print(f"    LS目录不存在: {ls_path}")
            return [] if isinstance(year, list) else None

        years = _as_year_list(year)
        files = []
        for y in years:
            files.extend(_glob_unique(ls_path, [
                f"China_Landsat_L8C2L2_{y}_AnnualMedian_ERA5SWE010deg.tif",
                f"China_Landsat*{y}*AnnualMedian*.tif",
                f"China_Landsat_{y}_reflectance.tif",
                f"*Landsat*{y}*.tif",
            ]))
        files = sorted(set(files), key=lambda p: p.name)

        if isinstance(year, list):
            print(f"    LS文件: 找到 {len(files)} 个")
            return files
        if files:
            print(f"    找到LS文件: {files[0].name}")
            return files[0]

        print(f"    ⚠ 未找到LS文件")
        return None

    elif var == "S1_VV" or var == "S1_VH":
        s1_path = FEATURE_ROOT / "s1" / "cn"
        if not s1_path.exists():
            print(f"    哨兵1目录不存在: {s1_path}")
            return []

        files = []
        for y in _as_year_list(year):
            files.extend(_glob_unique(s1_path, [
                f"S1_{y}??_ERA5SWE010deg_5band_bilinear_chinaDomain_monthly.tif",
                f"S1_{y}??_*.tif",
                f"S1_MONTHLY_{y}_*.tif",
                f"*S1*{y}*.tif",
            ]))
        files = sorted(set(files), key=lambda p: p.name)
        print(f"    哨兵1文件: 最终找到 {len(files)} 个")
        if files:
            print(f"      日期范围: {files[0].name} 到 {files[-1].name}")
        return files

    elif var == "SMAP_TBV" or var == "SMAP_TBH":
        smap_root = FEATURE_ROOT / "smap" / "cn"
        if not smap_root.exists():
            print(f"    SMAP目录不存在: {smap_root}")
            return []

        files = []
        for y in _as_year_list(year):
            files.extend(_glob_unique(smap_root, [
                f"SMAP_{y}??_AMTB_11132m_ERA5SWE010deg_float32_chinaDomain.tif",
                f"SMAP_{y}??_*.tif",
                f"SMAP_{y}_*_cube_drive-download-20260.tif",
                f"*SMAP*{y}*.tif",
            ]))
        files = sorted(set(files), key=lambda p: p.name)
        print(f"    SMAP文件: 最终找到 {len(files)} 个")
        if files:
            print(f"      日期范围: {files[0].name} 到 {files[-1].name}")
        return files

    else:
        raise ValueError(f"未知的点变量: {var}")


class SWEDataset(Dataset):
    def __init__(
            self,
            region: str = REGION,
            year_target: Union[int, List[int]] = YEAR_TARGET,
            feature_root: Path = FEATURE_ROOT,
            label_root: Path = LABEL_ROOT,
            patch_size: int = PATCH_SIZE,
            min_valid_pixels: int = MIN_VALID_PIXELS,
            samples_per_day: int = SAMPLES_PER_DAY,
            clamday_threshold: float = 0.5,
            s1_interp_method: str = "nearest",
            s1_max_gap_days: int = 7,
            s1_nodata_value: float = -9999.0,
            smap_interp_method: str = "nearest",
            smap_max_gap_days: int = 7,
            smap_nodata_value: float = -9999.0,
            use_tta: bool = False,
            cache_dir: Optional[Path] = None,  
            force_reload: bool = False,
            # 采样来源：auto=兼容旧逻辑；random=仅全国随机；
            # station=仅站点位置；hybrid=随机+站点。
            sampling_mode: str = "auto",
            use_station_guide: bool = False,
            # 推荐直接传具体文件；未传时才兼容旧 station_csv_dir 多文件逻辑。
            station_guide_file: Optional[Path] = None,
            station_csv_dir: Optional[Path] = None,
            station_neighborhood: int = 3,
            # <=0 表示每天使用全部有效站点格点。
            station_samples_per_day: int = 2000,
            station_filter_zero_target: bool = True,
            # positions_all_dates：旧逻辑，仅使用站点位置并遍历全部标签日期；
            # records：只使用站点文件中实际存在的“站点-日期”记录。
            station_sampling_unit: str = "positions_all_dates",
            # records 模式默认按 (date, ERA5 row, ERA5 col) 去重，避免完全相同
            # 的栅格样本同时进入十折训练集和验证集。
            station_record_dedup: str = "grid_date",
            station_date_column: Optional[str] = None,
            # 可选：正式Stage 0直接读取预先冻结的有效清单。
            station_record_manifest_path: Optional[Path] = None,
            # ============ 外部测试站点空间隔离 ============
            external_station_glob: Optional[str] = None,
            external_station_exclusion_radius: int = 0,
            external_station_strict: bool = False,
            external_station_report_path: Optional[Path] = None,
            # ============ 固定增量样本池 ============
            # incremental 模式只读取 manifest 中指定 stage 的新增样本。
            incremental_manifest_path: Optional[Path] = None,
            incremental_stage: int = 1,
            incremental_selection_mode: str = "package",
            incremental_exclude_manifest_path: Optional[Path] = None,
            build_incremental_manifest: bool = False,
            incremental_pool_size: int = 152000,
            incremental_stage_sizes: Optional[List[int]] = None,
            incremental_seed: int = 43,
            incremental_candidate_oversample_factor: float = 3.0,
            incremental_exclude_station_pixels: bool = True,
            incremental_ratio_config: Optional[Path] = None,
            incremental_glacier_mask_path: Optional[Path] = None,
            incremental_fold_block_pixels: int = 0,
            # ============ 季节性积雪判据（仅 incremental 随机池） ============
            seasonal_min_peak_swe_mm: float = 1.0,
            seasonal_max_swe_mm: float = 400.0,
            seasonal_snow_free_threshold_mm: float = 1.0,
            seasonal_min_warm_snow_free_ratio: float = 0.0,
            seasonal_min_consecutive_snow_free_days: int = 5,
            seasonal_min_snow_year_coverage_ratio: float = 0.90,
            # ============ 跨阶段固定归一化 ============
            normalization_config_path: Optional[Path] = None,
            normalization_mode: str = "auto",
            fixed_label_min_mm: Optional[float] = 0.0,
            fixed_label_max_mm: Optional[float] = 400.0,
            use_adaptive_supplement: bool = False,
            adaptive_alpha: float = 0.5,
            adaptive_threshold: float = 1.5,
            adaptive_swe_bins: Optional[List[float]] = None,
    ):
        super().__init__()

        # ============ 初始化标志位 ============
        self.fine_tune_mode = False
        self.load_fused_swe = True

        # ============ 基础参数 ============
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

        self.s1_interp_method = s1_interp_method
        self.s1_max_gap_days = s1_max_gap_days
        self.s1_nodata_value = s1_nodata_value

        self.smap_interp_method = smap_interp_method
        self.smap_max_gap_days = smap_max_gap_days
        self.smap_nodata_value = smap_nodata_value
        self.use_tta = use_tta

        sampling_mode = str(sampling_mode).strip().lower()
        valid_sampling_modes = {"auto", "random", "station", "hybrid", "incremental"}
        if sampling_mode not in valid_sampling_modes:
            raise ValueError(
                f"sampling_mode 必须是 {sorted(valid_sampling_modes)} 之一，当前={sampling_mode!r}"
            )

        # auto 保持旧命令兼容：传了 --use_station_guide 即 hybrid，否则 random。
        if sampling_mode == "auto":
            sampling_mode = "hybrid" if use_station_guide else "random"

        self.sampling_mode = sampling_mode
        self.use_station_guide = sampling_mode in {"station", "hybrid"}
        self.station_guide_file = (
            Path(station_guide_file).expanduser() if station_guide_file else None
        )
        self.station_csv_dir = Path(
            station_csv_dir or "/root/ablation"
        ).expanduser()
        self.station_neighborhood = int(station_neighborhood)
        if self.station_neighborhood < 0:
            raise ValueError("station_neighborhood 不能小于0")
        self.station_samples_per_day = int(station_samples_per_day)
        self.station_filter_zero_target = bool(station_filter_zero_target)
        self.station_sampling_unit = str(station_sampling_unit).strip().lower()
        if self.station_sampling_unit not in {"positions_all_dates", "records"}:
            raise ValueError(
                "station_sampling_unit 必须是 positions_all_dates 或 records，"
                f"当前={self.station_sampling_unit!r}"
            )
        self.station_record_dedup = str(station_record_dedup).strip().lower()
        if self.station_record_dedup not in {"grid_date", "none"}:
            raise ValueError(
                "station_record_dedup 必须是 grid_date 或 none，"
                f"当前={self.station_record_dedup!r}"
            )
        self.station_date_column = (
            str(station_date_column).strip() if station_date_column else None
        )
        self.station_record_manifest_path = (
            Path(station_record_manifest_path).expanduser()
            if station_record_manifest_path else None
        )
        self.station_pixels = set()
        self.station_record_samples = []
        self.station_record_stats = {}

        self.external_station_glob = (
            str(external_station_glob).strip() if external_station_glob else None
        )
        self.external_station_exclusion_radius = int(external_station_exclusion_radius)
        if self.external_station_exclusion_radius < 0:
            raise ValueError("external_station_exclusion_radius 不能小于0")
        self.external_station_strict = bool(external_station_strict)
        self.external_station_report_path = (
            Path(external_station_report_path).expanduser()
            if external_station_report_path else None
        )
        self.external_station_centers = set()
        self.external_excluded_cells = set()
        self.external_station_stats = {}

        # ============ 固定增量样本池配置 ============
        self.incremental_manifest_path = (
            Path(incremental_manifest_path).expanduser()
            if incremental_manifest_path else None
        )
        self.incremental_stage = int(incremental_stage)
        self.incremental_selection_mode = str(
            incremental_selection_mode
        ).strip().lower()
        if self.incremental_selection_mode not in {"package", "cumulative"}:
            raise ValueError(
                "incremental_selection_mode必须为package或cumulative，"
                f"当前={self.incremental_selection_mode}"
            )
        self.incremental_exclude_manifest_path = (
            Path(incremental_exclude_manifest_path).expanduser()
            if incremental_exclude_manifest_path else None
        )
        self.build_incremental_manifest = bool(build_incremental_manifest)
        self.incremental_pool_size = int(incremental_pool_size)
        self.incremental_stage_sizes = [
            int(x) for x in (incremental_stage_sizes or [12000, 20000, 40000, 80000])
        ]
        self.incremental_seed = int(incremental_seed)
        self.incremental_candidate_oversample_factor = float(
            incremental_candidate_oversample_factor
        )
        self.incremental_exclude_station_pixels = bool(
            incremental_exclude_station_pixels
        )
        self.incremental_ratio_config = (
            Path(incremental_ratio_config).expanduser()
            if incremental_ratio_config else None
        )
        self.incremental_glacier_mask_path = (
            Path(incremental_glacier_mask_path).expanduser()
            if incremental_glacier_mask_path else None
        )
        self.incremental_fold_block_pixels = max(0, int(incremental_fold_block_pixels))

        if self.sampling_mode == "incremental":
            if self.incremental_manifest_path is None:
                raise ValueError(
                    "sampling_mode=incremental 时必须提供 incremental_manifest_path"
                )
            if self.incremental_stage not in range(1, len(self.incremental_stage_sizes) + 1):
                raise ValueError(
                    f"incremental_stage 必须为 1-{len(self.incremental_stage_sizes)}，"
                    f"当前={self.incremental_stage}"
                )
            if sum(self.incremental_stage_sizes) != self.incremental_pool_size:
                raise ValueError(
                    "incremental_stage_sizes 之和必须等于 incremental_pool_size："
                    f"{self.incremental_stage_sizes} -> {sum(self.incremental_stage_sizes)}，"
                    f"pool={self.incremental_pool_size}"
                )

        # 季节性积雪判据只用于 incremental 随机样本池；station 模式完全不调用。
        self.seasonal_min_peak_swe_mm = float(seasonal_min_peak_swe_mm)
        self.seasonal_max_swe_mm = float(seasonal_max_swe_mm)
        self.seasonal_snow_free_threshold_mm = float(
            seasonal_snow_free_threshold_mm
        )
        self.seasonal_min_warm_snow_free_ratio = float(
            seasonal_min_warm_snow_free_ratio
        )
        self.seasonal_min_consecutive_snow_free_days = int(
            seasonal_min_consecutive_snow_free_days
        )
        self.seasonal_min_snow_year_coverage_ratio = float(
            seasonal_min_snow_year_coverage_ratio
        )

        if self.seasonal_max_swe_mm <= self.seasonal_min_peak_swe_mm:
            raise ValueError("seasonal_max_swe_mm 必须大于 seasonal_min_peak_swe_mm")
        if not (0.0 <= self.seasonal_min_warm_snow_free_ratio <= 1.0):
            raise ValueError("seasonal_min_warm_snow_free_ratio 必须在 [0,1]")
        # 兼容旧命令保留；当前不参与季节性硬筛选。
        if not (0.0 < self.seasonal_min_snow_year_coverage_ratio <= 1.0):
            raise ValueError("seasonal_min_snow_year_coverage_ratio 必须在 (0,1]")

        # 各阶段必须使用同一份归一化配置。
        self.normalization_config_path = (
            Path(normalization_config_path).expanduser()
            if normalization_config_path else None
        )
        self.normalization_mode = str(normalization_mode).strip().lower()
        if self.normalization_mode not in {"auto", "create", "load", "skip"}:
            raise ValueError("normalization_mode 必须是 auto/create/load/skip")
        self.fixed_label_min_mm = (
            None if fixed_label_min_mm is None else float(fixed_label_min_mm)
        )
        self.fixed_label_max_mm = (
            None if fixed_label_max_mm is None else float(fixed_label_max_mm)
        )
        self.sample_fold_ids = None

        self.use_adaptive_supplement = use_adaptive_supplement
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_swe_bins = adaptive_swe_bins or [0, 5, 10, 20, 30, 50, 80, 120, 200, 500, 800]

        # ============================================================
        # [SAMPLING-CONTRACT] 目标比例采样
        # ============================================================
        # 当前正式预训练默认启用目标比例采样，而不是按 0.3x / 5x 权重抽样。
        # 精确 SWE=0 单独作为一个 bin，目标比例为 3%。
        # TARGET_TOTAL_SAMPLES 控制最终数据集规模；默认约 16 万。
        self.use_target_quota_sampling = os.environ.get(
            "USE_TARGET_QUOTA_SAMPLING", "1"
        ) == "1"
        self.target_total_samples = int(
            os.environ.get("TARGET_TOTAL_SAMPLES", "160000")
        )
        # [DEFAULT-2026] 正式目标比例：低值 / 中值 / 高值约各占三分之一。
        #   SWE <= 5 mm : 33%
        #   5 < SWE <= 30 mm : 33%
        #   SWE > 30 mm : 34%
        #
        # [DANGER] 仅设置 quota 不足以保证最终达到目标。
        # 当前流程会在第一次候选生成后，对短缺区间执行第二遍定向补采；
        # 严格模式下禁止再用低值剩余样本回填高值短缺。
        self.target_swe_ratios = {
            "zero": 0.03,
            "0_1": 0.10,
            "1_5": 0.20,
            "5_10": 0.11,
            "10_20": 0.12,
            "20_30": 0.10,
            "30_50": 0.12,
            "50_80": 0.09,
            "80_120": 0.06,
            "120_200": 0.04,
            "200_500": 0.02,
            "500_plus": 0.01,
        }

        # 二次定向补采：只补当前候选池中低于 quota 的 SWE 区间。
        self.use_quota_shortage_supplement = os.environ.get(
            "USE_QUOTA_SHORTAGE_SUPPLEMENT", "1"
        ) == "1"
        self.quota_supplement_try_factor = float(
            os.environ.get("QUOTA_SUPPLEMENT_TRY_FACTOR", "6.0")
        )
        self.quota_supplement_max_rounds = int(
            os.environ.get("QUOTA_SUPPLEMENT_MAX_ROUNDS", "3")
        )
        self.strict_target_quota = os.environ.get(
            "STRICT_TARGET_QUOTA", "1"
        ) == "1"
        ratio_sum = sum(self.target_swe_ratios.values())
        if not np.isclose(ratio_sum, 1.0, atol=1e-8):
            raise ValueError(f"target_swe_ratios 之和必须为1，当前为 {ratio_sum}")

        # [COMPAT] 非目标比例模式仍保留旧的零值封顶逻辑。
        self.max_zero_target_ratio = self.target_swe_ratios["zero"]

        print(f"\n📌 采样配置:")
        print(f"   sampling_mode: {self.sampling_mode}")
        print(f"   站点引导: {'启用' if self.use_station_guide else '禁用'}")
        if self.use_station_guide:
            print(f"   站点文件: {self.station_guide_file or self.station_csv_dir}")
            print(f"   站点邻域半径: {self.station_neighborhood}")
            limit_text = "全部" if self.station_samples_per_day <= 0 else str(self.station_samples_per_day)
            print(f"   每日站点样本上限: {limit_text}")
            print(f"   过滤 ERA5-Land SWE=0: {self.station_filter_zero_target}")
        if self.external_station_glob:
            print(f"   外部测试站点: {self.external_station_glob}")
            print(f"   外部站点排除半径: {self.external_station_exclusion_radius}格")
            print(f"   外部站点严格模式: {self.external_station_strict}")
        if self.sampling_mode == "incremental":
            print(f"   增量清单: {self.incremental_manifest_path}")
            print(f"   当前增量阶段: {self.incremental_stage}")
            print(f"   固定总池: {self.incremental_pool_size:,}")
            print(f"   阶段大小: {self.incremental_stage_sizes}")
            print(f"   构建/覆盖清单: {self.build_incremental_manifest}")
            print(f"   随机池排除站点格点: {self.incremental_exclude_station_pixels}")
            print(
                f"   季节性判据: 年最大SWE严格位于 "
                f"({self.seasonal_min_peak_swe_mm}, {self.seasonal_max_swe_mm}) mm；"
                f"最后一次年峰值后连续近无雪≥"
                f"{self.seasonal_min_consecutive_snow_free_days}天；"
                f"近无雪阈值≤{self.seasonal_snow_free_threshold_mm} mm"
            )
        print(f"   目标比例采样: {'启用' if self.use_target_quota_sampling else '禁用'}")
        if self.use_target_quota_sampling:
            low_ratio = (
                self.target_swe_ratios['zero']
                + self.target_swe_ratios['0_1']
                + self.target_swe_ratios['1_5']
            )
            high_keys = ['30_50', '50_80', '80_120', '120_200', '200_500', '500_plus']
            high_ratio = sum(self.target_swe_ratios[k] for k in high_keys)
            print(f"   最终目标样本数: {self.target_total_samples:,}")
            print(f"   SWE=0 目标比例: {self.target_swe_ratios['zero']*100:.1f}%")
            print(f"   0-5 mm 合计目标比例: {low_ratio*100:.1f}%")
            print(f"   >30 mm 合计目标比例: {high_ratio*100:.1f}%")
            print(f"   短缺区间二次补采: {'启用' if self.use_quota_shortage_supplement else '禁用'}")
            print(f"   严格目标比例: {'启用' if self.strict_target_quota else '禁用'}")
            print("   quota 模式下旧 adaptive supplement 不参与最终分布控制")
        else:
            print(f"   自适应修正: {'启用' if use_adaptive_supplement else '禁用'}")
            if use_adaptive_supplement:
                print(f"   平衡强度 α = {adaptive_alpha}")
                print(f"   短缺阈值 = {adaptive_threshold}")
                print(f"   SWE区间: {self.adaptive_swe_bins}")

        if isinstance(year_target, list):
            self.load_years = [int(y) for y in year_target]
        else:
            self.load_years = [int(year_target)]

        # ============ 缓存逻辑 ============
        self.cache_dir = cache_dir
        cache_path = None
        cache_key = None

        if cache_dir is not None:
            import hashlib
            import json

            cache_dir_path = Path(cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)

            cache_params = {
                'region': region,
                'year_target': year_target,
                'load_years': self.load_years,
                'feature_root': str(feature_root),
                'label_root': str(label_root),
                'patch_size': patch_size,
                'min_valid_pixels': min_valid_pixels,
                'samples_per_day': samples_per_day,
                'clamday_threshold': clamday_threshold,
                's1_nodata_value': s1_nodata_value,
                'smap_nodata_value': smap_nodata_value,
                'sampling_mode': self.sampling_mode,
                'use_station_guide': self.use_station_guide,
                'station_guide_file': str(self.station_guide_file) if self.station_guide_file else None,
                'station_guide_file_size': (
                    self.station_guide_file.stat().st_size
                    if self.station_guide_file and self.station_guide_file.exists() else None
                ),
                'station_guide_file_mtime_ns': (
                    self.station_guide_file.stat().st_mtime_ns
                    if self.station_guide_file and self.station_guide_file.exists() else None
                ),
                'station_csv_dir': str(self.station_csv_dir),
                'station_neighborhood': self.station_neighborhood,
                'station_samples_per_day': self.station_samples_per_day,
                'station_sampling_unit': self.station_sampling_unit,
                'station_record_dedup': self.station_record_dedup,
                'station_date_column': self.station_date_column,
                'station_record_manifest_path': str(self.station_record_manifest_path) if self.station_record_manifest_path else None,
                'station_record_manifest_size': (self.station_record_manifest_path.stat().st_size if self.station_record_manifest_path and self.station_record_manifest_path.exists() else None),
                'station_record_manifest_mtime_ns': (self.station_record_manifest_path.stat().st_mtime_ns if self.station_record_manifest_path and self.station_record_manifest_path.exists() else None),
                'external_station_glob': self.external_station_glob,
                'external_station_exclusion_radius': self.external_station_exclusion_radius,
                'external_station_strict': self.external_station_strict,
                'external_station_report_path': str(self.external_station_report_path) if self.external_station_report_path else None,
                'use_adaptive_supplement': use_adaptive_supplement,
                'adaptive_alpha': adaptive_alpha,
                'adaptive_threshold': adaptive_threshold,
                'adaptive_swe_bins': self.adaptive_swe_bins,
                # 冰川/永久冰雪异常 SWE 过滤参数也必须进入 cache key；
                # 否则 1000/2000 阈值切换时可能误读旧缓存。
                'filter_glacier_swe_artifacts': FILTER_GLACIER_SWE_ARTIFACTS,
                'glacier_swe_threshold_mm': GLACIER_SWE_THRESHOLD_MM,
                'max_zero_target_ratio': self.max_zero_target_ratio,
                'station_filter_zero_target': self.station_filter_zero_target,
                'incremental_manifest_path': str(self.incremental_manifest_path) if self.incremental_manifest_path else None,
                'incremental_manifest_size': (
                    self.incremental_manifest_path.stat().st_size
                    if self.incremental_manifest_path and self.incremental_manifest_path.exists() else None
                ),
                'incremental_manifest_mtime_ns': (
                    self.incremental_manifest_path.stat().st_mtime_ns
                    if self.incremental_manifest_path and self.incremental_manifest_path.exists() else None
                ),
                'incremental_stage': self.incremental_stage,
                'incremental_selection_mode': self.incremental_selection_mode,
                'incremental_exclude_manifest_path': (
                    str(self.incremental_exclude_manifest_path)
                    if self.incremental_exclude_manifest_path else None
                ),
                'incremental_exclude_manifest_size': (
                    self.incremental_exclude_manifest_path.stat().st_size
                    if self.incremental_exclude_manifest_path
                    and self.incremental_exclude_manifest_path.exists()
                    else None
                ),
                'build_incremental_manifest': self.build_incremental_manifest,
                'incremental_pool_size': self.incremental_pool_size,
                'incremental_stage_sizes': self.incremental_stage_sizes,
                'incremental_seed': self.incremental_seed,
                'incremental_candidate_oversample_factor': self.incremental_candidate_oversample_factor,
                'incremental_exclude_station_pixels': self.incremental_exclude_station_pixels,
                'incremental_ratio_config': str(self.incremental_ratio_config) if self.incremental_ratio_config else None,
                'incremental_glacier_mask_path': str(self.incremental_glacier_mask_path) if self.incremental_glacier_mask_path else None,
                'incremental_fold_block_pixels': self.incremental_fold_block_pixels,
                'seasonal_min_peak_swe_mm': self.seasonal_min_peak_swe_mm,
                'seasonal_max_swe_mm': self.seasonal_max_swe_mm,
                'seasonal_snow_free_threshold_mm': self.seasonal_snow_free_threshold_mm,
                'seasonal_min_warm_snow_free_ratio': self.seasonal_min_warm_snow_free_ratio,
                'seasonal_min_consecutive_snow_free_days': self.seasonal_min_consecutive_snow_free_days,
                'seasonal_min_snow_year_coverage_ratio': self.seasonal_min_snow_year_coverage_ratio,
                'normalization_config_path': str(self.normalization_config_path) if self.normalization_config_path else None,
                'normalization_config_size': (
                    self.normalization_config_path.stat().st_size
                    if self.normalization_config_path and self.normalization_config_path.exists() else None
                ),
                'normalization_config_mtime_ns': (
                    self.normalization_config_path.stat().st_mtime_ns
                    if self.normalization_config_path and self.normalization_config_path.exists() else None
                ),
                'normalization_mode': self.normalization_mode,
                'fixed_label_min_mm': self.fixed_label_min_mm,
                'fixed_label_max_mm': self.fixed_label_max_mm,
                'use_target_quota_sampling': self.use_target_quota_sampling,
                'target_total_samples': self.target_total_samples,
                'target_swe_ratios': self.target_swe_ratios,
                'use_quota_shortage_supplement': self.use_quota_shortage_supplement,
                'quota_supplement_try_factor': self.quota_supplement_try_factor,
                'quota_supplement_max_rounds': self.quota_supplement_max_rounds,
                'strict_target_quota': self.strict_target_quota,
            }
            cache_str = json.dumps(cache_params, sort_keys=True, default=str)
            cache_key = hashlib.md5(cache_str.encode()).hexdigest()[:16]
            cache_path = cache_dir_path / f"swe_dataset_{cache_key}.pkl"

            if not force_reload and cache_path.exists():
                print(f"\n📦 发现缓存文件: {cache_path}")
                print("   正在加载缓存...")
                try:
                    import pickle
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)

                    for key, value in cached_data.items():
                        setattr(self, key, value)

                    print("   ✅ 缓存加载成功！跳过数据加载")

                    self._setup_unified_grid()

                    if self.external_station_glob and (
                        not hasattr(self, 'external_excluded_cells')
                        or not self.external_excluded_cells
                    ):
                        print("   ⚠ 缓存缺少外部测试站点排除掩膜，正在重建...")
                        self._load_external_station_exclusion()

                    if not hasattr(self, 'date_to_index') or self.date_to_index is None:
                        print("   ⚠ 缓存缺少 date_to_index，正在重建...")
                        if hasattr(self, 'all_dates') and self.all_dates:
                            self.date_to_index = {d: i for i, d in enumerate(self.all_dates)}
                            print(f"   ✅ date_to_index 重建完成")
                        else:
                            raise AttributeError("缓存数据损坏")

                    if hasattr(self, 'conv_dyn_data'):
                        for var, arr in self.conv_dyn_data.items():
                            if arr is not None and arr.dtype != np.float32:
                                self.conv_dyn_data[var] = arr.astype(np.float32)

                    if hasattr(self, 'label_data'):
                        for date, (label_arr, label_nodata) in self.label_data.items():
                            if label_arr is not None and label_arr.dtype != np.float32:
                                self.label_data[date] = (label_arr.astype(np.float32), label_nodata)

                    if self.use_station_guide:
                        if self.station_sampling_unit == "records":
                            if (
                                not hasattr(self, 'station_record_samples')
                                or not self.station_record_samples
                            ):
                                print("   ⚠ 缓存缺少 station_record_samples，正在重建...")
                                if self.station_record_manifest_path is not None:
                                    self._load_station_record_samples_from_manifest()
                                else:
                                    self._load_station_record_samples()
                                print("   ✅ station_record_samples 重建完成")
                        elif not hasattr(self, 'station_pixels') or not self.station_pixels:
                            print("   ⚠ 缓存缺少 station_pixels，正在重建...")
                            self._load_all_station_pixels()
                            print(f"   ✅ station_pixels 重建完成")

                    self._validate_cached_data()

                    print(f"\n{'='*60}")
                    print(f"✅ 从缓存加载数据集完成!")
                    print(f"  总样本数: {len(self.meta_index):,}")
                    print(f"  卷积特征维度: {self.C_conv}")
                    print(f"  点特征维度: {self.C_point}")
                    print(f"{'='*60}\n")

                    # [DIAG] 只重新生成基于原始毫米标签的分布图，不重建16GB缓存。
                    # 用法：export REGENERATE_RAW_MM_DISTRIBUTION=1
                    if os.environ.get("REGENERATE_RAW_MM_DISTRIBUTION", "0") == "1":
                        print("\n📊 REGENERATE_RAW_MM_DISTRIBUTION=1")
                        print("   重新生成原始毫米单位的分布统计和倍率图...")
                        self._analyze_swe_distribution()

                    return

                except Exception as e:
                    print(f"   ⚠ 缓存加载失败: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        cache_path.unlink()
                    except:
                        pass

        # ============ 正常数据加载流程 ============
        print(f"\n{'='*60}")
        if isinstance(year_target, list):
            print(f"初始化数据集 (多年份模式):")
            print(f"  区域: {region}")
            print(f"  目标年份: {year_target}")
        else:
            print(f"初始化数据集 (单年份模式):")
            print(f"  目标年份: {year_target}")

        print(f"  数据加载年份: {self.load_years}")

        # 数据存储
        self.s1_data = {}
        self.all_s1_dates = []
        self.smap_data = {}
        self.all_smap_dates = []
        self.clamday_data = None
        self.dem_data = []

        # 加载数据
        self._setup_unified_grid()
        self._load_data_unified()
        self._canonicalize_temporal_axes()

        # ============ 🔥 添加 SWE 原始分布验证 ============
        print("\n" + "="*70)
        print("🔍 检查 ERA5-Land SWE 标签原始分布（采样前，已应用冰川阈值过滤）")
        print("="*70)

        all_swe_values = []
        for date_dt, (label_arr, label_nodata) in self.label_data.items():
            if label_nodata is not None:
                valid_mask = (label_arr != label_nodata) & np.isfinite(label_arr)
            else:
                valid_mask = np.isfinite(label_arr)
            values = label_arr[valid_mask].flatten()
            all_swe_values.extend(values)

        all_swe_values = np.array(all_swe_values)
        print(f"  总像元数: {len(all_swe_values):,}")
        print(f"  SWE 范围: [{all_swe_values.min():.2f}, {all_swe_values.max():.2f}] mm")

        dist_bins = [0, 5, 10, 20, 30, 50, 80, 120, 200, 500, 800, 1000, 1500, 2000]
        for lo, hi in zip(dist_bins[:-1], dist_bins[1:]):
            ratio = np.sum((all_swe_values > lo) & (all_swe_values <= hi)) / len(all_swe_values) * 100
            print(f"  {lo}-{hi}mm 占比: {ratio:.4f}%")

        print(f"  >={GLACIER_SWE_THRESHOLD_MM:.0f}mm 占比: "
              f"{np.sum(all_swe_values >= GLACIER_SWE_THRESHOLD_MM) / len(all_swe_values) * 100:.6f}%")

        # 0值/低值细分
        eps = 1e-6
        n_total = len(all_swe_values)
        n_zero = np.sum(all_swe_values <= eps)
        n_trace = np.sum((all_swe_values > eps) & (all_swe_values <= 1))
        n_low_1_5 = np.sum((all_swe_values > 1) & (all_swe_values <= 5))
        n_snow_5_20 = np.sum((all_swe_values > 5) & (all_swe_values <= 20))
        n_snow_20_80 = np.sum((all_swe_values > 20) & (all_swe_values <= 80))
        n_high_80 = np.sum(all_swe_values > 80)

        print(f"\n📊 0值/低值细分:")
        print(f"  SWE = 0:        {n_zero:,} ({n_zero/n_total*100:.2f}%)")
        print(f"  0 < SWE ≤ 1:    {n_trace:,} ({n_trace/n_total*100:.2f}%)")
        print(f"  1 < SWE ≤ 5:    {n_low_1_5:,} ({n_low_1_5/n_total*100:.2f}%)")
        print(f"  5 < SWE ≤ 20:   {n_snow_5_20:,} ({n_snow_5_20/n_total*100:.2f}%)")
        print(f"  20 < SWE ≤ 80:  {n_snow_20_80:,} ({n_snow_20_80/n_total*100:.2f}%)")
        print(f"  SWE > 80:       {n_high_80:,} ({n_high_80/n_total*100:.2f}%)")
        print("="*70 + "\n")

        # 计算模型输入维度。点特征固定为18维：
        # LS(6) + S1(5) + SMAP(2值+2掩膜) + 经纬度(2) + 时间(1)。
        # normalization_mode=skip 时不会执行归一化统计，因此必须在这里初始化 C_point。
        self.C_conv = len(CONV_VARS) + 1 + len(self.dem_data)
        self.C_point = 18

        print(f"\n📊 卷积特征维度统计:")
        print(f"  动态变量: {len(CONV_VARS)}")
        print(f"  静态变量 (Clamday): 1")
        print(f"  DEM波段: {len(self.dem_data)}")
        print(f"  → C_conv = {self.C_conv}")

        # 外部CSV测试站点必须在Stage 0和固定152000随机池之前完成隔离。
        if self.external_station_glob:
            self._load_external_station_exclusion()

        # 加载站点数据。Stage 0 用于站点引导；incremental 模式仅用于
        # 从固定随机池中排除站点格点，二者都只读取指定 station 文件。
        need_station_pixels = self.use_station_guide or (
            self.sampling_mode == "incremental"
            and self.incremental_exclude_station_pixels
        )
        if need_station_pixels:
            if self.use_station_guide and self.station_sampling_unit == "records":
                if self.station_record_manifest_path is not None:
                    self._load_station_record_samples_from_manifest()
                else:
                    self._load_station_record_samples()
                print(
                    f"  站点实际记录样本候选: {len(self.station_record_samples):,}"
                )
            else:
                self._load_all_station_pixels()
            print(f"  站点像元数: {len(self.station_pixels):,}")

        # 构建样本索引
        self._build_sample_index()

        # 正式Stage 0-4读取同一份统一归一化。清单/统计准备阶段可skip。
        if self.normalization_mode == "skip":
            self.normalization_method = "skip"
            self._apply_fixed_label_scale()
            print("   ⏭ normalization_mode=skip：仅构建清单/原始统计，不计算归一化")
        else:
            norm_loaded = self._maybe_load_normalization_config()
            if not norm_loaded:
                self._compute_minmax_sampling()
                self._apply_fixed_label_scale()
                self.normalization_method = "minmax"
                self._maybe_save_normalization_config()

        print(f"\n{'='*60}")
        print(f"✅ 数据集初始化完成!")
        print(f"  总样本数: {len(self.meta_index):,}")
        print(f"  C_conv: {self.C_conv}")
        print(f"  C_point: {self.C_point}")
        print(f"{'='*60}\n")

        # 保存缓存
        if cache_dir is not None and cache_path is not None:
            self._save_cache(cache_path, cache_key)

        # ============================================================
        # 大规模预训练默认不要全样本预计算到内存。
        # 原来的 self._precompute_and_cache() 会逐样本调用 __getitem__，
        # 对百万级样本会非常慢、非常占内存。
        #
        # 需要调试小样本时再手动开启：
        #   export PRECOMPUTE_ALL_SAMPLES=1
        # ============================================================
        if os.environ.get("PRECOMPUTE_ALL_SAMPLES", "0") == "1":
            self._precompute_and_cache()
        else:
            print("\n⚡ PRECOMPUTE_ALL_SAMPLES=0，跳过全样本内存预计算")
            print("   只保存特征/索引缓存，训练时由 DataLoader 按需读取样本")
        
        
    def setup_chinese_fonts(self):
        """设置中文字体，解决乱码问题"""
        import matplotlib
        import matplotlib.font_manager as fm
        import os
        import platform
        
        try:
            system = platform.system()
            
            if system == 'Linux':
                print("Setting up Chinese fonts on Linux...")
                
                wqy_paths = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                    '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
                    '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
                ]
                
                for font_path in wqy_paths:
                    if os.path.exists(font_path):
                        print(f"Found Chinese font: {font_path}")
                        fm.fontManager.addfont(font_path)
                        font_prop = fm.FontProperties(fname=font_path)
                        font_name = font_prop.get_name()
                        
                        matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
                        matplotlib.rcParams['axes.unicode_minus'] = False
                        
                        print(f"Successfully set Chinese font: {font_name}")
                        return True
                
                print("Chinese fonts not found, using English fonts")
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
                matplotlib.rcParams['axes.unicode_minus'] = False
                
            else:
                matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
                matplotlib.rcParams['axes.unicode_minus'] = False
                
        except Exception as e:
            print(f"Error setting up Chinese fonts: {e}")
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
            matplotlib.rcParams['axes.unicode_minus'] = False
        
        return False
            
    def _precompute_and_cache(self):
        """预计算所有样本并缓存在内存中"""

        # 🔥 如果已经缓存过，跳过
        if hasattr(self, '_cached_conv') and self._cached_conv is not None:
            print(f"\n✅ 内存缓存已存在，跳过预计算")
            return

        print(f"\n{'='*60}")
        print(f"🔥 预计算所有样本到内存")
        print(f"   总样本数: {len(self.meta_index):,}")
        print(f"{'='*60}")

        start = time.time()
        initial_mem = psutil.Process().memory_info().rss / 1024**3
        print(f"   初始内存: {initial_mem:.2f} GB")

        # 存储为列表
        self._cached_conv = []
        self._cached_point = []
        self._cached_target = []
        self._cached_mask = []
        self._cached_grid = []

        for idx in tqdm(range(len(self.meta_index)), desc="缓存样本"):
            conv, point, target, mask, grid, _ = self.__getitem__(idx)
            self._cached_conv.append(conv)
            self._cached_point.append(point)
            self._cached_target.append(target)
            self._cached_mask.append(mask)
            self._cached_grid.append(grid)

            if (idx + 1) % 10000 == 0:
                current_mem = psutil.Process().memory_info().rss / 1024**3
                print(f"   已缓存 {idx+1:,} 样本, 内存: {current_mem:.2f} GB (+{current_mem - initial_mem:.2f})")

        elapsed = time.time() - start
        final_mem = psutil.Process().memory_info().rss / 1024**3

        # ============ 🔥 添加真实内存计算 ============
        def calc_tensor_mem(tensors):
            total = 0
            for t in tensors[:100]:  # 采样前100个估算
                total += t.element_size() * t.nelement()
            if len(tensors) > 100:
                total = total / 100 * len(tensors)
            return total / 1024**3

        conv_mem = calc_tensor_mem(self._cached_conv)
        point_mem = calc_tensor_mem(self._cached_point)
        target_mem = calc_tensor_mem(self._cached_target)
        mask_mem = calc_tensor_mem(self._cached_mask)
        grid_mem = calc_tensor_mem(self._cached_grid)
        total_tensor_mem = conv_mem + point_mem + target_mem + mask_mem + grid_mem

        print(f"\n✅ 预计算完成!")
        print(f"   耗时: {elapsed:.1f} 秒")
        print(f"   psutil 内存: {final_mem:.2f} GB (+{final_mem - initial_mem:.2f})")
        print(f"   Tensor 实际内存: {total_tensor_mem:.2f} GB")
        print(f"   平均每样本: {total_tensor_mem / len(self.meta_index) * 1024:.2f} KB")
        print(f"{'='*60}\n")
            
    def _save_cache(self, cache_path: Path, cache_key: str):
        """保存缓存到文件（包含完整元数据）"""

        print(f"\n💾 保存缓存到: {cache_path}")

        # 🔥 修复：明确列出需要保存的属性（白名单模式）
        required_attrs = [
            # 核心数据
            'meta_index', 'all_dates', 'date_to_index',
            'conv_dyn_data', 'clamday_data', 'dem_data',
            'ls_data', 'ls_data_default',
            's1_data', 'all_s1_dates',
            'smap_data', 'all_smap_dates',
            'label_data',

            # 🔥 新增：站点引导采样相关属性
            'station_pixels',           # 站点像元集合
            'sampling_mode',            # random / station / hybrid
            'use_station_guide',        # 是否启用
            'station_guide_file',       # 精确站点文件
            'station_neighborhood',     # 邻域半径
            'station_samples_per_day',  # 每日上限
            'station_filter_zero_target',
            'station_sampling_unit',
            'station_record_dedup',
            'station_date_column',
            'station_record_samples',
            'station_record_stats',
            'station_csv_dir',          # 兼容旧目录模式
            'external_station_glob', 'external_station_exclusion_radius',
            'external_station_strict', 'external_station_report_path',
            'external_station_centers', 'external_excluded_cells',
            'external_station_stats',
            # 固定增量清单与fold
            'incremental_manifest_path', 'incremental_stage',
            'incremental_selection_mode', 'incremental_exclude_manifest_path',
            'incremental_pool_size', 'incremental_stage_sizes',
            'incremental_seed', 'sample_fold_ids',
            'seasonal_min_peak_swe_mm', 'seasonal_max_swe_mm',
            'seasonal_snow_free_threshold_mm',
            'seasonal_min_warm_snow_free_ratio',
            'seasonal_min_consecutive_snow_free_days',
            'seasonal_min_snow_year_coverage_ratio',
            'normalization_config_path', 'normalization_mode',
            'fixed_label_min_mm', 'fixed_label_max_mm',

            # 维度信息
            'C_conv', 'C_point', 'H', 'W',

            # 归一化参数
            'normalization_method',
            'conv_min', 'conv_max', 'point_min', 'point_max',
            'conv_clip_low', 'conv_clip_high', 'conv_mean', 'conv_std',
            'point_clip_low', 'point_clip_high', 'point_mean', 'point_std',
            'point_transform',
            'label_min', 'label_max',
            'lon_raw_min', 'lon_raw_max', 'lat_raw_min', 'lat_raw_max',
            'doy_raw_min', 'doy_raw_max',

            # 网格信息
            'common_bounds', 'transform', 'crs_proj',

            # 配置参数
            'region', 'year_target', 'load_years', 'patch_size', 'R',
            'min_valid_pixels', 'samples_per_day',
            'clamday_threshold',
            's1_interp_method', 's1_max_gap_days', 's1_nodata_value',
            'smap_interp_method', 'smap_max_gap_days', 'smap_nodata_value',

            # 微调相关
            'fine_tune_mode', 'load_fused_swe',
        ]

        to_save = {}
        excluded_count = 0
        saved_count = 0

        for key in required_attrs:
            if hasattr(self, key):
                value = getattr(self, key)
                try:
                    # 尝试序列化
                    pickle.dumps(value)
                    to_save[key] = value
                    saved_count += 1
                except (TypeError, pickle.PicklingError) as e:
                    print(f"   跳过不可序列化属性: {key} ({str(e)[:50]})")
                    excluded_count += 1
            else:
                print(f"   ⚠ 警告: 缺少必要属性 {key}")

        # 额外保存其他可能存在的属性（兼容性）
        optional_attrs = [
            'abundance_maps', 'endmembers',
            'station_set', 'station_meta',
            'swe_min', 'swe_max',
            'random_sample_stats',   # 随机采样统计
            'station_sample_stats',  # 站点采样统计
            'stage0_manifest_rows',
        ]

        for key in optional_attrs:
            if hasattr(self, key):
                value = getattr(self, key)
                try:
                    pickle.dumps(value)
                    to_save[key] = value
                    saved_count += 1
                except (TypeError, pickle.PicklingError):
                    excluded_count += 1

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size = cache_path.stat().st_size / 1024 / 1024
            print(f"   ✅ 缓存保存成功!")
            print(f"      保存属性数: {saved_count}")
            print(f"      跳过属性数: {excluded_count}")
            print(f"      文件大小: {file_size:.2f} MB")

            # ============ 🔥 保存完整的元数据文件（用于区分不同缓存） ============
            metadata_path = cache_path.with_suffix('.meta.json')

            # 获取日期范围
            date_range = {}
            if hasattr(self, 'all_dates') and self.all_dates:
                date_range = {
                    'start': self.all_dates[0].strftime('%Y-%m-%d') if hasattr(self.all_dates[0], 'strftime') else str(self.all_dates[0]),
                    'end': self.all_dates[-1].strftime('%Y-%m-%d') if hasattr(self.all_dates[-1], 'strftime') else str(self.all_dates[-1]),
                    'total_days': len(self.all_dates)
                }

            # 获取年份信息
            year_info = {}
            if hasattr(self, 'year_target'):
                year_info['year_target'] = self.year_target
            if hasattr(self, 'load_years'):
                year_info['load_years'] = self.load_years

            # 获取标签数据年份分布
            label_years = {}
            if hasattr(self, 'label_data') and self.label_data:
                for dt in self.label_data.keys():
                    year = dt.year if hasattr(dt, 'year') else int(str(dt)[:4]) if isinstance(dt, str) else None
                    if year:
                        label_years[year] = label_years.get(year, 0) + 1

            metadata = {
                # 基础信息
                'cache_key': cache_key,
                'timestamp': datetime.now().isoformat(),
                'file_size_mb': round(file_size, 2),

                # 数据集统计
                'total_samples': len(self.meta_index) if hasattr(self, 'meta_index') else 0,
                'C_conv': self.C_conv if hasattr(self, 'C_conv') else None,
                'C_point': self.C_point if hasattr(self, 'C_point') else None,
                'H': self.H if hasattr(self, 'H') else None,
                'W': self.W if hasattr(self, 'W') else None,

                # 🔥 关键：训练参数（用于区分不同缓存）
                'parameters': {
                    'region': getattr(self, 'region', None),
                    'year_target': getattr(self, 'year_target', None),
                    'load_years': getattr(self, 'load_years', None),
                    'patch_size': getattr(self, 'patch_size', None),
                    'min_valid_pixels': getattr(self, 'min_valid_pixels', None),
                    'samples_per_day': getattr(self, 'samples_per_day', None),
                    'clamday_threshold': getattr(self, 'clamday_threshold', None),
                    's1_nodata_value': getattr(self, 's1_nodata_value', None),
                    'smap_nodata_value': getattr(self, 'smap_nodata_value', None),
                    # 站点引导参数
                    'sampling_mode': getattr(self, 'sampling_mode', 'random'),
                    'use_station_guide': getattr(self, 'use_station_guide', False),
                    'station_guide_file': str(getattr(self, 'station_guide_file', '') or ''),
                    'station_neighborhood': getattr(self, 'station_neighborhood', 3),
                    'station_samples_per_day': getattr(self, 'station_samples_per_day', 2000),
                    'station_filter_zero_target': getattr(self, 'station_filter_zero_target', True),
                    'station_sampling_unit': getattr(
                        self, 'station_sampling_unit', 'positions_all_dates'
                    ),
                    'station_record_dedup': getattr(
                        self, 'station_record_dedup', 'grid_date'
                    ),
                    'station_date_column': getattr(self, 'station_date_column', None),
                    'filter_glacier_swe_artifacts': FILTER_GLACIER_SWE_ARTIFACTS,
                    'glacier_swe_threshold_mm': GLACIER_SWE_THRESHOLD_MM,
                },

                # 站点引导统计
                'station_guide_stats': {
                    'station_pixels_count': len(self.station_pixels) if hasattr(self, 'station_pixels') else 0,
                },

                # 日期范围
                'date_range': date_range,

                # 年份分布
                'year_info': year_info,
                'label_years_distribution': label_years,

                # 保存的属性列表
                'saved_attrs_count': len(to_save),
                'saved_attrs': list(to_save.keys())[:20],  # 只保存前20个，避免太长
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"      元数据已保存: {metadata_path}")

            # 打印关键参数摘要
            print(f"\n   📊 缓存参数摘要:")
            print(f"      采样数 (samples_per_day): {getattr(self, 'samples_per_day', 'N/A')}")
            print(f"      目标年份: {getattr(self, 'year_target', 'N/A')}")
            print(f"      总样本数: {len(self.meta_index) if hasattr(self, 'meta_index') else 0}")
            print(f"      日期范围: {date_range.get('start', 'N/A')} → {date_range.get('end', 'N/A')}")
            if label_years:
                print(f"      标签年份: {', '.join([f'{k}({v})' for k, v in label_years.items()])}")

            # 🔥 新增：打印站点引导信息
            if getattr(self, 'use_station_guide', False):
                station_pixels_count = len(self.station_pixels) if hasattr(self, 'station_pixels') else 0
                print(f"      站点引导: 启用 (邻域={getattr(self, 'station_neighborhood', 3)}x2+1, 站点像元={station_pixels_count})")

        except Exception as e:
            print(f"   ⚠ 缓存保存失败: {e}")
            import traceback
            traceback.print_exc()
            
    def _validate_cached_data(self):
        """验证缓存数据的完整性"""
        print("\n🔍 验证缓存数据完整性...")

        errors = []
        warnings = []

        # 1. 检查核心属性
        required_attrs = ['meta_index', 'all_dates', 'date_to_index', 'conv_dyn_data', 'label_data']
        for attr in required_attrs:
            if not hasattr(self, attr):
                errors.append(f"缺少必要属性: {attr}")
            elif getattr(self, attr) is None:
                errors.append(f"属性 {attr} 为 None")

        # 2. 检查 date_to_index 的有效性
        if hasattr(self, 'date_to_index') and self.date_to_index is not None:
            if len(self.date_to_index) == 0:
                errors.append("date_to_index 为空")

            if hasattr(self, 'all_dates') and self.all_dates:
                expected_keys = set(self.all_dates)
                actual_keys = set(self.date_to_index.keys())
                missing = expected_keys - actual_keys
                extra = actual_keys - expected_keys
                if missing:
                    errors.append(f"date_to_index 缺失 {len(missing)} 个日期")
                if extra:
                    warnings.append(f"date_to_index 多出 {len(extra)} 个日期")
        else:
            errors.append("date_to_index 不存在")

        # 3. 检查样本索引
        if hasattr(self, 'meta_index') and self.meta_index:
            if len(self.meta_index) == 0:
                errors.append("meta_index 为空")

        # 4. 检查标签数据
        if hasattr(self, 'label_data') and self.label_data:
            if len(self.label_data) == 0:
                errors.append("label_data 为空")

        if errors:
            print(f"   ❌ 发现 {len(errors)} 个错误:")
            for err in errors[:10]:
                print(f"      - {err}")
            raise RuntimeError("缓存数据验证失败，请删除缓存重新生成")

        print("   ✅ 缓存数据验证通过")
        return True
    
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
        """加载卷积特征数据（统一到公共区域）- 支持多年份，统一时间轴"""
        print(f"\n加载卷积特征数据...")

        # 🔥 定义各变量的无效值（统一用数值）
        INVALID_VALUES = {
            "chelsa_sfxwind": -9999.0,
            "lst": -9999.0,
            "rh": -9999.0,
            "pr": -9999.0,
        }

        # 临时存储每个变量的数据和日期
        var_data_map = {}
        var_dates_map = {}

        # 第一遍：加载所有变量
        for var in CONV_VARS:
            print(f"\n加载 {var} 数据...")
            var_data, var_dates, src_bounds, src_transform = self._load_single_variable(var)
            if var_data is None:
                raise ValueError(f"无法加载 {var} 数据")

            # 统一到公共网格
            var_data_unified = self._unify_to_common_grid(
                var_data, var, src_bounds, src_transform
            )

            var_data_map[var] = var_data_unified
            var_dates_map[var] = var_dates
            print(f"  {var} 原始时间轴: {len(var_dates)} 天 ({var_dates[0].strftime('%Y-%m-%d')} 到 {var_dates[-1].strftime('%Y-%m-%d')})")

        # 🔥 计算所有变量的日期交集
        print(f"\n计算所有卷积变量的日期交集...")
        common_dates = set(var_dates_map[CONV_VARS[0]])
        for var in CONV_VARS[1:]:
            common_dates = common_dates.intersection(set(var_dates_map[var]))

        common_dates = sorted(common_dates)
        print(f"  共同日期数: {len(common_dates)}")
        print(f"  原始 chelsa: {len(var_dates_map[CONV_VARS[0]])} 天")
        print(f"  原始 lst: {len(var_dates_map['lst'])} 天")
        print(f"  原始 rh: {len(var_dates_map['rh'])} 天")
        print(f"  原始 pr: {len(var_dates_map['pr'])} 天")

        if len(common_dates) == 0:
            raise ValueError("卷积特征没有共同日期！请检查数据时间范围")

        # 🔥 第二遍：根据共同日期重新切片所有变量
        self.conv_dyn_data = {}
        for var in CONV_VARS:
            orig_dates = var_dates_map[var]
            orig_data = var_data_map[var]

            # 找到共同日期在原数组中的索引
            keep_indices = []
            for d in common_dates:
                try:
                    idx = orig_dates.index(d)
                    keep_indices.append(idx)
                except ValueError:
                    # 理论上不会发生，因为 common_dates 是交集
                    print(f"  警告: {var} 中找不到日期 {d}")
                    continue

            # 切片数据
            self.conv_dyn_data[var] = orig_data[keep_indices]
            print(f"  {var} 切片后: {self.conv_dyn_data[var].shape}")

        # 设置统一的时间轴
        self.all_dates = common_dates
        self.date_to_index = {d: i for i, d in enumerate(common_dates)}

        print(f"\n✅ 统一时间轴范围: {self.all_dates[0]} 到 {self.all_dates[-1]}")
        print(f"  总天数: {len(self.all_dates)}")

        # 🔥 统计每个变量的有效数据比例（统一时间轴后）
        print(f"\n【统一时间轴后的有效数据统计】")
        for var in CONV_VARS:
            arr = self.conv_dyn_data[var]
            invalid_val = INVALID_VALUES.get(var)

            if invalid_val is not None:
                valid_mask = (arr != invalid_val) & np.isfinite(arr)
            else:
                valid_mask = np.isfinite(arr)

            valid_data = arr[valid_mask]
            if len(valid_data) > 0:
                print(f"  {var}: 有效数据比例 = {len(valid_data)/arr.size*100:.1f}%, "
                      f"范围 = [{valid_data.min():.4f}, {valid_data.max():.4f}]")
            else:
                print(f"  {var}: ⚠ 没有有效数据")

        return

    def _load_sentinel1_data(self):
        """加载哨兵1数据 - 支持新格式（VV, VH, VV_cov, VH_cov, angle）"""
        print(f"  加载哨兵1数据...")

        s1_files = point_var_path("S1_VV", self.load_years)
        if not s1_files:
            print(f"  警告: 未找到哨兵1数据")
            return

        print(f"  找到 {len(s1_files)} 个哨兵1文件")

        self.s1_data = {}
        self.all_s1_dates = []

        for s1_file in s1_files:
            try:
                with rasterio.open(s1_file) as ds:
                    filename = s1_file.stem
                    try:
                        file_month_dt = self._parse_date_from_filename(filename)
                        year, month = file_month_dt.year, file_month_dt.month
                    except Exception:
                        print(f"    无法解析文件名: {filename}")
                        continue

                    n_bands = ds.count
                    band_descriptions = ds.descriptions if ds.descriptions else []
                    src_transform = ds.transform

                    print(f"    处理文件: {s1_file.name}, 波段数: {n_bands}")

                    file_data = {}

                    for band_idx in range(1, n_bands + 1):
                        band_desc = band_descriptions[band_idx - 1] if band_idx - 1 < len(band_descriptions) else f"Band_{band_idx}"

                        desc_clean = band_desc.replace('Band ', '').strip()
                        if ':' in desc_clean:
                            desc_clean = desc_clean.split(':', 1)[1].strip()

                        date_match = re.search(r'(\d{4})[_]?(\d{2})[_]?(\d{2})', desc_clean)
                        if not date_match:
                            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', desc_clean)
                        if not date_match:
                            print(f"      跳过无法解析日期的波段: {band_desc}")
                            continue

                        y, m, d = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                        try:
                            band_date = datetime(y, m, d)
                        except:
                            continue

                        desc_upper = desc_clean.upper()
                        if '_VV' in desc_upper and '_COV' not in desc_upper:
                            band_type = 'VV'
                        elif '_VH' in desc_upper and '_COV' not in desc_upper:
                            band_type = 'VH'
                        elif '_VV_COV' in desc_upper:
                            band_type = 'VV_cov'
                        elif '_VH_COV' in desc_upper:
                            band_type = 'VH_cov'
                        elif 'ANGLE' in desc_upper:
                            band_type = 'angle'
                        else:
                            print(f"      跳过未知波段类型: {band_desc}")
                            continue

                        band_data = ds.read(band_idx).astype(np.float32)

                        aligned_data = self._align_single_layer(
                            band_data, src_transform, self.transform, self.H, self.W
                        )

                        if band_date not in file_data:
                            file_data[band_date] = {
                                'VV': None, 'VH': None,
                                'VV_cov': None, 'VH_cov': None,
                                'angle': None
                            }

                        file_data[band_date][band_type] = aligned_data

                        valid_data = aligned_data[np.isfinite(aligned_data)]
                        if len(valid_data) > 0:
                            print(f"      波段: {band_desc}")
                            print(f"        日期: {band_date.strftime('%Y-%m-%d')}, 类型: {band_type}")
                            print(f"        有效数据范围: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
                            print(f"        有效数据比例: {len(valid_data)/aligned_data.size*100:.1f}%")

                    for date_dt, data_dict in file_data.items():
                        if date_dt not in self.s1_data:
                            self.s1_data[date_dt] = {
                                'VV': None, 'VH': None,
                                'VV_cov': None, 'VH_cov': None,
                                'angle': None
                            }

                        for key in ['VV', 'VH', 'VV_cov', 'VH_cov', 'angle']:
                            if data_dict[key] is not None:
                                self.s1_data[date_dt][key] = data_dict[key]

                        if date_dt not in self.all_s1_dates:
                            self.all_s1_dates.append(date_dt)

            except Exception as e:
                print(f"    处理文件 {s1_file.name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.all_s1_dates.sort()

        if self.all_s1_dates:
            print(f"\n    【总结】哨兵1数据加载完成:")
            print(f"      有效日期数: {len(self.all_s1_dates)}")
            print(f"      日期范围: {self.all_s1_dates[0].strftime('%Y-%m-%d')} 到 {self.all_s1_dates[-1].strftime('%Y-%m-%d')}")

            complete_count = 0
            for date_dt in self.all_s1_dates:
                data = self.s1_data[date_dt]
                if all(data[k] is not None for k in ['VV', 'VH', 'VV_cov', 'VH_cov', 'angle']):
                    complete_count += 1
            print(f"      完整数据（5个波段）日期数: {complete_count}/{len(self.all_s1_dates)}")
        else:
            print(f"    ⚠ 警告: 没有有效的哨兵1数据")

    @staticmethod
    def _dates_from_multiband_dataset(file_path: Path, ds, target_years):
        """从波段描述或文件名推断多波段逐日时间轴。"""
        descriptions = list(ds.descriptions or [])
        parsed = []
        if descriptions and any(descriptions):
            for desc in descriptions:
                if not desc:
                    parsed = []
                    break
                match = re.search(r'(\d{4})[-_/]?(\d{2})[-_/]?(\d{2})', str(desc))
                if not match:
                    parsed = []
                    break
                try:
                    parsed.append(datetime(*map(int, match.groups())))
                except ValueError:
                    parsed = []
                    break
            if len(parsed) == ds.count and all(d.year in target_years for d in parsed):
                return parsed

        name = file_path.stem
        ym = re.search(r'(?<!\d)(\d{4})[_-]?(\d{2})(?!\d)', name)
        if ym:
            year, month = map(int, ym.groups())
            if year in target_years and 1 <= month <= 12:
                n_days = calendar.monthrange(year, month)[1]
                if ds.count <= n_days:
                    return [datetime(year, month, day) for day in range(1, ds.count + 1)]

        ymatch = re.search(r'(?<!\d)(\d{4})(?!\d)', name)
        if ymatch:
            year = int(ymatch.group(1))
            expected = 366 if calendar.isleap(year) else 365
            if year in target_years and ds.count == expected:
                start = datetime(year, 1, 1)
                return [start + timedelta(days=i) for i in range(ds.count)]
        return None

    def _fill_missing_wind_dates(self, daily_layers):
        """
        将风速逐日序列补齐到 self.load_years 覆盖的完整日历。

        规则：
        1. 内部缺失日：按前后最近有效日期进行线性时间插值；
        2. 年初/年末缺失：使用最近有效日填充；
        3. 像元级无效值：两侧均有效时线性插值，仅一侧有效时使用该侧，
           两侧均无效时保持 -9999；
        4. 只补时间上缺失的整日文件，不修改已有日期的数据。
        """
        if not daily_layers:
            return daily_layers

        available_dates = sorted(self._canonical_day(d) for d in daily_layers)
        normalized_layers = {
            self._canonical_day(d): np.asarray(arr, dtype=np.float32)
            for d, arr in daily_layers.items()
        }

        full_dates = []
        for year in sorted(set(int(y) for y in self.load_years)):
            start = datetime(year, 1, 1)
            n_days = 366 if calendar.isleap(year) else 365
            full_dates.extend(start + timedelta(days=i) for i in range(n_days))

        missing_dates = [d for d in full_dates if d not in normalized_layers]
        if not missing_dates:
            print("    风速时间轴完整，无需插值")
            return normalized_layers

        print(f"    ⚠ 风速缺失整日: {len(missing_dates)} 天，开始时间插值")
        print(f"      缺失范围: {missing_dates[0].strftime('%Y-%m-%d')} .. "
              f"{missing_dates[-1].strftime('%Y-%m-%d')}")

        nodata = -9999.0
        internal_count = 0
        edge_count = 0

        # 插值基准始终使用原始有效日期，避免连续缺失时递归使用插值结果。
        original_dates = available_dates
        for target_date in missing_dates:
            pos = bisect_left(original_dates, target_date)
            prev_date = original_dates[pos - 1] if pos > 0 else None
            next_date = original_dates[pos] if pos < len(original_dates) else None

            if prev_date is None and next_date is None:
                raise RuntimeError("风速时间插值失败：没有任何可用日期")

            if prev_date is None:
                normalized_layers[target_date] = normalized_layers[next_date].copy()
                edge_count += 1
                continue

            if next_date is None:
                normalized_layers[target_date] = normalized_layers[prev_date].copy()
                edge_count += 1
                continue

            prev_arr = normalized_layers[prev_date]
            next_arr = normalized_layers[next_date]
            total_days = (next_date - prev_date).days
            if total_days <= 0:
                raise RuntimeError(
                    f"风速时间插值日期顺序异常: {prev_date} -> {next_date}"
                )

            alpha = (target_date - prev_date).days / float(total_days)
            prev_valid = np.isfinite(prev_arr) & (prev_arr != nodata)
            next_valid = np.isfinite(next_arr) & (next_arr != nodata)

            out = np.full(prev_arr.shape, nodata, dtype=np.float32)
            both = prev_valid & next_valid
            only_prev = prev_valid & ~next_valid
            only_next = ~prev_valid & next_valid

            out[both] = (
                prev_arr[both] * (1.0 - alpha)
                + next_arr[both] * alpha
            ).astype(np.float32, copy=False)
            out[only_prev] = prev_arr[only_prev]
            out[only_next] = next_arr[only_next]
            normalized_layers[target_date] = out
            internal_count += 1

        final_dates = sorted(normalized_layers)
        expected = len(full_dates)
        if len(final_dates) != expected:
            missing_after = [d for d in full_dates if d not in normalized_layers]
            raise RuntimeError(
                f"风速时间插值后仍缺 {len(missing_after)} 天: "
                f"{missing_after[:5]}"
            )

        print(f"    ✅ 风速时间轴补齐: {len(available_dates)} -> {len(final_dates)} 天")
        print(f"      线性插值: {internal_count} 天")
        print(f"      边界最近值填充: {edge_count} 天")
        return normalized_layers

    def _load_single_variable(self, var: str):
        """加载单个变量的数据 - 支持多年份"""
        print(f"  加载 {var} 数据...")

        # 🔥 统一使用 self.load_years
        target_years = self.load_years

        # 如果需要更多历史数据，可以扩展
        all_years = set(target_years)

        all_files = []
        for year in sorted(all_years):
            var_dir = conv_var_path(var, year)
            if not var_dir.exists():
                print(f"    {year}: 目录不存在，跳过")
                continue

            # 根据变量类型设置不同的文件匹配模式
            if var == "chelsa_sfxwind":
                files = list(var_dir.glob(f"*{year}*.tif"))
            elif var == "lst":
                files = _glob_unique(var_dir, [f"ERA5LAND_LST_{year}??_DAILYMEAN*.tif", f"*LST*{year}*.tif", f"ERA5_ST_{year}*.tif"])
            elif var == "rh":
                files = _glob_unique(var_dir, [f"ERA5LAND_RH_{year}??_DAILYMEAN*.tif", f"*RH*{year}*.tif", f"ERA5_RH_DailyMean_{year}_*.tif"])
            elif var == "pr":
                files = list(var_dir.glob(f"*{year}*.tif"))
            else:
                files = list(var_dir.glob("*.tif"))

            if not files:
                print(f"    {year}: 没有找到匹配的文件，跳过")
                continue

            print(f"    {year}: 找到 {len(files)} 个文件")
            all_files.extend(files)

        if not all_files:
            print(f"  未找到 {var} 文件")
            return None, [], None, None

        # 获取第一个文件的bounds和transform
        with rasterio.open(all_files[0]) as ds:
            src_bounds = ds.bounds
            src_transform = ds.transform

        # 处理月份文件（lst 和 rh）
        if var in ["lst", "rh"]:
            monthly_data = {}
            for f in all_files:
                try:
                    name = f.stem

                    if var == "rh":
                        match = re.search(r'ERA5_RH_DailyMean_(\d{4})_(\d{2})', name)
                        if not match:
                            match = re.search(r'(\d{4})(\d{2})', name)
                        if not match:
                            print(f"    无法解析文件名: {name}")
                            continue
                        year = int(match.group(1))
                        month = int(match.group(2))
                    else:
                        match = re.search(r'(\d{4})(\d{2})', name)
                        if not match:
                            continue
                        year = int(match.group(1))
                        month = int(match.group(2))

                    with rasterio.open(f) as ds:
                        data = ds.read()
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

        else:  # chelsa_sfxwind, pr等：支持单日文件、月/年多波段逐日立方体
            daily_layers = {}
            for f in all_files:
                try:
                    with rasterio.open(f) as ds:
                        if ds.count > 1:
                            band_dates = self._dates_from_multiband_dataset(f, ds, set(target_years))
                            if band_dates is None:
                                raise ValueError(
                                    f"多波段文件无法推断逐日日期: {f.name}, bands={ds.count}"
                                )
                            for band_idx, dt in enumerate(band_dates, start=1):
                                data = ds.read(band_idx).astype(np.float32)
                                if var == 'pr':
                                    data = np.nan_to_num(data, nan=-9999.0)
                                daily_layers[self._canonical_day(dt)] = data
                        else:
                            dt = self._canonical_day(self._parse_date_from_filename(f.stem))
                            data = ds.read(1).astype(np.float32)
                            if var == 'pr':
                                data = np.nan_to_num(data, nan=-9999.0)
                            daily_layers[dt] = data
                except Exception as e:
                    print(f"    读取/解析 {f.name} 失败: {e}")
                    continue

            if not daily_layers:
                print(f"  没有可用的{var}逐日数据")
                return None, [], None, None

            if var == "chelsa_sfxwind":
                daily_layers = self._fill_missing_wind_dates(daily_layers)

            var_dates = sorted(daily_layers)
            var_arr = np.stack([daily_layers[d] for d in var_dates], axis=0).astype(np.float32)
            if var == 'pr':
                print(f"\n  【验证】pr 数据返回前:")
                print(f"    数据形状: {var_arr.shape}")
                print(f"    NaN 数量: {np.isnan(var_arr).sum()}")
                print(f"    -9999 数量: {(var_arr == -9999).sum()}")
                print(f"    有限值数量: {np.isfinite(var_arr).sum()}")
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
        """对齐单个图层 - 使用 rasterio.reproject 高效实现"""
        import rasterio
        from rasterio.warp import reproject, Resampling

        # 记录输入形状（2D 还是带波段维度）
        is_2d = False
        if len(src_data.shape) == 2:
            is_2d = True
            src_data = src_data[np.newaxis, :, :]  # 添加波段维度

        n_bands, src_h, src_w = src_data.shape

        # 预分配目标数组
        aligned = np.full((n_bands, target_h, target_w), np.nan, dtype=src_data.dtype)

        # 统一处理最终产品 NoData=-9999
        src_data = src_data.astype(np.float32)
        src_data[~np.isfinite(src_data)] = np.nan
        src_data[src_data == -9999.0] = np.nan

        src_nodata = np.nan

        # 逐波段重投影
        for band in range(n_bands):
            reproject(
                source=src_data[band],
                destination=aligned[band],
                src_transform=src_transform,
                src_crs=self.crs_proj,
                dst_transform=target_transform,
                dst_crs=self.crs_proj,
                resampling=Resampling.nearest,
                src_nodata=src_nodata,
                dst_nodata=np.nan,
                num_threads=4,  # 多线程加速
            )

        # 如果输入是 2D，去掉波段维度
        if is_2d:
            aligned = aligned[0]

        return aligned

    def _idw_interpolate(self, data, invalid_mask, power=2, radius=10):
        """
        IDW (反距离加权) 插值填充二维数组中的无效值

        Args:
            data: 2D numpy array
            invalid_mask: bool mask，True 的位置需要插值
            power: 距离的幂次（通常为2）
            radius: 搜索半径（像素），None 表示全局搜索
        Returns:
            插值后的数组（与 data 形状相同）
        """
        if not np.any(invalid_mask):
            return data

        valid_mask = ~invalid_mask
        valid_coords = np.argwhere(valid_mask)
        valid_values = data[valid_mask]

        # 如果没有有效点，返回全零
        if len(valid_values) == 0:
            return np.zeros_like(data)

        # 需要插值的位置
        invalid_coords = np.argwhere(invalid_mask)
        interpolated = np.zeros(len(invalid_coords))

        for i, (x, y) in enumerate(invalid_coords):
            # 搜索邻域内的有效点
            if radius is not None:
                x_min = max(0, x - radius)
                x_max = min(data.shape[0], x + radius + 1)
                y_min = max(0, y - radius)
                y_max = min(data.shape[1], y + radius + 1)
                local_mask = (valid_coords[:, 0] >= x_min) & (valid_coords[:, 0] < x_max) & \
                             (valid_coords[:, 1] >= y_min) & (valid_coords[:, 1] < y_max)
                local_coords = valid_coords[local_mask]
                local_values = valid_values[local_mask]
            else:
                local_coords = valid_coords
                local_values = valid_values

            if len(local_coords) == 0:
                # 周围没有有效点，使用全局均值
                interpolated[i] = np.mean(valid_values)
                continue

            # 计算距离
            distances = np.sqrt((local_coords[:, 0] - x) ** 2 + (local_coords[:, 1] - y) ** 2)
            distances[distances == 0] = 1e-8
            weights = 1.0 / (distances ** power)
            weights /= weights.sum()
            interpolated[i] = np.sum(weights * local_values)

        result = data.copy()
        result[invalid_mask] = interpolated
        return result

    
    def _load_static_conv_features_unified(self):
        """加载静态卷积特征（统一到公共区域）- 支持多波段DEM，境外区域保持NaN"""
        print(f"\n加载静态卷积特征...")

        # 初始化属性
        self.clamday_data = None
        self.dem_data = []  # 存储所有 DEM 波段

        # 1. 加载 clamday - 🔥 使用 self.load_years
        clamday_files = conv_static_path("clamday", self.load_years, self.clamday_threshold)
        if clamday_files:
            # 如果有多个文件，选择第一个
            clamday_file = clamday_files[0]
            print(f"  使用Clamday文件: {clamday_file.name}")

            with rasterio.open(clamday_file) as ds:
                clamday_data_raw = ds.read(1).astype(np.float32)
                src_bounds = ds.bounds
                src_transform = ds.transform

            self.clamday_data = self._align_single_layer(
                clamday_data_raw, src_transform, self.transform, self.H, self.W
            )

            # 🔥 统计有效数据 - Clamday 无效值是 -9999
            CLAMDAY_INVALID = -9999.0
            valid_mask = (self.clamday_data != CLAMDAY_INVALID) & np.isfinite(self.clamday_data)
            valid_data = self.clamday_data[valid_mask]

            if len(valid_data) > 0:
                print(f"  Clamday形状: {self.clamday_data.shape}")
                print(f"  Clamday有效数据范围: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
                print(f"  Clamday有效数据比例: {len(valid_data)/self.clamday_data.size*100:.1f}%")
            else:
                print(f"  Clamday形状: {self.clamday_data.shape}")
                print(f"  ⚠ Clamday没有有效数据")
        else:
            self.clamday_data = np.zeros((self.H, self.W), dtype=np.float32)
            print(f"  警告: 未找到Clamday文件")

        # 2. 加载 DEM（读取所有波段）- DEM 不受年份影响，保持原样
        dem_files = conv_static_path("dem", self.year_target)

        # 🔥🔥🔥 过滤掉新疆区域的DEM文件 🔥🔥🔥
        if dem_files:
            original_count = len(dem_files)
            dem_files = [f for f in dem_files if 'Xinjiang' not in f.name and 'XINJIANG' not in f.name.upper()]
            if original_count != len(dem_files):
                print(f"  已排除 {original_count - len(dem_files)} 个新疆区域DEM文件")

        if dem_files:
            print(f"  找到 {len(dem_files)} 个DEM文件")

            # 定义无效值常量（根据实际文件调整）
            DEM_NODATA = -9999.0
            # DEM 合理高程范围（中国区域：-200 ~ 9000 米）
            DEM_MIN_VALID = -200.0
            DEM_MAX_VALID = 9000.0

            for dem_file in dem_files:
                with rasterio.open(dem_file) as ds:
                    src_bounds = ds.bounds
                    src_transform = ds.transform

                    # 获取文件的 nodata 值（如果未定义则使用默认值）
                    file_nodata = ds.nodata if ds.nodata is not None else DEM_NODATA

                    # 读取所有波段
                    dem_all_bands = ds.read().astype(np.float32)  # (n_bands, H, W)
                    n_bands = dem_all_bands.shape[0]

                    print(f"    文件: {dem_file.name}, 波段数: {n_bands}")

                    for band_idx in range(n_bands):
                        band_data = dem_all_bands[band_idx]

                        # ---------- 步骤1：标记并替换无效值 ----------
                        # 标记文件 nodata 和超出合理范围的值
                        invalid_mask = (band_data == file_nodata) | \
                                       (band_data < DEM_MIN_VALID) | \
                                       (band_data > DEM_MAX_VALID)
                        if np.any(invalid_mask):
                            print(f"      DEM波段{band_idx+1}: 发现 {np.sum(invalid_mask)} 个无效值（nodata或超出范围）")
                            band_data[invalid_mask] = np.nan

                        # ---------- 步骤2：地理对齐 ----------
                        aligned_band = self._align_single_layer(
                            band_data, src_transform, self.transform, self.H, self.W
                        )

                        # ---------- 🔥 步骤3：不再插值！境外区域保持 NaN ----------
                        nan_mask = np.isnan(aligned_band)
                        if np.any(nan_mask):
                            print(f"      DEM波段{band_idx+1}: 对齐后存在 {np.sum(nan_mask)} 个 NaN（境外区域），保持不变")

                        # 添加到 DEM 列表
                        self.dem_data.append(aligned_band)

                        # 获取波段描述
                        if ds.descriptions and band_idx < len(ds.descriptions):
                            band_desc = ds.descriptions[band_idx]
                        else:
                            band_desc = f"band_{band_idx+1}"

                        # 统计有效值（有限值）
                        valid_data = aligned_band[np.isfinite(aligned_band)]
                        if len(valid_data) > 0:
                            print(f"      DEM波段{len(self.dem_data)}: {band_desc} 形状: {aligned_band.shape}, "
                                  f"有效值范围: [{valid_data.min():.2f}, {valid_data.max():.2f}], "
                                  f"有效值比例: {len(valid_data)/aligned_band.size*100:.1f}%")
                        else:
                            print(f"      DEM波段{len(self.dem_data)}: {band_desc} 形状: {aligned_band.shape}, ⚠ 没有有效数据")
        else:
            # 如果没有DEM文件，创建一个默认波段
            self.dem_data = [np.zeros((self.H, self.W), dtype=np.float32)]
            print(f"  警告: 未找到DEM文件，使用默认零值波段")

        print(f"  DEM总波段数: {len(self.dem_data)}")

    def _load_smap_data(self):
        """加载SMAP亮温数据 - 支持新格式（H/V + mask）"""
        print(f"  加载SMAP数据...")

        smap_root = FEATURE_ROOT / "smap" / "cn"
        if not smap_root.exists():
            print(f"    ⚠ SMAP目录不存在: {smap_root}")
            return

        # 🔥 使用 self.load_years 获取要加载的年份列表
        target_years = self.load_years
        print(f"    加载年份: {target_years}")

        # 收集所有匹配的SMAP文件（兼容 SMAP_201507_AMTB_... 与旧 cube 命名）
        all_smap_files = []
        for year in target_years:
            files = point_var_path("SMAP_TBV", year)
            all_smap_files.extend(files)
            if files:
                print(f"    {year}年: 找到 {len(files)} 个文件")

        if not all_smap_files:
            print(f"    ⚠ 未找到SMAP文件")
            return

        all_smap_files = sorted(list(set(all_smap_files)))
        print(f"    共找到 {len(all_smap_files)} 个SMAP文件")

        # 初始化数据存储
        self.smap_data = {}
        self.all_smap_dates = []

        for smap_file in all_smap_files:
            try:
                filename = smap_file.stem
                match = re.search(r'SMAP_(\d{4})(\d{2})_', filename, re.IGNORECASE) or re.search(r'SMAP_(\d{4})_(\d{2})', filename, re.IGNORECASE)
                if not match:
                    print(f"    无法解析文件名: {filename}")
                    continue

                year = int(match.group(1))
                month = int(match.group(2))

                with rasterio.open(smap_file) as ds:
                    n_bands = ds.count
                    band_descriptions = ds.descriptions if ds.descriptions else []
                    src_transform = ds.transform

                    print(f"    处理文件: {smap_file.name}, 波段数: {n_bands}")

                    file_data = {}

                    for band_idx in range(1, n_bands + 1):
                        band_desc = band_descriptions[band_idx - 1] if band_idx - 1 < len(band_descriptions) else f"Band_{band_idx}"

                        desc_clean = band_desc.replace('Band ', '').strip()
                        if ':' in desc_clean:
                            desc_clean = desc_clean.split(':', 1)[1].strip()

                        is_mask = '_mask' in desc_clean.lower()

                        date_match = re.search(r'(\d{4})[_]?(\d{2})[_]?(\d{2})_?([HV])', desc_clean)

                        if date_match:
                            y, m, d = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                            pol = date_match.group(4)
                        else:
                            date_match = re.search(r'(\d{4})[_]?(\d{2})[_]?(\d{2})', desc_clean)
                            if not date_match:
                                print(f"      跳过无法解析的波段: {band_desc}")
                                continue
                            y, m, d = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))

                            if 'H' in desc_clean.upper():
                                pol = 'H'
                            elif 'V' in desc_clean.upper():
                                pol = 'V'
                            else:
                                pol = None
                                print(f"      警告: 无法推断极化: {band_desc}")

                        try:
                            band_date = datetime(y, m, d)
                        except:
                            continue

                        band_data = ds.read(band_idx).astype(np.float32)

                        aligned_data = self._align_single_layer(
                            band_data, src_transform, self.transform, self.H, self.W
                        )

                        if band_date not in file_data:
                            file_data[band_date] = {
                                'TBH': None, 'TBV': None,
                                'mask_H': None, 'mask_V': None
                            }

                        if pol is not None:
                            if is_mask:
                                file_data[band_date][f'mask_{pol}'] = aligned_data
                            else:
                                file_data[band_date][f'TB{pol}'] = aligned_data
                        else:
                            if is_mask:
                                if 'H' in desc_clean.upper():
                                    file_data[band_date]['mask_H'] = aligned_data
                                elif 'V' in desc_clean.upper():
                                    file_data[band_date]['mask_V'] = aligned_data
                            else:
                                if 'H' in desc_clean.upper():
                                    file_data[band_date]['TBH'] = aligned_data
                                elif 'V' in desc_clean.upper():
                                    file_data[band_date]['TBV'] = aligned_data

                        # 打印有效数据统计（简化版）
                        valid_data = aligned_data[np.isfinite(aligned_data)]
                        if len(valid_data) > 0:
                            print(f"      波段: {band_desc}")
                            print(f"        日期: {band_date.strftime('%Y-%m-%d')}, 极化: {pol if pol else 'N/A'}, mask: {is_mask}")
                            print(f"        有效数据范围: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
                            print(f"        有效数据比例: {len(valid_data)/aligned_data.size*100:.1f}%")

                    for date_dt, data_dict in file_data.items():
                        if date_dt not in self.smap_data:
                            self.smap_data[date_dt] = {
                                'TBH': None, 'TBV': None,
                                'mask_H': None, 'mask_V': None
                            }

                        if data_dict['TBH'] is not None:
                            self.smap_data[date_dt]['TBH'] = data_dict['TBH']
                        if data_dict['TBV'] is not None:
                            self.smap_data[date_dt]['TBV'] = data_dict['TBV']
                        if data_dict['mask_H'] is not None:
                            self.smap_data[date_dt]['mask_H'] = data_dict['mask_H']
                        if data_dict['mask_V'] is not None:
                            self.smap_data[date_dt]['mask_V'] = data_dict['mask_V']

                        if date_dt not in self.all_smap_dates:
                            self.all_smap_dates.append(date_dt)

            except Exception as e:
                print(f"    处理文件 {smap_file.name} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.all_smap_dates.sort()

        if self.all_smap_dates:
            print(f"\n    【总结】SMAP数据加载完成:")
            print(f"      有效日期数: {len(self.all_smap_dates)}")
            print(f"      日期范围: {self.all_smap_dates[0].strftime('%Y-%m-%d')} 到 {self.all_smap_dates[-1].strftime('%Y-%m-%d')}")

            complete_count = 0
            for date_dt in self.all_smap_dates:
                data = self.smap_data[date_dt]
                if data['TBH'] is not None and data['TBV'] is not None:
                    complete_count += 1
            print(f"      完整数据（H+V）日期数: {complete_count}/{len(self.all_smap_dates)}")
        else:
            print(f"    ⚠ 警告: 没有有效的SMAP数据")


    def _get_sentinel1_value(self, date_dt: datetime, r: int, c: int) -> Tuple[float, float, float, float, float]:
        """
        获取指定日期和位置的哨兵1值
        返回: (vv, vh, vv_cov, vh_cov, angle)
        注意：数据已是逐日格式，不需要时间插值
        """
        vv_value = self.s1_nodata_value
        vh_value = self.s1_nodata_value
        vv_cov_value = 0.0
        vh_cov_value = 0.0
        angle_value = self.s1_nodata_value

        # 如果没有该日期的数据，返回默认值
        if date_dt not in self.s1_data:
            return vv_value, vh_value, vv_cov_value, vh_cov_value, angle_value

        data = self.s1_data[date_dt]

        # VV
        if 'VV' in data and data['VV'] is not None:
            vv_value = float(data['VV'][r, c])

        # VH
        if 'VH' in data and data['VH'] is not None:
            vh_value = float(data['VH'][r, c])

        # VV_cov
        if 'VV_cov' in data and data['VV_cov'] is not None:
            vv_cov_value = float(data['VV_cov'][r, c])

        # VH_cov
        if 'VH_cov' in data and data['VH_cov'] is not None:
            vh_cov_value = float(data['VH_cov'][r, c])

        # angle
        if 'angle' in data and data['angle'] is not None:
            angle_value = float(data['angle'][r, c])

        return vv_value, vh_value, vv_cov_value, vh_cov_value, angle_value

    def _load_point_data_unified(self):
        """加载点特征数据（统一到公共区域）"""
        print(f"\n加载点特征数据...")

        # 🔥 LS 改为按年份分别加载，兼容新版 China_Landsat_L8C2L2_YYYY_AnnualMedian_ERA5SWE010deg.tif
        self.ls_data = {}  # year -> array

        for year in self.load_years:
            ls_file = point_var_path("ls", year)
            if ls_file is not None and Path(ls_file).exists():
                print(f"  处理LS文件: {Path(ls_file).name} (年份: {year})")

                with rasterio.open(ls_file) as ds:
                    ls_data_raw = ds.read().astype(np.float32)  # (C_ls, H, W)
                    src_transform = ds.transform

                aligned_bands = []
                for i in range(ls_data_raw.shape[0]):
                    band_aligned = self._align_single_layer(
                        ls_data_raw[i], src_transform, self.transform, self.H, self.W
                    )
                    aligned_bands.append(band_aligned)

                self.ls_data[year] = np.stack(aligned_bands, axis=0)
                print(f"    LS{year}数据形状: {self.ls_data[year].shape}")
            else:
                print(f"  警告: 未找到 {year} 年LS文件，使用6个零值波段")
                self.ls_data[year] = np.zeros((6, self.H, self.W), dtype=np.float32)

        # 默认使用第一年的LS数据（兼容旧代码）
        self.ls_data_default = self.ls_data.get(self.load_years[0], np.zeros((6, self.H, self.W)))

        # 加载哨兵1数据
        self._load_sentinel1_data()

        # 加载SMAP数据
        self._load_smap_data()
    
    def _filter_glacier_swe_artifacts(self, label_arr, label_nodata=None, date_dt=None):
        """
        剔除 ERA5-Land / HTESSEL 冰川占位型 SWE 极端值。

        不是只删 ==10000，因为实际 tif 里可能是 9900+、
        也可能因为插值/重采样变成几千 mm。
        """
        if not FILTER_GLACIER_SWE_ARTIFACTS:
            return label_arr

        arr = label_arr.astype(np.float32, copy=True)

        valid_mask = np.isfinite(arr)
        if label_nodata is not None:
            valid_mask &= (arr != label_nodata)

        # 阈值过滤：灵活处理 9900+、10000、以及冰川边缘插值极端值
        artifact_mask = valid_mask & (arr >= GLACIER_SWE_THRESHOLD_MM)

        n_bad = int(np.count_nonzero(artifact_mask))

        if n_bad > 0:
            bad_vals = arr[artifact_mask]
            date_str = date_dt.strftime("%Y-%m-%d") if date_dt is not None else "unknown-date"

            print(
                f"    🧊 冰川/极端SWE过滤: {date_str}, "
                f"threshold>={GLACIER_SWE_THRESHOLD_MM:.1f} mm, "
                f"n={n_bad}, "
                f"min_bad={np.nanmin(bad_vals):.2f}, "
                f"max_bad={np.nanmax(bad_vals):.2f}"
            )

            arr[artifact_mask] = np.nan

        return arr

    def _load_labels_unified(self):
        """加载ERA5-Land SWE target：支持新版月度daily-band立方体与旧版日文件。"""
        print(f"\n加载ERA5-Land SWE target数据...")

        self.label_data = {}
        target_years = self.load_years
        print(f"  加载标签年份: {target_years}")
        print(f"  标签目录: {self.label_root}")

        label_files = sorted(list(self.label_root.glob("*.tif")))
        print(f"  找到 {len(label_files)} 个标签文件")

        loaded_count = 0
        year_count = {}

        for label_file in label_files:
            try:
                name = label_file.stem

                # 新版：ERA5LAND_SWE_DAILY_AGGR_201507_11132m_chinaMask_float32.tif
                monthly_match = re.search(r'ERA5LAND_SWE_DAILY_AGGR_(\d{4})(\d{2})', name, re.IGNORECASE)

                with rasterio.open(label_file) as ds:
                    label_nodata = ds.nodata
                    src_transform = ds.transform

                    if monthly_match:
                        year = int(monthly_match.group(1))
                        month = int(monthly_match.group(2))
                        if year not in target_years:
                            continue

                        month_days = calendar.monthrange(year, month)[1]
                        n_bands = min(ds.count, month_days)

                        for day in range(1, n_bands + 1):
                            dt = datetime(year, month, day)
                            label_arr = ds.read(day).astype(np.float32)

                            if label_arr.shape != (self.H, self.W):
                                label_arr = self._align_single_layer(
                                    label_arr, src_transform, self.transform, self.H, self.W
                                )

                            label_arr = self._filter_glacier_swe_artifacts(
                                label_arr,
                                label_nodata=label_nodata,
                                date_dt=dt
                            )

                            self.label_data[dt] = (label_arr, label_nodata)
                            loaded_count += 1
                            year_count[year] = year_count.get(year, 0) + 1
                            if loaded_count <= 5:
                                print(f"  {dt.strftime('%Y-%m-%d')}: 加载成功 ({label_file.name} band {day})")
                    else:
                        # 旧版：单日单band文件
                        dt = self._parse_date_from_filename(name)
                        if dt.year not in target_years:
                            continue

                        label_arr = ds.read(1).astype(np.float32)
                        if label_arr.shape != (self.H, self.W):
                            label_arr = self._align_single_layer(
                                label_arr, src_transform, self.transform, self.H, self.W
                            )

                        label_arr = self._filter_glacier_swe_artifacts(
                            label_arr,
                            label_nodata=label_nodata,
                            date_dt=dt
                        )

                        self.label_data[dt] = (label_arr, label_nodata)
                        loaded_count += 1
                        year_count[dt.year] = year_count.get(dt.year, 0) + 1
                        if loaded_count <= 5:
                            print(f"  {dt.strftime('%Y-%m-%d')}: 加载成功 ({label_file.name})")

            except Exception as e:
                print(f"  加载标签文件 {label_file.name} 失败: {e}")
                continue

        if not self.label_data:
            raise ValueError(f"没有加载到任何ERA5-Land SWE target数据 (目标年份: {target_years}, 目录: {self.label_root})")

        print(f"\n✅ ERA5-Land SWE target加载完成:")
        print(f"  总日期数: {loaded_count}")
        for year, count in sorted(year_count.items()):
            print(f"    {year}年: {count} 天")

    def _parse_date_from_filename(self, filename: str) -> datetime:
        """从文件名解析日期 - 支持中国区域的新格式"""

        # 新版 ERA5-Land SWE target：ERA5LAND_SWE_DAILY_AGGR_201507_11132m_...
        match = re.search(r'ERA5LAND_SWE_DAILY_AGGR_(\d{4})(\d{2})', filename, re.IGNORECASE)
        if match:
            year, month = map(int, match.groups())
            return datetime(year, month, 15)

        # 新版 ERA5-Land LST/RH 月日尺度立方体：ERA5LAND_LST_201603_DAILYMEAN_...
        match = re.search(r'ERA5LAND_(?:LST|RH)_(\d{4})(\d{2})_DAILYMEAN', filename, re.IGNORECASE)
        if match:
            year, month = map(int, match.groups())
            return datetime(year, month, 15)

        # 新版 Landsat 年度中值：China_Landsat_L8C2L2_2015_AnnualMedian_...
        match = re.search(r'China_Landsat_L8C2L2_(\d{4})_AnnualMedian', filename, re.IGNORECASE)
        if match:
            return datetime(int(match.group(1)), 7, 1)

        # 新版 S1 月立方体：S1_201501_ERA5SWE010deg_5band_...
        match = re.search(r'S1_(\d{4})(\d{2})_', filename, re.IGNORECASE)
        if match:
            year, month = map(int, match.groups())
            return datetime(year, month, 15)

        # 新版 SMAP 月立方体：SMAP_201507_AMTB_...
        match = re.search(r'SMAP_(\d{4})(\d{2})_', filename, re.IGNORECASE)
        if match:
            year, month = map(int, match.groups())
            return datetime(year, month, 15)

        # 新版静风日：China_CalmDays_2016_threshold0p5.tif
        match = re.search(r'China_CalmDays_(\d{4})_threshold', filename, re.IGNORECASE)
        if match:
            return datetime(int(match.group(1)), 1, 1)


        # 🔥 新增：解析 ERA5_RH_DailyMean_2015_01_27830m.tif
        match = re.search(r'ERA5_RH_DailyMean_(\d{4})_(\d{2})', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return datetime(year, month, 15)  # 月中代表

        # 新增：解析 XGB_SWE_DAILY_025_20150101.tif
        match = re.search(r'XGB_SWE_DAILY_025_(\d{4})(\d{2})(\d{2})', filename)
        if match:
            year, month, day = map(int, match.groups())
            return datetime(year, month, day)

        # 新增：解析 China_Landsat_2015_reflectance.tif
        match = re.search(r'China_Landsat_(\d{4})_reflectance', filename)
        if match:
            year = int(match.group(1))
            return datetime(year, 7, 1)  # 用年中代表

        # 新增：解析 ERA5_ST_201501_UTC0_27830m.tif 或 ERA5_RH_201502_UTC8_27830m.tif（旧格式）
        match = re.search(r'ERA5_(?:ST|RH)_(\d{4})(\d{2})', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return datetime(year, month, 15)  # 月中

        # 新增：解析 resampled_cropped_CHELSA_pr_01_01_2017_V.2.1.tif 或 CHELSA_sfcWind
        match = re.search(r'CHELSA_(?:pr|sfcWind)_(\d{2})_(\d{2})_(\d{4})', filename)
        if match:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3))
            return datetime(year, month, day)

        # 新增：解析 S1_MONTHLY_2015_01.tif
        match = re.search(r'S1_MONTHLY_(\d{4})_(\d{2})', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return datetime(year, month, 15)

        # 新增：解析 China_SMAP_TB_2015_07_25km_days31.tif
        match = re.search(r'China_SMAP_TB_(\d{4})_(\d{2})', filename)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return datetime(year, month, 15)

        # 原有的解析逻辑...
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
            
    def _unpack_meta_item(self, item):
        """解包 meta_index 条目，兼容新旧格式"""
        if len(item) == 4:
            date_dt, r, c, source = item
            return date_dt, r, c, source
        else:
            date_dt, r, c = item
            return date_dt, r, c, 'random'  # 旧数据默认为随机采样
            
            
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
            # 🔥 兼容字典格式
            if isinstance(self.ls_data, dict):
                # 取第一个年份的LS数据检查形状
                first_year = list(self.ls_data.keys())[0]
                ls_arr = self.ls_data[first_year]
                C, H, W = ls_arr.shape
            else:
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


        
    def _interpolate_nan_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        对单个patch进行NaN插值填充
        改进：对全NaN的情况返回均值填充
        """
        if not np.isnan(patch).any():
            return patch

        # 检查是否全为 NaN
        if np.all(np.isnan(patch)):
            # 全NaN时返回0（或全局均值）
            return np.zeros_like(patch)

        try:
            from scipy.interpolate import griddata

            # 获取网格坐标
            x = np.arange(patch.shape[1])
            y = np.arange(patch.shape[0])
            xx, yy = np.meshgrid(x, y)

            # 有效点
            valid_mask = ~np.isnan(patch)
            if not valid_mask.any():
                return np.zeros_like(patch)

            # 如果有效点太少（少于3个），使用全局均值
            if np.sum(valid_mask) < 3:
                mean_value = np.nanmean(patch)
                if np.isnan(mean_value):
                    mean_value = 0.0
                result = patch.copy()
                result[np.isnan(result)] = mean_value
                return result

            # 获取有效点的坐标和值
            valid_points = np.column_stack([xx[valid_mask], yy[valid_mask]])
            valid_values = patch[valid_mask]

            # 判断是否所有有效点共线
            unique_x = np.unique(valid_points[:, 0])
            unique_y = np.unique(valid_points[:, 1])

            if len(unique_x) == 1 or len(unique_y) == 1:
                interpolation_method = 'nearest'
            else:
                interpolation_method = 'linear'

            # 无效点
            invalid_mask = np.isnan(patch)
            invalid_points = np.column_stack([xx[invalid_mask], yy[invalid_mask]])

            try:
                interpolated = griddata(valid_points, valid_values, invalid_points, 
                                        method=interpolation_method, fill_value=0.0)

                result = patch.copy()
                result[invalid_mask] = interpolated
                return result

            except Exception as e:
                # 插值失败时使用最近邻插值
                try:
                    interpolated = griddata(valid_points, valid_values, invalid_points, 
                                            method='nearest', fill_value=0.0)
                    result = patch.copy()
                    result[invalid_mask] = interpolated
                    return result
                except:
                    # 最后的备选方案：局部均值填充
                    result = patch.copy()
                    for i in range(result.shape[0]):
                        for j in range(result.shape[1]):
                            if np.isnan(result[i, j]):
                                i_min = max(0, i-1)
                                i_max = min(result.shape[0], i+2)
                                j_min = max(0, j-1)
                                j_max = min(result.shape[1], j+2)

                                neighborhood = result[i_min:i_max, j_min:j_max]
                                valid_neighbors = neighborhood[~np.isnan(neighborhood)]

                                if len(valid_neighbors) > 0:
                                    result[i, j] = np.mean(valid_neighbors)
                                else:
                                    result[i, j] = 0.0
                    return result

        except Exception as e:
            # 完全失败时返回0填充
            print(f"警告: 插值失败 ({e})，使用0填充")
            return np.nan_to_num(patch, nan=0.0)

    # ============================================================
    # 固定152000增量样本池
    # ============================================================
    @staticmethod
    def _snow_year_of_date(date_dt: datetime) -> int:
        """积雪年：当年9月1日至次年8月31日，以起始年份命名。"""
        return date_dt.year if (date_dt.month, date_dt.day) >= (9, 1) else date_dt.year - 1

    def _incremental_target_ratios(self):
        """默认保持低值/中值/高值约33%/33%/34%，并细分高值尾部。"""
        ratios = {
            "zero": 0.03,
            "0_1": 0.10,
            "1_5": 0.20,
            "5_10": 0.11,
            "10_20": 0.12,
            "20_30": 0.10,
            "30_50": 0.12,
            "50_80": 0.09,
            "80_120": 0.06,
            "120_200": 0.04,
            "200_max": 0.03,
        }
        if self.incremental_ratio_config is not None:
            if not self.incremental_ratio_config.exists():
                raise FileNotFoundError(
                    f"增量比例配置不存在: {self.incremental_ratio_config}"
                )
            with open(self.incremental_ratio_config, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            ratios = {str(k): float(v) for k, v in loaded.items()}

        ratio_sum = sum(ratios.values())
        if not np.isclose(ratio_sum, 1.0, atol=1e-8):
            raise ValueError(f"增量目标比例之和必须为1，当前={ratio_sum}")
        return ratios

    def _incremental_bin_name(self, swe_mm: float) -> Optional[str]:
        eps = 1e-6
        v = float(swe_mm)
        if not np.isfinite(v) or v < 0 or v >= self.seasonal_max_swe_mm:
            return None
        if v <= eps:
            return "zero"
        if v <= 1:
            return "0_1"
        if v <= 5:
            return "1_5"
        if v <= 10:
            return "5_10"
        if v <= 20:
            return "10_20"
        if v <= 30:
            return "20_30"
        if v <= 50:
            return "30_50"
        if v <= 80:
            return "50_80"
        if v <= 120:
            return "80_120"
        if v <= 200:
            return "120_200"
        return "200_max"

    @staticmethod
    def _allocate_counts_by_ratio(total: int, ratios: dict) -> dict:
        """最大余数法，保证整数配额之和严格等于 total。"""
        raw = {k: total * float(v) for k, v in ratios.items()}
        counts = {k: int(np.floor(v)) for k, v in raw.items()}
        remain = int(total - sum(counts.values()))
        order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
        for k in order[:remain]:
            counts[k] += 1
        return counts

    def _load_incremental_glacier_mask(self) -> np.ndarray:
        """可选冰川掩膜；未提供时返回全False。"""
        mask = np.zeros((self.H, self.W), dtype=bool)
        path = self.incremental_glacier_mask_path
        if path is None:
            print(
                "   ⚠ 未提供冰川掩膜：当前仅依靠积雪年内稳定融尽判据和"
                "SWE上限筛选；正式实验建议补充冰川空间掩膜。"
            )
            return mask
        if not path.exists():
            raise FileNotFoundError(f"冰川掩膜不存在: {path}")
        with rasterio.open(path) as ds:
            raw = ds.read(1).astype(np.float32)
            aligned = self._align_single_layer(
                raw, ds.transform, self.transform, self.H, self.W
            )
        mask = np.isfinite(aligned) & (aligned > 0)
        print(f"   冰川掩膜像元: {int(mask.sum()):,}")
        return mask

    def _compute_seasonal_masks(self):
        """逐积雪年计算季节性积雪格点掩膜。

        仅用于固定152000随机池；Stage 0 station模式不会调用。

        操作判据：
        1. 积雪年为当年9月1日至次年8月31日；
        2. 年最大SWE严格大于 ``seasonal_min_peak_swe_mm``；
        3. 年最大SWE严格小于 ``seasonal_max_swe_mm``；
        4. 从最后一次年最大SWE出现后开始，至少存在连续
           ``seasonal_min_consecutive_snow_free_days`` 个有效日，
           每日SWE不超过 ``seasonal_snow_free_threshold_mm``；
        5. 排除冰川掩膜，并满足积雪年日期覆盖要求。

        不再使用“7—8月无雪比例”作为硬筛选条件。旧参数
        ``seasonal_min_warm_snow_free_ratio`` 仅为命令兼容保留。
        """
        grouped = defaultdict(list)
        for dt in sorted(self.label_data):
            grouped[self._snow_year_of_date(dt)].append(dt)

        glacier_mask = self._load_incremental_glacier_mask()
        diagnostics = {}
        complete_dates = []

        print("\n" + "=" * 72)
        print("❄️ 构建季节性积雪格点-积雪年掩膜")
        print("   积雪年定义：9月1日 → 次年8月31日")
        print(
            f"   年最大SWE: ({self.seasonal_min_peak_swe_mm}, "
            f"{self.seasonal_max_swe_mm}) mm"
        )
        print(
            f"   稳定融尽: 最后一次年峰值后连续≥"
            f"{self.seasonal_min_consecutive_snow_free_days}天 "
            f"SWE≤{self.seasonal_snow_free_threshold_mm} mm"
        )
        print("=" * 72)

        for snow_year, dates in sorted(grouped.items()):
            start = datetime(snow_year, 9, 1)
            end = datetime(snow_year + 1, 8, 31)
            dates = sorted(d for d in dates if start <= d <= end)
            if not dates:
                continue

            expected_days = (end - start).days + 1
            date_coverage = len(dates) / expected_days
            start_ok = dates[0] <= start + timedelta(days=7)
            end_ok = dates[-1] >= end - timedelta(days=7)
            if (
                date_coverage < self.seasonal_min_snow_year_coverage_ratio
                or not start_ok
                or not end_ok
            ):
                print(
                    f"   跳过 SnowYear {snow_year}: 日期覆盖 {len(dates)}/{expected_days} "
                    f"({date_coverage:.1%}), range={dates[0].date()}~{dates[-1].date()}"
                )
                continue

            # 第一遍：计算年最大SWE、有效日数，以及每个像元最后一次年峰值日期。
            annual_max = np.full((self.H, self.W), -np.inf, dtype=np.float32)
            last_peak_day_index = np.full((self.H, self.W), -1, dtype=np.int16)
            valid_count = np.zeros((self.H, self.W), dtype=np.uint16)

            for day_index, dt in enumerate(dates):
                arr, nodata = self.label_data[dt]
                valid = np.isfinite(arr) & (arr >= 0)
                if nodata is not None:
                    valid &= arr != nodata

                valid_count += valid.astype(np.uint16)

                # 使用 >= 而不是 >，确保峰值平台取“最后一次”峰值日期。
                new_or_equal_peak = valid & (arr >= annual_max)
                annual_max = np.where(
                    new_or_equal_peak, arr, annual_max
                ).astype(np.float32, copy=False)
                last_peak_day_index = np.where(
                    new_or_equal_peak, day_index, last_peak_day_index
                ).astype(np.int16, copy=False)

            # 第二遍：只统计最后一次年峰值之后的连续近无雪日。
            current_post_peak_free_run = np.zeros((self.H, self.W), dtype=np.uint16)
            max_post_peak_free_run = np.zeros((self.H, self.W), dtype=np.uint16)

            for day_index, dt in enumerate(dates):
                arr, nodata = self.label_data[dt]
                valid = np.isfinite(arr) & (arr >= 0)
                if nodata is not None:
                    valid &= arr != nodata

                after_last_peak = valid & (day_index > last_peak_day_index)
                near_snow_free = after_last_peak & (
                    arr <= self.seasonal_snow_free_threshold_mm
                )

                # 非近无雪日或缺测日均中断连续序列。
                current_post_peak_free_run = np.where(
                    near_snow_free,
                    current_post_peak_free_run + 1,
                    0,
                ).astype(np.uint16)
                max_post_peak_free_run = np.maximum(
                    max_post_peak_free_run,
                    current_post_peak_free_run,
                )

            min_valid_days = int(
                np.ceil(
                    expected_days * self.seasonal_min_snow_year_coverage_ratio
                )
            )
            coverage_ratio = valid_count.astype(np.float32) / expected_days

            seasonal = (
                (valid_count >= min_valid_days)
                & (annual_max > self.seasonal_min_peak_swe_mm)
                & (annual_max < self.seasonal_max_swe_mm)
                & (
                    max_post_peak_free_run
                    >= self.seasonal_min_consecutive_snow_free_days
                )
                & (~glacier_mask)
            )

            diagnostics[snow_year] = {
                "dates": dates,
                "seasonal_mask": seasonal,
                "annual_max": annual_max,
                "last_peak_day_index": last_peak_day_index,
                "max_post_peak_snow_free_days": max_post_peak_free_run,
                "coverage_ratio": coverage_ratio,
            }
            complete_dates.extend(dates)

            print(
                f"   SnowYear {snow_year}: dates={len(dates)}, "
                f"seasonal pixels={int(seasonal.sum()):,}"
            )

        if not diagnostics:
            raise RuntimeError(
                "没有完整积雪年满足季节性判定所需的日期覆盖。"
                "例如仅加载2015-2018年时，通常可使用2015、2016、2017积雪年。"
            )
        return diagnostics, sorted(set(complete_dates))

    def _candidate_pool_by_bin(self, seasonal_diag, eligible_dates, total_quotas):
        """按日期和SWE箱均衡地产生候选，避免一次保存全国全部像元日。"""
        rng = np.random.default_rng(self.incremental_seed)
        n_dates = max(len(eligible_dates), 1)
        per_date_try = {
            name: max(
                1,
                int(np.ceil(
                    quota * self.incremental_candidate_oversample_factor / n_dates
                )),
            )
            for name, quota in total_quotas.items()
        }
        pools = {name: [] for name in total_quotas}
        station_mask = np.zeros((self.H, self.W), dtype=bool)
        if self.incremental_exclude_station_pixels:
            for r, c in self.station_pixels:
                if 0 <= r < self.H and 0 <= c < self.W:
                    station_mask[r, c] = True
        external_mask = np.zeros((self.H, self.W), dtype=bool)
        for r, c in self.external_excluded_cells:
            if 0 <= r < self.H and 0 <= c < self.W:
                external_mask[r, c] = True

        for dt in tqdm(eligible_dates, desc="固定池候选扫描"):
            sy = self._snow_year_of_date(dt)
            if sy not in seasonal_diag:
                continue
            seasonal = seasonal_diag[sy]["seasonal_mask"]
            arr, nodata = self.label_data[dt]
            valid = seasonal & np.isfinite(arr) & (arr >= 0) & (arr < self.seasonal_max_swe_mm)
            if nodata is not None:
                valid &= arr != nodata
            if self.incremental_exclude_station_pixels:
                valid &= ~station_mask
            if self.external_excluded_cells:
                valid &= ~external_mask

            # 先仅取合格季节雪像元，再一次性向量化分箱，避免对整幅栅格
            # 为每个SWE箱重复构造布尔掩膜。
            flat_valid = np.flatnonzero(valid)
            if flat_valid.size == 0:
                continue
            values = arr.ravel()[flat_valid]
            codes = np.full(values.shape, -1, dtype=np.int8)
            eps = 1e-6
            codes[values <= eps] = 0
            codes[(values > eps) & (values <= 1)] = 1
            codes[(values > 1) & (values <= 5)] = 2
            codes[(values > 5) & (values <= 10)] = 3
            codes[(values > 10) & (values <= 20)] = 4
            codes[(values > 20) & (values <= 30)] = 5
            codes[(values > 30) & (values <= 50)] = 6
            codes[(values > 50) & (values <= 80)] = 7
            codes[(values > 80) & (values <= 120)] = 8
            codes[(values > 120) & (values <= 200)] = 9
            codes[(values > 200) & (values < self.seasonal_max_swe_mm)] = 10
            code_by_name = {
                "zero": 0, "0_1": 1, "1_5": 2, "5_10": 3,
                "10_20": 4, "20_30": 5, "30_50": 6,
                "50_80": 7, "80_120": 8, "120_200": 9,
                "200_max": 10,
            }

            for name in total_quotas:
                if name not in code_by_name:
                    raise ValueError(f"未知增量SWE箱名称: {name}")
                positions = np.flatnonzero(codes == code_by_name[name])
                if positions.size == 0:
                    continue
                take = min(per_date_try[name], positions.size)
                chosen_pos = rng.choice(positions, size=take, replace=False)
                chosen = flat_valid[chosen_pos]
                rows, cols = np.unravel_index(chosen, (self.H, self.W))
                pools[name].extend(
                    (dt, int(r), int(c)) for r, c in zip(rows, cols)
                )

        return pools

    def _refill_incremental_bin_exhaustive(
        self,
        name,
        required,
        validated,
        seen,
        seasonal_diag,
        eligible_dates,
    ):
        """对单个不足分箱做一次定向穷举补采。

        该补采保持原有科学约束不变：
        - 仍然只使用季节性积雪格点；
        - 仍然排除站点格点和外部测试格点；
        - 仍然执行严格空间特征和点特征完整性验证；
        - 不放回、不复制样本；
        - 使用固定随机种子，保证重复运行结果一致。

        目的只是消除“不断提高全局 oversample 倍数”的盲目试错。
        如果穷举后仍不足，才说明当前数据与配额本身不可实现。
        """
        code_by_name = {
            "zero": 0, "0_1": 1, "1_5": 2, "5_10": 3,
            "10_20": 4, "20_30": 5, "30_50": 6,
            "50_80": 7, "80_120": 8, "120_200": 9,
            "200_max": 10,
        }
        if name not in code_by_name:
            raise ValueError(f"未知增量SWE箱名称: {name}")

        station_mask = np.zeros((self.H, self.W), dtype=bool)
        if self.incremental_exclude_station_pixels:
            for r, c in self.station_pixels:
                if 0 <= r < self.H and 0 <= c < self.W:
                    station_mask[r, c] = True

        external_mask = np.zeros((self.H, self.W), dtype=bool)
        for r, c in self.external_excluded_cells:
            if 0 <= r < self.H and 0 <= c < self.W:
                external_mask[r, c] = True

        # 每个箱使用独立且固定的随机流，保证确定性。
        rng = np.random.default_rng(
            self.incremental_seed + 10007 * (code_by_name[name] + 1)
        )
        dates = list(eligible_dates)
        rng.shuffle(dates)

        before = len(validated)
        checked_new = 0
        target_add = required - before
        pbar = tqdm(
            total=max(target_add, 0),
            desc=f"穷举补采 {name}",
            leave=False,
        )

        for dt in dates:
            if len(validated) >= required:
                break

            sy = self._snow_year_of_date(dt)
            if sy not in seasonal_diag:
                continue

            seasonal = seasonal_diag[sy]["seasonal_mask"]
            arr, nodata = self.label_data[dt]
            valid = (
                seasonal
                & np.isfinite(arr)
                & (arr >= 0)
                & (arr < self.seasonal_max_swe_mm)
            )
            if nodata is not None:
                valid &= arr != nodata
            if self.incremental_exclude_station_pixels:
                valid &= ~station_mask
            if self.external_excluded_cells:
                valid &= ~external_mask

            flat_valid = np.flatnonzero(valid)
            if flat_valid.size == 0:
                continue

            values = arr.ravel()[flat_valid]
            eps = 1e-6
            if name == "zero":
                in_bin = values <= eps
            elif name == "0_1":
                in_bin = (values > eps) & (values <= 1)
            elif name == "1_5":
                in_bin = (values > 1) & (values <= 5)
            elif name == "5_10":
                in_bin = (values > 5) & (values <= 10)
            elif name == "10_20":
                in_bin = (values > 10) & (values <= 20)
            elif name == "20_30":
                in_bin = (values > 20) & (values <= 30)
            elif name == "30_50":
                in_bin = (values > 30) & (values <= 50)
            elif name == "50_80":
                in_bin = (values > 50) & (values <= 80)
            elif name == "80_120":
                in_bin = (values > 80) & (values <= 120)
            elif name == "120_200":
                in_bin = (values > 120) & (values <= 200)
            else:  # 200_max
                in_bin = (
                    (values > 200)
                    & (values < self.seasonal_max_swe_mm)
                )

            candidates = flat_valid[np.flatnonzero(in_bin)]
            if candidates.size == 0:
                continue
            rng.shuffle(candidates)

            for flat_index in candidates:
                if len(validated) >= required:
                    break
                r, c = np.unravel_index(int(flat_index), (self.H, self.W))
                key = (dt, int(r), int(c))

                # STAGE6_REFILL_EXCLUSION_CHECK_V1
                # 初始候选路径已排除旧manifest；补采路径也必须执行相同检查。
                sample_id = f"{dt:%Y%m%d}_{int(r)}_{int(c)}"
                if sample_id in getattr(
                    self, "_incremental_excluded_sample_ids", set()
                ):
                    continue

                if key in seen:
                    continue
                seen.add(key)
                checked_new += 1

                # 先检查空间特征，失败时不要再无谓构建点特征。
                conv_patch = self._build_spatial_features(dt, int(r), int(c))
                if conv_patch is None:
                    continue
                point_feats = self._build_point_features(dt, int(r), int(c))
                if point_feats is None:
                    continue

                validated.append((dt, int(r), int(c)))
                pbar.update(1)

        pbar.close()
        return len(validated) - before, checked_new

    def _fixed_fold_id(self, row: int, col: int) -> int:
        """同一空间块固定进入同一折，避免同位置不同日期跨折。"""
        br = int(row) // self.incremental_fold_block_pixels
        bc = int(col) // self.incremental_fold_block_pixels
        token = f"{self.incremental_seed}:{br}:{bc}".encode("utf-8")
        digest = hashlib.md5(token).hexdigest()
        return int(digest[:8], 16) % 10 + 1

    def _build_incremental_manifest(self):
        """一次性固定152000个合格随机样本并分成12k/20k/40k/80k。"""
        manifest = self.incremental_manifest_path
        manifest.parent.mkdir(parents=True, exist_ok=True)
        ratios = self._incremental_target_ratios()

        # 构建Stage6新增包时，严格排除旧manifest已有sample_id。
        excluded_manifest_sample_ids = set()
        if self.incremental_exclude_manifest_path is not None:
            exclude_path = self.incremental_exclude_manifest_path
            if not exclude_path.exists():
                raise FileNotFoundError(f"排除清单不存在: {exclude_path}")
            exclude_df = pd.read_csv(exclude_path)
            if "sample_id" in exclude_df.columns:
                excluded_manifest_sample_ids = set(
                    exclude_df["sample_id"].astype(str).tolist()
                )
            elif {"date", "row", "col"}.issubset(exclude_df.columns):
                ex_dates = pd.to_datetime(exclude_df["date"], errors="raise")
                excluded_manifest_sample_ids = {
                    f"{dt:%Y%m%d}_{int(r)}_{int(c)}"
                    for dt, r, c in zip(
                        ex_dates, exclude_df["row"], exclude_df["col"]
                    )
                }
            else:
                raise ValueError(
                    "排除清单必须包含sample_id，或同时包含date/row/col"
                )
            print(
                f"\n🚫 新增样本排除旧清单: {exclude_path}；"
                f"sample_id={len(excluded_manifest_sample_ids):,}"
            )

        # STAGE6_REFILL_EXCLUSION_STORE_V1
        # 初始候选与定向穷举补采必须共享同一旧manifest排除集合。
        self._incremental_excluded_sample_ids = set(
            excluded_manifest_sample_ids
        )
        print(
            "   🚫 已将旧manifest排除集合绑定到补采路径: "
            f"{len(self._incremental_excluded_sample_ids):,} 个sample_id"
        )

        stage_quotas = {
            stage_id: self._allocate_counts_by_ratio(size, ratios)
            for stage_id, size in enumerate(self.incremental_stage_sizes, start=1)
        }
        total_quotas = {
            name: sum(q[name] for q in stage_quotas.values())
            for name in ratios
        }

        seasonal_diag, eligible_dates = self._compute_seasonal_masks()
        candidate_pools = self._candidate_pool_by_bin(
            seasonal_diag, eligible_dates, total_quotas
        )

        rng = np.random.default_rng(self.incremental_seed)
        validated = {name: [] for name in ratios}
        print("\n🔎 对固定池候选执行严格特征完整性验证")
        for name, required in total_quotas.items():
            candidates = candidate_pools[name]
            rng.shuffle(candidates)
            seen = set()
            for dt, r, c in tqdm(candidates, desc=f"验证 {name}", leave=False):
                key = (dt, r, c)
                if key in seen:
                    continue
                seen.add(key)
                sample_id = f"{dt:%Y%m%d}_{int(r)}_{int(c)}"
                if sample_id in excluded_manifest_sample_ids:
                    continue
                conv_patch = self._build_spatial_features(dt, r, c)
                if conv_patch is None:
                    continue
                point_feats = self._build_point_features(dt, r, c)
                if point_feats is None:
                    continue
                validated[name].append((dt, r, c))
                if len(validated[name]) >= required:
                    break

            initial_valid = len(validated[name])
            refill_added = 0
            refill_checked = 0
            if initial_valid < required:
                print(
                    f"   {name:>10s}: 初始候选验证仅 {initial_valid:,} / "
                    f"需要 {required:,}，开始对该箱定向穷举补采；"
                    "不再提高全局过采倍数。"
                )
                refill_added, refill_checked = self._refill_incremental_bin_exhaustive(
                    name=name,
                    required=required,
                    validated=validated[name],
                    seen=seen,
                    seasonal_diag=seasonal_diag,
                    eligible_dates=eligible_dates,
                )

            if len(validated[name]) < required:
                raise RuntimeError(
                    f"固定池 {name} 箱在定向穷举所有未检查候选后仍不足："
                    f"需要 {required:,}，实际最多找到 {len(validated[name]):,}。"
                    "这已经不是 oversample 倍数问题，而是当前季节性判据、"
                    "严格特征完整性和该分箱配额三者在现有数据下不可同时满足。"
                )

            extra = ""
            if refill_added > 0:
                extra = (
                    f"；定向补采检查 {refill_checked:,} 个新候选，"
                    f"补入 {refill_added:,}"
                )
            print(
                f"   {name:>10s}: 初始候选 {len(candidates):,} -> "
                f"有效 {len(validated[name]):,} / 需要 {required:,}{extra}"
            )

        rows_out = []
        offsets = {name: 0 for name in ratios}
        for stage_id, stage_size in enumerate(self.incremental_stage_sizes, start=1):
            for name in ratios:
                n_take = stage_quotas[stage_id][name]
                start = offsets[name]
                end = start + n_take
                selected = validated[name][start:end]
                offsets[name] = end
                for dt, r, c in selected:
                    swe = float(self.label_data[dt][0][r, c])
                    sy = self._snow_year_of_date(dt)
                    diag = seasonal_diag[sy]
                    lon, lat = self._pixel_to_lonlat(r, c)
                    rows_out.append({
                        "sample_id": f"{dt:%Y%m%d}_{r}_{c}",
                        "date": dt.strftime("%Y-%m-%d"),
                        "snow_year": int(sy),
                        "row": int(r),
                        "col": int(c),
                        "lon": float(lon),
                        "lat": float(lat),
                        "swe_mm": swe,
                        "swe_bin": name,
                        "annual_max_swe_mm": float(diag["annual_max"][r, c]),
                        "last_peak_day_index": int(
                            diag["last_peak_day_index"][r, c]
                        ),
                        "max_post_peak_snow_free_days": int(
                            diag["max_post_peak_snow_free_days"][r, c]
                        ),
                        "snow_year_coverage_ratio": float(
                            diag["coverage_ratio"][r, c]
                        ),
                        "stage_id": int(stage_id),
                        "fold_id": 0,
                        "source": "incremental_random",
                    })

            actual = sum(1 for x in rows_out if x["stage_id"] == stage_id)
            if actual != stage_size:
                raise RuntimeError(
                    f"Stage {stage_id} 分配数量错误: {actual} != {stage_size}"
                )

        df = pd.DataFrame(rows_out)

        # 固定十折：默认在每个stage、每个SWE箱内做可复现的随机分配，
        # 保证各折数量和数值分布接近。若 incremental_fold_block_pixels>0，
        # 才改用空间块哈希分折。
        if self.incremental_fold_block_pixels > 0:
            df["fold_id"] = [
                self._fixed_fold_id(int(r), int(c))
                for r, c in zip(df["row"], df["col"])
            ]
        else:
            df["fold_id"] = 0
            fold_rng = np.random.default_rng(self.incremental_seed + 1009)
            for (_, _), group in df.groupby(["stage_id", "swe_bin"], sort=True):
                idxs = group.index.to_numpy(copy=True)
                fold_rng.shuffle(idxs)
                folds = np.resize(np.arange(1, 11, dtype=np.int16), len(idxs))
                df.loc[idxs, "fold_id"] = folds
            df["fold_id"] = df["fold_id"].astype(int)

        if len(df) != self.incremental_pool_size:
            raise RuntimeError(
                f"最终清单数量错误: {len(df):,} != {self.incremental_pool_size:,}"
            )
        if df["sample_id"].duplicated().any():
            dup = int(df["sample_id"].duplicated().sum())
            raise RuntimeError(f"固定增量清单存在重复 sample_id: {dup}")

        df = df.sort_values(
            ["stage_id", "swe_bin", "date", "row", "col"]
        ).reset_index(drop=True)
        df.to_csv(manifest, index=False, encoding="utf-8-sig")

        meta = {
            "created_at": datetime.now().isoformat(),
            "manifest": str(manifest),
            "pool_size": self.incremental_pool_size,
            "stage_sizes": self.incremental_stage_sizes,
            "seed": self.incremental_seed,
            "ratios": ratios,
            "stage_counts": df.groupby("stage_id").size().astype(int).to_dict(),
            "stage_bin_counts": (
                df.groupby(["stage_id", "swe_bin"]).size().astype(int).to_dict()
            ),
            "seasonal_criteria": {
                "snow_year": "Sep-01 to next Aug-31",
                "min_peak_swe_mm": self.seasonal_min_peak_swe_mm,
                "max_swe_mm_exclusive": self.seasonal_max_swe_mm,
                "snow_free_threshold_mm": self.seasonal_snow_free_threshold_mm,
                "min_warm_snow_free_ratio": None,
                "post_last_peak_min_consecutive_snow_free_days": self.seasonal_min_consecutive_snow_free_days,
                "min_snow_year_coverage_ratio": self.seasonal_min_snow_year_coverage_ratio,
                "glacier_mask": str(self.incremental_glacier_mask_path or ""),
                "exclude_station_pixels": self.incremental_exclude_station_pixels,
            },
        }
        # JSON不支持tuple key，单独转换 stage_bin_counts。
        meta["stage_bin_counts"] = {
            f"stage{a}:{b}": int(v)
            for (a, b), v in df.groupby(["stage_id", "swe_bin"]).size().items()
        }
        meta_path = manifest.with_suffix(manifest.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 固定增量清单已保存: {manifest}")
        print(f"   总样本: {len(df):,}")
        print(f"   阶段分配: {df.groupby('stage_id').size().to_dict()}")
        print(f"   元数据: {meta_path}")

    def _load_incremental_stage_from_manifest(self):
        manifest = self.incremental_manifest_path
        if not manifest.exists():
            raise FileNotFoundError(f"增量清单不存在: {manifest}")
        df = pd.read_csv(manifest)
        required = {
            "date", "row", "col", "stage_id", "fold_id", "sample_id"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"增量清单缺少字段: {sorted(missing)}")

        if self.incremental_selection_mode == "cumulative":
            stage_df = df[
                df["stage_id"].astype(int) <= self.incremental_stage
            ].copy()
            expected = sum(
                self.incremental_stage_sizes[: self.incremental_stage]
            )
            source_name = (
                f"incremental_cumulative_stage_{self.incremental_stage}"
            )
        else:
            stage_df = df[
                df["stage_id"].astype(int) == self.incremental_stage
            ].copy()
            expected = self.incremental_stage_sizes[
                self.incremental_stage - 1
            ]
            source_name = (
                f"incremental_package_stage_{self.incremental_stage}"
            )
        if len(stage_df) != expected:
            raise RuntimeError(
                f"Stage {self.incremental_stage} 清单数量 {len(stage_df):,} "
                f"!= 预期 {expected:,}"
            )
        if stage_df["sample_id"].duplicated().any():
            raise RuntimeError(f"Stage {self.incremental_stage} 存在重复sample_id")

        stage_df["date"] = pd.to_datetime(stage_df["date"], errors="raise")
        meta_index = []
        fold_ids = []
        missing_dates = 0
        for row in stage_df.itertuples(index=False):
            dt = row.date.to_pydatetime()
            if dt not in self.label_data or dt not in self.date_to_index:
                missing_dates += 1
                continue
            r, c = int(row.row), int(row.col)
            if not (0 <= r < self.H and 0 <= c < self.W):
                continue
            meta_index.append((dt, r, c, source_name))
            fold_ids.append(int(row.fold_id))

        if missing_dates:
            raise RuntimeError(
                f"增量清单中有 {missing_dates} 条日期不在当前加载年份/特征时间轴内；"
                "请保持 PRETRAIN_YEARS 与生成清单时一致。"
            )
        self.meta_index = meta_index
        self.sample_fold_ids = np.asarray(fold_ids, dtype=np.int16)
        if len(self.sample_fold_ids) != len(self.meta_index):
            raise RuntimeError("sample_fold_ids 与 meta_index 长度不一致")

        print(f"\n📦 已加载累计样本池 Stage 1-{self.incremental_stage}")
        print(f"   样本数: {len(self.meta_index):,}")
        unique, counts = np.unique(self.sample_fold_ids, return_counts=True)
        print(f"   固定fold分布: {dict(zip(unique.tolist(), counts.tolist()))}")

    # ============================================================
    # 跨阶段固定归一化
    # ============================================================
    def _apply_fixed_label_scale(self):
        if self.fixed_label_min_mm is not None:
            self.label_min = float(self.fixed_label_min_mm)
        if self.fixed_label_max_mm is not None:
            self.label_max = float(self.fixed_label_max_mm)
        if self.label_max <= self.label_min:
            raise ValueError("固定label_max必须大于label_min")
        self.swe_min = self.label_min
        self.swe_max = self.label_max

        actual_values = []
        for item in self.meta_index[: min(len(self.meta_index), 20000)]:
            dt, r, c = item[:3]
            val = float(self.label_data[dt][0][r, c])
            if np.isfinite(val):
                actual_values.append(val)
        if actual_values:
            actual_min = float(np.min(actual_values))
            actual_max = float(np.max(actual_values))
            if actual_min < self.label_min or actual_max > self.label_max:
                print(
                    f"   ⚠ 当前样本实际SWE范围[{actual_min:.3f}, {actual_max:.3f}] "
                    f"超出固定归一化范围[{self.label_min:.3f}, {self.label_max:.3f}]；"
                    "归一化值可能落在[0,1]之外。"
                )
        print(
            f"   固定标签归一化范围: [{self.label_min:.3f}, "
            f"{self.label_max:.3f}] mm"
        )

    def _normalization_payload(self):
        keys = [
            "conv_min", "conv_max", "point_min", "point_max",
            "label_min", "label_max", "swe_min", "swe_max",
            "lon_raw_min", "lon_raw_max", "lat_raw_min", "lat_raw_max",
            "doy_raw_min", "doy_raw_max", "C_conv", "C_point",
        ]
        payload = {
            "created_at": datetime.now().isoformat(),
            "source_sampling_mode": self.sampling_mode,
            "fixed_label_min_mm": self.fixed_label_min_mm,
            "fixed_label_max_mm": self.fixed_label_max_mm,
        }
        for key in keys:
            if hasattr(self, key):
                value = getattr(self, key)
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif isinstance(value, np.generic):
                    value = value.item()
                payload[key] = value
        return payload

    def _maybe_save_normalization_config(self):
        path = self.normalization_config_path
        if path is None:
            return
        if self.normalization_mode == "load":
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._normalization_payload(), f, indent=2, ensure_ascii=False)
        print(f"   ✅ 固定归一化配置已保存: {path}")

    def _maybe_load_normalization_config(self) -> bool:
        path = self.normalization_config_path
        if path is None:
            if self.normalization_mode == "load":
                raise ValueError("normalization_mode=load 时必须提供 normalization_config_path")
            return False
        should_load = self.normalization_mode == "load" or (
            self.normalization_mode == "auto" and path.exists()
        )
        if not should_load:
            return False
        if not path.exists():
            raise FileNotFoundError(f"固定归一化配置不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        method = str(payload.get("method", "minmax")).strip().lower()
        if method == "clip_then_zscore":
            if int(payload.get("C_conv", -1)) != int(self.C_conv):
                raise ValueError(
                    f"统一归一化C_conv不一致: config={payload.get('C_conv')}, dataset={self.C_conv}"
                )
            if int(payload.get("C_point", -1)) != int(self.C_point):
                raise ValueError(
                    f"统一归一化C_point不一致: config={payload.get('C_point')}, dataset={self.C_point}"
                )
            if int(payload.get("patch_size", -1)) != int(self.patch_size):
                raise ValueError(
                    f"统一归一化patch_size不一致: config={payload.get('patch_size')}, dataset={self.patch_size}"
                )
            array_keys = [
                "conv_clip_low", "conv_clip_high", "conv_mean", "conv_std",
                "point_clip_low", "point_clip_high", "point_mean", "point_std",
            ]
            missing = [k for k in array_keys if k not in payload]
            if missing:
                raise ValueError(f"统一归一化配置缺少字段: {missing}")
            for key in array_keys:
                setattr(self, key, np.asarray(payload[key], dtype=np.float32))
            self.point_transform = list(payload.get("point_transform", ["zscore"] * self.C_point))
            if len(self.point_transform) != self.C_point:
                raise ValueError("point_transform长度与C_point不一致")
            self.label_min = float(payload["label_min"])
            self.label_max = float(payload["label_max"])
            self.swe_min = self.label_min
            self.swe_max = self.label_max
            self.normalization_method = "clip_then_zscore"
            print(f"   ✅ 已加载统一clip+zscore归一化: {path}")
            print(f"   标签范围: [{self.label_min}, {self.label_max}) mm")
            return True

        array_keys = {"conv_min", "conv_max", "point_min", "point_max"}
        for key, value in payload.items():
            if key in array_keys:
                setattr(self, key, np.asarray(value, dtype=np.float32))
            elif key in {
                "label_min", "label_max", "swe_min", "swe_max",
                "lon_raw_min", "lon_raw_max", "lat_raw_min", "lat_raw_max",
                "doy_raw_min", "doy_raw_max", "C_conv", "C_point",
            }:
                setattr(self, key, value)
        required = ["conv_min", "conv_max", "point_min", "point_max", "label_min", "label_max"]
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise ValueError(f"固定归一化配置缺少字段: {missing}")
        self.swe_min = float(getattr(self, "swe_min", self.label_min))
        self.swe_max = float(getattr(self, "swe_max", self.label_max))
        self.normalization_method = "minmax"
        print(f"   ✅ 已加载固定min-max归一化配置: {path}")
        print(f"   标签范围: [{self.label_min}, {self.label_max}] mm")
        return True

    def _build_sample_index(self):
        """按 sampling_mode 构建样本索引。

        Stage 0 的 station 模式不执行任何季节性积雪判定。
        季节性判据只在 incremental 固定随机池构建时执行。
        """
        self.meta_index = []
        self.sample_fold_ids = None

        if self.sampling_mode == "incremental":
            manifest_missing = not self.incremental_manifest_path.exists()
            if self.build_incremental_manifest or manifest_missing:
                if manifest_missing and not self.build_incremental_manifest:
                    raise FileNotFoundError(
                        f"增量清单不存在: {self.incremental_manifest_path}。"
                        "首次运行 Stage 1 请增加 --build_incremental_manifest。"
                    )
                self._build_incremental_manifest()
            self._load_incremental_stage_from_manifest()

            if not self.meta_index:
                raise RuntimeError(
                    f"累计清单 Stage 1-{self.incremental_stage} 没有可用样本"
                )

            # 清单已经按固定目标分布构建，严禁再次 quota/adaptive 补采或裁剪。
            print(
                "\n📦 cumulative incremental 模式：使用 stage_id <= 当前Stage 的累计池；"
                "不重新随机、不执行 quota/adaptive。"
            )
            self._print_sample_statistics()
            self._analyze_swe_distribution()
            return

        if self.sampling_mode in {"random", "hybrid"}:
            self._build_random_samples()

        if self.sampling_mode in {"station", "hybrid"}:
            if not self.station_pixels:
                raise RuntimeError(
                    "sampling_mode 包含 station，但没有加载到任何站点像元；"
                    "请检查 --station_guide_file 及经纬度列。"
                )
            self._build_station_guided_samples()

        if not self.meta_index:
            raise RuntimeError(f"sampling_mode={self.sampling_mode} 未生成任何训练样本")

        if self.sampling_mode == "station":
            # 纯站点模式必须保持空间来源纯净：不从全国其他像元补 quota。
            if self.use_target_quota_sampling or self.use_adaptive_supplement:
                print(
                    "\n📍 station-only 模式：忽略 quota/adaptive 补采与重平衡，"
                    "保留站点位置的自然 SWE 分布。"
                )
        elif self.use_target_quota_sampling:
            self._supplement_quota_shortages()
            self._rebalance_to_target_distribution()
        else:
            if self.use_adaptive_supplement:
                self._adaptive_supplement()
            self._cap_zero_target_samples(max_zero_ratio=self.max_zero_target_ratio)

        np.random.shuffle(self.meta_index)
        self._print_sample_statistics()
        self._analyze_swe_distribution()

    @property
    def _target_bin_names(self):
        return list(self.target_swe_ratios.keys())

    def _target_bin_index_array(self, values: np.ndarray) -> np.ndarray:
        """把 SWE 数组映射到目标比例 bin；精确0值单独成箱。"""
        values = np.asarray(values, dtype=np.float32)
        idx = np.full(values.shape, -1, dtype=np.int16)
        eps = 1e-6
        idx[values <= eps] = 0
        idx[(values > eps) & (values <= 1)] = 1
        idx[(values > 1) & (values <= 5)] = 2
        idx[(values > 5) & (values <= 10)] = 3
        idx[(values > 10) & (values <= 20)] = 4
        idx[(values > 20) & (values <= 30)] = 5
        idx[(values > 30) & (values <= 50)] = 6
        idx[(values > 50) & (values <= 80)] = 7
        idx[(values > 80) & (values <= 120)] = 8
        idx[(values > 120) & (values <= 200)] = 9
        idx[(values > 200) & (values <= 500)] = 10
        idx[values > 500] = 11
        return idx

    def _target_bin_name(self, swe_val: float) -> str:
        idx = int(self._target_bin_index_array(np.array([swe_val]))[0])
        if idx < 0:
            return "invalid"
        return self._target_bin_names[idx]

    def _allocate_target_counts(self, total: int) -> dict:
        """按最大余数法分配整数 quota，确保各箱数量之和严格等于 total。"""
        names = self._target_bin_names
        raw = np.array([self.target_swe_ratios[n] * total for n in names], dtype=float)
        counts = np.floor(raw).astype(int)
        remainder = int(total - counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            counts[order[:remainder]] += 1
        return {name: int(count) for name, count in zip(names, counts)}

    def _build_random_samples(self):
        """
        按每日目标配额构建候选样本。

        [CONTRACT]
        - 不再使用 0.3x / 5x 权重概率。
        - 每天先按 target_swe_ratios 分箱定额抽取。
        - 当天某个区间不足时，剩余名额从当天其他可用区间补齐；
          最终全局比例由 _rebalance_to_target_distribution() 再精确控制。
        """
        self.meta_index = []
        samples_per_date = {}
        samples_per_year = {}
        valid_by_bin = {name: 0 for name in self._target_bin_names}
        attempted_by_bin = {name: 0 for name in self._target_bin_names}
        total_candidates = 0
        skip_log_count = 0

        for date_dt, (label_arr, label_nodata) in self.label_data.items():
            if date_dt not in self.date_to_index:
                if skip_log_count < 10:
                    print(f"跳过日期 {date_dt.strftime('%Y-%m-%d')}，不在卷积特征时间轴中")
                    skip_log_count += 1
                continue

            if label_nodata is not None:
                valid_mask = (label_arr != label_nodata) & np.isfinite(label_arr)
            else:
                valid_mask = np.isfinite(label_arr)
            valid_mask &= (label_arr >= 0)

            rows, cols = np.where(valid_mask)
            if len(rows) < self.min_valid_pixels:
                continue

            interior = (
                (rows - self.R >= 0) & (rows + self.R < self.H) &
                (cols - self.R >= 0) & (cols + self.R < self.W)
            )
            rows = rows[interior]
            cols = cols[interior]
            if len(rows) == 0:
                continue

            values = label_arr[rows, cols].astype(np.float32)
            bin_idx = self._target_bin_index_array(values)
            keep = bin_idx >= 0
            rows, cols, values, bin_idx = rows[keep], cols[keep], values[keep], bin_idx[keep]
            total_candidates += len(rows)

            n_samples = len(rows) if self.samples_per_day is None else min(self.samples_per_day, len(rows))
            daily_quota = self._allocate_target_counts(n_samples)

            selected_local = []
            selected_mask = np.zeros(len(rows), dtype=bool)

            for b, name in enumerate(self._target_bin_names):
                pool = np.flatnonzero(bin_idx == b)
                n_take = min(daily_quota[name], len(pool))
                if n_take > 0:
                    chosen = np.random.choice(pool, size=n_take, replace=False)
                    selected_local.extend(chosen.tolist())
                    selected_mask[chosen] = True
                    attempted_by_bin[name] += n_take

            # 当天某些箱不足时，从当天其他剩余像元补齐候选数量。
            missing = n_samples - len(selected_local)
            if missing > 0:
                remaining = np.flatnonzero(~selected_mask)
                if len(remaining) > 0:
                    n_fill = min(missing, len(remaining))
                    fill = np.random.choice(remaining, size=n_fill, replace=False)
                    selected_local.extend(fill.tolist())

            added_today = 0
            for local_idx in selected_local:
                r = int(rows[local_idx])
                c = int(cols[local_idx])
                name = self._target_bin_names[int(bin_idx[local_idx])]

                try:
                    conv_patch = self._build_spatial_features(date_dt, r, c)
                    point_feats = self._build_point_features(date_dt, r, c)
                    if conv_patch is None or point_feats is None:
                        continue
                    self.meta_index.append((date_dt, r, c, 'random'))
                    valid_by_bin[name] += 1
                    added_today += 1
                except Exception:
                    continue

            samples_per_date[date_dt] = added_today
            samples_per_year[date_dt.year] = samples_per_year.get(date_dt.year, 0) + added_today

        self.random_sample_stats = {
            'total_candidates': total_candidates,
            'samples_per_year': samples_per_year,
            'samples_per_date': samples_per_date,
            'attempted_by_bin': attempted_by_bin,
            'valid_by_bin': valid_by_bin,
        }

        print("\n  每日目标配额候选采样完成:")
        print(f"    原始候选点: {total_candidates:,}")
        print(f"    通过特征验证的候选样本: {len(self.meta_index):,}")
        for name in self._target_bin_names:
            print(f"    {name:>10s}: {valid_by_bin[name]:,}")

    def _supplement_quota_shortages(self):
        """
        对第一次候选池中不足的 SWE 区间执行第二遍定向补采。

        [CONTRACT]
        - 只补 quota 短缺箱，不再增加已经充足的低值箱。
        - 使用宽松但可复现的特征验证，补采样本来源标记为 quota_supplement。
        - 日期顺序随机化，避免补采集中在某一年或少数日期。
        - 最终比例仍由 _rebalance_to_target_distribution() 精确裁剪。
        """
        if not self.use_quota_shortage_supplement:
            return
        if not self.meta_index:
            raise RuntimeError("二次定向补采失败：初始候选池为空")

        # 去重并统计当前每个 bin 的可用候选数量。
        existing = set()
        current = {name: 0 for name in self._target_bin_names}
        for item in self.meta_index:
            date_dt, r, c = item[:3]
            key = (date_dt, int(r), int(c))
            if key in existing:
                continue
            existing.add(key)
            y = float(self.label_data[date_dt][0][r, c])
            name = self._target_bin_name(y)
            if name in current:
                current[name] += 1

        quotas = self._allocate_target_counts(self.target_total_samples)
        remaining = {
            name: max(0, quotas[name] - current[name])
            for name in self._target_bin_names
        }

        if sum(remaining.values()) == 0:
            print("\n🎯 二次定向补采：各区间候选已经满足 quota，无需补采")
            return

        print("\n" + "=" * 72)
        print("🎯 第二遍：短缺 SWE 区间定向补采")
        print(f"   目标样本总数: {self.target_total_samples:,}")
        print(f"   最大轮数: {self.quota_supplement_max_rounds}")
        print(f"   尝试倍率: {self.quota_supplement_try_factor:.1f}x")
        print("   初始短缺:")
        for name in self._target_bin_names:
            if remaining[name] > 0:
                print(
                    f"   {name:>10s}: 当前={current[name]:>7,}, "
                    f"目标={quotas[name]:>7,}, 缺={remaining[name]:>7,}"
                )
        print("=" * 72)

        rng = np.random.default_rng(20260711)
        success = {name: 0 for name in self._target_bin_names}
        attempts = {name: 0 for name in self._target_bin_names}

        valid_dates = [d for d in self.label_data.keys() if d in self.date_to_index]

        for round_idx in range(self.quota_supplement_max_rounds):
            if sum(remaining.values()) == 0:
                break

            rng.shuffle(valid_dates)
            print(
                f"\n   补采轮次 {round_idx + 1}/{self.quota_supplement_max_rounds}, "
                f"剩余短缺={sum(remaining.values()):,}"
            )

            for date_pos, date_dt in enumerate(valid_dates):
                if sum(remaining.values()) == 0:
                    break

                label_arr, label_nodata = self.label_data[date_dt]
                if label_nodata is not None:
                    valid_mask = (label_arr != label_nodata) & np.isfinite(label_arr)
                else:
                    valid_mask = np.isfinite(label_arr)
                valid_mask &= (label_arr >= 0)

                # patch 必须完整落在栅格内部。
                valid_mask[:self.R, :] = False
                valid_mask[-self.R:, :] = False
                valid_mask[:, :self.R] = False
                valid_mask[:, -self.R:] = False

                rows, cols = np.where(valid_mask)
                if len(rows) == 0:
                    continue

                values = label_arr[rows, cols].astype(np.float32)
                bin_idx = self._target_bin_index_array(values)

                dates_left = max(1, len(valid_dates) - date_pos)

                for b, name in enumerate(self._target_bin_names):
                    need = remaining[name]
                    if need <= 0:
                        continue

                    pool = np.flatnonzero(bin_idx == b)
                    if len(pool) == 0:
                        continue

                    # 按“剩余缺口 / 剩余日期”分散补采，并乘尝试倍率抵消特征验证失败。
                    base_per_date = max(1, int(np.ceil(need / dates_left)))
                    n_try = int(np.ceil(base_per_date * self.quota_supplement_try_factor))
                    n_try = min(n_try, len(pool))
                    if n_try <= 0:
                        continue

                    chosen = rng.choice(pool, size=n_try, replace=False)
                    for local_idx in chosen:
                        if remaining[name] <= 0:
                            break

                        r = int(rows[local_idx])
                        c = int(cols[local_idx])
                        key = (date_dt, r, c)
                        if key in existing:
                            continue

                        attempts[name] += 1
                        is_valid, _, conv_patch, point_feats = self._validate_station_sample_with_reason(
                            date_dt, r, c
                        )
                        if not is_valid or conv_patch is None or point_feats is None:
                            continue

                        # 与 __getitem__ 中宽松样本的质量门槛保持一致。
                        if np.sum(np.abs(point_feats[:6]) > 0.01) < 2:
                            continue

                        self.meta_index.append((date_dt, r, c, 'quota_supplement'))
                        existing.add(key)
                        success[name] += 1
                        remaining[name] -= 1

            print("   本轮结束后的短缺:")
            for name in self._target_bin_names:
                if remaining[name] > 0:
                    print(f"   {name:>10s}: 仍缺 {remaining[name]:>7,}")

        print("\n   二次定向补采结果:")
        for name in self._target_bin_names:
            if attempts[name] > 0 or success[name] > 0:
                rate = success[name] / max(attempts[name], 1) * 100
                print(
                    f"   {name:>10s}: 尝试={attempts[name]:>8,}, "
                    f"成功={success[name]:>7,}, 成功率={rate:>5.1f}%, "
                    f"仍缺={remaining[name]:>7,}"
                )
        print(f"   新增候选合计: {sum(success.values()):,}")
        print("=" * 72)

        if self.strict_target_quota and sum(remaining.values()) > 0:
            details = ", ".join(
                f"{name}缺{remaining[name]}"
                for name in self._target_bin_names
                if remaining[name] > 0
            )
            raise RuntimeError(
                "严格目标比例无法满足：二次定向补采后仍有短缺。"
                f"{details}。可提高 QUOTA_SUPPLEMENT_TRY_FACTOR / "
                "QUOTA_SUPPLEMENT_MAX_ROUNDS，或降低 TARGET_TOTAL_SAMPLES。"
            )

    def _rebalance_to_target_distribution(self):
        """按全局目标比例从候选池中定额选择最终训练样本。"""
        if not self.meta_index:
            raise RuntimeError("目标比例重平衡失败：候选样本为空")

        # 去重；同一时空像元同时来自 random/station 时优先保留 station 标记。
        source_priority = {'station': 3, 'quota_supplement': 2, 'random': 1}
        unique = {}
        for item in self.meta_index:
            key = (item[0], int(item[1]), int(item[2]))
            old = unique.get(key)
            if old is None:
                unique[key] = item
            else:
                old_source = old[3] if len(old) > 3 else 'random'
                new_source = item[3] if len(item) > 3 else 'random'
                if source_priority.get(new_source, 0) > source_priority.get(old_source, 0):
                    unique[key] = item

        groups = {name: [] for name in self._target_bin_names}
        for item in unique.values():
            date_dt, r, c = item[:3]
            if date_dt not in self.label_data:
                continue
            label_arr, label_nodata = self.label_data[date_dt]
            y = float(label_arr[r, c])
            if (label_nodata is not None and y == label_nodata) or not np.isfinite(y) or y < 0:
                continue
            name = self._target_bin_name(y)
            if name in groups:
                groups[name].append(item)

        available_total = sum(len(v) for v in groups.values())
        target_total = min(self.target_total_samples, available_total)
        quotas = self._allocate_target_counts(target_total)

        print("\n" + "=" * 72)
        print("🎯 全局目标比例定额采样")
        print(f"   候选去重后: {available_total:,}")
        print(f"   最终目标总数: {target_total:,}")
        print("=" * 72)

        selected = []
        leftovers = {name: [] for name in self._target_bin_names}

        for name in self._target_bin_names:
            items = groups[name]
            station_items = [x for x in items if len(x) > 3 and x[3] == 'station']
            other_items = [x for x in items if not (len(x) > 3 and x[3] == 'station')]
            np.random.shuffle(station_items)
            np.random.shuffle(other_items)
            ordered = station_items + other_items

            take = min(quotas[name], len(ordered))
            selected.extend(ordered[:take])
            leftovers[name] = ordered[take:]
            shortage = quotas[name] - take
            status = "OK" if shortage == 0 else f"缺 {shortage:,}"
            print(
                f"   {name:>10s}: 目标={quotas[name]:>7,}, "
                f"可用={len(items):>7,}, 选中={take:>7,}  [{status}]"
            )

        # [DANGER] 严格 quota 模式下禁止用低值剩余样本回填高值短缺。
        # 否则虽然总数仍为16万，但 33% / 33% / 34% 会再次被破坏。
        missing_total = target_total - len(selected)
        if missing_total > 0:
            shortages = {
                name: max(0, quotas[name] - len(groups[name]))
                for name in self._target_bin_names
            }
            details = ", ".join(
                f"{name}缺{n}" for name, n in shortages.items() if n > 0
            )
            if self.strict_target_quota:
                raise RuntimeError(
                    f"严格目标比例重平衡失败，仍短缺 {missing_total:,} 个样本：{details}"
                )

            print(f"\n   ⚠ 非严格模式：短缺 {missing_total:,}，从非零剩余候选中补齐总数")
            fill_priority = [
                '30_50', '50_80', '20_30', '10_20', '80_120',
                '5_10', '1_5', '120_200', '0_1', '200_500', '500_plus'
            ]
            for name in fill_priority:
                if missing_total <= 0:
                    break
                pool = leftovers[name]
                if not pool:
                    continue
                n_take = min(missing_total, len(pool))
                selected.extend(pool[:n_take])
                missing_total -= n_take

        self.meta_index = selected

        actual = {name: 0 for name in self._target_bin_names}
        station_count = 0
        for item in self.meta_index:
            date_dt, r, c = item[:3]
            y = float(self.label_data[date_dt][0][r, c])
            actual[self._target_bin_name(y)] += 1
            if len(item) > 3 and item[3] == 'station':
                station_count += 1

        final_total = len(self.meta_index)
        print("\n   最终实际分布:")
        for name in self._target_bin_names:
            ratio = actual[name] / max(final_total, 1) * 100
            print(
                f"   {name:>10s}: {actual[name]:>7,} "
                f"({ratio:>6.2f}%), 目标={self.target_swe_ratios[name]*100:>5.1f}%"
            )
        print(f"   station 优先保留数: {station_count:,}")
        print(f"   最终样本总数: {final_total:,}")
        print("=" * 72)

    def _get_swe_bin(self, swe_val):
        """根据 SWE 值返回区间名称"""
        bins = self.adaptive_swe_bins
        for i in range(len(bins) - 1):
            if bins[i] <= swe_val < bins[i+1]:
                return f"{bins[i]}-{bins[i+1]}"
        return f"{bins[-1]}+"

    def _build_station_guided_samples(self):
        """根据参数选择站点位置全日期或站点实际记录采样。"""
        if self.station_sampling_unit == "records":
            return self._build_station_record_guided_samples()

        # positions_all_dates：旧逻辑，站点位置遍历全部标签日期。
        # 记录随机采样中已选中的像元，避免重复
        existing_pixels = set()
        for item in self.meta_index:
            if len(item) == 4:
                date_dt, r, c, _ = item
            else:
                date_dt, r, c = item
            existing_pixels.add((date_dt, r, c))

        station_samples_added = 0
        samples_per_date = {}

        # 🔥 详细特征统计
        feature_stats = {
            # 失败统计
            'conv_failed': 0,
            'point_failed': 0,
            'label_failed': 0,
            'total_attempted': 0,

            # 🔥 卷积特征统计
            'conv_means': [],
            'conv_stds': [],
            'conv_min': None,
            'conv_max': None,

            # 🔥 特征值分布统计（仅统计成功的样本）
            's1_vv_values': [],
            's1_vh_values': [],
            's1_vv_cov_values': [],
            's1_vh_cov_values': [],
            's1_angle_values': [],
            'smap_tbv_values': [],
            'smap_tbh_values': [],
            'smap_mask_v_values': [],
            'smap_mask_h_values': [],
            'ls_bands': [[] for _ in range(6)],

            # 🔥 标记统计
            'has_real_s1': 0,      # 有真实哨兵1
            'has_real_smap': 0,    # 有真实SMAP
            'has_real_microwave': 0,  # 有任一真实微波
            'all_default_microwave': 0,  # 全是默认值
        }

        for date_dt, (label_arr, label_nodata) in self.label_data.items():
            if date_dt not in self.date_to_index:
                continue

            if label_nodata is not None:
                valid_mask = (label_arr != label_nodata) & np.isfinite(label_arr)
            else:
                valid_mask = np.isfinite(label_arr)

            candidates = []
            for r, c in self.station_pixels:
                if not (0 <= r < self.H and 0 <= c < self.W):
                    continue
                if not valid_mask[r, c]:
                    continue
                if self.station_filter_zero_target and float(label_arr[r, c]) <= 1e-6:
                    continue
                if (date_dt, r, c) in existing_pixels:
                    continue
                candidates.append((r, c))

            if not candidates:
                continue

            np.random.shuffle(candidates)
            if self.station_samples_per_day <= 0:
                n_take = len(candidates)
            else:
                n_take = min(self.station_samples_per_day, len(candidates))
            selected = candidates[:n_take]

            added_today = 0
            for r, c in selected:
                feature_stats['total_attempted'] += 1

                # 🔥 修改：接收4个返回值
                is_valid, fail_reason, conv_patch, point_feats = self._validate_station_sample_with_reason(date_dt, r, c)

                if is_valid and point_feats is not None and conv_patch is not None:
                    # ========== 站点引导阶段：先过滤 target=0，再统计特征 ==========
                    swe = float(label_arr[r, c])
                    if self.station_filter_zero_target and swe <= 1e-6:
                        feature_stats['label_failed'] += 1
                        continue

                    # ========== 收集卷积特征统计 ==========
                    if conv_patch is not None:
                        feature_stats['conv_means'].append(float(np.mean(conv_patch)))
                        feature_stats['conv_stds'].append(float(np.std(conv_patch)))
                        if feature_stats['conv_min'] is None or np.min(conv_patch) < feature_stats['conv_min']:
                            feature_stats['conv_min'] = float(np.min(conv_patch))
                        if feature_stats['conv_max'] is None or np.max(conv_patch) > feature_stats['conv_max']:
                            feature_stats['conv_max'] = float(np.max(conv_patch))

                    # ========== 收集点特征统计 ==========
                    # LS波段
                    for band in range(6):
                        feature_stats['ls_bands'][band].append(float(point_feats[band]))

                    # 哨兵1
                    feature_stats['s1_vv_values'].append(float(point_feats[6]))
                    feature_stats['s1_vh_values'].append(float(point_feats[7]))
                    feature_stats['s1_vv_cov_values'].append(float(point_feats[8]))
                    feature_stats['s1_vh_cov_values'].append(float(point_feats[9]))
                    feature_stats['s1_angle_values'].append(float(point_feats[10]))

                    # SMAP
                    feature_stats['smap_tbv_values'].append(float(point_feats[11]))
                    feature_stats['smap_tbh_values'].append(float(point_feats[12]))
                    feature_stats['smap_mask_v_values'].append(float(point_feats[13]))
                    feature_stats['smap_mask_h_values'].append(float(point_feats[14]))

                    # 降水累积（注意索引位置）
                    # point_feats 索引: 0-5 LS, 6-10 S1, 11-12 SMAP_TB, 13-14 SMAP_mask,
                    # 15-16 经纬度, 17 DOY
                    # 总降水和降雪已从点特征中移除

                    # ========== 🔥 修正后的微波真实性检查（与 _build_point_features_station 一致） ==========
                    # 哨兵1有效：值不是0（因为无数据时填充0）
                    s1_vv_valid = (point_feats[6] != 0.0)
                    s1_vh_valid = (point_feats[7] != 0.0)
                    has_real_s1 = s1_vv_valid or s1_vh_valid

                    # SMAP有效：mask == 1（表示有真实数据）
                    smap_v_valid = (point_feats[13] == 1.0)  # mask_V
                    smap_h_valid = (point_feats[14] == 1.0)  # mask_H
                    has_real_smap = smap_v_valid or smap_h_valid

                    if has_real_s1:
                        feature_stats['has_real_s1'] += 1
                    if has_real_smap:
                        feature_stats['has_real_smap'] += 1
                    if has_real_s1 or has_real_smap:
                        feature_stats['has_real_microwave'] += 1
                    else:
                        feature_stats['all_default_microwave'] += 1

                    # 添加样本
                    self.meta_index.append((date_dt, r, c, 'station'))
                    existing_pixels.add((date_dt, r, c))
                    added_today += 1
                    station_samples_added += 1
                else:
                    if fail_reason == 'conv':
                        feature_stats['conv_failed'] += 1
                    elif fail_reason == 'point':
                        feature_stats['point_failed'] += 1
                    elif fail_reason == 'label':
                        feature_stats['label_failed'] += 1

            if added_today > 0:
                samples_per_date[date_dt] = added_today

            if station_samples_added % 5000 == 0 and station_samples_added > 0:
                print(f"    已添加站点样本: {station_samples_added}")

        # ========== 打印详细统计 ==========
        print(f"\n{'='*70}")
        print(f"📊 站点引导采样 - 完整特征统计")
        print(f"{'='*70}")

        print(f"\n【验证失败统计】")
        print(f"    总尝试: {feature_stats['total_attempted']}")
        print(f"    卷积失败: {feature_stats['conv_failed']}")
        print(f"    点特征失败: {feature_stats['point_failed']}")
        print(f"    标签失败: {feature_stats['label_failed']}")
        print(f"    成功添加: {station_samples_added}")
        if feature_stats['total_attempted'] > 0:
            success_rate = station_samples_added / feature_stats['total_attempted'] * 100
            print(f"    成功率: {success_rate:.1f}%")

        # 打印卷积特征统计
        if feature_stats['conv_means']:
            print(f"\n【卷积特征统计】(有效样本数: {len(feature_stats['conv_means'])})")
            print(f"    均值: {np.mean(feature_stats['conv_means']):.4f} ± {np.std(feature_stats['conv_means']):.4f}")
            print(f"    标准差: {np.mean(feature_stats['conv_stds']):.4f} ± {np.std(feature_stats['conv_stds']):.4f}")
            print(f"    全局范围: [{feature_stats['conv_min']:.4f}, {feature_stats['conv_max']:.4f}]")

        print(f"\n【微波特征真实性】")
        denom = max(station_samples_added, 1)
        print(f"    有真实哨兵1的样本: {feature_stats['has_real_s1']} ({feature_stats['has_real_s1']/denom*100:.1f}%)")
        print(f"    有真实SMAP的样本: {feature_stats['has_real_smap']} ({feature_stats['has_real_smap']/denom*100:.1f}%)")
        print(f"    有任一真实微波的样本: {feature_stats['has_real_microwave']} ({feature_stats['has_real_microwave']/denom*100:.1f}%)")
        print(f"    全是默认值的样本: {feature_stats['all_default_microwave']} ({feature_stats['all_default_microwave']/denom*100:.1f}%)")

        if feature_stats['s1_vv_values']:
            print(f"\n【哨兵1特征分布】(有效样本数: {len(feature_stats['s1_vv_values'])})")
            print(f"    S1_VV: 均值={np.mean(feature_stats['s1_vv_values']):.2f}, 标准差={np.std(feature_stats['s1_vv_values']):.2f}, "
                  f"范围=[{np.min(feature_stats['s1_vv_values']):.2f}, {np.max(feature_stats['s1_vv_values']):.2f}]")
            print(f"    S1_VH: 均值={np.mean(feature_stats['s1_vh_values']):.2f}, 标准差={np.std(feature_stats['s1_vh_values']):.2f}, "
                  f"范围=[{np.min(feature_stats['s1_vh_values']):.2f}, {np.max(feature_stats['s1_vh_values']):.2f}]")
            print(f"    S1_angle: 均值={np.mean(feature_stats['s1_angle_values']):.2f}, 标准差={np.std(feature_stats['s1_angle_values']):.2f}")

        if feature_stats['smap_tbv_values']:
            print(f"\n【SMAP特征分布】(有效样本数: {len(feature_stats['smap_tbv_values'])})")
            print(f"    TBV: 均值={np.mean(feature_stats['smap_tbv_values']):.2f}, 标准差={np.std(feature_stats['smap_tbv_values']):.2f}, "
                  f"范围=[{np.min(feature_stats['smap_tbv_values']):.2f}, {np.max(feature_stats['smap_tbv_values']):.2f}]")
            print(f"    TBH: 均值={np.mean(feature_stats['smap_tbh_values']):.2f}, 标准差={np.std(feature_stats['smap_tbh_values']):.2f}, "
                  f"范围=[{np.min(feature_stats['smap_tbh_values']):.2f}, {np.max(feature_stats['smap_tbh_values']):.2f}]")
            print(f"    mask_V: 均值={np.mean(feature_stats['smap_mask_v_values']):.4f}, 标准差={np.std(feature_stats['smap_mask_v_values']):.4f}")
            print(f"    mask_H: 均值={np.mean(feature_stats['smap_mask_h_values']):.4f}, 标准差={np.std(feature_stats['smap_mask_h_values']):.4f}")

        if feature_stats['ls_bands'][0]:
            print(f"\n【LS特征分布】(有效样本数: {len(feature_stats['ls_bands'][0])})")
            for band in range(6):
                vals = feature_stats['ls_bands'][band]
                print(f"    LS{band+1}: 均值={np.mean(vals):.4f}, 标准差={np.std(vals):.4f}, "
                      f"范围=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

        print(f"\n{'='*70}")

        # 保存统计
        self.station_sample_stats = {
            'added_count': station_samples_added,
            'samples_per_date': samples_per_date,
            'feature_stats': feature_stats
        }

        print(f"\n  站点引导采样完成:")
        print(f"    新增样本数: {station_samples_added}")
        print(f"    涉及日期数: {len(samples_per_date)}")
        


    def _adaptive_supplement(self):
        """自适应修正：分析当前样本的SWE分布，动态补充短缺区间，并记录高值样本位置"""
        print(f"\n📊 自适应修正: 分析当前分布短板...")

        if len(self.meta_index) == 0:
            print("   无样本，跳过修正")
            return

        swe_bins = self.adaptive_swe_bins
        n_bins = len(swe_bins) - 1

        # 1. 统计当前每个区间的样本数
        current_counts = [0] * n_bins
        for item in self.meta_index:
            date_dt, r, c = item[:3]
            if date_dt in self.label_data:
                label_arr, _ = self.label_data[date_dt]
                swe_val = label_arr[r, c]
                bin_idx = np.digitize(swe_val, swe_bins) - 1
                bin_idx = max(0, min(bin_idx, n_bins - 1))
                current_counts[bin_idx] += 1

        total_current = sum(current_counts)
        current_ratios = [c / total_current for c in current_counts]

        # 2. 计算目标分布
        natural_weights = self._get_natural_weights(swe_bins)
        target_weights = self._get_target_weights(swe_bins)

        final_target = (1 - self.adaptive_alpha) * np.array(natural_weights) + self.adaptive_alpha * np.array(target_weights)
        final_target = final_target / final_target.sum()

        # 3. 找出短缺区间（重点关注高值区间）
        shortages = []
        for i in range(n_bins):
            if i == 0:  # 跳过 0-5mm
                continue
            if current_counts[i] == 0:
                shortage_ratio = float('inf')
            else:
                shortage_ratio = final_target[i] / (current_ratios[i] + 1e-8)

            if shortage_ratio > self.adaptive_threshold:
                shortages.append((i, shortage_ratio, final_target[i], current_ratios[i]))

        if not shortages:
            print("   分布已合理，无需补充")
            return

        shortages.sort(key=lambda x: -x[1])

        print(f"   发现 {len(shortages)} 个短缺区间:")
        for i, ratio, target, current in shortages[:5]:
            print(f"     区间 [{swe_bins[i]}, {swe_bins[i+1]}): 当前={current*100:.1f}%, 目标={target*100:.1f}%, 短缺比={ratio:.2f}")

        # 4. 构建候选池
        candidates_by_bin = self._build_candidate_pool(swe_bins)

        # 🔥 记录高值样本的位置信息
        high_value_samples = []

        # 5. 补充样本
        supplemented = 0
        max_supplement = total_current * 0.3

        for bin_idx, ratio, target_ratio, current_ratio in shortages:
            if supplemented >= max_supplement:
                break

            target_count = int(target_ratio * total_current)
            current_count = current_counts[bin_idx]
            n_needed = min(target_count - current_count, max_supplement - supplemented)
            n_needed = max(1, n_needed)

            candidates = candidates_by_bin[bin_idx]
            if len(candidates) == 0:
                print(f"     区间 [{swe_bins[bin_idx]}, {swe_bins[bin_idx+1]}): 无候选点，跳过")
                continue

            available = []
            for c in candidates:
                if c not in self.meta_index:
                    available.append(c)

            if len(available) == 0:
                continue

            n_take = min(n_needed, len(available))
            import random
            new_samples = random.sample(available, n_take)

            # 🔥 记录每个补充的样本信息
            for item in new_samples:
                date_dt, r, c = item[:3]
                if date_dt in self.label_data:
                    label_arr, _ = self.label_data[date_dt]
                    swe_val = label_arr[r, c]

                    # 获取经纬度
                    lon, lat = self._pixel_to_lonlat(r, c)

                    high_value_samples.append({
                        'bin': self._get_swe_bin(swe_val),  # 🔥 使用辅助方法
                        'date': date_dt.strftime('%Y-%m-%d'),
                        'row': r,
                        'col': c,
                        'swe': float(swe_val),
                        'longitude': lon,
                        'latitude': lat,
                        'source': 'adaptive_supplement'
                    })

                self.meta_index.append(item)

            supplemented += n_take
            print(f"     区间 [{swe_bins[bin_idx]}, {swe_bins[bin_idx+1]}): 补充 {n_take} 个样本")

        print(f"\n   ✅ 自适应修正完成: 新增 {supplemented} 个样本")
        print(f"      总样本数: {len(self.meta_index):,}")

        # 🔥 保存高值样本位置信息到 CSV
        if high_value_samples:
            import pandas as pd
            df_high = pd.DataFrame(high_value_samples)

            # 保存到文件
            if hasattr(self, 'cache_dir') and self.cache_dir:
                save_path = Path(self.cache_dir) / "high_value_samples_locations.csv"
            else:
                save_path = Path("/root/autodl-tmp") / "high_value_samples_locations.csv"

            # 如果文件已存在，追加而不是覆盖
            if save_path.exists():
                df_existing = pd.read_csv(save_path)
                df_combined = pd.concat([df_existing, df_high], ignore_index=True)
                df_combined.to_csv(save_path, index=False, encoding='utf-8')
                print(f"\n   📍 高值样本位置已追加到: {save_path}")
                print(f"      本次新增 {len(high_value_samples)} 个，累计 {len(df_combined)} 个")
            else:
                df_high.to_csv(save_path, index=False, encoding='utf-8')
                print(f"\n   📍 高值样本位置已保存: {save_path}")
                print(f"      共 {len(high_value_samples)} 个高值样本")

            # 按区间统计
            print(f"\n   📊 本次补充的高值样本按区间统计:")
            for bin_name in df_high['bin'].unique():
                count = len(df_high[df_high['bin'] == bin_name])
                print(f"      {bin_name}: {count} 个样本")

        
    def _collect_sample_swe_raw_mm(self):
        """从 meta_index 对应的 label_data 直接读取原始 SWE（单位：mm）。

        [CONTRACT]
        - 分布统计只允许读取 label_data[date][row, col]。
        - 禁止使用 __getitem__ 返回的归一化 target。
        - 禁止通过 swe_min/swe_max（尤其旧默认200）反归一化。
        """
        values = []
        invalid = 0

        for item in self.meta_index:
            date_dt, row, col = item[:3]
            if date_dt not in self.label_data:
                invalid += 1
                continue

            label_arr, label_nodata = self.label_data[date_dt]
            swe_mm = float(label_arr[int(row), int(col)])

            if not np.isfinite(swe_mm):
                invalid += 1
                continue
            if label_nodata is not None and swe_mm == float(label_nodata):
                invalid += 1
                continue
            if swe_mm < 0:
                invalid += 1
                continue
            if FILTER_GLACIER_SWE_ARTIFACTS and swe_mm >= GLACIER_SWE_THRESHOLD_MM:
                invalid += 1
                continue

            values.append(swe_mm)

        values = np.asarray(values, dtype=np.float64)
        return values, invalid

    def _count_original_target_bins_raw_mm(self):
        """逐日累计原始 ERA5-Land 标签在目标分箱中的数量（原始mm）。"""
        bin_names = self._target_bin_names
        counts = np.zeros(len(bin_names), dtype=np.int64)
        total = 0

        for date_dt, (label_arr, label_nodata) in self.label_data.items():
            if date_dt not in self.date_to_index:
                continue

            valid = np.isfinite(label_arr) & (label_arr >= 0)
            if label_nodata is not None:
                valid &= (label_arr != label_nodata)
            if FILTER_GLACIER_SWE_ARTIFACTS:
                valid &= (label_arr < GLACIER_SWE_THRESHOLD_MM)

            vals_mm = label_arr[valid].astype(np.float64, copy=False)
            if vals_mm.size == 0:
                continue

            idx = self._target_bin_index_array(vals_mm)
            idx = idx[idx >= 0]
            counts += np.bincount(idx, minlength=len(bin_names))[:len(bin_names)]
            total += int(idx.size)

        return counts, total

    def _analyze_swe_distribution(self):
        """使用 label_data 原始毫米值分析并绘制最终采样分布。

        [CONTRACT]
        - 所有 bin、均值、分位数、倍率都使用原始 SWE mm。
        - 输出文件名带 raw_mm，避免与历史错误图混淆。

        [COMPAT]
        - 同时覆盖旧文件名 swe_distribution_after_adaptive.png、
          swe_sampling_amplification.png/csv；这些旧名称中的内容也会是正确mm尺度。
        """
        print(f"\n{'='*72}")
        print("📊 最终样本 SWE 分布分析（RAW MM；不经过归一化/反归一化）")
        print(f"{'='*72}")

        self.setup_chinese_fonts()

        sample_mm, invalid_count = self._collect_sample_swe_raw_mm()
        if sample_mm.size == 0:
            raise RuntimeError("无法从 meta_index/label_data 读取任何有效原始 SWE mm")

        if invalid_count > 0:
            print(f"⚠ 无效或被过滤样本: {invalid_count:,}")

        # [DANGER] 目标比例模式下，统计样本数应与最终 meta_index 一致。
        if self.use_target_quota_sampling and sample_mm.size != len(self.meta_index):
            raise RuntimeError(
                f"原始mm统计样本数({sample_mm.size:,})与meta_index({len(self.meta_index):,})不一致"
            )

        print("\n[PLOT DISTRIBUTION CHECK - RAW MM]")
        print(f"  样本总数: {sample_mm.size:,}")
        print(f"  SWE范围: [{sample_mm.min():.4f}, {sample_mm.max():.4f}] mm")
        print(f"  均值: {sample_mm.mean():.4f} mm")
        print(f"  标准差: {sample_mm.std():.4f} mm")
        print(f"  中位数: {np.median(sample_mm):.4f} mm")

        print("\n📊 分位数统计（RAW MM）:")
        for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
            print(f"  P{p:g}: {np.percentile(sample_mm, p):.4f} mm")

        bin_names = self._target_bin_names
        bin_labels = [
            "=0", "0–1", "1–5", "5–10", "10–20", "20–30",
            "30–50", "50–80", "80–120", "120–200", "200–500", ">500"
        ]

        sample_idx = self._target_bin_index_array(sample_mm)
        sample_idx = sample_idx[sample_idx >= 0]
        sample_counts = np.bincount(
            sample_idx, minlength=len(bin_names)
        )[:len(bin_names)]
        sample_total = int(sample_counts.sum())
        sample_ratio = sample_counts / max(sample_total, 1)

        raw_counts, raw_total = self._count_original_target_bins_raw_mm()
        raw_ratio = raw_counts / max(raw_total, 1)

        target_ratio = np.asarray(
            [self.target_swe_ratios[name] for name in bin_names],
            dtype=np.float64,
        )

        print("\n📋 精细分箱（RAW MM）:")
        print(f"{'区间':<12}{'原始占比':>13}{'采样数':>13}{'采样占比':>13}{'目标占比':>13}")
        print("-" * 65)
        for label, r0, count, r1, rt in zip(
            bin_labels, raw_ratio, sample_counts, sample_ratio, target_ratio
        ):
            print(
                f"{label:<12}{r0*100:>12.4f}%{int(count):>13,}"
                f"{r1*100:>12.4f}%{rt*100:>12.4f}%"
            )

        # 三大区间：≤5 / 5–30 / >30
        grouped_labels = ["≤5 mm", "5–30 mm", ">30 mm"]
        grouped_sample_counts = np.asarray([
            sample_counts[0:3].sum(),
            sample_counts[3:6].sum(),
            sample_counts[6:].sum(),
        ], dtype=np.int64)
        grouped_raw_counts = np.asarray([
            raw_counts[0:3].sum(),
            raw_counts[3:6].sum(),
            raw_counts[6:].sum(),
        ], dtype=np.int64)
        grouped_sample_ratio = grouped_sample_counts / max(sample_total, 1)
        grouped_raw_ratio = grouped_raw_counts / max(raw_total, 1)
        grouped_target_ratio = np.asarray([0.33, 0.33, 0.34], dtype=np.float64)

        print("\n🎯 三大区间检查（RAW MM）:")
        for label, count, ratio, target in zip(
            grouped_labels,
            grouped_sample_counts,
            grouped_sample_ratio,
            grouped_target_ratio,
        ):
            print(
                f"  {label:<10}: {int(count):>7,} "
                f"({ratio*100:6.2f}%), 目标={target*100:5.1f}%"
            )

        if self.use_target_quota_sampling and self.strict_target_quota:
            max_group_error = float(np.max(np.abs(grouped_sample_ratio - grouped_target_ratio)))
            if max_group_error > 0.005:
                raise RuntimeError(
                    "严格quota模式下三大区间偏离目标超过0.5个百分点："
                    f"实际={grouped_sample_ratio.tolist()}"
                )

        amplification = np.full(len(bin_names), np.nan, dtype=np.float64)
        valid_raw = raw_ratio > 0
        amplification[valid_raw] = sample_ratio[valid_raw] / raw_ratio[valid_raw]

        out_dir = Path(self.cache_dir) if getattr(self, "cache_dir", None) else Path("/root/autodl-tmp")
        out_dir.mkdir(parents=True, exist_ok=True)

        distribution_csv = out_dir / "swe_distribution_raw_mm.csv"
        amplification_csv = out_dir / "swe_sampling_amplification_raw_mm.csv"
        distribution_plot = out_dir / "swe_distribution_raw_mm.png"
        amplification_plot = out_dir / "swe_sampling_amplification_raw_mm.png"

        distribution_df = pd.DataFrame({
            "bin": bin_names,
            "label": bin_labels,
            "original_count": raw_counts,
            "original_ratio": raw_ratio,
            "sampled_count": sample_counts,
            "sampled_ratio": sample_ratio,
            "target_ratio": target_ratio,
        })
        distribution_df.to_csv(distribution_csv, index=False, encoding="utf-8-sig")

        amplification_df = distribution_df.copy()
        amplification_df["amplification"] = amplification
        amplification_df.to_csv(amplification_csv, index=False, encoding="utf-8-sig")

        # 图1：精细分箱 + 三大区间，均为原始毫米值。
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        x = np.arange(len(bin_labels))
        width = 0.36
        axes[0].bar(
            x - width / 2,
            raw_ratio * 100,
            width=width,
            label=f"Original ERA5-Land (n={raw_total:,})",
            edgecolor="black",
            alpha=0.80,
        )
        axes[0].bar(
            x + width / 2,
            sample_ratio * 100,
            width=width,
            label=f"Sampled dataset (n={sample_total:,})",
            edgecolor="black",
            alpha=0.85,
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(bin_labels, rotation=35, ha="right")
        axes[0].set_xlabel("SWE bin (raw mm)")
        axes[0].set_ylabel("Proportion (%)")
        axes[0].set_title("Original vs sampled SWE distribution — raw mm")
        axes[0].grid(True, axis="y", alpha=0.25)
        axes[0].legend(fontsize=9)

        gx = np.arange(3)
        gwidth = 0.25
        axes[1].bar(
            gx - gwidth,
            grouped_raw_ratio * 100,
            width=gwidth,
            label="Original",
            edgecolor="black",
            alpha=0.80,
        )
        axes[1].bar(
            gx,
            grouped_sample_ratio * 100,
            width=gwidth,
            label="Sampled",
            edgecolor="black",
            alpha=0.85,
        )
        axes[1].bar(
            gx + gwidth,
            grouped_target_ratio * 100,
            width=gwidth,
            label="Target",
            edgecolor="black",
            alpha=0.65,
        )
        axes[1].set_xticks(gx)
        axes[1].set_xticklabels(grouped_labels)
        axes[1].set_xlabel("Grouped SWE range (raw mm)")
        axes[1].set_ylabel("Proportion (%)")
        axes[1].set_title("Grouped quota check — raw mm")
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend(fontsize=9)

        for ax in axes:
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)

        plt.tight_layout()
        plt.savefig(distribution_plot, dpi=220, bbox_inches="tight")

        # [COMPAT] 覆盖旧文件名，避免旧流程找不到；内容已经是正确原始mm。
        legacy_distribution_plot = out_dir / "swe_distribution_after_adaptive.png"
        plt.savefig(legacy_distribution_plot, dpi=220, bbox_inches="tight")
        plt.close(fig)

        # 图2：sampled/original倍率，仍使用原始mm分箱。
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(x, amplification, edgecolor="black", alpha=0.85)
        ax.set_yscale("log")
        ax.axhline(1.0, linestyle="--", linewidth=1.2, label="1× unchanged")
        ax.axhline(2.0, linestyle=":", linewidth=1.0, label="2×")
        ax.axhline(5.0, linestyle="-.", linewidth=1.0, label="5×")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=35, ha="right")
        ax.set_xlabel("SWE bin (raw mm)")
        ax.set_ylabel("Sampled proportion / original proportion")
        ax.set_title("Sampling amplification by SWE bin — raw mm")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend()

        for bar, amp in zip(bars, amplification):
            if np.isfinite(amp) and amp > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    amp,
                    f"{amp:.2f}×",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=90,
                )

        plt.tight_layout()
        plt.savefig(amplification_plot, dpi=220, bbox_inches="tight")

        legacy_amp_plot = out_dir / "swe_sampling_amplification.png"
        plt.savefig(legacy_amp_plot, dpi=220, bbox_inches="tight")
        plt.close(fig)

        # [COMPAT] 旧CSV名同步为正确raw-mm结果。
        legacy_amp_csv = out_dir / "swe_sampling_amplification.csv"
        amplification_df.to_csv(legacy_amp_csv, index=False, encoding="utf-8-sig")

        summary_json = out_dir / "swe_distribution_raw_mm_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({
                "unit": "mm",
                "source": "label_data raw raster values",
                "sample_count": int(sample_total),
                "original_count": int(raw_total),
                "sample_min_mm": float(sample_mm.min()),
                "sample_max_mm": float(sample_mm.max()),
                "sample_mean_mm": float(sample_mm.mean()),
                "sample_median_mm": float(np.median(sample_mm)),
                "grouped_labels": grouped_labels,
                "grouped_sample_counts": grouped_sample_counts.tolist(),
                "grouped_sample_ratios": grouped_sample_ratio.tolist(),
                "grouped_target_ratios": grouped_target_ratio.tolist(),
            }, f, ensure_ascii=False, indent=2)

        print("\n✅ RAW-MM 分布结果已保存:")
        print(f"  分布图: {distribution_plot}")
        print(f"  分布表: {distribution_csv}")
        print(f"  倍率图: {amplification_plot}")
        print(f"  倍率表: {amplification_csv}")
        print(f"  摘要:   {summary_json}")
        print(f"  兼容旧图名: {legacy_distribution_plot}")
        print(f"{'='*72}\n")


    def _get_natural_weights(self, swe_bins):
        """获取候选池的自然分布权重"""
        n_bins = len(swe_bins) - 1
        counts = [0] * n_bins

        for date_dt, (label_arr, _) in self.label_data.items():
            if date_dt not in self.date_to_index:
                continue
            valid_mask = np.isfinite(label_arr) & (label_arr != -9999.0) & (label_arr >= 0)
            rows, cols = np.where(valid_mask)
            step = max(1, len(rows) // 5000)
            for r, c in zip(rows[::step], cols[::step]):
                swe_val = label_arr[r, c]
                bin_idx = np.digitize(swe_val, swe_bins) - 1
                bin_idx = max(0, min(bin_idx, n_bins - 1))
                counts[bin_idx] += 1

        total = sum(counts)
        if total == 0:
            return [1/n_bins] * n_bins
        return [c / total for c in counts]
    
    def _get_target_weights(self, swe_bins):
        """获取目标分布 - 更平缓，避免低值权重过高"""
        n_bins = len(swe_bins) - 1
        centers = [(swe_bins[i] + swe_bins[i+1]) / 2 for i in range(n_bins)]

        # 🔥 使用平方根倒数，而不是直接倒数
        # 这样低值的权重不会太大，高值的权重不会太小
        weights = [1.0 / np.sqrt(c + 10) for c in centers]

        total = sum(weights)
        return [w / total for w in weights]

    def _build_candidate_pool(self, swe_bins):
        """构建候选池（按SWE区间分组）"""
        n_bins = len(swe_bins) - 1
        candidates_by_bin = [[] for _ in range(n_bins)]

        for date_dt, (label_arr, label_nodata) in self.label_data.items():
            if date_dt not in self.date_to_index:
                continue

            valid_mask = (label_arr != label_nodata) & np.isfinite(label_arr)
            rows, cols = np.where(valid_mask)

            for r, c in zip(rows, cols):
                if r - self.R < 0 or r + self.R >= self.H or c - self.R < 0 or c + self.R >= self.W:
                    continue

                swe_val = label_arr[r, c]
                bin_idx = np.digitize(swe_val, swe_bins) - 1
                bin_idx = max(0, min(bin_idx, n_bins - 1))

                candidates_by_bin[bin_idx].append((date_dt, r, c))

        return candidates_by_bin
    
        
    def _build_spatial_features_station(self, date_dt: datetime, r: int, c: int) -> np.ndarray:
        """
        宽松版卷积特征 - 添加无效值检查
        无效值（-9999）先转为 NaN，再进行插值
        """

        date_idx = self.date_to_index.get(date_dt)
        if date_idx is None:
            return None

        r0 = max(0, r - self.R)
        r1 = min(self.H, r + self.R + 1)
        c0 = max(0, c - self.R)
        c1 = min(self.W, c + self.R + 1)

        actual_h = r1 - r0
        actual_w = c1 - c0

        if actual_h < 3 or actual_w < 3:
            return None

        # 🔥 定义各卷积变量的无效值
        INVALID_VALUES = {
            "chelsa_sfxwind": -9999.0,
            "lst": -9999.0,
            "rh": -9999.0,
            "pr": -9999.0,
        }
        CLAMDAY_INVALID = -9999.0

        conv_features = []

        # 1. 动态卷积特征（宽松：无效值先转NaN再插值）
        for var in CONV_VARS:
            var_arr = self.conv_dyn_data[var]
            if date_idx >= var_arr.shape[0]:
                patch = var_arr[-1, r0:r1, c0:c1]
            else:
                patch = var_arr[date_idx, r0:r1, c0:c1]

            if patch.shape != (self.P, self.P):
                patch = self._resize_to_standard(patch, self.P, self.P)

            invalid_val = INVALID_VALUES.get(var)
            if invalid_val is not None:
                # 🔥 将无效值转为 NaN
                patch = np.where(patch == invalid_val, np.nan, patch)

            # 检查是否有 NaN
            if np.any(np.isnan(patch)):
                # 如果全部是 NaN，使用全局有效均值填充（避免插值失败）
                if np.all(np.isnan(patch)):
                    # 使用该变量的全局有效均值
                    global_valid = var_arr[np.isfinite(var_arr) & (var_arr != invalid_val)]
                    if len(global_valid) > 0:
                        global_mean = np.mean(global_valid)
                        patch = np.full_like(patch, global_mean)
                    else:
                        patch = np.zeros_like(patch)
                else:
                    # 有部分有效值，进行插值
                    patch = self._interpolate_nan_patch(patch)

            conv_features.append(patch)

        # 2. clamday
        clamday_patch = self.clamday_data[r0:r1, c0:c1]
        clamday_patch = self._resize_to_standard(clamday_patch, self.P, self.P)

        # 🔥 Clamday 无效值 -9999 转为 NaN
        clamday_patch = np.where(clamday_patch == CLAMDAY_INVALID, np.nan, clamday_patch)
        if np.any(np.isnan(clamday_patch)):
            clamday_patch = self._interpolate_nan_patch(clamday_patch)
        conv_features.append(clamday_patch)

        # 3. DEM 波段（DEM 可能也有无效值）
        for dem_band in self.dem_data:
            dem_patch = dem_band[r0:r1, c0:c1]
            dem_patch = self._resize_to_standard(dem_patch, self.P, self.P)

            # 🔥 DEM 无效值可能是 -9999 或 NaN
            dem_patch = np.where(dem_patch == -9999, np.nan, dem_patch)
            if np.any(np.isnan(dem_patch)):
                dem_patch = self._interpolate_nan_patch(dem_patch)
            conv_features.append(dem_patch)

        try:
            conv_patch = np.stack(conv_features, axis=0)
            # 最终检查：如果还有 NaN，填 0
            if np.any(np.isnan(conv_patch)):
                conv_patch = np.nan_to_num(conv_patch, nan=0.0)
            return conv_patch
        except Exception as e:
            return None
        

    def _validate_station_sample(self, date_dt: datetime, r: int, c: int) -> bool:
        """宽松验证站点样本"""

        original_fine_tune_mode = self.fine_tune_mode
        self.fine_tune_mode = True

        try:
            # 🔥 使用宽松版卷积特征
            conv_patch = self._build_spatial_features_station(date_dt, r, c)
            if conv_patch is None:
                return False

            # 使用宽松版点特征
            point_feats = self._build_point_features_station(date_dt, r, c)
            if point_feats is None:
                return False

            label_arr, label_nodata = self.label_data[date_dt]
            y = label_arr[r, c]
            if (label_nodata is not None and y == label_nodata) or np.isnan(y):
                return False

            return True
        finally:
            self.fine_tune_mode = original_fine_tune_mode
            


    def _cap_zero_target_samples(self, max_zero_ratio: float = 0.10, eps: float = 1e-6):
        """
        全局控制 target=0 样本比例。
        注意：
        - 这里只压真正的 0 值；
        - 不压 0<SWE<=5 的痕量雪/薄雪样本；
        - 适合“ERA5-Land 有雪误差校准”的预训练目标。
        """
        if not hasattr(self, "meta_index") or not self.meta_index:
            return

        zero_items = []
        nonzero_items = []

        for item in self.meta_index:
            date_dt, r, c = item[:3]

            if date_dt not in self.label_data:
                nonzero_items.append(item)
                continue

            label_arr, label_nodata = self.label_data[date_dt]
            y = label_arr[r, c]

            if (label_nodata is not None and y == label_nodata) or (not np.isfinite(y)):
                continue

            if float(y) <= eps:
                zero_items.append(item)
            else:
                nonzero_items.append(item)

        if len(nonzero_items) == 0:
            print("⚠ 全局0值控制跳过：没有非零样本")
            return

        max_zero_allowed = int(len(nonzero_items) * max_zero_ratio / (1.0 - max_zero_ratio))

        before_total = len(self.meta_index)
        before_zero = len(zero_items)
        before_nonzero = len(nonzero_items)

        if len(zero_items) > max_zero_allowed:
            np.random.shuffle(zero_items)
            zero_items = zero_items[:max_zero_allowed]

        self.meta_index = nonzero_items + zero_items

        after_total = len(self.meta_index)
        after_zero = len(zero_items)
        after_nonzero = len(nonzero_items)

        print("\n🎯 全局 target=0 样本比例控制:")
        print(f"   max_zero_ratio: {max_zero_ratio:.2f}")
        print(f"   调整前: total={before_total:,}, zero={before_zero:,}, nonzero={before_nonzero:,}, zero_ratio={before_zero/max(before_total,1)*100:.2f}%")
        print(f"   调整后: total={after_total:,}, zero={after_zero:,}, nonzero={after_nonzero:,}, zero_ratio={after_zero/max(after_total,1)*100:.2f}%")

    def _print_sample_statistics(self):
        """打印最终样本统计信息；兼容旧权重采样和新目标比例采样。"""
        print(f"\n{'='*60}")
        print("✅ 样本构建完成:")
        print(f"  总样本数: {len(self.meta_index):,}")

        # [COMPAT] 旧权重采样统计字段：non_zero_count / zero_count。
        # 新 quota 采样统计字段：attempted_by_bin / valid_by_bin。
        # 这里不能再直接索引旧字段，否则 quota 模式会 KeyError。
        stats = getattr(self, 'random_sample_stats', None)
        if isinstance(stats, dict):
            print("\n  【随机候选采样】")

            if 'valid_by_bin' in stats:
                valid_by_bin = stats.get('valid_by_bin', {})
                random_valid_total = int(sum(valid_by_bin.values()))
                print(f"    通过特征验证的随机候选: {random_valid_total:,}")
                for name in getattr(self, '_target_bin_names', []):
                    print(f"    {name:>10s}: {int(valid_by_bin.get(name, 0)):,}")

            elif 'non_zero_count' in stats or 'zero_count' in stats:
                non_zero = int(stats.get('non_zero_count', 0))
                zero = int(stats.get('zero_count', 0))
                total = non_zero + zero
                print(f"    样本数: {total:,}")
                print(f"    其中 target>0: {non_zero:,}")
                print(f"    其中 target=0: {zero:,}")
                if total > 0:
                    print(f"    零值比例: {zero / total * 100:.1f}%")
            else:
                print("    统计字段存在，但不是已知格式；跳过旧字段打印")

        # 最终集合中的来源统计，比 station_sample_stats['added_count'] 更准确，
        # 因为 quota 重平衡会丢弃一部分最初加入的 station 候选。
        source_counts = Counter()
        for item in self.meta_index:
            source = item[3] if len(item) > 3 else 'unknown'
            source_counts[source] += 1

        if source_counts:
            print("\n  【最终样本来源】")
            for source, count in sorted(source_counts.items()):
                ratio = count / max(len(self.meta_index), 1) * 100
                print(f"    {source}: {count:,} ({ratio:.2f}%)")

        # 最终 0/非0 比例直接根据当前 meta_index 重新计算，
        # 不依赖候选阶段的 random_sample_stats。
        zero_count = 0
        valid_count = 0
        samples_per_year = defaultdict(int)
        samples_per_date = defaultdict(int)

        for item in self.meta_index:
            date_dt, r, c = item[:3]
            if date_dt not in self.label_data:
                continue
            label_arr, label_nodata = self.label_data[date_dt]
            y = float(label_arr[r, c])
            if (label_nodata is not None and y == label_nodata) or not np.isfinite(y):
                continue

            valid_count += 1
            if y <= 1e-6:
                zero_count += 1
            samples_per_year[date_dt.year] += 1
            samples_per_date[date_dt] += 1

        if valid_count > 0:
            non_zero_count = valid_count - zero_count
            print("\n  【最终目标值概况】")
            print(f"    有效样本: {valid_count:,}")
            print(f"    target>0: {non_zero_count:,}")
            print(f"    target=0: {zero_count:,}")
            print(f"    零值比例: {zero_count / valid_count * 100:.2f}%")

        print(f"{'='*60}")

        if samples_per_year:
            print("\n按年份统计:")
            for year, count in sorted(samples_per_year.items()):
                print(f"  {year}年: {count:,} 个样本")

        if samples_per_date:
            print("\n按日期统计（样本最多的前10天）:")
            sorted_dates = sorted(samples_per_date.items(), key=lambda x: x[1], reverse=True)
            for date, count in sorted_dates[:10]:
                print(f"  {date.strftime('%Y-%m-%d')}: {count:,} 个样本")

    @staticmethod
    def _read_station_guide_table(file_path: Path) -> pd.DataFrame:
        """读取站点引导表，支持 xlsx/xls/csv。"""
        suffix = file_path.suffix.lower()
        if suffix in {'.xlsx', '.xls'}:
            return pd.read_excel(file_path, engine='openpyxl')
        if suffix == '.csv':
            for encoding in ('utf-8', 'gbk', 'latin1'):
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(file_path)
        raise ValueError(f"不支持的站点文件格式: {file_path.suffix}")

    @staticmethod
    def _normalize_station_columns(df: pd.DataFrame) -> pd.DataFrame:
        """统一经纬度、日期和站点ID列名。"""
        aliases = {
            'longtitude': 'longitude', 'longitude': 'longitude',
            'lon': 'longitude', 'lng': 'longitude', 'long': 'longitude',
            '纬度': 'latitude', 'latitude': 'latitude', 'lat': 'latitude',
            '经度': 'longitude',
            'date': 'date', 'datetime': 'date', 'time': 'date',
            'obs_date': 'date', 'observation_date': 'date',
            '日期': 'date', '时间': 'date',
            'station_id': 'station_id', 'station': 'station_id',
            'stationid': 'station_id', 'site_id': 'station_id',
            'id': 'station_id', '站点': 'station_id', '站号': 'station_id',
        }
        return df.rename(
            columns=lambda x: aliases.get(str(x).strip().lower(), str(x).strip())
        )

    @staticmethod
    def _canonical_day(value):
        """把 datetime/date/Timestamp 统一为无时区的午夜 datetime。"""
        if value is None or pd.isna(value):
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.normalize().to_pydatetime()

    def _canonicalize_temporal_axes(self):
        """统一标签和动态特征时间轴键，避免 Timestamp/date/datetime 哈希不一致。"""
        if hasattr(self, 'label_data') and self.label_data:
            canonical_labels = {}
            for key, value in self.label_data.items():
                day = self._canonical_day(key)
                if day is not None:
                    canonical_labels[day] = value
            self.label_data = canonical_labels

        if hasattr(self, 'all_dates') and self.all_dates:
            canonical_dates = []
            seen = set()
            for key in self.all_dates:
                day = self._canonical_day(key)
                if day is not None and day not in seen:
                    canonical_dates.append(day)
                    seen.add(day)
            self.all_dates = sorted(canonical_dates)
            self.date_to_index = {d: i for i, d in enumerate(self.all_dates)}

        if hasattr(self, 's1_data') and self.s1_data:
            self.s1_data = {self._canonical_day(k): v for k, v in self.s1_data.items()}
            self.all_s1_dates = sorted(k for k in self.s1_data if k is not None)
        if hasattr(self, 'smap_data') and self.smap_data:
            self.smap_data = {self._canonical_day(k): v for k, v in self.smap_data.items()}
            self.all_smap_dates = sorted(k for k in self.smap_data if k is not None)

    @staticmethod
    def _parse_station_date_series(series: pd.Series) -> pd.Series:
        """兼容解析 Timestamp、日期字符串、YYYYMMDD 和 Excel 序列日期。

        不使用 ``format="mixed"``，因为部分旧版 pandas 会将其当作普通
        格式字符串并把所有日期静默解析为 NaT。
        """
        raw = series.copy()

        # 先按普通日期解析。Excel 读入的 Timestamp、datetime 和常见日期字符串
        # 都会在这里正确处理；旧版 pandas 也兼容。
        parsed = pd.to_datetime(raw, errors='coerce')
        parsed = pd.Series(parsed, index=raw.index, dtype='datetime64[ns]')

        # 对纯数值日期单独处理：8位整数按 YYYYMMDD，其余合理数值按
        # Excel 1900 日期系统解析。
        numeric = pd.to_numeric(raw, errors='coerce')
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            rounded = numeric.loc[numeric_mask].round().astype('Int64')
            numeric_text = rounded.astype(str)
            ymd_mask = numeric_text.str.fullmatch(r'\d{8}', na=False)

            if ymd_mask.any():
                idx = numeric_text.index[ymd_mask]
                parsed.loc[idx] = pd.to_datetime(
                    numeric_text.loc[idx], format='%Y%m%d', errors='coerce'
                )

            excel_mask = ~ymd_mask
            if excel_mask.any():
                idx = numeric_text.index[excel_mask]
                excel_values = numeric.loc[idx]
                # 只把合理的 Excel 序列日数当作日期，避免把年份等普通数字误判。
                plausible = excel_values.between(1, 100000)
                if plausible.any():
                    valid_idx = excel_values.index[plausible]
                    parsed.loc[valid_idx] = pd.to_datetime(
                        excel_values.loc[valid_idx],
                        unit='D',
                        origin='1899-12-30',
                        errors='coerce',
                    )

        return parsed.dt.normalize()

    def _load_external_station_exclusion(self):
        """读取明确指定的外部测试CSV，并生成中心格点及缓冲排除区。"""
        pattern = self.external_station_glob
        if not pattern:
            self.external_station_centers = set()
            self.external_excluded_cells = set()
            return

        files = [Path(x).expanduser() for x in sorted(glob.glob(pattern))]
        if not files:
            message = f"没有匹配到外部测试CSV: {pattern}"
            if self.external_station_strict:
                raise FileNotFoundError(message)
            print(f"   ⚠ {message}")
            self.external_station_centers = set()
            self.external_excluded_cells = set()
            return

        centers = set()
        center_sources = defaultdict(set)
        file_stats = []
        for file_path in files:
            try:
                df = self._normalize_station_columns(
                    self._read_station_guide_table(file_path)
                )
            except Exception as exc:
                if self.external_station_strict:
                    raise RuntimeError(f"读取外部测试CSV失败: {file_path}: {exc}") from exc
                print(f"   ⚠ 跳过外部CSV {file_path}: {exc}")
                continue
            if 'longitude' not in df.columns or 'latitude' not in df.columns:
                message = f"外部CSV缺少经纬度列: {file_path}"
                if self.external_station_strict:
                    raise ValueError(message)
                print(f"   ⚠ {message}")
                continue
            coords = df[['longitude', 'latitude']].copy()
            coords['longitude'] = pd.to_numeric(coords['longitude'], errors='coerce')
            coords['latitude'] = pd.to_numeric(coords['latitude'], errors='coerce')
            coords = coords.dropna()
            in_grid_rows = 0
            file_cells = set()
            for lon, lat in coords.itertuples(index=False, name=None):
                try:
                    col_f, row_f = ~self.transform * (float(lon), float(lat))
                    r, c = int(row_f), int(col_f)
                except (TypeError, ValueError, OverflowError):
                    continue
                if 0 <= r < self.H and 0 <= c < self.W:
                    in_grid_rows += 1
                    centers.add((r, c))
                    file_cells.add((r, c))
                    center_sources[(r, c)].add(file_path.name)
            file_stats.append({
                'file': str(file_path),
                'rows': int(len(df)),
                'in_grid_rows': int(in_grid_rows),
                'unique_cells': int(len(file_cells)),
            })

        if not centers:
            message = f"外部测试CSV未映射出任何有效ERA5格点: {pattern}"
            if self.external_station_strict:
                raise RuntimeError(message)
            print(f"   ⚠ {message}")

        radius = self.external_station_exclusion_radius
        excluded = set()
        for r, c in centers:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.H and 0 <= cc < self.W:
                        excluded.add((rr, cc))

        self.external_station_centers = centers
        self.external_excluded_cells = excluded
        self.external_station_stats = {
            'glob': pattern,
            'radius': int(radius),
            'files': file_stats,
            'unique_center_cells': int(len(centers)),
            'excluded_cells': int(len(excluded)),
        }

        report_path = self.external_station_report_path
        if report_path is None and self.cache_dir is not None:
            report_path = Path(self.cache_dir) / 'external_station_exclusion_report.csv'
        if report_path is not None:
            report_path = Path(report_path).expanduser()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_rows = []
            for r, c in sorted(excluded):
                lon, lat = self._pixel_to_lonlat(r, c)
                report_rows.append({
                    'row': int(r), 'col': int(c),
                    'longitude': float(lon), 'latitude': float(lat),
                    'is_external_center': int((r, c) in centers),
                    'source_files': ';'.join(sorted(center_sources.get((r, c), set()))),
                    'exclusion_radius': int(radius),
                })
            pd.DataFrame(report_rows).to_csv(report_path, index=False, encoding='utf-8-sig')
            meta_path = report_path.with_suffix(report_path.suffix + '.meta.json')
            with meta_path.open('w', encoding='utf-8') as f:
                json.dump(self.external_station_stats, f, ensure_ascii=False, indent=2)
            self.external_station_report_path = report_path

        print("\n🚫 外部测试站点空间隔离")
        print(f"   文件规则: {pattern}")
        print(f"   匹配CSV: {len(files)}")
        print(f"   唯一中心格点: {len(centers):,}")
        print(f"   排除半径: {radius}格")
        print(f"   排除格点总数: {len(excluded):,}")
        if report_path is not None:
            print(f"   排除报告: {report_path}")

    def _load_station_record_samples_from_manifest(self):
        """从预先冻结的Stage 0清单读取(date,row,col)，避免训练时重新选样。"""
        path = self.station_record_manifest_path
        if path is None:
            raise ValueError("station_record_manifest_path未设置")
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Stage 0清单不存在: {path}")

        print("\n📌 加载冻结的Stage 0样本清单")
        print(f"   清单: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        required = {"date", "row", "col"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Stage 0清单缺少列: {missing}; 现有列={list(df.columns)}")

        dates = self._parse_station_date_series(df["date"])
        rows = pd.to_numeric(df["row"], errors="coerce")
        cols = pd.to_numeric(df["col"], errors="coerce")
        source_rows = (
            pd.to_numeric(df["source_row"], errors="coerce")
            if "source_row" in df.columns else pd.Series(df.index, index=df.index)
        )

        samples = []
        invalid = 0
        date_not_loaded = 0
        out_of_grid = 0
        external_overlap = 0
        for idx in df.index:
            dt = dates.loc[idx]
            r_raw, c_raw = rows.loc[idx], cols.loc[idx]
            if pd.isna(dt) or pd.isna(r_raw) or pd.isna(c_raw):
                invalid += 1
                continue
            date_dt = self._canonical_day(dt)
            if date_dt not in self.label_data or date_dt not in self.date_to_index:
                date_not_loaded += 1
                continue
            r, c = int(r_raw), int(c_raw)
            if not (0 <= r < self.H and 0 <= c < self.W):
                out_of_grid += 1
                continue
            if (r, c) in self.external_excluded_cells:
                external_overlap += 1
                continue
            source = source_rows.loc[idx]
            source_row = int(source) if pd.notna(source) else int(idx)
            samples.append((date_dt, r, c, source_row))

        before = len(samples)
        if self.station_record_dedup == "grid_date":
            seen = set()
            deduped = []
            for item in samples:
                key = (item[0], item[1], item[2])
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            samples = deduped
        duplicates = before - len(samples)

        if external_overlap:
            raise RuntimeError(
                f"冻结Stage 0清单仍包含{external_overlap}条外部测试缓冲区记录；"
                "请重新运行prepare_stage0_manifest.sh"
            )
        if not samples:
            raise RuntimeError(f"冻结Stage 0清单没有可用记录: {path}")

        self.station_record_samples = samples
        self.station_pixels = {(r, c) for _, r, c, _ in samples}
        self.station_record_stats = {
            "source": "frozen_manifest",
            "manifest_path": str(path),
            "input_rows": int(len(df)),
            "invalid_rows": int(invalid),
            "date_not_loaded_rows": int(date_not_loaded),
            "out_of_grid_rows": int(out_of_grid),
            "external_overlap_rows": int(external_overlap),
            "duplicates_removed": int(duplicates),
            "record_candidates": int(len(samples)),
            "unique_pixels": int(len(self.station_pixels)),
        }
        print(f"   清单原始行数: {len(df):,}")
        print(f"   最终冻结候选: {len(samples):,}")
        print(f"   唯一ERA5格点: {len(self.station_pixels):,}")

    def _load_station_record_samples(self):
        """只加载站点文件中实际存在的站点-日期记录。

        默认按 (date, ERA5 row, ERA5 col) 去重。这样 Excel 中有约8000条
        有效记录时，Stage 0 最多生成约8000条候选，而不会扩展为
        “站点位置 × 2015-2018全部日期”。
        """
        file_path = self.station_guide_file
        if file_path is None:
            file_path = self.station_csv_dir / 'station_swe_data.xlsx'
        file_path = Path(file_path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"站点记录文件不存在: {file_path}")

        print("\n📍 加载站点实际记录 (station_sampling_unit=records)")
        print(f"   文件: {file_path}")

        df = self._normalize_station_columns(
            self._read_station_guide_table(file_path)
        )
        input_rows = len(df)

        if self.station_date_column:
            requested = self.station_date_column
            if requested not in df.columns:
                lower_map = {str(c).lower(): c for c in df.columns}
                requested = lower_map.get(requested.lower(), requested)
            if requested not in df.columns:
                raise ValueError(
                    f"指定日期列不存在: {self.station_date_column}; "
                    f"可用列={list(df.columns)}"
                )
            if requested != 'date':
                df = df.rename(columns={requested: 'date'})

        required = ['longitude', 'latitude', 'date']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"站点记录文件缺少必要列 {missing}; 可用列={list(df.columns)}"
            )

        work = df.copy()
        work['longitude'] = pd.to_numeric(work['longitude'], errors='coerce')
        work['latitude'] = pd.to_numeric(work['latitude'], errors='coerce')
        work['date'] = self._parse_station_date_series(work['date'])
        valid_basic = work[['longitude', 'latitude', 'date']].notna().all(axis=1)
        invalid_basic_rows = int((~valid_basic).sum())
        work = work.loc[valid_basic].copy()

        mapped = []
        out_of_grid = 0
        label_date_not_loaded = 0
        feature_date_not_loaded = 0
        external_excluded = 0
        label_days = set(self.label_data.keys())
        feature_days = set(self.date_to_index.keys())
        for source_row, row in work.iterrows():
            date_dt = self._canonical_day(row['date'])
            if date_dt is None:
                label_date_not_loaded += 1
                continue
            if date_dt not in label_days:
                label_date_not_loaded += 1
                continue
            if date_dt not in feature_days:
                feature_date_not_loaded += 1
                continue
            try:
                col_f, row_f = ~self.transform * (
                    float(row['longitude']), float(row['latitude'])
                )
                r, c = int(row_f), int(col_f)
            except (TypeError, ValueError, OverflowError):
                out_of_grid += 1
                continue
            if not (0 <= r < self.H and 0 <= c < self.W):
                out_of_grid += 1
                continue
            if (r, c) in self.external_excluded_cells:
                external_excluded += 1
                continue

            centers = {(r, c)}
            if self.station_neighborhood > 0:
                centers = self._expand_neighborhood(
                    centers, self.station_neighborhood
                )
            for rr, cc in centers:
                mapped.append((date_dt, rr, cc, int(source_row)))

        before_dedup = len(mapped)
        if self.station_record_dedup == 'grid_date':
            unique = {}
            for date_dt, r, c, source_row in mapped:
                unique.setdefault((date_dt, r, c), (date_dt, r, c, source_row))
            mapped = list(unique.values())
        duplicate_count = before_dedup - len(mapped)

        mapped.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        self.station_record_samples = mapped
        self.station_pixels = {(r, c) for _, r, c, _ in mapped}
        self.station_record_stats = {
            'input_rows': int(input_rows),
            'invalid_coordinate_or_date_rows': invalid_basic_rows,
            'label_date_not_loaded_rows': int(label_date_not_loaded),
            'feature_date_not_loaded_rows': int(feature_date_not_loaded),
            'out_of_grid_rows': int(out_of_grid),
            'external_buffer_excluded_rows': int(external_excluded),
            'mapped_before_dedup': int(before_dedup),
            'duplicates_removed': int(duplicate_count),
            'record_candidates': int(len(mapped)),
            'unique_pixels': int(len(self.station_pixels)),
            'dedup_mode': self.station_record_dedup,
            'neighborhood': int(self.station_neighborhood),
        }

        if not self.station_record_samples:
            label_range = (
                f"{min(label_days):%Y-%m-%d} 至 {max(label_days):%Y-%m-%d}"
                if label_days else "空"
            )
            feature_range = (
                f"{min(feature_days):%Y-%m-%d} 至 {max(feature_days):%Y-%m-%d}"
                if feature_days else "空"
            )
            raise RuntimeError(
                "站点文件实际记录全部被筛除："
                f"input={input_rows}, invalid_basic={invalid_basic_rows}, "
                f"label_date_missing={label_date_not_loaded}, "
                f"feature_common_date_missing={feature_date_not_loaded}, "
                f"out_of_grid={out_of_grid}, external_excluded={external_excluded}; "
                f"label_days={len(label_days)}({label_range}), "
                f"feature_common_days={len(feature_days)}({feature_range})。"
                "请查看上方四个动态变量的日期数与共同日期数。"
            )

        print(f"   原始记录行数: {input_rows:,}")
        print(f"   坐标/日期无效: {invalid_basic_rows:,}")
        print(f"   日期不在ERA5-Land标签中: {label_date_not_loaded:,}")
        print(f"   日期不在四动态特征共同时间轴中: {feature_date_not_loaded:,}")
        print(f"   网格范围外: {out_of_grid:,}")
        print(f"   外部测试站点缓冲区排除: {external_excluded:,}")
        print(f"   映射后候选: {before_dedup:,}")
        print(f"   完全重复(date,row,col)删除: {duplicate_count:,}")
        print(f"   最终实际记录候选: {len(mapped):,}")
        print(f"   涉及唯一ERA5格点: {len(self.station_pixels):,}")

    def _build_station_record_guided_samples(self):
        """验证并加入站点文件中的实际站点-日期记录。"""
        if not self.station_record_samples:
            raise RuntimeError(
                "station_sampling_unit=records，但 station_record_samples 为空"
            )

        existing = {(item[0], item[1], item[2]) for item in self.meta_index}
        by_date = defaultdict(list)
        for date_dt, r, c, source_row in self.station_record_samples:
            by_date[date_dt].append((r, c, source_row))

        rng = np.random.default_rng(43)
        added = 0
        failed_label = 0
        failed_feature = 0
        failed_conv = 0
        failed_point = 0
        zero_filtered = 0
        duplicate_existing = 0
        samples_per_date = {}
        manifest_rows = []

        for date_dt in sorted(by_date):
            records = list(by_date[date_dt])
            if self.station_samples_per_day > 0 and len(records) > self.station_samples_per_day:
                idx = rng.choice(
                    len(records), size=self.station_samples_per_day, replace=False
                )
                records = [records[int(i)] for i in sorted(idx)]

            label_arr, label_nodata = self.label_data.get(date_dt, (None, None))
            if label_arr is None:
                failed_label += len(records)
                continue

            added_today = 0
            for r, c, source_row in records:
                key = (date_dt, r, c)
                if key in existing:
                    duplicate_existing += 1
                    continue

                y = float(label_arr[r, c])
                if not np.isfinite(y) or (label_nodata is not None and y == label_nodata):
                    failed_label += 1
                    continue
                if (
                    self.fixed_label_min_mm is not None and y < self.fixed_label_min_mm
                ) or (
                    self.fixed_label_max_mm is not None and y >= self.fixed_label_max_mm
                ):
                    failed_label += 1
                    continue
                if self.station_filter_zero_target and y <= 1e-6:
                    zero_filtered += 1
                    continue

                is_valid, fail_reason, conv_patch, point_feats = (
                    self._validate_station_sample_with_reason(date_dt, r, c)
                )
                if not is_valid or conv_patch is None or point_feats is None:
                    if fail_reason == 'label':
                        failed_label += 1
                    else:
                        failed_feature += 1
                        if fail_reason == 'conv':
                            failed_conv += 1
                        elif fail_reason == 'point':
                            failed_point += 1
                    continue

                self.meta_index.append((date_dt, r, c, 'station'))
                manifest_rows.append({
                    'date': date_dt.strftime('%Y-%m-%d'),
                    'row': int(r),
                    'col': int(c),
                    'source_row': int(source_row),
                    'era5_swe_mm': float(y),
                })
                existing.add(key)
                added += 1
                added_today += 1

            if added_today:
                samples_per_date[date_dt] = added_today

        self.stage0_manifest_rows = list(manifest_rows)
        manifest_path = None
        stats_path = None
        if self.cache_dir is not None:
            cache_root = Path(self.cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            manifest_path = cache_root / 'stage0_station_record_manifest.csv'
            stats_path = cache_root / 'stage0_station_record_manifest.stats.json'
            pd.DataFrame(manifest_rows).to_csv(
                manifest_path, index=False, encoding='utf-8-sig'
            )

        self.station_sample_stats = {
            'sampling_unit': 'records',
            'record_candidates': len(self.station_record_samples),
            'added_count': added,
            'label_failed': failed_label,
            'feature_failed': failed_feature,
            'conv_failed': failed_conv,
            'point_failed': failed_point,
            'zero_filtered': zero_filtered,
            'duplicate_existing': duplicate_existing,
            'samples_per_date': samples_per_date,
            'source_stats': dict(self.station_record_stats),
            'manifest_path': str(manifest_path) if manifest_path else None,
        }
        if stats_path is not None:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.station_sample_stats, f, ensure_ascii=False, indent=2, default=str)

        print("\n" + "=" * 70)
        print("📊 Stage 0 站点实际记录采样统计")
        print("=" * 70)
        print(f"   实际记录候选: {len(self.station_record_samples):,}")
        print(f"   成功加入预训练: {added:,}")
        print(f"   标签无效: {failed_label:,}")
        print(f"   特征/Patch无效: {failed_feature:,}")
        print(f"     ├─ 卷积Patch无效: {failed_conv:,}")
        print(f"     └─ 点特征无效: {failed_point:,}")
        print(f"   ERA5 SWE=0被过滤: {zero_filtered:,}")
        print(f"   已存在重复样本: {duplicate_existing:,}")
        print(f"   涉及日期数: {len(samples_per_date):,}")
        if manifest_path is not None:
            print(f"   最终样本清单: {manifest_path}")
            print(f"   统计文件: {stats_path}")
        print("=" * 70)

    def _load_all_station_pixels(self):
        """加载站点引导位置；优先读取一个明确指定的文件。"""
        print("\n📍 加载站点数据 (用于引导采样)")

        if self.station_guide_file is not None:
            station_files = [self.station_guide_file]
        else:
            # [COMPAT] 未传具体文件时，保留旧目录多文件行为。
            station_files = [
                self.station_csv_dir / "station_swe_data.xlsx",
                self.station_csv_dir / "long_comb.csv",
                self.station_csv_dir / "long_comb2.csv",
                self.station_csv_dir / "long_comb3.csv",
                self.station_csv_dir / "one_record.csv",
            ]

        all_station_pixels = set()

        for file_path in station_files:
            file_path = Path(file_path).expanduser()
            if not file_path.exists():
                print(f"  ⚠ 文件不存在: {file_path}")
                continue

            print(f"  正在读取: {file_path}")

            try:
                if file_path.suffix.lower() in {'.xlsx', '.xls'}:
                    df = pd.read_excel(file_path, engine='openpyxl')
                elif file_path.suffix.lower() == '.csv':
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(file_path, encoding='gbk')
                        except UnicodeDecodeError:
                            df = pd.read_csv(file_path, encoding='latin1')
                else:
                    print(f"    跳过: 不支持的文件格式 {file_path.suffix}")
                    continue

                column_mapping = {
                    'longtitude': 'longitude',
                    'lon': 'longitude',
                    'lng': 'longitude',
                    'long': 'longitude',
                    'latitude': 'latitude',
                    'lat': 'latitude',
                }
                df = df.rename(
                    columns=lambda x: column_mapping.get(str(x).strip().lower(), str(x).strip())
                )

                if 'longitude' not in df.columns or 'latitude' not in df.columns:
                    print("    跳过: 缺少 longitude/latitude 列")
                    continue

                coords = df[['longitude', 'latitude']].copy()
                coords['longitude'] = pd.to_numeric(coords['longitude'], errors='coerce')
                coords['latitude'] = pd.to_numeric(coords['latitude'], errors='coerce')
                coords = coords.dropna().drop_duplicates()

                before = len(all_station_pixels)
                out_of_grid = 0
                for lon, lat in coords.itertuples(index=False, name=None):
                    try:
                        col, row = ~self.transform * (float(lon), float(lat))
                        r, c = int(row), int(col)
                        if 0 <= r < self.H and 0 <= c < self.W:
                            all_station_pixels.add((r, c))
                        else:
                            out_of_grid += 1
                    except (TypeError, ValueError, OverflowError):
                        continue

                added = len(all_station_pixels) - before
                print(
                    f"    唯一坐标={len(coords):,}，新增唯一格点={added:,}，"
                    f"网格外={out_of_grid:,}"
                )

            except Exception as e:
                print(f"    读取失败: {e}")

        if not all_station_pixels:
            source = self.station_guide_file or self.station_csv_dir
            raise RuntimeError(f"未从站点引导数据中得到有效格点: {source}")

        # 只在这里扩展一次，避免旧代码的重复邻域膨胀。
        if self.station_neighborhood > 0:
            print(f"\n  🔍 扩展邻域 (半径={self.station_neighborhood})")
            original_count = len(all_station_pixels)
            all_station_pixels = self._expand_neighborhood(
                all_station_pixels, self.station_neighborhood
            )
            print(f"      {original_count:,} → {len(all_station_pixels):,} 个像元")

        self.station_pixels = all_station_pixels
        limit_text = "全部" if self.station_samples_per_day <= 0 else f"{self.station_samples_per_day:,}"
        print("\n  ✅ 站点引导采样已启用")
        print(f"     sampling_mode: {self.sampling_mode}")
        print(f"     站点文件: {self.station_guide_file or self.station_csv_dir}")
        print(f"     邻域半径: {self.station_neighborhood}")
        print(f"     总引导格点数: {len(self.station_pixels):,}")
        print(f"     每天最多添加: {limit_text}")

    def _expand_neighborhood(self, pixels: set, radius: int) -> set:
        """扩展站点像元的邻域"""
        expanded = set()
        step = radius * 2 + 1  # 7x7 = 半径3

        for r, c in pixels:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.H and 0 <= nc < self.W:
                        expanded.add((nr, nc))

        return expanded
    
    def _build_point_features_station(self, date_dt: datetime, r: int, c: int) -> np.ndarray:
        """
        宽松版点特征构建（用于站点引导采样）
        维度必须与严格版一致：18维

        无效值处理策略：
        - LS: 无效值 -> 0
        - 哨兵1: -9999 -> 0
        - SMAP: -9999 -> 250, mask=0标识无效
        """
        point_features = []

        # ============ 1. LS特征 (6个) - 无效值用0填充 ============
        if hasattr(self, 'ls_data_default'):
            for i in range(min(6, self.ls_data_default.shape[0])):
                val = self.ls_data_default[i, r, c]
                if (not np.isfinite(val)) or val == -9999.0:
                    val = 0.0
                point_features.append(float(val))
        else:
            for _ in range(6):
                point_features.append(0.0)

        # ============ 2. 哨兵1特征 (5个) - 无数据时用0填充 ============
        s1_vv, s1_vh, s1_vv_cov, s1_vh_cov, s1_angle = self._get_sentinel1_value_loose(date_dt, r, c)
        point_features.append(float(s1_vv))
        point_features.append(float(s1_vh))
        point_features.append(float(s1_vv_cov) if s1_vv_cov >= 0 else 0.0)
        point_features.append(float(s1_vh_cov) if s1_vh_cov >= 0 else 0.0)
        point_features.append(float(s1_angle) if s1_angle != self.s1_nodata_value else 0.0)

        # ============ 3. SMAP亮温特征 (2个) + mask (2个) ============
        smap_tbv, smap_tbh = self._get_smap_value_loose(date_dt, r, c)
        point_features.append(float(smap_tbv))
        point_features.append(float(smap_tbh))

        mask_v = self._get_smap_mask_loose(date_dt, r, c, 'V')
        mask_h = self._get_smap_mask_loose(date_dt, r, c, 'H')
        point_features.append(float(mask_v))
        point_features.append(float(mask_h))

        # ============ 4. 经纬度特征 (2个) ============
        lon, lat = self._pixel_to_lonlat(r, c)
        lon_norm = (lon + 180) / 360
        lat_norm = (lat + 90) / 180
        point_features.extend([lon_norm, lat_norm])

        # ============ 5. 时间特征 (1个) ============
        time_feats = self._build_time_features(date_dt)
        point_features.extend(time_feats)

        point_feats_array = np.array(point_features, dtype=np.float32)
        point_feats_array = np.nan_to_num(point_feats_array, nan=0.0)

        # 维度检查
        if len(point_features) != 18:
            print(f"⚠ 宽松版点特征维度错误: {len(point_features)} != 18")
            return None

        # Stage 0 的站点-日期记录很珍贵，不再因为当天 S1 与 SMAP 同时缺失而删除。
        # 缺失信息已经显式编码：S1缺失值/覆盖度为0，SMAP使用mask=0；
        # 模型仍可使用卷积气象、地形、Landsat、经纬度与时间特征。
        return point_feats_array

    def _get_sentinel1_value_loose(self, date_dt: datetime, r: int, c: int):
        """
        宽松版获取哨兵1值 - 正确识别NODATA=-9999，无数据时设为0
        """

        # 🔥 哨兵1的真实NODATA值是 -9999
        S1_NODATA = -9999.0

        # 默认值（当没有任何数据时使用）
        default_vv = 0.0      # 改为0
        default_vh = 0.0      # 改为0
        default_cov = 0.0
        default_angle = 0.0   # 改为0

        if not self.all_s1_dates:
            return default_vv, default_vh, default_cov, default_cov, default_angle

        # 找最近的日期
        closest_date = min(self.all_s1_dates, key=lambda d: abs((d - date_dt).days))

        if closest_date not in self.s1_data:
            return default_vv, default_vh, default_cov, default_cov, default_angle

        data = self.s1_data[closest_date]

        vv = default_vv
        vh = default_vh
        vv_cov = default_cov
        vh_cov = default_cov
        angle = default_angle

        # 🔥 S1_VV：-9999 表示无数据，设为0
        if 'VV' in data and data['VV'] is not None:
            val = data['VV'][r, c]
            # 只有不是 -9999 且不是其他无效值时，才使用实际值
            if val != S1_NODATA and val != self.s1_nodata_value and np.isfinite(val):
                vv = float(val)
            else:
                vv = 0.0  # 无数据时设为0

        # 🔥 S1_VH：-9999 表示无数据，设为0
        if 'VH' in data and data['VH'] is not None:
            val = data['VH'][r, c]
            if val != S1_NODATA and val != self.s1_nodata_value and np.isfinite(val):
                vh = float(val)
            else:
                vh = 0.0  # 无数据时设为0

        # VV_cov 和 VH_cov 的处理保持不变
        if 'VV_cov' in data and data['VV_cov'] is not None:
            val = data['VV_cov'][r, c]
            if val >= 0 and np.isfinite(val):
                vv_cov = float(val)

        if 'VH_cov' in data and data['VH_cov'] is not None:
            val = data['VH_cov'][r, c]
            if val >= 0 and np.isfinite(val):
                vh_cov = float(val)

        # 🔥 S1_angle：-9999 表示无数据，设为0
        if 'angle' in data and data['angle'] is not None:
            val = data['angle'][r, c]
            if val != S1_NODATA and val != self.s1_nodata_value and np.isfinite(val):
                angle = float(val)
            else:
                angle = 0.0  # 无数据时设为0

        return vv, vh, vv_cov, vh_cov, angle

    def _get_smap_value_loose(self, date_dt: datetime, r: int, c: int):
        """
        宽松版获取SMAP值 - 正确识别无效值 -9999
        无数据时使用 250 填充（有效亮温的合理均值）
        """

        SMAP_NODATA = -9999.0
        DEFAULT_TB = 250.0  # 有效亮温的合理均值

        if not self.all_smap_dates:
            return DEFAULT_TB, DEFAULT_TB

        # 找最近的日期
        closest_date = min(self.all_smap_dates, key=lambda d: abs((d - date_dt).days))

        if closest_date not in self.smap_data:
            return DEFAULT_TB, DEFAULT_TB

        data = self.smap_data[closest_date]

        tbv = DEFAULT_TB
        tbh = DEFAULT_TB

        # 🔥 TBV：只有不是 -9999 才算有效
        if 'TBV' in data and data['TBV'] is not None:
            val = data['TBV'][r, c]
            if val != SMAP_NODATA and np.isfinite(val):
                tbv = float(val)
            # else: 保持 DEFAULT_TB，同时 mask 会是 0

        # 🔥 TBH：只有不是 -9999 才算有效
        if 'TBH' in data and data['TBH'] is not None:
            val = data['TBH'][r, c]
            if val != SMAP_NODATA and np.isfinite(val):
                tbh = float(val)
            # else: 保持 DEFAULT_TB，同时 mask 会是 0

        return tbv, tbh
            
        
    def _get_smap_mask_loose(self, date_dt: datetime, r: int, c: int, pol: str = 'V') -> float:
        """
        宽松版获取 SMAP mask
        mask 值：0=无效，1=有效（或其他正数表示有效）
        当亮温值为 -9999 时，mask 也会是 0
        """
        SMAP_NODATA = -9999.0

        if date_dt not in self.smap_data:
            return 0.0

        data = self.smap_data[date_dt]
        mask_key = f'mask_{pol}'

        if mask_key in data and data[mask_key] is not None:
            mask_val = data[mask_key][r, c]

            # 检查对应的亮温值是否有效
            tb_key = f'TB{pol}'
            if tb_key in data and data[tb_key] is not None:
                tb_val = data[tb_key][r, c]
                # 只有亮温值不是 -9999 时，mask 才有意义
                if tb_val != SMAP_NODATA and np.isfinite(tb_val):
                    return float(mask_val)

        return 0.0  # 无效时 mask=0
  
    def _compute_minmax_sampling(self):
        """计算特征的min/max用于标准化 - 基于meta_index中的实际样本"""
        print(f"\n计算特征统计量 (基于 {len(self.meta_index)} 个实际样本)...")

        # ============ 1. 卷积特征的统计量 ============
        conv_mins = []
        conv_maxs = []

        # 🔥 定义各卷积变量的无效值
        CONV_INVALID_VALUES = {
            "chelsa_sfxwind": -9999.0,
            "lst": -9999.0,
            "rh": -9999.0,
            "pr": -9999.0,
        }
        CLAMDAY_INVALID = -9999.0

        # 动态卷积特征 - 基于实际样本采样
        print(f"\n【卷积特征统计 - 基于实际样本采样】")

        # 为了效率，采样部分样本来估计范围（最多采样10000个）
        sample_size = min(10000, len(self.meta_index))
        sample_indices = np.random.choice(len(self.meta_index), sample_size, replace=False)
        print(f"  采样样本数: {sample_size}")

        # 收集每个变量的有效值
        var_values = {var: [] for var in CONV_VARS}
        clamday_values = []
        dem_values = [[] for _ in range(len(self.dem_data))]

        for idx in sample_indices:
            # 🔥 修复：兼容新旧格式
            item = self.meta_index[idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                date_dt, r, c = item

            date_idx = self.date_to_index.get(date_dt)
            if date_idx is None:
                continue

            # 动态卷积特征
            for var in CONV_VARS:
                var_arr = self.conv_dyn_data[var]
                if date_idx < var_arr.shape[0]:
                    val = var_arr[date_idx, r, c]
                    invalid_val = CONV_INVALID_VALUES.get(var)
                    if invalid_val is not None:
                        if val != invalid_val and np.isfinite(val):
                            var_values[var].append(float(val))
                    else:
                        if np.isfinite(val):
                            var_values[var].append(float(val))

            # clamday
            if self.clamday_data is not None:
                val = self.clamday_data[r, c]
                if val != CLAMDAY_INVALID and np.isfinite(val):
                    clamday_values.append(float(val))

            # DEM 波段
            for i, dem_band in enumerate(self.dem_data):
                val = dem_band[r, c]
                if np.isfinite(val) and val != -9999.0:
                    dem_values[i].append(float(val))

        # 计算各变量的范围
        for var in CONV_VARS:
            if var_values[var]:
                min_val = float(np.min(var_values[var]))
                max_val = float(np.max(var_values[var]))
            else:
                # 如果采样中没有有效值，回退到全局统计
                arr = self.conv_dyn_data[var]
                invalid_val = CONV_INVALID_VALUES.get(var)
                if invalid_val is not None:
                    valid_data = arr[(arr != invalid_val) & np.isfinite(arr)]
                else:
                    valid_data = arr[np.isfinite(arr)]
                if len(valid_data) > 0:
                    min_val = float(np.min(valid_data))
                    max_val = float(np.max(valid_data))
                else:
                    min_val, max_val = 0.0, 1.0
                    print(f"    ⚠ {var}: 无任何有效数据，使用默认[0.0, 1.0]")

            conv_mins.append(min_val)
            conv_maxs.append(max_val)
            print(f"  {var}: [{min_val:.4f}, {max_val:.4f}] (采样{len(var_values[var])}个)")

        # clamday
        if clamday_values:
            min_val = float(np.min(clamday_values))
            max_val = float(np.max(clamday_values))
        else:
            valid_mask = (self.clamday_data != CLAMDAY_INVALID) & np.isfinite(self.clamday_data)
            valid_data = self.clamday_data[valid_mask]
            if len(valid_data) > 0:
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))
                print(f"    ⚠ clamday: 采样无有效值，使用全局统计")
            else:
                min_val, max_val = 0.0, 1.0
                print(f"    ⚠ clamday: 无任何有效数据，使用默认[0.0, 1.0]")

        conv_mins.append(min_val)
        conv_maxs.append(max_val)
        print(f"  clamday: [{min_val:.4f}, {max_val:.4f}] (采样{len(clamday_values)}个)")

        # DEM 波段
        for i, dem_vals in enumerate(dem_values):
            if dem_vals:
                min_val = float(np.min(dem_vals))
                max_val = float(np.max(dem_vals))
            else:
                valid_data = self.dem_data[i][
                    np.isfinite(self.dem_data[i]) & (self.dem_data[i] != -9999.0)
                ]
                if len(valid_data) > 0:
                    min_val = float(np.min(valid_data))
                    max_val = float(np.max(valid_data))
                    print(f"    ⚠ DEM_band{i}: 采样无有效值，使用全局统计")
                else:
                    min_val, max_val = 0.0, 1.0
                    print(f"    ⚠ DEM_band{i}: 无任何有效数据，使用默认[0.0, 1.0]")

            conv_mins.append(min_val)
            conv_maxs.append(max_val)
            print(f"  DEM_band{i}: [{min_val:.4f}, {max_val:.4f}] (采样{len(dem_vals)}个)")

        self.conv_min = np.array(conv_mins, dtype=np.float32)
        self.conv_max = np.array(conv_maxs, dtype=np.float32)
        self.C_conv = len(self.conv_min)

        print(f"\n✅ 卷积特征维度统计:")
        print(f"  动态特征: {len(CONV_VARS)}")
        print(f"  Clamday: 1")
        print(f"  DEM波段: {len(self.dem_data)}")
        print(f"  总通道数 C_conv: {self.C_conv}")

        # ============ 2. 点特征的统计量 (基于实际样本) ============
        print(f"\n【点特征统计 - 基于 {len(self.meta_index)} 个实际样本】")
        point_mins = []
        point_maxs = []

        # ============ 2.1 LS特征 (6个) ============
        print(f"\n  LS特征:")

        ls_values = [[] for _ in range(6)]

        for idx in sample_indices:
            # 🔥 修复：兼容新旧格式
            item = self.meta_index[idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                date_dt, r, c = item

            year = date_dt.year

            if hasattr(self, 'ls_data') and isinstance(self.ls_data, dict) and year in self.ls_data:
                ls_arr = self.ls_data[year]
            elif hasattr(self, 'ls_data_default'):
                ls_arr = self.ls_data_default
            else:
                continue

            for band in range(min(6, ls_arr.shape[0])):
                val = ls_arr[band, r, c]
                if np.isfinite(val):
                    ls_values[band].append(float(val))

        for band in range(6):
            if ls_values[band]:
                min_val = float(np.min(ls_values[band]))
                max_val = float(np.max(ls_values[band]))
            else:
                min_val, max_val = 0.0, 1.0
                print(f"    ⚠ LS波段{band+1}: 采样无有效值，使用默认[0.0, 1.0]")

            point_mins.append(min_val)
            point_maxs.append(max_val)
            print(f"    LS波段{band+1}: [{min_val:.4f}, {max_val:.4f}] (采样{len(ls_values[band])}个)")

        # ============ 2.2 哨兵1特征 (5个) ============
        print(f"\n  哨兵1特征 (基于实际样本，VV/VH需要cov>0):")

        s1_vv_values = []
        s1_vh_values = []
        s1_angle_values = []
        s1_vv_cov_values = []
        s1_vh_cov_values = []

        for idx in sample_indices:
            # 🔥 修复：兼容新旧格式
            item = self.meta_index[idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                date_dt, r, c = item

            s1_vv, s1_vh, s1_vv_cov, s1_vh_cov, s1_angle = self._get_sentinel1_value(date_dt, r, c)

            if s1_vv != self.s1_nodata_value and np.isfinite(s1_vv) and s1_vv_cov > 0:
                s1_vv_values.append(float(s1_vv))

            if s1_vh != self.s1_nodata_value and np.isfinite(s1_vh) and s1_vh_cov > 0:
                s1_vh_values.append(float(s1_vh))

            if s1_vv_cov >= 0 and np.isfinite(s1_vv_cov):
                s1_vv_cov_values.append(float(s1_vv_cov))

            if s1_vh_cov >= 0 and np.isfinite(s1_vh_cov):
                s1_vh_cov_values.append(float(s1_vh_cov))

            if s1_angle != self.s1_nodata_value and np.isfinite(s1_angle):
                if s1_vv_cov > 0 or s1_vh_cov > 0:
                    s1_angle_values.append(float(s1_angle))

        # VV
        if s1_vv_values:
            min_val = float(np.min(s1_vv_values))
            max_val = float(np.max(s1_vv_values))
        else:
            min_val, max_val = -25.0, 25.0
            print(f"    ⚠ S1_VV: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_VV: [{min_val:.2f}, {max_val:.2f}] (采样{len(s1_vv_values)}个)")

        # VH
        if s1_vh_values:
            min_val = float(np.min(s1_vh_values))
            max_val = float(np.max(s1_vh_values))
        else:
            min_val, max_val = -30.0, 20.0
            print(f"    ⚠ S1_VH: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_VH: [{min_val:.2f}, {max_val:.2f}] (采样{len(s1_vh_values)}个)")

        # VV_cov
        if s1_vv_cov_values:
            min_val = float(np.min(s1_vv_cov_values))
            max_val = float(np.max(s1_vv_cov_values))
        else:
            min_val, max_val = 0.0, 1.0
            print(f"    ⚠ S1_VV_cov: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_VV_cov: [{min_val:.4f}, {max_val:.4f}] (采样{len(s1_vv_cov_values)}个)")

        # VH_cov
        if s1_vh_cov_values:
            min_val = float(np.min(s1_vh_cov_values))
            max_val = float(np.max(s1_vh_cov_values))
        else:
            min_val, max_val = 0.0, 1.0
            print(f"    ⚠ S1_VH_cov: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_VH_cov: [{min_val:.4f}, {max_val:.4f}] (采样{len(s1_vh_cov_values)}个)")

        # angle
        if s1_angle_values:
            min_val = float(np.min(s1_angle_values))
            max_val = float(np.max(s1_angle_values))
        else:
            min_val, max_val = 0.0, 90.0
            print(f"    ⚠ S1_angle: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_angle: [{min_val:.2f}, {max_val:.2f}] (采样{len(s1_angle_values)}个)")

        # ============ 2.3 SMAP亮温特征 ============
        print(f"\n  SMAP亮温特征 (基于实际样本):")

        smap_tbv_values = []
        smap_tbh_values = []
        smap_mask_v_values = []
        smap_mask_h_values = []

        for idx in sample_indices:
            # 🔥 修复：兼容新旧格式
            item = self.meta_index[idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                date_dt, r, c = item

            smap_tbv, smap_tbh = self._get_smap_value(date_dt, r, c)

            if smap_tbv != self.smap_nodata_value and np.isfinite(smap_tbv):
                smap_tbv_values.append(float(smap_tbv))

            if smap_tbh != self.smap_nodata_value and np.isfinite(smap_tbh):
                smap_tbh_values.append(float(smap_tbh))

            if smap_tbv != self.smap_nodata_value and np.isfinite(smap_tbv):
                mask_v = self._get_smap_mask(date_dt, r, c, 'V')
                if np.isfinite(mask_v):
                    smap_mask_v_values.append(float(mask_v))

            if smap_tbh != self.smap_nodata_value and np.isfinite(smap_tbh):
                mask_h = self._get_smap_mask(date_dt, r, c, 'H')
                if np.isfinite(mask_h):
                    smap_mask_h_values.append(float(mask_h))

        # TBV
        if smap_tbv_values:
            min_val = float(np.min(smap_tbv_values))
            max_val = float(np.max(smap_tbv_values))
        else:
            min_val, max_val = 180.0, 320.0
            print(f"    ⚠ SMAP_TBV: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    SMAP_TBV: [{min_val:.2f}, {max_val:.2f}] (采样{len(smap_tbv_values)}个)")

        # TBH
        if smap_tbh_values:
            min_val = float(np.min(smap_tbh_values))
            max_val = float(np.max(smap_tbh_values))
        else:
            min_val, max_val = 180.0, 320.0
            print(f"    ⚠ SMAP_TBH: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    SMAP_TBH: [{min_val:.2f}, {max_val:.2f}] (采样{len(smap_tbh_values)}个)")

        # SMAP mask V
        if smap_mask_v_values:
            min_val = float(np.min(smap_mask_v_values))
            max_val = float(np.max(smap_mask_v_values))
        else:
            min_val, max_val = 0.0, 1.0
            print(f"    ⚠ SMAP_mask_V: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    SMAP_mask_V: [{min_val:.4f}, {max_val:.4f}] (采样{len(smap_mask_v_values)}个)")

        # SMAP mask H
        if smap_mask_h_values:
            min_val = float(np.min(smap_mask_h_values))
            max_val = float(np.max(smap_mask_h_values))
        else:
            min_val, max_val = 0.0, 1.0
            print(f"    ⚠ SMAP_mask_H: 采样无有效值，使用默认")
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    SMAP_mask_H: [{min_val:.4f}, {max_val:.4f}] (采样{len(smap_mask_h_values)}个)")

        # ============ 2.4 经纬度特征 ============
        print(f"\n  空间特征:")
        lon_values = []
        lat_values = []

        for meta in self.meta_index:
            # 🔥 修复：兼容新旧格式
            if len(meta) == 4:
                date, r, c, source = meta
            else:
                date, r, c = meta

            lon, lat = self._pixel_to_lonlat(r, c)
            lon_values.append(lon)
            lat_values.append(lat)

        if lon_values:
            lon_min = min(lon_values)
            lon_max = max(lon_values)
            lat_min = min(lat_values)
            lat_max = max(lat_values)

            print(f"    经度范围: [{lon_min:.4f}, {lon_max:.4f}]")
            print(f"    纬度范围: [{lat_min:.4f}, {lat_max:.4f}]")

            self.lon_raw_min = lon_min
            self.lon_raw_max = lon_max
            self.lat_raw_min = lat_min
            self.lat_raw_max = lat_max

            point_mins.append(0.0)
            point_maxs.append(1.0)
            point_mins.append(0.0)
            point_maxs.append(1.0)
            print(f"    经纬度归一化: [0.0, 1.0]")
        else:
            point_mins.extend([0.0, 0.0])
            point_maxs.extend([1.0, 1.0])
            print(f"    ⚠ 无有效经纬度数据，使用默认[0.0, 1.0]")

        # ============ 2.5 DOY特征 ============
        print(f"\n  时间特征:")
        doy_values = []
        for meta in self.meta_index:
            # 🔥 修复：兼容新旧格式
            if len(meta) == 4:
                date, r, c, source = meta
            else:
                date, r, c = meta

            doy_values.append(date.timetuple().tm_yday)

        if doy_values:
            doy_min = min(doy_values)
            doy_max = max(doy_values)
            self.doy_raw_min = doy_min
            self.doy_raw_max = doy_max
            point_mins.append(0.0)
            point_maxs.append(1.0)
            print(f"    DOY范围: [{doy_min}, {doy_max}]")
            print(f"    DOY归一化: [0.0, 1.0]")
        else:
            point_mins.append(0.0)
            point_maxs.append(1.0)
            print(f"    ⚠ 无有效DOY数据，使用默认[0.0, 1.0]")



        # ============ 3. 最终维度检查和汇总 ============
        self.point_min = np.array(point_mins, dtype=np.float32)
        self.point_max = np.array(point_maxs, dtype=np.float32)
        self.C_point = len(self.point_min)

        expected = 6 + 5 + 2 + 2 + 2 + 1
        print(f"\n【最终点特征维度: {self.C_point}】")
        print(f"  组成: LS(6) + S1(5) + SMAP_TB(2) + SMAP_mask(2) + 经纬度(2) + DOY(1) = {expected}")
        print(f"  顺序确认: LS(6) → S1_VV(1) → S1_VH(1) → S1_VV_cov(1) → S1_VH_cov(1) → S1_angle(1) → SMAP_TB(2) → SMAP_mask(2) → 经纬度(2) → DOY(1)")

        if self.C_point != expected:
            print(f"  ⚠ 警告: 实际维度{self.C_point} != 预期{expected}")

        # ============ 4. 标签的统计量 ============
        print(f"\n【标签统计 - 基于实际样本】")
        label_values = []
        for idx in sample_indices:
            # 🔥 修复：兼容新旧格式
            item = self.meta_index[idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                date_dt, r, c = item

            if date_dt in self.label_data:
                label_arr, label_nodata = self.label_data[date_dt]
                val = label_arr[r, c]
                if (label_nodata is None or val != label_nodata) and np.isfinite(val):
                    label_values.append(float(val))

        if label_values:
            self.label_min = float(np.min(label_values))
            self.label_max = float(np.max(label_values))
            print(f"  标签范围: [{self.label_min:.4f}, {self.label_max:.4f}] (采样{len(label_values)}个)")
        else:
            self.label_min = 0.0
            self.label_max = 200.0
            print(f"  ⚠ 无有效标签，使用默认[0.0, 200.0]")

        # main_tune.py 中大量诊断/加权代码使用 swe_min/swe_max。
        # 预训练数据集原来只有 label_min/label_max，导致捕获 SWE 范围失败并退回默认值。
        self.swe_min = self.label_min
        self.swe_max = self.label_max

        print(f"\n✅ 统计完成!")
        print(f"  卷积特征: {self.C_conv} 个通道")
        print(f"  点特征: {self.C_point} 个维度")

        return

    def _build_time_features(self, date_dt: datetime) -> np.ndarray:
        """构建时间特征"""
        # 年日 (一年中的第几天)
        day_of_year = date_dt.timetuple().tm_yday
        # 归一化到0-1
        doy_norm = (day_of_year - 1) / 365.0
        return np.array([doy_norm], dtype=np.float32)

    def _build_spatial_features(self, date_dt: datetime, r: int, c: int, strict: bool = True) -> np.ndarray:
        """构建卷积特征 - 动态支持多波段DEM，检查NaN和无效值

        Args:
            date_dt: 日期
            r, c: 行列坐标
            strict: 是否严格检查无效值（True：发现无效值返回None；False：允许插值填充）
        """
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

        if actual_h < 3 or actual_w < 3:
            return None

        # 定义各卷积变量的无效值
        INVALID_VALUES = {
            "chelsa_sfxwind": -9999.0,
            "lst": -9999.0,
            "rh": -9999.0,
            "pr": -9999.0,
        }
        CLAMDAY_INVALID = -9999.0

        # 收集所有卷积特征
        conv_features = []
        has_invalid = False

        # 1. 动态卷积特征
        for var in CONV_VARS:
            var_arr = self.conv_dyn_data[var]
            if date_idx >= var_arr.shape[0]:
                patch = var_arr[-1, r0:r1, c0:c1]
            else:
                patch = var_arr[date_idx, r0:r1, c0:c1]

            if patch.shape != (self.P, self.P):
                patch = self._resize_to_standard(patch, self.P, self.P)

            invalid_val = INVALID_VALUES.get(var)
            if invalid_val is not None:
                if np.any(patch == invalid_val):
                    if strict:
                        return None
                    else:
                        # 宽松模式：插值填充
                        patch = self._interpolate_nan_patch(patch)
                        has_invalid = True
            if np.any(np.isnan(patch)):
                if strict:
                    return None
                else:
                    patch = self._interpolate_nan_patch(patch)
                    has_invalid = True

            conv_features.append(patch)

        # 2. clamday
        clamday_patch = self.clamday_data[r0:r1, c0:c1]
        clamday_patch = self._resize_to_standard(clamday_patch, self.P, self.P)
        if np.any(clamday_patch == CLAMDAY_INVALID) or np.any(np.isnan(clamday_patch)):
            if strict:
                return None
            else:
                clamday_patch = self._interpolate_nan_patch(clamday_patch)
                has_invalid = True
        conv_features.append(clamday_patch)

        # 3. DEM 波段
        for dem_band in self.dem_data:
            dem_patch = dem_band[r0:r1, c0:c1]
            dem_patch = self._resize_to_standard(dem_patch, self.P, self.P)
            if np.any(np.isnan(dem_patch)):
                if strict:
                    return None
                else:
                    dem_patch = self._interpolate_nan_patch(dem_patch)
                    has_invalid = True
            conv_features.append(dem_patch)

        try:
            conv_patch = np.stack(conv_features, axis=0)
            if has_invalid and not strict:
                # 最后再检查一次NaN
                if np.any(np.isnan(conv_patch)):
                    conv_patch = np.nan_to_num(conv_patch, nan=0.0)
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

    def _get_smap_value(self, date_dt: datetime, r: int, c: int) -> Tuple[float, float]:
        """
        获取指定日期和位置的SMAP亮温值
        返回: (tbv, tbh) 亮温值，无效时返回 NODATA 值
        注意：数据已是逐日格式，不需要时间插值
        """
        tbv_value = self.smap_nodata_value
        tbh_value = self.smap_nodata_value

        # 如果没有该日期的数据，返回 NODATA
        if date_dt not in self.smap_data:
            return tbv_value, tbh_value

        data = self.smap_data[date_dt]

        # ========== V 极化 ==========
        if 'TBV' in data and data['TBV'] is not None:
            val = data['TBV'][r, c]
            mask_v = data.get('mask_V')

            # 有效性判断：mask != 0 且 val 有效且 finite
            is_valid = (val != self.smap_nodata_value) and np.isfinite(val)
            if mask_v is not None:
                is_valid = is_valid and (mask_v[r, c] != 0)

            if is_valid:
                tbv_value = float(val)

        # ========== H 极化 ==========
        if 'TBH' in data and data['TBH'] is not None:
            val = data['TBH'][r, c]
            mask_h = data.get('mask_H')

            # 有效性判断：mask != 0 且 val 有效且 finite
            is_valid = (val != self.smap_nodata_value) and np.isfinite(val)
            if mask_h is not None:
                is_valid = is_valid and (mask_h[r, c] != 0)

            if is_valid:
                tbh_value = float(val)

        return tbv_value, tbh_value
    
    
    def _get_smap_mask(self, date_dt: datetime, r: int, c: int, pol: str = 'V') -> float:
        """
        获取 SMAP mask 值（用于特征输入）
        返回 mask 值（0=无效，>0=有效）
        """
        if date_dt not in self.smap_data:
            return 0.0

        data = self.smap_data[date_dt]
        mask_key = f'mask_{pol}'

        if mask_key in data and data[mask_key] is not None:
            mask_val = data[mask_key][r, c]
            # 检查对应的亮温值是否也是有效
            tb_key = f'TB{pol}'
            if tb_key in data and data[tb_key] is not None:
                tb_val = data[tb_key][r, c]
                # 只有亮温值也有效时，才返回 mask 值
                if tb_val != self.smap_nodata_value and np.isfinite(tb_val):
                    return float(mask_val)
            return 0.0

        return 0.0

    def _build_point_features(self, date_dt: datetime, r: int, c: int) -> np.ndarray:
        """构建点特征 - 预训练数据（18维，包含哨兵1完整特征 + SMAP mask）"""

        point_features = []

        # ============ 1. LS特征 (6个) - 根据年份选择 ============
        year = date_dt.year
        if hasattr(self, 'ls_data') and isinstance(self.ls_data, dict) and year in self.ls_data:
            ls_arr = self.ls_data[year]
        elif hasattr(self, 'ls_data_default'):
            ls_arr = self.ls_data_default
        else:
            ls_arr = self.ls_data

        # 🔥 LS 有效性检查
        for i in range(ls_arr.shape[0]):
            val = ls_arr[i, r, c]
            if (not np.isfinite(val)) or val == -9999.0:
                return None  # LS 无效，跳过该样本
            point_features.append(float(val))

        # ============ 2. 哨兵1特征 (5个: VV, VH, VV_cov, VH_cov, angle) ============
        s1_vv, s1_vh, s1_vv_cov, s1_vh_cov, s1_angle = self._get_sentinel1_value(date_dt, r, c)

        # 🔥 哨兵1有效性检查：两个极化都必须有效
        vv_valid = (s1_vv != self.s1_nodata_value) and np.isfinite(s1_vv) and (s1_vv_cov > 0)
        vh_valid = (s1_vh != self.s1_nodata_value) and np.isfinite(s1_vh) and (s1_vh_cov > 0)

        if not (vv_valid and vh_valid):
            return None  # 任何一个极化无效，跳过该样本

        # VV
        point_features.append(float(s1_vv))

        # VH
        point_features.append(float(s1_vh))

        # VV_cov - 🔥 只添加 >=0 的有效值，否则填 0
        if s1_vv_cov >= 0 and np.isfinite(s1_vv_cov):
            point_features.append(float(s1_vv_cov))
        else:
            point_features.append(0.0)

        # VH_cov - 🔥 只添加 >=0 的有效值，否则填 0
        if s1_vh_cov >= 0 and np.isfinite(s1_vh_cov):
            point_features.append(float(s1_vh_cov))
        else:
            point_features.append(0.0)

        # angle - 🔥 过滤无效值，有效时用原始值，无效时填 0
        if s1_angle != self.s1_nodata_value and np.isfinite(s1_angle):
            has_cov = (s1_vv_cov > 0) or (s1_vh_cov > 0)
            if has_cov:
                point_features.append(float(s1_angle))
            else:
                point_features.append(0.0)
        else:
            point_features.append(0.0)

        # ============ 3. SMAP亮温特征 (2个) ============
        smap_tbv, smap_tbh = self._get_smap_value(date_dt, r, c)

        # 🔥 SMAP 有效性检查：两个极化都必须有效
        if smap_tbv == self.smap_nodata_value or smap_tbh == self.smap_nodata_value:
            return None  # 任何一个极化无效，跳过该样本

        point_features.append(float(smap_tbv))
        point_features.append(float(smap_tbh))

        # ============ 4. SMAP mask特征 (2个) ============
        mask_v = self._get_smap_mask(date_dt, r, c, 'V')
        mask_h = self._get_smap_mask(date_dt, r, c, 'H')
        point_features.append(float(mask_v))
        point_features.append(float(mask_h))

        # ============ 5. 经纬度特征 (2个) - 总是有效 ============
        lon, lat = self._pixel_to_lonlat(r, c)
        lon_norm = (lon + 180) / 360
        lat_norm = (lat + 90) / 180
        point_features.extend([lon_norm, lat_norm])

        # ============ 6. 时间特征 (1个) - 总是有效 ============
        time_feats = self._build_time_features(date_dt)
        point_features.extend(time_feats)

        point_feats_array = np.array(point_features, dtype=np.float32)
        # 处理NaN值
        if np.any(np.isnan(point_feats_array)):
            point_feats_array = np.nan_to_num(point_feats_array, nan=0.0)

        # 维度防御检查
        if len(point_features) != 18:
            raise ValueError(f"预训练维度错误: 期望 18，实际 {len(point_features)}")

        return point_feats_array

    def _pixel_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        """将像素坐标转换为经纬度"""
        x, y = self.transform * (col + 0.5, row + 0.5)
        lon, lat = self.transformer.transform(x, y)
        return lon, lat

    def __len__(self):
        return len(self.meta_index)


    def __getitem__(self, idx: int):
        """获取一个样本 - 根据来源选择严格/宽松特征构建"""

        # ============ 🔥 优先从内存缓存读取 ============
        # 只有在缓存已完整构建时才使用
        if hasattr(self, '_cached_conv') and self._cached_conv is not None:
            # 🔥 检查缓存长度是否与 meta_index 一致
            if len(self._cached_conv) == len(self.meta_index):
                return (
                    self._cached_conv[idx],
                    self._cached_point[idx],
                    self._cached_target[idx],
                    self._cached_mask[idx],
                    self._cached_grid[idx],
                    idx
                )

        # ============ 缓存不存在或未完整时，正常处理 ============
        max_retry = 50
        cur_idx = idx

        for retry in range(max_retry):
            # 🔥 获取样本信息（兼容新旧格式）
            item = self.meta_index[cur_idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                # 兼容旧缓存（没有source字段，默认为随机采样）
                date_dt, r, c = item
                source = 'random'

            # 🔥 根据来源选择特征构建函数
            if source in ('station', 'quota_supplement'):
                # 站点样本：使用宽松版
                conv_patch = self._build_spatial_features_station(date_dt, r, c)
                point_feats = self._build_point_features_station(date_dt, r, c)
            else:
                # 随机样本：使用严格版
                conv_patch = self._build_spatial_features(date_dt, r, c)
                point_feats = self._build_point_features(date_dt, r, c)

            if conv_patch is None:
                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

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

            # 🔥 可选：对站点样本增加额外的质量检查（避免太差的样本）
            if source in ('station', 'quota_supplement'):
                # 检查点特征是否有基本有效值（至少2个LS波段有值）
                ls_bands = point_feats[:6]
                if np.sum(np.abs(ls_bands) > 0.01) < 2:
                    cur_idx = (cur_idx + 1) % len(self.meta_index)
                    continue

            # 转换为torch张量
            conv_t = torch.from_numpy(conv_patch)
            point_t = torch.from_numpy(point_feats)
            y_t = torch.tensor(y, dtype=torch.float32)

            # 标记原始target是否为0
            is_zero = 1 if y > 0 else 0
            is_zero_t = torch.tensor(is_zero, dtype=torch.float32)

            # 数据完整性检查（仅第一次，且只对随机样本详细打印）
            if not hasattr(self, '_data_checked'):
                print(f"\n【样本 {idx} 数据检查】来源: {source}")
                print(f"  卷积特征:")
                print(f"    shape: {conv_t.shape}")
                print(f"    dtype: {conv_t.dtype}")
                print(f"    range: [{conv_t.min():.4f}, {conv_t.max():.4f}]")
                print(f"    mean: {conv_t.mean():.4f} ± {conv_t.std():.4f}")
                print(f"    has nan: {torch.isnan(conv_t).any()}")
                print(f"    has inf: {torch.isinf(conv_t).any()}")

                print(f"\n  点特征:")
                print(f"    shape: {point_t.shape}")
                print(f"    dtype: {point_t.dtype}")
                print(f"    range: [{point_t.min():.4f}, {point_t.max():.4f}]")
                print(f"    mean: {point_t.mean():.4f} ± {point_t.std():.4f}")
                print(f"    has nan: {torch.isnan(point_t).any()}")
                print(f"    has inf: {torch.isinf(point_t).any()}")

                print(f"\n  目标值: {y_t.item():.4f}")
                print(f"  is_zero: {is_zero_t.item()}")
                self._data_checked = True

            # 固定归一化：正式渐进式流程使用clip+p01/p99后的z-score。
            eps = 1e-6
            if getattr(self, "normalization_method", "minmax") == "clip_then_zscore":
                conv_low = torch.from_numpy(self.conv_clip_low).view(-1, 1, 1)
                conv_high = torch.from_numpy(self.conv_clip_high).view(-1, 1, 1)
                conv_mean = torch.from_numpy(self.conv_mean).view(-1, 1, 1)
                conv_std = torch.from_numpy(self.conv_std).view(-1, 1, 1)
                conv_t = torch.clamp(conv_t, min=conv_low, max=conv_high)
                conv_t = (conv_t - conv_mean) / torch.clamp(conv_std, min=eps)

                raw_point = point_t.clone()
                valid_point = torch.ones(self.C_point, dtype=torch.bool)
                if self.C_point >= 15:
                    valid_point[0:6] = raw_point[0:6] != 0.0
                    valid_point[6] = raw_point[8] > 0.0
                    valid_point[7] = raw_point[9] > 0.0
                    valid_point[10] = (raw_point[8] > 0.0) | (raw_point[9] > 0.0)
                    valid_point[11] = raw_point[13] > 0.0
                    valid_point[12] = raw_point[14] > 0.0
                point_low = torch.from_numpy(self.point_clip_low)
                point_high = torch.from_numpy(self.point_clip_high)
                point_mean = torch.from_numpy(self.point_mean)
                point_std = torch.from_numpy(self.point_std)
                for i, transform_name in enumerate(self.point_transform):
                    if transform_name == "zscore":
                        if valid_point[i]:
                            val = torch.clamp(raw_point[i], point_low[i], point_high[i])
                            point_t[i] = (val - point_mean[i]) / torch.clamp(point_std[i], min=eps)
                        else:
                            point_t[i] = 0.0
                    else:
                        point_t[i] = raw_point[i]
            else:
                conv_t = (conv_t - torch.from_numpy(self.conv_min).view(-1, 1, 1)) / \
                         (torch.from_numpy(self.conv_max + eps).view(-1, 1, 1) - torch.from_numpy(self.conv_min).view(-1, 1, 1))
                point_t = (point_t - torch.from_numpy(self.point_min)) / \
                          (torch.from_numpy(self.point_max + eps) - torch.from_numpy(self.point_min))

            y_t = (y_t - self.label_min) / (self.label_max - self.label_min)

            # 维度检查
            expected_dim = 18
            if point_t.shape[0] != expected_dim:
                print("\n" + "!"*60)
                print(f"【维度报警】发现 {point_t.shape[0]} 维样本！（期望 {expected_dim} 维）")
                print(f"数据来源: {source}")
                print(f"当前样本索引: {idx}")
                print("!"*60 + "\n")
                import sys
                sys.exit(1)

            # 预训练数据产品值设为0
            grid_val_t = torch.tensor(0.0, dtype=torch.float32)

            # 返回: 卷积, 点特征, 标签, 零掩码, 产品值, 索引
            return conv_t, point_t, y_t, is_zero_t, grid_val_t, int(cur_idx)

        # 重试失败，打印诊断信息
        print(f"\n❌ 错误: 在idx={idx}附近连续{max_retry}个样本均无效")
        print(f"   meta_index长度: {len(self.meta_index)}")
        print(f"   请检查数据质量或降低采样密度")
        raise IndexError(f"在idx={idx}附近连续{max_retry}个样本均无效")
        
    def _validate_station_sample_with_reason(self, date_dt: datetime, r: int, c: int):
        """返回验证结果、失败原因、卷积特征和点特征（用于统计分析）"""

        original_fine_tune_mode = self.fine_tune_mode
        self.fine_tune_mode = True

        try:
            # 1. 卷积特征（宽松版）
            conv_patch = self._build_spatial_features_station(date_dt, r, c)
            if conv_patch is None:
                return False, 'conv', None, None

            # 2. 点特征（宽松版）
            point_feats = self._build_point_features_station(date_dt, r, c)
            if point_feats is None:
                return False, 'point', None, None

            # 3. 标签检查
            if date_dt not in self.label_data:
                return False, 'label', None, None

            label_arr, label_nodata = self.label_data[date_dt]
            y = label_arr[r, c]
            if (label_nodata is not None and y == label_nodata) or np.isnan(y):
                return False, 'label', None, None

            return True, 'success', conv_patch, point_feats

        finally:
            self.fine_tune_mode = original_fine_tune_mode


def build_dataloaders(
        batch_size: int = 32,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
        prefetch_factor: int = 2,  # 添加这个参数但不会传给dataset
        persistent_workers: bool = True,  # 添加这个参数
        **dataset_kwargs
):
    """构建数据加载器"""
    try:
        # 从dataset_kwargs中移除DataLoader特有的参数
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': persistent_workers,
            'pin_memory': True,
        }
        
        # 创建数据集（只传dataset需要的参数）
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

        # 创建数据加载器（使用dataloader_kwargs）
        train_loader = DataLoader(
            train_set,
            shuffle=True,
            **dataloader_kwargs,
        )
        val_loader = DataLoader(
            val_set,
            shuffle=False,
            **dataloader_kwargs,
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
            # 🔥 修复：兼容新旧格式
            item = dataset.meta_index[idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                date_dt, r, c = item
            
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
            # 🔥 修复：兼容新旧格式
            item = dataset.meta_index[idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                date_dt, r, c = item
            
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

        
def build_spatial_grid_cv_indices(
    dataset, 
    lon_step: float = 1.0, 
    lat_step: float = 1.0, 
    n_splits: int = 10, 
    seed: int = 42,
    min_samples_per_grid: int = 10
) -> List[Tuple[List[int], List[int]]]:
    """
    按经纬度网格分配样本索引，支持十折交叉验证
    
    Args:
        dataset: SWEDataset 实例
        lon_step: 经度网格步长（度），建议 1.0（约100km）
        lat_step: 纬度网格步长（度），建议 1.0
        n_splits: 折数
        seed: 随机种子
        min_samples_per_grid: 每个网格最少样本数，少于则合并到相邻网格
    
    Returns:
        list of (train_indices, val_indices)
    """
    print(f"\n{'='*70}")
    print(f"🌐 空间网格化十折划分 (Spatial Grid CV)")
    print(f"   网格大小: Lon={lon_step}°, Lat={lat_step}°")
    print(f"   折数: {n_splits}")
    print(f"   最小网格样本数: {min_samples_per_grid}")
    print(f"{'='*70}")
    
    # 1. 将样本归类到网格
    grid_to_indices = defaultdict(list)
    grid_to_centroids = {}
    
    print("  正在将样本分配到空间网格...")
    
    for idx in range(len(dataset)):
        # 🔥 修复：兼容新旧格式
        item = dataset.meta_index[idx]
        if len(item) == 4:
            date, r, c, source = item
        else:
            date, r, c = item
        
        lon, lat = dataset._pixel_to_lonlat(r, c)
        
        # 计算网格坐标 (使用 floor 确保连续性，处理负经度/纬度)
        gi = int(np.floor(lon / lon_step) if lon >= 0 else np.floor(lon / lon_step) - 1)
        gj = int(np.floor(lat / lat_step) if lat >= 0 else np.floor(lat / lat_step) - 1)
        
        grid_to_indices[(gi, gj)].append(idx)
        # 记录网格中心（用于调试）
        if (gi, gj) not in grid_to_centroids:
            grid_to_centroids[(gi, gj)] = (lon, lat)
    
    # 2. 过滤掉样本数过少的网格（合并到最近邻）
    if min_samples_per_grid > 0:
        print(f"  正在合并小样本网格 (<{min_samples_per_grid} 个样本)...")
        
        small_grids = []
        large_grids = []
        
        for grid, indices in grid_to_indices.items():
            if len(indices) < min_samples_per_grid:
                small_grids.append(grid)
            else:
                large_grids.append(grid)
        
        if small_grids:
            print(f"    发现 {len(small_grids)} 个小样本网格")
            
            # 将小网格合并到最近的相邻网格
            for grid in small_grids:
                # 计算网格中心
                center_lon, center_lat = grid_to_centroids[grid]
                min_dist = float('inf')
                nearest_large = None
                
                for large_grid in large_grids:
                    large_lon, large_lat = grid_to_centroids[large_grid]
                    dist = np.sqrt((center_lon - large_lon)**2 + (center_lat - center_lat)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_large = large_grid
                
                if nearest_large is not None:
                    grid_to_indices[nearest_large].extend(grid_to_indices[grid])
                    del grid_to_indices[grid]
        
        # 更新唯一网格列表
        unique_grids = list(grid_to_indices.keys())
        print(f"    合并后剩余 {len(unique_grids)} 个有效网格")
    else:
        unique_grids = list(grid_to_indices.keys())
    
    n_grids = len(unique_grids)
    print(f"\n📊 网格统计:")
    print(f"   总网格数: {n_grids}")
    
    # 统计网格样本数分布
    grid_sizes = [len(grid_to_indices[g]) for g in unique_grids]
    print(f"   每网格样本数: min={min(grid_sizes)}, max={max(grid_sizes)}, "
          f"mean={np.mean(grid_sizes):.1f}, median={np.median(grid_sizes):.1f}")
    
    # 3. 对“网格”进行 K-Fold 划分
    print(f"\n🎲 对 {n_grids} 个网格进行 {n_splits} 折划分 (seed={seed})...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = []
    
    # 统计每折的样本分布
    fold_stats = []
    
    for fold, (train_grid_idx, val_grid_idx) in enumerate(kf.split(unique_grids)):
        train_indices = []
        val_indices = []
        
        # 收集训练集样本
        for g_idx in train_grid_idx:
            train_indices.extend(grid_to_indices[unique_grids[g_idx]])
        
        # 收集验证集样本
        for g_idx in val_grid_idx:
            val_indices.extend(grid_to_indices[unique_grids[g_idx]])
        
        fold_indices.append((train_indices, val_indices))
        
        # 记录统计信息
        fold_stats.append({
            'fold': fold + 1,
            'n_train_grids': len(train_grid_idx),
            'n_val_grids': len(val_grid_idx),
            'n_train_samples': len(train_indices),
            'n_val_samples': len(val_indices),
        })
    
    # 4. 打印划分统计
    print(f"\n📊 十折划分统计:")
    print(f"{'折数':<6} {'训练网格':<10} {'验证网格':<10} {'训练样本':<12} {'验证样本':<12}")
    print("-" * 55)
    for stat in fold_stats:
        print(f"{stat['fold']:<6} {stat['n_train_grids']:<10} {stat['n_val_grids']:<10} "
              f"{stat['n_train_samples']:<12,} {stat['n_val_samples']:<12,}")
    
    # 5. 验证空间隔离性
    print(f"\n🔍 验证空间隔离性...")
    
    # 检查训练集和验证集是否有相同网格（理论上不应该有）
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        train_grids = set()
        val_grids = set()
        
        # 获取训练集所在的网格
        for idx in train_idx[:1000]:  # 采样检查，避免过慢
            # 🔥 修复：兼容新旧格式
            item = dataset.meta_index[idx]
            if len(item) == 4:
                date, r, c, source = item
            else:
                date, r, c = item
            
            lon, lat = dataset._pixel_to_lonlat(r, c)
            gi = int(np.floor(lon / lon_step))
            gj = int(np.floor(lat / lat_step))
            train_grids.add((gi, gj))
        
        # 获取验证集所在的网格
        for idx in val_idx[:1000]:
            # 🔥 修复：兼容新旧格式
            item = dataset.meta_index[idx]
            if len(item) == 4:
                date, r, c, source = item
            else:
                date, r, c = item
            
            lon, lat = dataset._pixel_to_lonlat(r, c)
            gi = int(np.floor(lon / lon_step))
            gj = int(np.floor(lat / lat_step))
            val_grids.add((gi, gj))
        
        overlap = train_grids & val_grids
        if overlap:
            print(f"  ⚠️ Fold {fold+1}: 发现 {len(overlap)} 个重叠网格（违反空间隔离）")
        else:
            print(f"  ✅ Fold {fold+1}: 训练集与验证集网格完全分离")
    
    print(f"\n✅ 空间网格划分完成!")
    
    return fold_indices
        
        
        
        
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
            # 🔥 修复：兼容新旧格式
            item = dataset.meta_index[idx]
            if len(item) == 4:
                date_dt, r, c, source = item
            else:
                date_dt, r, c = item
            
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
                    # 🔥 修复：接收6个返回值
                    conv, point, y, is_zero, grid_val, idx = dataset[i]
                    print(f"    conv shape: {conv.shape}")
                    print(f"    point shape: {point.shape}")
                    print(f"    y value: {y.item():.4f}")
                    print(f"    is_zero: {is_zero.item()}")
                    print(f"    grid_val: {grid_val.item()}")
                    print(f"    idx: {idx}")
                    
                    # 点特征维度检查
                    print(f"    point 维度: {point.shape[0]}")
                    
                    # 打印部分点特征值（前几个）
                    print(f"    point 前5个值: {point[:5].tolist()}")
                    
                except Exception as e:
                    print(f"    获取样本{i}失败: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("  数据集为空!")

        # 测试数据加载器
        print(f"\n3. 测试数据加载器...")
        train_loader, val_loader, shapes = build_dataloaders(batch_size=4, val_ratio=0.2)

        print(f"\n4. 测试批次加载...")
        if train_loader:
            batch = next(iter(train_loader))
            print(f"   批次大小: {len(batch)}")
            # 🔥 修复：batch 现在有6个元素
            conv_batch, point_batch, target_batch, is_zero_batch, grid_val_batch, idx_batch = batch
            print(f"   卷积特征批次形状: {conv_batch.shape}")
            print(f"   点特征批次形状: {point_batch.shape}")
            print(f"   目标批次形状: {target_batch.shape}")
            print(f"   is_zero批次形状: {is_zero_batch.shape}")
            print(f"   grid_val批次形状: {grid_val_batch.shape}")
            print(f"   idx批次形状: {idx_batch.shape}")

        print(f"\n✓ 数据加载测试完成!")

    except Exception as e:
        print(f"\n✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
