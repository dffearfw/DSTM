# -*- coding: utf-8 -*-
"""
第二阶段：站点SWE在线构建样本 Dataset + 按站点划分 DataLoader
用于站点实测SWE数据的微调训练

改进内容：
1. 使用与预训练相同的参考网格（标签数据网格）
2. 修复日期索引问题
3. 支持边界样本处理
4. 确保6个卷积特征一致性
5. 完整的点特征构建
6. 添加混合模式支持（站点+预训练样本）
"""

import json
import os
import calendar
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime, timedelta
from pyproj import Transformer
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import re
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# ============= 配置 =============

REGION = "CHINA"
YEAR_TARGET = 2016
PATCH_SIZE = 5
R = PATCH_SIZE // 2

# 全局无效值常量
FINAL_NODATA = -9999.0

# 与正式预训练 data_online_era5_swe.py 保持一致
FILTER_GLACIER_SWE_ARTIFACTS = os.environ.get(
    "FILTER_GLACIER_SWE_ARTIFACTS", "1"
) == "1"
GLACIER_SWE_THRESHOLD_MM = float(
    os.environ.get("GLACIER_SWE_THRESHOLD_MM", "2000.0")
)
STATION_FEATURE_LOADER_VERSION = "full_daily_wind_only_fill_2015_2018_v4"
DYNAMIC_TEMPORAL_FILL_MAX_GAP_DAYS = 7

# 特征配置（与第一阶段一致）
CONV_VARS = ["chelsa_sfxwind", "lst", "rh", "pr"]   # 动态卷积特征
CONV_STATIC_VARS = ["clamday", "dem"]  # 静态卷积特征
POINT_VARS = ["ls", "S1_VV", "S1_VH", "SMAP_TBV", "SMAP_TBH"]  # 点特征

# 数据路径
FEATURE_ROOT = Path(r"/root/ablation")
LABEL_ROOT = FEATURE_ROOT / "era5landswe"  # 用于获取参考网格

# 站点SWE数据路径
STATION_SWE_CSV = Path(r"/root/ablation/station_swe_data.xlsx")

def conv_var_path(var: str, year: int) -> Path:
    """卷积变量路径"""
    if var == "chelsa_sfxwind":
        return FEATURE_ROOT / "sfxwind" / "cn"
    elif var == "lst":
        return FEATURE_ROOT / "lst" / "cn"
    elif var == "rh":
        return FEATURE_ROOT / "rh" / "cn"
    elif var == "pr":
        return FEATURE_ROOT / "pr" / "cn"
    else:
        raise ValueError(f"未知的卷积变量: {var}")


def _as_year_list(year):
    """把 year / [years] 统一成 list[int]。"""
    if isinstance(year, (list, tuple, set)):
        return [int(y) for y in year]
    return [int(year)]




# STATION_THRESHOLD_TOKENS_PARITY_V1
def _threshold_tokens(threshold: float) -> List[str]:
    """同时兼容 threshold0.5 和 threshold0p5 这两类命名。"""
    if threshold is None:
        return []
    raw = f"{float(threshold):g}"
    return [raw, raw.replace(".", "p")]


def _glob_unique(root: Path, patterns: List[str]) -> List[Path]:
    """按多个 pattern 查找并去重排序。"""
    files = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(files), key=lambda path: path.name)
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
class StationSWEDataset(Dataset):
    """站点SWE数据集 - 改进版"""
    def __init__(
            self,
            station_csv: Path = STATION_SWE_CSV,
            region: str = REGION,
            year_target: Union[int, List[int]] = YEAR_TARGET,
            feature_root: Path = FEATURE_ROOT,
            patch_size: int = PATCH_SIZE,
            clamday_threshold: float = 0.5,
            s1_interp_method: str = "nearest",
            s1_max_gap_days: int = 7,
            s1_nodata_value: float = -9999.0,
            smap_interp_method: str = "nearest",
            smap_max_gap_days: int = 7,
            smap_nodata_value: float = -9999.0,
            fine_tune_mode: bool = False,
            load_fused_swe: bool = True,
            coordinate_jitter_std: float = 0.02,
            microwave_noise_std: float = 0.01,
            coordinate_mask_prob: float = 0.2,
            use_tta: bool = False,
            cache_dir: Optional[Path] = None,
            split_cache_file: str = None,
            force_recompute_split: bool = False,
            # 🔥 新增：共享缓存模式（用于十折CV）
            shared_cache_mode: bool = False,
            # 🔥 产品值修正开关（默认关闭，避免评估泄漏）
            use_product_correction: bool = False,
            **kwargs  # 🔥 添加这一行，吸收所有未声明的参数
    ):
        super().__init__()
        self.region = region

        # 处理 year_target
        if isinstance(year_target, list):
            self.year_target = year_target
            self.load_years = year_target
        else:
            self.year_target = year_target
            self.load_years = [year_target, year_target - 1, year_target - 2]

        self.feature_root = feature_root
        self.patch_size = patch_size
        self.P = patch_size
        self.R = patch_size // 2
        self.clamday_threshold = clamday_threshold
        self.fine_tune_mode = fine_tune_mode
        self.load_fused_swe = load_fused_swe

        self.s1_interp_method = s1_interp_method
        self.s1_max_gap_days = s1_max_gap_days
        self.s1_nodata_value = s1_nodata_value

        self.smap_interp_method = smap_interp_method
        self.smap_max_gap_days = smap_max_gap_days
        self.smap_nodata_value = smap_nodata_value

        self.coordinate_jitter_std = coordinate_jitter_std
        self.microwave_noise_std = microwave_noise_std
        self.coordinate_mask_prob = coordinate_mask_prob
        self.use_tta = use_tta

        self.tta_num_augmentations = 8
        self.tta_noise_scale = 0.01

        # 🔥 保存划分缓存参数（虽然 StationSWEDataset 本身不使用，但需要存储避免报错）
        self.split_cache_file = split_cache_file
        self.force_recompute_split = force_recompute_split
        self.force_reload = bool(
            kwargs.get("force_reload", False)
        )

        # PROGRESSIVE_FINETUNE_NORMALIZATION_V1
        # M0-M4微调、内部测试和外部测试统一复用预训练归一化。
        normalization_config_path = kwargs.get(
            "normalization_config_path"
        )
        self.normalization_config_path = (
            Path(normalization_config_path).expanduser()
            if normalization_config_path
            else None
        )

        self.normalization_mode = str(
            kwargs.get("normalization_mode", "auto")
        ).strip().lower()

        if self.normalization_mode not in {
            "auto",
            "load",
            "create",
            "skip",
            "legacy",
        }:
            raise ValueError(
                "StationSWEDataset normalization_mode非法: "
                f"{self.normalization_mode!r}"
            )

        self.fixed_label_min_mm = float(
            kwargs.get("fixed_label_min_mm", 0.0)
        )
        self.fixed_label_max_mm = float(
            kwargs.get("fixed_label_max_mm", 400.0)
        )

        if self.fixed_label_max_mm <= self.fixed_label_min_mm:
            raise ValueError(
                "fixed_label_max_mm必须大于fixed_label_min_mm"
            )

        # ============ 数据存储初始化 ============
        self.s1_data = {}
        self.all_s1_dates = []
        self.smap_data = {}
        self.all_smap_dates = []
        self.clamday_data = None
        self.dem_data = []
        self.label_data = {}
        self.current_augment = True

        # ============ 日志控制 ============
        # 默认关闭逐样本点特征/微波警告，避免 DataLoader 多 worker 刷屏
        self.verbose_point_debug = bool(kwargs.get("verbose_point_debug", False))

        # 默认 0：一条微波逐样本警告都不打印
        # 如果以后想调试，可以设成 3
        self.microwave_warning_print_limit = int(kwargs.get("microwave_warning_print_limit", 0))

        # 默认 0：不打印"每100个样本"的点特征统计
        # 如果以后想调试，可以设成 1000
        self.point_stats_interval = int(kwargs.get("point_stats_interval", 0))

        # ============ 🔥 缓存逻辑（支持共享缓存模式） ============
        cache_loaded = False
        if cache_dir is not None:
            import hashlib
            import json
            import pickle

            cache_dir_path = Path(cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)

            # 🔥 计算站点数据文件的内容哈希（不依赖路径）
            def get_file_hash(file_path):
                """计算文件内容的哈希值"""
                hasher = hashlib.md5()
                try:
                    with open(file_path, 'rb') as f:
                        # 读取前1MB（足够区分不同数据）
                        chunk = f.read(1024 * 1024)
                        hasher.update(chunk)
                except Exception as e:
                    print(f"  ⚠ 读取文件哈希失败: {e}")
                    hasher.update(str(file_path).encode())
                return hasher.hexdigest()[:16]

            # 🔥 如果启用共享缓存模式，使用固定 key
            if shared_cache_mode:
                cache_key = f"shared_station_features_{STATION_FEATURE_LOADER_VERSION}"
                print(f"📦 共享缓存模式: 使用固定缓存 key={cache_key}")
            else:
                # 正常模式：基于数据内容生成 hash
                try:
                    file_size = Path(station_csv).stat().st_size
                except:
                    file_size = 0

                cache_params = {
                    'region': region,
                    'feature_loader_version': STATION_FEATURE_LOADER_VERSION,
                    'year_target': year_target,
                    'load_years': self.load_years,
                    'patch_size': patch_size,
                    'clamday_threshold': clamday_threshold,
                    'fine_tune_mode': fine_tune_mode,
                    'load_fused_swe': load_fused_swe,
                    'station_data_hash': get_file_hash(station_csv),
                    'station_data_size': file_size,
                }

                cache_str = json.dumps(cache_params, sort_keys=True, default=str)
                cache_key = hashlib.md5(cache_str.encode()).hexdigest()[:16]

            cache_path = cache_dir_path / f"station_dataset_features_{cache_key}.pkl"

            print(f"📦 缓存Key: {cache_key}")
            print(f"   缓存文件: {cache_path}")

            # 尝试加载特征缓存
            if not hasattr(self, 'force_reload') or not self.force_reload:
                if cache_path.exists():
                    print(f"\n📦 发现站点特征缓存: {cache_path}")
                    print("   正在加载特征数据...")
                    try:
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)

                        # 恢复特征数据
                        for key, value in cached_data.items():
                            setattr(self, key, value)

                        print("   ✅ 特征数据加载成功")

                        # 🔥 如果 ls_data_default 缺失但 ls_data 存在，则重建
                        if not hasattr(self, 'ls_data_default') and hasattr(self, 'ls_data'):
                            if isinstance(self.ls_data, dict) and self.ls_data:
                                first_year = list(self.ls_data.keys())[0]
                                self.ls_data_default = self.ls_data[first_year]
                                print(f"   ✅ 重建 ls_data_default (年份: {first_year})")
                            else:
                                self.ls_data_default = self.ls_data

                        # 重建不可序列化的对象
                        self._setup_unified_grid()

                        cache_loaded = True

                    except Exception as e:
                        print(f"   ⚠ 缓存加载失败: {e}")
                        import traceback
                        traceback.print_exc()
                        if cache_path.exists():
                            cache_path.unlink()  # 删除损坏的缓存
                            print(f"   已删除损坏的缓存文件")

        # ============ 如果缓存未加载，正常加载特征数据 ============
        if not cache_loaded:
            print(f"\n初始化站点SWE数据集 (加载特征数据):")
            print(f"  模式: {'微调' if fine_tune_mode else '训练'}")
            print(f"  区域: {region}")
            print(f"  目标年份: {self.year_target}")
            print(f"  加载年份: {self.load_years}")
            print(f"  站点数据: {station_csv}")
            print(f"  Patch大小: {patch_size}")

            # 1. 设置统一网格
            self._setup_unified_grid()

            # 2. 加载所有特征数据
            self._load_all_features()

            # 3. 计算卷积通道数
            self.C_conv = len(CONV_VARS) + 1 + len(self.dem_data)

            print(f"\n📊 卷积特征维度统计:")
            print(f"  动态变量: {len(CONV_VARS)}")
            print(f"  静态变量 (Clamday): 1")
            print(f"  DEM波段: {len(self.dem_data)}")
            print(f"  → 总卷积通道数 C_conv = {self.C_conv}")

            # 4. 加载FusedSWE
            if self.load_fused_swe:
                self._load_fused_swe_labels()

            # 5. 保存特征缓存
            if cache_dir is not None:
                self._save_feature_cache(cache_path)

        # 无论现场加载还是缓存恢复，都执行最终特征契约检查。
        required_dynamic = [
            "chelsa_sfxwind", "lst", "rh", "pr"
        ]
        missing_dynamic = [
            name for name in required_dynamic
            if name not in getattr(self, "conv_dyn_data", {})
        ]
        if missing_dynamic:
            raise RuntimeError(
                "缓存/加载结果缺少动态特征: "
                f"{missing_dynamic}"
            )

        self._validate_complete_daily_feature_timeline()

        if not hasattr(self, "ls_data_default"):
            raise RuntimeError(
                "缓存/加载结果缺少ls_data_default"
            )
        if not getattr(self, "all_s1_dates", []):
            raise RuntimeError(
                "缓存/加载结果没有Sentinel-1日期"
            )
        if not getattr(self, "all_smap_dates", []):
            raise RuntimeError(
                "缓存/加载结果没有SMAP日期"
            )
        if not getattr(self, "label_data", {}):
            raise RuntimeError(
                "缓存/加载结果没有ERA5-Land SWE日期"
            )

        self.C_point = 18
        print("\n✅ 站点微调特征契约检查通过")
        print(
            "   动态: "
            + ", ".join(
                f"{name}={self.conv_dyn_data[name].shape}"
                for name in required_dynamic
            )
        )
        print(
            f"   LS={self.ls_data_default.shape}, "
            f"S1日期={len(self.all_s1_dates)}, "
            f"SMAP日期={len(self.all_smap_dates)}, "
            f"SWE日期={len(self.label_data)}, "
            f"C_conv={self.C_conv}, C_point={self.C_point}"
        )

        # ============ 以下步骤每次都要执行（保证随机性） ============
        print(f"\n构建样本索引 (每次运行重新生成，保证随机划分):")

        # 6. 加载站点数据并构建索引（每次都重新生成）
        self._load_station_data(station_csv)

        # 7. 归一化
        # 正式渐进式流程必须加载Stage0-4共用的clip+z-score配置。
        loaded_progressive_norm = (
            self._load_progressive_normalization()
        )

        if not loaded_progressive_norm:
            if self.normalization_mode == "load":
                raise RuntimeError(
                    "normalization_mode=load，"
                    "但统一归一化配置没有成功加载"
                )

            print(
                "   ⚠ 未加载渐进式统一归一化，"
                "使用旧版站点子集min-max"
            )
            self._compute_minmax()
            self.label_min = float(self.swe_min)
            self.label_max = float(self.swe_max)
            self.normalization_method = "minmax"

        # 8. 检查SWE差异
        self._check_swe_discrepancies(threshold=30.0)

        # ============ 🔥 产品值修正数据（默认关闭，避免评估泄漏） ============
        self.correction_map = {}
        self.use_product_correction = use_product_correction
        if self.use_product_correction:
            self._load_corrections()
        else:
            print("   ℹ product correction 已关闭（默认），跳过修正数据加载")

        print(f"\n站点SWE数据集初始化完成!")
        print(f"  总样本数: {len(self.meta_index)}")
        print(f"  站点数: {len(self.station_set)}")
        print(f"  卷积特征维度: {self.C_conv}")
        print(f"  点特征维度: {self.C_point}")
        print(f"  图像尺寸: {self.H}行 × {self.W}列")
        
    def _load_progressive_normalization(self) -> bool:
        """加载M0-M4共用的clip+p01/p99+z-score归一化配置。"""
        path = self.normalization_config_path

        should_load = (
            self.normalization_mode == "load"
            or (
                self.normalization_mode == "auto"
                and path is not None
                and path.exists()
            )
        )

        if not should_load:
            return False

        if path is None:
            raise ValueError(
                "normalization_mode=load时必须提供"
                "normalization_config_path"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"统一归一化配置不存在: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        method = str(
            payload.get("method", "")
        ).strip().lower()

        if method != "clip_then_zscore":
            raise ValueError(
                "渐进式微调要求method=clip_then_zscore，"
                f"当前={method!r}"
            )

        expected_dimensions = {
            "C_conv": int(self.C_conv),
            "C_point": int(self.C_point),
            "patch_size": int(self.patch_size),
        }

        for key, expected in expected_dimensions.items():
            actual = int(payload.get(key, -1))

            if actual != expected:
                raise ValueError(
                    f"统一归一化{key}不一致: "
                    f"config={actual}, dataset={expected}"
                )

        array_keys = [
            "conv_clip_low",
            "conv_clip_high",
            "conv_mean",
            "conv_std",
            "point_clip_low",
            "point_clip_high",
            "point_mean",
            "point_std",
        ]

        missing = [
            key
            for key in array_keys
            if key not in payload
        ]

        if missing:
            raise ValueError(
                f"统一归一化配置缺少字段: {missing}"
            )

        for key in array_keys:
            setattr(
                self,
                key,
                np.asarray(
                    payload[key],
                    dtype=np.float32,
                ),
            )

        self.point_transform = list(
            payload.get(
                "point_transform",
                ["zscore"] * self.C_point,
            )
        )

        if len(self.point_transform) != self.C_point:
            raise ValueError(
                "point_transform长度与C_point不一致"
            )

        self.label_min = float(payload["label_min"])
        self.label_max = float(payload["label_max"])

        if (
            abs(
                self.label_min
                - self.fixed_label_min_mm
            )
            > 1e-6
        ):
            raise ValueError(
                "统一标签下限不一致: "
                f"config={self.label_min}, "
                f"requested={self.fixed_label_min_mm}"
            )

        if (
            abs(
                self.label_max
                - self.fixed_label_max_mm
            )
            > 1e-6
        ):
            raise ValueError(
                "统一标签上限不一致: "
                f"config={self.label_max}, "
                f"requested={self.fixed_label_max_mm}"
            )

        self.swe_min = self.label_min
        self.swe_max = self.label_max
        self.normalization_method = "clip_then_zscore"

        print(
            f"   ✅ 已加载渐进式统一归一化: {path}"
        )
        print(
            "      method: clip_then_zscore"
        )
        print(
            f"      C_conv={self.C_conv}, "
            f"C_point={self.C_point}, "
            f"patch={self.patch_size}"
        )
        print(
            f"      SWE范围: "
            f"[{self.label_min}, {self.label_max}] mm"
        )

        return True

    def _save_feature_cache(self, cache_path: Path):
        """保存特征数据缓存（只保存已存在的属性）"""
        print(f"\n💾 保存特征缓存到: {cache_path}")

        to_save = {}

        # 核心属性（应该都存在）
        core_attrs = [
            'H', 'W', 'common_bounds', 'transform', 'crs_proj',
            'conv_dyn_data', 'clamday_data', 'dem_data',
            'label_data', 'all_dates', 'date_to_index',
            'C_conv', 'C_point'
        ]

        for attr in core_attrs:
            if hasattr(self, attr):
                to_save[attr] = getattr(self, attr)
            else:
                print(f"   ⚠ 警告: 属性 {attr} 不存在，跳过")

        # 可选属性（可能不存在）
        optional_attrs = ['ls_data', 'ls_data_default', 's1_data', 'smap_data', 'all_s1_dates', 'all_smap_dates', 'dynamic_time_fill_report']
        for attr in optional_attrs:
            if hasattr(self, attr):
                to_save[attr] = getattr(self, attr)

        try:
            import pickle
            # ATOMIC_STATION_CACHE_SAVE_V1
            # 先写临时文件；完整成功后再替换正式缓存。
            tmp_path = cache_path.with_name(cache_path.name + ".tmp")
            try:
                import os
                with open(tmp_path, "wb") as f:
                    pickle.dump(
                        to_save,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    f.flush()
                    os.fsync(f.fileno())
            
                os.replace(tmp_path, cache_path)
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                raise
            file_size = cache_path.stat().st_size / 1024 / 1024
            print(f"   ✅ 特征缓存保存成功! 大小: {file_size:.2f} MB")
            print(f"   保存的属性: {list(to_save.keys())}")
        except Exception as e:
            print(f"   ⚠ 特征缓存保存失败: {e}")
        
    def set_augmentation_mode(self, is_train: bool = True):
        """
        设置是否启用数据增强
        
        Args:
            is_train: True 表示训练模式（启用增强），False 表示验证模式（禁用增强）
        """
        self.current_augment = is_train
        mode = "训练" if is_train else "验证"
        print(f"  [StationSWEDataset] 切换到{mode}模式 - 数据增强={'启用' if is_train else '禁用'}")
    
    def _setup_unified_grid(self):
        """设置统一网格 - 使用与预训练相同的标签数据网格"""
        print(f"\n设置统一网格...")
        
        label_files = list(LABEL_ROOT.glob("*.tif"))
        if not label_files:
            raise FileNotFoundError(f"找不到标签文件: {LABEL_ROOT}")
        
        with rasterio.open(label_files[0]) as ds:
            self.common_bounds = ds.bounds
            self.transform = ds.transform
            self.crs_proj = ds.crs.to_string()
            self.H, self.W = ds.shape
        
        print(f"参考网格（使用标签数据）:")
        print(f"  范围: {self.common_bounds}")
        print(f"  尺寸: {self.H}行 × {self.W}列")
        print(f"  分辨率: {abs(self.transform.a):.3f}° × {abs(self.transform.e):.3f}°")
        
        self.transformer = Transformer.from_crs(self.crs_proj, "EPSG:4326", always_xy=True)
    
    def _filter_glacier_swe_artifacts(
        self,
        label_arr,
        label_nodata=None,
        date_dt=None,
    ):
        """与正式预训练一致地过滤ERA5-Land冰川占位型极端SWE。"""
        if not FILTER_GLACIER_SWE_ARTIFACTS:
            return label_arr

        arr = label_arr.astype(np.float32, copy=True)
        valid_mask = np.isfinite(arr)
        if label_nodata is not None:
            valid_mask &= arr != label_nodata

        artifact_mask = (
            valid_mask
            & (arr >= GLACIER_SWE_THRESHOLD_MM)
        )
        n_bad = int(np.count_nonzero(artifact_mask))
        if n_bad:
            date_text = (
                date_dt.strftime("%Y-%m-%d")
                if date_dt is not None
                else "unknown-date"
            )
            print(
                f"    冰川/极端SWE过滤: {date_text}, "
                f"threshold>={GLACIER_SWE_THRESHOLD_MM:.1f}, "
                f"n={n_bad}"
            )
            arr[artifact_mask] = np.nan
        return arr

    def _load_fused_swe_labels(self):
        """加载ERA5-Land SWE；支持月度daily-band立方体与旧版日文件。"""
        print(f"\n加载ERA5-Land SWE/FusedSWE数据...")

        self.label_data = {}
        target_years = [int(year) for year in self.load_years]
        label_root = self.feature_root / "era5landswe"

        if not label_root.exists():
            raise FileNotFoundError(
                f"ERA5-Land SWE目录不存在: {label_root}"
            )

        label_files = sorted(label_root.glob("*.tif"))
        print(f"  目标年份: {target_years}")
        print(f"  标签目录: {label_root}")
        print(f"  找到 {len(label_files)} 个标签文件")

        loaded_count = 0
        year_count = {}

        for label_file in label_files:
            try:
                name = label_file.stem
                monthly_match = re.search(
                    r"ERA5LAND_SWE_DAILY_AGGR_(\d{4})(\d{2})",
                    name,
                    re.IGNORECASE,
                )

                with rasterio.open(label_file) as ds:
                    label_nodata = ds.nodata
                    src_transform = ds.transform

                    if monthly_match:
                        year = int(monthly_match.group(1))
                        month = int(monthly_match.group(2))
                        if year not in target_years:
                            continue

                        month_days = calendar.monthrange(
                            year, month
                        )[1]
                        n_bands = min(ds.count, month_days)

                        for day in range(1, n_bands + 1):
                            date_dt = datetime(
                                year, month, day
                            )
                            label_arr = ds.read(
                                day
                            ).astype(np.float32)

                            if label_arr.shape != (
                                self.H,
                                self.W,
                            ):
                                label_arr = self._align_single_layer(
                                    label_arr,
                                    src_transform,
                                    self.transform,
                                    self.H,
                                    self.W,
                                )

                            label_arr = (
                                self._filter_glacier_swe_artifacts(
                                    label_arr,
                                    label_nodata=label_nodata,
                                    date_dt=date_dt,
                                )
                            )
                            self.label_data[date_dt] = (
                                label_arr,
                                label_nodata,
                            )
                            loaded_count += 1
                            year_count[year] = (
                                year_count.get(year, 0) + 1
                            )
                    else:
                        date_dt = self._parse_date_from_filename(
                            name
                        )
                        if date_dt.year not in target_years:
                            continue

                        label_arr = ds.read(1).astype(np.float32)
                        if label_arr.shape != (
                            self.H,
                            self.W,
                        ):
                            label_arr = self._align_single_layer(
                                label_arr,
                                src_transform,
                                self.transform,
                                self.H,
                                self.W,
                            )

                        label_arr = (
                            self._filter_glacier_swe_artifacts(
                                label_arr,
                                label_nodata=label_nodata,
                                date_dt=date_dt,
                            )
                        )
                        self.label_data[date_dt] = (
                            label_arr,
                            label_nodata,
                        )
                        loaded_count += 1
                        year_count[date_dt.year] = (
                            year_count.get(date_dt.year, 0) + 1
                        )
            except Exception as exc:
                print(
                    f"  加载标签文件 {label_file.name} 失败: "
                    f"{exc}"
                )

        if not self.label_data:
            raise ValueError(
                "没有加载到任何ERA5-Land SWE日期；"
                "禁止继续微调"
            )

        expected_dates = {
            date_dt for date_dt in self.all_dates
        }
        missing_dates = sorted(
            expected_dates - set(self.label_data)
        )
        if missing_dates:
            raise RuntimeError(
                "ERA5-Land SWE没有覆盖统一时间轴，"
                f"缺少{len(missing_dates)}天，"
                f"首个={missing_dates[0]:%Y-%m-%d}"
            )

        print(f"\n  ✅ ERA5-Land SWE加载完成:")
        print(f"     总日期数: {loaded_count}")
        for year, count in sorted(year_count.items()):
            print(f"     {year}年: {count}天")
    def _load_corrections(self):
        """
        加载产品值修正数据。

        correction_map 使用字符串 key，避免 Timestamp / datetime / 日期格式不一致导致匹配失败。

        key 格式：
            (str(station_id), "YYYY-MM-DD")

        value:
            corrected SWE in mm
        """
        import pandas as pd
        import numpy as np
        from pathlib import Path

        correction_file = Path("/root/autodl-tmp/full_sample_predictions.csv")
        zero_file = Path("/root/autodl-tmp/zero_misclassifications.csv")

        self.correction_map = {}

        print("\n📊 加载产品值修正数据...")

        if not correction_file.exists():
            print(f"   ⚠ 未找到修正文件: {correction_file}")
            print("   跳过产品值修正")
            return

        if not zero_file.exists():
            print(f"   ⚠ 未找到 zero 文件: {zero_file}")
            print("   跳过产品值修正")
            return

        def parse_mixed_date(col):
            """
            兼容混合日期格式：
            - 2015/1/1
            - 2015/1/1 0:00
            - 2015-01-01
            - 2015-01-01 00:00:00
            """
            col = col.astype(str).str.strip()

            try:
                dt = pd.to_datetime(col, errors="coerce", format="mixed")
            except TypeError:
                # 老版本 pandas 不支持 format="mixed"
                dt = pd.to_datetime(col, errors="coerce", infer_datetime_format=True)

            # 统一到当天 00:00:00
            dt = dt.dt.normalize()
            return dt

        # ============================================================
        # 1. 读取 zero_misclassifications.csv
        # ============================================================
        try:
            zero_df = pd.read_csv(zero_file)
        except Exception as e:
            print(f"   ❌ 读取 zero 文件失败: {e}")
            return

        if "station_id" not in zero_df.columns or "date" not in zero_df.columns:
            print(f"   ❌ zero_misclassifications.csv 必须包含 station_id 和 date 列")
            print(f"      当前列: {list(zero_df.columns)}")
            return

        zero_df["station_id"] = zero_df["station_id"].astype(str).str.strip()
        zero_df["date"] = parse_mixed_date(zero_df["date"])

        before_zero = len(zero_df)
        zero_df = zero_df.dropna(subset=["station_id", "date"]).copy()
        after_zero = len(zero_df)

        zero_df["date_key"] = zero_df["date"].dt.strftime("%Y-%m-%d")

        print(f"   zero文件原始行数: {before_zero}")
        print(f"   zero文件有效行数: {after_zero}")

        if len(zero_df) == 0:
            print("   ⚠ zero 文件没有有效 station_id/date，跳过修正")
            return

        # ============================================================
        # 2. 读取 full_sample_predictions.csv
        # ============================================================
        try:
            correction_df = pd.read_csv(correction_file)

            # 如果读出来没有 predicted_swe，说明可能没有表头
            if "predicted_swe" not in correction_df.columns:
                correction_df = pd.read_csv(
                    correction_file,
                    header=None,
                    names=["station_id", "date", "predicted_swe"]
                )

        except Exception as e:
            print(f"   ⚠ 常规读取 full_sample_predictions.csv 失败: {e}")
            print("   尝试按无表头三列格式读取...")

            try:
                correction_df = pd.read_csv(
                    correction_file,
                    header=None,
                    names=["station_id", "date", "predicted_swe"]
                )
            except Exception as e2:
                print(f"   ❌ 读取 correction 文件失败: {e2}")
                return

        required_cols = ["station_id", "date", "predicted_swe"]
        missing_cols = [c for c in required_cols if c not in correction_df.columns]

        if missing_cols:
            print(f"   ❌ full_sample_predictions.csv 缺少必要列: {missing_cols}")
            print(f"      当前列: {list(correction_df.columns)}")
            return

        correction_df["station_id"] = correction_df["station_id"].astype(str).str.strip()
        correction_df["date"] = parse_mixed_date(correction_df["date"])
        correction_df["predicted_swe"] = pd.to_numeric(
            correction_df["predicted_swe"],
            errors="coerce"
        )

        before_corr = len(correction_df)
        correction_df = correction_df.dropna(
            subset=["station_id", "date", "predicted_swe"]
        ).copy()
        after_corr = len(correction_df)

        correction_df["date_key"] = correction_df["date"].dt.strftime("%Y-%m-%d")

        print(f"   correction文件原始行数: {before_corr}")
        print(f"   correction文件有效行数: {after_corr}")

        if len(correction_df) == 0:
            print("   ⚠ correction 文件没有有效 station_id/date/predicted_swe，跳过修正")
            return

        # ============================================================
        # 3. 打印日期范围，方便检查
        # ============================================================
        print(f"   zero日期范围: {zero_df['date_key'].min()} 到 {zero_df['date_key'].max()}")
        print(f"   correction日期范围: {correction_df['date_key'].min()} 到 {correction_df['date_key'].max()}")

        # ============================================================
        # 4. 合并 zero 文件和 correction 文件
        #    只修正 zero_misclassifications 里记录的样本
        # ============================================================
        merged = zero_df.merge(
            correction_df[["station_id", "date_key", "predicted_swe"]],
            on=["station_id", "date_key"],
            how="inner"
        )

        print(f"   匹配到的修正样本数: {len(merged)}")

        if len(merged) == 0:
            print("   ⚠ zero文件和 full_sample_predictions.csv 没有匹配样本")
            print("   请检查 station_id 是否一致、date 是否对应同一天")
            print("   zero 示例:")
            print(zero_df[["station_id", "date_key"]].head())
            print("   correction 示例:")
            print(correction_df[["station_id", "date_key", "predicted_swe"]].head())
            return

        # ============================================================
        # 5. 构建 correction_map
        #    key = (station_id字符串, YYYY-MM-DD字符串)
        # ============================================================
        self.correction_map = {}

        for _, row in merged.iterrows():
            sid_key = str(row["station_id"]).strip()
            date_key = str(row["date_key"]).strip()

            corrected_mm = float(row["predicted_swe"])

            # 可选：限制在合理 SWE 范围，防止异常预测污染输入
            corrected_mm = float(np.clip(corrected_mm, 0.0, 300.0))

            key = (sid_key, date_key)
            self.correction_map[key] = corrected_mm

        print(f"   ✅ 加载了 {len(self.correction_map)} 个修正样本")

        # ============================================================
        # 6. 打印示例
        # ============================================================
        print("   示例修正:")
        for i, ((sid, d), val) in enumerate(list(self.correction_map.items())[:10]):
            print(f"      {sid} - {d}: {val:.2f} mm")

        # ============================================================
        # 7. 调试：检查重复 key
        # ============================================================
        n_merged = len(merged)
        n_map = len(self.correction_map)

        if n_map < n_merged:
            print(f"   ⚠ 注意：merged 有 {n_merged} 行，但 correction_map 只有 {n_map} 个 key")
            print("      说明存在重复 station_id + date，后面的值覆盖了前面的值")

        # ============================================================
        # 8. 标记 debug 计数器
        # ============================================================
        self._correction_debug_count = 0
    
    def _build_complete_daily_timeline(self):
        """
        根据 self.load_years 构建完整日时间轴。

        2015—2018应得到1461天，不再使用多个动态变量的日期交集
        作为站点样本时间轴。
        """
        years = sorted({
            int(year)
            for year in self.load_years
        })

        if not years:
            raise RuntimeError(
                "load_years为空，无法构建完整日时间轴"
            )

        complete_dates = []

        for year in years:
            current_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)

            while current_date <= end_date:
                complete_dates.append(current_date)
                current_date += timedelta(days=1)

        if len(set(complete_dates)) != len(complete_dates):
            raise RuntimeError(
                "完整日时间轴内部存在重复日期"
            )

        return complete_dates


    def _align_to_timeline(
        self,
        var: str,
        var_data,
        var_dates,
        allow_temporal_fill: bool = False,
    ):
        """
        将单个动态变量对齐到完整日时间轴。

        规则
        ----
        1. LST、RH和PR必须完整覆盖目标时间轴，不允许静默补日期；
        2. 只有chelsa_sfxwind允许对缺失日期进行单变量填补；
        3. 内部缺口使用前后有效日期线性插值；
        4. 时间轴首尾缺口使用最近有效日期；
        5. 最近有效日期距离超过7天时直接报错；
        6. 此函数只改变当前变量，不改变站点样本日期，也不改变
           其他动态变量的日期。
        """
        var_data = np.asarray(
            var_data,
            dtype=np.float32,
        )

        normalized_dates = [
            pd.Timestamp(date_value)
            .normalize()
            .to_pydatetime()
            for date_value in var_dates
        ]

        target_dates = [
            pd.Timestamp(date_value)
            .normalize()
            .to_pydatetime()
            for date_value in self.all_dates
        ]

        if var_data.ndim != 3:
            raise RuntimeError(
                f"{var}必须是(T,H,W)三维数组，"
                f"当前shape={var_data.shape}"
            )

        if var_data.shape[0] != len(normalized_dates):
            raise RuntimeError(
                f"{var}时间维与日期数量不一致："
                f"data={var_data.shape[0]}, "
                f"dates={len(normalized_dates)}"
            )

        if len(set(normalized_dates)) != len(normalized_dates):
            duplicate_count = (
                len(normalized_dates)
                - len(set(normalized_dates))
            )
            raise RuntimeError(
                f"{var}日期标准化后存在"
                f"{duplicate_count}个重复日期"
            )

        date_to_source_index = {
            date_value: index
            for index, date_value
            in enumerate(normalized_dates)
        }

        target_date_set = set(target_dates)

        missing_dates = [
            date_value
            for date_value in target_dates
            if date_value not in date_to_source_index
        ]

        extra_dates = [
            date_value
            for date_value in normalized_dates
            if date_value not in target_date_set
        ]

        if missing_dates and not allow_temporal_fill:
            preview = ", ".join(
                date_value.strftime("%Y-%m-%d")
                for date_value in missing_dates[:10]
            )

            raise RuntimeError(
                f"{var}缺少{len(missing_dates)}个日尺度日期，"
                "该变量禁止静默时间填补。"
                f"前10个缺失日期：{preview}"
            )

        # 日期和顺序已经完全一致时，直接返回原数组，避免额外复制。
        if normalized_dates == target_dates:
            report = {
                "variable": var,
                "source_date_count": len(normalized_dates),
                "target_date_count": len(target_dates),
                "exact_count": len(target_dates),
                "filled_count": 0,
                "linear_count": 0,
                "nearest_count": 0,
                "missing_dates": [],
                "extra_dates": [
                    value.strftime("%Y-%m-%d")
                    for value in extra_dates
                ],
                "max_nearest_gap_days": 0,
                "fill_details": [],
            }

            print(
                f"  {var}: 完整日覆盖，"
                f"{len(target_dates)}天全部精确匹配"
            )

            return var_data, report

        source_dates_sorted = sorted(
            date_to_source_index
        )

        source_ordinals = np.asarray(
            [
                date_value.toordinal()
                for date_value in source_dates_sorted
            ],
            dtype=np.int64,
        )

        target_count = len(target_dates)
        _, height, width = var_data.shape

        aligned_data = np.empty(
            (target_count, height, width),
            dtype=np.float32,
        )

        exact_count = 0
        fill_details = []

        for target_index, target_date in enumerate(
            target_dates
        ):
            source_index = date_to_source_index.get(
                target_date
            )

            if source_index is not None:
                aligned_data[target_index] = (
                    var_data[source_index]
                )
                exact_count += 1
                continue

            insert_position = int(
                np.searchsorted(
                    source_ordinals,
                    target_date.toordinal(),
                    side="left",
                )
            )

            left_date = (
                source_dates_sorted[
                    insert_position - 1
                ]
                if insert_position > 0
                else None
            )

            right_date = (
                source_dates_sorted[
                    insert_position
                ]
                if insert_position
                < len(source_dates_sorted)
                else None
            )

            if left_date is None and right_date is None:
                raise RuntimeError(
                    f"{var}没有任何可用于时间填补的数据"
                )

            left_gap = (
                abs((target_date - left_date).days)
                if left_date is not None
                else None
            )

            right_gap = (
                abs((right_date - target_date).days)
                if right_date is not None
                else None
            )

            available_gaps = [
                gap
                for gap in [left_gap, right_gap]
                if gap is not None
            ]

            nearest_gap = min(available_gaps)

            if (
                nearest_gap
                > DYNAMIC_TEMPORAL_FILL_MAX_GAP_DAYS
            ):
                raise RuntimeError(
                    f"{var}在{target_date:%Y-%m-%d}"
                    "附近缺口过长："
                    f"最近有效日期相距{nearest_gap}天，"
                    "超过允许的"
                    f"{DYNAMIC_TEMPORAL_FILL_MAX_GAP_DAYS}天"
                )

            if (
                left_date is not None
                and right_date is not None
            ):
                left_layer = var_data[
                    date_to_source_index[left_date]
                ]

                right_layer = var_data[
                    date_to_source_index[right_date]
                ]

                total_gap = (
                    right_date - left_date
                ).days

                if total_gap <= 0:
                    raise RuntimeError(
                        f"{var}插值日期顺序异常："
                        f"{left_date} -> {right_date}"
                    )

                weight = (
                    (target_date - left_date).days
                    / total_gap
                )

                left_valid = (
                    np.isfinite(left_layer)
                    & (left_layer != FINAL_NODATA)
                )

                right_valid = (
                    np.isfinite(right_layer)
                    & (right_layer != FINAL_NODATA)
                )

                both_valid = (
                    left_valid & right_valid
                )

                only_left = (
                    left_valid & ~right_valid
                )

                only_right = (
                    right_valid & ~left_valid
                )

                filled_layer = np.full(
                    left_layer.shape,
                    np.nan,
                    dtype=np.float32,
                )

                filled_layer[both_valid] = (
                    left_layer[both_valid]
                    + (
                        right_layer[both_valid]
                        - left_layer[both_valid]
                    )
                    * weight
                )

                filled_layer[only_left] = (
                    left_layer[only_left]
                )

                filled_layer[only_right] = (
                    right_layer[only_right]
                )

                aligned_data[target_index] = (
                    filled_layer
                )

                method = "linear"
                source_dates_text = (
                    f"{left_date:%Y-%m-%d}|"
                    f"{right_date:%Y-%m-%d}"
                )

            else:
                nearest_date = (
                    left_date
                    if left_date is not None
                    else right_date
                )

                aligned_data[target_index] = (
                    var_data[
                        date_to_source_index[
                            nearest_date
                        ]
                    ]
                )

                method = "nearest_edge"
                source_dates_text = (
                    nearest_date.strftime(
                        "%Y-%m-%d"
                    )
                )

            fill_details.append({
                "target_date":
                    target_date.strftime(
                        "%Y-%m-%d"
                    ),
                "method":
                    method,
                "source_dates":
                    source_dates_text,
                "nearest_gap_days":
                    int(nearest_gap),
            })

        linear_count = sum(
            detail["method"] == "linear"
            for detail in fill_details
        )

        nearest_count = sum(
            detail["method"] == "nearest_edge"
            for detail in fill_details
        )

        max_nearest_gap = max(
            (
                detail["nearest_gap_days"]
                for detail in fill_details
            ),
            default=0,
        )

        report = {
            "variable": var,
            "source_date_count":
                len(normalized_dates),
            "target_date_count":
                len(target_dates),
            "exact_count":
                int(exact_count),
            "filled_count":
                int(len(fill_details)),
            "linear_count":
                int(linear_count),
            "nearest_count":
                int(nearest_count),
            "missing_dates": [
                value.strftime("%Y-%m-%d")
                for value in missing_dates
            ],
            "extra_dates": [
                value.strftime("%Y-%m-%d")
                for value in extra_dates
            ],
            "max_nearest_gap_days":
                int(max_nearest_gap),
            "fill_details":
                fill_details,
        }

        print(
            f"  {var}: source={len(normalized_dates)}, "
            f"target={len(target_dates)}, "
            f"exact={exact_count}, "
            f"filled={len(fill_details)}, "
            f"linear={linear_count}, "
            f"nearest_edge={nearest_count}, "
            f"max_gap={max_nearest_gap}天"
        )

        for detail in fill_details[:10]:
            print(
                "     填补 "
                f"{detail['target_date']} <- "
                f"{detail['source_dates']} "
                f"({detail['method']}, "
                f"最近间隔"
                f"{detail['nearest_gap_days']}天)"
            )

        if len(fill_details) > 10:
            print(
                f"     其余"
                f"{len(fill_details) - 10}"
                "个填补日期省略"
            )

        return aligned_data, report


    def _validate_complete_daily_feature_timeline(
        self,
    ):
        """
        检查所有动态变量是否严格采用完整日时间轴。
        """
        expected_dates = (
            self._build_complete_daily_timeline()
        )

        actual_dates = [
            pd.Timestamp(date_value)
            .normalize()
            .to_pydatetime()
            for date_value in self.all_dates
        ]

        if actual_dates != expected_dates:
            raise RuntimeError(
                "动态特征时间轴不是完整日时间轴："
                f"expected={len(expected_dates)}, "
                f"actual={len(actual_dates)}, "
                f"expected_range="
                f"{expected_dates[0]:%Y-%m-%d}—"
                f"{expected_dates[-1]:%Y-%m-%d}, "
                f"actual_range="
                f"{actual_dates[0]:%Y-%m-%d}—"
                f"{actual_dates[-1]:%Y-%m-%d}"
            )

        self.all_dates = actual_dates
        self.date_to_index = {
            date_value: index
            for index, date_value
            in enumerate(actual_dates)
        }

        for var in CONV_VARS:
            if var not in self.conv_dyn_data:
                raise RuntimeError(
                    f"动态特征缺少变量：{var}"
                )

            array = self.conv_dyn_data[var]

            if array.shape[0] != len(expected_dates):
                raise RuntimeError(
                    f"{var}时间维错误："
                    f"{array.shape[0]} "
                    f"!= {len(expected_dates)}"
                )

            report = getattr(
                self,
                "dynamic_time_fill_report",
                {},
            ).get(var, {})

            filled_count = int(
                report.get("filled_count", 0)
            )

            if (
                var != "chelsa_sfxwind"
                and filled_count != 0
            ):
                raise RuntimeError(
                    f"{var}不应进行时间填补，"
                    f"但报告filled_count="
                    f"{filled_count}"
                )

        expected_count = len(expected_dates)

        print(
            "\n✅ 完整日时间轴检查通过："
            f"{expected_count}天，"
            f"{expected_dates[0]:%Y-%m-%d}—"
            f"{expected_dates[-1]:%Y-%m-%d}"
        )

        for var in CONV_VARS:
            report = getattr(
                self,
                "dynamic_time_fill_report",
                {},
            ).get(var, {})

            print(
                f"   {var}: "
                f"shape="
                f"{self.conv_dyn_data[var].shape}, "
                f"filled="
                f"{report.get('filled_count', 0)}"
            )


    def _load_all_features(self):
        """
        加载完整日尺度动态特征。

        不再取四变量日期交集。站点日期始终保留原始观测日期；
        只有缺日的chelsa_sfxwind通道单独做时间填补。
        """
        print("\n加载特征数据...")

        expected_years = [
            2015,
            2016,
            2017,
            2018,
        ]

        actual_years = sorted({
            int(year)
            for year in self.load_years
        })

        if actual_years != expected_years:
            raise RuntimeError(
                "渐进式微调年份必须与预训练一致："
                f"expected={expected_years}, "
                f"actual={actual_years}"
            )

        self.all_dates = (
            self._build_complete_daily_timeline()
        )

        self.date_to_index = {
            date_value: index
            for index, date_value
            in enumerate(self.all_dates)
        }

        print(
            "\n📅 站点微调采用完整日时间轴："
        )

        print(
            f"   日期范围："
            f"{self.all_dates[0]:%Y-%m-%d}—"
            f"{self.all_dates[-1]:%Y-%m-%d}"
        )

        print(
            f"   总天数：{len(self.all_dates)}"
        )

        self.conv_dyn_data = {}
        self.dynamic_time_fill_report = {}

        for var in CONV_VARS:
            print(f"\n加载 {var} 数据...")

            var_data, var_dates = (
                self._load_single_variable(var)
            )

            if (
                var_data is None
                or not var_dates
            ):
                raise RuntimeError(
                    f"必需动态特征{var}加载失败"
                )

            if (
                int(var_data.shape[0])
                != len(var_dates)
            ):
                raise RuntimeError(
                    f"{var}数据长度与日期长度"
                    "不一致："
                    f"data={var_data.shape[0]}, "
                    f"dates={len(var_dates)}"
                )

            normalized_var_dates = [
                pd.Timestamp(date_value)
                .normalize()
                .to_pydatetime()
                for date_value in var_dates
            ]

            if (
                len(set(normalized_var_dates))
                != len(normalized_var_dates)
            ):
                duplicate_count = (
                    len(normalized_var_dates)
                    - len(set(normalized_var_dates))
                )

                raise RuntimeError(
                    f"{var}存在"
                    f"{duplicate_count}个重复日期"
                )

            print(
                f"  {var}原始时间轴："
                f"{len(normalized_var_dates)}天，"
                f"{normalized_var_dates[0]:%Y-%m-%d}—"
                f"{normalized_var_dates[-1]:%Y-%m-%d}"
            )

            aligned_data, fill_report = (
                self._align_to_timeline(
                    var=var,
                    var_data=var_data,
                    var_dates=normalized_var_dates,
                    allow_temporal_fill=(
                        var == "chelsa_sfxwind"
                    ),
                )
            )

            self.conv_dyn_data[var] = (
                aligned_data
            )

            self.dynamic_time_fill_report[var] = (
                fill_report
            )

            del var_data

        self._validate_complete_daily_feature_timeline()

        self._load_static_conv_features()
        self._load_point_features()


    def _load_single_variable(self, var: str):
        """加载单个变量数据 - 使用地理对齐"""
        print(f"  加载 {var} 数据...")

        # 🔥 使用 self.load_years（已在 __init__ 中设置为3年）
        target_years = self.load_years
        print(f"    目标年份: {target_years}")

        all_files = []
        for year in target_years:
            var_dir = conv_var_path(var, year)
            if not var_dir.exists():
                print(f"    {year}: 目录不存在")
                continue

            # STATION_LST_RH_GLOB_MATCH_PRETRAIN_V1
            # 与 data_online_era5_swe.py 的正式预训练匹配规则保持一致。
            if var == "chelsa_sfxwind":
                patterns = [f"*{year}*.tif"]
            elif var == "lst":
                patterns = [
                    f"ERA5LAND_LST_{year}??_DAILYMEAN*.tif",
                    f"*LST*{year}*.tif",
                    f"ERA5_ST_{year}*.tif",
                ]
            elif var == "rh":
                patterns = [
                    f"ERA5LAND_RH_{year}??_DAILYMEAN*.tif",
                    f"*RH*{year}*.tif",
                    f"ERA5_RH_DailyMean_{year}_*.tif",
                ]
            elif var == "pr":
                patterns = [f"*{year}*.tif"]
            else:
                patterns = ["*.tif"]

            files = []
            seen_files = set()
            for pattern in patterns:
                for file_path in sorted(var_dir.glob(pattern)):
                    resolved = str(file_path.resolve())
                    if resolved not in seen_files:
                        seen_files.add(resolved)
                        files.append(file_path)

            all_files.extend(files)
            print(
                f"    {year}: 找到 {len(files)} 个文件 "
                f"(目录={var_dir}, patterns={patterns})"
            )

        if not all_files:
            print(f"  警告: 未找到 {var} 文件")
            return None, []

        import calendar

        # 对于 chelsa_sfxwind 和 pr，是单文件单波段日数据
        if var in ["chelsa_sfxwind", "pr"]:
            var_dates = []
            var_data = []

            print(f"    以日尺度单波段模式加载 {var}...")

            for f in all_files:
                try:
                    dt = self._parse_date_from_filename(f.name)

                    # 🔥 使用 target_years 过滤
                    if dt.year not in target_years:
                        continue

                    with rasterio.open(f) as ds:
                        data = ds.read(1).astype(np.float32)

                    aligned_data = self._align_single_layer(data, f)
                    var_dates.append(dt)
                    var_data.append(aligned_data)
                except Exception as e:
                    print(f"    处理文件 {f.name} 失败: {e}")
                    continue

            if not var_data:
                print(f"  警告: {var} 无有效数据")
                return None, []

            sorted_indices = np.argsort(var_dates)
            var_dates = [var_dates[i] for i in sorted_indices]
            var_data = np.stack([var_data[i] for i in sorted_indices], axis=0)

            print(f"  {var}: 最终形状 {var_data.shape}, {len(var_dates)} 个时间点")
            print(f"     日期范围: {var_dates[0].strftime('%Y-%m-%d')} 到 {var_dates[-1].strftime('%Y-%m-%d')}")

            return var_data, var_dates

        # 对于 lst 和 rh，是多波段月度立方体文件（每个文件包含一个月所有天的数据）
        else:  # var in ["lst", "rh"]
            daily_data = {}  # date -> data
            src_bounds = None
            src_transform = None

            total_days_loaded = 0
            file_count = 0

            print(f"    以多波段月度立方体模式加载 {var}...")

            for f in all_files:
                try:
                    file_count += 1

                    # 🔥 修复：支持多种文件名格式
                    # 格式1: ERA5_ST_201501_UTC0_27830m.tif (YYYYMM)
                    # 格式2: ERA5_RH_DailyMean_2015_01_27830m.tif (YYYY_MM)
                    # 格式3: ERA5_RH_DailyMean_2015-01_27830m.tif (YYYY-MM)

                    # 先尝试匹配 YYYYMM 格式（6位连续数字）
                    match = re.search(r'(\d{4})(\d{2})', f.name)
                    if not match:
                        # 再尝试匹配 YYYY_MM 或 YYYY-MM 格式
                        match = re.search(r'(\d{4})[_-](\d{2})', f.name)

                    if not match:
                        print(f"    无法解析年月: {f.name}")
                        continue

                    year = int(match.group(1))
                    month = int(match.group(2))

                    # 验证月份有效性
                    if month < 1 or month > 12:
                        print(f"    无效月份: {f.name} (month={month})")
                        continue

                    # 🔥 检查年份是否在目标范围内
                    if year not in target_years:
                        print(f"    跳过非目标年份: {f.name} (year={year})")
                        continue

                    # 获取该月的天数
                    month_days = calendar.monthrange(year, month)[1]

                    with rasterio.open(f) as ds:
                        if src_bounds is None:
                            src_bounds = ds.bounds
                            src_transform = ds.transform

                        n_bands = ds.count
                        print(f"    [{file_count}/{len(all_files)}] 处理文件: {f.name}")
                        print(f"       年份: {year}, 月份: {month}, 波段数: {n_bands}, 本月实际天数: {month_days}")

                        days_in_file = 0
                        # 遍历每个波段（每天）
                        for day in range(1, min(n_bands, month_days) + 1):
                            date_dt = datetime(year, month, day)

                            band_data = ds.read(day).astype(np.float32)

                            # 处理无效值
                            if var == "lst":
                                # LST 无效值 -9999
                                band_data = np.where(band_data == -9999.0, np.nan, band_data)
                            elif var == "rh":
                                # RH 无效值 -9999
                                band_data = np.where(band_data == -9999.0, np.nan, band_data)

                            aligned_band = self._align_single_layer(
                                band_data, src_transform, self.transform, self.H, self.W
                            )
                            daily_data[date_dt] = aligned_band
                            days_in_file += 1
                            total_days_loaded += 1

                            # 每加载100天打印一次进度
                            if total_days_loaded % 100 == 0:
                                print(f"       已累计加载 {total_days_loaded} 天数据...")

                        print(f"       本文件成功加载 {days_in_file} 天")

                except Exception as e:
                    print(f"    处理文件 {f.name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if not daily_data:
                print(f"  警告: {var} 无有效数据")
                return None, []

            sorted_dates = sorted(daily_data.keys())
            var_arr = np.stack([daily_data[dt] for dt in sorted_dates], axis=0)

            print(f"\n  ✅ {var} 加载完成!")
            print(f"     总文件数: {len(all_files)}")
            print(f"     成功加载天数: {len(sorted_dates)}")
            print(f"     日期范围: {sorted_dates[0].strftime('%Y-%m-%d')} 到 {sorted_dates[-1].strftime('%Y-%m-%d')}")
            print(f"     数据形状: {var_arr.shape}")

            # 打印前5天和后5天的日期
            print(f"     前5天: {[d.strftime('%Y-%m-%d') for d in sorted_dates[:5]]}")
            print(f"     后5天: {[d.strftime('%Y-%m-%d') for d in sorted_dates[-5:]]}")

            return var_arr, sorted_dates
    
    def _align_single_layer(self, src_data, src_file_or_transform, target_transform=None, target_h=None, target_w=None):
        """对齐单个图层"""
        try:
            src_data = src_data.astype(np.float32)
            src_data[~np.isfinite(src_data)] = np.nan
            src_data[src_data == FINAL_NODATA] = np.nan

            if isinstance(src_file_or_transform, (Path, str)):
                src_file = Path(src_file_or_transform)
                with rasterio.open(src_file) as src:
                    src_transform = src.transform
                target_transform = self.transform
                target_h, target_w = self.H, self.W
            else:
                src_transform = src_file_or_transform
                if target_transform is None or target_h is None or target_w is None:
                    raise ValueError("方式2调用时必须提供 target_transform, target_h, target_w")
            
            aligned = np.full((target_h, target_w), np.nan, dtype=src_data.dtype)
            src_h, src_w = src_data.shape
            
            rows, cols = np.meshgrid(range(target_h), range(target_w), indexing='ij')
            
            target_xs = (target_transform.a * cols.ravel() + 
                         target_transform.b * rows.ravel() + 
                         target_transform.c + 
                         target_transform.a * 0.5 + 
                         target_transform.b * 0.5)
            
            target_ys = (target_transform.d * cols.ravel() + 
                         target_transform.e * rows.ravel() + 
                         target_transform.f + 
                         target_transform.d * 0.5 + 
                         target_transform.e * 0.5)
            
            src_rows, src_cols = rasterio.transform.rowcol(
                src_transform,
                target_xs,
                target_ys,
                op=round
            )
            
            for i in range(len(src_rows)):
                r, c = src_rows[i], src_cols[i]
                row_idx, col_idx = rows.ravel()[i], cols.ravel()[i]
                
                if 0 <= r < src_h and 0 <= c < src_w:
                    aligned[row_idx, col_idx] = src_data[r, c]
            
            # 不在整幅图范围内做最近邻填充。
            # NoData 保持为 NaN，后续 patch 内部再局部插值。
            return aligned
            
        except Exception as e:
            print(f"      地理对齐失败 ({e})，使用简单调整")
            if target_h is None or target_w is None:
                target_h, target_w = self.H, self.W
            return self._resize_to_standard(src_data, target_h, target_w)
    
    def _adjust_monthly_to_daily(self, var: str, var_data, var_dates):
        """
        将数据对齐到统一时间轴
        现在 var_data 已经是日尺度，只需要按 self.all_dates 重新索引
        """
        T_daily = len(self.all_dates)
        _, H, W = var_data.shape

        print(f"\n  📅 对齐 {var} 到统一时间轴:")
        print(f"     原始数据: {len(var_dates)} 天")
        print(f"     原始范围: {var_dates[0].strftime('%Y-%m-%d')} 到 {var_dates[-1].strftime('%Y-%m-%d')}")
        print(f"     目标时间轴: {T_daily} 天")
        print(f"     目标范围: {self.all_dates[0].strftime('%Y-%m-%d')} 到 {self.all_dates[-1].strftime('%Y-%m-%d')}")

        # 创建日期到数据的映射
        date_to_data = {dt: var_data[i] for i, dt in enumerate(var_dates)}

        aligned_data = np.zeros((T_daily, H, W), dtype=np.float32)
        matched_count = 0
        interpolated_count = 0

        for i, daily_date in enumerate(self.all_dates):
            if daily_date in date_to_data:
                aligned_data[i] = date_to_data[daily_date]
                matched_count += 1
            else:
                # 找最近的日期
                closest_date = min(var_dates, key=lambda d: abs((d - daily_date).days))
                aligned_data[i] = date_to_data[closest_date]
                interpolated_count += 1
                if interpolated_count <= 5:  # 只打印前5个插值
                    day_gap = abs((closest_date - daily_date).days)
                    print(f"       插值: {daily_date.strftime('%Y-%m-%d')} -> 使用 {closest_date.strftime('%Y-%m-%d')} (间隔{day_gap}天)")

        print(f"     对齐结果: 精确匹配 {matched_count} 天, 最近邻插值 {interpolated_count} 天")

        return aligned_data
    
    def _parse_date_from_filename(self, filename: str) -> datetime:
        """从文件名解析日期"""
        filename_lower = filename.lower()
        
        calmday_match = re.search(r'China_CalmDays_Frequency_(\d{4})_threshold', filename)
        if calmday_match:
            year = int(calmday_match.group(1))
            return datetime(year, 1, 1)
        
        landsat_match = re.search(r'China_Landsat_(\d{4})_reflectance', filename)
        if landsat_match:
            year = int(landsat_match.group(1))
            return datetime(year, 7, 1)
        
        era5_match = re.search(r'ERA5_(?:ST|RH)_(\d{4})(\d{2})', filename)
        if era5_match:
            year = int(era5_match.group(1))
            month = int(era5_match.group(2))
            return datetime(year, month, 15)
        
        s1_match = re.search(r'S1_MONTHLY_(\d{4})_(\d{2})', filename)
        if s1_match:
            year = int(s1_match.group(1))
            month = int(s1_match.group(2))
            return datetime(year, month, 15)
        
        smap_match = re.search(r'China_SMAP_TB_(\d{4})_(\d{2})', filename)
        if smap_match:
            year = int(smap_match.group(1))
            month = int(smap_match.group(2))
            return datetime(year, month, 15)
        
        xgb_match = re.search(r'XGB_SWE_DAILY_025_(\d{4})(\d{2})(\d{2})', filename)
        if xgb_match:
            year = int(xgb_match.group(1))
            month = int(xgb_match.group(2))
            day = int(xgb_match.group(3))
            return datetime(year, month, day)
        
        chelsa_match = re.search(r'CHELSA_(?:pr|sfcWind)_(\d{2})_(\d{2})_(\d{4})', filename)
        if chelsa_match:
            day = int(chelsa_match.group(1))
            month = int(chelsa_match.group(2))
            year = int(chelsa_match.group(3))
            return datetime(year, month, day)
        
        patterns = [
            r'(\d{2})_(\d{2})_(\d{4})',
            r'(\d{4})(\d{2})(\d{2})',
            r'(\d{4})-(\d{2})-(\d{2})',
            r'(\d{4})_(\d{2})_(\d{2})',
            r'(\d{4})_(\d{2})',
            r'(\d{4})(\d{2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                try:
                    if len(groups) == 3:
                        if pattern == r'(\d{2})_(\d{2})_(\d{4})':
                            day, month, year = map(int, groups)
                        else:
                            year, month, day = map(int, groups)
                        return datetime(year, month, day)
                    elif len(groups) == 2:
                        year, month = map(int, groups)
                        return datetime(year, month, 15)
                except:
                    continue
        
        if 's1' in filename_lower:
            s1_match = re.search(r'(\d{4})_(\d{2})', filename)
            if s1_match:
                year, month = map(int, s1_match.groups())
                return datetime(year, month, 15)
        
        print(f"  警告: 无法解析日期: {filename}, 使用默认日期")
        if isinstance(self.year_target, list):
            return datetime(self.year_target[0], 1, 1)
        else:
            return datetime(self.year_target, 1, 1)
    
    def _load_static_conv_features(self):
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
    def _idw_interpolate(self, data, invalid_mask, power=2, radius=10):
        """IDW (反距离加权) 插值填充二维数组中的无效值"""
        if not np.any(invalid_mask):
            return data

        valid_mask = ~invalid_mask
        valid_coords = np.argwhere(valid_mask)
        valid_values = data[valid_mask]

        if len(valid_values) == 0:
            return np.zeros_like(data)

        invalid_coords = np.argwhere(invalid_mask)
        interpolated = np.zeros(len(invalid_coords))

        for i, (x, y) in enumerate(invalid_coords):
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
                interpolated[i] = np.mean(valid_values)
                continue

            distances = np.sqrt((local_coords[:, 0] - x) ** 2 + (local_coords[:, 1] - y) ** 2)
            distances[distances == 0] = 1e-8
            weights = 1.0 / (distances ** power)
            weights /= weights.sum()
            interpolated[i] = np.sum(weights * local_values)

        result = data.copy()
        result[invalid_mask] = interpolated
        return result
    
    
    def _load_point_features(self):
        """加载点特征；文件发现和读取逻辑与正式预训练保持一致。"""
        print(f"\n加载点特征...")

        self.ls_data = {}
        missing_ls_years = []

        for year in self.load_years:
            ls_file = point_var_path("ls", year)
            if ls_file is None or not Path(ls_file).exists():
                missing_ls_years.append(int(year))
                continue

            print(f"  处理LS文件: {Path(ls_file).name} (年份: {year})")
            with rasterio.open(ls_file) as ds:
                ls_data_raw = ds.read().astype(np.float32)
                src_transform = ds.transform

            aligned_bands = []
            for band_index in range(ls_data_raw.shape[0]):
                aligned_bands.append(
                    self._align_single_layer(
                        ls_data_raw[band_index],
                        src_transform,
                        self.transform,
                        self.H,
                        self.W,
                    )
                )

            self.ls_data[int(year)] = np.stack(
                aligned_bands, axis=0
            )
            print(
                f"    LS{year}数据形状: "
                f"{self.ls_data[int(year)].shape}"
            )

        if missing_ls_years:
            raise FileNotFoundError(
                "缺少正式预训练命名的Landsat文件，年份="
                f"{missing_ls_years}"
            )

        self.ls_data_default = self.ls_data[
            int(self.load_years[0])
        ]

        print(f"\n  加载哨兵1数据...")
        self._load_sentinel1_data()
        if not self.all_s1_dates:
            raise RuntimeError(
                "未加载到任何Sentinel-1日期，禁止用全0替代后继续微调"
            )

        print(f"\n  加载SMAP数据...")
        self._load_smap_data()
        if not self.all_smap_dates:
            raise RuntimeError(
                "未加载到任何SMAP日期，禁止用默认亮温替代后继续微调"
            )

        self.C_point = 18

        print(f"\n  【点特征加载统计】")
        print(
            f"    LS年份: {sorted(self.ls_data.keys())}; "
            f"默认形状: {self.ls_data_default.shape}"
        )
        print(
            f"    哨兵1数据: {len(self.all_s1_dates)} 个日期; "
            f"{self.all_s1_dates[0]:%Y-%m-%d} → "
            f"{self.all_s1_dates[-1]:%Y-%m-%d}"
        )
        print(
            f"    SMAP数据: {len(self.all_smap_dates)} 个日期; "
            f"{self.all_smap_dates[0]:%Y-%m-%d} → "
            f"{self.all_smap_dates[-1]:%Y-%m-%d}"
        )
        print(f"    C_point: {self.C_point}")
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

    def _get_sentinel1_value_loose(self, date_dt: datetime, r: int, c: int):
        """
        宽松版获取哨兵1值 - 正确识别NODATA=-9999，无数据时设为0

        返回: (vv, vh, vv_cov, vh_cov, angle)
        """
        S1_NODATA = FINAL_NODATA
        default_vv = 0.0
        default_vh = 0.0
        default_cov = 0.0
        default_angle = 0.0

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

        # VV：-9999 表示无数据，设为0
        if 'VV' in data and data['VV'] is not None:
            val = data['VV'][r, c]
            if val != S1_NODATA and val != self.s1_nodata_value and np.isfinite(val):
                vv = float(val)

        # VH：-9999 表示无数据，设为0
        if 'VH' in data and data['VH'] is not None:
            val = data['VH'][r, c]
            if val != S1_NODATA and val != self.s1_nodata_value and np.isfinite(val):
                vh = float(val)

        # VV_cov：cov >= 0 即有效
        if 'VV_cov' in data and data['VV_cov'] is not None:
            val = data['VV_cov'][r, c]
            if val >= 0 and np.isfinite(val):
                vv_cov = float(val)

        # VH_cov：cov >= 0 即有效
        if 'VH_cov' in data and data['VH_cov'] is not None:
            val = data['VH_cov'][r, c]
            if val >= 0 and np.isfinite(val):
                vh_cov = float(val)

        # angle：-9999 表示无数据，设为0
        if 'angle' in data and data['angle'] is not None:
            val = data['angle'][r, c]
            if val != S1_NODATA and val != self.s1_nodata_value and np.isfinite(val):
                angle = float(val)

        return vv, vh, vv_cov, vh_cov, angle
    
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
    def _get_smap_value_loose(self, date_dt: datetime, r: int, c: int):
        """
        宽松版获取SMAP值 - 正确识别无效值-9999，无数据时用250填充

        返回: (tbv, tbh)
        """
        SMAP_NODATA = FINAL_NODATA
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

        # TBV：只有不是 -9999 才算有效
        if 'TBV' in data and data['TBV'] is not None:
            val = data['TBV'][r, c]
            if val != SMAP_NODATA and np.isfinite(val):
                tbv = float(val)

        # TBH：只有不是 -9999 才算有效
        if 'TBH' in data and data['TBH'] is not None:
            val = data['TBH'][r, c]
            if val != SMAP_NODATA and np.isfinite(val):
                tbh = float(val)

        return tbv, tbh
    
    def _get_smap_mask_loose(self, date_dt: datetime, r: int, c: int, pol: str = 'V') -> float:
        """
        宽松版获取 SMAP mask
        mask 值：0=无效，1=有效（或其他正数表示有效）
        当亮温值为 -9999 时，mask 也会是 0
        """
        SMAP_NODATA = FINAL_NODATA

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
    
    def _load_station_data(self, station_csv: Path):
        """加载站点数据并构建索引"""
        print(f"\n加载站点数据...")
        
        station_data_sources = []
        
        if station_csv.exists():
            station_data_sources.append(station_csv)
        
        extra_data_sources = ["long_comb.csv", "long_comb2.csv"]
        
        for data_file in extra_data_sources:
            data_path = station_csv.parent / data_file
            if data_path.exists():
                station_data_sources.append(data_path)
                print(f"  发现额外数据源: {data_file}")
        
        if not station_data_sources:
            raise FileNotFoundError(f"站点数据文件不存在: {station_csv}")
        
        all_dataframes = []
        
        for data_source in station_data_sources:
            try:
                print(f"\n  处理数据源: {data_source.name}")
                
                if data_source.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(data_source, engine='openpyxl')
                elif data_source.suffix.lower() == '.csv':
                    try:
                        df = pd.read_csv(data_source, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(data_source, encoding='gbk')
                        except UnicodeDecodeError:
                            df = pd.read_csv(data_source, encoding='latin1')
                else:
                    print(f"  警告: 不支持的格式: {data_source.suffix}")
                    continue
                
                column_mapping = {
                    'longtitude': 'longitude',
                    'lon': 'longitude',
                    'lng': 'longitude',
                    'long': 'longitude',
                    'latitude': 'latitude',
                    'lat': 'latitude',
                    'swe': 'swe',
                    'date': 'date',
                    'station_id': 'station_id'
                }
                
                df = df.rename(columns=lambda x: column_mapping.get(str(x).strip().lower(), x))
                
                required_cols = ['station_id', 'date', 'swe', 'longitude', 'latitude']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"    警告: 缺少必要列: {missing_cols}")
                    continue
                
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
                df = df.dropna(subset=['date'])
                
                if isinstance(self.year_target, list):
                    df = df[df['date'].dt.year.isin(self.year_target)].copy()
                else:
                    df = df[df['date'].dt.year == self.year_target].copy()
                
                if len(df) > 0:
                    all_dataframes.append(df)
                    
            except Exception as e:
                print(f"    处理数据源 {data_source.name} 失败: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError(f"没有找到 {self.year_target} 年的有效数据")
        
        if len(all_dataframes) > 1:
            print(f"\n  合并 {len(all_dataframes)} 个数据源...")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
        else:
            combined_df = all_dataframes[0]
        
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['station_id', 'date', 'longitude', 'latitude'])
        after_dedup = len(combined_df)
        print(f"  去重: {before_dedup} -> {after_dedup} 行")
        
        print(f"  转换经纬度为像素坐标...")
        combined_df['row'], combined_df['col'] = self._lonlat_to_pixel(
            combined_df['longitude'].values, 
            combined_df['latitude'].values
        )
        
        print(f"\n  【按日期和像元聚合】")
        aggregated_df = self._aggregate_stations_by_pixel(combined_df)
        
        self.meta_index = []
        self.station_set = set()
        
        valid_count = 0
        boundary_count = 0
        invalid_date_count = 0
        
        print(f"\n  构建样本索引...")
        for idx, row in aggregated_df.iterrows():
            dt = row['date']
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            
            r = int(row['row'])
            c = int(row['col'])
            swe_value = float(row['swe'])
            station_count = int(row['station_count'])
            stations = row['stations']
            
            if not (0 <= r < self.H and 0 <= c < self.W):
                boundary_count += 1
                continue
            
            dt = (
                pd.Timestamp(dt)
                .normalize()
                .to_pydatetime()
            )

            if dt not in self.date_to_index:
                invalid_date_count += 1
                raise RuntimeError(
                    "站点原始观测日期不在完整日时间轴中："
                    f"station={stations}, "
                    f"date={dt:%Y-%m-%d}, "
                    f"row={r}, col={c}"
                )

            # 关键约束：模型特征日期始终等于原始站点观测日期。
            feature_date = dt

            self.meta_index.append({
                'feature_date': feature_date,      # 与原始站点日期严格一致
                'label_date': dt,                  # 原始站点观测日期
                'day_gap': 0,                      # 禁止整样本日期折叠
                'row': r,
                'col': c,
                'swe': swe_value,
                'station_id': stations,
                'station_count': station_count,
                'original_longitude': row['longitude'],
                'original_latitude': row['latitude']
            })
            
            for sid in str(stations).split(','):
                self.station_set.add(sid)
            
            valid_count += 1
            
            if valid_count % 1000 == 0:
                print(f"    已处理 {valid_count} 个样本...")
        
        collapsed_date_samples = [
            meta
            for meta in self.meta_index
            if (
                pd.Timestamp(meta["feature_date"]).normalize()
                != pd.Timestamp(meta["label_date"]).normalize()
                or int(meta.get("day_gap", 0)) != 0
            )
        ]

        if collapsed_date_samples:
            example = collapsed_date_samples[0]
            raise RuntimeError(
                "仍检测到站点日期折叠："
                f"count={len(collapsed_date_samples)}, "
                f"example={example}"
            )

        print(
            "  ✅ 站点日期完整性检查通过："
            f"{len(self.meta_index)}个样本全部满足"
            "feature_date == label_date，day_gap=0"
        )

        print(f"\n数据统计:")
        print(f"  总样本数: {len(aggregated_df)}")
        print(f"  有效样本数: {valid_count}")
        print(f"  边界外点数: {boundary_count}")
        print(f"  无效日期点: {invalid_date_count}")
        print(f"  涉及站点数: {len(self.station_set)}")
        
        dup_samples = [m for m in self.meta_index if m['station_count'] > 1]
        if dup_samples:
            print(f"\n  【重复像元统计】")
            print(f"    包含重复像元的样本: {len(dup_samples)}")
            print(f"    平均每个样本有 {np.mean([m['station_count'] for m in dup_samples]):.1f} 个站点")
        
        if valid_count == 0:
            raise ValueError("没有有效样本数据")
            
    def _build_point_features_station(self, date_dt: datetime, r: int, c: int) -> np.ndarray:
        """
        宽松版点特征构建（用于站点引导采样和微调）
        维度必须与严格版一致：18维

        无效值处理策略：
        - LS: 无效值 -> 0
        - 哨兵1: -9999 -> 0
        - SMAP: -9999 -> 250, mask=0标识无效
        """

        # 🔥 调试计数器
        if not hasattr(self, '_point_fail_stats'):
            self._point_fail_stats = {
                'total_attempts': 0,
                'ls_failed': 0,
                's1_failed': 0,
                'smap_failed': 0,
                'dimension_failed': 0,
                'microwave_warnings': 0,
                'success': 0,
                'reasons': {}
            }

        self._point_fail_stats['total_attempts'] += 1

        point_features = []

        # ============ 1. LS特征 (6个) - 与预训练按年份取年度合成 ============
        ls_ok = True
        ls_array = None
        if isinstance(getattr(self, "ls_data", None), dict):
            ls_array = self.ls_data.get(
                int(date_dt.year),
                getattr(self, "ls_data_default", None),
            )
        else:
            ls_array = getattr(self, "ls_data_default", None)

        if ls_array is not None:
            for i in range(min(6, ls_array.shape[0])):
                val = ls_array[i, r, c]
                if (not np.isfinite(val)) or val == FINAL_NODATA:
                    val = 0.0
                point_features.append(float(val))
        else:
            self._point_fail_stats['ls_failed'] += 1
            for _ in range(6):
                point_features.append(0.0)

        # ============ 2. 哨兵1特征 (5个) - 无数据时用0填充 ============
        s1_vv, s1_vh, s1_vv_cov, s1_vh_cov, s1_angle = self._get_sentinel1_value_loose(date_dt, r, c)
        point_features.append(float(s1_vv))
        point_features.append(float(s1_vh))
        point_features.append(float(s1_vv_cov) if s1_vv_cov >= 0 else 0.0)
        point_features.append(float(s1_vh_cov) if s1_vh_cov >= 0 else 0.0)
        point_features.append(float(s1_angle) if np.isfinite(s1_angle) and s1_angle != FINAL_NODATA else 0.0)

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



        # ============ 转换为 numpy 数组 ============
        point_feats_array = np.array(point_features, dtype=np.float32)
        point_feats_array = np.nan_to_num(point_feats_array, nan=0.0)

        # ============ 维度检查 ============
        if len(point_features) != 18:
            reason = f"维度错误: 实际{len(point_features)} != 18"
            self._point_fail_stats['dimension_failed'] += 1
            self._point_fail_stats['reasons'][reason] = self._point_fail_stats['reasons'].get(reason, 0) + 1

            if self._point_fail_stats['dimension_failed'] <= 5:
                print(f"\n  [点特征调试] 维度错误:")
                print(f"    日期: {date_dt.strftime('%Y-%m-%d')}")
                print(f"    位置: ({r}, {c})")
                print(f"    实际维度: {len(point_features)}")
                print(f"    各组件维度:")
                print(f"      LS: 6")
                print(f"      哨兵1: 5")
                print(f"      SMAP TB: 2")
                print(f"      SMAP mask: 2")
                print(f"      经纬度: 2")
                print(f"      DOY: 1")
                print(f"      总和: 18")
                print(f"    实际各组件值数量: {[len(point_features[:6]), len(point_features[6:11]), len(point_features[11:13]), len(point_features[13:15]), len(point_features[15:17]), len(point_features[17:])]}")
            return None

        # ============ 🔥 微波有效性检查 - 改为警告而不是拒绝 ============
        s1_vv_valid = (point_features[6] != 0.0)
        s1_vh_valid = (point_features[7] != 0.0)
        s1_valid = s1_vv_valid or s1_vh_valid

        smap_v_valid = (point_features[13] == 1.0)
        smap_h_valid = (point_features[14] == 1.0)
        smap_valid = smap_v_valid or smap_h_valid

        if not (s1_valid or smap_valid):
            # 只累计，不默认逐样本打印
            self._point_fail_stats['microwave_warnings'] += 1

            reason = "microwave_all_invalid"
            self._point_fail_stats['reasons'][reason] = (
                self._point_fail_stats['reasons'].get(reason, 0) + 1
            )

            n_warn = self._point_fail_stats['microwave_warnings']

            if self.verbose_point_debug and n_warn <= self.microwave_warning_print_limit:
                print(f"\n  [⚠️ 微波数据警告] 第{n_warn}次")
                print(f"    日期: {date_dt.strftime('%Y-%m-%d')}")
                print(f"    位置: ({r}, {c})")
                print(
                    f"    哨兵1: VV={point_features[6]:.2f}, "
                    f"VH={point_features[7]:.2f}, "
                    f"angle={point_features[10]:.2f}"
                )
                print(
                    f"    SMAP: TBV={point_features[11]:.2f}, "
                    f"TBH={point_features[12]:.2f}, "
                    f"mask_V={point_features[13]:.1f}, "
                    f"mask_H={point_features[14]:.1f}"
                )
                print("    → 将使用默认值(0/250)继续训练")

            elif (
                self.verbose_point_debug
                and self.microwave_warning_print_limit > 0
                and n_warn == self.microwave_warning_print_limit + 1
            ):
                print(
                    f"\n  [⚠️ 微波数据警告] 已超过 "
                    f"{self.microwave_warning_print_limit} 次，后续同类警告静默累计。"
                )

        self._point_fail_stats['success'] += 1

        if (
            self.verbose_point_debug
            and self.point_stats_interval > 0
            and self._point_fail_stats['success'] % self.point_stats_interval == 0
            and self._point_fail_stats['success'] > 0
        ):
            print(
                f"\n  [点特征统计] 已处理 {self._point_fail_stats['success']} 个成功样本, "
                f"失败: 维度={self._point_fail_stats['dimension_failed']}, "
                f"微波警告={self._point_fail_stats['microwave_warnings']}"
            )

        return point_feats_array
            
            
    def _build_spatial_features_station(self, date_dt: datetime, r: int, c: int) -> np.ndarray:
        """
        宽松版卷积特征 - 添加无效值检查
        无效值（-9999）先转为 NaN，再进行插值
        用于微调脚本中的站点样本特征构建
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
            "chelsa_sfxwind": FINAL_NODATA,
            "lst": FINAL_NODATA,
            "rh": FINAL_NODATA,
            "pr": FINAL_NODATA,
        }
        CLAMDAY_INVALID = FINAL_NODATA

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
                # 如果全部是 NaN，使用全局有效均值填充
                if np.all(np.isnan(patch)):
                    global_valid = var_arr[np.isfinite(var_arr) & (var_arr != invalid_val)]
                    if len(global_valid) > 0:
                        global_mean = np.mean(global_valid)
                        patch = np.full_like(patch, global_mean)
                        print(f"    ⚠ {var}: patch全NaN，使用全局均值 {global_mean:.4f} 填充")
                    else:
                        patch = np.zeros_like(patch)
                        print(f"    ⚠ {var}: patch全NaN且无全局有效值，使用0填充")
                else:
                    # 有部分有效值，进行插值
                    patch = self._interpolate_nan_patch(patch)
                    # 🔥 插值后检查是否还有NaN
                    if np.any(np.isnan(patch)):
                        nan_count = np.isnan(patch).sum()
                        print(f"    ⚠ {var}: 插值后仍有 {nan_count} 个NaN，使用中位数填充")
                        median_val = np.nanmedian(patch)
                        if np.isnan(median_val):
                            median_val = 0.0
                        patch = np.nan_to_num(patch, nan=median_val)

            conv_features.append(patch)

        # 2. clamday
        clamday_patch = self.clamday_data[r0:r1, c0:c1]
        clamday_patch = self._resize_to_standard(clamday_patch, self.P, self.P)

        clamday_patch = np.where(clamday_patch == CLAMDAY_INVALID, np.nan, clamday_patch)
        if np.any(np.isnan(clamday_patch)):
            if np.all(np.isnan(clamday_patch)):
                global_mean = np.nanmean(self.clamday_data)
                if np.isnan(global_mean):
                    global_mean = 0.0
                clamday_patch = np.full_like(clamday_patch, global_mean)
                print(f"    ⚠ clamday: patch全NaN，使用全局均值 {global_mean:.4f} 填充")
            else:
                clamday_patch = self._interpolate_nan_patch(clamday_patch)
                if np.any(np.isnan(clamday_patch)):
                    median_val = np.nanmedian(clamday_patch)
                    clamday_patch = np.nan_to_num(clamday_patch, nan=median_val if not np.isnan(median_val) else 0.0)
        conv_features.append(clamday_patch)

        # 3. DEM 波段
        for idx, dem_band in enumerate(self.dem_data):
            dem_patch = dem_band[r0:r1, c0:c1]
            dem_patch = self._resize_to_standard(dem_patch, self.P, self.P)

            dem_patch = np.where(dem_patch == -9999, np.nan, dem_patch)
            if np.any(np.isnan(dem_patch)):
                if np.all(np.isnan(dem_patch)):
                    global_mean = np.nanmean(dem_band)
                    if np.isnan(global_mean):
                        global_mean = 0.0
                    dem_patch = np.full_like(dem_patch, global_mean)
                    print(f"    ⚠ DEM_band{idx}: patch全NaN，使用全局均值 {global_mean:.2f} 填充")
                else:
                    dem_patch = self._interpolate_nan_patch(dem_patch)
                    if np.any(np.isnan(dem_patch)):
                        median_val = np.nanmedian(dem_patch)
                        dem_patch = np.nan_to_num(dem_patch, nan=median_val if not np.isnan(median_val) else 0.0)
            conv_features.append(dem_patch)

        try:
            conv_patch = np.stack(conv_features, axis=0)

            # 🔥 最终检查：如果还有NaN，报错而不是静默填0
            if np.any(np.isnan(conv_patch)):
                nan_channels = np.where(np.isnan(conv_patch).any(axis=(1,2)))[0]
                print(f"  ❌ 错误: 卷积特征仍有NaN残留，通道索引: {nan_channels}")
                print(f"     日期: {date_dt}, 位置: ({r}, {c})")
                # 打印每个通道的NaN数量
                for ch in nan_channels:
                    nan_count = np.isnan(conv_patch[ch]).sum()
                    print(f"     通道 {ch}: {nan_count} 个NaN")
                return None  # 返回None，让上层跳过该样本

            return conv_patch
        except Exception as e:
            print(f"  ❌ 堆叠卷积特征失败: {e}")
            return None
    
    def _aggregate_stations_by_pixel(self, df):
        """按日期和像元聚合站点数据 - 多站点时合并为一个样本（SWE取平均）"""
        print(f"\n  【聚合前】")
        print(f"    记录数: {len(df)}")
        print(f"    唯一(日期,行,列)组合数: {df.groupby(['date', 'row', 'col']).ngroups}")

        pixel_groups = df.groupby(['date', 'row', 'col'])

        aggregated_records = []
        duplicate_pixels = 0
        total_stations_in_dup = 0

        for (date, row, col), group in pixel_groups:
            stations = group['station_id'].unique()
            station_count = len(group)

            if station_count > 1:
                # 🔥 多站点：合并为一个样本（SWE取平均）
                swe_values = group['swe'].values
                swe_mean = np.mean(swe_values)
                swe_std = np.std(swe_values)

                duplicate_pixels += 1
                total_stations_in_dup += station_count

                if duplicate_pixels <= 10:
                    print(f"      📍 多站点像元: 日期 {date}, 像元 ({row},{col}), {station_count} 个站点 {list(stations)}")
                    print(f"        SWE值: {swe_values}")
                    print(f"        平均值: {swe_mean:.2f} ± {swe_std:.2f}")
                    print(f"        → 合并为 1 个样本 (SWE={swe_mean:.2f})")

                # 🔥 关键修改：合并成一个样本
                first_row = group.iloc[0]
                aggregated_records.append({
                    'date': date,
                    'row': row,
                    'col': col,
                    'swe': swe_mean,                    # 平均值
                    'swe_std': swe_std,
                    'station_count': station_count,     # 记录包含几个站点
                    'stations': ','.join(map(str, stations)),  # 所有站点ID用逗号连接
                    'longitude': first_row['longitude'],
                    'latitude': first_row['latitude'],
                    'original_swe_values': swe_values.tolist(),
                    'is_aggregated': True,
                    'source_stations': ','.join(map(str, stations))
                })
            else:
                # 单站点：正常处理
                record = group.iloc[0].to_dict()
                record['swe_std'] = 0
                record['station_count'] = 1
                record['stations'] = str(record['station_id'])
                record['original_swe_values'] = [record['swe']]
                record['is_aggregated'] = False
                record['source_stations'] = record['stations']
                aggregated_records.append(record)

        aggregated_df = pd.DataFrame(aggregated_records)

        print(f"\n  【聚合结果】")
        print(f"    聚合前记录数: {len(df)}")
        print(f"    聚合后记录数: {len(aggregated_df)}")
        print(f"    多站点像元数: {duplicate_pixels} (涉及 {total_stations_in_dup} 个站点记录)")

        if duplicate_pixels > 0:
            dup_df = aggregated_df[aggregated_df['is_aggregated'] == True]
            print(f"    多站点像元合并为 {len(dup_df)} 个样本")
            print(f"    平均每个多站点像元合并 {len(dup_df)/duplicate_pixels:.1f} 个样本")

        return aggregated_df
    
    def _lonlat_to_pixel(self, lon: np.ndarray, lat: np.ndarray):
        """经纬度转换为像素坐标"""
        rows = []
        cols = []
        
        for lng, lt in zip(lon, lat):
            col, row = ~self.transform * (lng, lt)
            rows.append(int(row))
            cols.append(int(col))
        
        return np.array(rows), np.array(cols)
    
    def _precompute_unmixing(self, n_endmembers=5):
        """预计算所有像元的光谱解混丰度"""
        try:
            from sklearn.cluster import KMeans
            from scipy.optimize import nnls
            
            n_bands, H, W = self.ls_data.shape
            
            pixels = self.ls_data.reshape(n_bands, -1).T
            valid_mask = ~np.isnan(pixels).any(axis=1)
            valid_pixels = pixels[valid_mask]
            
            if len(valid_pixels) < n_endmembers * 10:
                print(f"    警告: 有效像素太少 ({len(valid_pixels)}), 使用默认丰度")
                self.abundance_maps = np.zeros((n_endmembers, H, W), dtype=np.float32)
                self.abundance_maps[0, :, :] = 1.0
                return
            
            print(f"    有效像素数: {len(valid_pixels)}")
            
            kmeans = KMeans(n_clusters=n_endmembers, random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            
            self.endmembers = kmeans.cluster_centers_
            
            print("    正在计算丰度图...")
            self.abundance_maps = np.zeros((n_endmembers, H, W), dtype=np.float32)
            
            for i in range(H):
                for j in range(W):
                    spectrum = self.ls_data[:, i, j]
                    if not np.isnan(spectrum).any():
                        fractions, _ = nnls(self.endmembers.T, spectrum)
                        if fractions.sum() > 0:
                            fractions = fractions / fractions.sum()
                        self.abundance_maps[:, i, j] = fractions
            
            print(f"    丰度图形状: {self.abundance_maps.shape}")
            print(f"    丰度范围: [{self.abundance_maps.min():.3f}, {self.abundance_maps.max():.3f}]")
            
        except Exception as e:
            print(f"    光谱解混失败: {e}")
            n_endmembers = 5
            self.abundance_maps = np.zeros((n_endmembers, self.H, self.W), dtype=np.float32)
            self.abundance_maps[0, :, :] = 1.0
    
    def _compute_minmax(self):
        """计算归一化参数 - 支持多波段DEM (18维版本，与预训练对齐)"""
        print(f"\n计算归一化参数...")
        print(f"微调模式: {self.fine_tune_mode}")

        # ============ 1. 卷积特征的统计量 ============
        conv_mins = []
        conv_maxs = []

        # 动态卷积特征
        for var in CONV_VARS:
            if var in self.conv_dyn_data:
                arr = self.conv_dyn_data[var]
                valid_data = arr[np.isfinite(arr) & (arr != FINAL_NODATA)]
                if len(valid_data) > 0:
                    min_val = float(np.min(valid_data))
                    max_val = float(np.max(valid_data))
                    conv_mins.append(min_val)
                    conv_maxs.append(max_val)
                else:
                    conv_mins.append(0.0)
                    conv_maxs.append(1.0)
            else:
                conv_mins.append(0.0)
                conv_maxs.append(1.0)
            print(f"  {var}: [{conv_mins[-1]:.4f}, {conv_maxs[-1]:.4f}]")

        # clamday
        if self.clamday_data is not None:
            valid_data = self.clamday_data[
                np.isfinite(self.clamday_data) & (self.clamday_data != FINAL_NODATA)
            ]
            if len(valid_data) > 0:
                conv_mins.append(float(np.min(valid_data)))
                conv_maxs.append(float(np.max(valid_data)))
            else:
                conv_mins.append(0.0)
                conv_maxs.append(1.0)
            print(f"  clamday: [{conv_mins[-1]:.4f}, {conv_maxs[-1]:.4f}]")
        else:
            conv_mins.append(0.0)
            conv_maxs.append(1.0)
            print(f"  clamday: 无数据，使用默认[0.0, 1.0]")

        # 🔥 动态添加所有 DEM 波段
        for i, dem_band in enumerate(self.dem_data):
            if dem_band is not None:
                valid_data = dem_band[
                    np.isfinite(dem_band) & (dem_band != FINAL_NODATA)
                ]
                if len(valid_data) > 0:
                    conv_mins.append(float(np.min(valid_data)))
                    conv_maxs.append(float(np.max(valid_data)))
                else:
                    conv_mins.append(0.0)
                    conv_maxs.append(1.0)
                print(f"  DEM_band{i}: [{conv_mins[-1]:.4f}, {conv_maxs[-1]:.4f}]")
            else:
                conv_mins.append(0.0)
                conv_maxs.append(1.0)
                print(f"  DEM_band{i}: 无数据，使用默认[0.0, 1.0]")

        self.conv_min = np.array(conv_mins, dtype=np.float32)
        self.conv_max = np.array(conv_maxs, dtype=np.float32)
        self.C_conv = len(self.conv_min)

        print(f"\n✅ 卷积特征维度统计:")
        print(f"  动态特征: {len(CONV_VARS)}")
        print(f"  Clamday: 1")
        print(f"  DEM波段: {len(self.dem_data)}")
        print(f"  总通道数 C_conv: {self.C_conv}")

        # ============ 2. 点特征的统计量 (18维) ============
        print(f"\n【点特征统计 - 18维】")
        point_mins = []
        point_maxs = []

        # 2.1 LS特征 (6个)
        if hasattr(self, 'ls_data_default') and self.ls_data_default is not None:
            num_ls_bands = min(6, self.ls_data_default.shape[0])
            print(f"  LS特征: {num_ls_bands} 个波段")

            for i in range(num_ls_bands):
                band_data = self.ls_data_default[i]
                valid_data = band_data[
                    np.isfinite(band_data) & (band_data != FINAL_NODATA)
                ]
                if len(valid_data) > 0:
                    min_val = float(np.min(valid_data))
                    max_val = float(np.max(valid_data))
                    point_mins.append(min_val)
                    point_maxs.append(max_val)
                else:
                    point_mins.append(0.0)
                    point_maxs.append(1.0)
                print(f"    LS波段{i+1}: [{point_mins[-1]:.4f}, {point_maxs[-1]:.4f}]")
        else:
            for i in range(6):
                point_mins.append(0.0)
                point_maxs.append(1.0)
            print(f"  LS特征: 数据缺失，使用6个默认波段[0.0, 1.0]")

        # 2.2 哨兵1特征 (5个: VV, VH, VV_cov, VH_cov, angle)
        print(f"\n  哨兵1特征 (基于站点样本统计):")

        s1_vv_values = []
        s1_vh_values = []
        s1_vv_cov_values = []
        s1_vh_cov_values = []
        s1_angle_values = []

        total_samples = len(self.meta_index)
        print(f"    总样本数: {total_samples}")

        for idx, meta in enumerate(self.meta_index):
            if (idx + 1) % 5000 == 0:
                print(f"    已处理 {idx + 1}/{total_samples} 个样本")

            try:
                date = meta['feature_date']  # 🔥 使用 feature_date
                r = meta['row']
                c = meta['col']

                s1_vv, s1_vh, s1_vv_cov, s1_vh_cov, s1_angle = self._get_sentinel1_value_loose(date, r, c)

                if s1_vv != 0.0 and np.isfinite(s1_vv):
                    s1_vv_values.append(s1_vv)
                if s1_vh != 0.0 and np.isfinite(s1_vh):
                    s1_vh_values.append(s1_vh)
                if s1_vv_cov >= 0 and np.isfinite(s1_vv_cov):
                    s1_vv_cov_values.append(s1_vv_cov)
                if s1_vh_cov >= 0 and np.isfinite(s1_vh_cov):
                    s1_vh_cov_values.append(s1_vh_cov)
                if s1_angle != 0.0 and np.isfinite(s1_angle):
                    s1_angle_values.append(s1_angle)

            except Exception as e:
                continue

        print(f"    处理完成")

        # S1_VV
        if s1_vv_values:
            min_val = min(s1_vv_values)
            max_val = max(s1_vv_values)
        else:
            min_val = -25.0
            max_val = 25.0
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_VV: [{min_val:.2f}, {max_val:.2f}] (采样{len(s1_vv_values)}个)")

        # S1_VH
        if s1_vh_values:
            min_val = min(s1_vh_values)
            max_val = max(s1_vh_values)
        else:
            min_val = -30.0
            max_val = 20.0
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_VH: [{min_val:.2f}, {max_val:.2f}] (采样{len(s1_vh_values)}个)")

        # S1_VV_cov
        if s1_vv_cov_values:
            min_val = min(s1_vv_cov_values)
            max_val = max(s1_vv_cov_values)
        else:
            min_val = 0.0
            max_val = 0.5
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_VV_cov: [{min_val:.4f}, {max_val:.4f}] (采样{len(s1_vv_cov_values)}个)")

        # S1_VH_cov
        if s1_vh_cov_values:
            min_val = min(s1_vh_cov_values)
            max_val = max(s1_vh_cov_values)
        else:
            min_val = 0.0
            max_val = 0.5
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_VH_cov: [{min_val:.4f}, {max_val:.4f}] (采样{len(s1_vh_cov_values)}个)")

        # S1_angle
        if s1_angle_values:
            min_val = min(s1_angle_values)
            max_val = max(s1_angle_values)
        else:
            min_val = 0.0
            max_val = 60.0
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    S1_angle: [{min_val:.2f}, {max_val:.2f}] (采样{len(s1_angle_values)}个)")

        # 2.3 SMAP特征 (2个亮温 + 2个mask)
        print(f"\n  SMAP特征 (基于站点样本统计):")

        smap_tbv_values = []
        smap_tbh_values = []
        smap_mask_v_values = []
        smap_mask_h_values = []

        for idx, meta in enumerate(self.meta_index):
            try:
                date = meta['feature_date']  # 🔥 使用 feature_date
                r = meta['row']
                c = meta['col']

                smap_tbv, smap_tbh = self._get_smap_value_loose(date, r, c)
                mask_v = self._get_smap_mask_loose(date, r, c, 'V')
                mask_h = self._get_smap_mask_loose(date, r, c, 'H')

                if smap_tbv != 250.0 and np.isfinite(smap_tbv):
                    smap_tbv_values.append(smap_tbv)
                if smap_tbh != 250.0 and np.isfinite(smap_tbh):
                    smap_tbh_values.append(smap_tbh)
                if np.isfinite(mask_v):
                    smap_mask_v_values.append(mask_v)
                if np.isfinite(mask_h):
                    smap_mask_h_values.append(mask_h)

            except Exception as e:
                continue

        # SMAP_TBV
        if smap_tbv_values:
            min_val = min(smap_tbv_values)
            max_val = max(smap_tbv_values)
        else:
            min_val = 200.0
            max_val = 320.0
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    SMAP_TBV: [{min_val:.2f}, {max_val:.2f}] (采样{len(smap_tbv_values)}个)")

        # SMAP_TBH
        if smap_tbh_values:
            min_val = min(smap_tbh_values)
            max_val = max(smap_tbh_values)
        else:
            min_val = 200.0
            max_val = 320.0
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    SMAP_TBH: [{min_val:.2f}, {max_val:.2f}] (采样{len(smap_tbh_values)}个)")

        # SMAP_mask_V
        if smap_mask_v_values:
            min_val = min(smap_mask_v_values)
            max_val = max(smap_mask_v_values)
        else:
            min_val = 0.0
            max_val = 1.0
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    SMAP_mask_V: [{min_val:.4f}, {max_val:.4f}] (采样{len(smap_mask_v_values)}个)")

        # SMAP_mask_H
        if smap_mask_h_values:
            min_val = min(smap_mask_h_values)
            max_val = max(smap_mask_h_values)
        else:
            min_val = 0.0
            max_val = 1.0
        point_mins.append(min_val)
        point_maxs.append(max_val)
        print(f"    SMAP_mask_H: [{min_val:.4f}, {max_val:.4f}] (采样{len(smap_mask_h_values)}个)")

        # 2.4 空间特征 (2个)
        print(f"\n  空间特征:")
        lon_values = []
        lat_values = []

        for meta in self.meta_index:
            lon, lat = self._pixel_to_lonlat(meta['row'], meta['col'])
            lon_values.append(lon)
            lat_values.append(lat)

        if lon_values:
            lon_min = min(lon_values)
            lon_max = max(lon_values)
            lat_min = min(lat_values)
            lat_max = max(lat_values)

            self.lon_raw_min = lon_min
            self.lon_raw_max = lon_max
            self.lat_raw_min = lat_min
            self.lat_raw_max = lat_max

            point_mins.append(0.0)
            point_maxs.append(1.0)
            point_mins.append(0.0)
            point_maxs.append(1.0)
            print(f"    经度范围: [{lon_min:.4f}, {lon_max:.4f}]")
            print(f"    纬度范围: [{lat_min:.4f}, {lat_max:.4f}]")
            print(f"    经纬度归一化: [0.0, 1.0]")
        else:
            point_mins.extend([0.0, 0.0])
            point_maxs.extend([1.0, 1.0])
            print(f"    无经纬度数据，使用默认[0.0, 1.0]")

        # 2.5 时间特征 (1个)
        print(f"\n  时间特征:")
        doy_values = []
        for meta in self.meta_index:
            date_obj = meta['feature_date']  # 🔥 使用 feature_date
            if isinstance(date_obj, pd.Timestamp):
                date_obj = date_obj.to_pydatetime()
            doy_values.append(date_obj.timetuple().tm_yday)

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
            print(f"    无DOY数据，使用默认[0.0, 1.0]")



        # ============ 3. 最终维度检查和汇总 ============
        self.point_min = np.array(point_mins, dtype=np.float32)
        self.point_max = np.array(point_maxs, dtype=np.float32)
        self.C_point = len(self.point_min)

        # 预期维度: LS(6) + S1(5) + SMAP_TB(2) + SMAP_mask(2) + 经纬度(2) + DOY(1) = 18
        expected = 6 + 5 + 2 + 2 + 2 + 1

        print(f"\n【最终点特征维度: {self.C_point}】")
        print(f"  组成: LS(6) + S1(5) + SMAP_TB(2) + SMAP_mask(2) + 经纬度(2) + DOY(1) = {expected}")
        print(f"  顺序确认: LS(6) → S1_VV(1) → S1_VH(1) → S1_VV_cov(1) → S1_VH_cov(1) → S1_angle(1) → SMAP_TB(2) → SMAP_mask(2) → 经纬度(2) → DOY(1)")

        if self.C_point != expected:
            print(f"  ⚠ 警告: 实际维度{self.C_point} != 预期{expected}")

        # ============ 4. SWE统计 ============
        print(f"\n【SWE统计 - 全样本】")
        swe_values = [meta['swe'] for meta in self.meta_index]
        if swe_values:
            swe_values = np.array(swe_values)
            self.swe_min = float(np.min(swe_values))
            self.swe_max = float(np.max(swe_values))
            print(f"  SWE原始范围: [{self.swe_min:.2f}, {self.swe_max:.2f}] mm")
        else:
            self.swe_min = 0.0
            self.swe_max = 100.0
            print(f"  无SWE数据，使用默认[0.0, 100.0] mm")

        print(f"\n归一化参数统计:")
        print(f"  卷积特征维度: {self.C_conv}")
        print(f"  点特征维度: {self.C_point}")

        return
    
    def _resize_to_standard(self, data: np.ndarray, target_h: int, target_w: int):
        """调整到标准尺寸"""
        h, w = data.shape
        if h == target_h and w == target_w:
            return data
        
        result = np.zeros((target_h, target_w), dtype=data.dtype)
        
        h_start = max(0, (target_h - h) // 2)
        w_start = max(0, (target_w - w) // 2)
        h_end = min(h_start + h, target_h)
        w_end = min(w_start + w, target_w)
        
        result[h_start:h_end, w_start:w_end] = data[:h_end-h_start, :w_end-w_start]
        return result
    
    def _get_microwave_value(self, date: datetime, r: int, c: int):
        """获取微波特征值"""
        s1_vv, s1_vh = self.s1_nodata_value, self.s1_nodata_value
        smap_tbv, smap_tbh = self.smap_nodata_value, self.smap_nodata_value
        
        if self.all_s1_dates:
            closest_date = min(self.all_s1_dates, key=lambda d: abs((d - date).days))
            if abs((closest_date - date).days) <= self.s1_max_gap_days:
                if closest_date in self.s1_data:
                    if 'VV' in self.s1_data[closest_date]:
                        s1_vv = float(self.s1_data[closest_date]['VV'][r, c])
                    if 'VH' in self.s1_data[closest_date]:
                        s1_vh = float(self.s1_data[closest_date]['VH'][r, c])
        
        if self.all_smap_dates:
            closest_date = min(self.all_smap_dates, key=lambda d: abs((d - date).days))
            if abs((closest_date - date).days) <= self.smap_max_gap_days:
                if closest_date in self.smap_data:
                    if 'TBV' in self.smap_data[closest_date]:
                        smap_tbv = float(self.smap_data[closest_date]['TBV'][r, c])
                    if 'TBH' in self.smap_data[closest_date]:
                        smap_tbh = float(self.smap_data[closest_date]['TBH'][r, c])
        
        return s1_vv, s1_vh, smap_tbv, smap_tbh
    
    def _build_time_features(self, date: datetime):
        """构建时间特征"""
        day_of_year = date.timetuple().tm_yday
        doy_norm = (day_of_year - 1) / 365.0
        return np.array([doy_norm], dtype=np.float32)
    
    def _pixel_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        """像素坐标转经纬度"""
        x, y = self.transform * (col + 0.5, row + 0.5)
        lon, lat = self.transformer.transform(x, y)
        return lon, lat

    def _interpolate_nan_patch(self, patch: np.ndarray) -> np.ndarray:
        """使用scipy插值填充NaN"""
        if not self.fine_tune_mode or not np.isnan(patch).any():
            return patch
        
        x = np.arange(patch.shape[1])
        y = np.arange(patch.shape[0])
        xx, yy = np.meshgrid(x, y)
        
        valid_mask = ~np.isnan(patch)
        if not valid_mask.any():
            return np.zeros_like(patch)
        
        if np.sum(valid_mask) < 3:
            mean_value = np.nanmean(patch)
            if np.isnan(mean_value):
                mean_value = 0.0
            
            result = patch.copy()
            result[np.isnan(result)] = mean_value
            return result
        
        valid_points = np.column_stack([xx[valid_mask], yy[valid_mask]])
        valid_values = patch[valid_mask]
        
        unique_x = np.unique(valid_points[:, 0])
        unique_y = np.unique(valid_points[:, 1])
        
        if len(unique_x) == 1 or len(unique_y) == 1:
            interpolation_method = 'nearest'
        else:
            interpolation_method = 'linear'
        
        invalid_mask = np.isnan(patch)
        if not invalid_mask.any():
            return patch
        
        invalid_points = np.column_stack([xx[invalid_mask], yy[invalid_mask]])
        
        try:
            interpolated = griddata(valid_points, valid_values, invalid_points, 
                                    method=interpolation_method, fill_value=0.0)
            
            result = patch.copy()
            result[invalid_mask] = interpolated
            return result
            
        except Exception as e:
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

    def _get_microwave_value_with_interpolation(self, date: datetime, r: int, c: int):
        """获取微波特征值（带插值）"""
        s1_vv, s1_vh = self.s1_nodata_value, self.s1_nodata_value
        smap_tbv, smap_tbh = self.smap_nodata_value, self.smap_nodata_value
        
        if self.all_s1_dates:
            closest_date = min(self.all_s1_dates, key=lambda d: abs((d - date).days))
            day_gap = abs((closest_date - date).days)
            
            if day_gap <= self.s1_max_gap_days:
                if closest_date in self.s1_data:
                    if 'VV' in self.s1_data[closest_date]:
                        value = self.s1_data[closest_date]['VV'][r, c]
                        if value != self.s1_nodata_value and np.isfinite(value):
                            s1_vv = float(value)
                        else:
                            s1_vv = self._get_spatial_nearest_value(date, r, c, 'S1_VV')
                    
                    if 'VH' in self.s1_data[closest_date]:
                        value = self.s1_data[closest_date]['VH'][r, c]
                        if value != self.s1_nodata_value and np.isfinite(value):
                            s1_vh = float(value)
                        else:
                            s1_vh = self._get_spatial_nearest_value(date, r, c, 'S1_VH')
        
        if self.all_smap_dates:
            valid_smap_dates = []
            for d in self.all_smap_dates:
                if d in self.smap_data:
                    has_tbv = 'TBV' in self.smap_data[d]
                    has_tbh = 'TBH' in self.smap_data[d]
                    if has_tbv and has_tbh:
                        tbv_val = self.smap_data[d]['TBV'][r, c]
                        tbh_val = self.smap_data[d]['TBH'][r, c]
                        if (tbv_val != self.smap_nodata_value and np.isfinite(tbv_val) and
                            tbh_val != self.smap_nodata_value and np.isfinite(tbh_val)):
                            valid_smap_dates.append(d)
            
            if valid_smap_dates:
                closest_date = min(valid_smap_dates, key=lambda d: abs((d - date).days))
                
                if closest_date in self.smap_data:
                    if 'TBV' in self.smap_data[closest_date]:
                        value = self.smap_data[closest_date]['TBV'][r, c]
                        if value != self.smap_nodata_value and np.isfinite(value):
                            smap_tbv = float(value)
                    
                    if 'TBH' in self.smap_data[closest_date]:
                        value = self.smap_data[closest_date]['TBH'][r, c]
                        if value != self.smap_nodata_value and np.isfinite(value):
                            smap_tbh = float(value)
        
        if s1_vv == self.s1_nodata_value:
            s1_vv = self._get_alternative_microwave_value(date, r, c, 'S1_VV')
        if s1_vh == self.s1_nodata_value:
            s1_vh = self._get_alternative_microwave_value(date, r, c, 'S1_VH')
        
        return s1_vv, s1_vh, smap_tbv, smap_tbh
    
    def _get_alternative_microwave_value(self, date: datetime, r: int, c: int, band: str):
        """备选微波值获取方案"""
        if band in ['S1_VV', 'S1_VH']:
            data_source = self.s1_data
            dates_source = self.all_s1_dates
            band_key = 'VV' if band == 'S1_VV' else 'VH'
            nodata = self.s1_nodata_value
        else:
            data_source = self.smap_data
            dates_source = self.all_smap_dates
            band_key = 'TBV' if band == 'SMAP_TBV' else 'TBH'
            nodata = self.smap_nodata_value
        
        if not dates_source or not data_source:
            if band in ['S1_VV', 'S1_VH']:
                return 0.0
            else:
                return 250.0
        
        all_values = []
        for data_date, data_dict in data_source.items():
            if band_key in data_dict:
                value = data_dict[band_key][r, c]
                if value != nodata and np.isfinite(value):
                    all_values.append(float(value))
        
        if all_values:
            return float(np.median(all_values))
        
        all_layer_values = []
        for data_date, data_dict in data_source.items():
            if band_key in data_dict:
                layer = data_dict[band_key]
                valid_values = layer[
                    np.isfinite(layer) & (layer != nodata) & (layer != FINAL_NODATA)
                ]
                all_layer_values.extend(valid_values.flatten())
        
        if all_layer_values:
            return float(np.median(all_layer_values))
        
        if band in ['S1_VV', 'S1_VH']:
            return 0.0
        else:
            return 250.0
    
    def _get_spatial_nearest_value(self, date: datetime, r: int, c: int, band: str):
        """空间最近邻插值"""
        if band in ['S1_VV', 'S1_VH']:
            data_source = self.s1_data
            dates_source = self.all_s1_dates
            band_key = 'VV' if band == 'S1_VV' else 'VH'
            nodata = self.s1_nodata_value
        else:
            data_source = self.smap_data
            dates_source = self.all_smap_dates
            band_key = 'TBV' if band == 'SMAP_TBV' else 'TBH'
            nodata = self.smap_nodata_value
        
        if not dates_source:
            return 0.0
        
        closest_date = min(dates_source, key=lambda d: abs((d - date).days))
        
        if closest_date not in data_source or band_key not in data_source[closest_date]:
            return 0.0
        
        data_layer = data_source[closest_date][band_key]
        
        if 0 <= r < self.H and 0 <= c < self.W:
            value = data_layer[r, c]
            if value != nodata and np.isfinite(value):
                return float(value)
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    value = data_layer[nr, nc]
                    if value != nodata and np.isfinite(value):
                        return float(value)
        
        valid_values = data_layer[
            np.isfinite(data_layer) & (data_layer != nodata) & (data_layer != FINAL_NODATA)
        ]
        if len(valid_values) > 0:
            return float(np.mean(valid_values))
        
        return 0.0
    
    def _check_swe_discrepancies(self, threshold=30.0):
        """
        检查站点实测值与 FusedSWE 产品值差异过大的异常样本

        Args:
            threshold: 差异阈值（mm），默认 30mm
        """
        print("\n" + "="*70)
        print(f"🔍 【异常差异检查】站点实测 SWE vs FusedSWE 产品值 (差异阈值: {threshold}mm)")
        print("="*70)

        discrepancies = []
        valid_pairs = 0

        for meta in self.meta_index:
            date_dt = meta['feature_date']  # 🔥 改为 feature_date
            r = meta['row']
            c = meta['col']
            station_swe = float(meta['swe'])
            station_id = meta.get('station_id', 'Unknown')
            lon = meta.get('original_longitude', 0)
            lat = meta.get('original_latitude', 0)

            fused_swe = None
            # 获取对应的 FusedSWE 栅格值
            if date_dt in self.label_data:
                label_arr, label_nodata = self.label_data[date_dt]
                val = label_arr[r, c]
                if (label_nodata is None or val != label_nodata) and np.isfinite(val):
                    fused_swe = float(val)

            if fused_swe is not None:
                valid_pairs += 1
                abs_diff = abs(station_swe - fused_swe)

                # 如果绝对差值大于设定的阈值
                if abs_diff > threshold:
                    discrepancies.append({
                        'date': date_dt.strftime('%Y-%m-%d'),
                        'station_id': str(station_id),
                        'lon': lon,
                        'lat': lat,
                        'station_swe': station_swe,
                        'fused_swe': fused_swe,
                        'abs_diff': abs_diff,
                        'type': '产品低估' if station_swe > fused_swe else '产品高估'
                    })

        # 按差异大小降序排列
        discrepancies.sort(key=lambda x: x['abs_diff'], reverse=True)

        print(f"匹配到有效对比样本: {valid_pairs} 个")

        if valid_pairs > 0:
            print(f"发现差异 > {threshold}mm 的异常样本: {len(discrepancies)} 个 ({len(discrepancies)/valid_pairs*100:.1f}%)")

        if discrepancies:
            # 统计高估和低估的比例
            under_est = sum(1 for d in discrepancies if d['type'] == '产品低估')
            over_est = len(discrepancies) - under_est
            print(f"  ├─ FusedSWE 严重低估 (实测 > 产品): {under_est} 个")
            print(f"  └─ FusedSWE 严重高估 (实测 < 产品): {over_est} 个")

            print(f"\nTop 15 差异最大的样本:")
            print(f"{'日期':<12} {'站点ID':<10} {'经度':<8} {'纬度':<8} {'实测(mm)':<10} {'产品(mm)':<10} {'差值(mm)':<10} {'类型'}")
            print("-" * 85)
            for d in discrepancies[:15]:
                print(f"{d['date']:<12} {d['station_id']:<10} {d['lon']:<8.2f} {d['lat']:<8.2f} "
                      f"{d['station_swe']:<10.2f} {d['fused_swe']:<10.2f} {d['abs_diff']:<10.2f} {d['type']}")

        print("="*70 + "\n")
    
    
#     def _build_point_features(self, date_dt: datetime, r: int, c: int, 
#                               original_lon=None, original_lat=None, 
#                               grid_val=None, augment: bool = False) -> np.ndarray:
#         """
#         构建点特征 - 16维（新增：原产品值作为第16维）

#         Args:
#             date_dt: 日期
#             r, c: 行列坐标
#             original_lon: 原始经度
#             original_lat: 原始纬度
#             grid_val: 原产品值 (FusedSWE)
#             augment: 是否应用数据增强（训练时为True，验证/测试时为False）
#         """
#         point_features = []

#         # 1. LS特征 (6个)
#         if hasattr(self, 'ls_data'):
#             for i in range(self.ls_data.shape[0]):
#                 point_features.append(float(self.ls_data[i, r, c]))
#         else:
#             for _ in range(6): 
#                 point_features.append(0.0)

#         # 2 & 3. 微波特征 (4个)
#         s1_vv, s1_vh, smap_tbv, smap_tbh = self._get_microwave_value_with_interpolation(date_dt, r, c)
#         point_features.extend([float(s1_vv), float(s1_vh), float(smap_tbv), float(smap_tbh)])

#         # 4. 经纬度特征 (2个)
#         lon = original_lon if original_lon is not None else self._pixel_to_lonlat(r, c)[0]
#         lat = original_lat if original_lat is not None else self._pixel_to_lonlat(r, c)[1]

#         lon_norm = (lon + 180) / 360
#         lat_norm = (lat + 90) / 180
                    
#         # ============ 数据增强：坐标抖动和掩码 ============
#         if augment:
#             # 策略1: 坐标抖动 (Coordinate Jittering)
#             # 让坐标变得模糊，迫使模型学习物理规律而非死记位置
#             lon_norm += np.random.normal(0, self.coordinate_jitter_std)
#             lat_norm += np.random.normal(0, self.coordinate_jitter_std)

#             # 策略2: 坐标掩码 (Coordinate Masking)
#             # 随机将经纬度清零，强制模型依赖其他特征
#             if np.random.random() < self.coordinate_mask_prob:
#                 lon_norm = 0.5  # 归一化后的全局均值
#                 lat_norm = 0.5

#             # 策略3: 微波信号噪声增强
#             # 模拟仪器误差，防止过拟合特定波段
#             microwave_indices = [6, 7, 8, 9]  # S1_VV, S1_VH, SMAP_TBV, SMAP_TBH
#             for m_idx in microwave_indices:
#                 if m_idx < len(point_features):
#                     noise = np.random.normal(0, self.microwave_noise_std)
#                     point_features[m_idx] *= (1 + noise)
                    

#         point_features.extend([lon_norm, lat_norm])

#         # 5. 时间特征 (1个)
#         time_feats = self._build_time_features(date_dt)
#         point_features.extend(time_feats)

#         # 6. 物理累积特征 (2个)
#         cum_pr_30d = 0.0
#         cum_snow_30d = 0.0

#         if "pr" in self.conv_dyn_data and "lst" in self.conv_dyn_data:
#             date_idx = self.date_to_index.get(date_dt)
#             if date_idx is not None:
#                 start_idx = max(0, date_idx - 30)
#                 pr_history = self.conv_dyn_data["pr"][start_idx:date_idx + 1, r, c]
#                 temp_history = self.conv_dyn_data["lst"][start_idx:date_idx + 1, r, c]

#                 valid_days = min(len(pr_history), len(temp_history))
#                 for i in range(valid_days):
#                     p, t = pr_history[i], temp_history[i]
#                     if np.isfinite(p) and np.isfinite(t):
#                         cum_pr_30d += float(p)
#                         if t < 1.0:  # LST 单位摄氏度，阈值 1°C
#                             cum_snow_30d += float(p)

#         point_features.append(cum_pr_30d)
#         point_features.append(cum_snow_30d)

#         # ============ 7. 新增第16维：归一化后的原产品值 (FusedSWE) ============
#         if grid_val is not None:
#             # 获取当前点的 LST (用于物理公式)
#             date_idx = self.date_to_index.get(date_dt)
#             current_lst = 0.0
#             if date_idx is not None and "lst" in self.conv_dyn_data:
#                 current_lst = float(self.conv_dyn_data["lst"][date_idx, r, c])

#             # 调用物理模型转换
#             # 传入：原始产品值, 当前气温, 30天累积降水, 行列号(用于取DEM)
#             y_phys = self._build_transformed_physical_feature(
#                 grid_val, current_lst, cum_pr_30d, r, c
#             )
            
#             # 归一化转换后的物理指标
#             eps = 1e-6
#             y_phys_norm = (y_phys - self.swe_min) / (self.swe_max - self.swe_min + eps)
#             point_features.append(float(np.clip(y_phys_norm, 0.0, 1.0)))
#         else:
#             point_features.append(0.0)            

#         # ================================================================

#         point_feats_array = np.array(point_features, dtype=np.float32)

#         # 处理NaN值
#         point_feats_array = np.nan_to_num(point_feats_array, nan=0.0)

#         # 裁剪到合理范围（防止噪声导致超出归一化范围）
#         if augment:
#             point_feats_array = np.clip(point_feats_array, 0.0, 1.0)

#         # 维度检查：现在应该是16维
#         if len(point_features) != 16:
#             print(f"\n🚨 维度错误！所在文件: {self.__class__.__name__}")
#             print(f"实际维度: {len(point_features)}, 期望: 16")
#             print(f"完整特征: {point_features}")
#             import sys; sys.exit(1)

#         return point_feats_array
    
    
    
        
    def _build_spatial_patch(self, date: datetime, r: int, c: int):
        """构建空间特征patch - 支持多波段DEM"""
        if r < self.R or r >= self.H - self.R or c < self.R or c >= self.W - self.R:
            return None

        r0, r1 = r - self.R, r + self.R + 1
        c0, c1 = c - self.R, c + self.R + 1

        if date not in self.date_to_index:
            return None

        date_idx = self.date_to_index[date]

        conv_features = []
        has_nan = False

        # 1. 动态卷积特征
        for var in CONV_VARS:
            if var in self.conv_dyn_data:
                arr = self.conv_dyn_data[var]
                if date_idx < arr.shape[0]:
                    patch = arr[date_idx, r0:r1, c0:c1]
                    if patch.shape != (self.P, self.P):
                        patch = self._resize_to_standard(patch, self.P, self.P)

                    if np.isnan(patch).any():
                        has_nan = True
                        if self.fine_tune_mode:
                            patch = self._interpolate_nan_patch(patch)

                    conv_features.append(patch)
                else:
                    patch = np.zeros((self.P, self.P), dtype=np.float32)
                    conv_features.append(patch)
            else:
                patch = np.zeros((self.P, self.P), dtype=np.float32)
                conv_features.append(patch)

        # 2. clamday
        if self.clamday_data is not None:
            patch = self.clamday_data[r0:r1, c0:c1]
            patch = self._resize_to_standard(patch, self.P, self.P)

            if np.isnan(patch).any():
                has_nan = True
                if self.fine_tune_mode:
                    patch = self._interpolate_nan_patch(patch)

            conv_features.append(patch)
        else:
            patch = np.zeros((self.P, self.P), dtype=np.float32)
            conv_features.append(patch)

        # 3. 🔥 动态添加所有 DEM 波段
        for dem_band in self.dem_data:
            patch = dem_band[r0:r1, c0:c1]
            patch = self._resize_to_standard(patch, self.P, self.P)

            if np.isnan(patch).any():
                has_nan = True
                if self.fine_tune_mode:
                    patch = self._interpolate_nan_patch(patch)

            conv_features.append(patch)

        if has_nan and not self.fine_tune_mode:
            return None

        # 不再硬编码检查 len(conv_features) != 7
        # 动态计算期望的通道数
        expected_channels = len(CONV_VARS) + 1 + len(self.dem_data)
        if len(conv_features) != expected_channels:
            return None

        try:
            conv_patch = np.stack(conv_features, axis=0)

            if self.fine_tune_mode and np.isnan(conv_patch).any():
                conv_patch = np.nan_to_num(conv_patch, nan=0.0)

            return conv_patch
        except Exception as e:
            return None
    
    def __len__(self):
        return len(self.meta_index)

    def __getitem__(self, idx: int):
        """
        获取一个站点样本。

        返回：
            conv_t:          (C_conv, P, P)
            point_t:         (21,)
            y_t:             station SWE, normalized
            is_zero_t:       station SWE 是否 > 0
            grid_val_norm_t: 原始 FusedSWE 栅格值，normalized
            cur_idx:         实际样本索引

        产品值修正逻辑：
            correction_map 的 key 必须是：
                (str(station_id), "YYYY-MM-DD")

            correction_map 的 value 是：
                corrected SWE in mm

            当前 Clean-18D 不把产品值写入 point_t。
            grid_val_norm_t 作为独立返回值，仅用于产品对比或显式
            residual/gate 模型；普通模型仍只接收18维 point_t。
        """
        import pandas as pd
        import numpy as np
        import torch

        max_retry = 50
        cur_idx = idx

        # 多进程 DataLoader 中每个 worker 都有独立 Dataset 副本。
        # 只允许主进程/worker 0 打印一次样本级 DEBUG，避免重复刷屏。
        worker_info = torch.utils.data.get_worker_info()
        is_debug_worker = worker_info is None or worker_info.id == 0
        debug_enabled = bool(getattr(self, "verbose_point_debug", False)) and is_debug_worker

        # ============ 调试计数器 ============
        if not hasattr(self, "_debug_stats"):
            self._debug_stats = {
                "total_attempts": 0,
                "conv_failed": 0,
                "label_failed": 0,
                "point_failed": 0,
                "swe_failed": 0,
                "norm_failed": 0,
                "success": 0,
                "failed_samples": [],
            }
            self._debug_logged = False

        for retry in range(max_retry):
            self._debug_stats["total_attempts"] += 1

            # ============================================================
            # 1. 读取 meta 信息
            # ============================================================
            meta = self.meta_index[cur_idx]

            if "feature_date" in meta:
                feature_date = meta["feature_date"]
                label_date = meta.get("label_date", feature_date)
            else:
                feature_date = meta["date"]
                label_date = meta["date"]

            # 统一 feature_date 类型
            feature_date = pd.to_datetime(feature_date).to_pydatetime()
            label_date = pd.to_datetime(label_date).to_pydatetime()

            r = int(meta["row"])
            c = int(meta["col"])
            station_id = meta.get("station_id", "unknown")
            original_swe = float(meta["swe"])

            # ============================================================
            # 2. 构建卷积特征
            # ============================================================
            conv_patch = self._build_spatial_features_station(feature_date, r, c)

            if conv_patch is None:
                self._debug_stats["conv_failed"] += 1

                if debug_enabled and not self._debug_logged and retry < 5:
                    print(
                        f"  [DEBUG] 样本 {cur_idx} "
                        f"(站点:{station_id}, 日期:{feature_date.strftime('%Y-%m-%d')}) "
                        f"卷积特征构建失败"
                    )
                    print(f"     位置: ({r},{c}), SWE={original_swe}")

                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            # ============================================================
            # 3. 提取原始 FusedSWE 栅格值
            # ============================================================
            if feature_date not in self.label_data:
                self._debug_stats["label_failed"] += 1

                if debug_enabled and not self._debug_logged and retry < 5:
                    print(
                        f"  [DEBUG] 样本 {cur_idx} "
                        f"(站点:{station_id}, 日期:{feature_date.strftime('%Y-%m-%d')}) "
                        f"标签数据不存在"
                    )
                    print(f"     可用标签日期: {list(self.label_data.keys())[:5]}...")

                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            label_arr, label_nodata = self.label_data[feature_date]

            if r < 0 or r >= label_arr.shape[0] or c < 0 or c >= label_arr.shape[1]:
                self._debug_stats["label_failed"] += 1

                if debug_enabled and not self._debug_logged and retry < 5:
                    print(
                        f"  [DEBUG] 样本 {cur_idx}: 行列超出范围: "
                        f"({r},{c}), 标签形状: {label_arr.shape}"
                    )

                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            grid_val = float(label_arr[r, c])

            is_invalid = False
            if label_nodata is not None and grid_val == label_nodata:
                is_invalid = True
            if (not np.isfinite(grid_val)) or grid_val == FINAL_NODATA or grid_val < 0:
                is_invalid = True

            if is_invalid:
                self._debug_stats["label_failed"] += 1

                if debug_enabled and not self._debug_logged and retry < 5:
                    print(
                        f"  [DEBUG] 样本 {cur_idx}: "
                        f"栅格值无效: {grid_val}, nodata={label_nodata}"
                    )

                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            # ============================================================
            # 4. 构建点特征
            # ============================================================
            point_feats = self._build_point_features_station(feature_date, r, c)

            if point_feats is None:
                self._debug_stats["point_failed"] += 1

                if debug_enabled and not self._debug_logged and retry < 5:
                    print(f"  [DEBUG] 样本 {cur_idx} (站点:{station_id}) 点特征构建失败")

                    try:
                        s1_vv, s1_vh, _, _, _ = self._get_sentinel1_value_loose(
                            feature_date, r, c
                        )
                        smap_tbv, smap_tbh = self._get_smap_value_loose(
                            feature_date, r, c
                        )
                        print(f"     哨兵1: VV={s1_vv:.2f}, VH={s1_vh:.2f}")
                        print(f"     SMAP: TBV={smap_tbv:.2f}, TBH={smap_tbh:.2f}")
                    except Exception as e:
                        print(f"     点特征调试失败: {e}")

                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            # ============================================================
            # 5. 获取站点真值
            # ============================================================
            y = float(meta["swe"])

            if np.isnan(y) or y < 0:
                self._debug_stats["swe_failed"] += 1

                if debug_enabled and not self._debug_logged and retry < 5:
                    print(f"  [DEBUG] 样本 {cur_idx}: SWE值无效: {y}")

                cur_idx = (cur_idx + 1) % len(self.meta_index)
                continue

            # ============================================================
            # 6. 第一次成功时打印诊断
            # ============================================================
            self._debug_stats["success"] += 1

            if debug_enabled and not self._debug_logged:
                print(f"\n{'=' * 60}")
                print(f"✅ [DEBUG] 成功获取样本 (第{retry + 1}次尝试)")
                print(f"{'=' * 60}")
                print(f"  原始索引: {idx}")
                print(f"  实际索引: {cur_idx}")
                print(f"  站点ID: {station_id}")
                print(f"  日期: {feature_date.strftime('%Y-%m-%d')}")
                print(f"  位置: 行={r}, 列={c}")
                print(f"  站点SWE: {y:.2f} mm")
                print(f"  栅格SWE: {grid_val:.2f} mm")
                print(f"\n  卷积特征形状: {conv_patch.shape}")
                print(f"  点特征形状: {point_feats.shape}")
                print(f"  点特征前5个值: {point_feats[:5]}")
                print(f"\n  失败统计:")
                print(f"    卷积失败: {self._debug_stats['conv_failed']}")
                print(f"    标签失败: {self._debug_stats['label_failed']}")
                print(f"    点特征失败: {self._debug_stats['point_failed']}")
                print(f"    SWE失败: {self._debug_stats['swe_failed']}")
                print(f"    成功: {self._debug_stats['success']}")

                self._debug_logged = True

            # ============================================================
            # 7. 转成 Tensor
            # ============================================================
            conv_t = torch.from_numpy(conv_patch).float()
            point_t = torch.from_numpy(point_feats).float()
            y_t = torch.tensor(y, dtype=torch.float32)
            grid_val_t = torch.tensor(grid_val, dtype=torch.float32)

            is_zero_t = torch.tensor(
                1.0 if y > 0 else 0.0,
                dtype=torch.float32
            )

            # ============================================================
            # 8. 标准化：与M0-M4预训练严格一致
            # ============================================================
            eps = 1e-6

            try:
                if (
                    getattr(
                        self,
                        "normalization_method",
                        "minmax",
                    )
                    == "clip_then_zscore"
                ):
                    conv_low = torch.from_numpy(
                        self.conv_clip_low
                    ).float().view(-1, 1, 1)

                    conv_high = torch.from_numpy(
                        self.conv_clip_high
                    ).float().view(-1, 1, 1)

                    conv_mean = torch.from_numpy(
                        self.conv_mean
                    ).float().view(-1, 1, 1)

                    conv_std = torch.from_numpy(
                        self.conv_std
                    ).float().view(-1, 1, 1)

                    conv_t = torch.clamp(
                        conv_t,
                        min=conv_low,
                        max=conv_high,
                    )

                    conv_t = (
                        conv_t - conv_mean
                    ) / torch.clamp(
                        conv_std,
                        min=eps,
                    )

                    raw_point = point_t.clone()

                    # 与预训练build_progressive_normalization.py
                    # 中的point_valid_mask保持一致。
                    valid_point = torch.ones(
                        self.C_point,
                        dtype=torch.bool,
                    )

                    if self.C_point >= 15:
                        # Landsat六波段：0表示缺失填充值
                        valid_point[0:6] = (
                            raw_point[0:6] != 0.0
                        )

                        # S1 VV/VH的有效性由coverage判断
                        valid_point[6] = (
                            raw_point[8] > 0.0
                        )
                        valid_point[7] = (
                            raw_point[9] > 0.0
                        )

                        # S1 angle
                        valid_point[10] = (
                            (raw_point[8] > 0.0)
                            | (raw_point[9] > 0.0)
                        )

                        # SMAP TBV/TBH的有效性由mask判断
                        valid_point[11] = (
                            raw_point[13] > 0.0
                        )
                        valid_point[12] = (
                            raw_point[14] > 0.0
                        )

                    point_low = torch.from_numpy(
                        self.point_clip_low
                    ).float()

                    point_high = torch.from_numpy(
                        self.point_clip_high
                    ).float()

                    point_mean = torch.from_numpy(
                        self.point_mean
                    ).float()

                    point_std = torch.from_numpy(
                        self.point_std
                    ).float()

                    for feature_index, transform_name in enumerate(
                        self.point_transform
                    ):
                        if transform_name == "zscore":
                            if valid_point[feature_index]:
                                value = torch.clamp(
                                    raw_point[feature_index],
                                    point_low[feature_index],
                                    point_high[feature_index],
                                )

                                point_t[feature_index] = (
                                    value
                                    - point_mean[feature_index]
                                ) / torch.clamp(
                                    point_std[feature_index],
                                    min=eps,
                                )
                            else:
                                # 缺失连续变量保持0，
                                # 不把缺失填充值变成极端z-score。
                                point_t[feature_index] = 0.0
                        else:
                            # coverage、mask、坐标、DOY保持原语义
                            point_t[feature_index] = (
                                raw_point[feature_index]
                            )

                    y_t = (
                        y_t - self.label_min
                    ) / (
                        self.label_max
                        - self.label_min
                    )

                    grid_val_norm_t = (
                        grid_val_t - self.label_min
                    ) / (
                        self.label_max
                        - self.label_min
                    )

                    grid_val_norm_t = torch.clamp(
                        grid_val_norm_t,
                        0.0,
                        1.0,
                    )

                else:
                    c_min = torch.from_numpy(
                        self.conv_min
                    ).float().view(-1, 1, 1)

                    c_max = torch.from_numpy(
                        self.conv_max
                    ).float().view(-1, 1, 1)

                    p_min = torch.from_numpy(
                        self.point_min
                    ).float()

                    p_max = torch.from_numpy(
                        self.point_max
                    ).float()

                    conv_t = (
                        conv_t - c_min
                    ) / (
                        c_max - c_min + eps
                    )
                    conv_t = torch.clamp(
                        conv_t,
                        0.0,
                        1.0,
                    )

                    point_t = (
                        point_t - p_min
                    ) / (
                        p_max - p_min + eps
                    )
                    point_t = torch.clamp(
                        point_t,
                        0.0,
                        1.0,
                    )

                    y_t = (
                        y_t - self.swe_min
                    ) / (
                        self.swe_max
                        - self.swe_min
                        + eps
                    )
                    y_t = torch.clamp(
                        y_t,
                        0.0,
                        1.0,
                    )

                    grid_val_norm_t = (
                        grid_val_t - self.swe_min
                    ) / (
                        self.swe_max
                        - self.swe_min
                        + eps
                    )
                    grid_val_norm_t = torch.clamp(
                        grid_val_norm_t,
                        0.0,
                        1.0,
                    )

            except Exception as exc:
                self._debug_stats["norm_failed"] += 1

                if debug_enabled and not self._debug_logged:
                    print(
                        f"  [DEBUG] 标准化失败: {exc}"
                    )

                cur_idx = (
                    cur_idx + 1
                ) % len(self.meta_index)
                continue

            return conv_t, point_t, y_t, is_zero_t, grid_val_norm_t, int(cur_idx)

        # ============================================================
        # 重试失败：打印完整诊断
        # ============================================================
        print(f"\n{'=' * 80}")
        print(f"❌ 严重错误: 在idx={idx}附近连续{max_retry}个样本均无效")
        print(f"{'=' * 80}")

        print(f"\n📊 最终失败统计:")
        print(f"  总尝试次数: {self._debug_stats['total_attempts']}")
        print(f"  卷积失败: {self._debug_stats['conv_failed']}")
        print(f"  标签失败: {self._debug_stats['label_failed']}")
        print(f"  点特征失败: {self._debug_stats['point_failed']}")
        print(f"  SWE失败: {self._debug_stats['swe_failed']}")
        print(f"  标准化失败: {self._debug_stats['norm_failed']}")
        print(f"  成功: {self._debug_stats['success']}")

        print(f"\n📋 附近样本诊断 (idx={idx} 周围10个):")

        for offset in range(-5, 6):
            check_idx = (idx + offset) % len(self.meta_index)

            if check_idx < 0:
                continue

            meta = self.meta_index[check_idx]
            feature_date = meta.get("feature_date", meta.get("date"))
            feature_date_str = (
                feature_date.strftime("%Y-%m-%d")
                if hasattr(feature_date, "strftime")
                else str(feature_date)
            )

            print(
                f"  [{check_idx}] 日期={feature_date_str}, "
                f"行={meta['row']}, 列={meta['col']}, "
                f"SWE={meta['swe']:.2f}, "
                f"站点={meta.get('station_id', 'unknown')}"
            )

        raise IndexError(f"在idx={idx}附近连续{max_retry}个样本均无效")
    
class MixedFineTuneDataset(Dataset):
    """
    混合数据集：站点数据 + 高质量预训练样本
    新增：样本筛选功能
    """
    def __init__(
            self,
            station_csv: Path = STATION_SWE_CSV,
            pretrain_dataset=None,
            station_ratio: float = 0.5,
            year_target: Union[int, List[int]] = [2015, 2016, 2017],
            quality_threshold: float = 0.7,
            spatial_balance: bool = True,
            temporal_balance: bool = True,
            max_pretrain_samples: int = 10000,
            coordinate_jitter_std: float = 0.02,
            microwave_noise_std: float = 0.01,
            coordinate_mask_prob: float = 0.2,
            use_tta: bool = False,
            cache_dir: Optional[Path] = None,
            # 🔥 添加这两个参数
            split_cache_file: str = None,
            force_recompute_split: bool = False,
            # 🔥 雪量优先参数
            pretrain_snow_priority_ratio: float = 1.0,
            **kwargs
        ):
            super().__init__()

            print(f"\n{'='*60}")
            print("创建混合微调数据集（带样本筛选）")
            print(f"{'='*60}")

            print(f"\n数据增强配置:")
            print(f"  坐标抖动标准差: {coordinate_jitter_std}")
            print(f"  微波噪声标准差: {microwave_noise_std}")
            print(f"  坐标掩码概率: {coordinate_mask_prob}")
            if cache_dir:
                print(f"  缓存目录: {cache_dir}")

            # 🔥 打印划分缓存配置
            if split_cache_file:
                print(f"  划分缓存文件: {split_cache_file}")
            if force_recompute_split:
                print(f"  强制重新计算划分: {force_recompute_split}")

            # ============================================================
            # 🔥 关键修改：产品值修正只给 StationSWEDataset，不给 SWEDataset
            # ============================================================
            use_product_correction = kwargs.get("use_product_correction", False)

            station_kwargs = dict(kwargs)
            pretrain_kwargs = dict(kwargs)

            # StationSWEDataset 需要这个参数
            station_kwargs["use_product_correction"] = use_product_correction

            # SWEDataset 不认识这个参数，必须删掉
            pretrain_kwargs.pop("use_product_correction", None)

            print(f"  🔧 use_product_correction for StationSWEDataset = {use_product_correction}")

            # ============================================================
            # 1. 加载站点数据集
            # ============================================================
            self.station_dataset = StationSWEDataset(
                station_csv=station_csv,
                year_target=year_target,
                fine_tune_mode=True,
                load_fused_swe=True,
                coordinate_jitter_std=coordinate_jitter_std,
                microwave_noise_std=microwave_noise_std,
                coordinate_mask_prob=coordinate_mask_prob,
                use_tta=use_tta,
                cache_dir=cache_dir,
                # 🔥 传递划分缓存参数
                split_cache_file=split_cache_file,
                force_recompute_split=force_recompute_split,
                **station_kwargs
            )

            print(f"\n站点数据集: {len(self.station_dataset)} 个样本")

            # ============================================================
            # 2. 加载预训练数据集
            # ============================================================
            if pretrain_dataset is None:
                from data_online_era5_swe import SWEDataset

                # ============ 修改：直接使用原始的 year_target ============
                print(f"  预训练使用年份: {year_target}")

                # 🔥 这里必须用 pretrain_kwargs，而不是 kwargs
                # 进一步删除只属于 StationSWEDataset 的增强参数
                pretrain_kwargs = {
                    k: v for k, v in pretrain_kwargs.items()
                    if k not in [
                        "coordinate_jitter_std",
                        "microwave_noise_std",
                        "coordinate_mask_prob",
                        "use_product_correction",
                    ]
                }

                if cache_dir:
                    pretrain_kwargs["cache_dir"] = cache_dir

                self.pretrain_dataset = SWEDataset(
                    year_target=year_target,
                    use_tta=use_tta,
                    **pretrain_kwargs
                )
            else:
                self.pretrain_dataset = pretrain_dataset

            print(f"预训练数据集: {len(self.pretrain_dataset)} 个样本")

            # ============================================================
            # 3. 筛选高质量预训练样本
            # ============================================================
            self.pretrain_indices = self._select_high_quality_samples(
                max_samples=max_pretrain_samples,
                quality_threshold=quality_threshold,
                spatial_balance=spatial_balance,
                temporal_balance=temporal_balance
            )

            # ============================================================
            # 4. 创建索引映射
            # ============================================================
            self.station_indices = list(range(len(self.station_dataset)))

            # 🔥 保存 snow_priority_ratio 供后续使用
            self.pretrain_snow_priority_ratio = pretrain_snow_priority_ratio

            # ============================================================
            # 5. 计算各取多少样本
            # ============================================================
            n_station = len(self.station_dataset)
            self.station_ratio = station_ratio
            n_pretrain_target = int(n_station * (1 - station_ratio) / station_ratio)
            n_pretrain = min(n_pretrain_target, len(self.pretrain_indices))

            # 不再随机抽预训练样本，改成按 FusedSWE 分层抽样
            self.selected_pretrain = self._select_pretrain_by_swe_distribution(
                candidate_indices=self.pretrain_indices,
                n_pretrain=n_pretrain,
                seed=42
            )

            print(f"\n混合比例: 站点={station_ratio*100:.0f}%, 预训练={(1-station_ratio)*100:.0f}%")
            print(f"站点样本: {n_station}")
            print(f"高质量预训练样本: {len(self.pretrain_indices)} (筛选后)")
            print(f"实际使用预训练: {n_pretrain}")

            self.total_samples = n_station + n_pretrain
            self.station_meta = self.station_dataset.meta_index

            print(f"\n总样本数: {self.total_samples}")

            self.C_conv = self.station_dataset.C_conv
            self.C_point = self.station_dataset.C_point
            self.swe_min = self.station_dataset.swe_min
            self.swe_max = self.station_dataset.swe_max
            self.conv_min = self.station_dataset.conv_min
            self.conv_max = self.station_dataset.conv_max
            self.point_min = self.station_dataset.point_min
            self.point_max = self.station_dataset.point_max
            self.mode = 'train'

            print(f"\n归一化参数已继承:")
            print(f"  C_conv: {self.C_conv}")
            print(f"  C_point: {self.C_point}")
            print(f"  SWE范围: [{self.swe_min:.2f}, {self.swe_max:.2f}]")
    
    
    def set_mode(self, mode='train'):
        """
        设置数据集模式

        Args:
            mode: 'train' 训练模式（启用数据增强）或 'val' 验证模式（禁用数据增强）
        """
        self.mode = mode

        # 传递给站点数据集
        if hasattr(self, 'station_dataset'):
            if mode == 'train':
                self.station_dataset.set_augmentation_mode(True)
                print(f"  [MixedFineTuneDataset] 切换到训练模式 - 数据增强已启用")
            else:
                self.station_dataset.set_augmentation_mode(False)
                print(f"  [MixedFineTuneDataset] 切换到验证模式 - 数据增强已关闭")
    
    def _get_pretrain_label_mm(self, idx):
        """
        根据 pretrain_dataset 的 meta_index 和 label_data 获取预训练样本的 FusedSWE 标签，单位 mm。
        如果获取失败，返回 np.nan。
        """
        import numpy as np

        try:
            item = self.pretrain_dataset.meta_index[idx]

            if len(item) == 4:
                date, r, c, _ = item
            else:
                date, r, c = item

            if not hasattr(self.pretrain_dataset, "label_data"):
                return np.nan

            label_data = self.pretrain_dataset.label_data

            # 1. 直接用 date 匹配
            if date in label_data:
                label_arr, label_nodata = label_data[date]
            else:
                # 2. 尝试字符串日期匹配
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
                matched = False
                label_arr, label_nodata = None, None

                for key, value in label_data.items():
                    key_str = key.strftime("%Y-%m-%d") if hasattr(key, "strftime") else str(key)

                    if key_str == date_str:
                        label_arr, label_nodata = value
                        matched = True
                        break

                if not matched:
                    return np.nan

            if label_arr is None:
                return np.nan

            if not (0 <= r < label_arr.shape[0] and 0 <= c < label_arr.shape[1]):
                return np.nan

            val = label_arr[r, c]

            if label_nodata is not None and val == label_nodata:
                return np.nan

            if not np.isfinite(val):
                return np.nan

            return float(val)

        except Exception:
            return np.nan
    
    
    def _select_high_quality_samples(self, max_samples=10000, quality_threshold=0.7,
                                     spatial_balance=True, temporal_balance=True):
        """
        筛选高质量预训练样本 - 严格禁止与站点位置重复
        """
        print(f"\n【筛选高质量预训练样本】")

        # 获取所有预训练样本的元数据
        pretrain_meta = []
        for idx in range(len(self.pretrain_dataset)):
            # 🔥 预训练数据集 meta_index 格式: (date, r, c, source) 或 (date, r, c)
            item = self.pretrain_dataset.meta_index[idx]
            if len(item) == 4:
                date, r, c, source = item
            else:
                date, r, c = item

            pretrain_meta.append({
                'idx': idx,
                'date': date,
                'row': r,
                'col': c,
                'doy': date.timetuple().tm_yday,
                'fused_swe_mm': self._get_pretrain_label_mm(idx)
            })

        print(f"  总预训练样本: {len(pretrain_meta)}")

        # ============ 1. 质量筛选（雪样本阈值放宽） ============
        quality_scores = []
        fused_swe_values = []

        for meta in pretrain_meta:
            score = self._compute_quality_score(meta)
            quality_scores.append(score)
            fused_swe_values.append(meta['fused_swe_mm'] if np.isfinite(meta['fused_swe_mm']) else 0.0)

        quality_scores = np.array(quality_scores)
        fused_swe_values = np.array(fused_swe_values)

        # ============ 🔥 关键修改：雪样本阈值放宽 ============
        pretrain_snow_min_mm = getattr(self, 'pretrain_snow_min_mm', 20.0)
        quality_threshold = getattr(self, 'quality_threshold', 0.83)
        snow_quality_threshold = getattr(self, 'snow_quality_threshold', 0.60)

        print(f"\n  质量筛选配置:")
        print(f"    雪样本阈值 (FusedSWE >= {pretrain_snow_min_mm} mm): 质量阈值 {snow_quality_threshold}")
        print(f"    非雪样本 (FusedSWE < {pretrain_snow_min_mm} mm): 质量阈值 {quality_threshold}")

        snow_mask = fused_swe_values >= pretrain_snow_min_mm
        snow_count = np.sum(snow_mask)
        print(f"    雪样本数量: {snow_count} ({(snow_count / len(pretrain_meta) * 100):.1f}%)")

        # 🔥 核心修改：分别应用不同的质量阈值
        keep = (
            (snow_mask & (quality_scores >= snow_quality_threshold)) |
            ((~snow_mask) & (quality_scores >= quality_threshold))
        )

        high_quality_indices = [pretrain_meta[i]['idx'] for i, keep_flag in enumerate(keep) if keep_flag]

        print(f"  质量筛选后: {len(high_quality_indices)} 个样本")

        # 打印雪样本保留情况
        kept_snow = np.sum(keep & snow_mask)
        print(f"    其中雪样本: {kept_snow} (保留率 {kept_snow / max(1, snow_count) * 100:.1f}%)")

        # ============ 2. 🔥 获取所有站点位置（像素级） ============
        station_locations = set()
        for meta in self.station_dataset.meta_index:
            station_locations.add((meta['row'], meta['col']))

        print(f"  站点位置总数: {len(station_locations)}")

        # ============ 3. 🔥 第一步过滤：移除与站点位置重复的样本 ============
        non_station_indices = []
        for idx in high_quality_indices:
            item = self.pretrain_dataset.meta_index[idx]
            if len(item) == 4:
                _, r, c, _ = item
            else:
                _, r, c = item

            # 严格检查：位置不能在站点位置中
            if (r, c) not in station_locations:
                non_station_indices.append(idx)

        print(f"  移除站点位置重复后: {len(non_station_indices)} 个样本")

        # 如果过滤后样本太少，警告但不添加重复位置
        if len(non_station_indices) < max_samples // 2:
            print(f"  ⚠️ 警告: 与站点位置不同的样本只有 {len(non_station_indices)} 个")
            print(f"  将使用全部 {len(non_station_indices)} 个样本，不添加重复位置（避免数据泄露）")

        # 使用过滤后的样本
        filtered_indices = non_station_indices

        # ============ 4. 空间平衡（在非站点位置中采样） ============
        if spatial_balance and len(filtered_indices) > max_samples:
            from collections import defaultdict

            # 对预训练样本按空间网格分组
            spatial_groups = defaultdict(list)
            for idx in filtered_indices:
                item = self.pretrain_dataset.meta_index[idx]
                if len(item) == 4:
                    _, r, c, _ = item
                else:
                    _, r, c = item
                grid_key = (r // 10, c // 10)  # 10x10网格
                spatial_groups[grid_key].append(idx)

            selected = []
            samples_per_grid = max_samples // len(spatial_groups)

            for grid, indices in spatial_groups.items():
                n_take = min(samples_per_grid, len(indices))
                if n_take > 0:
                    selected.extend(np.random.choice(indices, n_take, replace=False))

            # 补足剩余
            if len(selected) < max_samples:
                remaining = max_samples - len(selected)
                remaining_indices = [idx for idx in filtered_indices if idx not in selected]
                if remaining_indices:
                    selected.extend(np.random.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False))

            filtered_indices = selected[:max_samples]
            print(f"  空间平衡后: {len(filtered_indices)} 个样本")

        # ============ 5. 时间平衡 ============
        if temporal_balance and len(filtered_indices) > max_samples:
            from collections import defaultdict

            month_groups = defaultdict(list)
            for idx in filtered_indices:
                item = self.pretrain_dataset.meta_index[idx]
                if len(item) == 4:
                    date, _, _, _ = item
                else:
                    date, _, _ = item
                month_key = f"{date.year}-{date.month:02d}"
                month_groups[month_key].append(idx)

            selected = []
            samples_per_month = max_samples // len(month_groups)

            for month, indices in month_groups.items():
                n_take = min(samples_per_month, len(indices))
                if n_take > 0:
                    selected.extend(np.random.choice(indices, n_take, replace=False))

            # 补足剩余
            if len(selected) < max_samples:
                remaining = max_samples - len(selected)
                remaining_indices = [idx for idx in filtered_indices if idx not in selected]
                if remaining_indices:
                    selected.extend(np.random.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False))

            filtered_indices = selected[:max_samples]
            print(f"  时间平衡后: {len(filtered_indices)} 个样本")

        print(f"\n  ✅ 最终筛选出 {len(filtered_indices)} 个高质量预训练样本")
        print(f"     （全部与站点位置不同，无数据泄露风险）")

        return filtered_indices
    
    
    
    def _select_pretrain_by_swe_distribution(self, candidate_indices, n_pretrain, seed=42):
        """
        从已经筛好的高质量预训练样本中，按 FusedSWE 分层抽样。
        雪量优先：>=20mm 样本优先全选，比例由 pretrain_snow_priority_ratio 控制
        """
        import numpy as np
        import random
        from collections import defaultdict

        random.seed(seed)
        np.random.seed(seed)

        snow_ratio = getattr(self, 'pretrain_snow_priority_ratio', 1.0)

        print("\n【按 FusedSWE 分层选择预训练样本 (雪量优先)】")
        print(f"  候选预训练样本: {len(candidate_indices)}")
        print(f"  目标选择数量: {n_pretrain}")
        print(f"  雪量优先比例: {snow_ratio*100:.0f}%")

        # 获取所有候选样本的 FusedSWE 值
        records = []
        for idx in candidate_indices:
            swe = self._get_pretrain_label_mm(idx)
            if np.isfinite(swe) and swe >= 0:
                records.append((idx, swe))

        if len(records) == 0:
            print("  ⚠ 无法获取预训练 FusedSWE 标签，退回随机抽样")
            return random.sample(candidate_indices, min(n_pretrain, len(candidate_indices)))

        candidate_indices = np.asarray([r[0] for r in records])
        fused = np.asarray([r[1] for r in records], dtype=np.float32)

        valid = np.isfinite(fused)
        candidate_indices = candidate_indices[valid]
        fused = fused[valid]

        # 分箱
        snow20 = candidate_indices[fused >= 20.0]
        mid10 = candidate_indices[(fused >= 10.0) & (fused < 20.0)]
        low1 = candidate_indices[(fused >= 1.0) & (fused < 10.0)]
        zero = candidate_indices[fused < 1.0]

        print("\n  候选样本 FusedSWE 分布:")
        print(f"    >=20 mm:      {len(snow20)}")
        print(f"    10-20 mm:     {len(mid10)}")
        print(f"    1-10 mm:      {len(low1)}")
        print(f"    <1 mm:        {len(zero)}")

        # 雪量优先：>=20mm 占 snow_ratio
        n_snow20 = min(int(n_pretrain * snow_ratio), len(snow20))
        remaining = n_pretrain - n_snow20

        # 剩余 60% 按比例分配给其他箱
        other_total = len(mid10) + len(low1) + len(zero)
        if other_total > 0:
            n_mid10 = min(int(remaining * len(mid10) / other_total), len(mid10))
            n_low1 = min(int(remaining * len(low1) / other_total), len(low1))
            n_zero = remaining - n_mid10 - n_low1
            n_zero = min(n_zero, len(zero))
        else:
            n_mid10 = n_low1 = n_zero = 0

        selected = []
        if n_snow20 > 0:
            selected.extend(np.random.choice(snow20, n_snow20, replace=False))
        if n_mid10 > 0:
            selected.extend(np.random.choice(mid10, n_mid10, replace=False))
        if n_low1 > 0:
            selected.extend(np.random.choice(low1, n_low1, replace=False))
        if n_zero > 0:
            selected.extend(np.random.choice(zero, n_zero, replace=False))

        # 补足不足的
        if len(selected) < n_pretrain:
            selected_set = set(selected)
            remaining_indices = [idx for idx in candidate_indices if idx not in selected_set]
            need = n_pretrain - len(selected)
            if remaining_indices:
                selected.extend(np.random.choice(remaining_indices, min(need, len(remaining_indices)), replace=False))

        selected = selected[:n_pretrain]

        # 打印最终分布
        final_swe = [self._get_pretrain_label_mm(idx) for idx in selected]
        final_swe = np.array([x for x in final_swe if np.isfinite(x)])

        print("\n  最终选择的预训练样本 FusedSWE 分布:")
        if len(final_swe) > 0:
            final_snow20 = int((final_swe >= 20.0).sum())
            print(f"    N:       {len(final_swe)}")
            print(f"    mean:    {final_swe.mean():.2f} mm")
            print(f"    p50:     {np.percentile(final_swe, 50):.2f} mm")
            print(f"    p75:     {np.percentile(final_swe, 75):.2f} mm")
            print(f"    p90:     {np.percentile(final_swe, 90):.2f} mm")
            print(f"    max:     {final_swe.max():.2f} mm")
            print(f"    >=20 mm: {final_snow20} ({final_snow20 / len(final_swe) * 100:.1f}%)")

        return [int(x) for x in selected]
    
    def reselect_pretrain_by_train_count(self, n_train_station):
        """
        根据实际训练站点样本数重新选择预训练样本。
        在 build_mixed_dataloaders 确定 train_indices 后调用。
        
        Args:
            n_train_station: 训练集站点样本数（len(train_indices)）
        """
        station_ratio = getattr(self, 'station_ratio', 0.5)
        n_pretrain_target = int(n_train_station * (1 - station_ratio) / station_ratio)
        n_pretrain = min(n_pretrain_target, len(self.pretrain_indices))
        
        print(f"\n🔄 根据训练集重新选择预训练样本:")
        print(f"  训练集站点样本数: {n_train_station}")
        print(f"  station_ratio: {station_ratio}")
        print(f"  目标 pretrain: {n_pretrain_target}")
        print(f"  实际 pretrain: {n_pretrain}")
        
        self.selected_pretrain = self._select_pretrain_by_swe_distribution(
            candidate_indices=self.pretrain_indices,
            n_pretrain=n_pretrain,
            seed=42
        )
        self.total_samples = n_train_station + n_pretrain
        print(f"  总样本数: {self.total_samples}")
        
        return n_pretrain
    
    
    
    def set_validation_mode(self, is_validation: bool = True):
        """
        设置验证模式（关闭数据增强）

        Args:
            is_validation: True 表示验证模式（无增强），False 表示训练模式（有增强）
        """
        # 传递给站点数据集
        if hasattr(self, 'station_dataset'):
            self.station_dataset.set_validation_mode(is_validation)

        if is_validation:
            print(f"  [MixedFineTuneDataset] 切换到验证模式 - 数据增强已关闭")
        else:
            print(f"  [MixedFineTuneDataset] 切换到训练模式 - 数据增强已启用")
    
    def _compute_quality_score(self, meta):
        """
        计算样本质量分数
        """
        score = 1.0
        
        # 1. 基于云量（如果有）
        # TODO: 如果有云量信息，可以加入
        
        # 2. 基于季节（夏季可能质量更好）
        month = meta['date'].month
        if 6 <= month <= 9:  # 夏季
            score *= 1.2
        elif 12 <= month or month <= 2:  # 冬季
            score *= 0.8
        
        # 3. 基于空间位置（中心区域可能更好）
        r, c = meta['row'], meta['col']
        h, w = self.pretrain_dataset.H, self.pretrain_dataset.W
        center_dist = np.sqrt((r - h/2)**2 + (c - w/2)**2)
        max_dist = np.sqrt((h/2)**2 + (w/2)**2)
        score *= (1 - 0.3 * center_dist / max_dist)  # 中心区域加分
        
        return score
    
    
    def _apply_tta(self, conv_tensor, point_tensor, num_augmentations=8):
        """
        测试时增强 (Test Time Augmentation)
        对输入进行多次微小扰动，取平均预测

        Args:
            conv_tensor: 卷积特征 (1, C_conv, P, P)
            point_tensor: 点特征 (1, 16)
            num_augmentations: 增强次数

        Returns:
            增强后的平均预测
        """
        predictions = []

        for _ in range(num_augmentations):
            # 复制原始点特征
            point_aug = point_tensor.clone()

            # 对经纬度添加小噪声
            lon_idx, lat_idx = 10, 11
            point_aug[0, lon_idx] += torch.randn(1) * 0.01
            point_aug[0, lat_idx] += torch.randn(1) * 0.01

            # 对微波信号添加小噪声
            microwave_indices = [6, 7, 8, 9]
            for m_idx in microwave_indices:
                point_aug[0, m_idx] *= (1 + torch.randn(1) * 0.005)

            # 裁剪到有效范围
            point_aug = torch.clamp(point_aug, 0.0, 1.0)

            # 预测
            with torch.no_grad():
                pred = self.model(conv_tensor, point_aug)
                predictions.append(pred)

        # 取中位数（对异常值更鲁棒）
        predictions = torch.stack(predictions)
        return predictions.median(dim=0)[0]

    
    def __getattr__(self, name):
        """
        魔法方法：当访问的属性在当前类找不到时，自动调用此方法。
        我们将其转发给内部的 station_dataset。
        """
        try:
            # 尝试从 station_dataset 中获取属性
            return getattr(self.station_dataset, name)
        except AttributeError:
            # 如果连 station_dataset 也没有，才抛出错误
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        获取一个样本 - 统一槽位法
        确保出口只有这一个，且返回数量绝对是 6

        核心改进：
        - 直接使用18维点特征，不再追加 SHTSI
        - 第6个返回值：样本来源标记 (0=站点实测, 1=预训练伪标签)
        """
        # 1. 初始定义 6 个槽位
        res_conv, res_point, res_target, res_mask, res_grid, res_source = [None] * 6

        # ============ 新增：样本来源标记 ============
        # 0 = 站点实测样本
        # 1 = 预训练伪标签样本
        is_pretrain_sample = idx >= len(self.station_dataset)
        source_flag = 1 if is_pretrain_sample else 0

        # 2. 拿货（区分数据源）
        if not is_pretrain_sample:  # 原 if idx < len(self.station_dataset)
            # ============ 站点数据仓库：直接使用 ============
            raw_res = self.station_dataset[idx]
            if len(raw_res) >= 6:
                res_conv, res_point, res_target, res_mask, res_grid, _ = raw_res[:6]
            elif len(raw_res) == 5:
                res_conv, res_point, res_target, res_mask, res_grid = raw_res[:5]
            else:
                res_conv, res_point, res_target, res_mask = raw_res[:4]
                res_grid = res_target.clone() if torch.is_tensor(res_target) else torch.tensor(res_target)
        else:
            # ============ 预训练数据仓库：直接使用18维点特征，不再追加SHTSI ============
            p_idx = self.selected_pretrain[idx - len(self.station_dataset)]
            raw_res = self.pretrain_dataset[p_idx]

            if len(raw_res) >= 4:
                res_conv, res_point, res_target, res_mask = raw_res[:4]
            else:
                res_conv, res_point, res_target = raw_res[:3]
                res_mask = torch.where(
                    torch.as_tensor(res_target) > 0,
                    torch.ones_like(torch.as_tensor(res_target, dtype=torch.float32)),
                    torch.zeros_like(torch.as_tensor(res_target, dtype=torch.float32))
                )

            res_grid = res_target.clone() if torch.is_tensor(res_target) else torch.tensor(res_target)

        # 3. 模具加工（形状强制统一）
        res_target = torch.as_tensor(res_target, dtype=torch.float32).reshape(())
        res_mask = torch.as_tensor(res_mask, dtype=torch.float32).reshape(())
        res_grid = torch.as_tensor(res_grid, dtype=torch.float32).reshape(())

        # 点特征: 确保是18维
        res_point = torch.as_tensor(res_point, dtype=torch.float32).flatten()

        expected_point_dim = self.C_point  # 现在应该是18

        if res_point.shape[0] < expected_point_dim:
            padding = torch.zeros(expected_point_dim - res_point.shape[0])
            res_point = torch.cat([res_point, padding])
        elif res_point.shape[0] > expected_point_dim:
            res_point = res_point[:expected_point_dim]

        # 卷积特征维度对齐
        target_channels = self.C_conv
        res_conv = torch.as_tensor(res_conv, dtype=torch.float32)
        if res_conv.dim() == 4:
            res_conv = res_conv.squeeze(0)
        if res_conv.dim() == 2:
            res_conv = res_conv.unsqueeze(0)
        if res_conv.shape[0] < target_channels:
            padding = torch.zeros(target_channels - res_conv.shape[0], res_conv.shape[1], res_conv.shape[2])
            res_conv = torch.cat([res_conv, padding], dim=0)
        elif res_conv.shape[0] > target_channels:
            res_conv = res_conv[:target_channels, :, :]

        # ============ 第6个槽位：样本来源标记 ============
        # 0 = station, 1 = pretrain
        source_flag_tensor = torch.tensor(source_flag, dtype=torch.long)

        # 4. 归一化（预训练分支）
        if is_pretrain_sample:
            eps = 1e-6
            try:
                c_min = torch.from_numpy(self.conv_min).view(-1, 1, 1).float()
                c_max = torch.from_numpy(self.conv_max).view(-1, 1, 1).float()
                res_conv = torch.clamp((res_conv - c_min) / (c_max - c_min + eps), 0.0, 1.0)

                p_min = torch.from_numpy(self.point_min).float()
                p_max = torch.from_numpy(self.point_max).float()
                res_point = torch.clamp((res_point - p_min) / (p_max - p_min + eps), 0.0, 1.0)

                res_target = torch.clamp((res_target - self.swe_min) / (self.swe_max - self.swe_min + eps), 0.0, 1.0)
                res_grid = res_target.clone()
            except:
                pass

        # 返回 6 个值：卷积, 点, 目标, 零掩码, 产品值, 来源标记
        return res_conv, res_point, res_target, res_mask, res_grid, source_flag_tensor


def build_station_dataloaders_swe(
    station_csv: Path = STATION_SWE_CSV,
    batch_size: int = 128,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    seed: int = 42,
    fine_tune_mode: bool = False,
    split_strategy: str = 'station_test',
    coordinate_jitter_std: float = 0.02,
    microwave_noise_std: float = 0.01,
    coordinate_mask_prob: float = 0.2,
    use_tta: bool = False,
    split_cache_file: str = None,  # 🔥 新增：划分缓存文件路径
    force_recompute_split: bool = False,  # 🔥 新增：强制重新计算
    # 🔥 新增：共享缓存模式（用于十折CV）
    shared_cache_mode: bool = False,
    **dataset_kwargs
):
    """
    构建站点SWE数据的数据加载器
    split_strategy:
        'station_all': 全部按站点划分（默认）
        'station_test': 测试集按站点划分，训练验证集随机划分
    
    Args:
        split_cache_file: 划分缓存文件路径，多个策略共享时使用相同划分
        force_recompute_split: 是否强制重新计算划分（忽略缓存）
        shared_cache_mode: 是否启用共享缓存模式（用于十折CV，所有折共享特征缓存）
    """
    print("\n" + "="*70)
    print("🚀 构建站点SWE数据加载器")
    print(f"模式: {'微调' if fine_tune_mode else '训练'}")
    print(f"划分策略: {split_strategy}")
    print(f"随机种子: {seed}")
    print(f"共享缓存模式: {'启用' if shared_cache_mode else '禁用'}")
    print(f"数据增强: 坐标抖动={coordinate_jitter_std}, 微波噪声={microwave_noise_std}, 坐标掩码={coordinate_mask_prob}")
    print("="*70)
    
    try:
        dataset_kwargs['fine_tune_mode'] = fine_tune_mode
        dataset_kwargs['coordinate_jitter_std'] = coordinate_jitter_std
        dataset_kwargs['microwave_noise_std'] = microwave_noise_std
        dataset_kwargs['coordinate_mask_prob'] = coordinate_mask_prob
        dataset_kwargs['use_tta'] = use_tta
        # 🔥 传递共享缓存模式到 dataset
        dataset_kwargs['shared_cache_mode'] = shared_cache_mode

        dataset = StationSWEDataset(
            station_csv=station_csv,
            **dataset_kwargs
        )
        
        all_indices = list(range(len(dataset)))
        
        # ============ 🔥 辅助函数：提取所有关联站点ID ============
        def extract_all_station_ids(meta):
            """从 meta 中提取所有关联的站点ID"""
            station_ids = set()
            
            # 优先使用 source_stations（聚合样本）
            if 'source_stations' in meta and meta['source_stations']:
                for sid in str(meta['source_stations']).split(','):
                    station_ids.add(sid.strip())
            else:
                # 普通样本
                station_id = meta['station_id']
                if ',' in str(station_id):
                    for sid in str(station_id).split(','):
                        station_ids.add(sid.strip())
                else:
                    station_ids.add(str(station_id))
            
            return station_ids
        
        # ============ 🔥 划分缓存逻辑 ============
        import pickle
        from pathlib import Path
        from datetime import datetime
        
        # 生成默认缓存文件名（如果未指定）
        if split_cache_file is None:
            import hashlib
            # 基于关键参数生成缓存键
            cache_key_str = f"{station_csv}_{val_ratio}_{test_ratio}_{seed}_{split_strategy}"
            cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()[:16]
            split_cache_file = f"./split_cache/split_{cache_key}.pkl"
        
        split_cache_path = Path(split_cache_file)
        split_cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 尝试加载缓存
        train_indices = None
        val_indices = None
        test_indices = None
        splits_info = None
        cache_loaded = False
        
        if not force_recompute_split and split_cache_path.exists():
            print(f"\n📦 从缓存加载划分: {split_cache_path}")
            try:
                with open(split_cache_path, 'rb') as f:
                    cached = pickle.load(f)
                
                # 验证缓存的一致性
                if cached.get('total_samples') == len(dataset):
                    train_indices = cached['train_indices']
                    val_indices = cached['val_indices']
                    test_indices = cached['test_indices']
                    splits_info = cached['splits_info']
                    cache_loaded = True
                    # CV_TEMP_SPLIT_LOG_CLEANUP_V1
                    if shared_cache_mode:
                        print("   ✅ 初始 DataLoader 划分缓存加载成功")
                        print("      说明: 正式十折CV不采用该临时train/val/test划分")
                        print("      正式CV样本数将在 run_cv_workflow 中进行完整性检查")
                        print(f"      缓存时间: {cached.get('timestamp', 'unknown')}")
                    else:
                        print(f"   ✅ 加载成功！")
                        print(f"      训练集: {len(train_indices)} 样本")
                        print(f"      验证集: {len(val_indices)} 样本")
                        print(f"      测试集: {len(test_indices)} 样本")
                        print(f"      缓存时间: {cached.get('timestamp', 'unknown')}")
                else:
                    print(f"   ⚠️ 缓存样本数({cached.get('total_samples')})与当前数据集({len(dataset)})不一致，重新计算")
            except Exception as e:
                print(f"   ⚠️ 缓存加载失败: {e}，重新计算")
        
        # ============ 如果需要重新计算划分 ============
        if not cache_loaded:
            print(f"\n🔨 计算新的划分...")
            
            # ============ 原有的划分逻辑 ============
            if split_strategy == 'station_test':
                print("\n【混合划分策略】")
                print("  测试集: 按站点划分（严格泛化测试）")
                print("  训练/验证集: 随机划分（更多训练数据）")
                
                # 1. 先按站点分组（处理多站点关联）
                station_to_indices = {}
                multi_station_warning = set()  # 记录多站点样本
                
                for idx, meta in enumerate(dataset.meta_index):
                    station_ids = extract_all_station_ids(meta)
                    
                    # 检查是否多站点
                    if len(station_ids) > 1:
                        multi_station_warning.add(idx)
                    
                    # 为每个关联的站点ID都添加这个样本的索引
                    for station_id in station_ids:
                        if station_id not in station_to_indices:
                            station_to_indices[station_id] = []
                        station_to_indices[station_id].append(idx)
                
                # 对每个站点的索引排序
                for station_id in station_to_indices:
                    station_to_indices[station_id].sort()
                
                # 打印多站点样本警告
                if multi_station_warning:
                    print(f"\n  ⚠️ 发现 {len(multi_station_warning)} 个样本关联多个站点")
                    print(f"     这些样本将被分配到多个站点的集合中")
                    print(f"     如果这些站点被分到不同的数据集（训练/验证/测试），会造成数据泄露！")
                
                # 对站点键排序
                stations = sorted(station_to_indices.keys())
                n_stations = len(stations)
                
                # 打印前5个站点用于调试
                print(f"\n  🔍 划分确定性验证 (seed={seed}):")
                print(f"     前5个站点: {stations[:5]}")
                
                # 按站点划分出测试集
                np.random.seed(seed)
                shuffled_stations = stations.copy()
                np.random.shuffle(shuffled_stations)
                
                print(f"     shuffle后前5个: {shuffled_stations[:5]}")
                
                n_test_stations = max(1, int(n_stations * test_ratio))
                test_stations = set(shuffled_stations[:n_test_stations])
                
                # 收集测试集样本
                test_indices_set = set()
                train_val_indices_set = set()
                test_station_samples = {}
                
                for station_id, indices in station_to_indices.items():
                    if station_id in test_stations:
                        test_indices_set.update(indices)
                        test_station_samples[station_id] = len(indices)
                    else:
                        train_val_indices_set.update(indices)
                
                test_indices = list(test_indices_set)
                train_val_indices = list(train_val_indices_set)
                
                print(f"\n【测试集统计】")
                print(f"  测试集站点数: {len(test_stations)}")
                print(f"  测试集样本数: {len(test_indices)}")
                
                # 检查泄露
                overlap_indices = set(test_indices) & set(train_val_indices)
                if overlap_indices:
                    print(f"  ❌ 严重警告: 发现 {len(overlap_indices)} 个样本同时出现在测试集和训练/验证集！")
                
                # 剩下的样本随机划分训练集和验证集
                np.random.seed(seed)
                np.random.shuffle(train_val_indices)
                n_train_val = len(train_val_indices)
                n_val = int(n_train_val * val_ratio)
                
                val_indices = train_val_indices[:n_val]
                train_indices = train_val_indices[n_val:]
                
                print(f"\n【训练/验证集统计（随机划分）】")
                print(f"  训练集样本数: {len(train_indices)}")
                print(f"  验证集样本数: {len(val_indices)}")
                
                # 对最终索引排序
                train_indices.sort()
                val_indices.sort()
                test_indices.sort()
                
                # 数据泄露检查
                test_site_ids = set(test_station_samples.keys())
                train_val_site_ids = set()
                for idx in train_val_indices:
                    station_ids = extract_all_station_ids(dataset.meta_index[idx])
                    train_val_site_ids.update(station_ids)
                
                overlap = test_site_ids & train_val_site_ids
                if overlap:
                    print(f"  ⚠️ 警告: 测试集站点出现在训练/验证集中!")
                    print(f"     重叠站点数: {len(overlap)}")
                else:
                    print(f"  ✓ 测试集站点与训练/验证集无重叠")
                
                splits_info = {
                    'split_strategy': 'station_test',
                    'test_stations': list(test_stations),
                    'test_samples': len(test_indices),
                    'train_samples': len(train_indices),
                    'val_samples': len(val_indices),
                    'total_samples': len(dataset),
                    'test_stations_count': len(test_stations),
                    'multi_station_samples': len(multi_station_warning),
                    'has_data_leakage': len(overlap_indices) > 0,
                    'seed': seed,
                    'data_augmentation': {
                        'coordinate_jitter_std': coordinate_jitter_std,
                        'microwave_noise_std': microwave_noise_std,
                        'coordinate_mask_prob': coordinate_mask_prob,
                        'use_tta': use_tta
                    }
                }
                
            else:
                # ============ 严格站点划分策略 ============
                print("\n【严格站点划分策略】")
                print("  所有数据集均按站点划分")
                
                station_to_indices = {}
                multi_station_warning = set()
                
                for idx, meta in enumerate(dataset.meta_index):
                    station_ids = extract_all_station_ids(meta)
                    
                    if len(station_ids) > 1:
                        multi_station_warning.add(idx)
                    
                    for station_id in station_ids:
                        if station_id not in station_to_indices:
                            station_to_indices[station_id] = []
                        station_to_indices[station_id].append(idx)
                
                for station_id in station_to_indices:
                    station_to_indices[station_id].sort()
                
                if multi_station_warning:
                    print(f"\n  ⚠️ 发现 {len(multi_station_warning)} 个样本关联多个站点")
                
                stations = sorted(station_to_indices.keys())
                n_stations = len(stations)
                
                print(f"\n  🔍 划分确定性验证 (seed={seed}):")
                print(f"     前5个站点: {stations[:5]}")
                
                np.random.seed(seed)
                shuffled_stations = stations.copy()
                np.random.shuffle(shuffled_stations)
                
                print(f"     shuffle后前5个: {shuffled_stations[:5]}")
                
                n_test = max(1, int(n_stations * test_ratio))
                n_val = max(1, int(n_stations * val_ratio))
                
                test_stations = set(shuffled_stations[:n_test])
                val_stations = set(shuffled_stations[n_test:n_test + n_val])
                train_stations = set(shuffled_stations[n_test + n_val:])
                
                train_indices_set = set()
                val_indices_set = set()
                test_indices_set = set()
                
                for station_id, indices in station_to_indices.items():
                    if station_id in train_stations:
                        train_indices_set.update(indices)
                    elif station_id in val_stations:
                        val_indices_set.update(indices)
                    elif station_id in test_stations:
                        test_indices_set.update(indices)
                
                train_indices = list(train_indices_set)
                val_indices = list(val_indices_set)
                test_indices = list(test_indices_set)
                
                train_indices.sort()
                val_indices.sort()
                test_indices.sort()
                
                overlap_train_val = set(train_indices) & set(val_indices)
                overlap_train_test = set(train_indices) & set(test_indices)
                overlap_val_test = set(val_indices) & set(test_indices)
                
                if overlap_train_val or overlap_train_test or overlap_val_test:
                    print(f"\n  ❌ 严重警告: 发现数据泄露！")
                
                print(f"\n【数据集统计】")
                print(f"  训练集: {len(train_stations)} 个站点, {len(train_indices)} 个样本")
                print(f"  验证集: {len(val_stations)} 个站点, {len(val_indices)} 个样本")
                print(f"  测试集: {len(test_stations)} 个站点, {len(test_indices)} 个样本")
                
                splits_info = {
                    'split_strategy': 'station_all',
                    'train_stations': list(train_stations),
                    'val_stations': list(val_stations),
                    'test_stations': list(test_stations),
                    'train_samples': len(train_indices),
                    'val_samples': len(val_indices),
                    'test_samples': len(test_indices),
                    'total_samples': len(dataset),
                    'train_stations_count': len(train_stations),
                    'val_stations_count': len(val_stations),
                    'test_stations_count': len(test_stations),
                    'multi_station_samples': len(multi_station_warning),
                    'has_data_leakage': len(overlap_train_val) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0,
                    'seed': seed,
                    'data_augmentation': {
                        'coordinate_jitter_std': coordinate_jitter_std,
                        'microwave_noise_std': microwave_noise_std,
                        'coordinate_mask_prob': coordinate_mask_prob,
                        'use_tta': use_tta
                    }
                }
            
            # ============ 🔥 保存划分到缓存 ============
            to_cache = {
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'splits_info': splits_info,
                'total_samples': len(dataset),
                'seed': seed,
                'timestamp': datetime.now().isoformat(),
                'station_csv': str(station_csv),
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'split_strategy': split_strategy,
            }
            
            with open(split_cache_path, 'wb') as f:
                pickle.dump(to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"\n💾 划分已保存到: {split_cache_path}")
            print(f"   文件大小: {split_cache_path.stat().st_size / 1024:.2f} KB")
        
        # ============ 创建数据加载器 ============
        from torch.utils.data import Subset, DataLoader
        
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'prefetch_factor': prefetch_factor if num_workers > 0 else None,
            'persistent_workers': persistent_workers if num_workers > 0 else False,
        }
        
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            shuffle=True,
            drop_last=True,
            **loader_kwargs
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            shuffle=False,
            drop_last=False,
            **loader_kwargs
        )
        
        test_loader = DataLoader(
            Subset(dataset, test_indices),
            shuffle=False,
            drop_last=False,
            **loader_kwargs
        )
        
        print(f"\n{'='*70}")
        print(f"✅ 数据加载器构建完成!")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  验证批次: {len(val_loader)}")
        print(f"  测试批次: {len(test_loader)}")
        print(f"  划分缓存: {split_cache_path}")
        print(f"  共享缓存模式: {'启用' if shared_cache_mode else '禁用'}")
        if splits_info.get('multi_station_samples', 0) > 0:
            print(f"  ⚠️ 多站点样本数: {splits_info['multi_station_samples']}")
        if splits_info.get('has_data_leakage', False):
            print(f"  ❌ 存在数据泄露！请检查多站点样本！")
        print(f"{'='*70}")
        
        return train_loader, val_loader, test_loader, (dataset.C_conv, dataset.C_point), splits_info
        
    except Exception as e:
        print(f"❌ 构建站点数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        raise
        
def build_mixed_dataloaders(
    station_csv: Path = STATION_SWE_CSV,
    batch_size: int = 32,
    station_ratio: float = 0.5,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    seed: int = 42,
    # ============ 数据增强参数 ============
    coordinate_jitter_std: float = 0.02,
    microwave_noise_std: float = 0.01,
    coordinate_mask_prob: float = 0.2,
    use_tta: bool = False,
    # 🔥 新增：划分缓存参数
    split_cache_file: str = None,
    force_recompute_split: bool = False,
    **kwargs
):
    """
    混合数据加载器 - 严格防止数据泄露
    - 测试集：按站点 ID 划分（保证空间独立性）
    - 训练/验证集：按样本随机划分（保证十折验证稳定性）
    """
    import random
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset
    
    print("\n" + "="*60)
    print(f"创建混合微调数据集（严格站点划分）")
    print(f"站点比例: {station_ratio*100:.0f}%")
    print(f"测试集比例: {test_ratio*100:.0f}% (按站点划分)")
    print(f"验证集比例: {val_ratio*100:.0f}% (按样本随机)")
    print(f"数据增强: 坐标抖动={coordinate_jitter_std}, 微波噪声={microwave_noise_std}, 坐标掩码={coordinate_mask_prob}")
    
    # 🔥 打印划分缓存配置
    if split_cache_file:
        print(f"📦 划分缓存文件: {split_cache_file}")
    if force_recompute_split:
        print(f"   ⚠️ 强制重新计算划分模式")
    print("="*60)
    
    # 🔥 划分缓存逻辑（在创建数据集之前）
    import pickle
    from pathlib import Path
    from datetime import datetime
    
    # 生成默认缓存文件名（如果未指定）
    if split_cache_file is None:
        import hashlib
        cache_key_str = f"{station_csv}_{val_ratio}_{test_ratio}_{seed}_mixed"
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()[:16]
        split_cache_file = f"./split_cache/mixed_split_{cache_key}.pkl"
    
    split_cache_path = Path(split_cache_file)
    split_cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 尝试加载划分缓存（但数据集还需要创建，所以这里先检查）
    cached_split = None
    if not force_recompute_split and split_cache_path.exists():
        print(f"\n📦 发现划分缓存: {split_cache_path}")
        try:
            with open(split_cache_path, 'rb') as f:
                cached_split = pickle.load(f)
            print(f"   ✅ 将使用缓存的划分")
        except Exception as e:
            print(f"   ⚠️ 缓存加载失败: {e}，将重新计算")
            cached_split = None
    
    try:
        # 🔥 过滤掉不应该传给 Dataset 的参数（预训练专用参数）
        exclude_keys = [
            'prefetch_factor', 'persistent_workers', 'num_workers',
            'use_station_guide', 'station_neighborhood', 
            'station_samples_per_day', 'station_csv_dir',
            'force_reload', 'shared_cache_dir',
            's1_interp_method', 's1_max_gap_days', 's1_nodata_value',
            'smap_interp_method', 'smap_max_gap_days', 'smap_nodata_value',
            'min_valid_pixels', 'samples_per_day', 'clamday_threshold',
            'patch_size', 'region'
        ]
        
        # 只保留 MixedFineTuneDataset 接受的参数
        dataset_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in exclude_keys}
        
        # 添加数据增强参数
        dataset_kwargs['coordinate_jitter_std'] = coordinate_jitter_std
        dataset_kwargs['microwave_noise_std'] = microwave_noise_std
        dataset_kwargs['coordinate_mask_prob'] = coordinate_mask_prob
        dataset_kwargs['use_tta'] = use_tta
        
        # 🔥 添加划分缓存参数到 dataset_kwargs
        dataset_kwargs['split_cache_file'] = split_cache_file
        dataset_kwargs['force_recompute_split'] = force_recompute_split
        
        # 可选：打印过滤后的参数（调试用）
        if dataset_kwargs:
            print(f"\n📋 传递给数据集的额外参数: {list(dataset_kwargs.keys())}")
        
        # 1. 创建混合数据集
        dataset = MixedFineTuneDataset(
            station_csv=station_csv,
            station_ratio=station_ratio,
            **dataset_kwargs
        )
        
        # 获取站点数据集部分
        station_ds = dataset.station_dataset
        
        # ============ 🔥 如果已有缓存，直接使用缓存的划分索引 ============
        if cached_split is not None and cached_split.get('total_samples') == len(station_ds):
            print(f"\n📦 使用缓存的划分索引...")
            # 🔥 统一使用不带 station_ 前缀的键名（兼容旧缓存格式）
            train_indices = cached_split['train_indices']
            val_indices = cached_split['val_indices']
            test_indices = cached_split['test_indices']
            test_stations_set = set(cached_split.get('test_stations', []))
            splits_info = cached_split.get('splits_info', {})
            
            # 🔥 兼容旧缓存：如果顶层没有 test_stations，尝试从 splits_info 或 test_indices 反推
            if not test_stations_set:
                if 'test_stations' in splits_info:
                    test_stations_set = set(splits_info['test_stations'])
                elif test_indices:
                    # 从 test_indices 反推站点
                    for idx in test_indices:
                        meta = station_ds.meta_index[idx]
                        sid = meta.get('station_id', '')
                        if isinstance(sid, str) and ',' in sid:
                            for s in sid.split(','):
                                test_stations_set.add(s.strip())
                        else:
                            test_stations_set.add(str(sid))
                if test_stations_set:
                    print(f"   ⚠ 旧缓存缺少 test_stations，已从 splits_info/test_indices 反推: {len(test_stations_set)} 个站点")
                else:
                    print(f"   ⚠ 旧缓存缺少 test_stations，且无法反推，可能是真正的 0 个独立测试站点")
            
            print(f"\n📊 从缓存加载的划分:")
            print(f"  训练集站点样本数: {len(train_indices)}")
            print(f"  验证集站点样本数: {len(val_indices)}")
            print(f"  测试集样本数: {len(test_indices)}")
            
        else:
            # ============ 重新计算划分 ============
            print(f"\n🔨 计算新的划分...")
            
            # ============ 🔥 辅助函数：提取所有关联站点ID ============
            def extract_all_station_ids(meta):
                """从 meta 中提取所有关联的站点ID"""
                station_ids = set()
                
                # 优先使用 source_stations（聚合样本）
                if 'source_stations' in meta and meta['source_stations']:
                    for sid in str(meta['source_stations']).split(','):
                        station_ids.add(sid.strip())
                else:
                    # 普通样本
                    station_id = meta['station_id']
                    if ',' in str(station_id):
                        for sid in str(station_id).split(','):
                            station_ids.add(sid.strip())
                    else:
                        station_ids.add(str(station_id))
                
                return station_ids
            
            # 构建站点到索引的映射（处理多站点关联）
            station_to_indices = {}
            multi_station_samples = set()
            
            print(f"\n📊 正在构建站点到样本的映射...")
            for idx, meta in enumerate(station_ds.meta_index):
                station_ids = extract_all_station_ids(meta)
                
                if len(station_ids) > 1:
                    multi_station_samples.add(idx)
                
                for sid in station_ids:
                    if sid not in station_to_indices:
                        station_to_indices[sid] = []
                    station_to_indices[sid].append(idx)
            
            # ============ 🔥 关键修改：强制排序，确保不同运行间顺序一致 ============
            unique_stations = sorted(list(station_to_indices.keys()))
            # ====================================================================
            
            n_unique_stations = len(unique_stations)
            n_total_samples = len(station_ds)
            
            print(f"\n📊 站点数据集统计:")
            print(f"  总样本数: {n_total_samples}")
            print(f"  唯一站点数: {n_unique_stations}")
            
            if multi_station_samples:
                print(f"  ⚠️ 多站点样本数: {len(multi_station_samples)}")
                print(f"     这些样本关联多个站点，可能造成数据泄露")
            
            # 2. 【按站点划分】出独立的测试集
            train_val_stations, test_stations_list = train_test_split(
                unique_stations, 
                test_size=test_ratio, 
                random_state=seed
            )
            test_stations_set = set(test_stations_list)
            train_val_stations_set = set(train_val_stations)
            
            print(f"\n📊 站点级划分:")
            print(f"  测试集站点数: {len(test_stations_set)}")
            print(f"  训练/验证池站点数: {len(train_val_stations_set)}")
            
            # 3. 收集索引（使用集合去重）
            test_indices_set = set()
            train_val_pool_set = set()
            
            for station_id, indices in station_to_indices.items():
                if station_id in test_stations_set:
                    test_indices_set.update(indices)
                else:
                    train_val_pool_set.update(indices)
            
            test_indices = list(test_indices_set)
            train_val_pool_indices = list(train_val_pool_set)
            
            # 检查泄露：测试集和训练/验证池是否有重叠
            overlap_indices = test_indices_set & train_val_pool_set
            if overlap_indices:
                print(f"\n  ❌ 严重警告: 发现 {len(overlap_indices)} 个样本同时出现在测试集和训练/验证池！")
                print(f"     这些样本关联多个站点，被分配到了不同的数据集。")
            
            print(f"\n📊 样本级统计:")
            print(f"  测试集样本数: {len(test_indices)}")
            print(f"  训练/验证池样本数: {len(train_val_pool_indices)}")
            
            # 4. 【按样本划分】训练集和验证集
            val_size_ratio = val_ratio / (1 - test_ratio) if test_ratio < 1 else 0
            
            if len(train_val_pool_indices) > 0 and val_size_ratio > 0:
                train_indices, val_indices = train_test_split(
                    train_val_pool_indices,
                    test_size=min(val_size_ratio, 0.5),  # 最多50%作为验证集
                    random_state=seed,
                    shuffle=True
                )
            else:
                train_indices = train_val_pool_indices
                val_indices = []
            
            print(f"\n📊 训练/验证集划分（样本随机）:")
            print(f"  训练集站点样本数: {len(train_indices)}")
            print(f"  验证集站点样本数: {len(val_indices)}")
            
            # 数据泄露检查（确保测试集站点不在训练/验证集中）
            train_site_ids = set()
            for idx in train_indices:
                site_ids = extract_all_station_ids(station_ds.meta_index[idx])
                train_site_ids.update(site_ids)
            
            val_site_ids = set()
            for idx in val_indices:
                site_ids = extract_all_station_ids(station_ds.meta_index[idx])
                val_site_ids.update(site_ids)
            
            overlap_train_test = train_site_ids & test_stations_set
            overlap_val_test = val_site_ids & test_stations_set
            
            print(f"\n🔍 数据泄露检查:")
            print(f"  测试集站点出现在训练集: {len(overlap_train_test)} 个")
            print(f"  测试集站点出现在验证集: {len(overlap_val_test)} 个")
            
            if len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
                print(f"  ✅ 测试集站点完全独立，无数据泄露！")
            else:
                print(f"  ⚠️ 警告: 存在数据泄露！")
                if overlap_train_test:
                    print(f"     重叠站点: {list(overlap_train_test)[:10]}")
            
            # 收集划分信息
            splits_info = {
                'split_strategy': 'mixed_site_sample_split',
                'test_stations': list(test_stations_set),
                'test_stations_count': len(test_stations_set),
                'test_samples': len(test_indices),
                'train_station_samples': len(train_indices),
                'val_samples': len(val_indices),
                'total_samples': len(station_ds),
                'station_ratio': station_ratio,
                'multi_station_samples': len(multi_station_samples),
                'has_data_leakage': len(overlap_indices) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0,
                'no_data_leakage': (len(overlap_train_test) == 0 and len(overlap_val_test) == 0),
                'seed': seed,
                'data_augmentation': {
                    'coordinate_jitter_std': coordinate_jitter_std,
                    'microwave_noise_std': microwave_noise_std,
                    'coordinate_mask_prob': coordinate_mask_prob,
                    'use_tta': use_tta
                }
            }
            
            # 🔥 保存划分到缓存（统一使用不带 station_ 前缀的键名）
            to_cache = {
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'test_stations': list(test_stations_set),
                'splits_info': splits_info,
                'total_samples': len(station_ds),
                'seed': seed,
                'timestamp': datetime.now().isoformat(),
            }
            
            with open(split_cache_path, 'wb') as f:
                pickle.dump(to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"\n💾 划分已保存到: {split_cache_path}")
        
        # 5. 处理预训练样本（预训练样本全部进入训练集）
        # 🔥 根据实际训练站点样本数重新选择 pretrain
        dataset.reselect_pretrain_by_train_count(len(train_indices))
        pretrain_indices = [len(station_ds) + i for i in range(len(dataset.selected_pretrain))]
        final_train_indices = train_indices + pretrain_indices
        
        print(f"\n📊 最终数据划分:")
        print(f"  训练集: {len(final_train_indices)} 个样本")
        print(f"    ├─ 站点样本: {len(train_indices)}")
        print(f"    └─ 预训练样本: {len(pretrain_indices)}")
        print(f"  验证集: {len(val_indices)} 个样本 (仅站点)")
        print(f"  测试集: {len(test_indices)} 个样本 (仅站点, {len(test_stations_set)} 个独立站点)")
        
        # 6. 构建 DataLoader
        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': True,
            'prefetch_factor': prefetch_factor if num_workers > 0 else None,
            'persistent_workers': persistent_workers if num_workers > 0 else False,
        }
        
        train_loader = DataLoader(
            Subset(dataset, final_train_indices), 
            shuffle=True, 
            **loader_kwargs
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_indices), 
            shuffle=False, 
            **loader_kwargs
        )
        
        test_loader = DataLoader(
            Subset(dataset, test_indices), 
            shuffle=False, 
            **loader_kwargs
        )
        
        # 更新 splits_info 添加最终样本数
        splits_info['train_samples'] = len(final_train_indices)
        splits_info['train_pretrain_samples'] = len(pretrain_indices)
        
        print(f"\n✅ 混合数据加载器构建成功!")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  验证批次: {len(val_loader)}")
        print(f"  测试批次: {len(test_loader)}")
        print(f"  划分缓存: {split_cache_path}")
        if splits_info.get('multi_station_samples', 0) > 0:
            print(f"  ⚠️ 多站点样本数: {splits_info['multi_station_samples']}")
        if splits_info.get('has_data_leakage', False):
            print(f"  ❌ 存在数据泄露！请检查多站点样本！")
        
        return train_loader, val_loader, test_loader, (dataset.C_conv, dataset.C_point), splits_info
        
    except Exception as e:
        print(f"❌ 构建混合数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # 测试代码
    print("测试站点SWE数据集...")
    
    try:
        dataset = StationSWEDataset()
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            conv, point, target, is_zero = dataset[0]
            print(f"卷积特征形状: {conv.shape}")
            print(f"点特征形状: {point.shape}")
            print(f"目标值: {target:.4f}")
            print(f"is_zero: {is_zero}")
        
        train_loader, val_loader, test_loader, shapes, splits_info = build_station_dataloaders_swe(
            batch_size=8,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        print(f"\n数据加载器构建成功!")
        print(f"特征维度: {shapes}")
        print(f"划分信息: {splits_info}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
