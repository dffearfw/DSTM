# data_online_era5_swe.py
# -*- coding: utf-8 -*-
"""
在线从栅格构建 SWE 反演训练样本
- 卷积特征: chelsa_sfxwind, lst, rh, clamday, dem
- 点特征: ls, S1_VV, S1_VH, 经纬度, doy
- 标签: fusedSWE
"""
from sklearn.model_selection import KFold
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
import pickle
import hashlib
import json
import pandas as pd
from scipy.interpolate import griddata
import time
import psutil
import gc
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
# ============= 配置区域 =============
REGION = "XINJIANG"
YEAR_TARGET = 2016
PATCH_SIZE = 5  # patch大小
R = PATCH_SIZE // 2

MIN_VALID_PIXELS = 50
SAMPLES_PER_DAY = 50000

# 卷积特征（参与卷积的变量）
CONV_VARS = ["chelsa_sfxwind", "lst", "rh", "pr"]
CONV_STATIC_VARS = ["clamday", "dem"]  # 静态卷积特征

# 点特征（不参与卷积）
POINT_VARS = ["ls", "S1_VV", "S1_VH", "SMAP_TBV", "SMAP_TBH"]  # 添加哨兵1和SMAP亮温

# 数据路径
FEATURE_ROOT = Path(r"/root/autodl-tmp/ablation")
LABEL_ROOT = Path(r"/root/autodl-tmp/ablation/fusedswe/cn")  # 标签路径


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

def conv_static_path(var: str, year, threshold: float = 0.5) -> List[Path]:
    """静态卷积变量路径 - 支持多波段DEM，返回所有DEM文件
    
    Args:
        var: 变量名，支持 "clamday" 或 "dem"
        year: 年份，可以是单个年份或年份列表
        threshold: clamday 的阈值参数（仅对 clamday 有效）
    
    Returns:
        文件路径列表
    """
    if var == "clamday":
        clamday_path = FEATURE_ROOT / "clamday" / "cn"
        
        if not clamday_path.exists():
            print(f"    ⚠ Clamday目录不存在: {clamday_path}")
            return []
        
        # 🔥 处理 year 可能是列表的情况
        if isinstance(year, list):
            # 多年份模式：匹配所有年份的文件
            files = []
            for y in year:
                # 匹配模式：*2015*threshold*.tif
                pattern = f"*{y}*threshold*.tif"
                found = list(clamday_path.glob(pattern))
                files.extend(found)
                if found:
                    print(f"      找到 {len(found)} 个 {y} 年的文件")
            files = list(set(files))  # 去重
        else:
            # 单年份模式
            pattern = f"*{year}*threshold*.tif"
            files = list(clamday_path.glob(pattern))
        
        # 如果没找到，尝试不指定年份（匹配所有threshold文件）
        if not files:
            pattern = f"*threshold*.tif"
            files = list(clamday_path.glob(pattern))
            if files:
                print(f"      使用模糊匹配: 找到 {len(files)} 个文件")
        
        # 如果还没找到，尝试所有tif
        if not files:
            files = list(clamday_path.glob("*.tif"))
            if files:
                print(f"      使用全部tif: 找到 {len(files)} 个文件")
        
        print(f"    Clamday文件: 最终找到 {len(files)} 个")
        if files:
            print(f"      例如: {files[0].name}")
        
        # 根据阈值过滤（如果需要）
        if threshold is not None and files:
            # 按threshold过滤
            threshold_str = f"threshold{threshold}"
            filtered = [f for f in files if threshold_str in f.name]
            if filtered:
                files = filtered
                print(f"      按阈值{threshold}过滤后: {len(files)} 个")
        
        return files
    
    elif var == "dem":
        dem_path = FEATURE_ROOT / "DEM"
        dem_files = []
        
        if not dem_path.exists():
            print(f"    ⚠ DEM目录不存在: {dem_path}")
            return []
        
        # 🔥 获取所有地形文件（主DEM）
        terrain_files = list(dem_path.glob("*_Terrain_China_27830m.tif"))
        if terrain_files:
            dem_files.append(terrain_files[0])
            print(f"    找到DEM Terrain文件: {terrain_files[0].name}")
        else:
            # 尝试备用名称
            terrain_files = list(dem_path.glob("*Terrain*.tif"))
            if terrain_files:
                dem_files.append(terrain_files[0])
                print(f"    找到DEM Terrain文件(备用): {terrain_files[0].name}")
            else:
                print(f"    ⚠ 未找到DEM Terrain文件")
        
        # 🔥 获取标准差文件
        stddev_files = list(dem_path.glob("*_Terrain_StdDev_China_27830m.tif"))
        if stddev_files:
            dem_files.append(stddev_files[0])
            print(f"    找到DEM StdDev文件: {stddev_files[0].name}")
        else:
            # 尝试备用名称
            stddev_files = list(dem_path.glob("*StdDev*.tif"))
            if stddev_files:
                dem_files.append(stddev_files[0])
                print(f"    找到DEM StdDev文件(备用): {stddev_files[0].name}")
            else:
                print(f"    ⚠ 未找到DEM StdDev文件")
        
        # 🔥 可选：查找其他DEM相关文件（如坡度、坡向等）
        # 如果你有其他DEM衍生数据，可以在这里添加
        other_dem_files = list(dem_path.glob("*_Terrain_*.tif"))
        for f in other_dem_files:
            if f not in dem_files:
                dem_files.append(f)
                print(f"    找到额外DEM文件: {f.name}")
        
        print(f"    DEM文件: 找到 {len(dem_files)} 个有效文件")
        return dem_files
    
    else:
        raise ValueError(f"未知的静态卷积变量: {var}")

def point_var_path(var: str, year) -> Union[Path, List[Path]]:
    """点变量路径 - 支持多年份列表"""
    
    if var == "ls":
        ls_path = FEATURE_ROOT / "ls" / "cn"
        
        if not ls_path.exists():
            print(f"    LS目录不存在: {ls_path}")
            return None
        
        # 🔥 处理多年份列表
        if isinstance(year, list):
            # 优先匹配指定年份的文件
            for y in year:
                ls_file = ls_path / f"China_Landsat_{y}_reflectance.tif"
                if ls_file.exists():
                    print(f"    找到LS文件: {ls_file.name} (年份: {y})")
                    return ls_file
            # 如果都没找到，使用第一个匹配的
            files = list(ls_path.glob("China_Landsat_*_reflectance.tif"))
            if files:
                print(f"    使用模糊匹配: {files[0].name}")
                return files[0]
        else:
            # 单年份
            ls_file = ls_path / f"China_Landsat_{year}_reflectance.tif"
            if ls_file.exists():
                print(f"    找到LS文件: {ls_file.name}")
                return ls_file
        
        print(f"    ⚠ 未找到LS文件")
        return None
    
    elif var == "S1_VV" or var == "S1_VH":
        s1_path = FEATURE_ROOT / "s1" / "cn"
        
        if not s1_path.exists():
            print(f"    哨兵1目录不存在: {s1_path}")
            return []
        
        # 🔥 处理多年份列表
        if isinstance(year, list):
            files = []
            for y in year:
                pattern = f"S1_MONTHLY_{y}_*.tif"
                found = list(s1_path.glob(pattern))
                files.extend(found)
                if found:
                    print(f"      找到 {len(found)} 个 {y} 年的哨兵1文件")
            files = sorted(list(set(files)))  # 去重并排序
        else:
            pattern = f"S1_MONTHLY_{year}_*.tif"
            files = sorted(list(s1_path.glob(pattern)))
        
        # 如果没找到，尝试模糊匹配
        if not files:
            files = sorted(list(s1_path.glob("S1_MONTHLY_*.tif")))
            if files:
                print(f"      使用模糊匹配: 找到 {len(files)} 个哨兵1文件")
        
        print(f"    哨兵1文件: 最终找到 {len(files)} 个")
        if files:
            print(f"      日期范围: {files[0].name} 到 {files[-1].name}")
        return files
    
    elif var == "SMAP_TBV" or var == "SMAP_TBH":
        smap_root = Path(r"/root/autodl-tmp/ablation/smap/cn")
        
        if not smap_root.exists():
            print(f"    SMAP目录不存在: {smap_root}")
            return []
        
        # 🔥 修改：新的文件命名模式
        if isinstance(year, list):
            files = []
            for y in year:
                # 新模式: SMAP_2015_07_cube_drive-download-20260.tif
                pattern = f"SMAP_{y}_*_cube_drive-download-20260.tif"
                found = list(smap_root.glob(pattern))
                files.extend(found)
                if found:
                    print(f"      找到 {len(found)} 个 {y} 年的SMAP文件")
            files = sorted(list(set(files)))
        else:
            pattern = f"SMAP_{year}_*_cube_drive-download-20260.tif"
            files = sorted(list(smap_root.glob(pattern)))
        
        # 如果没找到，尝试模糊匹配
        if not files:
            files = sorted(list(smap_root.glob("SMAP_*_cube_drive-download-20260.tif")))
            if files:
                print(f"      使用模糊匹配: 找到 {len(files)} 个SMAP文件")
        
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
            use_station_guide: bool = False,           
            station_csv_dir: Optional[Path] = None,    
            station_neighborhood: int = 3,             
            station_samples_per_day: int = 2000,
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

        self.use_station_guide = use_station_guide
        self.station_csv_dir = station_csv_dir or Path("/root/autodl-tmp/ablation")
        self.station_neighborhood = station_neighborhood
        self.station_samples_per_day = station_samples_per_day
        self.station_pixels = set()

        self.use_adaptive_supplement = use_adaptive_supplement
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_swe_bins = adaptive_swe_bins or [0, 5, 10, 20, 30, 50, 80, 120, 200, 500]

        print(f"\n📌 采样配置:")
        print(f"   站点引导: {'启用' if use_station_guide else '禁用'}")
        print(f"   自适应修正: {'启用' if use_adaptive_supplement else '禁用'}")
        if use_adaptive_supplement:
            print(f"   平衡强度 α = {adaptive_alpha}")
            print(f"   短缺阈值 = {adaptive_threshold}")
            print(f"   SWE区间: {self.adaptive_swe_bins}")

        if isinstance(year_target, list):
            self.load_years = year_target
        else:
            self.load_years = [year_target, year_target - 1]

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
                'use_station_guide': use_station_guide,
                'station_neighborhood': station_neighborhood,
                'station_samples_per_day': station_samples_per_day,
                'use_adaptive_supplement': use_adaptive_supplement,
                'adaptive_alpha': adaptive_alpha,
                'adaptive_threshold': adaptive_threshold,
                'adaptive_swe_bins': adaptive_swe_bins,
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
                        if not hasattr(self, 'station_pixels') or not self.station_pixels:
                            print("   ⚠ 缓存缺少 station_pixels，正在重建...")
                            self._load_all_station_pixels()
                            if self.station_neighborhood > 0 and self.station_pixels:
                                self.station_pixels = self._expand_neighborhood(
                                    self.station_pixels, 
                                    self.station_neighborhood
                                )
                            print(f"   ✅ station_pixels 重建完成")

                    self._validate_cached_data()

                    print(f"\n{'='*60}")
                    print(f"✅ 从缓存加载数据集完成!")
                    print(f"  总样本数: {len(self.meta_index):,}")
                    print(f"  卷积特征维度: {self.C_conv}")
                    print(f"  点特征维度: {self.C_point}")
                    print(f"{'='*60}\n")

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

        # ============ 🔥 添加 FusedSWE 原始分布验证 ============
        print("\n" + "="*70)
        print("🔍 检查 FusedSWE 产品原始分布（采样前）")
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
        print(f"  0-5mm 占比: {np.sum(all_swe_values <= 5) / len(all_swe_values) * 100:.4f}%")
        print(f"  5-10mm 占比: {np.sum((all_swe_values > 5) & (all_swe_values <= 10)) / len(all_swe_values) * 100:.4f}%")
        print(f"  10-20mm 占比: {np.sum((all_swe_values > 10) & (all_swe_values <= 20)) / len(all_swe_values) * 100:.4f}%")
        print(f"  20-30mm 占比: {np.sum((all_swe_values > 20) & (all_swe_values <= 30)) / len(all_swe_values) * 100:.4f}%")
        print(f"  30-50mm 占比: {np.sum((all_swe_values > 30) & (all_swe_values <= 50)) / len(all_swe_values) * 100:.4f}%")
        print(f"  50-80mm 占比: {np.sum((all_swe_values > 50) & (all_swe_values <= 80)) / len(all_swe_values) * 100:.4f}%")
        print(f"  80-120mm 占比: {np.sum((all_swe_values > 80) & (all_swe_values <= 120)) / len(all_swe_values) * 100:.4f}%")
        print(f"  120-200mm 占比: {np.sum((all_swe_values > 120) & (all_swe_values <= 200)) / len(all_swe_values) * 100:.4f}%")
        print(f"  >200mm 占比: {np.sum(all_swe_values > 200) / len(all_swe_values) * 100:.1f}%")
        print("="*70 + "\n")

        # 计算卷积通道数
        self.C_conv = len(CONV_VARS) + 1 + len(self.dem_data)

        print(f"\n📊 卷积特征维度统计:")
        print(f"  动态变量: {len(CONV_VARS)}")
        print(f"  静态变量 (Clamday): 1")
        print(f"  DEM波段: {len(self.dem_data)}")
        print(f"  → C_conv = {self.C_conv}")

        # 加载站点数据
        if self.use_station_guide:
            self._load_all_station_pixels()
            if self.station_neighborhood > 0 and self.station_pixels:
                self.station_pixels = self._expand_neighborhood(
                    self.station_pixels, 
                    self.station_neighborhood
                )
            print(f"  站点像元数: {len(self.station_pixels):,}")

        # 构建样本索引
        self._build_sample_index()
        self._compute_minmax_sampling()

        print(f"\n{'='*60}")
        print(f"✅ 数据集初始化完成!")
        print(f"  总样本数: {len(self.meta_index):,}")
        print(f"  C_conv: {self.C_conv}")
        print(f"  C_point: {self.C_point}")
        print(f"{'='*60}\n")

        # 保存缓存
        if cache_dir is not None and cache_path is not None:
            self._save_cache(cache_path, cache_key)

        self._precompute_and_cache()
        
        
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
            'use_station_guide',        # 是否启用
            'station_neighborhood',     # 邻域半径
            'station_samples_per_day',  # 每日上限
            'station_csv_dir',          # 站点CSV目录

            # 维度信息
            'C_conv', 'C_point', 'H', 'W',

            # 归一化参数
            'conv_min', 'conv_max', 'point_min', 'point_max',
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
                    # 🔥 新增：站点引导参数
                    'use_station_guide': getattr(self, 'use_station_guide', False),
                    'station_neighborhood': getattr(self, 'station_neighborhood', 3),
                    'station_samples_per_day': getattr(self, 'station_samples_per_day', 2000),
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
            "chelsa_sfxwind": 0.0,
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
                    match = re.search(r'S1_MONTHLY_(\d{4})_(\d{2})', filename)
                    if match:
                        year = int(match.group(1))
                        month = int(match.group(2))
                    else:
                        match = re.search(r'(\d{4})_(\d{2})', filename)
                        if match:
                            year = int(match.group(1))
                            month = int(match.group(2))
                        else:
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

                        date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', desc_clean)
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

    def _load_single_variable(self, var: str):
        """加载单个变量的数据 - 支持多年份"""
        print(f"  加载 {var} 数据...")

        # 🔥 统一使用 self.load_years
        target_years = self.load_years

        # 如果需要更多历史数据，可以扩展
        all_years = set(target_years)
        for year in target_years:
            all_years.add(year - 1)  # 添加前一年用于时间序列

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
                files = list(var_dir.glob(f"ERA5_ST_{year}*.tif"))
            elif var == "rh":
                files = list(var_dir.glob(f"ERA5_RH_DailyMean_{year}_*.tif"))
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

        else:  # chelsa_sfxwind, pr等
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

            if not dated_files:
                print(f"  没有可用的文件")
                return None, [], None, None

            with rasterio.open(dated_files[0][1]) as ds:
                H, W = ds.shape

            var_arr = np.zeros((len(dated_files), H, W), dtype=np.float32)

            for i, (dt, f) in enumerate(dated_files):
                try:
                    with rasterio.open(f) as ds:
                        data = ds.read(1).astype(np.float32)

                        # 🔥 关键修改：pr 的 NaN 替换为 -9999
                        if var == "pr":
                            data = np.nan_to_num(data, nan=-9999.0)

                        var_arr[i] = data
                except Exception as e:
                    print(f"    读取 {f.name} 失败: {e}")
                    if var == "pr":
                        var_arr[i] = np.full((H, W), -9999.0, dtype=np.float32)
                    else:
                        var_arr[i] = np.zeros((H, W), dtype=np.float32)

            # ========== 🔥 验证代码 ==========
            if var == "pr":
                print(f"\n  【验证】pr 数据返回前:")
                print(f"    数据形状: {var_arr.shape}")
                print(f"    NaN 数量: {np.isnan(var_arr).sum()}")
                print(f"    -9999 数量: {(var_arr == -9999).sum()}")
                print(f"    有限值数量: {np.isfinite(var_arr).sum()}")
            # ================================

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

        # 处理源数据中的 NaN（设为 nodata）
        src_nodata = np.nan if np.isnan(src_data).any() else None

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

            # 🔥 统计有效数据 - Clamday 无效值是 -11
            CLAMDAY_INVALID = -11.0
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

        smap_root = Path(r"/root/autodl-tmp/ablation/smap/cn")
        if not smap_root.exists():
            print(f"    ⚠ SMAP目录不存在: {smap_root}")
            return

        # 🔥 使用 self.load_years 获取要加载的年份列表
        target_years = self.load_years
        print(f"    加载年份: {target_years}")

        # 收集所有匹配的SMAP文件
        all_smap_files = []
        for year in target_years:
            pattern = f"SMAP_{year}_*_cube_drive-download-20260.tif"
            files = list(smap_root.glob(pattern))
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
                match = re.search(r'SMAP_(\d{4})_(\d{2})', filename)
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

                        date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})_([HV])', desc_clean)

                        if date_match:
                            y, m, d = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                            pol = date_match.group(4)
                        else:
                            date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', desc_clean)
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

        # 🔥 LS 改为按年份分别加载
        self.ls_data = {}  # 改为字典: year -> array
        ls_path = FEATURE_ROOT / "ls" / "cn"

        for year in self.load_years:
            ls_file = ls_path / f"China_Landsat_{year}_reflectance.tif"
            if ls_file.exists():
                print(f"  处理LS文件: {ls_file.name} (年份: {year})")

                with rasterio.open(ls_file) as ds:
                    ls_data_raw = ds.read()  # (C_ls, H, W)
                    src_transform = ds.transform

                # 对齐每个波段
                aligned_bands = []
                for i in range(ls_data_raw.shape[0]):
                    band_aligned = self._align_single_layer(
                        ls_data_raw[i], src_transform, self.transform, self.H, self.W
                    )
                    aligned_bands.append(band_aligned)

                self.ls_data[year] = np.stack(aligned_bands, axis=0)
                print(f"    LS{year}数据形状: {self.ls_data[year].shape}")
            else:
                print(f"  警告: 未找到LS文件 China_Landsat_{year}_reflectance.tif")
                self.ls_data[year] = np.zeros((6, self.H, self.W), dtype=np.float32)

        # 默认使用第一年的LS数据（兼容旧代码）
        self.ls_data_default = self.ls_data.get(self.load_years[0], np.zeros((6, self.H, self.W)))

        # 加载哨兵1数据
        self._load_sentinel1_data()

        # 加载SMAP数据
        self._load_smap_data()
    
    def _load_labels_unified(self):
        """加载标签数据 - 支持多年份"""
        print(f"\n加载标签数据...")

        self.label_data = {}

        # 🔥 使用 self.load_years 而不是 self.year_target
        target_years = self.load_years
        print(f"  加载标签年份: {target_years}")

        # 查找所有标签文件
        label_files = sorted(list(self.label_root.glob("*.tif")))
        print(f"  找到 {len(label_files)} 个标签文件")

        loaded_count = 0
        year_count = {}

        for label_file in label_files:
            try:
                name = label_file.name
                dt = self._parse_date_from_filename(name)

                if dt.year not in target_years:
                    continue

                with rasterio.open(label_file) as ds:
                    label_arr = ds.read(1).astype(np.float32)
                    label_nodata = ds.nodata

                if label_arr.shape != (self.H, self.W):
                    label_arr = self._resize_to_standard(label_arr, self.H, self.W)

                self.label_data[dt] = (label_arr, label_nodata)
                loaded_count += 1
                year_count[dt.year] = year_count.get(dt.year, 0) + 1

                if loaded_count <= 5:
                    print(f"  {dt.strftime('%Y-%m-%d')}: 加载成功")

            except Exception as e:
                print(f"  加载标签文件 {label_file.name} 失败: {e}")
                continue

        if not self.label_data:
            raise ValueError(f"没有加载到任何标签数据 (目标年份: {target_years})")

        print(f"\n✅ 标签数据加载完成:")
        print(f"  总加载数: {loaded_count}")
        for year, count in sorted(year_count.items()):
            print(f"    {year}年: {count} 个文件")
            

    def _parse_date_from_filename(self, filename: str) -> datetime:
        """从文件名解析日期 - 支持中国区域的新格式"""

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

    def _build_sample_index(self):
        """构建样本索引 - 支持自适应修正"""

        # ============ 阶段1: 随机采样 ============
        self._build_random_samples()

        # ============ 阶段2: 站点引导采样（叠加） ============
        if self.use_station_guide and self.station_pixels:
            self._build_station_guided_samples()

        # ============ 阶段3: 自适应修正（补充短缺区间） ============
        if self.use_adaptive_supplement:
            self._adaptive_supplement()

        # 打乱顺序
        np.random.shuffle(self.meta_index)
        self._print_sample_statistics()


    def _build_random_samples(self):
        """原有的随机采样逻辑（增加高值采样权重，降低低值采样权重）"""

        # ============ 质量控制参数 ============
        ZERO_TARGET_MAX_RATIO = 0.1  # target=0样本的最大比例

        self.meta_index = []
        samples_per_date = {}
        samples_per_year = {}

        # 统计信息
        total_candidates = 0
        zero_target_samples = []
        non_zero_target_samples = []

        # 用于控制跳过日期的打印次数
        skip_log_count = 0

        # 🔥 采样权重配置
        LOW_SWE_THRESHOLD = 5      # 小于等于5mm视为低值
        LOW_SWE_WEIGHT = 0.3       # 低值采样权重（降低）
        HIGH_SWE_THRESHOLD = 30    # 大于30mm视为高值
        HIGH_SWE_WEIGHT = 5.0      # 高值采样权重倍数（提高）

        # 🔥 记录高值样本的位置信息（>80mm）
        high_value_samples = []

        for date_dt, (label_arr, label_nodata) in self.label_data.items():
            # 🔥 严格匹配：日期必须在卷积特征时间轴中
            if date_dt not in self.date_to_index:
                if skip_log_count < 10:
                    print(f"跳过日期 {date_dt.strftime('%Y-%m-%d')}，不在卷积特征时间轴中")
                    skip_log_count += 1
                continue

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

            # ============ 🔥 候选像素（带权重：低值降权，高值提权）============
            candidate_with_weights = []
            for (r, c) in np.argwhere(valid_mask):
                # 检查边界
                r0, r1 = r - self.R, r + self.R + 1
                c0, c1 = c - self.R, c + self.R + 1
                if r0 < 0 or r1 > self.H or c0 < 0 or c1 > self.W:
                    continue

                target_val = label_arr[r, c]

                # 🔥 根据 SWE 值设置权重
                if target_val <= LOW_SWE_THRESHOLD:
                    # 低值：降权
                    weight = LOW_SWE_WEIGHT
                elif target_val > HIGH_SWE_THRESHOLD:
                    # 高值：提权
                    weight = HIGH_SWE_WEIGHT
                else:
                    # 中间值：正常权重
                    weight = 1.0

                candidate_with_weights.append(((r, c), weight))

            if not candidate_with_weights:
                continue

            total_candidates += len(candidate_with_weights)

            # ============ 🔥 带权重的采样 ============
            # 提取像素和权重
            pixels = [p for p, w in candidate_with_weights]
            weights = np.array([w for p, w in candidate_with_weights])

            # 归一化权重
            weights = weights / weights.sum()

            # 确定采样数量
            if self.samples_per_day is not None:
                n_samples = min(self.samples_per_day, len(pixels))
            else:
                n_samples = len(pixels)

            # 🔥 使用权重进行采样
            if len(pixels) > n_samples:
                selected_indices = np.random.choice(
                    len(pixels), 
                    size=n_samples, 
                    replace=False, 
                    p=weights
                )
                candidate_indices = [pixels[i] for i in selected_indices]
            else:
                candidate_indices = pixels

            # ============ 按target值分类 ============
            date_zero_samples = []
            date_non_zero_samples = []
            high_swe_sampled = 0  # 统计高值采样数量
            low_swe_sampled = 0   # 统计低值采样数量

            for r, c in candidate_indices:
                try:
                    # 检查target值
                    target_val = label_arr[r, c]
                    is_zero_target = (target_val == 0)

                    # 🔥 统计高低值采样
                    if target_val > HIGH_SWE_THRESHOLD:
                        high_swe_sampled += 1
                    elif target_val <= LOW_SWE_THRESHOLD:
                        low_swe_sampled += 1

                    # 验证特征能否正常构建（严格标准）
                    conv_patch = self._build_spatial_features(date_dt, r, c)
                    point_feats = self._build_point_features(date_dt, r, c)

                    if conv_patch is not None and point_feats is not None:
                        if is_zero_target:
                            date_zero_samples.append((r, c))
                        else:
                            date_non_zero_samples.append((r, c))

                            # 🔥 记录高值样本（>80mm）的位置
                            if target_val > 80:
                                lon, lat = self._pixel_to_lonlat(r, c)
                                high_value_samples.append({
                                    'source': 'random_sampling',
                                    'bin': self._get_swe_bin(target_val),
                                    'date': date_dt.strftime('%Y-%m-%d'),
                                    'row': r,
                                    'col': c,
                                    'swe': float(target_val),
                                    'longitude': lon,
                                    'latitude': lat
                                })

                except Exception as e:
                    continue

            # 打印采样统计（每10天打印一次）
            if len(samples_per_date) % 10 == 0 and len(samples_per_date) > 0:
                total_sampled = len(candidate_indices)
                print(f"    {date_dt.strftime('%Y-%m-%d')}: 采样 {total_sampled} 个, "
                      f"低值({LOW_SWE_THRESHOLD}mm) {low_swe_sampled}/{total_sampled} ({low_swe_sampled/total_sampled*100:.1f}%), "
                      f"高值(>{HIGH_SWE_THRESHOLD}mm) {high_swe_sampled}/{total_sampled} ({high_swe_sampled/total_sampled*100:.1f}%)")

            # 记录当天样本数
            samples_per_date[date_dt] = len(date_zero_samples) + len(date_non_zero_samples)
            samples_per_year[year] += len(date_zero_samples) + len(date_non_zero_samples)

            # 先添加所有非零样本（标记来源 'random'）
            for r, c in date_non_zero_samples:
                non_zero_target_samples.append((date_dt, r, c, 'random'))

            # 根据当前总体比例，决定添加多少零样本
            current_total = len(non_zero_target_samples) + len(zero_target_samples)
            if current_total > 0:
                current_zero_ratio = len(zero_target_samples) / current_total
            else:
                current_zero_ratio = 0

            # 如果当前零样本比例还低于阈值，可以添加一些
            if current_zero_ratio < ZERO_TARGET_MAX_RATIO:
                # 计算还能添加多少个零样本
                max_zero_allowed = int(len(non_zero_target_samples) * ZERO_TARGET_MAX_RATIO / (1 - ZERO_TARGET_MAX_RATIO))
                can_add_zero = max_zero_allowed - len(zero_target_samples)

                # 添加零样本（不超过可用数量）
                add_zero_count = min(len(date_zero_samples), max(0, can_add_zero))
                if add_zero_count > 0:
                    # 随机选择一些零样本
                    np.random.shuffle(date_zero_samples)
                    for r, c in date_zero_samples[:add_zero_count]:
                        zero_target_samples.append((date_dt, r, c, 'random'))

        # 合并样本
        for item in non_zero_target_samples:
            self.meta_index.append(item)
        for item in zero_target_samples:
            self.meta_index.append(item)

        # 保存随机采样的统计信息供后续使用
        self.random_sample_stats = {
            'total_candidates': total_candidates,
            'non_zero_count': len(non_zero_target_samples),
            'zero_count': len(zero_target_samples),
            'samples_per_year': samples_per_year,
            'samples_per_date': samples_per_date,
        }

        print(f"\n  随机采样完成:")
        print(f"    候选点: {total_candidates:,}")
        print(f"    样本数: {len(self.meta_index):,}")
        print(f"    其中 target>0: {len(non_zero_target_samples):,}")
        print(f"    其中 target=0: {len(zero_target_samples):,}")

        # 🔥 打印采样权重配置
        print(f"    采样权重: 低值(≤{LOW_SWE_THRESHOLD}mm)权重 {LOW_SWE_WEIGHT}x, "
              f"高值(>{HIGH_SWE_THRESHOLD}mm)权重 {HIGH_SWE_WEIGHT}x")

        # 🔥 保存高值样本位置信息到 CSV
        if high_value_samples:
            import pandas as pd
            df_high = pd.DataFrame(high_value_samples)

            # 保存到文件
            if hasattr(self, 'cache_dir') and self.cache_dir:
                save_path = Path(self.cache_dir) / "random_high_value_samples.csv"
            else:
                save_path = Path("/root/autodl-tmp") / "random_high_value_samples.csv"

            df_high.to_csv(save_path, index=False, encoding='utf-8')
            print(f"\n   📍 随机采样高值样本位置已保存: {save_path}")
            print(f"      共 {len(high_value_samples)} 个高值样本(>80mm)")

            # 按区间统计
            print(f"\n   📊 随机采样高值样本按区间统计:")
            for bin_name in df_high['bin'].unique():
                count = len(df_high[df_high['bin'] == bin_name])
                print(f"      {bin_name}: {count} 个样本")

    def _get_swe_bin(self, swe_val):
        """根据 SWE 值返回区间名称"""
        bins = self.adaptive_swe_bins
        for i in range(len(bins) - 1):
            if bins[i] <= swe_val < bins[i+1]:
                return f"{bins[i]}-{bins[i+1]}"
        return f"{bins[-1]}+"

    def _build_station_guided_samples(self):
        """站点引导采样（宽松标准，纯粹叠加）- 添加完整特征统计"""

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
            'cum_pr_values': [],
            'cum_snow_values': [],
            'product_values': [],

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
                if (0 <= r < self.H and 0 <= c < self.W and 
                    valid_mask[r, c] and
                    (date_dt, r, c) not in existing_pixels):
                    candidates.append((r, c))

            if not candidates:
                continue

            np.random.shuffle(candidates)
            n_take = min(self.station_samples_per_day, len(candidates))
            selected = candidates[:n_take]

            added_today = 0
            for r, c in selected:
                feature_stats['total_attempted'] += 1

                # 🔥 修改：接收4个返回值
                is_valid, fail_reason, conv_patch, point_feats = self._validate_station_sample_with_reason(date_dt, r, c)

                if is_valid and point_feats is not None and conv_patch is not None:
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
                    # 15-16 经纬度, 17 DOY, 18 总降水, 19 降雪, 20 产品值
                    feature_stats['cum_pr_values'].append(float(point_feats[18]))
                    feature_stats['cum_snow_values'].append(float(point_feats[19]))

                    # 产品值
                    feature_stats['product_values'].append(float(point_feats[20]))

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
        print(f"    有真实哨兵1的样本: {feature_stats['has_real_s1']} ({feature_stats['has_real_s1']/station_samples_added*100:.1f}%)")
        print(f"    有真实SMAP的样本: {feature_stats['has_real_smap']} ({feature_stats['has_real_smap']/station_samples_added*100:.1f}%)")
        print(f"    有任一真实微波的样本: {feature_stats['has_real_microwave']} ({feature_stats['has_real_microwave']/station_samples_added*100:.1f}%)")
        print(f"    全是默认值的样本: {feature_stats['all_default_microwave']} ({feature_stats['all_default_microwave']/station_samples_added*100:.1f}%)")

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

        if feature_stats['cum_pr_values']:
            print(f"\n【降水特征分布】")
            print(f"    30天总降水: 均值={np.mean(feature_stats['cum_pr_values']):.2f}, 标准差={np.std(feature_stats['cum_pr_values']):.2f}")
            print(f"    30天有效降雪: 均值={np.mean(feature_stats['cum_snow_values']):.2f}, 标准差={np.std(feature_stats['cum_snow_values']):.2f}")

        if feature_stats['product_values']:
            print(f"\n【产品值分布】")
            print(f"    FusedSWE: 均值={np.mean(feature_stats['product_values']):.2f}, 标准差={np.std(feature_stats['product_values']):.2f}, "
                  f"范围=[{np.min(feature_stats['product_values']):.2f}, {np.max(feature_stats['product_values']):.2f}]")

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
            self._analyze_swe_distribution()
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

        self._analyze_swe_distribution()
        
        
    def _analyze_swe_distribution(self):
        """分析并可视化补充后的SWE分布"""

        print(f"\n{'='*70}")
        print(f"📊 补充后样本SWE分布分析")
        print(f"{'='*70}")

        self.setup_chinese_fonts()

        # 1. 收集所有样本的SWE值
        swe_values = []
        swe_by_bin = {f"{self.adaptive_swe_bins[i]}-{self.adaptive_swe_bins[i+1]}": [] 
                      for i in range(len(self.adaptive_swe_bins)-1)}
        swe_by_bin["200+"] = []

        for item in self.meta_index:
            date_dt, r, c = item[:3]
            if date_dt in self.label_data:
                label_arr, label_nodata = self.label_data[date_dt]
                swe = label_arr[r, c]
                if (label_nodata is None or swe != label_nodata) and np.isfinite(swe):
                    swe_values.append(float(swe))

                    # 按区间分类
                    if swe < 200:
                        for i in range(len(self.adaptive_swe_bins)-1):
                            if self.adaptive_swe_bins[i] <= swe < self.adaptive_swe_bins[i+1]:
                                bin_key = f"{self.adaptive_swe_bins[i]}-{self.adaptive_swe_bins[i+1]}"
                                swe_by_bin[bin_key].append(swe)
                                break
                    else:
                        swe_by_bin["200+"].append(swe)

        swe_values = np.array(swe_values)

        # 2. 打印统计信息
        print(f"\n📈 基础统计:")
        print(f"   总样本数: {len(swe_values):,}")
        print(f"   SWE范围: [{swe_values.min():.2f}, {swe_values.max():.2f}] mm")
        print(f"   均值: {swe_values.mean():.2f} ± {swe_values.std():.2f} mm")
        print(f"   中位数: {np.median(swe_values):.2f} mm")

        print(f"\n📊 分位数统计:")
        for p in [50, 75, 90, 95, 99]:
            print(f"   {p}%: {np.percentile(swe_values, p):.2f} mm")

        print(f"\n📋 按区间分布:")
        print(f"{'区间 (mm)':<15} {'样本数':<12} {'占比':<12} {'状态'}")
        print("-" * 55)

        total = len(swe_values)
        for bin_key, values in swe_by_bin.items():
            count = len(values)
            ratio = count / total * 100
            if count == 0:
                status = "❌ 无样本"
            elif ratio < 1:
                status = "⚠️ 偏低"
            elif ratio < 5:
                status = "✓ 正常"
            else:
                status = "✅ 充足"
            # 🔥 改成 .4f，显示4位小数
            print(f"{bin_key:<15} {count:<12,} {ratio:<12.4f}% {status}")

        # 3. 绘制直方图 - 🔥 使用等宽区间
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 左图：完整分布（等宽区间，每10mm）
            ax1 = axes[0]
            # 🔥 等宽区间：0-200mm，每10mm一个区间
            equal_bins = np.arange(0, 201, 10)  # [0,10,20,...,200]

            # 只取 <=200 的样本（高值太少，单独处理）
            swe_low = swe_values[swe_values <= 200]
            ax1.hist(swe_low, bins=equal_bins, edgecolor='black', alpha=0.7, color='steelblue')
            ax1.set_xlabel('SWE (mm)', fontsize=12)
            ax1.set_ylabel('样本数', fontsize=12)
            ax1.set_title(f'补充后样本SWE分布 (等宽区间, n={len(swe_low):,})', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # 添加高值样本标注
            high_count = len(swe_values[swe_values > 200])
            if high_count > 0:
                ax1.text(0.95, 0.95, f'SWE >200mm: {high_count} 个样本', 
                        transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

            # 右图：低值区放大（0-50mm，更细的等宽区间）
            ax2 = axes[1]
            # 🔥 0-50mm，每5mm一个区间
            fine_bins = np.arange(0, 55, 5)
            swe_fine = swe_values[swe_values <= 50]
            ax2.hist(swe_fine, bins=fine_bins, edgecolor='black', alpha=0.7, color='coral')
            ax2.set_xlabel('SWE (mm)', fontsize=12)
            ax2.set_ylabel('样本数', fontsize=12)
            ax2.set_title(f'低值区放大 (0-50mm, n={len(swe_fine):,})', fontsize=14)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图片
            if hasattr(self, 'cache_dir') and self.cache_dir:
                save_path = Path(self.cache_dir) / "swe_distribution_after_adaptive.png"
            else:
                save_path = Path("/root/autodl-tmp") / "swe_distribution_after_adaptive.png"

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\n📊 直方图已保存: {save_path}")
            print(f"   (使用等宽区间: 左图10mm/格, 右图5mm/格)")

        except Exception as e:
            print(f"   ⚠ 绘图失败: {e}")

        print(f"{'='*70}\n")
        
        
    def _get_natural_weights(self, swe_bins):
        """获取候选池的自然分布权重"""
        n_bins = len(swe_bins) - 1
        counts = [0] * n_bins

        for date_dt, (label_arr, _) in self.label_data.items():
            if date_dt not in self.date_to_index:
                continue
            valid_mask = np.isfinite(label_arr)
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
            "chelsa_sfxwind": 0.0,      # 风速无效值可能是 0
            "lst": -9999.0,
            "rh": -9999.0,
            "pr": -9999.0,
        }
        CLAMDAY_INVALID = -11.0

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

        # 🔥 Clamday 无效值 -11 转为 NaN
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
            


    def _print_sample_statistics(self):
        """打印最终样本统计信息"""
        print(f"\n{'='*60}")
        print(f"✅ 样本构建完成:")
        print(f"  总样本数: {len(self.meta_index):,}")

        # 统计随机采样部分
        if hasattr(self, 'random_sample_stats'):
            print(f"\n  【随机采样】")
            print(f"    样本数: {self.random_sample_stats['non_zero_count'] + self.random_sample_stats['zero_count']:,}")
            print(f"    其中 target>0: {self.random_sample_stats['non_zero_count']:,}")
            print(f"    其中 target=0: {self.random_sample_stats['zero_count']:,}")
            zero_ratio = self.random_sample_stats['zero_count'] / (self.random_sample_stats['non_zero_count'] + self.random_sample_stats['zero_count']) * 100
            print(f"    零值比例: {zero_ratio:.1f}%")

        # 统计站点引导部分
        if hasattr(self, 'station_sample_stats') and self.station_sample_stats['added_count'] > 0:
            print(f"\n  【站点引导采样】（宽松标准）")
            print(f"    新增样本数: {self.station_sample_stats['added_count']:,}")
            print(f"    总样本中占比: {self.station_sample_stats['added_count'] / len(self.meta_index) * 100:.1f}%")

        print(f"{'='*60}")

        # 按年份统计
        if hasattr(self, 'random_sample_stats') and 'samples_per_year' in self.random_sample_stats:
            print(f"\n按年份统计:")
            for year, count in sorted(self.random_sample_stats['samples_per_year'].items()):
                print(f"  {year}年: {count} 个样本")

        # 按日期统计（前10天）
        if hasattr(self, 'random_sample_stats') and self.random_sample_stats['samples_per_date']:
            print(f"\n按日期统计（前10天）:")
            sorted_dates = sorted(self.random_sample_stats['samples_per_date'].items(), key=lambda x: x[1], reverse=True)
            for date, count in sorted_dates[:10]:
                print(f"  {date.strftime('%Y-%m-%d')}: {count} 个样本")
            
    def _load_all_station_pixels(self):
        """加载所有站点CSV文件中的站点位置"""
        print("\n📍 加载站点数据 (用于引导采样)")

        csv_files = [
            "station_swe_data.xlsx",
            "long_comb.csv", 
            "long_comb2.csv",
            "long_comb3.csv",
            "one_record.csv"
        ]

        all_station_pixels = set()

        # 🔥 确保 station_csv_dir 是 Path 对象
        station_dir = Path(self.station_csv_dir) if not isinstance(self.station_csv_dir, Path) else self.station_csv_dir

        for csv_file in csv_files:
            file_path = station_dir / csv_file
            if not file_path.exists():
                print(f"  ⚠ 文件不存在: {csv_file}")
                continue

            print(f"  正在读取: {csv_file}")

            try:
                # 读取文件
                if file_path.suffix in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            df = pd.read_csv(file_path, encoding='gbk')
                        except:
                            df = pd.read_csv(file_path, encoding='latin1')

                # 列名标准化
                column_mapping = {
                    'longtitude': 'longitude', 'lon': 'longitude', 'lng': 'longitude', 'long': 'longitude',
                    'latitude': 'latitude', 'lat': 'latitude',
                }
                df = df.rename(columns=lambda x: column_mapping.get(str(x).strip().lower(), x))

                # 检查必要列
                if 'longitude' not in df.columns or 'latitude' not in df.columns:
                    print(f"    跳过: 缺少经纬度列")
                    continue

                # 转换为像素坐标
                pixels_in_file = 0
                for _, row in df.iterrows():
                    lon, lat = row['longitude'], row['latitude']
                    if pd.isna(lon) or pd.isna(lat):
                        continue

                    try:
                        col, row_idx = ~self.transform * (lon, lat)
                        r, c = int(row_idx), int(col)
                        if 0 <= r < self.H and 0 <= c < self.W:
                            all_station_pixels.add((r, c))
                            pixels_in_file += 1
                    except:
                        continue

                print(f"    添加了 {pixels_in_file} 个站点像元")

            except Exception as e:
                print(f"    读取失败: {e}")
                continue

        # 扩展邻域
        if self.station_neighborhood > 0:
            print(f"\n  🔍 扩展邻域 (半径={self.station_neighborhood})")
            original_count = len(all_station_pixels)
            all_station_pixels = self._expand_neighborhood(all_station_pixels, self.station_neighborhood)
            print(f"      {original_count} → {len(all_station_pixels)} 个像元")

        self.station_pixels = all_station_pixels
        print(f"\n  ✅ 站点引导采样已启用")
        print(f"     总站点像元数: {len(self.station_pixels)}")
        print(f"     每天最多添加: {self.station_samples_per_day} 个站点样本")
        
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
        维度必须与严格版一致：21维

        无效值处理策略：
        - LS: 无效值 -> 0
        - 哨兵1: -11 -> 0
        - SMAP: -11 -> 250, mask=0标识无效
        - 降水累积: -9999 -> 跳过，不累积
        - 产品值: 恒为0（预训练不使用原产品值作为输入）
        """
        point_features = []

        # ============ 1. LS特征 (6个) - 无效值用0填充 ============
        if hasattr(self, 'ls_data_default'):
            for i in range(min(6, self.ls_data_default.shape[0])):
                val = self.ls_data_default[i, r, c]
                if not np.isfinite(val) or val == 0.0:
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
        point_features.append(float(s1_angle) if s1_angle != -11 else 0.0)

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

        # ============ 6. 物理累积特征 (2个) - 🔥 过滤 -9999 无效值 ============
        cum_pr_30d = 0.0
        cum_snow_30d = 0.0

        INVALID_PR = -9999.0
        INVALID_LST = -9999.0

        if "pr" in self.conv_dyn_data and "lst" in self.conv_dyn_data:
            date_idx = self.date_to_index.get(date_dt)
            if date_idx is not None:
                start_idx = max(0, date_idx - 30)
                pr_history = self.conv_dyn_data["pr"][start_idx:date_idx + 1, r, c]
                temp_history = self.conv_dyn_data["lst"][start_idx:date_idx + 1, r, c]

                valid_days = min(len(pr_history), len(temp_history))
                for i in range(valid_days):
                    p = float(pr_history[i])
                    t = float(temp_history[i])

                    # 🔥 关键：过滤 -9999 无效值
                    pr_valid = (p != INVALID_PR) and np.isfinite(p)
                    lst_valid = (t != INVALID_LST) and np.isfinite(t)

                    if pr_valid and lst_valid:
                        cum_pr_30d += p
                        if t < 1.0:  # 温度阈值 1°C
                            cum_snow_30d += p

        point_features.append(cum_pr_30d)
        point_features.append(cum_snow_30d)

        # ============ 7. 原产品值 (1个) - 🔥 修改：恒为0，避免数据泄露 ============
        point_features.append(0.0)  # 预训练时不使用原产品值作为输入

        point_feats_array = np.array(point_features, dtype=np.float32)
        point_feats_array = np.nan_to_num(point_feats_array, nan=0.0)

        # 维度检查
        if len(point_features) != 21:
            print(f"⚠ 宽松版点特征维度错误: {len(point_features)} != 21")
            return None

        # ============ 🔥 微波有效性检查（基于真实数据，不是填充值） ============
        # 哨兵1有效：值不是0（因为无数据时填充0）
        s1_vv_valid = (point_features[6] != 0.0)
        s1_vh_valid = (point_features[7] != 0.0)
        s1_valid = s1_vv_valid or s1_vh_valid

        # SMAP有效：mask == 1（表示有真实数据）
        smap_v_valid = (point_features[13] == 1.0)  # mask_V
        smap_h_valid = (point_features[14] == 1.0)  # mask_H
        smap_valid = smap_v_valid or smap_h_valid

        # 至少有一类微波数据有效才保留样本
        if not (s1_valid or smap_valid):
            return None

        return point_feats_array

    def _get_sentinel1_value_loose(self, date_dt: datetime, r: int, c: int):
        """
        宽松版获取哨兵1值 - 正确识别NODATA=-11，无数据时设为0
        """

        # 🔥 哨兵1的真实NODATA值是 -11
        S1_NODATA = -11.0

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

        # 🔥 S1_VV：-11 表示无数据，设为0
        if 'VV' in data and data['VV'] is not None:
            val = data['VV'][r, c]
            # 只有不是 -11 且不是其他无效值时，才使用实际值
            if val != S1_NODATA and val != self.s1_nodata_value and np.isfinite(val):
                vv = float(val)
            else:
                vv = 0.0  # 无数据时设为0

        # 🔥 S1_VH：-11 表示无数据，设为0
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

        # 🔥 S1_angle：-11 表示无数据，设为0
        if 'angle' in data and data['angle'] is not None:
            val = data['angle'][r, c]
            if val != S1_NODATA and val != self.s1_nodata_value and np.isfinite(val):
                angle = float(val)
            else:
                angle = 0.0  # 无数据时设为0

        return vv, vh, vv_cov, vh_cov, angle

    def _get_smap_value_loose(self, date_dt: datetime, r: int, c: int):
        """
        宽松版获取SMAP值 - 正确识别无效值 -11
        无数据时使用 250 填充（有效亮温的合理均值）
        """

        SMAP_NODATA = -11.0
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

        # 🔥 TBV：只有不是 -11 才算有效
        if 'TBV' in data and data['TBV'] is not None:
            val = data['TBV'][r, c]
            if val != SMAP_NODATA and np.isfinite(val):
                tbv = float(val)
            # else: 保持 DEFAULT_TB，同时 mask 会是 0

        # 🔥 TBH：只有不是 -11 才算有效
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
        当亮温值为 -11 时，mask 也会是 0
        """
        SMAP_NODATA = -11.0

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
                # 只有亮温值不是 -11 时，mask 才有意义
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
            "chelsa_sfxwind": 0.0,
            "lst": -9999.0,
            "rh": -9999.0,
            "pr": -9999.0,
        }
        CLAMDAY_INVALID = -11.0

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
                if np.isfinite(val):
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
                valid_data = self.dem_data[i][np.isfinite(self.dem_data[i])]
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

            if s1_angle != self.s1_nodata_value and np.isfinite(s1_angle) and s1_angle != -11:
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

        # ============ 2.6 降水相关特征 ============
        print(f"\n  降水相关特征:")
        print(f"    顺序: [总降水, 累积降雪]")

        cum_pr_values = []
        cum_snow_values = []

        for idx in sample_indices:
            # 🔥 修复：兼容新旧格式
            item = self.meta_index[idx]
            if len(item) == 4:
                date, r, c, source = item
            else:
                date, r, c = item

            date_idx = self.date_to_index.get(date)

            if date_idx is not None and "pr" in self.conv_dyn_data and "lst" in self.conv_dyn_data:
                start_idx = max(0, date_idx - 30)
                pr_history = self.conv_dyn_data["pr"][start_idx:date_idx + 1, r, c]
                temp_history = self.conv_dyn_data["lst"][start_idx:date_idx + 1, r, c]

                valid_days = min(len(pr_history), len(temp_history))
                cum_pr = 0.0
                cum_snow = 0.0

                for i in range(valid_days):
                    pr_val = pr_history[i]
                    temp_val = temp_history[i]

                    pr_invalid = CONV_INVALID_VALUES.get("pr")
                    pr_ok = (pr_val != pr_invalid) if pr_invalid is not None else True
                    pr_ok = pr_ok and np.isfinite(pr_val)

                    lst_invalid = CONV_INVALID_VALUES.get("lst")
                    temp_ok = (temp_val != lst_invalid) if lst_invalid is not None else True
                    temp_ok = temp_ok and np.isfinite(temp_val)

                    if pr_ok and temp_ok:
                        cum_pr += float(pr_val)
                        if temp_val < 1.0:
                            cum_snow += float(pr_val)

                cum_pr_values.append(cum_pr)
                cum_snow_values.append(cum_snow)

        # 总降水
        if cum_pr_values:
            min_pr = float(np.min(cum_pr_values))
            max_pr = float(np.max(cum_pr_values))
            point_mins.append(min_pr)
            point_maxs.append(max_pr)
            print(f"    过去30天总降水: [{min_pr:.2f}, {max_pr:.2f}] mm (采样{len(cum_pr_values)}个)")
            print(f"    平均总降水: {np.mean(cum_pr_values):.2f} mm")
        else:
            point_mins.append(0.0)
            point_maxs.append(300.0)
            print(f"    ⚠ 过去30天总降水: 无有效数据，使用手动范围[0.0, 300.0] mm")

        # 有效降雪
        if cum_snow_values:
            min_snow = float(np.min(cum_snow_values))
            max_snow = float(np.max(cum_snow_values))
            point_mins.append(min_snow)
            point_maxs.append(max_snow)
            print(f"    过去30天有效降雪: [{min_snow:.2f}, {max_snow:.2f}] mm (采样{len(cum_snow_values)}个)")
            print(f"    平均有效降雪: {np.mean(cum_snow_values):.2f} mm")

            snow_ratios = [s/p if p > 0 else 0 for p, s in zip(cum_pr_values, cum_snow_values)]
            if snow_ratios:
                print(f"    平均降雪率: {np.mean(snow_ratios):.3f}")
        else:
            point_mins.append(0.0)
            point_maxs.append(300.0)
            print(f"    ⚠ 过去30天有效降雪: 无有效数据，使用手动范围[0.0, 300.0] mm")

        # ============ 2.7 原产品值 (FusedSWE) ============
        print(f"\n  原产品值 (FusedSWE):")
        product_values = []

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
                    product_values.append(float(val))

        if product_values:
            min_product = float(np.min(product_values))
            max_product = float(np.max(product_values))
            point_mins.append(min_product)
            point_maxs.append(max_product)
            print(f"    范围: [{min_product:.2f}, {max_product:.2f}] mm (采样{len(product_values)}个)")
            print(f"    均值: {np.mean(product_values):.2f} ± {np.std(product_values):.2f} mm")
        else:
            point_mins.append(0.0)
            point_maxs.append(200.0)
            print(f"    ⚠ 无有效数据，使用默认[0.0, 200.0] mm")

        # ============ 3. 最终维度检查和汇总 ============
        self.point_min = np.array(point_mins, dtype=np.float32)
        self.point_max = np.array(point_maxs, dtype=np.float32)
        self.C_point = len(self.point_min)

        expected = 6 + 5 + 2 + 2 + 2 + 1 + 2 + 1
        print(f"\n【最终点特征维度: {self.C_point}】")
        print(f"  组成: LS(6) + S1(5) + SMAP_TB(2) + SMAP_mask(2) + 经纬度(2) + DOY(1) + 降水(2) + 产品(1) = {expected}")
        print(f"  顺序确认: LS(6) → S1_VV(1) → S1_VH(1) → S1_VV_cov(1) → S1_VH_cov(1) → S1_angle(1) → SMAP_TB(2) → SMAP_mask(2) → 经纬度(2) → DOY(1) → 总降水(1) → 降雪(1) → 产品(1)")

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
            "chelsa_sfxwind": 0.0,
            "lst": -9999.0,
            "rh": -9999.0,
            "pr": -9999.0,
        }
        CLAMDAY_INVALID = -11.0

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

            # 有效性判断：mask != 0 且 val != -11 且 finite
            is_valid = (val != self.smap_nodata_value) and np.isfinite(val)
            if mask_v is not None:
                is_valid = is_valid and (mask_v[r, c] != 0)

            if is_valid:
                tbv_value = float(val)

        # ========== H 极化 ==========
        if 'TBH' in data and data['TBH'] is not None:
            val = data['TBH'][r, c]
            mask_h = data.get('mask_H')

            # 有效性判断：mask != 0 且 val != -11 且 finite
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
        """构建点特征 - 预训练数据（21维，包含哨兵1完整特征 + SMAP mask）"""

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
            if not np.isfinite(val):
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

        # angle - 🔥 过滤 -11 无效值，有效时用原始值，无效时填 0
        if s1_angle != self.s1_nodata_value and np.isfinite(s1_angle) and s1_angle != -11:
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

        # ============ 7. 物理累积特征 (2个) ============
        cum_pr_30d = 0.0
        cum_snow_30d = 0.0

        if "pr" in self.conv_dyn_data and "lst" in self.conv_dyn_data:
            date_idx = self.date_to_index.get(date_dt)
            if date_idx is not None:
                start_idx = max(0, date_idx - 30)
                pr_history = self.conv_dyn_data["pr"][start_idx:date_idx + 1, r, c]
                temp_history = self.conv_dyn_data["lst"][start_idx:date_idx + 1, r, c]

                valid_days = min(len(pr_history), len(temp_history))
                for i in range(valid_days):
                    p, t = pr_history[i], temp_history[i]
                    # 🔥 同时过滤 pr 和 lst 的无效值（-9999）
                    if (np.isfinite(p) and p != -9999 and 
                        np.isfinite(t) and t != -9999):
                        cum_pr_30d += float(p)
                        if t < 1.0:
                            cum_snow_30d += float(p)

        point_features.append(cum_pr_30d)    # 第19维：总降水
        point_features.append(cum_snow_30d)  # 第20维：累积降雪

        # ============ 8. 原产品值 (1个) ============
        point_features.append(0.0)           # 第21维：原产品值（预训练时补0）

        point_feats_array = np.array(point_features, dtype=np.float32)

        # 处理NaN值
        if np.any(np.isnan(point_feats_array)):
            point_feats_array = np.nan_to_num(point_feats_array, nan=0.0)

        # 维度防御检查
        if len(point_features) != 21:
            raise ValueError(f"预训练维度错误: 期望 21，实际 {len(point_features)}")

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
            if source == 'station':
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
            if source == 'station':
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

            # Min-Max标准化
            eps = 1e-6

            conv_t = (conv_t - torch.from_numpy(self.conv_min).view(-1, 1, 1)) / \
                     (torch.from_numpy(self.conv_max + eps).view(-1, 1, 1) - torch.from_numpy(self.conv_min).view(-1, 1, 1))

            point_t = (point_t - torch.from_numpy(self.point_min)) / \
                      (torch.from_numpy(self.point_max + eps) - torch.from_numpy(self.point_min))

            y_t = (y_t - self.label_min) / (self.label_max - self.label_min)

            # 维度检查
            expected_dim = 21
            if point_t.shape[0] != expected_dim:
                print("\n" + "!"*60)
                print(f"【维度报警】发现 {point_t.shape[0]} 维样本！（期望 {expected_dim} 维）")
                print(f"数据来源: {source}")
                print(f"当前样本索引: {idx}")
                print("!"*60 + "\n")
                import sys
                sys.exit(1)

            # 预训练数据第21维设为0
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
