# data_online_smap.py
# -*- coding: utf-8 -*-
"""
在线从 1 km 栅格构建 SMAP 反演训练样本（不再使用 NPZ）
- 动态变量: NDVI, LST, Alb, ET, Pre （2018+2019，）
- 对 2019 年有 SMAP 覆盖的日期构建样本：
    * 时间窗口: 前 364 天 + 当天，共 365 天（T=364）
    * patch: P x P（默认 5x5）
- Dataset 内部直接从 numpy 数组切片，不做 npz 解压
- 初始化时一次性读取所有栅格到内存，计算 min-max，后续训练只做 numpy 切片 + 标准化
- 样本级 NaN/Inf 过滤，只保留“完全没有缺失值”的样本（不做 0 填补）
"""

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import rasterio
from rasterio.windows import Window
from datetime import datetime
from pyproj import Transformer

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ============= 基本配置（根据需要改） =============

REGION = "XINJIANG"
YEAR_TARGET = 2016

# 动态窗口：
LAG_DAYS = 364
T = LAG_DAYS + 1  # 364

P = 5            # patch 大小，奇数
R = P // 2

# ERA5 有效像元阈值 & 每日采样上限
MIN_VALID_PIXELS = 100
SAMPLES_PER_DAY = 2000

# 动态变量顺序（）
VAR_ORDER = ["TB", "LST", "2m-TEMPERATURE", "sfcWind", "Pr"]
DYN_YEARS = [2015, 2016]

# 数据根目录（
FEATURE_ROOT = Path(r"D:\data")
ERA5_ROOT    = Path(r"D:\data\era5_swe") / REGION

# ERA5 XINGJIANG 文件命名模板：例如
# NSIDC-0779_EASE2_G1km_SMAP_SM_DS_20190110_XINJIANG.tif
ERA5_PREFIX = "NSIDC-0779_EASE2_G1km_SMAP_SM_DS_"
SMAP_SUFFIX = "_" + REGION + ".tif"


def dyn_path(var: str, year: int) -> Path:
    """动态多波段栅格路径，例如 NDVI_NAQU_2018_1km_daily_EASE2.tif"""
    return FEATURE_ROOT / REGION / f"{var}_{REGION}_{year}_1km_daily_EASE2.tif"


def load_raster_as_array(path: Path) -> Tuple[np.ndarray, rasterio.Affine, str]:
    """读取多波段栅格为 numpy 数组 (C,H,W)，同时返回 transform 和 CRS"""
    ds = rasterio.open(path)
    arr = ds.read().astype(np.float32)
    transform = ds.transform
    crs = ds.crs.to_string()
    ds.close()
    return arr, transform, crs


# ============= 在线 Dataset =============

class OnlineSMAPPatchDataset(Dataset):
    def __init__(
        self,
        region: str = REGION,
        year_target: int = YEAR_TARGET,
        feature_root: Path = FEATURE_ROOT,
        smap_root: Path = ERA5_ROOT,
        lag_days: int = LAG_DAYS,
        patch_size: int = P,
        min_valid_pixels: int = MIN_VALID_PIXELS,
        samples_per_day: int = SAMPLES_PER_DAY,
    ):
        super().__init__()
        self.region = region
        self.year_target = year_target
        self.feature_root = feature_root
        self.smap_root = smap_root
        self.lag_days = lag_days
        self.patch_size = patch_size
        self.R = patch_size // 2
        self.min_valid_pixels = min_valid_pixels
        self.samples_per_day = samples_per_day

        # 1) 加载动态变量（2018+2019），并过滤掉每年的最后 4 天
        self._load_dynamic_all_years()

        # 2) 加载静态栅格（S2med / SoilGrids / Terrain / LC / Koppen）
        self._load_static_rasters()

        # 3) 根据 SMAP 日产品，构建 meta_index 和 SMAP 网格
        self._build_meta_from_smap()

        # 4) 计算动态 & 空间 patch & 点特征 的全局 min/max（按通道），用于 Min-Max 标准化
        self._compute_minmax()

        print(f"\n[OnlineDataset] 最终样本数: {len(self.meta_index)}")
        print(f"  时间轴长度（过滤后）: {len(self.all_dates_dt)}")
        print(f"  动态窗口 T = {self.T}, patch P = {self.P}")

    # ---------- 1) 动态变量加载与时间轴过滤 ----------

    def _load_dynamic_all_years(self):
        """
        为每个动态变量 var 读取 2018 和 2019 两年的多波段栅格，
        拼成 (T_all, H, W)，同时构建时间轴 all_dates_dt。
        然后去掉 “每年的最后 4 天”，压缩时间轴。
        """
        self.dyn_data: Dict[str, np.ndarray] = {}   # var -> (T_all_filtered, H, W)
        self.dyn_dates: Dict[str, List[datetime]] = {}

        all_dates_dt = None

        for var in VAR_ORDER:
            arr_list = []
            date_list: List[datetime] = []

            for year in DYN_YEARS:
                path = dyn_path(var, year)
                if not path.exists():
                    raise FileNotFoundError(f"[{var}] {year} 文件不存在: {path}")
                print(f"[{var}] 读取 {year}: {path}")
                with rasterio.open(path) as ds:
                    data = ds.read().astype(np.float32)  # (nband,H,W)
                    descs = ds.descriptions

                if data.shape[0] != 365:
                    print(f"  警告：{var} {year} 波段数 != 365, 实际={data.shape[0]}")

                # 每个波段对应一个日期
                for d in descs:
                    if d.startswith(f"{var}-"):
                        date_str = d.replace(f"{var}-", "")
                    else:
                        date_str = d
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    date_list.append(dt)
                arr_list.append(data)

            data_all = np.concatenate(arr_list, axis=0)  # (T_all, H, W)
            dates_all = date_list  # 长度 T_all=2*365=730

            if all_dates_dt is None:
                # 第一次，用 NDVI 作为参考时间轴
                all_dates_dt = dates_all
                self.H = data_all.shape[1]
                self.W = data_all.shape[2]
            else:
                if len(dates_all) != len(all_dates_dt):
                    raise ValueError(f"[{var}] 时间轴长度与 NDVI 不一致")

            # 过滤掉每年的最后 4 天
            valid_idx = []
            for i, d in enumerate(dates_all):
                if d.month == 12 and d.day >= 28:
                    continue
                valid_idx.append(i)

            data_filt = data_all[valid_idx, :, :]
            dates_filt = [dates_all[i] for i in valid_idx]

            self.dyn_data[var] = data_filt      # (T_all_filt, H, W)
            self.dyn_dates[var] = dates_filt
            print(f"[{var}] 过滤后 shape={data_filt.shape}, 天数={len(dates_filt)}")

        # 所有变量使用同一条过滤后的时间轴
        self.all_dates_dt = self.dyn_dates["NDVI"]
        self.T_all = len(self.all_dates_dt)
        self.T = T
        self.P = P

        # 构建日期 -> index 映射
        self.date_to_index: Dict[datetime, int] = {
            d: i for i, d in enumerate(self.all_dates_dt)
        }

        print("\n[TimeAxis] 过滤后时间轴：")
        print("  总天数:", self.T_all)
        print("  起始日期:", self.all_dates_dt[0].strftime("%Y-%m-%d"),
              "结束日期:", self.all_dates_dt[-1].strftime("%Y-%m-%d"))

    # ---------- 2) 静态栅格加载 ----------

    def _load_static_rasters(self):
        # S2 年中值 (6 band)
        s2_path = self.feature_root / self.region / f"S2med_{self.region}_2018_1000m_EASE2.tif"
        soil_path = self.feature_root / self.region / f"SoilGrids_{self.region}_1000m_EASE2.tif"
        terrain_path = self.feature_root / self.region / f"TerrainIndices_{self.region}_1000m_EASE2.tif"
        lc_path = self.feature_root / self.region / f"LC_{self.region}_1000m_EASE2.tif"
        kop_path = self.feature_root / self.region / f"Koppen_{self.region}_1km_EASE2.tif"

        print("\n[Static] 读取静态栅格...")

        self.s2_arr, transform, crs = load_raster_as_array(s2_path)      # (6,H,W)
        self.soil_arr, _, _ = load_raster_as_array(soil_path)            # (7,H,W)
        self.terrain_arr, _, _ = load_raster_as_array(terrain_path)      # (8,H,W)
        self.lc_arr, _, _ = load_raster_as_array(lc_path)                # (1,H,W)
        self.kop_arr, _, _ = load_raster_as_array(kop_path)              # (1,H,W)

        # 用 S2 的 transform / CRS 做经纬度转换
        self.transform = transform
        self.crs_proj = crs
        self.transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        # 一些简单检查
        H, W = self.H, self.W
        for name, arr in [
            ("S2", self.s2_arr),
            ("Soil", self.soil_arr),
            ("Terrain", self.terrain_arr),
            ("LC", self.lc_arr),
            ("Koppen", self.kop_arr),
        ]:
            if arr.shape[1] != H or arr.shape[2] != W:
                raise ValueError(f"[Static] {name} 栅格尺寸 {arr.shape[1:]} 与动态数据 ({H},{W}) 不一致")

        print(f"  栅格尺寸: H={H}, W={W}")
        print("  S2 bands:", self.s2_arr.shape[0])
        print("  Soil bands:", self.soil_arr.shape[0])
        print("  Terrain bands:", self.terrain_arr.shape[0])

    # ---------- 3) SMAP -> meta_index / smap_grids ----------

    def _build_meta_from_smap(self):
        """
        扫描 SMAP_ROOT 下所有 SMAP 日产品：
        - 解析日期 -> 对应 dyn 时间轴索引 d_idx
        - 判断该日有效像元数量是否足够
        - 在该日中随机抽取若干像元作为样本（存到 meta_index 里）
        - 同时将该日 SMAP 网格缓存到 smap_grids[d_idx]
        """
        print("\n[SMAP] 构建 meta_index...")

        self.meta_index: List[Tuple[int, int, int]] = []   # (d_idx, r, c)
        self.smap_grids: Dict[int, np.ndarray] = {}        # d_idx -> (H,W)

        smap_files = sorted(self.smap_root.glob("*.tif"))
        if not smap_files:
            raise FileNotFoundError(f"[SMAP] 在 {self.smap_root} 下没有找到任何 SMAP 栅格")

        for f in smap_files:
            name = f.name
            # 从文件名中提取日期（yyyyMMdd）
            if not (name.startswith(SMAP_PREFIX) and name.endswith(SMAP_SUFFIX)):
                print(f"  [跳过] 文件名不符合模板: {name}")
                continue

            date_str = name[len(SMAP_PREFIX): len(SMAP_PREFIX) + 8]  # "20190110"
            date_dt = datetime.strptime(date_str, "%Y%m%d")

            # 只用目标年份
            if date_dt.year != self.year_target:
                continue

            # 该日期是否在动态时间轴中（被去掉的 12-28~12-31 会不在）
            if date_dt not in self.date_to_index:
                print(f"  [跳过] {date_str} 不在动态时间轴（可能是被过滤掉的最后 4 天）")
                continue

            d_idx = self.date_to_index[date_dt]
            # 确保前面有足够的 lag
            if d_idx < self.lag_days:
                print(f"  [跳过] {date_str} 前面不足 {self.lag_days} 天历史")
                continue

            # 读取 SMAP
            with rasterio.open(f) as ds:
                sm = ds.read(1).astype(np.float32)
                sm_nodata = ds.nodata

            if sm.shape != (self.H, self.W):
                print(f"  [跳过] {date_str} SMAP 尺寸 {sm.shape} 与动态数据不一致 ({self.H},{self.W})")
                continue

            # 有效像元： != nodata 且是有限值
            if sm_nodata is not None:
                valid_mask = (sm != sm_nodata) & np.isfinite(sm)
            else:
                valid_mask = np.isfinite(sm)

            valid_pixels = np.count_nonzero(valid_mask)
            if valid_pixels < self.min_valid_pixels:
                print(f"  [跳过] {date_str} 有效像元 {valid_pixels} < {self.min_valid_pixels}")
                continue

            # 边缘裁剪，保证能够取到完整 patch
            candidate_indices = np.argwhere(valid_mask)
            candidate_indices = [
                (r, c) for (r, c) in candidate_indices
                if (self.R <= r < self.H - self.R) and (self.R <= c < self.W - self.R)
            ]
            if not candidate_indices:
                print(f"  [跳过] {date_str} 边缘裁剪后无可用像元")
                continue

            # 随机下采样
            np.random.shuffle(candidate_indices)
            if self.samples_per_day is not None:
                candidate_indices = candidate_indices[: self.samples_per_day]

            # 记录 meta 和 SMAP 网格
            self.smap_grids[d_idx] = sm  # 整张网格缓存
            for (r, c) in candidate_indices:
                self.meta_index.append((d_idx, int(r), int(c)))

            print(f"  [SMAP] {date_str} -> 样本数 {len(candidate_indices)}")

        print(f"[SMAP] meta_index 总样本数: {len(self.meta_index)}")

    # ---------- 4) 计算 Min-Max ----------

    def _compute_minmax(self):
        """
        动态 / 空间 patch / 点特征 的全局 min / max（按通道）：
        - dyn_seq: 各 var 的所有时间/空间上的 min / max（nanmin/nanmax 忽略 NaN）
        - spatial: [S2(6) + Soil(7) + Terrain(8) + LC(1) + Kop(1) + dyn(5)]
        - point:   [Soil(7) + Terrain(8) + S2(6) + LC(1) + Kop(1) + dyn(5) + time(6)]
        """
        print("\n[Norm] 计算 Min-Max ...")

        # 1) 动态变量
        dyn_min_list = []
        dyn_max_list = []
        for var in VAR_ORDER:
            arr = self.dyn_data[var]  # (T_all_filt,H,W)
            dyn_min_list.append(float(np.nanmin(arr)))
            dyn_max_list.append(float(np.nanmax(arr)))
        dyn_min = np.array(dyn_min_list, dtype=np.float32)
        dyn_max = np.array(dyn_max_list, dtype=np.float32)
        self.C_dyn = len(VAR_ORDER)

        # 2) 静态部分
        s2_min = np.nanmin(self.s2_arr, axis=(1, 2))
        s2_max = np.nanmax(self.s2_arr, axis=(1, 2))

        soil_min = np.nanmin(self.soil_arr, axis=(1, 2))
        soil_max = np.nanmax(self.soil_arr, axis=(1, 2))

        terrain_min = np.nanmin(self.terrain_arr, axis=(1, 2))
        terrain_max = np.nanmax(self.terrain_arr, axis=(1, 2))

        lc_min = np.nanmin(self.lc_arr, axis=(1, 2))
        lc_max = np.nanmax(self.lc_arr, axis=(1, 2))

        kop_min = np.nanmin(self.kop_arr, axis=(1, 2))
        kop_max = np.nanmax(self.kop_arr, axis=(1, 2))

        # spatial: S2(6)+Soil(7)+Terrain(8)+LC(1)+Kop(1)+dyn(5)
        spatial_min = np.concatenate(
            [s2_min, soil_min, terrain_min, lc_min, kop_min, dyn_min], axis=0
        ).astype(np.float32)
        spatial_max = np.concatenate(
            [s2_max, soil_max, terrain_max, lc_max, kop_max, dyn_max], axis=0
        ).astype(np.float32)

        # point: Soil(7)+Terrain(8)+S2(6)+LC(1)+Kop(1)+dyn(5)+time(6)
        time_min = np.full(6, -1.0, dtype=np.float32)
        time_max = np.full(6,  1.0, dtype=np.float32)

        point_min = np.concatenate(
            [soil_min, terrain_min, s2_min, lc_min, kop_min, dyn_min, time_min],
            axis=0
        ).astype(np.float32)
        point_max = np.concatenate(
            [soil_max, terrain_max, s2_max, lc_max, kop_max, dyn_max, time_max],
            axis=0
        ).astype(np.float32)

        # 保存为 torch 张量便于标准化
        eps = 1e-6
        self.eps = eps

        self.min_dyn_t = torch.from_numpy(dyn_min).view(-1, 1, 1, 1)
        self.max_dyn_t = torch.from_numpy(dyn_max + eps).view(-1, 1, 1, 1)

        self.min_spatial_t = torch.from_numpy(spatial_min).view(-1, 1, 1)
        self.max_spatial_t = torch.from_numpy(spatial_max + eps).view(-1, 1, 1)

        self.min_point_t = torch.from_numpy(point_min)
        self.max_point_t = torch.from_numpy(point_max + eps)

        self.C_spatial = spatial_min.shape[0]
        self.C_point = point_min.shape[0]

        print(f"  dyn_min shape: {dyn_min.shape}")
        print(f"  spatial_min shape: {spatial_min.shape}")
        print(f"  point_min shape: {point_min.shape}")

    # ---------- Dataset 内部：构建一个 numpy 样本 + NaN/Inf 检查 ----------

    def _pixel_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        x, y = self.transform * (col + 0.5, row + 0.5)
        lon, lat = self.transformer.transform(x, y)
        return lon, lat

    def _build_sample_numpy(self, d_idx: int, r: int, c: int):
        """
        用 numpy 构建一个样本（不做标准化），并做 NaN/Inf 检查。
        返回 (dyn_cube, spatial_patch, point_feats, y) 或 None（表示无效样本）
        """
        # --- A. 动态时序 cube ---
        dyn_cube = np.zeros((self.C_dyn, T, self.P, self.P), dtype=np.float32)
        start_idx = d_idx - self.lag_days
        end_idx = d_idx  # inclusive

        if start_idx < 0 or end_idx >= self.T_all:
            return None

        for v_i, var in enumerate(VAR_ORDER):
            arr = self.dyn_data[var]  # (T_all_filt,H,W)
            window = arr[start_idx:end_idx + 1,
                         r - self.R: r + self.R + 1,
                         c - self.R: c + self.R + 1]
            if window.shape != (T, self.P, self.P):
                return None
            dyn_cube[v_i] = window

        # --- B. 空间 patch ---
        r0, r1 = r - self.R, r + self.R + 1
        c0, c1 = c - self.R, c + self.R + 1

        s2_patch = self.s2_arr[:, r0:r1, c0:c1]
        soil_patch = self.soil_arr[:, r0:r1, c0:c1]
        terrain_patch = self.terrain_arr[:, r0:r1, c0:c1]
        lc_patch = self.lc_arr[:, r0:r1, c0:c1]
        kop_patch = self.kop_arr[:, r0:r1, c0:c1]
        day_patch = dyn_cube[:, -1, :, :]  # (C_dyn,P,P)

        if (s2_patch.shape[1:] != (self.P, self.P) or
            soil_patch.shape[1:] != (self.P, self.P) or
            terrain_patch.shape[1:] != (self.P, self.P) or
            lc_patch.shape[1:] != (self.P, self.P) or
            kop_patch.shape[1:] != (self.P, self.P)):
            return None

        spatial_patch = np.concatenate(
            [s2_patch, soil_patch, terrain_patch, lc_patch, kop_patch, day_patch],
            axis=0
        ).astype(np.float32)

        # --- C. 点特征 ---
        soil_point = self.soil_arr[:, r, c]
        terrain_point = self.terrain_arr[:, r, c]
        s2_point = self.s2_arr[:, r, c]
        lc_point = self.lc_arr[:, r, c]
        kop_point = self.kop_arr[:, r, c]
        center_dyn = dyn_cube[:, -1, self.R, self.R]  # (C_dyn,)

        date_dt = self.all_dates_dt[d_idx]
        doy = date_dt.timetuple().tm_yday
        doy_sin = np.sin(2 * np.pi * doy / 365.0)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)

        lon, lat = self._pixel_to_lonlat(r, c)
        lon_sin = np.sin(np.deg2rad(lon))
        lon_cos = np.cos(np.deg2rad(lon))
        lat_sin = np.sin(np.deg2rad(lat))
        lat_cos = np.cos(np.deg2rad(lat))

        time_feats = np.array(
            [doy_sin, doy_cos, lon_sin, lon_cos, lat_sin, lat_cos],
            dtype=np.float32
        )

        point_feats = np.concatenate([
            soil_point, terrain_point, s2_point,
            lc_point, kop_point, center_dyn, time_feats
        ]).astype(np.float32)

        # --- D. SMAP 标签 ---
        smap_grid = self.smap_grids[d_idx]
        y = float(smap_grid[r, c])

        # ---- 样本级 NaN/Inf 检查（只保留完全无缺失的样本）----
        if not np.isfinite(dyn_cube).all():
            return None
        if not np.isfinite(spatial_patch).all():
            return None
        if not np.isfinite(point_feats).all():
            return None
        if not np.isfinite(y):
            return None

        return dyn_cube, spatial_patch, point_feats, y

    # ---------- Dataset 接口 ----------

    def __len__(self):
        return len(self.meta_index)

    def __getitem__(self, idx: int):
        """
        返回：
          dyn_seq:      (C_dyn, T, P, P)
          spatial_patch:(C_spatial, P, P)
          point_feats:  (C_point,)
          target:       (1,) 标量

        为了保证只返回“完全没有缺失值”的样本：
        - 如果当前 idx 对应的样本有 NaN/Inf，会向后尝试若干个邻近样本
        - 默认最多重试 10 次，仍无效则抛出 IndexError
        """
        max_retry = 10
        cur_idx = idx

        for _ in range(max_retry):
            d_idx, r, c = self.meta_index[cur_idx]
            sample = self._build_sample_numpy(d_idx, r, c)

            if sample is not None:
                dyn_cube, spatial_patch, point_feats, y = sample

                dyn_t = torch.from_numpy(dyn_cube)
                spatial_t = torch.from_numpy(spatial_patch)
                point_t = torch.from_numpy(point_feats)
                y_t = torch.tensor(y, dtype=torch.float32)

                # Min-Max 标准化
                dyn_t = (dyn_t - self.min_dyn_t) / (self.max_dyn_t - self.min_dyn_t)
                spatial_t = (spatial_t - self.min_spatial_t) / (self.max_spatial_t - self.min_spatial_t)
                point_t = (point_t - self.min_point_t) / (self.max_point_t - self.min_point_t)

                return dyn_t, spatial_t, point_t, y_t

            # 当前样本无效（含 NaN/Inf），尝试下一个
            cur_idx = (cur_idx + 1) % len(self.meta_index)

        # 连续 max_retry 个样本都无效，抛异常方便排查
        raise IndexError(f"在 idx={idx} 附近连续 {max_retry} 个样本均包含缺失值（NaN/Inf），请检查源数据。")


# ============= DataLoader 构建函数（训练脚本直接调用） =============

def build_dataloaders(
    batch_size: int = 32,
    val_ratio: float = 0.2,
    num_workers: int = 0,  # 在线构建版本建议先用 0/1，避免多进程复制大数组
    seed: int = 42,
):
    dataset = OnlineSMAPPatchDataset()

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    print(f"\n[DataLoader] 样本总数: {n_total}, train={n_train}, val={n_val}")

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

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

    # 方便主程序获知维度
    sample_dyn, sample_spatial, sample_point, _ = next(iter(train_loader))
    C_dyn, T_real, P_real, _ = sample_dyn.shape[1:]
    C_spatial = sample_spatial.shape[1]
    C_point = sample_point.shape[1]

    print("\n[DataLoader] 推断出来的维度：")
    print(f"  C_dyn={C_dyn}, T={T_real}, P={P_real}, C_spatial={C_spatial}, C_point={C_point}")

    return train_loader, val_loader, (C_dyn, T_real, P_real, C_spatial, C_point)


if __name__ == "__main__":
    train_loader, val_loader, shapes = build_dataloaders(batch_size=8, val_ratio=0.2, num_workers=0)
    print("shapes:", shapes)
