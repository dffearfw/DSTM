# data_station_online.py
# -*- coding: utf-8 -*-
"""
第二阶段：站点 Ms 在线构建样本 Dataset + 按站点划分 DataLoader

- 使用与第一阶段相同的 1km 特征栅格：
    * 动态变量: NDVI, LST, Alb, ET, Pre （2018+2019，去掉两年的最后4天）
    * 静态: S2med, SoilGrids, Terrain, LC, Koppen
- 使用站点日尺度 Ms 表（已包含 lon, lat, row, col）构建样本：
    * 时间窗口: 前 360 天 + 当天，共 361 天 (T=361)
    * patch 大小: 5x5
- 目标：只保留“完全没有缺失值”的样本
- 按 station 进行 train/val/test 的“留站点”划分
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import rasterio
from datetime import datetime
from pyproj import Transformer

import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ============= 基本配置（根据需要修改） =============

REGION = "NAQU"
YEAR_TARGET = 2019

# 动态窗口：前 360 天 + 当天，共 361
LAG_DAYS = 360
T = LAG_DAYS + 1  # 361

P = 5                  # patch 大小，奇数
R = P // 2

# 动态变量顺序（必须和你的动态栅格命名一致）
VAR_ORDER = ["NDVI", "LST", "Alb", "ET", "Pre"]
DYN_YEARS = [2018, 2019]

# 数据根目录（1km 栅格）
FEATURE_ROOT = Path(r"D:\多尺度SM\临时")

# 站点 Ms+行列号表
STATION_CSV = Path(
    r"H:\Soil_water\SM_是测站点\下载\青藏高原中部土壤温湿度多尺度观测网数据集（2010-2021）"
    r"\SM_NQ_202109\NAQU_SM_05cm_daily_2019_grid.csv"
)

# Ms 列名（请根据你的实际列名修改）
SM_COL = "sm"   # 比如你 CSV 里是 'sm_05cm' 就改成 'sm_05cm'


def dyn_path(var: str, year: int) -> Path:
    """动态多波段栅格路径，例如 NDVI_NAQU_2018_1km_daily_EASE2.tif"""
    return FEATURE_ROOT / REGION / f"{var}_{REGION}_{year}_1km_daily_EASE2.tif"


def load_raster_as_array(path: Path):
    """读取多波段栅格为 numpy 数组 (C,H,W)，同时返回 transform 和 CRS"""
    ds = rasterio.open(path)
    arr = ds.read().astype(np.float32)
    transform = ds.transform
    crs = ds.crs.to_string()
    ds.close()
    return arr, transform, crs


# ============= 站点 Dataset =============

class StationOnlineDataset(Dataset):
    """
    使用站点实测 Ms 构建样本：

    meta_index: List[dict]，每个元素：
        {
            "d_idx": int,      # 在动态时间轴中的 index
            "r": int, "c": int,
            "y": float,        # 站点实测 Ms
            "station": str,
            "date": datetime
        }

    __getitem__ 返回：
        dyn_seq:       (C_dyn, T, P, P)
        spatial_patch: (C_spatial, P, P)
        point_feats:   (C_point,)
        target:        (标量 Ms)
    """

    def __init__(
        self,
        station_csv: Path = STATION_CSV,
        region: str = REGION,
        year_target: int = YEAR_TARGET,
        feature_root: Path = FEATURE_ROOT,
        lag_days: int = LAG_DAYS,
        patch_size: int = P,
    ):
        super().__init__()
        self.region = region
        self.year_target = year_target
        self.feature_root = feature_root
        self.station_csv = Path(station_csv)
        self.lag_days = lag_days
        self.patch_size = patch_size
        self.R = patch_size // 2

        # 1) 加载动态变量（2018+2019），并过滤掉每年的最后 4 天
        self._load_dynamic_all_years()

        # 2) 加载静态栅格
        self._load_static_rasters()

        # 3) 从站点 Ms CSV 构建 meta_index
        self._build_meta_from_station_csv()

        # 4) 计算 Min-Max 以便做归一化
        self._compute_minmax()

        print(f"\n[StationDataset] 最终样本数: {len(self.meta_index)}")
        print(f"  覆盖站点数: {len(self.station_set)}")
        print(f"  时间轴长度（过滤后）: {len(self.all_dates_dt)}")
        print(f"  动态窗口 T = {self.T}, patch P = {self.P}")

    # ---------- 1) 动态变量加载与时间轴过滤 ----------

    def _load_dynamic_all_years(self):
        """
        为每个动态变量 var 读取 2018 和 2019 两年的多波段栅格，
        拼成 (T_all, H, W)，构建时间轴 all_dates_dt，
        然后去掉 “每年的最后 4 天”（12-28~12-31），压缩时间轴。
        """
        self.dyn_data: Dict[str, np.ndarray] = {}
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
                    # 预期: "NDVI-2018-01-01"
                    if d.startswith(f"{var}-"):
                        date_str = d.replace(f"{var}-", "")
                    else:
                        date_str = d
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    date_list.append(dt)
                arr_list.append(data)

            data_all = np.concatenate(arr_list, axis=0)  # (T_all,H,W)
            dates_all = date_list  # 长度 2*365=730

            if all_dates_dt is None:
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

            self.dyn_data[var] = data_filt
            self.dyn_dates[var] = dates_filt
            print(f"[{var}] 过滤后 shape={data_filt.shape}, 天数={len(dates_filt)}")

        self.all_dates_dt = self.dyn_dates["NDVI"]
        self.T_all = len(self.all_dates_dt)
        self.T = T
        self.P = P

        # 日期 -> index 映射
        self.date_to_index: Dict[datetime, int] = {
            d: i for i, d in enumerate(self.all_dates_dt)
        }

        print("\n[TimeAxis] 过滤后时间轴：")
        print("  总天数:", self.T_all)
        print("  起始日期:", self.all_dates_dt[0].strftime("%Y-%m-%d"),
              "结束日期:", self.all_dates_dt[-1].strftime("%Y-%m-%d"))

    # ---------- 2) 静态栅格加载 ----------

    def _load_static_rasters(self):
        s2_path = self.feature_root / self.region / f"S2med_{self.region}_2018_1000m_EASE2.tif"
        soil_path = self.feature_root / self.region / f"SoilGrids_{self.region}_1000m_EASE2.tif"
        terrain_path = self.feature_root / self.region / f"TerrainIndices_{self.region}_1000m_EASE2.tif"
        lc_path = self.feature_root / self.region / f"LC_{self.region}_1000m_EASE2.tif"
        kop_path = self.feature_root / self.region / f"Koppen_{self.region}_1km_EASE2.tif"

        print("\n[Static] 读取静态栅格...")

        self.s2_arr, transform, crs = load_raster_as_array(s2_path)
        self.soil_arr, _, _ = load_raster_as_array(soil_path)
        self.terrain_arr, _, _ = load_raster_as_array(terrain_path)
        self.lc_arr, _, _ = load_raster_as_array(lc_path)
        self.kop_arr, _, _ = load_raster_as_array(kop_path)

        self.transform = transform
        self.crs_proj = crs
        self.transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

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

    # ---------- 3) 站点 CSV -> meta_index ----------

    def _build_meta_from_station_csv(self):
        """
        从 NAQU_SM_05cm_daily_2019_grid.csv 构建 meta_index:
        每一行 -> 一个样本入口 (d_idx, r, c, y, station, date)
        """
        print("\n[Station] 从 CSV 构建 meta_index ...")
        if not self.station_csv.exists():
            raise FileNotFoundError(f"站点 Ms CSV 不存在: {self.station_csv}")

        df = pd.read_csv(self.station_csv)

        # 尝试解析日期
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            # 如果没有 date 列，你可以根据 year/month/day 自己改
            raise ValueError("CSV 中没有 'date' 列，请检查或自行修改解析方式。")

        # 只保留目标年份
        df = df[df["date"].dt.year == self.year_target].copy()

        # 检查必需列
        for col in ["station", "row", "col", SM_COL]:
            if col not in df.columns:
                raise ValueError(f"CSV 中缺少必要列: {col}")

        # 剔除异常 Ms（例如 <= -90）
        df = df[df[SM_COL] > -90].copy()

        meta_index: List[Dict[str, Any]] = []
        station_set = set()

        for _, row in df.iterrows():
            dt = row["date"].to_pydatetime()
            if dt not in self.date_to_index:
                # 比如被过滤掉的 12-28~12-31
                continue

            d_idx = self.date_to_index[dt]
            if d_idx < self.lag_days:
                # 前面历史不足 360 天
                continue

            r = int(row["row"])
            c = int(row["col"])
            if not (0 <= r < self.H and 0 <= c < self.W):
                continue

            y = float(row[SM_COL])
            if not np.isfinite(y):
                continue

            station_id = str(row["station"])
            station_set.add(station_id)

            meta_index.append({
                "d_idx": d_idx,
                "r": r,
                "c": c,
                "y": y,
                "station": station_id,
                "date": dt,
            })

        self.meta_index = meta_index
        self.station_set = station_set

        print(f"  有效样本数: {len(self.meta_index)}")
        print(f"  有效站点数: {len(self.station_set)}")

    # ---------- 4) 计算 Min-Max ----------

    def _compute_minmax(self):
        """
        动态 / 空间 patch / 点特征 的全局 min / max（按通道）：
        - dyn_seq: 各 var 的所有时间/空间上的 min/max
        - spatial: [S2(6) + Soil(7) + Terrain(8) + LC(1) + Kop(1) + dyn(5)]
        - point:   [Soil(7)+Terrain(8)+S2(6)+LC(1)+Kop(1)+dyn(5)+time(6)]
        """
        print("\n[Norm] 计算 Min-Max ...")

        dyn_min_list = []
        dyn_max_list = []
        for var in VAR_ORDER:
            arr = self.dyn_data[var]
            dyn_min_list.append(float(np.nanmin(arr)))
            dyn_max_list.append(float(np.nanmax(arr)))
        dyn_min = np.array(dyn_min_list, dtype=np.float32)
        dyn_max = np.array(dyn_max_list, dtype=np.float32)
        self.C_dyn = len(VAR_ORDER)

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

    # ---------- Dataset 接口 ----------

    def __len__(self):
        return len(self.meta_index)

    def _pixel_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        x, y = self.transform * (col + 0.5, row + 0.5)
        lon, lat = self.transformer.transform(x, y)
        return lon, lat

    def _build_sample_numpy(self, meta: Dict[str, Any]):
        """
        用 numpy 构建一个样本（不做标准化），并做 NaN/Inf 检查。
        返回 (dyn_cube, spatial_patch, point_feats, y) 或 None（无效样本）
        """
        d_idx = meta["d_idx"]
        r = meta["r"]
        c = meta["c"]
        y = meta["y"]

        dyn_cube = np.zeros((self.C_dyn, T, self.P, self.P), dtype=np.float32)
        start_idx = d_idx - self.lag_days
        end_idx = d_idx

        if start_idx < 0 or end_idx >= self.T_all:
            return None

        for v_i, var in enumerate(VAR_ORDER):
            arr = self.dyn_data[var]
            window = arr[start_idx:end_idx + 1,
                         r - self.R: r + self.R + 1,
                         c - self.R: c + self.R + 1]
            if window.shape != (T, self.P, self.P):
                return None
            dyn_cube[v_i] = window

        # 空间 patch
        r0, r1 = r - self.R, r + self.R + 1
        c0, c1 = c - self.R, c + self.R + 1

        s2_patch = self.s2_arr[:, r0:r1, c0:c1]
        soil_patch = self.soil_arr[:, r0:r1, c0:c1]
        terrain_patch = self.terrain_arr[:, r0:r1, c0:c1]
        lc_patch = self.lc_arr[:, r0:r1, c0:c1]
        kop_patch = self.kop_arr[:, r0:r1, c0:c1]
        day_patch = dyn_cube[:, -1, :, :]

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

        # 点特征
        soil_point = self.soil_arr[:, r, c]
        terrain_point = self.terrain_arr[:, r, c]
        s2_point = self.s2_arr[:, r, c]
        lc_point = self.lc_arr[:, r, c]
        kop_point = self.kop_arr[:, r, c]
        center_dyn = dyn_cube[:, -1, self.R, self.R]

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

        point_feats = np.concatenate(
            [soil_point, terrain_point, s2_point,
             lc_point, kop_point, center_dyn, time_feats],
            axis=0
        ).astype(np.float32)

        # Ms 目标
        if not np.isfinite(y):
            return None

        # NaN/Inf 检查
        if (not np.isfinite(dyn_cube).all() or
            not np.isfinite(spatial_patch).all() or
            not np.isfinite(point_feats).all()):
            return None

        return dyn_cube, spatial_patch, point_feats, y

    def __getitem__(self, idx: int):
        """
        只使用“完全无缺失值”的样本：
        - 当前 idx 无效时，随机换样本尝试，最多 50 次
        """
        max_retry = 50
        n = len(self.meta_index)
        cur_idx = idx

        for _ in range(max_retry):
            meta = self.meta_index[cur_idx]
            sample = self._build_sample_numpy(meta)

            if sample is not None:
                dyn_cube, spatial_patch, point_feats, y = sample

                dyn_t = torch.from_numpy(dyn_cube)
                spatial_t = torch.from_numpy(spatial_patch)
                point_t = torch.from_numpy(point_feats)
                y_t = torch.tensor(y, dtype=torch.float32)

                dyn_t = (dyn_t - self.min_dyn_t) / (self.max_dyn_t - self.min_dyn_t)
                spatial_t = (spatial_t - self.min_spatial_t) / (self.max_spatial_t - self.min_spatial_t)
                point_t = (point_t - self.min_point_t) / (self.max_point_t - self.min_point_t)

                return dyn_t, spatial_t, point_t, y_t

            cur_idx = int(np.random.randint(0, n))

        raise RuntimeError(
            f"连续尝试 {max_retry} 个随机样本仍未找到无缺失值样本，"
            f"请检查站点/栅格数据质量。"
        )


# ============= 按站点划分的 DataLoader 构建函数 =============

def build_station_dataloaders(
    station_csv: Path = STATION_CSV,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    - 先构建一个包含“所有站点”样本的 StationOnlineDataset
    - 然后按 station 划分为 train/val/test 三部分
    - 最终返回 train_loader, val_loader, test_loader, (shape_info), splits
    """
    dataset = StationOnlineDataset(station_csv=station_csv)

    # 站点列表
    stations = sorted(list(dataset.station_set))
    n_st = len(stations)

    if n_st < 3:
        raise ValueError(f"站点数太少 ({n_st})，无法划分 train/val/test。")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_st)
    stations_shuffled = [stations[i] for i in perm]

    n_test = max(1, int(round(n_st * test_ratio)))
    n_val = max(1, int(round((n_st - n_test) * val_ratio)))
    n_train = n_st - n_test - n_val
    if n_train <= 0:
        n_train = 1
        # 简单保证，不纠结极端情况

    test_stations = stations_shuffled[:n_test]
    val_stations = stations_shuffled[n_test:n_test + n_val]
    train_stations = stations_shuffled[n_test + n_val:]

    print("\n[StationSplit] 站点划分：")
    print(f"  总站点数: {n_st}")
    print(f"  train 站点数: {len(train_stations)}")
    print(f"  val   站点数: {len(val_stations)}")
    print(f"  test  站点数: {len(test_stations)}")

    # 为每个样本指定所属集合
    train_indices = []
    val_indices = []
    test_indices = []

    train_set_station = set(train_stations)
    val_set_station = set(val_stations)
    test_set_station = set(test_stations)

    for i, meta in enumerate(dataset.meta_index):
        st = meta["station"]
        if st in train_set_station:
            train_indices.append(i)
        elif st in val_set_station:
            val_indices.append(i)
        elif st in test_set_station:
            test_indices.append(i)

    print(f"[StationSplit] 样本数：train={len(train_indices)}, "
          f"val={len(val_indices)}, test={len(test_indices)}")

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

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
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 推断形状
    sample_dyn, sample_spatial, sample_point, _ = next(iter(train_loader))
    C_dyn, T_real, P_real, _ = sample_dyn.shape[1:]
    C_spatial = sample_spatial.shape[1]
    C_point = sample_point.shape[1]

    print("\n[StationDataLoader] 推断出来的维度：")
    print(f"  C_dyn={C_dyn}, T={T_real}, P={P_real}, "
          f"C_spatial={C_spatial}, C_point={C_point}")

    shape_info = (C_dyn, T_real, P_real, C_spatial, C_point)
    splits = {
        "train_stations": train_stations,
        "val_stations": val_stations,
        "test_stations": test_stations,
    }
    return train_loader, val_loader, test_loader, shape_info, splits


if __name__ == "__main__":
    # 简单测试
    train_loader, val_loader, test_loader, shapes, splits = build_station_dataloaders(
        station_csv=STATION_CSV,
        batch_size=8,
        val_ratio=0.2,
        test_ratio=0.2,
        num_workers=0,
        seed=42,
    )
    print("shapes:", shapes)
    print("splits:", {k: len(v) for k, v in splits.items()})
