#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build one immutable normalization config for Stage 0-4.

Reference population:
    valid Stage 0 station-date manifest + fixed 152000 incremental manifest.

Method:
    1. Estimate per-channel p01/p99 from a deterministic representative subset.
    2. Scan every reference sample's actual input patch once and calculate
       clipped mean/std with streaming sums.
    3. Keep bounded/mask/time/location point features unchanged.
    4. Fix target scaling to [0, 400) mm; target statistics are not learned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

# Set these before importing the dataset because its sampling contract reads env.
os.environ.setdefault("PRETRAIN_SAMPLES_PER_DAY", "0")
os.environ.setdefault("USE_TARGET_QUOTA_SAMPLING", "0")
os.environ.setdefault("USE_QUOTA_SHORTAGE_SUPPLEMENT", "0")
os.environ.setdefault("STRICT_TARGET_QUOTA", "0")
os.environ.setdefault("PRECOMPUTE_ALL_SAMPLES", "0")

from data_online_era5_swe import SWEDataset, CONV_VARS  # noqa: E402


POINT_FEATURE_NAMES = [
    "ls_band_1", "ls_band_2", "ls_band_3", "ls_band_4", "ls_band_5", "ls_band_6",
    "s1_vv", "s1_vh", "s1_vv_coverage", "s1_vh_coverage", "s1_angle",
    "smap_tbv", "smap_tbh", "smap_mask_v", "smap_mask_h",
    "longitude_norm", "latitude_norm", "doy_norm",
]

# Only real continuous measurements are z-scored. Coverage/masks and already
# bounded coordinate/time variables keep their original semantics.
POINT_TRANSFORM = [
    "zscore", "zscore", "zscore", "zscore", "zscore", "zscore",
    "zscore", "zscore", "identity", "identity", "zscore",
    "zscore", "zscore", "identity", "identity",
    "identity", "identity", "identity",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "row", "col"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["row"] = pd.to_numeric(out["row"], errors="raise").astype(np.int64)
    out["col"] = pd.to_numeric(out["col"], errors="raise").astype(np.int64)
    out["source"] = source
    return out[["date", "row", "col", "source"]]


def point_valid_mask(point: np.ndarray) -> np.ndarray:
    """Validity mask for point channels before normalization."""
    valid = np.ones(point.shape[0], dtype=bool)
    # Loose Stage 0 loader uses zero for missing Landsat bands.
    valid[0:6] = point[0:6] != 0.0
    # S1 values/angle are valid only when coverage is positive.
    valid[6] = point[8] > 0.0
    valid[7] = point[9] > 0.0
    valid[10] = (point[8] > 0.0) or (point[9] > 0.0)
    # SMAP brightness temperatures are valid only when corresponding mask=1.
    valid[11] = point[13] > 0.0
    valid[12] = point[14] > 0.0
    # Coverage, masks, coordinates and DOY remain valid bounded variables.
    return valid


def extract_features(dataset: SWEDataset, row) -> tuple[np.ndarray, np.ndarray]:
    dt = row.date.to_pydatetime()
    r, c = int(row.row), int(row.col)
    if row.source == "stage0_station":
        conv = dataset._build_spatial_features_station(dt, r, c)
        point = dataset._build_point_features_station(dt, r, c)
    else:
        conv = dataset._build_spatial_features(dt, r, c)
        point = dataset._build_point_features(dt, r, c)
    if conv is None or point is None:
        raise RuntimeError(
            f"清单样本重新读取失败: source={row.source}, date={dt:%Y-%m-%d}, row={r}, col={c}"
        )
    conv = np.asarray(conv, dtype=np.float32)
    point = np.asarray(point, dtype=np.float32)
    if conv.shape != (dataset.C_conv, dataset.patch_size, dataset.patch_size):
        raise RuntimeError(f"卷积输入shape异常: {conv.shape}")
    if point.shape != (dataset.C_point,):
        raise RuntimeError(f"点输入shape异常: {point.shape}")
    if not np.all(np.isfinite(conv)) or not np.all(np.isfinite(point)):
        raise RuntimeError("清单样本包含NaN/Inf")
    return conv, point


def choose_quantile_indices(union: pd.DataFrame, max_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stage0_idx = np.flatnonzero(union["source"].to_numpy() == "stage0_station")
    random_idx = np.flatnonzero(union["source"].to_numpy() == "incremental_random")
    if len(union) <= max_samples:
        return np.arange(len(union), dtype=np.int64)
    # Always include every Stage 0 sample; fill the remaining capacity from the
    # nationally distributed fixed random pool.
    keep_stage0 = stage0_idx
    remaining = max(0, max_samples - len(keep_stage0))
    if remaining >= len(random_idx):
        keep_random = random_idx
    else:
        keep_random = np.sort(rng.choice(random_idx, size=remaining, replace=False))
    return np.concatenate([keep_stage0, keep_random]).astype(np.int64)


def validate_union(dataset: SWEDataset, union: pd.DataFrame, label_min: float, label_max: float) -> None:
    duplicates = union.duplicated(["date", "row", "col"], keep=False)
    if duplicates.any():
        examples = union.loc[duplicates].head(10).to_dict("records")
        raise RuntimeError(f"Stage0与随机池存在重复时空样本，例如: {examples}")

    for row in union.itertuples(index=False):
        dt = row.date.to_pydatetime()
        r, c = int(row.row), int(row.col)
        if dt not in dataset.label_data or dt not in dataset.date_to_index:
            raise RuntimeError(f"清单日期未加载: {dt:%Y-%m-%d}")
        if not (0 <= r < dataset.H and 0 <= c < dataset.W):
            raise RuntimeError(f"清单像元越界: row={r}, col={c}")
        arr, nodata = dataset.label_data[dt]
        y = float(arr[r, c])
        if not np.isfinite(y) or (nodata is not None and y == nodata):
            raise RuntimeError(f"清单标签无效: {dt:%Y-%m-%d}, row={r}, col={c}")
        if not (label_min <= y < label_max):
            raise RuntimeError(
                f"清单标签超出固定范围[{label_min}, {label_max}): y={y}, "
                f"date={dt:%Y-%m-%d}, row={r}, col={c}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-guide-file", required=True)
    ap.add_argument("--stage0-manifest", required=True)
    ap.add_argument("--incremental-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shared-cache-dir", default="/root/autodl-tmp/shared_cache")
    ap.add_argument("--years", nargs="+", type=int, default=[2015, 2016, 2017, 2018])
    ap.add_argument("--patch-size", type=int, default=5)
    ap.add_argument("--min-valid-pixels", type=int, default=100)
    ap.add_argument("--clamday-threshold", type=float, default=0.5)
    ap.add_argument("--label-min-mm", type=float, default=0.0)
    ap.add_argument("--label-max-mm", type=float, default=400.0)
    ap.add_argument("--quantile-samples", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--external-station-glob", default=None)
    ap.add_argument("--external-station-exclusion-radius", type=int, default=0)
    ap.add_argument("--external-station-strict", action="store_true")
    ap.add_argument("--external-station-report-path", default=None)
    ap.add_argument("--force-reload", action="store_true")
    args = ap.parse_args()

    stage0_path = Path(args.stage0_manifest).expanduser().resolve()
    incremental_path = Path(args.incremental_manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    station_path = Path(args.station_guide_file).expanduser().resolve()
    for path in [stage0_path, incremental_path, station_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 78)
    print("构建 Stage 0-4 统一归一化配置")
    print(f"Stage 0清单: {stage0_path}")
    print(f"随机池清单: {incremental_path}")
    print(f"输出: {output_path}")
    print("=" * 78)

    dataset = SWEDataset(
        year_target=args.years,
        patch_size=args.patch_size,
        min_valid_pixels=args.min_valid_pixels,
        samples_per_day=0,
        clamday_threshold=args.clamday_threshold,
        cache_dir=None,
        force_reload=args.force_reload,
        sampling_mode="station",
        station_guide_file=station_path,
        station_neighborhood=0,
        station_samples_per_day=-1,
        station_filter_zero_target=False,
        station_sampling_unit="records",
        station_record_dedup="grid_date",
        external_station_glob=args.external_station_glob,
        external_station_exclusion_radius=args.external_station_exclusion_radius,
        external_station_strict=args.external_station_strict,
        external_station_report_path=args.external_station_report_path,
        normalization_mode="skip",
        fixed_label_min_mm=args.label_min_mm,
        fixed_label_max_mm=args.label_max_mm,
    )

    stage0 = load_manifest(stage0_path, "stage0_station")
    incremental = load_manifest(incremental_path, "incremental_random")
    union = pd.concat([stage0, incremental], ignore_index=True)

    # The stage0 file must describe exactly the valid Stage 0 set produced by
    # the current data code and input files.
    dataset_stage0_keys = {
        (pd.Timestamp(dt), int(r), int(c)) for dt, r, c, *_ in dataset.meta_index
    }
    file_stage0_keys = {
        (pd.Timestamp(x.date), int(x.row), int(x.col)) for x in stage0.itertuples(index=False)
    }
    if dataset_stage0_keys != file_stage0_keys:
        missing = list(dataset_stage0_keys - file_stage0_keys)[:5]
        extra = list(file_stage0_keys - dataset_stage0_keys)[:5]
        raise RuntimeError(
            "Stage 0清单与当前代码重新构建的有效样本不一致。"
            f" 文件缺少示例={missing}; 文件多出示例={extra}"
        )

    if len(incremental) != 152000:
        raise RuntimeError(f"固定随机池必须为152000条，当前={len(incremental)}")

    validate_union(dataset, union, args.label_min_mm, args.label_max_mm)
    print(f"Stage 0有效样本: {len(stage0):,}")
    print(f"固定随机样本: {len(incremental):,}")
    print(f"统一参考样本: {len(union):,}")

    conv_names = list(CONV_VARS) + ["clamday"] + [
        f"dem_band_{i + 1}" for i in range(len(dataset.dem_data))
    ]
    if len(conv_names) != dataset.C_conv:
        raise RuntimeError("卷积特征名称数量与C_conv不一致")
    if len(POINT_FEATURE_NAMES) != dataset.C_point:
        raise RuntimeError("点特征名称数量与C_point不一致")

    q_idx = choose_quantile_indices(union, args.quantile_samples, args.seed)
    q_n = len(q_idx)
    p2 = dataset.patch_size * dataset.patch_size
    conv_store = np.empty((q_n, dataset.C_conv, p2), dtype=np.float32)
    point_store = np.empty((q_n, dataset.C_point), dtype=np.float32)

    print(f"\n第一遍：用 {q_n:,} 个代表性样本估计p01/p99")
    for j, idx in enumerate(tqdm(q_idx, desc="quantile pass")):
        row = union.iloc[int(idx)]
        conv, point = extract_features(dataset, row)
        conv_store[j] = conv.reshape(dataset.C_conv, p2)
        point_store[j] = point

    conv_low = np.percentile(conv_store, 1.0, axis=(0, 2)).astype(np.float32)
    conv_high = np.percentile(conv_store, 99.0, axis=(0, 2)).astype(np.float32)
    bad = conv_high <= conv_low
    conv_high[bad] = conv_low[bad] + 1.0

    point_low = np.zeros(dataset.C_point, dtype=np.float32)
    point_high = np.ones(dataset.C_point, dtype=np.float32)
    point_mean = np.zeros(dataset.C_point, dtype=np.float32)
    point_std = np.ones(dataset.C_point, dtype=np.float32)
    point_valid_q = np.vstack([point_valid_mask(x) for x in point_store])
    for i, transform in enumerate(POINT_TRANSFORM):
        if transform == "identity":
            vals = point_store[:, i]
            point_low[i] = float(np.min(vals))
            point_high[i] = float(np.max(vals))
            if point_high[i] <= point_low[i]:
                point_high[i] = point_low[i] + 1.0
            continue
        vals = point_store[point_valid_q[:, i], i]
        if vals.size == 0:
            raise RuntimeError(f"点特征 {POINT_FEATURE_NAMES[i]} 没有有效值")
        point_low[i] = float(np.percentile(vals, 1.0))
        point_high[i] = float(np.percentile(vals, 99.0))
        if point_high[i] <= point_low[i]:
            point_high[i] = point_low[i] + 1.0

    del conv_store, point_store, point_valid_q

    conv_sum = np.zeros(dataset.C_conv, dtype=np.float64)
    conv_sumsq = np.zeros(dataset.C_conv, dtype=np.float64)
    conv_count = np.zeros(dataset.C_conv, dtype=np.int64)
    conv_low_count = np.zeros(dataset.C_conv, dtype=np.int64)
    conv_high_count = np.zeros(dataset.C_conv, dtype=np.int64)

    point_sum = np.zeros(dataset.C_point, dtype=np.float64)
    point_sumsq = np.zeros(dataset.C_point, dtype=np.float64)
    point_count = np.zeros(dataset.C_point, dtype=np.int64)
    point_low_count = np.zeros(dataset.C_point, dtype=np.int64)
    point_high_count = np.zeros(dataset.C_point, dtype=np.int64)

    print("\n第二遍：扫描全部实际Patch，计算截断后的mean/std")
    for row in tqdm(union.itertuples(index=False), total=len(union), desc="full stats pass"):
        conv, point = extract_features(dataset, row)
        flat = conv.reshape(dataset.C_conv, -1)
        conv_low_count += np.sum(flat < conv_low[:, None], axis=1)
        conv_high_count += np.sum(flat > conv_high[:, None], axis=1)
        clipped = np.clip(flat, conv_low[:, None], conv_high[:, None]).astype(np.float64)
        conv_sum += clipped.sum(axis=1)
        conv_sumsq += np.square(clipped).sum(axis=1)
        conv_count += clipped.shape[1]

        valid = point_valid_mask(point)
        for i, transform in enumerate(POINT_TRANSFORM):
            if transform != "zscore" or not valid[i]:
                continue
            val = float(point[i])
            point_low_count[i] += int(val < point_low[i])
            point_high_count[i] += int(val > point_high[i])
            val = float(np.clip(val, point_low[i], point_high[i]))
            point_sum[i] += val
            point_sumsq[i] += val * val
            point_count[i] += 1

    if np.any(conv_count == 0):
        raise RuntimeError("存在没有统计值的卷积通道")
    conv_mean = conv_sum / conv_count
    conv_var = np.maximum(conv_sumsq / conv_count - np.square(conv_mean), 0.0)
    conv_std = np.sqrt(conv_var)
    conv_std[conv_std < 1e-6] = 1.0

    for i, transform in enumerate(POINT_TRANSFORM):
        if transform == "identity":
            point_mean[i] = 0.0
            point_std[i] = 1.0
            continue
        if point_count[i] == 0:
            raise RuntimeError(f"点特征 {POINT_FEATURE_NAMES[i]} 没有流式统计值")
        mean = point_sum[i] / point_count[i]
        var = max(point_sumsq[i] / point_count[i] - mean * mean, 0.0)
        point_mean[i] = mean
        point_std[i] = max(np.sqrt(var), 1e-6)

    conv_total_values = conv_count.astype(np.float64)
    point_total_values = np.maximum(point_count, 1).astype(np.float64)
    payload = {
        "version": 2,
        "created_at": datetime.now().isoformat(),
        "method": "clip_then_zscore",
        "description": "Stage0有效站点-日期样本与固定152000随机样本共同确定，Stage0-4永久复用",
        "reference_samples": {
            "stage0_count": int(len(stage0)),
            "incremental_count": int(len(incremental)),
            "total_count": int(len(union)),
            "quantile_sample_count": int(q_n),
        },
        "reference_manifests": {
            "stage0": {"path": str(stage0_path), "sha256": sha256_file(stage0_path)},
            "incremental": {"path": str(incremental_path), "sha256": sha256_file(incremental_path)},
        },
        "years": [int(x) for x in args.years],
        "patch_size": int(dataset.patch_size),
        "C_conv": int(dataset.C_conv),
        "C_point": int(dataset.C_point),
        "conv_feature_names": conv_names,
        "point_feature_names": POINT_FEATURE_NAMES,
        "point_transform": POINT_TRANSFORM,
        "conv_clip_low": conv_low.tolist(),
        "conv_clip_high": conv_high.tolist(),
        "conv_mean": conv_mean.astype(np.float32).tolist(),
        "conv_std": conv_std.astype(np.float32).tolist(),
        "point_clip_low": point_low.tolist(),
        "point_clip_high": point_high.tolist(),
        "point_mean": point_mean.tolist(),
        "point_std": point_std.tolist(),
        "label_min": float(args.label_min_mm),
        "label_max": float(args.label_max_mm),
        "target_method": "fixed_minmax",
        "diagnostics": {
            "conv_below_p01_ratio": (conv_low_count / conv_total_values).tolist(),
            "conv_above_p99_ratio": (conv_high_count / conv_total_values).tolist(),
            "point_below_p01_ratio": (point_low_count / point_total_values).tolist(),
            "point_above_p99_ratio": (point_high_count / point_total_values).tolist(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(output_path)

    print("\n" + "=" * 78)
    print(f"✅ 统一归一化配置已生成: {output_path}")
    print(f"SHA256: {sha256_file(output_path)}")
    print("标签固定范围: [0, 400) mm")
    print("Stage 0-4必须使用 normalization_mode=load")
    print("=" * 78)


if __name__ == "__main__":
    main()
