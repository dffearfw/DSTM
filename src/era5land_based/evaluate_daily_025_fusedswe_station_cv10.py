#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the daily 0.25-degree China FusedSWE product with the exact sample
pool and fold assignment already saved by the Frozen M0 station-CV10 run.

This script does not load or run a neural-network model. It:
1. reads frozen_station_cv10_oof_predictions.csv;
2. restores each sample's original station longitude/latitude;
3. samples XGB_SWE_DAILY_025_YYYYMMDD.tif on the exact observation date;
4. produces a fixed 0-400 mm, 2x5 held-out-fold scatter panel.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRODUCT_NAME = (
    "Daily 0.25° Fused Snow Water Equivalent Product "
    "in China (1980–2020)"
)
PRODUCT_FILE_RE = re.compile(r"^XGB_SWE_DAILY_025_(\d{8})\.tif$", re.IGNORECASE)

AXIS_MIN_MM = 0.0
AXIS_MAX_MM = 400.0
AXIS_TICK_MM = 50.0
EXPECTED_CV_SAMPLES = 7936
EXPECTED_FOLDS = list(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the daily 0.25-degree China FusedSWE product using "
            "the existing balanced station-wise 10-fold OOF sample list."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp"),
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=None,
        help=(
            "M0 frozen_station_cv10_oof_predictions.csv. When omitted, "
            "the newest valid Frozen M0-M6/M0 baseline result is selected."
        ),
    )
    parser.add_argument(
        "--station-csv",
        type=Path,
        default=Path(
            "/root/autodl-tmp/shared_cache/progressive_finetune/"
            "internal_progressive_station.csv"
        ),
    )
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/root/ablation/fusedswe/cn"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--scale-to-mm",
        type=float,
        default=1.0,
        help="Multiplicative conversion from raster value to mm; default=1.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Allow missing product values. By default any missing value "
            "stops the run so all methods retain exactly the same samples."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="gbk")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        match = lower_map.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def normalize_station_id(value: Any) -> str:
    text = str(value).split(",")[0].strip()
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".")[0]
    return text


def normalize_date(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d")


def validate_oof_frame(frame: pd.DataFrame, path: Path) -> None:
    required = {
        "dataset_index",
        "fold",
        "station_id",
        "label_date",
        "target_mm",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"OOF文件缺少列 {missing}: {path}")

    if len(frame) != EXPECTED_CV_SAMPLES:
        raise RuntimeError(
            "OOF样本数不是当前正式7936口径："
            f"expected={EXPECTED_CV_SAMPLES}, actual={len(frame)}, file={path}"
        )

    folds = sorted(
        pd.to_numeric(frame["fold"], errors="raise").astype(int).unique().tolist()
    )
    if folds != EXPECTED_FOLDS:
        raise RuntimeError(f"OOF fold异常: expected={EXPECTED_FOLDS}, actual={folds}")

    dataset_index = pd.to_numeric(frame["dataset_index"], errors="raise").astype(int)
    if dataset_index.duplicated().any():
        raise RuntimeError("OOF文件存在重复dataset_index")


def discover_oof_predictions(root: Path) -> Path:
    patterns = [
        (
            "experiments/frozen_M0_M6_baselines_*/M0/internal_cv10/"
            "frozen_station_cv10_oof_predictions.csv"
        ),
        (
            "experiments/frozen_M0_station_cv10_*/internal_cv10/"
            "frozen_station_cv10_oof_predictions.csv"
        ),
    ]

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    rejected: list[str] = []
    for candidate in candidates:
        try:
            frame = read_csv(candidate)
            validate_oof_frame(frame, candidate)
            print(f"✅ 自动选中最新有效M0 OOF清单: {candidate}")
            return candidate
        except Exception as exc:
            rejected.append(f"{candidate}: {exc}")

    detail = "\n".join(rejected[:10]) if rejected else "没有发现候选文件"
    raise FileNotFoundError(
        "无法自动找到有效的Frozen M0 7936样本OOF文件。"
        "可通过--oof-predictions显式指定。\n"
        f"{detail}"
    )


def attach_station_coordinates(
    predictions: pd.DataFrame,
    station_csv: Path,
) -> pd.DataFrame:
    result = predictions.copy()

    existing_lon = find_column(
        result,
        ["longitude", "original_longitude", "lon", "longtitude"],
    )
    existing_lat = find_column(
        result,
        ["latitude", "original_latitude", "lat"],
    )
    if existing_lon is not None and existing_lat is not None:
        result["longitude"] = pd.to_numeric(result[existing_lon], errors="coerce")
        result["latitude"] = pd.to_numeric(result[existing_lat], errors="coerce")
        if result[["longitude", "latitude"]].notna().all().all():
            print("✅ OOF文件已包含完整原始经纬度")
            return result

    source = read_csv(station_csv)
    station_col = find_column(source, ["station_id", "station", "id"])
    date_col = find_column(source, ["date", "label_date", "feature_date"])
    lon_col = find_column(
        source,
        ["longitude", "original_longitude", "lon", "longtitude"],
    )
    lat_col = find_column(
        source,
        ["latitude", "original_latitude", "lat"],
    )
    target_col = find_column(source, ["swe", "target_mm", "站点SWE_raw"])

    missing_names = [
        name
        for name, column in [
            ("station_id", station_col),
            ("date", date_col),
            ("longitude", lon_col),
            ("latitude", lat_col),
        ]
        if column is None
    ]
    if missing_names:
        raise RuntimeError(
            f"站点CSV缺少坐标匹配列 {missing_names}: "
            f"columns={source.columns.tolist()}"
        )

    source = source.copy()
    source["_station_key"] = source[station_col].map(normalize_station_id)
    source["_date_key"] = source[date_col].map(normalize_date)
    source["_source_order"] = np.arange(len(source), dtype=np.int64)
    source["_longitude"] = pd.to_numeric(source[lon_col], errors="coerce")
    source["_latitude"] = pd.to_numeric(source[lat_col], errors="coerce")
    source["_target"] = (
        pd.to_numeric(source[target_col], errors="coerce")
        if target_col is not None
        else np.nan
    )
    source = source.dropna(subset=["_longitude", "_latitude"])

    result["_station_key"] = result["station_id"].map(normalize_station_id)
    result["_date_key"] = result["label_date"].map(normalize_date)
    result["_target"] = pd.to_numeric(result["target_mm"], errors="coerce")
    if (result["_date_key"] == "").any():
        raise RuntimeError("OOF文件存在无法解析的label_date")

    groups = {
        key: group.sort_values("_source_order")
        for key, group in source.groupby(["_station_key", "_date_key"], sort=False)
    }

    longitudes: list[float] = []
    latitudes: list[float] = []
    unmatched: list[str] = []

    match_rows = result[
        ["_station_key", "_date_key", "_target"]
    ].itertuples(index=False, name=None)
    for station_key, date_key, target_value in match_rows:
        key = (station_key, date_key)
        candidates = groups.get(key)
        if candidates is None or candidates.empty:
            longitudes.append(float("nan"))
            latitudes.append(float("nan"))
            if len(unmatched) < 20:
                unmatched.append(f"station={key[0]}, date={key[1]}")
            continue

        if len(candidates) > 1 and candidates["_target"].notna().any():
            target = float(target_value)
            distance = np.abs(candidates["_target"].to_numpy(dtype=float) - target)
            distance[~np.isfinite(distance)] = np.inf
            selected = candidates.iloc[int(np.argmin(distance))]
        else:
            selected = candidates.iloc[0]

        longitudes.append(float(selected["_longitude"]))
        latitudes.append(float(selected["_latitude"]))

    result["longitude"] = longitudes
    result["latitude"] = latitudes

    missing_coordinate = ~np.isfinite(result["longitude"]) | ~np.isfinite(
        result["latitude"]
    )
    if missing_coordinate.any():
        raise RuntimeError(
            "无法为全部OOF样本恢复原始站点坐标："
            f"missing={int(missing_coordinate.sum())}/{len(result)}；"
            f"示例={unmatched}"
        )

    result = result.drop(columns=["_station_key", "_date_key", "_target"])
    print(f"✅ 已从站点CSV恢复 {len(result):,} 条OOF样本的原始经纬度")
    return result


def index_product_files(product_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    duplicates: list[str] = []

    for path in sorted(product_root.glob("XGB_SWE_DAILY_025_*.tif")):
        match = PRODUCT_FILE_RE.match(path.name)
        if match is None:
            continue
        date_text = datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        if date_text in files:
            duplicates.append(date_text)
        files[date_text] = path

    if duplicates:
        raise RuntimeError(f"同一日期存在重复产品文件: {sorted(set(duplicates))[:20]}")
    if not files:
        raise FileNotFoundError(
            f"没有找到XGB_SWE_DAILY_025_YYYYMMDD.tif: {product_root}"
        )

    print(
        f"✅ 已索引 {len(files):,} 个逐日产品文件: "
        f"{min(files)} 至 {max(files)}"
    )
    return files


def sample_product(
    frame: pd.DataFrame,
    product_files: dict[str, Path],
    scale_to_mm: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.warp import transform as transform_coordinates
    except ImportError as exc:
        raise RuntimeError(
            "缺少rasterio。AutoDL环境请先执行: pip install rasterio"
        ) from exc

    result = frame.copy()
    result["_date_key"] = result["label_date"].map(normalize_date)
    values = np.full(len(result), np.nan, dtype=np.float64)

    missing_dates: dict[str, int] = {}
    outside_count = 0
    nodata_count = 0
    sampled_count = 0
    first_raster_metadata: dict[str, Any] | None = None

    for date_text, index_values in result.groupby("_date_key", sort=True).groups.items():
        positions = np.asarray(list(index_values), dtype=np.int64)
        path = product_files.get(date_text)
        if path is None:
            missing_dates[date_text] = int(len(positions))
            continue

        with rasterio.open(path) as dataset:
            if dataset.count < 1:
                raise RuntimeError(f"产品文件没有波段: {path}")

            if first_raster_metadata is None:
                first_raster_metadata = {
                    "example_file": str(path),
                    "crs": str(dataset.crs) if dataset.crs is not None else None,
                    "width": int(dataset.width),
                    "height": int(dataset.height),
                    "bounds": [float(value) for value in dataset.bounds],
                    "transform": [float(value) for value in dataset.transform[:6]],
                    "nodata": (
                        float(dataset.nodata)
                        if dataset.nodata is not None
                        else None
                    ),
                }

            xs = result.loc[positions, "longitude"].to_numpy(dtype=np.float64)
            ys = result.loc[positions, "latitude"].to_numpy(dtype=np.float64)

            if dataset.crs is not None and dataset.crs != CRS.from_epsg(4326):
                xs_list, ys_list = transform_coordinates(
                    CRS.from_epsg(4326),
                    dataset.crs,
                    xs.tolist(),
                    ys.tolist(),
                )
                xs = np.asarray(xs_list, dtype=np.float64)
                ys = np.asarray(ys_list, dtype=np.float64)

            inside = (
                (xs >= dataset.bounds.left)
                & (xs <= dataset.bounds.right)
                & (ys >= dataset.bounds.bottom)
                & (ys <= dataset.bounds.top)
            )
            outside_count += int(np.sum(~inside))

            inside_positions = positions[inside]
            coordinates = list(zip(xs[inside], ys[inside]))
            sampled = list(
                dataset.sample(coordinates, indexes=1, masked=True)
            )
            for output_position, sample in zip(inside_positions, sampled):
                scalar = sample[0]
                if np.ma.is_masked(scalar):
                    nodata_count += 1
                    continue
                value = float(scalar)
                if not np.isfinite(value):
                    nodata_count += 1
                    continue
                if dataset.nodata is not None and np.isclose(
                    value,
                    float(dataset.nodata),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    nodata_count += 1
                    continue
                values[output_position] = value * scale_to_mm
                sampled_count += 1

    result["daily_025_fusedswe_mm"] = values
    result = result.drop(columns=["_date_key"])

    audit = {
        "n_requested": int(len(result)),
        "n_sampled": int(sampled_count),
        "n_missing": int(np.sum(~np.isfinite(values))),
        "n_missing_date": int(sum(missing_dates.values())),
        "missing_dates": missing_dates,
        "n_outside_raster": int(outside_count),
        "n_nodata_or_nonfinite": int(nodata_count),
        "scale_to_mm": float(scale_to_mm),
        "first_raster_metadata": first_raster_metadata,
    }
    return result, audit


def clean_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def compute_metrics(target: Any, prediction: Any) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]
    if target.size == 0:
        raise RuntimeError("没有有效目标/产品值")

    error = prediction - target
    target_std = float(np.std(target))
    prediction_std = float(np.std(prediction))
    if (
        target.size > 1
        and target_std > 1e-12
        and prediction_std > 1e-12
    ):
        r = float(np.corrcoef(target, prediction)[0, 1])
    else:
        r = float("nan")

    centered_target = target - np.mean(target)
    denominator = float(np.sum(centered_target ** 2))
    if denominator > 1e-12:
        slope = float(
            np.sum(centered_target * (prediction - np.mean(prediction)))
            / denominator
        )
        intercept = float(np.mean(prediction) - slope * np.mean(target))
    else:
        slope = float("nan")
        intercept = float("nan")

    return {
        "n_samples": int(target.size),
        "r": clean_float(r),
        "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "mae_mm": float(np.mean(np.abs(error))),
        "bias_mm": float(np.mean(error)),
        "slope": clean_float(slope),
        "intercept_mm": clean_float(intercept),
        "std_ratio": (
            clean_float(prediction_std / target_std)
            if target_std > 1e-12
            else None
        ),
        "target_mean_mm": float(np.mean(target)),
        "prediction_mean_mm": float(np.mean(prediction)),
        "target_std_mm": target_std,
        "prediction_std_mm": prediction_std,
        "prediction_min_mm": float(np.min(prediction)),
        "prediction_max_mm": float(np.max(prediction)),
    }


def build_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        fold_frame = frame[frame["fold"] == fold]
        metrics = compute_metrics(
            fold_frame["target_mm"],
            fold_frame["daily_025_fusedswe_mm"],
        )
        metrics.update(
            {
                "fold": fold,
                "method": PRODUCT_NAME,
                "n_total_fold_samples": int(len(fold_frame)),
                "n_missing_product": int(
                    fold_frame["daily_025_fusedswe_mm"].isna().sum()
                ),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def draw_fold_axis(
    axis,
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, Any],
    fold: int,
) -> None:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]

    axis.scatter(target, prediction, s=10, alpha=0.38)
    axis.plot(
        [AXIS_MIN_MM, AXIS_MAX_MM],
        [AXIS_MIN_MM, AXIS_MAX_MM],
        "--",
        linewidth=1.2,
        label="1:1",
    )

    if len(target) >= 2 and np.std(target) > 0:
        slope, intercept = np.polyfit(target, prediction, 1)
        xx = np.array([AXIS_MIN_MM, AXIS_MAX_MM], dtype=np.float64)
        axis.plot(
            xx,
            intercept + slope * xx,
            linewidth=1.4,
            color="red",
            label="Fit",
        )

    ticks = np.arange(
        AXIS_MIN_MM,
        AXIS_MAX_MM + AXIS_TICK_MM,
        AXIS_TICK_MM,
    )
    axis.set_xlim(AXIS_MIN_MM, AXIS_MAX_MM)
    axis.set_ylim(AXIS_MIN_MM, AXIS_MAX_MM)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_title(f"Fold {fold}", fontsize=10, fontweight="bold")
    axis.grid(alpha=0.22)
    axis.text(
        0.03,
        0.97,
        (
            f"N={metrics['n_samples']}\n"
            f"R={metrics['r']:.3f}\n"
            f"RMSE={metrics['rmse_mm']:.2f}\n"
            f"MAE={metrics['mae_mm']:.2f}\n"
            f"Bias={metrics['bias_mm']:.2f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
    )


def plot_fold_panel(
    frame: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(19, 7.8),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    for fold in EXPECTED_FOLDS:
        fold_frame = frame[frame["fold"] == fold]
        metrics = (
            fold_metrics.loc[fold_metrics["fold"] == fold]
            .iloc[0]
            .to_dict()
        )
        draw_fold_axis(
            axes[fold - 1],
            fold_frame["target_mm"].to_numpy(),
            fold_frame["daily_025_fusedswe_mm"].to_numpy(),
            metrics,
            fold,
        )

    for row in range(2):
        axes[row * 5].set_ylabel("Product SWE (mm)")
    for column in range(5):
        axes[5 + column].set_xlabel("Station SWE (mm)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
        )
    figure.text(
        0.995,
        0.012,
        "All x/y axes fixed at 0–400 mm",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="dimgray",
    )
    figure.suptitle(
        f"{PRODUCT_NAME}: balanced station-wise 10-fold held-out scatter",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    args.station_csv = args.station_csv.expanduser().resolve()
    args.product_root = args.product_root.expanduser().resolve()

    if not args.station_csv.is_file():
        raise FileNotFoundError(args.station_csv)
    if not args.product_root.is_dir():
        raise FileNotFoundError(args.product_root)
    if not np.isfinite(args.scale_to_mm) or args.scale_to_mm <= 0:
        raise ValueError("--scale-to-mm必须是有限正数")

    if args.oof_predictions is None:
        oof_path = discover_oof_predictions(args.root)
    else:
        oof_path = args.oof_predictions.expanduser().resolve()
        if not oof_path.is_file():
            raise FileNotFoundError(oof_path)

    predictions = read_csv(oof_path).reset_index(drop=True)
    validate_oof_frame(predictions, oof_path)
    predictions["fold"] = pd.to_numeric(
        predictions["fold"],
        errors="raise",
    ).astype(int)

    if args.output_dir is None:
        output_dir = oof_path.parent / "daily_025_fusedswe_baseline"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(PRODUCT_NAME)
    print("确定性平衡站点级10折产品基线")
    print("=" * 100)
    print(f"OOF样本/折清单: {oof_path}")
    print(f"站点CSV:         {args.station_csv}")
    print(f"产品目录:        {args.product_root}")
    print(f"输出目录:        {output_dir}")
    print(f"产品值转mm系数:  {args.scale_to_mm}")
    print("坐标范围:        所有fold固定为0–400 mm")
    print("=" * 100)

    predictions = attach_station_coordinates(predictions, args.station_csv)
    product_files = index_product_files(args.product_root)
    predictions, sampling_audit = sample_product(
        predictions,
        product_files,
        args.scale_to_mm,
    )

    missing_count = int(
        predictions["daily_025_fusedswe_mm"].isna().sum()
    )
    if missing_count and not args.allow_missing:
        audit_path = output_dir / "sampling_audit_failed.json"
        audit_path.write_text(
            json.dumps(sampling_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            "产品取值存在缺失，为保证与ERA5-Land/M0完全相同的样本口径，"
            f"默认拒绝继续：missing={missing_count}/{len(predictions)}。"
            f"审计={audit_path}"
        )

    predictions.to_csv(
        output_dir / "daily_025_fusedswe_station_cv10_oof_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics = build_fold_metrics(predictions)
    fold_metrics.to_csv(
        output_dir / "daily_025_fusedswe_station_cv10_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled_metrics = compute_metrics(
        predictions["target_mm"],
        predictions["daily_025_fusedswe_mm"],
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "product_name": PRODUCT_NAME,
        "protocol": {
            "sample_pool": (
                "exact rows from the existing Frozen M0 7936-sample OOF file"
            ),
            "fold_assignment": (
                "reused unchanged from the existing balanced station-wise "
                "10-fold OOF file"
            ),
            "n_total_samples": int(len(predictions)),
            "n_valid_product_samples": int(
                predictions["daily_025_fusedswe_mm"].notna().sum()
            ),
            "n_splits": 10,
            "scatter_axis_mm": [AXIS_MIN_MM, AXIS_MAX_MM],
            "zero_product_values_retained": True,
            "missing_values_allowed": bool(args.allow_missing),
        },
        "files": {
            "oof_predictions": str(oof_path),
            "station_csv": str(args.station_csv),
            "product_root": str(args.product_root),
        },
        "sampling_audit": sampling_audit,
        "pooled_metrics": pooled_metrics,
        "fold_metrics_file": (
            "daily_025_fusedswe_station_cv10_fold_metrics.csv"
        ),
        "panel_file": (
            "daily_025_fusedswe_station_cv10_fold_scatter_panel.png"
        ),
    }
    (output_dir / "daily_025_fusedswe_station_cv10_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    panel_path = (
        output_dir
        / "daily_025_fusedswe_station_cv10_fold_scatter_panel.png"
    )
    plot_fold_panel(predictions, fold_metrics, panel_path)

    print()
    print("=" * 100)
    print("✅ 0.25° FusedSWE站点10折产品基线完成")
    print(
        f"Pooled: N={pooled_metrics['n_samples']}, "
        f"R={pooled_metrics['r']:.4f}, "
        f"RMSE={pooled_metrics['rmse_mm']:.2f} mm, "
        f"MAE={pooled_metrics['mae_mm']:.2f} mm, "
        f"Bias={pooled_metrics['bias_mm']:.2f} mm"
    )
    print(f"10折图: {panel_path}")
    print(f"逐折指标: {output_dir / 'daily_025_fusedswe_station_cv10_fold_metrics.csv'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
