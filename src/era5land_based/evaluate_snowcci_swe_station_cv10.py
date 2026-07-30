#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate ESA Snow CCI SWE CRDP v4.0 on the exact station samples and fold
assignment saved by the Frozen M0 station-wise CV10 baseline.

The Snow CCI SWE product intentionally masks mountains, glaciers/permanent ice,
water, and land pixels where no retrieval was attempted. Those values remain
missing; they are never replaced by zero or by a neighbouring date/pixel.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRODUCT_NAME = (
    "ESA Snow Climate Change Initiative (Snow_cci): Snow Water Equivalent "
    "(SWE) level 3C daily global climate research data package (CRDP) "
    "(1979–2023), version 4.0"
)
PLOT_NAME = "ESA Snow CCI SWE CRDP v4.0"
PRODUCT_FILE_RE = re.compile(
    r"^(\d{8})-ESACCI-L3C_SNOW-SWE-.*-fv4[.]0[.]nc$",
    re.IGNORECASE,
)

EXPECTED_CV_SAMPLES = 7936
EXPECTED_FOLDS = list(range(1, 11))
AXIS_MIN_MM = 0.0
AXIS_MAX_MM = 400.0
AXIS_TICK_MM = 50.0
VALID_SWE_MIN_MM = 0.0
VALID_SWE_MAX_MM = 500.0

LATITUDE_CANDIDATES = ("lat", "latitude", "y")
LONGITUDE_CANDIDATES = ("lon", "longitude", "x")
SWE_CANDIDATES = (
    "SWE",
    "swe",
    "snow_water_equivalent",
    "lwe_thickness_of_surface_snow_amount",
)
UNCERTAINTY_TOKENS = ("std", "uncert", "variance", "var")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ESA Snow CCI SWE CRDP v4.0 using the existing balanced "
            "station-wise Frozen M0 CV10 OOF sample list."
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
            "Frozen M0 frozen_station_cv10_oof_predictions.csv. If omitted, "
            "the newest valid 7936-sample M0 baseline is selected."
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
        default=Path("/root/ablation/snowcci"),
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
        help=(
            "Additional multiplicative conversion after the NetCDF "
            "scale_factor/add_offset and unit conversion; default=1."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail if any OOF row has no valid Snow CCI retrieval. The default "
            "continues because Snow CCI deliberately masks mountains, water, "
            "glaciers, and no-retrieval land pixels."
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
    if re.fullmatch(r"[+-]?\d+[.]0+", text):
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
        raise RuntimeError(
            f"OOF fold异常: expected={EXPECTED_FOLDS}, actual={folds}"
        )

    dataset_index = pd.to_numeric(
        frame["dataset_index"],
        errors="raise",
    ).astype(int)
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
        result["longitude"] = pd.to_numeric(
            result[existing_lon],
            errors="coerce",
        )
        result["latitude"] = pd.to_numeric(
            result[existing_lat],
            errors="coerce",
        )
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
        for key, group in source.groupby(
            ["_station_key", "_date_key"],
            sort=False,
        )
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
            distance = np.abs(
                candidates["_target"].to_numpy(dtype=float) - target
            )
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
    for path in sorted(product_root.glob("*.nc")):
        match = PRODUCT_FILE_RE.match(path.name)
        if match is None:
            continue
        date_text = datetime.strptime(
            match.group(1),
            "%Y%m%d",
        ).strftime("%Y-%m-%d")
        if date_text in files:
            duplicates.append(date_text)
        files[date_text] = path

    if duplicates:
        raise RuntimeError(
            f"同一日期存在重复Snow CCI文件: "
            f"{sorted(set(duplicates))[:20]}"
        )
    if not files:
        raise FileNotFoundError(
            "没有找到YYYYMMDD-ESACCI-L3C_SNOW-SWE-*-fv4.0.nc："
            f"{product_root}"
        )

    print(
        f"✅ 已索引 {len(files):,} 个Snow CCI逐日文件: "
        f"{min(files)} 至 {max(files)}"
    )
    return files


def choose_variable_name(
    dataset,
    candidates: tuple[str, ...],
) -> str | None:
    lower_map = {name.lower(): name for name in dataset.variables}
    for candidate in candidates:
        if candidate in dataset.variables:
            return candidate
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    return None


def detect_swe_variable(dataset, lat_name: str, lon_name: str) -> str:
    exact = choose_variable_name(dataset, SWE_CANDIDATES)
    if exact is not None:
        return exact

    lat_dim = dataset.variables[lat_name].dimensions[0]
    lon_dim = dataset.variables[lon_name].dimensions[0]
    candidates: list[str] = []
    for name, variable in dataset.variables.items():
        lowered = name.lower()
        if any(token in lowered for token in UNCERTAINTY_TOKENS):
            continue
        if lat_dim not in variable.dimensions or lon_dim not in variable.dimensions:
            continue
        if not np.issubdtype(variable.dtype, np.number):
            continue
        standard_name = str(
            getattr(variable, "standard_name", "")
        ).lower()
        long_name = str(getattr(variable, "long_name", "")).lower()
        if (
            "snow_water_equivalent" in standard_name
            or "snow water equivalent" in long_name
            or lowered == "swe"
        ):
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        "无法唯一识别Snow CCI SWE变量。"
        f"候选={candidates}, variables={list(dataset.variables)}"
    )


def nearest_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < 2 or not np.all(np.isfinite(axis)):
        raise RuntimeError("NetCDF经纬度坐标不是有效的一维规则坐标")

    descending = axis[0] > axis[-1]
    work_axis = axis[::-1] if descending else axis
    insertion = np.searchsorted(work_axis, values, side="left")
    insertion = np.clip(insertion, 1, len(work_axis) - 1)
    left = insertion - 1
    right = insertion
    choose_right = np.abs(work_axis[right] - values) < np.abs(
        work_axis[left] - values
    )
    selected = np.where(choose_right, right, left)
    if descending:
        selected = len(axis) - 1 - selected
    return selected.astype(np.int64)


def raw_missing_values(variable) -> list[float]:
    values: list[float] = []
    for name in ("_FillValue", "missing_value"):
        if not hasattr(variable, name):
            continue
        for value in np.asarray(getattr(variable, name)).reshape(-1):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return values


def units_to_mm_factor(units: str) -> float:
    normalized = units.lower().replace(" ", "").replace("**", "^")
    if normalized in {"m", "meter", "metre", "meters", "metres"}:
        return 1000.0
    # Snow CCI uses mm. kg m-2 is numerically equivalent to mm water.
    return 1.0


def classify_raw_value(
    raw_value: Any,
    missing_values: list[float],
    scale_factor: float,
    add_offset: float,
    unit_factor: float,
    additional_scale: float,
) -> tuple[float, str]:
    try:
        raw = float(raw_value)
    except (TypeError, ValueError):
        return float("nan"), "nonfinite"
    if not np.isfinite(raw):
        return float("nan"), "nonfinite"
    if any(np.isclose(raw, value, rtol=0.0, atol=1e-12) for value in missing_values):
        return float("nan"), "fill_or_missing_value"

    integer_code = int(round(raw))
    if np.isclose(raw, integer_code, rtol=0.0, atol=1e-12):
        special_status = {
            -1: "masked_no_retrieval",
            -10: "masked_water",
            -20: "masked_mountain",
            -30: "masked_glacier_or_permanent_ice",
        }.get(integer_code)
        if special_status is not None:
            return float("nan"), special_status

    value_mm = (
        (raw * scale_factor + add_offset)
        * unit_factor
        * additional_scale
    )
    if not np.isfinite(value_mm):
        return float("nan"), "nonfinite"
    if value_mm < VALID_SWE_MIN_MM or value_mm > VALID_SWE_MAX_MM:
        return float("nan"), "outside_valid_swe_code_range"
    return float(value_mm), "valid"


def sample_snowcci(
    frame: pd.DataFrame,
    product_files: dict[str, Path],
    additional_scale: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "缺少netCDF4。AutoDL环境请先执行: pip install netCDF4"
        ) from exc

    result = frame.copy()
    result["_date_key"] = result["label_date"].map(normalize_date)
    values = np.full(len(result), np.nan, dtype=np.float64)
    statuses = np.full(len(result), "not_sampled", dtype=object)

    status_counts: Counter[str] = Counter()
    missing_dates: dict[str, int] = {}
    first_file_metadata: dict[str, Any] | None = None
    date_groups = list(result.groupby("_date_key", sort=True).groups.items())

    for date_number, (date_text, index_values) in enumerate(date_groups, start=1):
        positions = np.asarray(list(index_values), dtype=np.int64)
        path = product_files.get(date_text)
        if path is None:
            statuses[positions] = "missing_date_file"
            status_counts["missing_date_file"] += int(len(positions))
            missing_dates[date_text] = int(len(positions))
            continue

        with Dataset(path, mode="r") as dataset:
            lat_name = choose_variable_name(dataset, LATITUDE_CANDIDATES)
            lon_name = choose_variable_name(dataset, LONGITUDE_CANDIDATES)
            if lat_name is None or lon_name is None:
                raise RuntimeError(
                    f"NetCDF缺少一维经纬度变量: {path}; "
                    f"variables={list(dataset.variables)}"
                )

            lat_variable = dataset.variables[lat_name]
            lon_variable = dataset.variables[lon_name]
            latitudes = np.asarray(lat_variable[:], dtype=np.float64).squeeze()
            longitudes = np.asarray(
                lon_variable[:],
                dtype=np.float64,
            ).squeeze()
            if latitudes.ndim != 1 or longitudes.ndim != 1:
                raise RuntimeError(
                    f"仅支持Snow CCI一维规则经纬网格: {path}; "
                    f"lat_shape={latitudes.shape}, lon_shape={longitudes.shape}"
                )

            swe_name = detect_swe_variable(dataset, lat_name, lon_name)
            swe_variable = dataset.variables[swe_name]
            lat_dim = lat_variable.dimensions[0]
            lon_dim = lon_variable.dimensions[0]
            if lat_dim not in swe_variable.dimensions:
                raise RuntimeError(
                    f"SWE变量不含纬度维 {lat_dim}: {swe_variable.dimensions}"
                )
            if lon_dim not in swe_variable.dimensions:
                raise RuntimeError(
                    f"SWE变量不含经度维 {lon_dim}: {swe_variable.dimensions}"
                )

            input_lon = result.loc[
                positions,
                "longitude",
            ].to_numpy(dtype=np.float64)
            input_lat = result.loc[
                positions,
                "latitude",
            ].to_numpy(dtype=np.float64)
            if np.nanmin(longitudes) >= 0.0 and np.nanmax(longitudes) > 180.0:
                input_lon = np.mod(input_lon, 360.0)
            elif np.nanmax(longitudes) <= 180.0:
                input_lon = ((input_lon + 180.0) % 360.0) - 180.0

            inside = (
                (input_lon >= np.nanmin(longitudes))
                & (input_lon <= np.nanmax(longitudes))
                & (input_lat >= np.nanmin(latitudes))
                & (input_lat <= np.nanmax(latitudes))
            )
            outside_positions = positions[~inside]
            statuses[outside_positions] = "outside_grid"
            status_counts["outside_grid"] += int(len(outside_positions))

            inside_positions = positions[inside]
            if len(inside_positions):
                lon_indices = nearest_indices(longitudes, input_lon[inside])
                lat_indices = nearest_indices(latitudes, input_lat[inside])

                missing_values = raw_missing_values(swe_variable)
                scale_factor = float(
                    np.asarray(
                        getattr(swe_variable, "scale_factor", 1.0)
                    ).reshape(-1)[0]
                )
                add_offset = float(
                    np.asarray(
                        getattr(swe_variable, "add_offset", 0.0)
                    ).reshape(-1)[0]
                )
                units = str(getattr(swe_variable, "units", "mm"))
                unit_factor = units_to_mm_factor(units)
                if hasattr(swe_variable, "set_auto_maskandscale"):
                    swe_variable.set_auto_maskandscale(False)

                for output_position, lat_index, lon_index in zip(
                    inside_positions,
                    lat_indices,
                    lon_indices,
                ):
                    selection: list[int] = []
                    for dimension in swe_variable.dimensions:
                        if dimension == lat_dim:
                            selection.append(int(lat_index))
                        elif dimension == lon_dim:
                            selection.append(int(lon_index))
                        elif len(dataset.dimensions[dimension]) == 1:
                            selection.append(0)
                        else:
                            raise RuntimeError(
                                "SWE变量存在无法自动选择的非单例维度："
                                f"dimension={dimension}, "
                                f"size={len(dataset.dimensions[dimension])}, "
                                f"file={path}"
                            )

                    raw_value = swe_variable[tuple(selection)]
                    value_mm, status = classify_raw_value(
                        raw_value,
                        missing_values,
                        scale_factor,
                        add_offset,
                        unit_factor,
                        additional_scale,
                    )
                    values[output_position] = value_mm
                    statuses[output_position] = status
                    status_counts[status] += 1

            if first_file_metadata is None:
                first_file_metadata = {
                    "example_file": str(path),
                    "variables": list(dataset.variables),
                    "latitude_variable": lat_name,
                    "longitude_variable": lon_name,
                    "swe_variable": swe_name,
                    "swe_dimensions": list(swe_variable.dimensions),
                    "swe_dtype": str(swe_variable.dtype),
                    "swe_units": str(getattr(swe_variable, "units", "")),
                    "swe_scale_factor": float(
                        np.asarray(
                            getattr(swe_variable, "scale_factor", 1.0)
                        ).reshape(-1)[0]
                    ),
                    "swe_add_offset": float(
                        np.asarray(
                            getattr(swe_variable, "add_offset", 0.0)
                        ).reshape(-1)[0]
                    ),
                    "swe_fill_or_missing_values": raw_missing_values(
                        swe_variable
                    ),
                    "latitude_size": int(latitudes.size),
                    "longitude_size": int(longitudes.size),
                    "latitude_range": [
                        float(np.nanmin(latitudes)),
                        float(np.nanmax(latitudes)),
                    ],
                    "longitude_range": [
                        float(np.nanmin(longitudes)),
                        float(np.nanmax(longitudes)),
                    ],
                }

        if date_number % 100 == 0 or date_number == len(date_groups):
            print(
                f"  Snow CCI采样进度: "
                f"{date_number}/{len(date_groups)} 个观测日期"
            )

    result["snowcci_swe_mm"] = values
    result["snowcci_sampling_status"] = statuses
    result = result.drop(columns=["_date_key"])

    final_counts = Counter(statuses.tolist())
    n_valid = int(np.sum(np.isfinite(values)))
    audit = {
        "n_requested": int(len(result)),
        "n_valid": n_valid,
        "n_missing": int(len(result) - n_valid),
        "valid_ratio": float(n_valid / len(result)),
        "status_counts": dict(sorted(final_counts.items())),
        "n_missing_date": int(sum(missing_dates.values())),
        "missing_dates": missing_dates,
        "additional_scale_to_mm": float(additional_scale),
        "valid_swe_range_mm": [
            VALID_SWE_MIN_MM,
            VALID_SWE_MAX_MM,
        ],
        "sampling_method": (
            "nearest 0.1-degree Snow CCI grid-cell centre; no spatial or "
            "temporal infilling"
        ),
        "first_file_metadata": first_file_metadata,
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
        raise RuntimeError("没有有效目标/Snow CCI产品值")

    error = prediction - target
    target_std = float(np.std(target))
    prediction_std = float(np.std(prediction))
    if target.size > 1 and target_std > 1e-12 and prediction_std > 1e-12:
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
        nse = float(
            1.0
            - np.sum(error ** 2)
            / np.sum((target - np.mean(target)) ** 2)
        )
    else:
        slope = float("nan")
        intercept = float("nan")
        nse = float("nan")

    return {
        "n_samples": int(target.size),
        "r": clean_float(r),
        "nse": clean_float(nse),
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
            fold_frame["snowcci_swe_mm"],
        )
        metrics.update(
            {
                "fold": fold,
                "method": PLOT_NAME,
                "n_total_fold_samples": int(len(fold_frame)),
                "n_missing_product": int(
                    fold_frame["snowcci_swe_mm"].isna().sum()
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
            fold_frame["snowcci_swe_mm"].to_numpy(),
            metrics,
            fold,
        )

    for row in range(2):
        axes[row * 5].set_ylabel("Snow CCI SWE (mm)")
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
        "All x/y axes fixed at 0–400 mm; masked Snow CCI pixels omitted",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="dimgray",
    )
    figure.suptitle(
        f"{PLOT_NAME}: balanced station-wise 10-fold held-out scatter",
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
        output_dir = oof_path.parent / "snowcci_swe_baseline"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print(PRODUCT_NAME)
    print("确定性平衡站点级10折产品基线")
    print("=" * 108)
    print(f"OOF样本/折清单: {oof_path}")
    print(f"站点CSV:         {args.station_csv}")
    print(f"产品目录:        {args.product_root}")
    print(f"输出目录:        {output_dir}")
    print(f"额外转mm系数:    {args.scale_to_mm}")
    print("空间采样:        最近0.1°网格中心，不做邻域填补")
    print("掩膜处理:        保留缺失，不填0；有效样本参与指标和绘图")
    print("坐标范围:        所有fold固定为0–400 mm")
    print("=" * 108)

    predictions = attach_station_coordinates(predictions, args.station_csv)
    product_files = index_product_files(args.product_root)
    predictions, sampling_audit = sample_snowcci(
        predictions,
        product_files,
        args.scale_to_mm,
    )

    audit_path = output_dir / "snowcci_swe_sampling_audit.json"
    audit_path.write_text(
        json.dumps(sampling_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    missing_count = int(predictions["snowcci_swe_mm"].isna().sum())
    if missing_count:
        print(
            f"⚠ Snow CCI缺失/掩膜: {missing_count:,}/{len(predictions):,} "
            f"({100.0 * missing_count / len(predictions):.2f}%)"
        )
        print(f"  分类: {sampling_audit['status_counts']}")
        if args.require_complete:
            raise RuntimeError(
                "Snow CCI存在缺失/掩膜值且启用了--require-complete。"
                f"审计={audit_path}"
            )

    values_path = (
        output_dir / "snowcci_swe_station_cv10_oof_values.csv"
    )
    predictions.to_csv(values_path, index=False, encoding="utf-8-sig")

    fold_metrics = build_fold_metrics(predictions)
    fold_metrics_path = (
        output_dir / "snowcci_swe_station_cv10_fold_metrics.csv"
    )
    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    pooled_metrics = compute_metrics(
        predictions["target_mm"],
        predictions["snowcci_swe_mm"],
    )

    plot_path = (
        output_dir / "snowcci_swe_station_cv10_fold_scatter_panel.png"
    )
    plot_fold_panel(predictions, fold_metrics, plot_path)

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
                predictions["snowcci_swe_mm"].notna().sum()
            ),
            "n_missing_or_masked_product_samples": missing_count,
            "n_splits": 10,
            "scatter_axis_mm": [AXIS_MIN_MM, AXIS_MAX_MM],
            "zero_product_values_retained": True,
            "masked_values_filled": False,
            "require_complete": bool(args.require_complete),
        },
        "files": {
            "oof_predictions": str(oof_path),
            "station_csv": str(args.station_csv),
            "product_root": str(args.product_root),
            "sampled_values": str(values_path),
            "fold_metrics": str(fold_metrics_path),
            "scatter_panel": str(plot_path),
            "sampling_audit": str(audit_path),
        },
        "sampling_audit": sampling_audit,
        "pooled_metrics": pooled_metrics,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "comparison_note": (
            "For a strict numerical comparison with M0, ERA5-Land, and other "
            "products, recompute all methods on the common rows having a valid "
            "Snow CCI retrieval."
        ),
    }
    summary_path = output_dir / "snowcci_swe_station_cv10_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Snow CCI pooled有效样本指标")
    print("=" * 72)
    print(
        f"N={pooled_metrics['n_samples']:,}/{len(predictions):,}, "
        f"R={pooled_metrics['r']:.4f}, "
        f"NSE={pooled_metrics['nse']:.4f}, "
        f"RMSE={pooled_metrics['rmse_mm']:.2f} mm, "
        f"MAE={pooled_metrics['mae_mm']:.2f} mm, "
        f"Bias={pooled_metrics['bias_mm']:.2f} mm"
    )
    print("=" * 72)
    print(f"✅ 十折散点图: {plot_path}")
    print(f"✅ 逐折指标:   {fold_metrics_path}")
    print(f"✅ 采样明细:   {values_path}")
    print(f"✅ 掩膜审计:   {audit_path}")
    print(f"✅ 汇总:       {summary_path}")


if __name__ == "__main__":
    main()
