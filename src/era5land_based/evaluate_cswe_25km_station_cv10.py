#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the 25 km daily China SWE HDF5 product on the exact 7,936 samples
and fold assignment used by the Frozen M0 balanced station-wise CV10 baseline.

Official product code handling:
  0..240 : numerical SWE in mm (zero is a valid snow-free value)
  250    : dry-snow category, no numerical SWE -> missing
  251    : wet-snow category, no numerical SWE -> missing
  252    : snow-free -> 0 mm by default
  253    : water/building -> missing
  254    : missing -> missing
  255    : outside China -> missing

No temporal or spatial gap filling is performed. HDF5 dataset paths are
auto-detected and recorded, and can be overridden from the command line.
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

from evaluate_snowcci_swe_station_cv10 import (
    EXPECTED_FOLDS,
    attach_station_coordinates,
    clean_float,
    compute_metrics,
    discover_oof_predictions,
    normalize_date,
    read_csv,
    validate_oof_frame,
)


PRODUCT_NAME_ZH = "中国雪水当量25公里逐日产品（1980–2020年）"
PRODUCT_NAME_EN = (
    "Daily snow water equivalent product from 1980 to 2020 over China"
)
PLOT_NAME = "China Daily SWE 25 km (1980–2020)"
PRODUCT_DOI = "10.12072/ncdc.I-SNOW.db0002.2020"
PRODUCT_CSTR = "11738.11.ncdc.I-SNOW.2020.6"

AXIS_MIN_MM = 0.0
AXIS_MAX_MM = 400.0
AXIS_TICK_MM = 50.0
EARTH_RADIUS_KM = 6371.0088
DATE_RE = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")
H5_SUFFIXES = {".h5", ".hdf5", ".he5"}

SWE_NAME_TOKENS = (
    "snow_water_equivalent",
    "snow water equivalent",
    "swe",
)
LAT_NAME_TOKENS = ("latitude", "lat")
LON_NAME_TOKENS = ("longitude", "lon")
BAD_SWE_NAME_TOKENS = (
    "quality",
    "flag",
    "mask",
    "uncert",
    "longitude",
    "latitude",
    "snow_depth",
    "snow depth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the 25 km China daily SWE HDF5 product using the exact "
            "Frozen M0 7,936-row balanced station-wise CV10 sample list."
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
            "the newest valid 7,936-sample M0 baseline is selected."
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
        default=Path("/root/ablation/cswe"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--scale-to-mm",
        type=float,
        default=1.0,
        help=(
            "Additional multiplicative factor after HDF5 scale_factor, "
            "add_offset and unit conversion. The documented product is "
            "already in mm, so the default is 1."
        ),
    )
    parser.add_argument(
        "--swe-dataset",
        type=str,
        default=None,
        help="Explicit HDF5 path of the SWE dataset if auto-detection fails.",
    )
    parser.add_argument(
        "--latitude-dataset",
        type=str,
        default=None,
        help="Explicit HDF5 path of the latitude dataset.",
    )
    parser.add_argument(
        "--longitude-dataset",
        type=str,
        default=None,
        help="Explicit HDF5 path of the longitude dataset.",
    )
    parser.add_argument(
        "--code-252-policy",
        choices=("zero", "missing"),
        default="zero",
        help=(
            "The product documentation defines code 252 as snow-free. "
            "Default 'zero' maps it to 0 mm; 'missing' is available for a "
            "sensitivity audit."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any OOF row has no valid numerical product value.",
    )
    parser.add_argument(
        "--inspect-file",
        type=Path,
        default=None,
        help=(
            "HDF5 file used for schema/code inspection. If omitted, the "
            "first indexed file needed by the OOF dates is used."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Write/print the HDF5 schema audit and stop before CV evaluation.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_h5_path(path: str) -> str:
    return path.strip().lstrip("/")


def list_h5_datasets(h5_file) -> dict[str, Any]:
    import h5py

    datasets: dict[str, Any] = {}

    def collect(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    h5_file.visititems(collect)
    return datasets


def is_numeric_dataset(dataset: Any) -> bool:
    return np.issubdtype(dataset.dtype, np.number)


def effective_shape(dataset: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in dataset.shape if int(value) != 1)


def select_dataset(
    datasets: dict[str, Any],
    role: str,
    explicit_path: str | None,
) -> str:
    if explicit_path:
        normalized = normalize_h5_path(explicit_path)
        if normalized not in datasets:
            raise RuntimeError(
                f"显式指定的{role}数据集不存在: {explicit_path}; "
                f"available={sorted(datasets)}"
            )
        if not is_numeric_dataset(datasets[normalized]):
            raise RuntimeError(f"{role}数据集不是数值类型: {explicit_path}")
        return normalized

    if role == "SWE":
        positive_tokens = SWE_NAME_TOKENS
        negative_tokens = BAD_SWE_NAME_TOKENS
    elif role == "latitude":
        positive_tokens = LAT_NAME_TOKENS
        negative_tokens = ("longitude", "lon")
    elif role == "longitude":
        positive_tokens = LON_NAME_TOKENS
        negative_tokens = ("latitude", "lat")
    else:
        raise ValueError(role)

    scored: list[tuple[int, str]] = []
    for path, dataset in datasets.items():
        if not is_numeric_dataset(dataset):
            continue
        lowered = path.lower().replace("-", "_")
        base = lowered.rsplit("/", 1)[-1]
        score = 0
        for token in positive_tokens:
            normalized_token = token.lower().replace("-", "_")
            if base == normalized_token:
                score += 120
            elif normalized_token in base:
                score += 70
            elif normalized_token in lowered:
                score += 30
        for token in negative_tokens:
            if token.lower().replace("-", "_") in lowered:
                score -= 100

        shape = effective_shape(dataset)
        if role == "SWE":
            if len(shape) == 2:
                score += 35
            elif len(shape) > 2:
                score += 5
            else:
                score -= 80
        else:
            if len(shape) in (1, 2):
                score += 20
            else:
                score -= 40

        if score > 0:
            scored.append((score, path))

    if not scored:
        raise RuntimeError(
            f"无法自动识别{role}数据集。available={sorted(datasets)}"
        )
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    best_score = scored[0][0]
    tied = [path for score, path in scored if score == best_score]
    if len(tied) > 1:
        raise RuntimeError(
            f"{role}数据集自动识别存在并列，请显式指定: {tied}"
        )
    return scored[0][1]


def attr_float(dataset: Any, name: str, default: float) -> float:
    attr_map = {
        str(key).strip().lower().replace(" ", "_"): key
        for key in dataset.attrs.keys()
    }
    actual_name = attr_map.get(name.strip().lower().replace(" ", "_"))
    if actual_name is None:
        return default
    values = np.asarray(dataset.attrs[actual_name]).reshape(-1)
    if len(values) == 0:
        return default
    return float(values[0])


def missing_attr_values(dataset: Any) -> list[float]:
    result: list[float] = []
    attr_map = {
        str(key).strip().lower().replace(" ", "_"): key
        for key in dataset.attrs.keys()
    }
    for name in ("_FillValue", "missing_value", "fill_value"):
        actual_name = attr_map.get(name.strip().lower().replace(" ", "_"))
        if actual_name is None:
            continue
        for value in np.asarray(dataset.attrs[actual_name]).reshape(-1):
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                pass
    return sorted(set(result))


def read_scaled_coordinates(dataset: Any) -> np.ndarray:
    raw = np.asarray(dataset[...], dtype=np.float64).squeeze()
    for missing_value in missing_attr_values(dataset):
        raw[np.isclose(raw, missing_value, rtol=0.0, atol=1e-12)] = np.nan
    scale = attr_float(dataset, "scale_factor", 1.0)
    offset = attr_float(dataset, "add_offset", 0.0)
    return raw * scale + offset


def read_raw_swe(dataset: Any) -> np.ndarray:
    raw = np.asarray(dataset[...]).squeeze()
    if raw.ndim != 2:
        raise RuntimeError(
            "SWE数据集去除单例维后必须为二维："
            f"path={dataset.name}, shape={dataset.shape}, squeezed={raw.shape}"
        )
    return raw


def unit_to_mm_factor(units: str) -> float:
    normalized = units.lower().replace(" ", "").replace("**", "^")
    if normalized in {"m", "meter", "metre", "meters", "metres"}:
        return 1000.0
    return 1.0


def inspect_h5_schema(
    path: Path,
    swe_dataset: str | None,
    latitude_dataset: str | None,
    longitude_dataset: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "缺少h5py。AutoDL环境请先执行: pip install h5py"
        ) from exc

    with h5py.File(path, mode="r") as h5_file:
        datasets = list_h5_datasets(h5_file)
        if not datasets:
            raise RuntimeError(f"HDF5没有数据集: {path}")
        selected = {
            "swe": select_dataset(datasets, "SWE", swe_dataset),
            "latitude": select_dataset(
                datasets,
                "latitude",
                latitude_dataset,
            ),
            "longitude": select_dataset(
                datasets,
                "longitude",
                longitude_dataset,
            ),
        }
        swe = datasets[selected["swe"]]
        raw = read_raw_swe(swe)
        finite = np.asarray(raw, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        code_counts = {
            str(code): int(np.sum(np.isclose(finite, float(code))))
            for code in range(250, 256)
        }
        dataset_audit = []
        for name, dataset in sorted(datasets.items()):
            dataset_audit.append(
                {
                    "path": name,
                    "shape": list(dataset.shape),
                    "dtype": str(dataset.dtype),
                    "attrs": {
                        str(key): json_safe(value)
                        for key, value in dataset.attrs.items()
                    },
                }
            )
        audit = {
            "file": str(path),
            "file_attrs": {
                str(key): json_safe(value)
                for key, value in h5_file.attrs.items()
            },
            "datasets": dataset_audit,
            "selected_datasets": selected,
            "selected_swe": {
                "shape": list(swe.shape),
                "squeezed_shape": list(raw.shape),
                "dtype": str(swe.dtype),
                "attrs": {
                    str(key): json_safe(value)
                    for key, value in swe.attrs.items()
                },
                "raw_min": (
                    float(np.min(finite)) if finite.size else None
                ),
                "raw_max": (
                    float(np.max(finite)) if finite.size else None
                ),
                "n_raw_0_to_240": int(
                    np.sum((finite >= 0.0) & (finite <= 240.0))
                ),
                "special_code_counts": code_counts,
            },
        }
    return audit, selected


def index_product_files(product_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(product_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in H5_SUFFIXES:
            continue
        match = DATE_RE.search(path.name)
        if match is None:
            continue
        try:
            date_text = datetime.strptime(
                match.group(1),
                "%Y%m%d",
            ).strftime("%Y-%m-%d")
        except ValueError:
            continue
        if date_text in files:
            duplicates.append(date_text)
        else:
            files[date_text] = path

    if duplicates:
        raise RuntimeError(
            "同一日期存在重复CSWE HDF5文件，请先消除歧义："
            f"{sorted(set(duplicates))[:20]}"
        )
    if not files:
        raise FileNotFoundError(
            f"没有找到文件名含YYYYMMDD的H5/HDF5/HE5文件: {product_root}"
        )
    print(
        f"✅ 已索引 {len(files):,} 个CSWE逐日文件: "
        f"{min(files)} 至 {max(files)}"
    )
    return files


def normalize_longitudes(
    station_lon: np.ndarray,
    grid_lon: np.ndarray,
) -> np.ndarray:
    station_lon = np.asarray(station_lon, dtype=np.float64).copy()
    finite_grid = np.asarray(grid_lon, dtype=np.float64)
    finite_grid = finite_grid[np.isfinite(finite_grid)]
    if finite_grid.size and np.nanmin(finite_grid) >= 0.0:
        station_lon = np.mod(station_lon, 360.0)
    else:
        station_lon = ((station_lon + 180.0) % 360.0) - 180.0
    return station_lon


def nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_positions = np.flatnonzero(np.isfinite(axis))
    if finite_positions.size < 2:
        raise RuntimeError("HDF5一维坐标轴无效")
    order = np.argsort(axis[finite_positions])
    sorted_positions = finite_positions[order]
    work = axis[sorted_positions]
    insertion = np.searchsorted(work, values, side="left")
    insertion = np.clip(insertion, 1, len(work) - 1)
    left = insertion - 1
    right = insertion
    selected = np.where(
        np.abs(work[right] - values) < np.abs(work[left] - values),
        right,
        left,
    )
    return sorted_positions[selected].astype(np.int64)


def lon_lat_to_unit(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64))
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        (
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        )
    )


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    chord = np.clip(np.asarray(chord, dtype=np.float64), 0.0, 2.0)
    return 2.0 * np.arcsin(chord / 2.0) * EARTH_RADIUS_KM


def nearest_irregular_grid(
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    station_lon: np.ndarray,
    station_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_grid = np.isfinite(grid_lon) & np.isfinite(grid_lat)
    valid_flat = np.flatnonzero(valid_grid.reshape(-1))
    if valid_flat.size == 0:
        raise RuntimeError("二维经纬度网格没有有限坐标")

    grid_units = lon_lat_to_unit(
        grid_lon.reshape(-1)[valid_flat],
        grid_lat.reshape(-1)[valid_flat],
    )
    station_units = lon_lat_to_unit(station_lon, station_lat)
    try:
        from scipy.spatial import cKDTree

        distances, local_indices = cKDTree(grid_units).query(
            station_units,
            k=1,
        )
    except ImportError:
        local_indices = np.empty(len(station_units), dtype=np.int64)
        distances = np.empty(len(station_units), dtype=np.float64)
        for start in range(0, len(station_units), 256):
            stop = min(start + 256, len(station_units))
            dots = station_units[start:stop] @ grid_units.T
            selected = np.argmax(dots, axis=1)
            local_indices[start:stop] = selected
            distances[start:stop] = np.sqrt(
                np.maximum(0.0, 2.0 - 2.0 * dots[np.arange(stop - start), selected])
            )
    return valid_flat[local_indices], chord_to_km(distances)


def build_grid_mapper(
    h5_path: Path,
    selected: dict[str, str],
    station_lon: np.ndarray,
    station_lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import h5py

    with h5py.File(h5_path, mode="r") as h5_file:
        swe = read_raw_swe(h5_file[selected["swe"]])
        lat = read_scaled_coordinates(h5_file[selected["latitude"]])
        lon = read_scaled_coordinates(h5_file[selected["longitude"]])

    station_lon = normalize_longitudes(station_lon, lon)
    grid_lon_for_bounds = normalize_longitudes(
        np.asarray(lon, dtype=np.float64),
        lon,
    )
    inside = (
        np.isfinite(station_lon)
        & np.isfinite(station_lat)
        & (station_lon >= np.nanmin(grid_lon_for_bounds))
        & (station_lon <= np.nanmax(grid_lon_for_bounds))
        & (station_lat >= np.nanmin(lat))
        & (station_lat <= np.nanmax(lat))
    )
    flat_indices = np.full(len(station_lon), -1, dtype=np.int64)
    distance_km = np.full(len(station_lon), np.nan, dtype=np.float64)

    inside_positions = np.flatnonzero(inside)
    if lat.ndim == 1 and lon.ndim == 1:
        if lat.size == swe.shape[0] and lon.size == swe.shape[1]:
            rows = nearest_axis_indices(lat, station_lat[inside])
            columns = nearest_axis_indices(
                grid_lon_for_bounds,
                station_lon[inside],
            )
        elif lon.size == swe.shape[0] and lat.size == swe.shape[1]:
            rows = nearest_axis_indices(
                grid_lon_for_bounds,
                station_lon[inside],
            )
            columns = nearest_axis_indices(lat, station_lat[inside])
        else:
            raise RuntimeError(
                "一维经纬度长度与SWE二维形状不匹配："
                f"swe={swe.shape}, lat={lat.shape}, lon={lon.shape}"
            )
        selected_flat = np.ravel_multi_index(
            (rows, columns),
            swe.shape,
        )
        flat_indices[inside_positions] = selected_flat
        if lat.size == swe.shape[0]:
            sampled_lat = lat[rows]
            sampled_lon = grid_lon_for_bounds[columns]
        else:
            sampled_lat = lat[columns]
            sampled_lon = grid_lon_for_bounds[rows]
        station_unit = lon_lat_to_unit(
            station_lon[inside],
            station_lat[inside],
        )
        sampled_unit = lon_lat_to_unit(sampled_lon, sampled_lat)
        chord = np.linalg.norm(station_unit - sampled_unit, axis=1)
        distance_km[inside_positions] = chord_to_km(chord)
        grid_type = "one-dimensional regular longitude/latitude axes"
    elif lat.ndim == 2 and lon.ndim == 2:
        if lat.shape == swe.shape and lon.shape == swe.shape:
            grid_lat = lat
            grid_lon = grid_lon_for_bounds
        elif lat.T.shape == swe.shape and lon.T.shape == swe.shape:
            grid_lat = lat.T
            grid_lon = grid_lon_for_bounds.T
        else:
            raise RuntimeError(
                "二维经纬度网格与SWE形状不匹配："
                f"swe={swe.shape}, lat={lat.shape}, lon={lon.shape}"
            )
        selected_flat, selected_distance = nearest_irregular_grid(
            grid_lon,
            grid_lat,
            station_lon[inside],
            station_lat[inside],
        )
        flat_indices[inside_positions] = selected_flat
        distance_km[inside_positions] = selected_distance
        grid_type = "two-dimensional longitude/latitude grid"
    else:
        raise RuntimeError(
            "仅支持一维坐标轴或二维经纬网格："
            f"lat={lat.shape}, lon={lon.shape}"
        )

    valid_distance = distance_km[np.isfinite(distance_km)]
    grid_audit = {
        "example_file": str(h5_path),
        "grid_type": grid_type,
        "swe_shape": list(swe.shape),
        "latitude_shape": list(lat.shape),
        "longitude_shape": list(lon.shape),
        "latitude_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
        "longitude_range": [
            float(np.nanmin(grid_lon_for_bounds)),
            float(np.nanmax(grid_lon_for_bounds)),
        ],
        "n_station_rows_inside_grid_bounds": int(np.sum(inside)),
        "n_station_rows_outside_grid_bounds": int(np.sum(~inside)),
        "nearest_distance_km": {
            "median": (
                float(np.median(valid_distance))
                if valid_distance.size
                else None
            ),
            "p95": (
                float(np.quantile(valid_distance, 0.95))
                if valid_distance.size
                else None
            ),
            "max": (
                float(np.max(valid_distance))
                if valid_distance.size
                else None
            ),
        },
    }
    return flat_indices, inside, grid_audit


def classify_product_value(
    raw_value: Any,
    dataset: Any,
    additional_scale: float,
    code_252_policy: str,
) -> tuple[float, str]:
    try:
        raw = float(raw_value)
    except (TypeError, ValueError):
        return float("nan"), "nonfinite"
    if not np.isfinite(raw):
        return float("nan"), "nonfinite"

    special_codes = {
        250: "dry_snow_code_250_no_numeric_swe",
        251: "wet_snow_code_251_no_numeric_swe",
        253: "water_or_building_code_253",
        254: "missing_code_254",
        255: "outside_china_code_255",
    }
    for code, status in special_codes.items():
        if np.isclose(raw, float(code), rtol=0.0, atol=1e-8):
            return float("nan"), status
    if np.isclose(raw, 252.0, rtol=0.0, atol=1e-8):
        if code_252_policy == "zero":
            return 0.0, "snow_free_code_252_mapped_to_zero"
        return float("nan"), "snow_free_code_252_treated_as_missing"

    for missing_value in missing_attr_values(dataset):
        if np.isclose(raw, missing_value, rtol=0.0, atol=1e-8):
            return float("nan"), "hdf5_fill_or_missing_value"

    if raw < 0.0 or raw > 240.0:
        return float("nan"), "outside_documented_numeric_range_0_240"

    scale = attr_float(dataset, "scale_factor", 1.0)
    offset = attr_float(dataset, "add_offset", 0.0)
    units_key = next(
        (
            key
            for key in dataset.attrs.keys()
            if str(key).strip().lower() in {"units", "unit"}
        ),
        None,
    )
    units = str(dataset.attrs[units_key]) if units_key is not None else "mm"
    value_mm = (
        (raw * scale + offset)
        * unit_to_mm_factor(units)
        * additional_scale
    )
    if not np.isfinite(value_mm) or value_mm < 0.0:
        return float("nan"), "invalid_scaled_swe"
    return float(value_mm), "valid_numeric_swe_0_240"


def sample_product(
    frame: pd.DataFrame,
    product_files: dict[str, Path],
    selected: dict[str, str],
    mapper_path: Path,
    additional_scale: float,
    code_252_policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import h5py

    result = frame.copy()
    result["_date_key"] = result["label_date"].map(normalize_date)
    station_lon = result["longitude"].to_numpy(dtype=np.float64)
    station_lat = result["latitude"].to_numpy(dtype=np.float64)
    flat_indices, inside, grid_audit = build_grid_mapper(
        mapper_path,
        selected,
        station_lon,
        station_lat,
    )

    values = np.full(len(result), np.nan, dtype=np.float64)
    statuses = np.full(len(result), "not_sampled", dtype=object)
    statuses[~inside] = "outside_grid_bounds"
    missing_dates: dict[str, int] = {}
    date_groups = list(result.groupby("_date_key", sort=True).groups.items())
    preferred_swe_path = selected["swe"]
    per_file_swe_paths: Counter[str] = Counter()

    for date_number, (date_text, index_values) in enumerate(
        date_groups,
        start=1,
    ):
        positions = np.asarray(list(index_values), dtype=np.int64)
        path = product_files.get(date_text)
        if path is None:
            statuses[positions] = "missing_date_file"
            missing_dates[date_text] = int(len(positions))
            continue

        with h5py.File(path, mode="r") as h5_file:
            datasets = list_h5_datasets(h5_file)
            if preferred_swe_path in datasets:
                swe_path = preferred_swe_path
            else:
                swe_path = select_dataset(datasets, "SWE", None)
            per_file_swe_paths[swe_path] += 1
            dataset = datasets[swe_path]
            raw = read_raw_swe(dataset)
            expected_shape = tuple(grid_audit["swe_shape"])
            if raw.shape != expected_shape:
                raise RuntimeError(
                    "不同日期SWE网格形状发生变化，禁止复用空间映射："
                    f"expected={expected_shape}, actual={raw.shape}, file={path}"
                )

            eligible = positions[inside[positions]]
            if len(eligible):
                raw_values = raw.reshape(-1)[flat_indices[eligible]]
                for output_position, raw_value in zip(eligible, raw_values):
                    value_mm, status = classify_product_value(
                        raw_value,
                        dataset,
                        additional_scale,
                        code_252_policy,
                    )
                    values[output_position] = value_mm
                    statuses[output_position] = status

        if date_number % 100 == 0 or date_number == len(date_groups):
            print(
                f"  CSWE采样进度: "
                f"{date_number}/{len(date_groups)} 个观测日期"
            )

    result["cswe_25km_swe_mm"] = values
    result["cswe_25km_sampling_status"] = statuses
    result = result.drop(columns=["_date_key"])
    status_counts = Counter(statuses.tolist())
    n_valid = int(np.sum(np.isfinite(values)))
    audit = {
        "n_requested": int(len(result)),
        "n_valid": n_valid,
        "n_missing": int(len(result) - n_valid),
        "valid_ratio": float(n_valid / len(result)),
        "status_counts": dict(sorted(status_counts.items())),
        "n_missing_date": int(sum(missing_dates.values())),
        "missing_dates": missing_dates,
        "additional_scale_to_mm": float(additional_scale),
        "code_252_policy": code_252_policy,
        "documented_code_mapping": {
            "0_240": "valid numerical SWE in mm",
            "250": "dry-snow category; excluded because no numerical SWE",
            "251": "wet-snow category; excluded because no numerical SWE",
            "252": (
                "snow-free; mapped to 0 mm"
                if code_252_policy == "zero"
                else "snow-free; sensitivity policy treats it as missing"
            ),
            "253": "water/building; missing",
            "254": "missing",
            "255": "outside China; missing",
        },
        "sampling_method": (
            "nearest native 25-km HDF5 grid-cell centre; no spatial or "
            "temporal infilling"
        ),
        "grid_audit": grid_audit,
        "swe_dataset_paths_used": dict(sorted(per_file_swe_paths.items())),
    }
    return result, audit


def build_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        fold_frame = frame[frame["fold"] == fold]
        valid = (
            pd.to_numeric(fold_frame["target_mm"], errors="coerce").notna()
            & fold_frame["cswe_25km_swe_mm"].notna()
        )
        if int(valid.sum()) == 0:
            raise RuntimeError(f"Fold {fold}没有有效CSWE产品值")
        metrics = compute_metrics(
            fold_frame.loc[valid, "target_mm"],
            fold_frame.loc[valid, "cswe_25km_swe_mm"],
        )
        metrics.update(
            {
                "fold": fold,
                "method": PLOT_NAME,
                "n_total_fold_samples": int(len(fold_frame)),
                "n_missing_product": int((~valid).sum()),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def metric_text(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "NA"


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
    ticks = np.arange(
        AXIS_MIN_MM,
        AXIS_MAX_MM + AXIS_TICK_MM,
        AXIS_TICK_MM,
    )
    for fold in EXPECTED_FOLDS:
        axis = axes[fold - 1]
        fold_frame = frame[frame["fold"] == fold]
        target = pd.to_numeric(
            fold_frame["target_mm"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        prediction = fold_frame["cswe_25km_swe_mm"].to_numpy(
            dtype=np.float64
        )
        valid = np.isfinite(target) & np.isfinite(prediction)
        target = target[valid]
        prediction = prediction[valid]
        metrics = (
            fold_metrics.loc[fold_metrics["fold"] == fold]
            .iloc[0]
            .to_dict()
        )

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
            xx = np.array([AXIS_MIN_MM, AXIS_MAX_MM])
            axis.plot(
                xx,
                intercept + slope * xx,
                color="red",
                linewidth=1.4,
                label="Fit",
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
                f"R={metric_text(metrics['r'])}\n"
                f"RMSE={metric_text(metrics['rmse_mm'], 2)}\n"
                f"MAE={metric_text(metrics['mae_mm'], 2)}\n"
                f"Bias={metric_text(metrics['bias_mm'], 2)}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
        )

    for row in range(2):
        axes[row * 5].set_ylabel("China Daily SWE 25 km (mm)")
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
        (
            "All x/y axes fixed at 0–400 mm; categorical/masked product "
            "codes omitted; code 252 treated as snow-free"
        ),
        ha="right",
        va="bottom",
        fontsize=8.2,
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
    if not args.product_root.is_dir():
        raise FileNotFoundError(args.product_root)
    if not np.isfinite(args.scale_to_mm) or args.scale_to_mm <= 0:
        raise ValueError("--scale-to-mm必须是有限正数")

    product_files = index_product_files(args.product_root)
    inspect_path = (
        args.inspect_file.expanduser().resolve()
        if args.inspect_file is not None
        else next(iter(product_files.values()))
    )
    if not inspect_path.is_file():
        raise FileNotFoundError(inspect_path)

    if args.output_dir is None:
        output_dir = args.root / "experiments" / (
            "cswe_25km_station_cv10_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_audit, selected = inspect_h5_schema(
        inspect_path,
        args.swe_dataset,
        args.latitude_dataset,
        args.longitude_dataset,
    )
    schema_path = output_dir / "cswe_25km_h5_schema_audit.json"
    schema_path.write_text(
        json.dumps(schema_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ HDF5结构审计: {schema_path}")
    print(f"   自动选中数据集: {selected}")
    print(
        "   特殊码计数: "
        f"{schema_audit['selected_swe']['special_code_counts']}"
    )
    if args.inspect_only:
        print("✅ --inspect-only完成，未执行站点采样和十折评估")
        return

    if not args.station_csv.is_file():
        raise FileNotFoundError(args.station_csv)
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
    predictions = attach_station_coordinates(predictions, args.station_csv)

    sample_dates = predictions["label_date"].map(normalize_date)
    mapper_path = next(
        (
            product_files[date_text]
            for date_text in sample_dates
            if date_text in product_files
        ),
        inspect_path,
    )
    if mapper_path != inspect_path:
        _, mapper_selected = inspect_h5_schema(
            mapper_path,
            args.swe_dataset,
            args.latitude_dataset,
            args.longitude_dataset,
        )
        selected = mapper_selected

    print("=" * 108)
    print(PRODUCT_NAME_EN)
    print(PRODUCT_NAME_ZH)
    print("确定性平衡站点级10折产品基线")
    print("=" * 108)
    print(f"OOF样本/折清单: {oof_path}")
    print(f"站点CSV:         {args.station_csv}")
    print(f"产品目录:        {args.product_root}")
    print(f"输出目录:        {output_dir}")
    print(f"HDF5数据集:      {selected}")
    print(f"额外转mm系数:    {args.scale_to_mm}")
    print(f"252处理:         {args.code_252_policy}")
    print("空间采样:        最近25-km原生网格中心，不做邻域填补")
    print("缺失处理:        250/251/253/254/255不参与指标，不填补")
    print("坐标范围:        所有fold固定为0–400 mm")
    print("=" * 108)

    predictions, sampling_audit = sample_product(
        predictions,
        product_files,
        selected,
        mapper_path,
        args.scale_to_mm,
        args.code_252_policy,
    )
    audit_path = output_dir / "cswe_25km_sampling_audit.json"
    audit_path.write_text(
        json.dumps(sampling_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    missing_count = int(predictions["cswe_25km_swe_mm"].isna().sum())
    if missing_count:
        print(
            f"⚠ CSWE缺失/类别码: {missing_count:,}/{len(predictions):,} "
            f"({100.0 * missing_count / len(predictions):.2f}%)"
        )
        print(f"  分类: {sampling_audit['status_counts']}")
        if args.require_complete:
            raise RuntimeError(
                "CSWE存在缺失/类别码且启用了--require-complete。"
                f"审计={audit_path}"
            )

    values_path = output_dir / "cswe_25km_station_cv10_oof_values.csv"
    predictions.to_csv(values_path, index=False, encoding="utf-8-sig")
    fold_metrics = build_fold_metrics(predictions)
    metrics_path = output_dir / "cswe_25km_station_cv10_fold_metrics.csv"
    fold_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pooled_metrics = compute_metrics(
        predictions["target_mm"],
        predictions["cswe_25km_swe_mm"],
    )

    plot_path = output_dir / (
        "cswe_25km_station_cv10_fold_scatter_panel.png"
    )
    plot_fold_panel(predictions, fold_metrics, plot_path)

    summary = {
        "created_at": datetime.now().isoformat(),
        "product": {
            "name_zh": PRODUCT_NAME_ZH,
            "name_en": PRODUCT_NAME_EN,
            "plot_name": PLOT_NAME,
            "doi": PRODUCT_DOI,
            "cstr": PRODUCT_CSTR,
            "documented_resolution": "25 km",
            "documented_period": "1980-01-01 to 2020-01-31",
        },
        "protocol": {
            "sample_pool": (
                "exact rows from the existing Frozen M0 "
                "7,936-sample OOF file"
            ),
            "fold_assignment": (
                "reused unchanged from the existing balanced station-wise "
                "10-fold OOF file"
            ),
            "n_total_samples": int(len(predictions)),
            "n_valid_product_samples": int(
                predictions["cswe_25km_swe_mm"].notna().sum()
            ),
            "n_missing_or_categorical_samples": missing_count,
            "n_splits": 10,
            "scatter_axis_mm": [AXIS_MIN_MM, AXIS_MAX_MM],
            "numeric_zero_retained": True,
            "code_252_policy": args.code_252_policy,
            "masked_values_filled": False,
            "require_complete": bool(args.require_complete),
        },
        "files": {
            "oof_predictions": str(oof_path),
            "station_csv": str(args.station_csv),
            "product_root": str(args.product_root),
            "schema_audit": str(schema_path),
            "sampling_audit": str(audit_path),
            "sampled_values": str(values_path),
            "fold_metrics": str(metrics_path),
            "scatter_panel": str(plot_path),
        },
        "schema_audit": schema_audit,
        "sampling_audit": sampling_audit,
        "pooled_metrics": pooled_metrics,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "comparison_note": (
            "For strict cross-product numerical comparison, recompute all "
            "methods on the common subset having valid values from every "
            "compared product. This product-specific panel uses all valid "
            "CSWE rows and reports per-fold N."
        ),
    }
    summary_path = output_dir / "cswe_25km_station_cv10_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("CSWE 25 km pooled有效样本指标")
    print("=" * 72)
    print(
        f"N={pooled_metrics['n_samples']:,}/{len(predictions):,}, "
        f"R={metric_text(pooled_metrics['r'], 4)}, "
        f"NSE={metric_text(pooled_metrics['nse'], 4)}, "
        f"RMSE={pooled_metrics['rmse_mm']:.2f} mm, "
        f"MAE={pooled_metrics['mae_mm']:.2f} mm, "
        f"Bias={pooled_metrics['bias_mm']:.2f} mm"
    )
    print("=" * 72)
    print(f"✅ 十折散点图: {plot_path}")
    print(f"✅ 逐折指标:   {metrics_path}")
    print(f"✅ 采样明细:   {values_path}")
    print(f"✅ 编码审计:   {audit_path}")
    print(f"✅ HDF5审计:   {schema_path}")
    print(f"✅ 汇总:       {summary_path}")


if __name__ == "__main__":
    main()
