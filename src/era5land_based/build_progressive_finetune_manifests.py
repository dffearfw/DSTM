#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import KFold


COLUMN_ALIASES = {
    "longtitude": "longitude",
    "longitude": "longitude",
    "lon": "longitude",
    "lng": "longitude",
    "long": "longitude",
    "latitude": "latitude",
    "lat": "latitude",
    "date": "date",
    "time": "date",
    "datetime": "date",
    "日期": "date",
    "station_id": "station_id",
    "station": "station_id",
    "stationid": "station_id",
    "site_id": "station_id",
    "site": "station_id",
    "id": "station_id",
    "swe": "swe",
    "swe_mm": "swe",
    "swe_value": "swe",
    "snow_water_equivalent": "swe",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")

    errors = []

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "latin1",
    ):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"无法识别文件编码: {path}\n" + "\n".join(errors)
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}

    for column in df.columns:
        original = str(column).strip()
        key = original.lower()
        rename[column] = COLUMN_ALIASES.get(key, original)

    return df.rename(columns=rename)


def parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    excel_mask = parsed.isna() & numeric.notna()

    if excel_mask.any():
        parsed.loc[excel_mask] = pd.to_datetime(
            numeric.loc[excel_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return parsed.dt.normalize()


def clean_and_map(
    df: pd.DataFrame,
    source_name: str,
    transform,
    height: int,
    width: int,
    years: set[int],
) -> tuple[pd.DataFrame, dict]:
    df = normalize_columns(df).copy()

    required = {
        "date",
        "station_id",
        "longitude",
        "latitude",
        "swe",
    }
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{source_name} 缺少字段: {missing}\n"
            f"当前字段: {list(df.columns)}"
        )

    stats = {
        "source": source_name,
        "input_rows": int(len(df)),
    }

    df["date"] = parse_dates(df["date"])

    for column in ["longitude", "latitude", "swe"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["station_id"] = (
        df["station_id"]
        .astype(str)
        .str.strip()
    )

    valid_basic = (
        df[["date", "longitude", "latitude", "swe"]]
        .notna()
        .all(axis=1)
        & ~df["station_id"].isin(["", "nan", "None"])
    )

    stats["invalid_basic_rows"] = int((~valid_basic).sum())
    df = df.loc[valid_basic].copy()

    year_mask = df["date"].dt.year.isin(years)
    stats["rows_outside_years"] = int((~year_mask).sum())
    df = df.loc[year_mask].copy()

    swe_mask = (
        np.isfinite(df["swe"])
        & (df["swe"] >= 0.0)
        & (df["swe"] < 400.0)
    )

    stats["swe_outside_0_400"] = int((~swe_mask).sum())
    df = df.loc[swe_mask].copy()

    before_exact_dedup = len(df)

    df = df.drop_duplicates(
        subset=[
            "date",
            "station_id",
            "longitude",
            "latitude",
            "swe",
        ]
    ).copy()

    stats["exact_duplicates_removed"] = int(
        before_exact_dedup - len(df)
    )

    inverse_transform = ~transform

    rows = []
    cols = []

    for longitude, latitude in zip(
        df["longitude"],
        df["latitude"],
    ):
        col_float, row_float = inverse_transform * (
            float(longitude),
            float(latitude),
        )

        rows.append(int(row_float))
        cols.append(int(col_float))

    df["row"] = rows
    df["col"] = cols

    in_grid = (
        (df["row"] >= 0)
        & (df["row"] < height)
        & (df["col"] >= 0)
        & (df["col"] < width)
    )

    stats["out_of_grid_rows"] = int((~in_grid).sum())
    df = df.loc[in_grid].copy()

    stats["mapped_rows"] = int(len(df))
    stats["unique_grid_dates"] = int(
        df.groupby(["date", "row", "col"]).ngroups
    )
    stats["unique_cells"] = int(
        len(df[["row", "col"]].drop_duplicates())
    )

    return df, stats


def aggregate_grid_date(
    df: pd.DataFrame,
    transform,
    prefix: str,
) -> pd.DataFrame:
    records = []

    grouped = df.groupby(
        ["date", "row", "col"],
        sort=True,
        dropna=False,
    )

    for (date, row, col), group in grouped:
        source_station_ids = sorted(
            set(group["station_id"].astype(str))
        )

        longitude, latitude = rasterio.transform.xy(
            transform,
            int(row),
            int(col),
            offset="center",
        )

        swe_values = group["swe"].to_numpy(
            dtype=np.float64
        )

        records.append(
            {
                "date": pd.Timestamp(date).strftime(
                    "%Y-%m-%d"
                ),
                "station_id": (
                    f"{prefix}_R{int(row):04d}_C{int(col):04d}"
                ),
                "longitude": float(longitude),
                "latitude": float(latitude),
                "swe": float(np.mean(swe_values)),
                "row": int(row),
                "col": int(col),
                "n_source_records": int(len(group)),
                "n_source_stations": int(
                    len(source_station_ids)
                ),
                "source_station_ids": ";".join(
                    source_station_ids
                ),
                "source_swe_std": float(
                    np.std(swe_values)
                ),
            }
        )

    result = pd.DataFrame(records)

    if len(result) == 0:
        raise RuntimeError(
            f"{prefix} 聚合后没有有效样本"
        )

    return result


def swe_group_counts(df: pd.DataFrame) -> np.ndarray:
    swe = df["swe"].to_numpy(dtype=np.float64)

    return np.array(
        [
            np.sum(swe <= 5.0),
            np.sum(
                (swe > 5.0)
                & (swe <= 30.0)
            ),
            np.sum(swe > 30.0),
        ],
        dtype=np.float64,
    )


def year_group_counts(
    df: pd.DataFrame,
    years: list[int],
) -> np.ndarray:
    dates = pd.to_datetime(df["date"])

    return np.array(
        [
            np.sum(dates.dt.year == year)
            for year in years
        ],
        dtype=np.float64,
    )


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    total = float(np.sum(counts))

    if total <= 0:
        return np.zeros_like(
            counts,
            dtype=np.float64,
        )

    return counts / total


def select_internal_test_cells(
    df: pd.DataFrame,
    target_samples: int,
    tolerance: int,
    seed: int,
    trials: int,
    years: list[int],
) -> tuple[set[str], dict]:
    grouped = {
        str(station_id): group.copy()
        for station_id, group
        in df.groupby("station_id", sort=True)
    }

    station_ids = np.array(
        sorted(grouped.keys()),
        dtype=object,
    )

    if len(station_ids) < 20:
        raise RuntimeError(
            f"可用内部网格太少: {len(station_ids)}"
        )

    cell_count = {
        sid: int(len(grouped[sid]))
        for sid in station_ids
    }
    cell_swe_counts = {
        sid: swe_group_counts(grouped[sid])
        for sid in station_ids
    }
    cell_year_counts = {
        sid: year_group_counts(
            grouped[sid],
            years,
        )
        for sid in station_ids
    }

    overall_swe_ratio = normalize_counts(
        swe_group_counts(df)
    )
    overall_year_ratio = normalize_counts(
        year_group_counts(df, years)
    )

    rng = np.random.default_rng(seed)
    best = None

    for _ in range(trials):
        order = station_ids.copy()
        rng.shuffle(order)

        selected = []
        sample_count = 0
        selected_swe_counts = np.zeros(
            3,
            dtype=np.float64,
        )
        selected_year_counts = np.zeros(
            len(years),
            dtype=np.float64,
        )

        for sid_value in order:
            sid = str(sid_value)
            n_cell = cell_count[sid]

            old_error = abs(
                sample_count - target_samples
            )
            new_error = abs(
                sample_count
                + n_cell
                - target_samples
            )

            if (
                sample_count >= target_samples - tolerance
                and new_error > old_error
            ):
                continue

            selected.append(sid)
            sample_count += n_cell
            selected_swe_counts += (
                cell_swe_counts[sid]
            )
            selected_year_counts += (
                cell_year_counts[sid]
            )

            if (
                sample_count
                >= target_samples + tolerance
            ):
                break

        if not selected:
            continue

        count_error = (
            abs(sample_count - target_samples)
            / max(target_samples, 1)
        )

        swe_error = float(
            np.abs(
                normalize_counts(
                    selected_swe_counts
                )
                - overall_swe_ratio
            ).sum()
        )

        year_error = float(
            np.abs(
                normalize_counts(
                    selected_year_counts
                )
                - overall_year_ratio
            ).sum()
        )

        within_tolerance = (
            abs(sample_count - target_samples)
            <= tolerance
        )

        score = (
            count_error
            + 0.35 * swe_error
            + 0.15 * year_error
        )

        ranking_key = (
            0 if within_tolerance else 1,
            score,
            abs(sample_count - target_samples),
        )

        if (
            best is None
            or ranking_key < best["ranking_key"]
        ):
            best = {
                "ranking_key": ranking_key,
                "selected": set(selected),
                "sample_count": int(sample_count),
                "swe_ratio": normalize_counts(
                    selected_swe_counts
                ),
                "year_ratio": normalize_counts(
                    selected_year_counts
                ),
            }

    if best is None:
        raise RuntimeError(
            "内部测试网格搜索失败"
        )

    if (
        abs(
            best["sample_count"]
            - target_samples
        )
        > tolerance
    ):
        raise RuntimeError(
            "未找到满足样本数量要求的内部测试集："
            f"target={target_samples}, "
            f"actual={best['sample_count']}, "
            f"tolerance={tolerance}"
        )

    info = {
        "target_samples": int(target_samples),
        "tolerance": int(tolerance),
        "actual_samples": int(
            best["sample_count"]
        ),
        "selected_grid_cells": int(
            len(best["selected"])
        ),
        "seed": int(seed),
        "trials": int(trials),
        "overall_swe_group_ratio": (
            overall_swe_ratio.tolist()
        ),
        "test_swe_group_ratio": (
            best["swe_ratio"].tolist()
        ),
        "overall_year_ratio": (
            overall_year_ratio.tolist()
        ),
        "test_year_ratio": (
            best["year_ratio"].tolist()
        ),
    }

    return best["selected"], info


def assign_cv_fold_audit(
    internal_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    cv_df = internal_df[
        internal_df["split"] != "test"
    ].copy()

    ordered_station_ids = list(
        dict.fromkeys(
            cv_df["station_id"].astype(str).tolist()
        )
    )

    if len(ordered_station_ids) < 10:
        raise RuntimeError(
            "内部CV池网格数量少于10，无法十折"
        )

    station_array = np.array(
        ordered_station_ids,
        dtype=object,
    )

    kfold = KFold(
        n_splits=10,
        shuffle=True,
        random_state=seed,
    )

    fold_map = {}

    for fold_id, (_, val_indices) in enumerate(
        kfold.split(station_array),
        start=1,
    ):
        for station_id in station_array[
            val_indices
        ]:
            fold_map[str(station_id)] = int(
                fold_id
            )

    result = internal_df.copy()

    result["cv_fold"] = [
        (
            0
            if split == "test"
            else fold_map[str(station_id)]
        )
        for station_id, split in zip(
            result["station_id"],
            result["split"],
        )
    ]

    return result


def write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--internal-excel",
        type=Path,
        default=Path(
            "/root/ablation/"
            "station_swe_data.xlsx"
        ),
    )
    parser.add_argument(
        "--external-glob",
        default=(
            "/root/ablation/"
            "external_test/*.csv"
        ),
    )
    parser.add_argument(
        "--stage0-manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/"
            "shared_cache/"
            "stage0_station_record_manifest.csv"
        ),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path(
            "/root/ablation/era5landswe"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/"
            "shared_cache/"
            "progressive_finetune"
        ),
    )
    parser.add_argument(
        "--target-test-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--test-tolerance",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
    )
    parser.add_argument(
        "--search-trials",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2015, 2016, 2017],
    )

    args = parser.parse_args()

    for required_path in [
        args.internal_excel,
        args.stage0_manifest,
        args.label_dir,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                required_path
            )

    external_paths = sorted(
        Path(path)
        for path in glob.glob(
            args.external_glob
        )
    )

    if not external_paths:
        raise FileNotFoundError(
            "没有找到外部测试CSV: "
            f"{args.external_glob}"
        )

    label_paths = sorted(
        args.label_dir.glob("*.tif")
    )

    if not label_paths:
        raise FileNotFoundError(
            f"标签目录中没有TIF: "
            f"{args.label_dir}"
        )

    with rasterio.open(
        label_paths[0]
    ) as reference:
        transform = reference.transform
        height = reference.height
        width = reference.width
        crs = str(reference.crs)

    years = sorted(
        set(int(year) for year in args.years)
    )
    year_set = set(years)

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("构建渐进式微调与双测试固定清单")
    print("=" * 80)
    print(f"参考网格: {label_paths[0]}")
    print(f"网格大小: {height} × {width}")
    print(f"年份: {years}")
    print(f"随机种子: {args.seed}")

    internal_raw, internal_stats = (
        clean_and_map(
            read_table_auto(
                args.internal_excel
            ),
            str(args.internal_excel),
            transform,
            height,
            width,
            year_set,
        )
    )

    internal = aggregate_grid_date(
        internal_raw,
        transform,
        prefix="INTGRID",
    )

    stage0_manifest = pd.read_csv(
        args.stage0_manifest,
        encoding="utf-8-sig",
    )

    required_stage0_columns = {
        "date",
        "row",
        "col",
    }

    missing_stage0_columns = sorted(
        required_stage0_columns
        - set(stage0_manifest.columns)
    )

    if missing_stage0_columns:
        raise ValueError(
            "Stage0清单缺少字段: "
            f"{missing_stage0_columns}"
        )

    stage0_manifest["date"] = parse_dates(
        stage0_manifest["date"]
    )
    stage0_manifest["row"] = pd.to_numeric(
        stage0_manifest["row"],
        errors="coerce",
    )
    stage0_manifest["col"] = pd.to_numeric(
        stage0_manifest["col"],
        errors="coerce",
    )

    stage0_manifest = (
        stage0_manifest
        .dropna(
            subset=["date", "row", "col"]
        )
        .copy()
    )

    stage0_manifest["row"] = (
        stage0_manifest["row"]
        .astype(int)
    )
    stage0_manifest["col"] = (
        stage0_manifest["col"]
        .astype(int)
    )

    stage0_manifest = stage0_manifest[
        stage0_manifest[
            "date"
        ].dt.year.isin(year_set)
    ].copy()

    stage0_valid_keys = set(
        zip(
            stage0_manifest[
                "date"
            ].dt.strftime("%Y-%m-%d"),
            stage0_manifest["row"],
            stage0_manifest["col"],
        )
    )

    internal_keys = list(
        zip(
            internal["date"],
            internal["row"],
            internal["col"],
        )
    )

    internal_valid_mask = np.array(
        [
            key in stage0_valid_keys
            for key in internal_keys
        ],
        dtype=bool,
    )

    internal_before_validity = len(
        internal
    )
    internal = internal.loc[
        internal_valid_mask
    ].copy()

    internal_not_stage0_valid = int(
        internal_before_validity
        - len(internal)
    )

    external_frames = []
    external_stats = []

    for external_path in external_paths:
        cleaned, stats = clean_and_map(
            read_table_auto(external_path),
            str(external_path),
            transform,
            height,
            width,
            year_set,
        )

        cleaned["_source_file"] = (
            external_path.name
        )

        external_frames.append(cleaned)
        external_stats.append(stats)

        print(
            f"外部文件: {external_path.name}, "
            f"有效记录={len(cleaned):,}"
        )

    external_raw = pd.concat(
        external_frames,
        ignore_index=True,
    )

    external_before_cross_dedup = len(
        external_raw
    )

    external_raw = (
        external_raw
        .drop_duplicates(
            subset=[
                "date",
                "station_id",
                "longitude",
                "latitude",
                "swe",
            ]
        )
        .copy()
    )

    external_cross_duplicates = int(
        external_before_cross_dedup
        - len(external_raw)
    )

    external = aggregate_grid_date(
        external_raw,
        transform,
        prefix="EXTGRID",
    )

    external_cells = set(
        zip(
            external["row"].astype(int),
            external["col"].astype(int),
        )
    )

    internal_overlap_mask = np.array(
        [
            (int(row), int(col))
            in external_cells
            for row, col in zip(
                internal["row"],
                internal["col"],
            )
        ],
        dtype=bool,
    )

    overlap_rows_removed = int(
        internal_overlap_mask.sum()
    )
    overlap_cells_removed = int(
        internal.loc[
            internal_overlap_mask,
            ["row", "col"],
        ]
        .drop_duplicates()
        .shape[0]
    )

    internal = internal.loc[
        ~internal_overlap_mask
    ].copy()

    selected_test_cells, test_info = (
        select_internal_test_cells(
            internal,
            target_samples=(
                args.target_test_samples
            ),
            tolerance=args.test_tolerance,
            seed=args.seed,
            trials=args.search_trials,
            years=years,
        )
    )

    internal["split"] = np.where(
        internal["station_id"].isin(
            selected_test_cells
        ),
        "test",
        "unknown",
    )

    internal = (
        internal
        .sort_values(
            ["date", "row", "col"]
        )
        .reset_index(drop=True)
    )

    internal = assign_cv_fold_audit(
        internal,
        seed=args.seed,
    )

    internal_test = internal[
        internal["split"] == "test"
    ].copy()

    internal_cv = internal[
        internal["split"] != "test"
    ].copy()

    external["split"] = "test"
    external["cv_fold"] = 0

    external = (
        external
        .sort_values(
            ["date", "row", "col"]
        )
        .reset_index(drop=True)
    )

    # evaluate模式要求同时存在非test池。
    # 这里只是为加载器提供内部CV池；
    # 实际评估只使用split=test的外部样本。
    external_evaluation_input = pd.concat(
        [
            internal_cv,
            external,
        ],
        ignore_index=True,
        sort=False,
    )

    external_evaluation_input = (
        external_evaluation_input
        .sort_values(
            ["split", "date", "row", "col"]
        )
        .reset_index(drop=True)
    )

    internal_path = (
        output_dir
        / "internal_progressive_station.csv"
    )
    internal_cv_path = (
        output_dir
        / "internal_cv_pool.csv"
    )
    internal_test_path = (
        output_dir
        / "internal_test_approximately_1000.csv"
    )
    external_path = (
        output_dir
        / "external_test_all_aggregated.csv"
    )
    external_input_path = (
        output_dir
        / "external_evaluation_input.csv"
    )
    fold_audit_path = (
        output_dir
        / "internal_cv_fold_audit.csv"
    )

    write_csv(
        internal,
        internal_path,
    )
    write_csv(
        internal_cv,
        internal_cv_path,
    )
    write_csv(
        internal_test,
        internal_test_path,
    )
    write_csv(
        external,
        external_path,
    )
    write_csv(
        external_evaluation_input,
        external_input_path,
    )

    fold_audit = (
        internal_cv[
            [
                "station_id",
                "row",
                "col",
                "cv_fold",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["cv_fold", "row", "col"]
        )
    )

    write_csv(
        fold_audit,
        fold_audit_path,
    )

    fold_sample_counts = (
        internal_cv
        .groupby("cv_fold")
        .size()
        .astype(int)
        .to_dict()
    )

    fold_grid_counts = (
        internal_cv
        .groupby("cv_fold")[
            "station_id"
        ]
        .nunique()
        .astype(int)
        .to_dict()
    )

    summary = {
        "created_at": (
            datetime.now().isoformat()
        ),
        "seed": int(args.seed),
        "years": years,
        "reference_grid": {
            "file": str(label_paths[0]),
            "height": int(height),
            "width": int(width),
            "crs": crs,
            "transform": list(transform)[:6],
        },
        "definitions": {
            "internal_test": (
                "Excel站点留出微调测试集；"
                "对微调训练独立，但当前Stage0"
                "已经使用过该Excel，不能称为"
                "全流程完全独立测试集。"
            ),
            "external_test": (
                "外部CSV全量聚合测试集；"
                "预训练已排除外部中心格点。"
            ),
            "aggregation": (
                "删除完全重复记录后，"
                "按date+ERA5 row+ERA5 col分组，"
                "SWE取平均。"
            ),
            "split_unit": (
                "完整ERA5网格；同一网格所有日期"
                "只能全部属于CV池或内部测试集。"
            ),
        },
        "internal_source_stats": (
            internal_stats
        ),
        "internal_stage0_validity": {
            "before": int(
                internal_before_validity
            ),
            "after": int(
                internal_before_validity
                - internal_not_stage0_valid
            ),
            "removed": int(
                internal_not_stage0_valid
            ),
        },
        "external_source_stats": (
            external_stats
        ),
        "external_cross_file_exact_duplicates_removed": (
            external_cross_duplicates
        ),
        "internal_external_overlap_removed": {
            "rows": overlap_rows_removed,
            "cells": overlap_cells_removed,
        },
        "counts": {
            "internal_total": int(
                len(internal)
            ),
            "internal_cv_samples": int(
                len(internal_cv)
            ),
            "internal_test_samples": int(
                len(internal_test)
            ),
            "internal_cv_grid_cells": int(
                internal_cv[
                    "station_id"
                ].nunique()
            ),
            "internal_test_grid_cells": int(
                internal_test[
                    "station_id"
                ].nunique()
            ),
            "external_raw_valid_records": int(
                len(external_raw)
            ),
            "external_aggregated_samples": int(
                len(external)
            ),
            "external_grid_cells": int(
                external[
                    "station_id"
                ].nunique()
            ),
        },
        "internal_test_selection": (
            test_info
        ),
        "cv_fold_sample_counts": {
            str(key): int(value)
            for key, value
            in fold_sample_counts.items()
        },
        "cv_fold_grid_counts": {
            str(key): int(value)
            for key, value
            in fold_grid_counts.items()
        },
        "files": {},
    }

    output_paths = [
        internal_path,
        internal_cv_path,
        internal_test_path,
        external_path,
        external_input_path,
        fold_audit_path,
    ]

    for output_path in output_paths:
        summary["files"][
            output_path.name
        ] = {
            "path": str(output_path),
            "sha256": sha256_file(
                output_path
            ),
        }

    summary_path = (
        output_dir
        / "progressive_finetune_manifest_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("✅ 渐进式微调固定清单构建完成")
    print("=" * 80)
    print(
        f"内部聚合后有效样本: "
        f"{len(internal):,}"
    )
    print(
        f"内部CV池: "
        f"{len(internal_cv):,}"
    )
    print(
        f"内部留出测试: "
        f"{len(internal_test):,} "
        f"(目标={args.target_test_samples}"
        f"±{args.test_tolerance})"
    )
    print(
        f"内部测试网格数: "
        f"{internal_test['station_id'].nunique():,}"
    )
    print(
        f"外部原始有效记录: "
        f"{len(external_raw):,}"
    )
    print(
        f"外部同日同网格聚合后: "
        f"{len(external):,}"
    )
    print(
        f"内部/外部中心格点重叠删除: "
        f"{overlap_rows_removed:,} 条，"
        f"{overlap_cells_removed:,} 格"
    )
    print(f"输出目录: {output_dir}")
    print(f"摘要文件: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
