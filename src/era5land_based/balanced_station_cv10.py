#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic, lightly balanced station-level 10-fold manifest.

The design is intentionally simple:
1. station_id is indivisible;
2. balance total sample count;
3. balance high-SWE (>= threshold) sample count;
4. lightly balance station count;
5. no random seed and no region/year constraints.

The manifest is created from the internal station CSV and then reused by
Frozen and every fine-tuning strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path(
    "/root/autodl-tmp/shared_cache/progressive_finetune/"
    "balanced_station_cv10_manifest.csv"
)


def normalize_station_id(value: Any) -> str:
    return str(value).split(",")[0].strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise RuntimeError(
        f"找不到列，候选={list(candidates)}, 实际={frame.columns.tolist()}"
    )


@dataclass(frozen=True)
class StationStat:
    station_id: str
    n_samples: int
    n_high_swe: int
    swe_mean_mm: float
    swe_max_mm: float


def station_stats_from_csv(
    station_csv: Path,
    high_threshold_mm: float = 80.0,
    include_fixed_test: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    station_csv = Path(station_csv).expanduser().resolve()
    source = pd.read_csv(station_csv)

    station_col = _find_column(source, ["station_id", "station", "站点ID"])
    target_col = _find_column(source, ["swe", "target_mm", "SWE"])
    split_col = _find_column(source, ["split", "Split", "subset"])

    split_norm = source[split_col].astype(str).str.strip().str.lower()
    fixed_test_mask = split_norm.str.contains("test", regex=False)
    if include_fixed_test:
        cv_source = source.copy()
        pool_policy = "all_internal_rows"
    else:
        cv_source = source.loc[~fixed_test_mask].copy()
        pool_policy = "exclude_fixed_internal_test"

    if cv_source.empty:
        raise RuntimeError("内部CV池为空")

    cv_source["_station_id"] = cv_source[station_col].map(normalize_station_id)
    cv_source["_target_mm"] = pd.to_numeric(
        cv_source[target_col], errors="coerce"
    )

    if cv_source["_target_mm"].isna().any():
        bad = cv_source.loc[cv_source["_target_mm"].isna()].head(10)
        raise RuntimeError(f"内部CV池存在非法SWE：\n{bad}")

    grouped = cv_source.groupby("_station_id", sort=True)["_target_mm"]
    stats = grouped.agg(["size", "mean", "max"]).reset_index()
    high = grouped.apply(lambda series: int((series >= high_threshold_mm).sum()))

    stats = stats.rename(
        columns={
            "_station_id": "station_id",
            "size": "n_samples",
            "mean": "swe_mean_mm",
            "max": "swe_max_mm",
        }
    )
    stats["n_high_swe"] = stats["station_id"].map(high.to_dict()).astype(int)
    stats["n_samples"] = stats["n_samples"].astype(int)

    metadata = {
        "station_csv": str(station_csv),
        "station_csv_sha256": sha256_file(station_csv),
        "split_column": split_col,
        "station_column": station_col,
        "target_column": target_col,
        "n_cv_samples": int(len(cv_source)),
        "n_cv_stations": int(stats["station_id"].nunique()),
        "n_fixed_test_rows_in_source": int(fixed_test_mask.sum()),
        "include_fixed_test": bool(include_fixed_test),
        "pool_policy": pool_policy,
        "n_high_swe_samples": int(
            (cv_source["_target_mm"] >= high_threshold_mm).sum()
        ),
        "high_threshold_mm": float(high_threshold_mm),
    }
    return stats, metadata


def assign_balanced_folds(
    station_stats: pd.DataFrame,
    n_splits: int = 10,
) -> pd.DataFrame:
    """Deterministic greedy assignment with only three balance targets."""
    required = {
        "station_id",
        "n_samples",
        "n_high_swe",
        "swe_mean_mm",
        "swe_max_mm",
    }
    missing = required - set(station_stats.columns)
    if missing:
        raise RuntimeError(f"station_stats缺少列: {sorted(missing)}")

    stats = station_stats.copy()
    stats["station_id"] = stats["station_id"].map(normalize_station_id)

    if stats["station_id"].duplicated().any():
        raise RuntimeError("station_stats存在重复station_id")
    if len(stats) < n_splits:
        raise RuntimeError(
            f"站点数{len(stats)}小于n_splits={n_splits}"
        )

    # Difficult/large stations are placed first. Tie-breaking is station_id,
    # making the manifest fully deterministic.
    stats = stats.sort_values(
        ["n_high_swe", "n_samples", "swe_max_mm", "station_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    total_samples = float(stats["n_samples"].sum())
    total_high = float(stats["n_high_swe"].sum())
    total_stations = float(len(stats))

    target_samples = total_samples / n_splits
    target_high = total_high / n_splits
    target_stations = total_stations / n_splits

    fold_samples = np.zeros(n_splits, dtype=np.float64)
    fold_high = np.zeros(n_splits, dtype=np.float64)
    fold_stations = np.zeros(n_splits, dtype=np.float64)

    assignments: list[int] = []

    def global_cost(
        samples: np.ndarray,
        high: np.ndarray,
        stations: np.ndarray,
    ) -> float:
        sample_cost = np.sum(
            ((samples - target_samples) / max(target_samples, 1.0)) ** 2
        )
        high_cost = (
            np.sum(((high - target_high) / max(target_high, 1.0)) ** 2)
            if total_high > 0
            else 0.0
        )
        station_cost = np.sum(
            ((stations - target_stations) / max(target_stations, 1.0)) ** 2
        )
        # Sample balance and high-SWE balance are equally important;
        # station count is only a light tie stabilizer.
        return float(sample_cost + high_cost + 0.20 * station_cost)

    for row in stats.itertuples(index=False):
        candidate_scores: list[tuple[float, float, float, float, int]] = []

        for fold_zero in range(n_splits):
            next_samples = fold_samples.copy()
            next_high = fold_high.copy()
            next_stations = fold_stations.copy()

            next_samples[fold_zero] += float(row.n_samples)
            next_high[fold_zero] += float(row.n_high_swe)
            next_stations[fold_zero] += 1.0

            score = global_cost(next_samples, next_high, next_stations)
            candidate_scores.append(
                (
                    score,
                    next_samples[fold_zero],
                    next_high[fold_zero],
                    next_stations[fold_zero],
                    fold_zero,
                )
            )

        _, _, _, _, selected = min(candidate_scores)
        assignments.append(selected + 1)
        fold_samples[selected] += float(row.n_samples)
        fold_high[selected] += float(row.n_high_swe)
        fold_stations[selected] += 1.0

    stats["fold"] = assignments
    stats = stats[
        [
            "station_id",
            "fold",
            "n_samples",
            "n_high_swe",
            "swe_mean_mm",
            "swe_max_mm",
        ]
    ].sort_values(["fold", "station_id"]).reset_index(drop=True)

    validate_manifest(stats, station_stats, n_splits=n_splits)
    return stats


def validate_manifest(
    manifest: pd.DataFrame,
    station_stats: pd.DataFrame,
    n_splits: int = 10,
) -> None:
    required = {
        "station_id",
        "fold",
        "n_samples",
        "n_high_swe",
        "swe_mean_mm",
        "swe_max_mm",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"fold manifest缺少列: {sorted(missing)}")

    work = manifest.copy()
    work["station_id"] = work["station_id"].map(normalize_station_id)
    work["fold"] = pd.to_numeric(work["fold"], errors="raise").astype(int)

    if work["station_id"].duplicated().any():
        duplicate = work.loc[work["station_id"].duplicated(), "station_id"].tolist()
        raise RuntimeError(f"fold manifest站点重复: {duplicate[:20]}")

    expected_folds = set(range(1, n_splits + 1))
    actual_folds = set(work["fold"].unique().tolist())
    if actual_folds != expected_folds:
        raise RuntimeError(
            f"fold编号异常: expected={sorted(expected_folds)}, "
            f"actual={sorted(actual_folds)}"
        )

    current = station_stats.copy()
    current["station_id"] = current["station_id"].map(normalize_station_id)

    expected_stations = set(current["station_id"])
    manifest_stations = set(work["station_id"])
    if manifest_stations != expected_stations:
        raise RuntimeError(
            "fold manifest与当前CV站点集合不一致: "
            f"missing={sorted(expected_stations-manifest_stations)[:20]}, "
            f"extra={sorted(manifest_stations-expected_stations)[:20]}"
        )

    check = work.merge(
        current[["station_id", "n_samples", "n_high_swe"]],
        on="station_id",
        suffixes=("_manifest", "_current"),
        how="inner",
    )
    mismatched = check[
        (check["n_samples_manifest"] != check["n_samples_current"])
        | (check["n_high_swe_manifest"] != check["n_high_swe_current"])
    ]
    if not mismatched.empty:
        raise RuntimeError(
            "fold manifest中的站点统计与当前CSV不一致，拒绝复用：\n"
            + mismatched.head(20).to_string(index=False)
        )


def fold_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby("fold", sort=True)
        .agg(
            n_stations=("station_id", "nunique"),
            n_samples=("n_samples", "sum"),
            n_high_swe=("n_high_swe", "sum"),
            mean_station_swe_mm=("swe_mean_mm", "mean"),
            max_station_swe_mm=("swe_max_mm", "max"),
        )
        .reset_index()
    )


def create_or_load_manifest(
    station_csv: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    n_splits: int = 10,
    high_threshold_mm: float = 80.0,
    force_rebuild: bool = False,
    include_fixed_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    station_csv = Path(station_csv).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()

    stats, metadata = station_stats_from_csv(
        station_csv=station_csv,
        high_threshold_mm=high_threshold_mm,
        include_fixed_test=include_fixed_test,
    )

    if manifest_path.exists() and not force_rebuild:
        manifest = pd.read_csv(manifest_path)
        validate_manifest(manifest, stats, n_splits=n_splits)
        status = "loaded"
    else:
        manifest = assign_balanced_folds(stats, n_splits=n_splits)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        status = "created"

    summary = fold_summary(manifest)
    metadata = {
        **metadata,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_status": status,
        "created_or_checked_at": datetime.now().isoformat(),
        "method": "deterministic_balanced_greedy_v1",
        "balance_targets": [
            "total_sample_count",
            "high_swe_sample_count_ge_80mm",
            "station_count_light_weight",
        ],
        "randomized": False,
        "n_splits": int(n_splits),
    }

    metadata_path = manifest_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary.to_csv(
        manifest_path.with_suffix(".fold_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return manifest, summary, metadata


def mapping_for_dataset_indices(
    dataset,
    cv_indices: list[int],
    manifest: pd.DataFrame,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    station_to_fold = {
        normalize_station_id(row.station_id): int(row.fold)
        for row in manifest.itertuples(index=False)
    }

    fold_by_index: dict[int, int] = {}
    station_to_indices: dict[str, list[int]] = {}

    for dataset_idx in cv_indices:
        meta = dataset.meta_index[int(dataset_idx)]
        station_id = normalize_station_id(meta.get("station_id", "unknown"))
        if station_id not in station_to_fold:
            raise RuntimeError(f"站点不在平衡fold manifest中: {station_id}")
        fold_by_index[int(dataset_idx)] = station_to_fold[station_id]
        station_to_indices.setdefault(station_id, []).append(int(dataset_idx))

    if len(fold_by_index) != len(cv_indices):
        raise RuntimeError(
            f"fold映射未完整覆盖CV池: {len(fold_by_index)}/{len(cv_indices)}"
        )

    records: list[dict[str, Any]] = []
    all_stations = set(station_to_indices)
    for fold in sorted(set(station_to_fold.values())):
        test_stations = sorted(
            station for station in all_stations if station_to_fold[station] == fold
        )
        train_stations = sorted(all_stations - set(test_stations))
        test_indices = [
            idx for station in test_stations for idx in station_to_indices[station]
        ]
        train_indices = [
            idx for station in train_stations for idx in station_to_indices[station]
        ]
        records.append(
            {
                "fold": int(fold),
                "n_train_stations": len(train_stations),
                "n_test_stations": len(test_stations),
                "n_train_samples": len(train_indices),
                "n_test_samples": len(test_indices),
                "train_stations": train_stations,
                "test_stations": test_stations,
                "train_indices": train_indices,
                "test_indices": test_indices,
            }
        )

    covered = [idx for record in records for idx in record["test_indices"]]
    if sorted(covered) != sorted(int(i) for i in cv_indices):
        raise RuntimeError("十折test_indices没有恰好覆盖CV池一次")

    return fold_by_index, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/validate deterministic balanced station 10-fold manifest"
    )
    parser.add_argument(
        "--station-csv",
        type=Path,
        default=Path(
            "/root/autodl-tmp/shared_cache/progressive_finetune/"
            "internal_progressive_station.csv"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--high-threshold-mm", type=float, default=80.0)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--include-fixed-test",
        action="store_true",
        help=(
            "将旧split=test内部1000条并回Nested CV池；"
            "不修改原CSV，仅改变本次fold统计范围"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, summary, metadata = create_or_load_manifest(
        station_csv=args.station_csv,
        manifest_path=args.output,
        n_splits=args.n_splits,
        high_threshold_mm=args.high_threshold_mm,
        force_rebuild=args.force_rebuild,
        include_fixed_test=args.include_fixed_test,
    )

    print("=" * 88)
    print("确定性平衡站点级10折")
    print("=" * 88)
    print(f"状态: {metadata['manifest_status']}")
    print(f"方法: {metadata['method']}")
    print(f"随机: {metadata['randomized']}")
    print(f"CV样本: {metadata['n_cv_samples']}")
    print(f"CV站点: {metadata['n_cv_stations']}")
    print(f"manifest: {metadata['manifest']}")
    print("\n折间组成:")
    print(summary.to_string(index=False))
    print("\n范围:")
    for column in ["n_stations", "n_samples", "n_high_swe"]:
        values = summary[column]
        print(
            f"  {column}: min={int(values.min())}, "
            f"max={int(values.max())}, range={int(values.max()-values.min())}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
