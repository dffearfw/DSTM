#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/autodl-tmp")
CACHE_DIR = ROOT / "shared_cache"
MANIFEST_DIR = CACHE_DIR / "progressive_finetune"

STATION_FILE = MANIFEST_DIR / "internal_progressive_station.csv"
NORMALIZATION_FILE = CACHE_DIR / "progressive_pretrain_normalization.json"

EXPECTED_CACHE = (
    CACHE_DIR
    / "station_dataset_features_"
      "shared_station_features_"
      "full_daily_wind_only_fill_2015_2018_v4.pkl"
)

OUTPUT_DIR = ROOT / "diagnostics" / "v4_full_input_hash_audit"

LEGACY_GROUP50_DATES = {
    "2015-01-02",
    "2015-01-03",
    "2015-01-05",
}
LEGACY_GROUP50_ROW = 254
LEGACY_GROUP50_COL = 125


def sha256_parts(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "little", signed=False))
        h.update(part)
    return h.hexdigest()


def tensor_to_array(value: Any, decimals: int | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    arr = np.asarray(value, dtype=np.float32)

    if decimals is not None:
        arr = np.round(arr, decimals=decimals).astype(
            np.float32,
            copy=False,
        )

    # 固定字节序和连续布局，保证哈希可复现
    arr = np.ascontiguousarray(arr.astype("<f4", copy=False))
    return arr


def array_hash(arr: np.ndarray) -> str:
    shape_bytes = np.asarray(arr.shape, dtype="<i8").tobytes()
    return sha256_parts(shape_bytes, arr.tobytes())


def full_hash(conv: np.ndarray, point: np.ndarray) -> str:
    return sha256_parts(
        np.asarray(conv.shape, dtype="<i8").tobytes(),
        conv.tobytes(),
        np.asarray(point.shape, dtype="<i8").tobytes(),
        point.tobytes(),
    )


def normalize_date(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def normalize_station_id(value: Any) -> str:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return ",".join(str(x) for x in value)
    return str(value)


def choose_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def build_source_lookup(source: pd.DataFrame):
    station_col = choose_column(
        source,
        ["station_id", "站点ID", "station", "Station_ID"],
    )
    date_col = choose_column(
        source,
        ["date", "日期", "datetime", "label_date"],
    )
    split_col = choose_column(source, ["split", "Split"])
    target_col = choose_column(
        source,
        ["swe", "SWE", "target_mm", "站点SWE_raw"],
    )
    row_col = choose_column(source, ["row", "行列号_row"])
    col_col = choose_column(source, ["col", "行列号_col"])

    if station_col is None or date_col is None:
        raise RuntimeError(
            "源CSV缺少station_id或date列："
            f"columns={source.columns.tolist()}"
        )

    source = source.copy()
    source["_station_key"] = source[station_col].map(normalize_station_id)
    source["_date_key"] = pd.to_datetime(
        source[date_col],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    exact_lookup = {}
    station_date_lookup = {}

    for _, row in source.iterrows():
        station = row["_station_key"]
        date = row["_date_key"]

        payload = {
            "split": (
                str(row[split_col])
                if split_col is not None
                else "unknown"
            ),
            "source_target_mm": (
                float(row[target_col])
                if target_col is not None
                and pd.notna(row[target_col])
                else np.nan
            ),
        }

        station_date_lookup[(station, date)] = payload

        if (
            row_col is not None
            and col_col is not None
            and pd.notna(row[row_col])
            and pd.notna(row[col_col])
        ):
            exact_lookup[
                (
                    station,
                    date,
                    int(row[row_col]),
                    int(row[col_col]),
                )
            ] = payload

    return exact_lookup, station_date_lookup


def make_group_tables(
    frame: pd.DataFrame,
    hash_column: str,
    prefix: str,
):
    sizes = frame.groupby(hash_column, sort=False).size()
    duplicate_hashes = sizes[sizes > 1].index

    members = frame[
        frame[hash_column].isin(duplicate_hashes)
    ].copy()

    if members.empty:
        members["duplicate_group_id"] = pd.Series(dtype="int64")
        groups = pd.DataFrame(
            columns=[
                "duplicate_group_id",
                hash_column,
                "member_count",
                "target_min_mm",
                "target_max_mm",
                "target_range_mm",
                "date_count",
                "station_count",
                "splits",
                "cross_split",
            ]
        )
        return members, groups

    target_ranges = (
        members.groupby(hash_column)["target_mm"]
        .agg(["min", "max"])
    )
    target_ranges["range"] = target_ranges["max"] - target_ranges["min"]

    ordered_hashes = (
        target_ranges.sort_values(
            ["range", "max"],
            ascending=False,
        )
        .index.tolist()
    )

    id_map = {
        hash_value: group_id
        for group_id, hash_value in enumerate(ordered_hashes, start=1)
    }

    members["duplicate_group_id"] = members[hash_column].map(id_map)
    members = members.sort_values(
        ["duplicate_group_id", "label_date", "target_mm"]
    )

    rows = []
    for hash_value, group in members.groupby(hash_column, sort=False):
        splits = sorted(set(group["split"].astype(str)))
        rows.append(
            {
                "duplicate_group_id": id_map[hash_value],
                hash_column: hash_value,
                "member_count": int(len(group)),
                "target_min_mm": float(group["target_mm"].min()),
                "target_max_mm": float(group["target_mm"].max()),
                "target_range_mm": float(
                    group["target_mm"].max()
                    - group["target_mm"].min()
                ),
                "date_count": int(group["label_date"].nunique()),
                "station_count": int(group["station_id"].nunique()),
                "splits": "|".join(splits),
                "cross_split": len(splits) > 1,
            }
        )

    groups = pd.DataFrame(rows).sort_values(
        ["target_range_mm", "member_count"],
        ascending=False,
    )

    members.to_csv(
        OUTPUT_DIR / f"{prefix}_duplicate_members.csv",
        index=False,
        encoding="utf-8-sig",
    )
    groups.to_csv(
        OUTPUT_DIR / f"{prefix}_duplicate_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return members, groups


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    torch.set_num_threads(1)

    for required in [
        STATION_FILE,
        NORMALIZATION_FILE,
        EXPECTED_CACHE,
        ROOT / "data_station_online_swe.py",
    ]:
        if not required.is_file():
            raise FileNotFoundError(f"缺少必要文件：{required}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("v4完整模型输入哈希审计")
    print("=" * 100)
    print(f"站点文件：{STATION_FILE}")
    print(f"v4缓存：  {EXPECTED_CACHE}")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 100)

    sys.path.insert(0, str(ROOT))
    from data_station_online_swe import StationSWEDataset

    source = pd.read_csv(STATION_FILE)
    exact_lookup, station_date_lookup = build_source_lookup(source)

    dataset = StationSWEDataset(
        station_csv=STATION_FILE,
        year_target=[2015, 2016, 2017, 2018],
        fine_tune_mode=True,
        load_fused_swe=True,
        coordinate_jitter_std=0.0,
        microwave_noise_std=0.0,
        coordinate_mask_prob=0.0,
        use_tta=False,
        cache_dir=CACHE_DIR,
        shared_cache_mode=True,
        use_product_correction=False,
        normalization_config_path=NORMALIZATION_FILE,
        normalization_mode="load",
        fixed_label_min_mm=0.0,
        fixed_label_max_mm=400.0,
        force_reload=False,
        verbose_point_debug=False,
        microwave_warning_print_limit=0,
        point_stats_interval=0,
    )

    if hasattr(dataset, "set_augmentation_mode"):
        dataset.set_augmentation_mode(False)
    elif hasattr(dataset, "current_augment"):
        dataset.current_augment = False

    print(f"\nDataset样本数：{len(dataset)}")

    if len(dataset) != len(source):
        print(
            "⚠ Dataset与源CSV行数不同："
            f"dataset={len(dataset)}, csv={len(source)}"
        )

    rows = []
    redirected_count = 0

    with torch.inference_mode():
        for requested_idx in range(len(dataset)):
            sample = dataset[requested_idx]

            if not isinstance(sample, (tuple, list)) or len(sample) < 2:
                raise RuntimeError(
                    f"样本返回格式异常：idx={requested_idx}, "
                    f"type={type(sample)}, value={sample}"
                )

            conv_t = sample[0]
            point_t = sample[1]

            if len(sample) >= 6:
                cur_idx = sample[5]
                if torch.is_tensor(cur_idx):
                    cur_idx = int(cur_idx.item())
                else:
                    cur_idx = int(cur_idx)
            else:
                cur_idx = requested_idx

            if cur_idx != requested_idx:
                redirected_count += 1

            meta = dataset.meta_index[cur_idx]

            label_date = normalize_date(
                meta.get(
                    "label_date",
                    meta.get("feature_date", meta.get("date")),
                )
            )
            feature_date = normalize_date(
                meta.get(
                    "feature_date",
                    meta.get("label_date", meta.get("date")),
                )
            )

            station_id = normalize_station_id(
                meta.get("station_id", "unknown")
            )
            row_idx = int(meta["row"])
            col_idx = int(meta["col"])
            target_mm = float(meta["swe"])
            day_gap = int(meta.get("day_gap", 0))

            payload = exact_lookup.get(
                (station_id, label_date, row_idx, col_idx)
            )
            if payload is None:
                payload = station_date_lookup.get(
                    (station_id, label_date),
                    {
                        "split": "unknown",
                        "source_target_mm": np.nan,
                    },
                )

            conv_exact = tensor_to_array(conv_t)
            point_exact = tensor_to_array(point_t)

            conv_round6 = tensor_to_array(conv_t, decimals=6)
            point_round6 = tensor_to_array(point_t, decimals=6)

            rows.append(
                {
                    "requested_index": requested_idx,
                    "returned_index": cur_idx,
                    "station_id": station_id,
                    "split": payload["split"],
                    "label_date": label_date,
                    "feature_date": feature_date,
                    "day_gap": day_gap,
                    "row": row_idx,
                    "col": col_idx,
                    "target_mm": target_mm,
                    "source_target_mm": payload["source_target_mm"],
                    "conv_hash_exact": array_hash(conv_exact),
                    "point_hash_exact": array_hash(point_exact),
                    "full_hash_exact": full_hash(
                        conv_exact,
                        point_exact,
                    ),
                    "full_hash_round6": full_hash(
                        conv_round6,
                        point_round6,
                    ),
                }
            )

            if (requested_idx + 1) % 500 == 0:
                print(
                    f"已完成 {requested_idx + 1:,}/"
                    f"{len(dataset):,}"
                )

    frame = pd.DataFrame(rows)

    frame.to_csv(
        OUTPUT_DIR / "all_sample_input_hashes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    exact_members, exact_groups = make_group_tables(
        frame,
        "full_hash_exact",
        "full_exact",
    )
    rounded_members, rounded_groups = make_group_tables(
        frame,
        "full_hash_round6",
        "full_round6",
    )
    conv_members, conv_groups = make_group_tables(
        frame,
        "conv_hash_exact",
        "conv_exact",
    )
    point_members, point_groups = make_group_tables(
        frame,
        "point_hash_exact",
        "point_exact",
    )

    group50 = frame[
        (frame["row"] == LEGACY_GROUP50_ROW)
        & (frame["col"] == LEGACY_GROUP50_COL)
        & (frame["label_date"].isin(LEGACY_GROUP50_DATES))
    ].copy()

    group50.to_csv(
        OUTPUT_DIR / "legacy_group50_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group50_unique_full_hashes = int(
        group50["full_hash_exact"].nunique()
    )
    group50_still_collapsed = (
        len(group50) == 3
        and group50_unique_full_hashes == 1
    )

    feature_date_mismatch_count = int(
        (frame["feature_date"] != frame["label_date"]).sum()
    )
    nonzero_day_gap_count = int((frame["day_gap"] != 0).sum())

    summary = {
        "dataset_size": int(len(frame)),
        "source_csv_size": int(len(source)),
        "redirected_getitem_count": int(redirected_count),
        "feature_date_mismatch_count": feature_date_mismatch_count,
        "nonzero_day_gap_count": nonzero_day_gap_count,
        "full_exact_duplicate_group_count": int(len(exact_groups)),
        "full_exact_duplicate_member_count": int(len(exact_members)),
        "full_exact_cross_split_group_count": (
            int(exact_groups["cross_split"].sum())
            if not exact_groups.empty
            else 0
        ),
        "full_round6_duplicate_group_count": int(len(rounded_groups)),
        "full_round6_duplicate_member_count": int(len(rounded_members)),
        "conv_exact_duplicate_group_count": int(len(conv_groups)),
        "conv_exact_duplicate_member_count": int(len(conv_members)),
        "point_exact_duplicate_group_count": int(len(point_groups)),
        "point_exact_duplicate_member_count": int(len(point_members)),
        "legacy_group50_record_count": int(len(group50)),
        "legacy_group50_unique_full_hashes": group50_unique_full_hashes,
        "legacy_group50_still_collapsed": bool(
            group50_still_collapsed
        ),
        "old_55_groups_117_members_definitively_gone": bool(
            len(exact_groups) == 0
            and len(exact_members) == 0
            and not group50_still_collapsed
        ),
    }

    (OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("审计结果")
    print("=" * 100)
    print(f"总样本：{summary['dataset_size']}")
    print(
        "feature_date != label_date："
        f"{summary['feature_date_mismatch_count']}"
    )
    print(
        "day_gap != 0："
        f"{summary['nonzero_day_gap_count']}"
    )
    print(
        "完整输入精确重复："
        f"{summary['full_exact_duplicate_group_count']}组，"
        f"{summary['full_exact_duplicate_member_count']}条"
    )
    print(
        "完整输入round6重复："
        f"{summary['full_round6_duplicate_group_count']}组，"
        f"{summary['full_round6_duplicate_member_count']}条"
    )
    print(
        "跨split完整输入重复组："
        f"{summary['full_exact_cross_split_group_count']}"
    )
    print(
        "旧Group 50："
        f"找到{summary['legacy_group50_record_count']}条，"
        f"唯一完整哈希数="
        f"{summary['legacy_group50_unique_full_hashes']}，"
        f"是否仍折叠="
        f"{summary['legacy_group50_still_collapsed']}"
    )
    print(
        "旧55组/117条是否已明确归零："
        f"{summary['old_55_groups_117_members_definitively_gone']}"
    )
    print(f"完整结果：{OUTPUT_DIR}")
    print("=" * 100)

    del dataset
    gc.collect()


if __name__ == "__main__":
    main()
