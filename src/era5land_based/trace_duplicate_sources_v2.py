# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
import pandas as pd


DUPLICATE_FILE = Path(
    "/root/autodl-tmp/diagnostics/"
    "fold01_true_split_and_duplicates/"
    "duplicate_input_members.csv"
)

SOURCE_CSV = Path(
    "/root/autodl-tmp/shared_cache/"
    "progressive_finetune/"
    "internal_progressive_station.csv"
)

OUTPUT_DIR = Path(
    "/root/autodl-tmp/diagnostics/"
    "trace_duplicate_sources_v2"
)


def read_csv(path):
    for encoding in [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
    ]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"无法读取：{path}")


def normalize_date(series):
    return pd.to_datetime(
        series,
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")


def unique_text(series):
    result = []

    for value in series:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if text and text not in result:
            result.append(text)

    return result


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicates = read_csv(DUPLICATE_FILE)
    source = read_csv(SOURCE_CSV)

    duplicates["dataset_index"] = pd.to_numeric(
        duplicates["dataset_index"],
        errors="raise",
    ).astype(int)

    source["split_normalized"] = (
        source["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # 固定独立测试集的局部数据池
    test_pool = (
        source[
            source["split_normalized"] == "test"
        ]
        .drop(columns=["split_normalized"])
        .reset_index()
        .rename(
            columns={
                "index": "source_csv_global_index"
            }
        )
    )

    # 交叉验证池的局部数据池
    cv_pool = (
        source[
            source["split_normalized"] != "test"
        ]
        .drop(columns=["split_normalized"])
        .reset_index()
        .rename(
            columns={
                "index": "source_csv_global_index"
            }
        )
    )

    print("=" * 100)
    print("数据池")
    print("=" * 100)
    print(f"完整CSV：{len(source)}")
    print(f"固定Test池：{len(test_pool)}")
    print(f"CV池：{len(cv_pool)}")

    if len(test_pool) != 1000:
        raise ValueError(
            f"Test池应为1000，实际为{len(test_pool)}"
        )

    if len(cv_pool) != 6936:
        raise ValueError(
            f"CV池应为6936，实际为{len(cv_pool)}"
        )

    traced_rows = []

    for _, diag in duplicates.iterrows():
        split = str(diag["split"])
        local_index = int(diag["dataset_index"])

        if split == "TEST_FIXED":
            pool_name = "TEST_POOL"
            pool = test_pool

        elif split in {
            "TRAIN_TRUE_FOLD",
            "VAL_TRUE_FOLD",
        }:
            pool_name = "CV_POOL"
            pool = cv_pool

        else:
            raise ValueError(
                f"未知诊断划分：{split}"
            )

        if not 0 <= local_index < len(pool):
            raise IndexError(
                f"{split}中的dataset_index越界："
                f"{local_index}, pool_size={len(pool)}"
            )

        src = pool.iloc[local_index]

        row = {
            # 诊断字段
            "duplicate_group_id":
                int(diag["duplicate_group_id"]),
            "record_id":
                int(diag["record_id"]),
            "diagnostic_split":
                split,
            "diagnostic_split_position":
                int(diag["split_position"]),
            "diagnostic_dataset_index":
                local_index,
            "diagnostic_station_id":
                diag["station_id"],
            "diagnostic_date":
                diag["date"],
            "diagnostic_row":
                diag["row"],
            "diagnostic_col":
                diag["col"],
            "diagnostic_target_mm":
                diag["target_mm"],
            "model_prediction_mm":
                diag["model_prediction_mm"],
            "era5_land_mm":
                diag["era5_land_mm"],
            "full_hash_exact":
                diag["full_hash_exact"],

            # 正确回查信息
            "source_pool":
                pool_name,
            "source_csv_global_index":
                int(src["source_csv_global_index"]),
        }

        for column in source.columns:
            if column == "split_normalized":
                continue

            row[f"source__{column}"] = src[column]

        traced_rows.append(row)

    trace = pd.DataFrame(traced_rows)

    # -------------------------------------------------
    # 映射验证
    # -------------------------------------------------
    trace["station_match"] = (
        trace["diagnostic_station_id"]
        .astype(str)
        .str.strip()
        ==
        trace["source__station_id"]
        .astype(str)
        .str.strip()
    )

    trace["row_match"] = np.isclose(
        pd.to_numeric(
            trace["diagnostic_row"],
            errors="coerce",
        ),
        pd.to_numeric(
            trace["source__row"],
            errors="coerce",
        ),
        atol=0,
        rtol=0,
        equal_nan=True,
    )

    trace["col_match"] = np.isclose(
        pd.to_numeric(
            trace["diagnostic_col"],
            errors="coerce",
        ),
        pd.to_numeric(
            trace["source__col"],
            errors="coerce",
        ),
        atol=0,
        rtol=0,
        equal_nan=True,
    )

    trace["target_match"] = np.isclose(
        pd.to_numeric(
            trace["diagnostic_target_mm"],
            errors="coerce",
        ),
        pd.to_numeric(
            trace["source__swe"],
            errors="coerce",
        ),
        atol=1e-3,
        rtol=0,
        equal_nan=True,
    )

    trace["date_match"] = (
        normalize_date(
            trace["diagnostic_date"]
        )
        ==
        normalize_date(
            trace["source__date"]
        )
    )

    core_match_columns = [
        "station_match",
        "row_match",
        "col_match",
        "target_match",
    ]

    trace["core_mapping_match"] = trace[
        core_match_columns
    ].all(axis=1)

    match_rates = {
        column: float(trace[column].mean())
        for column in [
            "station_match",
            "row_match",
            "col_match",
            "target_match",
            "date_match",
            "core_mapping_match",
        ]
    }

    trace.to_csv(
        OUTPUT_DIR
        / "corrected_duplicate_source_trace_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------
    # 按完整输入重复组汇总
    # -------------------------------------------------
    group_rows = []

    for group_id, group in trace.groupby(
        "duplicate_group_id",
        sort=True,
    ):
        source_station_ids = unique_text(
            group["source__source_station_ids"]
        )

        source_dates = unique_text(
            normalize_date(
                group["source__date"]
            )
        )

        synthetic_station_ids = unique_text(
            group["source__station_id"]
        )

        source_swe = pd.to_numeric(
            group["source__swe"],
            errors="coerce",
        )

        target_min = float(source_swe.min())
        target_max = float(source_swe.max())
        target_range = target_max - target_min

        n_original_stations = len(
            source_station_ids
        )

        n_dates = len(source_dates)

        if (
            n_original_stations > 1
            and n_dates == 1
        ):
            classification = (
                "MULTIPLE_STATIONS_"
                "SAME_GRID_SAME_DATE"
            )

        elif (
            n_original_stations == 1
            and n_dates > 1
        ):
            classification = (
                "SAME_STATION_MULTIPLE_DATES_"
                "IDENTICAL_MODEL_INPUT"
            )

        elif (
            n_original_stations > 1
            and n_dates > 1
        ):
            classification = (
                "MULTIPLE_STATIONS_MULTIPLE_DATES_"
                "IDENTICAL_MODEL_INPUT"
            )

        elif (
            n_original_stations == 1
            and n_dates == 1
            and target_range > 1.0
        ):
            classification = (
                "SAME_STATION_SAME_DATE_"
                "CONFLICTING_SWE"
            )

        else:
            classification = (
                "EXACT_OR_NEAR_DUPLICATE"
            )

        group_rows.append({
            "duplicate_group_id":
                int(group_id),
            "member_count":
                int(len(group)),
            "diagnostic_splits":
                "|".join(
                    unique_text(
                        group[
                            "diagnostic_split"
                        ]
                    )
                ),
            "synthetic_station_ids":
                "|".join(
                    synthetic_station_ids
                ),
            "source_dates":
                "|".join(source_dates),
            "source_station_ids":
                "|".join(
                    source_station_ids
                ),
            "n_original_stations":
                n_original_stations,
            "n_source_dates":
                n_dates,
            "source_swe_min_mm":
                target_min,
            "source_swe_max_mm":
                target_max,
            "source_swe_range_mm":
                target_range,
            "all_core_mapping_match":
                bool(
                    group[
                        "core_mapping_match"
                    ].all()
                ),
            "all_date_match":
                bool(
                    group[
                        "date_match"
                    ].all()
                ),
            "classification":
                classification,
            "network_status":
                "NOT_PRESERVED_IN_INTERNAL_CSV",
        })

    group_summary = (
        pd.DataFrame(group_rows)
        .sort_values(
            [
                "source_swe_range_mm",
                "duplicate_group_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    group_summary.to_csv(
        OUTPUT_DIR
        / "corrected_duplicate_source_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    classification_counts = (
        group_summary[
            "classification"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "source_csv_rows":
            int(len(source)),
        "test_pool_rows":
            int(len(test_pool)),
        "cv_pool_rows":
            int(len(cv_pool)),
        "duplicate_groups":
            int(
                trace[
                    "duplicate_group_id"
                ].nunique()
            ),
        "duplicate_members":
            int(len(trace)),
        "mapping_match_rates":
            match_rates,
        "core_mapping_verified":
            bool(
                trace[
                    "core_mapping_match"
                ].all()
            ),
        "diagnostic_dates_verified":
            bool(
                trace[
                    "date_match"
                ].all()
            ),
        "classification_counts": {
            str(key): int(value)
            for key, value
            in classification_counts.items()
        },
        "network_status":
            (
                "internal_progressive_station.csv "
                "does not contain a network field"
            ),
    }

    with open(
        OUTPUT_DIR
        / "corrected_trace_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("正确映射验证")
    print("=" * 100)

    for key, value in match_rates.items():
        print(f"{key}: {value:.4%}")

    print("\n分类统计：")

    for key, value in classification_counts.items():
        print(f"  {key}: {value}")

    print("\n冲突最大的前15组：")

    print(
        group_summary[
            [
                "duplicate_group_id",
                "member_count",
                "diagnostic_splits",
                "synthetic_station_ids",
                "source_dates",
                "source_station_ids",
                "n_original_stations",
                "n_source_dates",
                "source_swe_min_mm",
                "source_swe_max_mm",
                "source_swe_range_mm",
                "all_date_match",
                "classification",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\n输出目录：")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
