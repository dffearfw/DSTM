# -*- coding: utf-8 -*-

import json
import re
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
    "trace_duplicate_sources"
)


def read_csv_robust(path):
    errors = []

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "latin1",
    ]:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )
            return df, encoding

        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"无法读取CSV：{path}\n"
        + "\n".join(errors)
    )


def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(name).lower(),
    )


def find_columns(columns, keywords):
    results = []

    for column in columns:
        normalized = normalize_name(column)

        if any(
            keyword in normalized
            for keyword in keywords
        ):
            results.append(column)

    return results


def unique_values(series):
    values = []

    for value in series:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if not text:
            continue

        if text.lower() in {
            "nan",
            "none",
            "null",
            "na",
        }:
            continue

        if text not in values:
            values.append(text)

    return values


def is_synthetic_station_column(df, column):
    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    if len(values) == 0:
        return False

    pattern = re.compile(
        r"^(INTGRID|EXTGRID)_R\d+_C\d+$",
        re.IGNORECASE,
    )

    fraction = values.map(
        lambda value: bool(pattern.match(value))
    ).mean()

    return fraction >= 0.5


def compare_rate(source, diagnostic, kind):
    if kind == "date":
        source_value = pd.to_datetime(
            source,
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        diagnostic_value = pd.to_datetime(
            diagnostic,
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

        valid = (
            source_value.notna()
            & diagnostic_value.notna()
        )

        if valid.sum() == 0:
            return np.nan, 0

        rate = (
            source_value[valid]
            == diagnostic_value[valid]
        ).mean()

        return float(rate), int(valid.sum())

    if kind in {
        "row",
        "col",
        "target",
    }:
        source_value = pd.to_numeric(
            source,
            errors="coerce",
        )

        diagnostic_value = pd.to_numeric(
            diagnostic,
            errors="coerce",
        )

        valid = (
            source_value.notna()
            & diagnostic_value.notna()
        )

        if valid.sum() == 0:
            return np.nan, 0

        tolerance = (
            1e-3
            if kind == "target"
            else 1e-6
        )

        equal = np.isclose(
            source_value[valid].to_numpy(float),
            diagnostic_value[valid].to_numpy(float),
            atol=tolerance,
            rtol=0.0,
        )

        return float(equal.mean()), int(valid.sum())

    source_value = (
        source
        .astype(str)
        .str.strip()
        .str.lower()
    )

    diagnostic_value = (
        diagnostic
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid = (
        source.notna()
        & diagnostic.notna()
    )

    if valid.sum() == 0:
        return np.nan, 0

    rate = (
        source_value[valid]
        == diagnostic_value[valid]
    ).mean()

    return float(rate), int(valid.sum())


def values_by_column(group, columns):
    result = {}

    for column in columns:
        trace_column = f"src__{column}"

        if trace_column not in group.columns:
            continue

        result[column] = unique_values(
            group[trace_column]
        )

    return result


def max_unique_count(values_dict):
    if not values_dict:
        return 0

    return max(
        len(values)
        for values in values_dict.values()
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicate_df, duplicate_encoding = (
        read_csv_robust(DUPLICATE_FILE)
    )

    source_df, source_encoding = (
        read_csv_robust(SOURCE_CSV)
    )

    required = {
        "duplicate_group_id",
        "record_id",
        "split",
        "dataset_index",
        "station_id",
        "date",
        "row",
        "col",
        "target_mm",
        "model_prediction_mm",
        "era5_land_mm",
    }

    missing = sorted(
        required - set(duplicate_df.columns)
    )

    if missing:
        raise ValueError(
            "duplicate_input_members.csv "
            f"缺少字段：{missing}"
        )

    duplicate_df["dataset_index"] = (
        pd.to_numeric(
            duplicate_df["dataset_index"],
            errors="raise",
        )
        .astype(int)
    )

    minimum_index = int(
        duplicate_df["dataset_index"].min()
    )

    maximum_index = int(
        duplicate_df["dataset_index"].max()
    )

    print("=" * 100)
    print("原始CSV基本信息")
    print("=" * 100)
    print(f"原始CSV行数：{len(source_df)}")
    print(f"原始CSV列数：{len(source_df.columns)}")
    print(
        "重复记录dataset_index范围："
        f"{minimum_index}—{maximum_index}"
    )

    if minimum_index < 0:
        raise IndexError(
            f"存在负dataset_index：{minimum_index}"
        )

    if maximum_index >= len(source_df):
        raise IndexError(
            "dataset_index超出原始CSV行数："
            f"max_index={maximum_index}, "
            f"CSV_rows={len(source_df)}"
        )

    # dataset_index按CSV位置直接回查原始记录
    positions = (
        duplicate_df["dataset_index"]
        .to_numpy(int)
    )

    selected_source = (
        source_df
        .iloc[positions]
        .reset_index()
        .rename(
            columns={
                "index": "source_csv_index_label"
            }
        )
    )

    diagnostic_part = (
        duplicate_df
        .reset_index(drop=True)
        .add_prefix("diag__")
    )

    source_part = (
        selected_source
        .reset_index(drop=True)
        .add_prefix("src__")
    )

    trace_df = pd.concat(
        [
            diagnostic_part,
            source_part,
        ],
        axis=1,
    )

    trace_df.insert(
        len(diagnostic_part.columns),
        "source_csv_row_position",
        positions,
    )

    columns = list(source_df.columns)

    station_candidates = find_columns(
        columns,
        [
            "stationid",
            "stationcode",
            "stationname",
            "station",
            "siteid",
            "sitecode",
            "sitename",
            "site",
            "wmo",
            "ismn",
            "snotel",
            "snowcourse",
            "snowpillow",
        ],
    )

    network_candidates = find_columns(
        columns,
        [
            "network",
            "obsnetwork",
            "stationnetwork",
            "sitenetwork",
        ],
    )

    source_candidates = find_columns(
        columns,
        [
            "source",
            "dataset",
            "provider",
            "agency",
            "project",
            "database",
            "archive",
            "origin",
        ],
    )

    date_candidates = find_columns(
        columns,
        [
            "date",
            "datetime",
            "timestamp",
            "obstime",
            "observationtime",
        ],
    )

    row_candidates = find_columns(
        columns,
        [
            "gridrow",
            "rasterrow",
            "era5row",
        ],
    )

    col_candidates = find_columns(
        columns,
        [
            "gridcol",
            "rastercol",
            "era5col",
            "column",
        ],
    )

    target_candidates = find_columns(
        columns,
        [
            "swemm",
            "observedswe",
            "obsswe",
            "targetswe",
            "snowwaterequivalent",
            "swe",
            "target",
            "label",
        ],
    )

    # 精确名称补充
    for column in columns:
        normalized = normalize_name(column)

        if (
            normalized == "row"
            and column not in row_candidates
        ):
            row_candidates.append(column)

        if (
            normalized in {"col", "c"}
            and column not in col_candidates
        ):
            col_candidates.append(column)

        if (
            normalized in {
                "date",
                "time",
                "datetime",
            }
            and column not in date_candidates
        ):
            date_candidates.append(column)

    synthetic_station_columns = [
        column
        for column in station_candidates
        if is_synthetic_station_column(
            source_df,
            column,
        )
    ]

    original_station_candidates = [
        column
        for column in station_candidates
        if column not in synthetic_station_columns
    ]

    candidate_information = {
        "station_candidates":
            station_candidates,
        "synthetic_station_columns":
            synthetic_station_columns,
        "original_station_candidates":
            original_station_candidates,
        "network_candidates":
            network_candidates,
        "source_candidates":
            source_candidates,
        "date_candidates":
            date_candidates,
        "row_candidates":
            row_candidates,
        "col_candidates":
            col_candidates,
        "target_candidates":
            target_candidates,
    }

    with open(
        OUTPUT_DIR
        / "source_column_candidates.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            candidate_information,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 验证dataset_index是否确实映射到了正确CSV行
    verification_rows = []

    verification_groups = {
        "station": (
            station_candidates,
            "diag__station_id",
        ),
        "date": (
            date_candidates,
            "diag__date",
        ),
        "row": (
            row_candidates,
            "diag__row",
        ),
        "col": (
            col_candidates,
            "diag__col",
        ),
        "target": (
            target_candidates,
            "diag__target_mm",
        ),
    }

    for kind, (
        candidate_columns,
        diagnostic_column,
    ) in verification_groups.items():

        if diagnostic_column not in trace_df.columns:
            continue

        for source_column in candidate_columns:
            trace_source_column = (
                f"src__{source_column}"
            )

            rate, compared_n = compare_rate(
                trace_df[trace_source_column],
                trace_df[diagnostic_column],
                kind,
            )

            verification_rows.append({
                "field_type": kind,
                "source_column": source_column,
                "match_rate": rate,
                "compared_N": compared_n,
            })

    verification_df = pd.DataFrame(
        verification_rows
    )

    verification_df.to_csv(
        OUTPUT_DIR
        / "dataset_index_mapping_verification.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_rates = {}

    if len(verification_df) > 0:
        for field_type, group in (
            verification_df
            .dropna(subset=["match_rate"])
            .groupby("field_type")
        ):
            best_row = group.sort_values(
                [
                    "match_rate",
                    "compared_N",
                ],
                ascending=False,
            ).iloc[0]

            best_rates[field_type] = {
                "source_column":
                    best_row["source_column"],
                "match_rate":
                    float(best_row["match_rate"]),
                "compared_N":
                    int(best_row["compared_N"]),
            }

    critical_rates = [
        information["match_rate"]
        for field, information
        in best_rates.items()
        if field in {
            "date",
            "row",
            "col",
            "target",
        }
    ]

    if critical_rates:
        mapping_verified = (
            float(np.mean(critical_rates))
            >= 0.95
        )
    else:
        mapping_verified = False

    trace_df["dataset_index_mapping_verified"] = (
        mapping_verified
    )

    trace_df.to_csv(
        OUTPUT_DIR
        / "duplicate_source_trace_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 每个重复组汇总原始站点、网络和来源
    group_rows = []

    for group_id, group in trace_df.groupby(
        "diag__duplicate_group_id",
        sort=True,
    ):
        original_station_values = (
            values_by_column(
                group,
                original_station_candidates,
            )
        )

        network_values = values_by_column(
            group,
            network_candidates,
        )

        source_values = values_by_column(
            group,
            source_candidates,
        )

        original_station_max_count = (
            max_unique_count(
                original_station_values
            )
        )

        network_max_count = (
            max_unique_count(
                network_values
            )
        )

        targets = pd.to_numeric(
            group["diag__target_mm"],
            errors="coerce",
        )

        target_min = float(targets.min())
        target_max = float(targets.max())
        target_range = target_max - target_min

        if (
            original_station_max_count > 1
            or network_max_count > 1
        ):
            classification = (
                "MULTIPLE_ORIGINAL_STATIONS_"
                "OR_NETWORKS_COLLAPSED"
            )

        elif (
            not original_station_candidates
            and not network_candidates
        ):
            classification = (
                "IDENTITY_FIELDS_NOT_FOUND"
            )

        elif target_range > 1.0:
            classification = (
                "SAME_SOURCE_IDENTITY_"
                "CONFLICTING_TARGETS"
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
            "splits":
                "|".join(
                    sorted(
                        unique_values(
                            group["diag__split"]
                        )
                    )
                ),
            "synthetic_station_ids":
                "|".join(
                    unique_values(
                        group[
                            "diag__station_id"
                        ]
                    )
                ),
            "dates":
                "|".join(
                    unique_values(
                        group["diag__date"]
                    )
                ),
            "target_min_mm":
                target_min,
            "target_max_mm":
                target_max,
            "target_range_mm":
                target_range,
            "original_station_values":
                json.dumps(
                    original_station_values,
                    ensure_ascii=False,
                ),
            "network_values":
                json.dumps(
                    network_values,
                    ensure_ascii=False,
                ),
            "source_values":
                json.dumps(
                    source_values,
                    ensure_ascii=False,
                ),
            "classification":
                classification,
        })

    group_df = (
        pd.DataFrame(group_rows)
        .sort_values(
            [
                "target_range_mm",
                "duplicate_group_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    group_df.to_csv(
        OUTPUT_DIR
        / "duplicate_source_group_identity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    column_df = pd.DataFrame({
        "column_position":
            range(len(source_df.columns)),
        "column_name":
            source_df.columns,
        "dtype": [
            str(source_df[column].dtype)
            for column in source_df.columns
        ],
        "non_null_count": [
            int(
                source_df[column]
                .notna()
                .sum()
            )
            for column in source_df.columns
        ],
        "unique_count": [
            int(
                source_df[column]
                .nunique(dropna=True)
            )
            for column in source_df.columns
        ],
    })

    column_df.to_csv(
        OUTPUT_DIR
        / "source_csv_columns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    classification_counts = (
        group_df["classification"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "source_csv":
            str(SOURCE_CSV),
        "source_csv_encoding":
            source_encoding,
        "source_csv_rows":
            int(len(source_df)),
        "source_csv_columns":
            int(len(source_df.columns)),
        "duplicate_file_encoding":
            duplicate_encoding,
        "duplicate_groups":
            int(
                duplicate_df[
                    "duplicate_group_id"
                ].nunique()
            ),
        "duplicate_members":
            int(len(duplicate_df)),
        "dataset_index_min":
            minimum_index,
        "dataset_index_max":
            maximum_index,
        "mapping_verified":
            mapping_verified,
        "best_mapping_match_rates":
            best_rates,
        "original_station_candidates":
            original_station_candidates,
        "synthetic_station_columns":
            synthetic_station_columns,
        "network_candidates":
            network_candidates,
        "source_candidates":
            source_candidates,
        "classification_counts": {
            str(key): int(value)
            for key, value
            in classification_counts.items()
        },
    }

    with open(
        OUTPUT_DIR / "trace_summary.json",
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
    print("追踪完成")
    print("=" * 100)

    print(
        f"dataset_index映射是否通过核验："
        f"{mapping_verified}"
    )

    print(
        "最佳字段匹配率："
        f"{best_rates}"
    )

    print(
        "原始站点编号候选列："
        f"{original_station_candidates}"
    )

    print(
        "网络候选列："
        f"{network_candidates}"
    )

    print(
        "来源候选列："
        f"{source_candidates}"
    )

    print("\n分类数量：")

    for key, value in (
        classification_counts.items()
    ):
        print(f"  {key}: {value}")

    print("\n冲突最大的前15组：")

    print(
        group_df[
            [
                "duplicate_group_id",
                "member_count",
                "splits",
                "synthetic_station_ids",
                "dates",
                "target_min_mm",
                "target_max_mm",
                "target_range_mm",
                "original_station_values",
                "network_values",
                "source_values",
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
