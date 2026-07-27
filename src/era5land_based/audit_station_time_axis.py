# -*- coding: utf-8 -*-

import calendar
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import rasterio


FEATURE_ROOT = Path("/root/ablation")

TRACE_FILE = Path(
    "/root/autodl-tmp/diagnostics/"
    "trace_duplicate_sources_v2/"
    "corrected_duplicate_source_trace_rows.csv"
)

OUTPUT_DIR = Path(
    "/root/autodl-tmp/diagnostics/"
    "station_time_axis_audit"
)

YEARS = [2015, 2016, 2017, 2018]

VARIABLE_DIRS = {
    "chelsa_sfxwind": FEATURE_ROOT / "sfxwind" / "cn",
    "lst": FEATURE_ROOT / "lst" / "cn",
    "rh": FEATURE_ROOT / "rh" / "cn",
    "pr": FEATURE_ROOT / "pr" / "cn",
}


def parse_daily_filename(filename):
    """复制StationSWEDataset中日文件的日期解析规则。"""

    specific = re.search(
        r"CHELSA_(?:pr|sfcWind)_(\d{2})_(\d{2})_(\d{4})",
        filename,
        flags=re.IGNORECASE,
    )

    if specific:
        day = int(specific.group(1))
        month = int(specific.group(2))
        year = int(specific.group(3))
        return datetime(year, month, day)

    patterns = [
        (r"(\d{2})_(\d{2})_(\d{4})", "dmy"),
        (r"(\d{4})(\d{2})(\d{2})", "ymd"),
        (r"(\d{4})-(\d{2})-(\d{2})", "ymd"),
        (r"(\d{4})_(\d{2})_(\d{2})", "ymd"),
    ]

    for pattern, order in patterns:
        match = re.search(pattern, filename)

        if not match:
            continue

        try:
            a, b, c = map(int, match.groups())

            if order == "dmy":
                day, month, year = a, b, c
            else:
                year, month, day = a, b, c

            return datetime(year, month, day)

        except (TypeError, ValueError):
            continue

    return None


def find_files(variable, year):
    root = VARIABLE_DIRS[variable]

    if variable in {"chelsa_sfxwind", "pr"}:
        patterns = [f"*{year}*.tif"]

    elif variable == "lst":
        patterns = [
            f"ERA5LAND_LST_{year}??_DAILYMEAN*.tif",
            f"*LST*{year}*.tif",
            f"ERA5_ST_{year}*.tif",
        ]

    elif variable == "rh":
        patterns = [
            f"ERA5LAND_RH_{year}??_DAILYMEAN*.tif",
            f"*RH*{year}*.tif",
            f"ERA5_RH_DailyMean_{year}_*.tif",
        ]

    else:
        raise ValueError(variable)

    files = []
    seen = set()

    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            resolved = str(path.resolve())

            if resolved not in seen:
                seen.add(resolved)
                files.append(path)

    return files


def read_daily_file_dates(variable):
    dates = []
    unparsed_files = []

    for year in YEARS:
        for path in find_files(variable, year):
            date = parse_daily_filename(path.name)

            if date is None:
                unparsed_files.append(str(path))
                continue

            if date.year in YEARS:
                dates.append(date)

    return dates, unparsed_files


def parse_cube_year_month(filename):
    match = re.search(r"(\d{4})(\d{2})", filename)

    if not match:
        match = re.search(r"(\d{4})[_-](\d{2})", filename)

    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))

    if year not in YEARS or not 1 <= month <= 12:
        return None

    return year, month


def read_monthly_cube_dates(variable):
    dates = []
    file_rows = []
    unparsed_files = []

    for year in YEARS:
        for path in find_files(variable, year):
            parsed = parse_cube_year_month(path.name)

            if parsed is None:
                unparsed_files.append(str(path))
                continue

            file_year, month = parsed
            calendar_days = calendar.monthrange(file_year, month)[1]

            try:
                with rasterio.open(path) as dataset:
                    band_count = int(dataset.count)
            except Exception as exc:
                file_rows.append({
                    "variable": variable,
                    "file": str(path),
                    "year": file_year,
                    "month": month,
                    "band_count": None,
                    "calendar_days": calendar_days,
                    "loaded_days": 0,
                    "error": str(exc),
                })
                continue

            loaded_days = min(band_count, calendar_days)

            for day in range(1, loaded_days + 1):
                dates.append(datetime(file_year, month, day))

            file_rows.append({
                "variable": variable,
                "file": str(path),
                "year": file_year,
                "month": month,
                "band_count": band_count,
                "calendar_days": calendar_days,
                "loaded_days": loaded_days,
                "error": "",
            })

    return dates, file_rows, unparsed_files


def nearest_date(target_date, available_dates):
    return min(
        available_dates,
        key=lambda date: abs((date - target_date).days),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRACE_FILE.exists():
        raise FileNotFoundError(TRACE_FILE)

    trace = pd.read_csv(TRACE_FILE)

    required = {
        "duplicate_group_id",
        "diagnostic_date",
        "source__date",
        "source__station_id",
        "source__source_station_ids",
        "source__swe",
        "source__row",
        "source__col",
    }

    missing = sorted(required - set(trace.columns))

    if missing:
        raise ValueError(f"追踪文件缺少字段：{missing}")

    raw_dates = {}
    file_rows = []
    unparsed = {}

    for variable in ["chelsa_sfxwind", "pr"]:
        dates, bad_files = read_daily_file_dates(variable)
        raw_dates[variable] = sorted(set(dates))
        unparsed[variable] = bad_files

    for variable in ["lst", "rh"]:
        dates, rows, bad_files = read_monthly_cube_dates(variable)
        raw_dates[variable] = sorted(set(dates))
        file_rows.extend(rows)
        unparsed[variable] = bad_files

    date_sets = {
        variable: set(dates)
        for variable, dates in raw_dates.items()
    }

    common_dates = set(date_sets["chelsa_sfxwind"])

    for variable in ["lst", "rh", "pr"]:
        common_dates &= date_sets[variable]

    common_dates = sorted(common_dates)

    if not common_dates:
        raise RuntimeError("四个动态变量没有共同日期")

    coverage_rows = []

    for variable, dates in raw_dates.items():
        coverage_rows.append({
            "variable": variable,
            "date_count": len(dates),
            "first_date": dates[0].strftime("%Y-%m-%d") if dates else "",
            "last_date": dates[-1].strftime("%Y-%m-%d") if dates else "",
            "unparsed_file_count": len(unparsed[variable]),
        })

    coverage_rows.append({
        "variable": "COMMON_INTERSECTION",
        "date_count": len(common_dates),
        "first_date": common_dates[0].strftime("%Y-%m-%d"),
        "last_date": common_dates[-1].strftime("%Y-%m-%d"),
        "unparsed_file_count": 0,
    })

    audit_rows = []

    for _, row in trace.iterrows():
        label_date = pd.to_datetime(
            row["source__date"]
        ).to_pydatetime()

        diagnostic_feature_date = pd.to_datetime(
            row["diagnostic_date"]
        ).to_pydatetime()

        availability = {
            variable: label_date in date_sets[variable]
            for variable in date_sets
        }

        missing_variables = [
            variable
            for variable, available in availability.items()
            if not available
        ]

        if label_date in set(common_dates):
            calculated_feature_date = label_date
        else:
            calculated_feature_date = nearest_date(
                label_date,
                common_dates,
            )

        calculated_gap = abs(
            (calculated_feature_date - label_date).days
        )

        result = {
            "duplicate_group_id":
                int(row["duplicate_group_id"]),
            "synthetic_station_id":
                row["source__station_id"],
            "original_station_id":
                row["source__source_station_ids"],
            "row":
                int(row["source__row"]),
            "col":
                int(row["source__col"]),
            "station_swe_mm":
                float(row["source__swe"]),
            "label_date":
                label_date.strftime("%Y-%m-%d"),
            "diagnostic_feature_date":
                diagnostic_feature_date.strftime("%Y-%m-%d"),
            "calculated_feature_date":
                calculated_feature_date.strftime("%Y-%m-%d"),
            "calculated_day_gap":
                calculated_gap,
            "label_date_in_common_intersection":
                label_date in set(common_dates),
            "missing_variables":
                "|".join(missing_variables),
            "feature_date_matches_diagnostic":
                calculated_feature_date
                == diagnostic_feature_date,
        }

        for variable in [
            "chelsa_sfxwind",
            "lst",
            "rh",
            "pr",
        ]:
            result[f"{variable}_has_label_date"] = availability[
                variable
            ]

            nearest_variable_date = nearest_date(
                label_date,
                raw_dates[variable],
            )

            result[f"{variable}_nearest_date"] = (
                nearest_variable_date.strftime("%Y-%m-%d")
            )

            result[f"{variable}_nearest_gap_days"] = abs(
                (nearest_variable_date - label_date).days
            )

        audit_rows.append(result)

    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        OUTPUT_DIR / "duplicate_time_axis_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(coverage_rows).to_csv(
        OUTPUT_DIR / "dynamic_variable_date_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if file_rows:
        pd.DataFrame(file_rows).to_csv(
            OUTPUT_DIR / "lst_rh_cube_band_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )

    missing_reason = (
        audit["missing_variables"]
        .value_counts(dropna=False)
        .to_dict()
    )

    summary = {
        "dynamic_date_counts": {
            variable: len(dates)
            for variable, dates in raw_dates.items()
        },
        "common_intersection_count":
            len(common_dates),
        "common_first_date":
            common_dates[0].strftime("%Y-%m-%d"),
        "common_last_date":
            common_dates[-1].strftime("%Y-%m-%d"),
        "duplicate_members":
            int(len(audit)),
        "duplicate_groups":
            int(audit["duplicate_group_id"].nunique()),
        "label_dates_in_common_intersection":
            int(
                audit[
                    "label_date_in_common_intersection"
                ].sum()
            ),
        "collapsed_members":
            int(
                (
                    ~audit[
                        "label_date_in_common_intersection"
                    ]
                ).sum()
            ),
        "calculated_feature_date_matches_diagnostic":
            int(
                audit[
                    "feature_date_matches_diagnostic"
                ].sum()
            ),
        "missing_variable_combinations": {
            str(key): int(value)
            for key, value in missing_reason.items()
        },
        "unparsed_files": unparsed,
    }

    with open(
        OUTPUT_DIR / "time_axis_audit_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print("动态变量日期覆盖")
    print("=" * 100)

    print(
        pd.DataFrame(coverage_rows).to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("造成日期折叠的缺失变量组合")
    print("=" * 100)

    print(
        audit[
            "missing_variables"
        ].value_counts(
            dropna=False
        ).to_string()
    )

    print("\n" + "=" * 100)
    print("冲突最大的第50组")
    print("=" * 100)

    columns = [
        "duplicate_group_id",
        "original_station_id",
        "station_swe_mm",
        "label_date",
        "diagnostic_feature_date",
        "calculated_feature_date",
        "calculated_day_gap",
        "missing_variables",
        "chelsa_sfxwind_has_label_date",
        "lst_has_label_date",
        "rh_has_label_date",
        "pr_has_label_date",
        "feature_date_matches_diagnostic",
    ]

    print(
        audit[
            audit["duplicate_group_id"] == 50
        ][columns].to_string(index=False)
    )

    print("\n完整结果：")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
