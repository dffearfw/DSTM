#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose Stage-0 station-record filtering.

Supports both:
1) monthly multi-band ERA5-Land SWE cubes, e.g.
   ERA5LAND_SWE_DAILY_AGGR_201507_...tif (band 1 = July 1, ...)
2) legacy daily single-band files containing YYYYMMDD in the filename.

This script is read-only.
"""
from __future__ import annotations

import argparse
import calendar
import glob
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio

ALIASES = {
    "longtitude": "longitude", "longitude": "longitude", "lon": "longitude",
    "lng": "longitude", "long": "longitude", "经度": "longitude",
    "latitude": "latitude", "lat": "latitude", "纬度": "latitude",
    "date": "date", "datetime": "date", "time": "date", "日期": "date",
    "观测日期": "date", "测量日期": "date",
    "station_id": "station_id", "stationid": "station_id", "station": "station_id",
    "site_id": "station_id", "id": "station_id", "站点": "station_id", "站号": "station_id",
    "swe": "swe", "swe_mm": "swe", "swe_value": "swe", "swedepth": "swe",
    "value": "swe", "积雪水当量": "swe",
}
MONTHLY_RE = re.compile(r"ERA5LAND_SWE_DAILY_AGGR_(20\d{2})([01]\d)", re.I)
DAILY_RE = re.compile(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def f(x):
        s = str(x).strip()
        return ALIASES.get(s.lower(), ALIASES.get(s, s))
    return df.rename(columns=f)


def parse_dates(s: pd.Series) -> pd.Series:
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # Avoid pandas treating numeric YYYYMMDD as ns since epoch.
    num = pd.to_numeric(s, errors="coerce")
    ymd = num.between(19000101, 21001231, inclusive="both")
    if ymd.any():
        parsed = pd.to_datetime(
            num.loc[ymd].round().astype("Int64").astype(str),
            format="%Y%m%d", errors="coerce"
        )
        out.loc[ymd] = parsed

    excel_serial = num.between(20000, 80000, inclusive="both") & ~ymd
    if excel_serial.any():
        parsed = pd.to_datetime(
            num.loc[excel_serial], unit="D", origin="1899-12-30", errors="coerce"
        )
        out.loc[excel_serial] = parsed

    remaining = out.isna()
    if remaining.any():
        parsed = pd.to_datetime(s.loc[remaining], errors="coerce")
        out.loc[remaining] = parsed

    return out.dt.normalize()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    last = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"无法读取 {path}: {last}")


def discover_label_dates(root: Path, years: set[int]):
    files = sorted(root.rglob("*.tif"))
    if not files:
        raise SystemExit(f"标签目录下没有tif: {root}")

    label_dates: set[pd.Timestamp] = set()
    recognized = []
    unrecognized = []
    grid_file = None

    for p in files:
        name = p.stem
        mm = MONTHLY_RE.search(name)
        if mm:
            year, month = map(int, mm.groups())
            if year not in years:
                continue
            try:
                with rasterio.open(p) as src:
                    if grid_file is None:
                        grid_file = p
                    ndays = calendar.monthrange(year, month)[1]
                    nbands = min(src.count, ndays)
                for day in range(1, nbands + 1):
                    label_dates.add(pd.Timestamp(year=year, month=month, day=day))
                recognized.append({"file": str(p), "type": "monthly_cube", "bands_used": nbands})
            except Exception as exc:
                unrecognized.append({"file": str(p), "reason": f"open failed: {exc}"})
            continue

        dm = DAILY_RE.search(name)
        if dm:
            year, month, day = map(int, dm.groups())
            if year not in years:
                continue
            try:
                dt = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                unrecognized.append({"file": str(p), "reason": "invalid date"})
                continue
            label_dates.add(dt)
            recognized.append({"file": str(p), "type": "daily_file", "bands_used": 1})
            if grid_file is None:
                grid_file = p
        else:
            unrecognized.append({"file": str(p), "reason": "filename pattern not recognized"})

    if not recognized or not label_dates or grid_file is None:
        examples = "\n".join(f"  {p.name}" for p in files[:20])
        raise SystemExit(
            "找到了tif，但无法识别ERA5-Land SWE日期格式。前20个文件:\n" + examples
        )
    return grid_file, label_dates, recognized, unrecognized


def map_cells(df: pd.DataFrame, transform, h: int, w: int):
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    ok_coord = lon.notna() & lat.notna()
    in_grid = np.zeros(len(df), dtype=bool)
    rows = np.full(len(df), -1, dtype=int)
    cols = np.full(len(df), -1, dtype=int)
    cells = set()
    for i, (x, y, ok) in enumerate(zip(lon, lat, ok_coord)):
        if not ok:
            continue
        c_f, r_f = ~transform * (float(x), float(y))
        r, c = int(np.floor(r_f)), int(np.floor(c_f))
        rows[i], cols[i] = r, c
        if 0 <= r < h and 0 <= c < w:
            in_grid[i] = True
            cells.add((r, c))
    return ok_coord.to_numpy(), in_grid, rows, cols, cells


def expand(cells: Iterable[tuple[int, int]], radius: int, h: int, w: int):
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="/root/ablation/station_swe_data.xlsx")
    ap.add_argument("--label-root", default="/root/ablation/era5landswe")
    ap.add_argument(
        "--external-glob",
        default="/root/ablation/external_test/*.csv",
        help="Use an explicit external-test directory; pass '' to disable external exclusion diagnosis."
    )
    ap.add_argument("--external-radius", type=int, default=0)
    ap.add_argument("--years", nargs="+", type=int, default=[2015, 2016, 2017, 2018])
    ap.add_argument("--report", default="/root/autodl-tmp/shared_cache/stage0_record_diagnosis.json")
    args = ap.parse_args()

    excel = Path(args.excel)
    root = Path(args.label_root)
    if not excel.exists():
        raise SystemExit(f"Excel不存在: {excel}")
    if not root.exists():
        raise SystemExit(f"标签目录不存在: {root}")

    grid_file, label_dates, recognized, unrecognized = discover_label_dates(root, set(args.years))
    with rasterio.open(grid_file) as src:
        transform, h, w, grid_crs = src.transform, src.height, src.width, str(src.crs)

    print("\n=== ERA5-Land SWE labels ===")
    print("标签目录:", root)
    print("识别文件数:", len(recognized))
    print("月立方体数:", sum(x["type"] == "monthly_cube" for x in recognized))
    print("单日文件数:", sum(x["type"] == "daily_file" for x in recognized))
    print("标签日期数:", len(label_dates))
    print("日期范围:", min(label_dates).date(), "至", max(label_dates).date())
    print("参考网格:", grid_file.name, h, "x", w, grid_crs)

    df0 = read_table(excel)
    df = normalize_columns(df0.copy())
    print("\n=== Excel ===")
    print("原始列:", list(df0.columns))
    print("标准化列:", list(df.columns))
    print("原始行数:", len(df))
    missing = [c for c in ("longitude", "latitude", "date") if c not in df.columns]
    if missing:
        raise SystemExit(f"缺少必要列: {missing}")

    parsed = parse_dates(df["date"])
    df["_date"] = parsed
    valid_date = parsed.notna()
    year_ok = parsed.dt.year.isin(args.years)
    date_match = parsed.isin(label_dates)
    coord_ok, in_grid, rows, cols, _ = map_cells(df, transform, h, w)
    df["_row"], df["_col"] = rows, cols

    print("日期原值示例:", df["date"].head(10).tolist())
    print("解析日期示例:", parsed.head(10).astype(str).tolist())
    print("可解析日期:", int(valid_date.sum()))
    print("目标年份日期:", int(year_ok.sum()))
    print("与标签日期精确匹配:", int(date_match.sum()))
    print("有效经纬度:", int(coord_ok.sum()))
    print("落入标签网格:", int(in_grid.sum()))

    external_cells = set()
    ext_details = []
    print("\n=== External CSV ===")
    ext_paths = [Path(p) for p in glob.glob(args.external_glob)] if args.external_glob else []
    if not ext_paths:
        print("未匹配外部CSV（本次只诊断日期和网格，不应用外部排除）")
    for p in ext_paths:
        try:
            e0 = read_table(p)
            e = normalize_columns(e0.copy())
            if not {"longitude", "latitude"}.issubset(e.columns):
                print(f"跳过（无经纬度列）: {p.name}")
                continue
            _, ext_in, _, _, cells = map_cells(e, transform, h, w)
            external_cells |= cells
            ext_details.append({
                "file": str(p), "rows": len(e),
                "in_grid_rows": int(ext_in.sum()), "unique_cells": len(cells)
            })
            print(f"{p.name}: 行={len(e)}, 网格内={int(ext_in.sum())}, 唯一格点={len(cells)}")
        except Exception as exc:
            print(f"读取失败 {p.name}: {exc}")

    excluded = expand(external_cells, args.external_radius, h, w)
    base = valid_date.to_numpy() & year_ok.to_numpy() & date_match.to_numpy() & in_grid
    ext_mask = np.array([
        ((r, c) not in excluded) if ok else False
        for r, c, ok in zip(rows, cols, in_grid)
    ]) if excluded else np.ones(len(df), dtype=bool)
    after_external = base & ext_mask

    overlap_center = sum(
        (r, c) in external_cells for r, c, ok in zip(rows, cols, in_grid) if ok
    )
    overlap_buffer = sum(
        (r, c) in excluded for r, c, ok in zip(rows, cols, in_grid) if ok
    )

    print("\n=== Funnel ===")
    print("日期+年份+标签匹配+网格内:", int(base.sum()))
    print("Excel记录命中外部中心格点:", overlap_center)
    print(f"Excel记录命中外部±{args.external_radius}格缓冲:", overlap_buffer)
    print("外部排除后剩余:", int(after_external.sum()))
    tmp = df.loc[after_external, ["_date", "_row", "_col"]].drop_duplicates()
    print("按(date,row,col)去重后:", len(tmp))

    # Extra failure clues.
    if int(date_match.sum()) == 0:
        excel_dates = sorted(d.date() for d in parsed.dropna().unique()[:10])
        label_sample = sorted(d.date() for d in label_dates)[:10]
        print("\n❌ 日期完全不匹配")
        print("Excel日期示例:", excel_dates)
        print("标签日期示例:", label_sample)
    elif int(base.sum()) == 0 and int(in_grid.sum()) == 0:
        print("\n❌ 日期有匹配，但经纬度全部落在网格外，请检查经纬度列或CRS。")
    elif int(base.sum()) > 0 and int(after_external.sum()) == 0:
        print("\n❌ 所有候选都被外部站点缓冲区排除，请缩小external-glob或检查外部CSV目录。")

    report = {
        "excel": str(excel), "excel_rows": len(df),
        "raw_columns": list(map(str, df0.columns)),
        "normalized_columns": list(map(str, df.columns)),
        "label_root": str(root), "recognized_label_files": len(recognized),
        "monthly_cubes": sum(x["type"] == "monthly_cube" for x in recognized),
        "daily_files": sum(x["type"] == "daily_file" for x in recognized),
        "label_date_count": len(label_dates),
        "label_date_min": str(min(label_dates).date()),
        "label_date_max": str(max(label_dates).date()),
        "parseable_dates": int(valid_date.sum()), "year_ok": int(year_ok.sum()),
        "date_matches": int(date_match.sum()), "coord_ok": int(coord_ok.sum()),
        "in_grid": int(in_grid.sum()), "base_candidates": int(base.sum()),
        "external_glob": args.external_glob, "external_radius": args.external_radius,
        "external_files": ext_details, "external_unique_cells": len(external_cells),
        "excluded_cells_with_buffer": len(excluded),
        "overlap_center_records": overlap_center,
        "overlap_buffer_records": overlap_buffer,
        "after_external": int(after_external.sum()),
        "deduplicated_date_row_col": len(tmp),
        "unrecognized_tif_count": len(unrecognized),
        "unrecognized_tif_examples": unrecognized[:20],
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n报告:", out)


if __name__ == "__main__":
    main()
