#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验渐进式增量预训练 manifest。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

EXPECTED_STAGE_COUNTS = {1: 12000, 2: 20000, 3: 40000, 4: 80000}


def fail(message: str) -> None:
    print(f"❌ {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-annual-max-exclusive", type=float, default=1.0)
    parser.add_argument("--max-annual-max-exclusive", type=float, default=400.0)
    parser.add_argument("--min-post-peak-free-days", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.manifest)
    if not path.exists():
        fail(f"manifest不存在: {path}")

    df = pd.read_csv(path)
    required = {
        "sample_id", "date", "row", "col", "swe_mm", "swe_bin",
        "annual_max_swe_mm", "max_post_peak_snow_free_days",
        "stage_id", "fold_id", "source",
    }
    missing = required - set(df.columns)
    if missing:
        fail(f"缺少字段: {sorted(missing)}")

    expected_total = sum(EXPECTED_STAGE_COUNTS.values())
    if len(df) != expected_total:
        fail(f"总样本数={len(df):,}，预期={expected_total:,}")

    if df["sample_id"].duplicated().any():
        fail(f"重复sample_id={int(df['sample_id'].duplicated().sum()):,}")

    stage_counts = df.groupby("stage_id").size().astype(int).to_dict()
    if stage_counts != EXPECTED_STAGE_COUNTS:
        fail(f"阶段数量错误: {stage_counts}，预期={EXPECTED_STAGE_COUNTS}")

    annual = pd.to_numeric(df["annual_max_swe_mm"], errors="coerce")
    if annual.isna().any():
        fail("annual_max_swe_mm存在无法解析的值")
    bad_min = annual <= args.min_annual_max_exclusive
    bad_max = annual >= args.max_annual_max_exclusive
    if bad_min.any() or bad_max.any():
        fail(
            f"年最大SWE不满足严格范围 "
            f"({args.min_annual_max_exclusive}, {args.max_annual_max_exclusive}) mm；"
            f"下界失败={int(bad_min.sum())}, 上界失败={int(bad_max.sum())}"
        )

    runs = pd.to_numeric(df["max_post_peak_snow_free_days"], errors="coerce")
    if runs.isna().any():
        fail("max_post_peak_snow_free_days存在无法解析的值")
    bad_runs = runs < args.min_post_peak_free_days
    if bad_runs.any():
        fail(f"峰值后连续近无雪天数不足的样本={int(bad_runs.sum()):,}")

    folds = set(pd.to_numeric(df["fold_id"], errors="coerce").dropna().astype(int))
    if not folds.issubset(set(range(1, 11))) or len(folds) != 10:
        fail(f"fold_id异常: {sorted(folds)}")

    print("✅ 增量样本清单校验通过")
    print(f"   文件: {path}")
    print(f"   总样本: {len(df):,}")
    print(f"   阶段: {stage_counts}")
    print(
        f"   年最大SWE范围: [{annual.min():.3f}, {annual.max():.3f}] mm "
        f"（要求严格位于({args.min_annual_max_exclusive}, "
        f"{args.max_annual_max_exclusive})）"
    )
    print(
        f"   峰值后连续近无雪天数: "
        f"min={int(runs.min())}, median={float(runs.median()):.1f}, "
        f"max={int(runs.max())}"
    )
    print("   各阶段SWE箱数量:")
    table = df.groupby(["stage_id", "swe_bin"]).size().unstack(fill_value=0)
    print(table.to_string())


if __name__ == "__main__":
    main()
