#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_frozen_station_cv10 import (
    STATION_SCATTER_AXIS_MAX_MM,
    STATION_SCATTER_AXIS_MIN_MM,
    compute_metrics,
    plot_fold_scatter_outputs,
    plot_scatter,
)


REQUIRED_COLUMNS = {
    "fold",
    "target_mm",
    "frozen_prediction_mm",
    "era5_land_mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Redraw existing ERA5-Land and Frozen M0-M6 scatter plots "
            "with fixed 0-400 mm x/y axes; no model inference."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Frozen M0-M6 batch output directory.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=list(range(7)),
        help="Stages to redraw; default: 0 1 2 3 4 5 6.",
    )
    return parser.parse_args()


def redraw_stage(run_dir: Path, stage: int) -> None:
    output_dir = run_dir / f"M{stage}" / "internal_cv10"
    prediction_path = output_dir / "frozen_station_cv10_oof_predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)

    predictions = pd.read_csv(prediction_path)
    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise RuntimeError(
            f"{prediction_path}缺少字段: {sorted(missing)}"
        )

    folds = sorted(
        pd.to_numeric(predictions["fold"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if folds != list(range(1, 11)):
        raise RuntimeError(
            f"{prediction_path}不是完整10折: {folds}"
        )

    frozen_metrics = compute_metrics(
        predictions["target_mm"],
        predictions["frozen_prediction_mm"],
    )
    era5_metrics = compute_metrics(
        predictions["target_mm"],
        predictions["era5_land_mm"],
    )

    plot_scatter(
        predictions["target_mm"].to_numpy(),
        predictions["frozen_prediction_mm"].to_numpy(),
        frozen_metrics,
        f"Frozen M{stage}",
        output_dir / "frozen_station_cv10_pooled_scatter.png",
    )
    plot_scatter(
        predictions["target_mm"].to_numpy(),
        predictions["era5_land_mm"].to_numpy(),
        era5_metrics,
        "ERA5-Land",
        output_dir / "era5_station_cv10_pooled_scatter.png",
    )
    plot_fold_scatter_outputs(
        predictions,
        prediction_column="frozen_prediction_mm",
        method_label=f"Frozen M{stage}",
        prefix="frozen",
        output_dir=output_dir,
    )
    plot_fold_scatter_outputs(
        predictions,
        prediction_column="era5_land_mm",
        method_label="ERA5-Land",
        prefix="era5",
        output_dir=output_dir,
    )
    print(f"✅ M{stage}: 已统一重绘 {output_dir}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    stages = sorted(args.stages)

    if not stages:
        raise RuntimeError("至少需要一个stage")
    if len(stages) != len(set(stages)):
        raise RuntimeError(f"--stages包含重复值: {stages}")
    if any(stage < 0 or stage > 6 for stage in stages):
        raise RuntimeError(f"--stages只允许0-6: {stages}")

    for stage in stages:
        redraw_stage(run_dir, stage)

    print(
        "✅ 全部完成：所有横纵坐标均固定为 "
        f"{STATION_SCATTER_AXIS_MIN_MM:.0f}–"
        f"{STATION_SCATTER_AXIS_MAX_MM:.0f} mm；"
        "指标未重新定义，未运行模型推理。"
    )


if __name__ == "__main__":
    main()
