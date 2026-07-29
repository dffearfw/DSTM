#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("rmse_mm", "Station-wise 10-fold distribution: RMSE (mm)", "RMSE (mm)"),
    ("r", "Station-wise 10-fold distribution: R", "R"),
    ("mae_mm", "Station-wise 10-fold distribution: MAE (mm)", "MAE (mm)"),
    ("bias_mm", "Station-wise 10-fold distribution: Bias (mm)", "Bias (mm)"),
]
REQUIRED_COLUMNS = {
    "fold",
    "method",
    "r",
    "rmse_mm",
    "mae_mm",
    "bias_mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine ERA5-Land and Frozen M0-M6 fold metrics into one 4-panel boxplot."
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
        help="Stages to include; default: 0 1 2 3 4 5 6.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_stage_metrics(run_dir: Path, stage: int) -> pd.DataFrame:
    path = (
        run_dir
        / f"M{stage}"
        / "internal_cv10"
        / "frozen_station_cv10_fold_metrics.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path}缺少字段: {sorted(missing)}")

    expected_methods = {"ERA5-Land", "Frozen"}
    actual_methods = set(frame["method"].dropna().astype(str))
    if actual_methods != expected_methods:
        raise RuntimeError(
            f"{path}方法标签异常: expected={expected_methods}, actual={actual_methods}"
        )

    for method in expected_methods:
        subset = frame.loc[frame["method"] == method]
        folds = sorted(subset["fold"].astype(int).tolist())
        if folds != list(range(1, 11)):
            raise RuntimeError(
                f"{path}的{method}不是完整10折: {folds}"
            )
    return frame


def build_combined_frame(run_dir: Path, stages: list[int]) -> pd.DataFrame:
    if not stages:
        raise RuntimeError("至少需要一个stage")
    if len(stages) != len(set(stages)):
        raise RuntimeError(f"--stages包含重复值: {stages}")
    if any(stage < 0 or stage > 6 for stage in stages):
        raise RuntimeError(f"--stages只允许0-6: {stages}")

    era5_reference: pd.DataFrame | None = None
    model_frames: list[pd.DataFrame] = []
    comparison_columns = ["fold", "r", "rmse_mm", "mae_mm", "bias_mm"]

    for stage in stages:
        frame = read_stage_metrics(run_dir, stage)
        era5 = (
            frame.loc[frame["method"] == "ERA5-Land"]
            .sort_values("fold")
            .reset_index(drop=True)
        )
        if era5_reference is None:
            era5_reference = era5.copy()
        else:
            reference_values = era5_reference[comparison_columns].to_numpy(
                dtype=float
            )
            current_values = era5[comparison_columns].to_numpy(dtype=float)
            if not np.allclose(
                reference_values,
                current_values,
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"M{stage}中的ERA5-Land逐折指标与首个stage不一致，"
                    "拒绝生成不可比图。"
                )

        frozen = frame.loc[frame["method"] == "Frozen"].copy()
        frozen["method"] = f"M{stage}"
        model_frames.append(frozen)

    assert era5_reference is not None
    return pd.concat(
        [era5_reference, *model_frames],
        ignore_index=True,
    )


def plot_four_panel(
    frame: pd.DataFrame,
    method_order: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.reshape(-1)

    meanprops = {
        "marker": "^",
        "markerfacecolor": "#2ca02c",
        "markeredgecolor": "#2ca02c",
        "markersize": 8,
    }
    medianprops = {"color": "#ff7f0e", "linewidth": 1.6}

    for ax, (metric, title, ylabel) in zip(axes, METRICS):
        values = [
            frame.loc[frame["method"] == method, metric]
            .astype(float)
            .to_numpy()
            for method in method_order
        ]
        ax.boxplot(
            values,
            labels=method_order,
            showmeans=True,
            meanprops=meanprops,
            medianprops=medianprops,
            widths=0.58,
        )
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    stages = sorted(args.stages)
    method_order = ["ERA5-Land", *[f"M{stage}" for stage in stages]]

    frame = build_combined_frame(run_dir, stages)
    combined_csv = run_dir / "frozen_M0_M6_fold_metrics_combined.csv"
    output_png = run_dir / "frozen_M0_M6_fold_distribution_4panel.png"

    frame.to_csv(combined_csv, index=False, encoding="utf-8-sig")
    plot_four_panel(frame, method_order, output_png, args.dpi)

    print(f"✅ 四联箱线图: {output_png}")
    print(f"✅ 合并逐折指标: {combined_csv}")


if __name__ == "__main__":
    main()
