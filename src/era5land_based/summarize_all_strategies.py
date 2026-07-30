#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    ("Frozen M0", "frozen"),
    ("Fusion FT", "fusion_ft"),
    ("Point FT", "point_ft"),
    ("Spatial FT", "spatial_ft"),
    ("Partial FT", "partial"),
    ("Full FT", "none"),
]


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    headers = [str(column) for column in frame.columns]
    rows = [
        [str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(
            len(headers[index]),
            *[len(row[index]) for row in rows],
        )
        for index in range(len(headers))
    ]

    def render(row):
        return "| " + " | ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        ) + " |"

    separator = "| " + " | ".join(
        "-" * width
        for width in widths
    ) + " |"
    return "\n".join(
        [render(headers), separator]
        + [render(row) for row in rows]
    )


def newest_unique(root: Path, pattern: str) -> Path:
    matches = sorted(
        root.rglob(pattern),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"{root} 下找不到 {pattern}")
    if len(matches) > 1:
        print(f"⚠ {root} 下找到{len(matches)}个 {pattern}，使用最新文件")
    return matches[-1]


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def compute_metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]
    if target.size == 0:
        raise RuntimeError("没有有效target/prediction")

    error = prediction - target
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    target_std = float(np.std(target))
    prediction_std = float(np.std(prediction))
    std_ratio = (
        prediction_std / target_std
        if target_std > 1e-12
        else float("nan")
    )

    centered_target = target - np.mean(target)
    denominator = float(np.sum(centered_target ** 2))
    slope = (
        float(
            np.sum(
                centered_target
                * (prediction - np.mean(prediction))
            )
            / denominator
        )
        if denominator > 1e-12
        else float("nan")
    )

    r = (
        float(np.corrcoef(target, prediction)[0, 1])
        if target.size > 1
        and target_std > 1e-12
        and prediction_std > 1e-12
        else float("nan")
    )
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    nse = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-12
        else float("nan")
    )

    ge50 = target >= 50.0
    ge80 = target >= 80.0
    rmse_ge50 = (
        float(np.sqrt(np.mean(error[ge50] ** 2)))
        if np.any(ge50)
        else float("nan")
    )
    bias_ge80 = (
        float(np.mean(error[ge80]))
        if np.any(ge80)
        else float("nan")
    )

    return {
        "n_samples": int(target.size),
        "r": finite_float(r),
        "nse": finite_float(nse),
        "rmse_mm": rmse,
        "mae_mm": mae,
        "bias_mm": bias,
        "slope": finite_float(slope),
        "std_ratio": finite_float(std_ratio),
        "rmse_obs_ge50_mm": finite_float(rmse_ge50),
        "bias_obs_ge80_mm": finite_float(bias_ge80),
        "n_obs_ge50": int(np.sum(ge50)),
        "n_obs_ge80": int(np.sum(ge80)),
    }


def canonical_frame(path: Path, method_key: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if method_key == "frozen":
        target_col = "target_mm"
        prediction_col = "frozen_prediction_mm"
    else:
        target_col = "target_mm"
        prediction_col = "prediction_mm"

    missing = [
        column
        for column in ["fold", target_col, prediction_col]
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            f"{path} 缺少列 {missing}; actual={list(frame.columns)}"
        )

    result = pd.DataFrame({
        "fold": pd.to_numeric(frame["fold"], errors="raise").astype(int),
        "target_mm": pd.to_numeric(
            frame[target_col],
            errors="coerce",
        ),
        "prediction_mm": pd.to_numeric(
            frame[prediction_col],
            errors="coerce",
        ),
    })
    if "dataset_index" in frame.columns:
        result["dataset_index"] = pd.to_numeric(
            frame["dataset_index"],
            errors="raise",
        ).astype(int)
    else:
        result["dataset_index"] = np.arange(len(result), dtype=np.int64)
    return result


def load_method_frame(run_root: Path, method_key: str):
    if method_key == "frozen":
        search_root = run_root / "frozen_station_cv10"
        path = newest_unique(
            search_root,
            "frozen_station_cv10_oof_predictions.csv",
        )
    else:
        search_root = run_root / method_key
        path = newest_unique(
            search_root,
            "station_cv10_oof_predictions.csv",
        )
    return canonical_frame(path, method_key), path


def plot_fold_distributions(
    fold_metrics: pd.DataFrame,
    output_path: Path,
):
    specs = [
        ("rmse_mm", "RMSE (mm)", None),
        ("r", "R", None),
        ("mae_mm", "MAE (mm)", None),
        ("bias_mm", "Bias (mm)", 0.0),
        ("slope", "Slope", 1.0),
        ("std_ratio", "Std(pred) / Std(obs)", 1.0),
    ]
    labels = [label for label, _ in METHODS]
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    for ax, (column, title, reference) in zip(
        axes.reshape(-1),
        specs,
    ):
        values = []
        for label in labels:
            series = pd.to_numeric(
                fold_metrics.loc[
                    fold_metrics["method"] == label,
                    column,
                ],
                errors="coerce",
            ).dropna()
            values.append(series.to_numpy(dtype=np.float64))

        box = ax.boxplot(
            values,
            labels=labels,
            showmeans=True,
            meanprops={
                "marker": "^",
                "markerfacecolor": "#2ca02c",
                "markeredgecolor": "#2ca02c",
                "markersize": 7,
            },
            medianprops={
                "color": "#ff7f0e",
                "linewidth": 1.8,
            },
            patch_artist=True,
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#f7f7f7")
            patch.set_edgecolor("#333333")
        if reference is not None:
            ax.axhline(
                reference,
                color="#777777",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
        ax.set_title(f"Nested station-wise 10-fold: {title}")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "M0 Frozen and all fine-tuning strategies\n"
        "Frozen-relative gate temporarily disabled for FT checkpoint selection",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Orange line = median; green triangle = mean. "
        "Outer folds are not used for checkpoint selection.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)

    output_dir = run_root / "all_strategy_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled_rows = []
    fold_rows = []
    source_files = {}
    identity_reference = None
    target_reference = None

    for method_label, method_key in METHODS:
        frame, source_path = load_method_frame(
            run_root,
            method_key,
        )
        frame = frame.sort_values("dataset_index").reset_index(drop=True)
        source_files[method_label] = str(source_path)

        identity = frame["dataset_index"].to_numpy(dtype=np.int64)
        target = frame["target_mm"].to_numpy(dtype=np.float64)
        if identity_reference is None:
            identity_reference = identity
            target_reference = target
        else:
            if not np.array_equal(identity, identity_reference):
                raise RuntimeError(
                    f"{method_label} OOF样本身份与Frozen不一致"
                )
            if not np.allclose(
                target,
                target_reference,
                atol=1e-8,
                rtol=0.0,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"{method_label} OOF target与Frozen不一致"
                )

        pooled = compute_metrics(
            frame["target_mm"],
            frame["prediction_mm"],
        )
        pooled_rows.append({
            "method": method_label,
            "method_key": method_key,
            **pooled,
            "source_file": str(source_path),
        })

        for fold, fold_frame in frame.groupby("fold", sort=True):
            metrics = compute_metrics(
                fold_frame["target_mm"],
                fold_frame["prediction_mm"],
            )
            fold_rows.append({
                "method": method_label,
                "method_key": method_key,
                "fold": int(fold),
                **metrics,
            })

    pooled_frame = pd.DataFrame(pooled_rows)
    fold_frame = pd.DataFrame(fold_rows)

    pooled_path = output_dir / "M0_all_strategies_pooled_oof_metrics.csv"
    fold_path = output_dir / "M0_all_strategies_fold_metrics.csv"
    figure_path = output_dir / "M0_all_strategies_boxplots.png"
    markdown_path = output_dir / "M0_all_strategies_summary.md"
    audit_path = output_dir / "M0_all_strategies_audit.json"

    pooled_frame.to_csv(
        pooled_path,
        index=False,
        encoding="utf-8-sig",
    )
    fold_frame.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )
    plot_fold_distributions(fold_frame, figure_path)

    display_columns = [
        "method",
        "n_samples",
        "r",
        "nse",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
        "slope",
        "std_ratio",
        "rmse_obs_ge50_mm",
        "bias_obs_ge80_mm",
    ]
    table = pooled_frame[display_columns].copy()
    numeric_columns = [
        column
        for column in display_columns
        if column not in {"method", "n_samples"}
    ]
    for column in numeric_columns:
        table[column] = table[column].map(
            lambda value: (
                f"{float(value):.4f}"
                if pd.notna(value)
                else "NA"
            )
        )

    markdown = [
        "# M0 Frozen and all fine-tuning strategies",
        "",
        "Checkpoint policy for the five fine-tuning strategies: "
        "**Frozen-relative gate temporarily disabled; the minimum composite "
        "selection score among non-collapsed fine-tuned epochs is selected "
        "on each Inner fold.**",
        "",
        "Outer folds are used only for OOF evaluation. The fixed external "
        "987-sample set is not evaluated in this run.",
        "",
        markdown_table(table),
        "",
    ]
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")

    audit = {
        "run_root": str(run_root),
        "n_methods": len(METHODS),
        "methods": [label for label, _ in METHODS],
        "n_oof_samples_per_method": int(len(identity_reference)),
        "sample_identity_aligned": True,
        "target_values_aligned": True,
        "frozen_relative_gate_enabled_for_finetuning": False,
        "collapse_guard_retained": True,
        "outer_used_for_checkpoint_selection": False,
        "external_987_evaluated": False,
        "source_files": source_files,
        "outputs": {
            "pooled_metrics": str(pooled_path),
            "fold_metrics": str(fold_path),
            "boxplots": str(figure_path),
            "markdown_summary": str(markdown_path),
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ 全策略汇总完成")
    print(table.to_string(index=False))
    print(f"汇总目录: {output_dir}")


if __name__ == "__main__":
    main()
