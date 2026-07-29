#!/usr/bin/env python3
# CV10_FOLD_SCATTER_PANEL_V1
# -*- coding: utf-8 -*-
"""
Frozen M0–M6 的站点级 10 折内部评估。

目的：
1. 正式模式使用 internal_progressive_station.csv 的全部 7936 条内部样本；
2. 按 station_id 做确定性平衡10折，每个站点只属于一个测试折；
3. Frozen 模型只前向一次，随后按折计算指标；
4. 10 折合并为一套 OOF 预测，覆盖 7936 条样本且每条只评估一次；
5. 同一折同时计算 ERA5-Land 基线，输出 Frozen/ERA5 的箱线图；
6. 不训练、不改权重；旧固定1000条作为开发数据并回Nested CV。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


# ERA5-Land、Frozen M0-M6及后续微调图统一使用同一物理坐标域，
# 防止自动缩放造成跨模型视觉偏差。指标仍基于全部原始数值计算。
STATION_SCATTER_AXIS_MIN_MM = 0.0
STATION_SCATTER_AXIS_MAX_MM = 400.0
STATION_SCATTER_TICK_MM = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen model deterministic balanced station-wise 10-fold evaluation"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp"),
    )
    parser.add_argument(
        "--station-csv",
        type=Path,
        default=Path(
            "/root/autodl-tmp/shared_cache/progressive_finetune/"
            "internal_progressive_station.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage-label",
        type=str,
        default="M0",
        help="用于标题和汇总的阶段标签，例如M0、M1、...、M6",
    )
    parser.add_argument(
        "--normalization-config",
        type=Path,
        default=Path(
            "/root/autodl-tmp/shared_cache/"
            "progressive_pretrain_normalization.json"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/shared_cache"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=None,
    )
    # 保留seed参数仅兼容旧命令；平衡分折本身完全确定性，不使用随机数。
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--include-fixed-test",
        action="store_true",
        help="将旧split=test内部1000条并回站点级CV池",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def compute_metrics(target: Any, prediction: Any) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)

    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]

    if target.size == 0:
        raise RuntimeError("没有有效目标/预测值")

    error = prediction - target
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))

    target_std = float(np.std(target))
    pred_std = float(np.std(prediction))

    if target.size > 1 and target_std > 1e-12 and pred_std > 1e-12:
        r = float(np.corrcoef(target, prediction)[0, 1])
    else:
        r = float("nan")

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    nse = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    centered_target = target - np.mean(target)
    slope_den = float(np.sum(centered_target ** 2))
    if slope_den > 1e-12:
        slope = float(
            np.sum(centered_target * (prediction - np.mean(prediction)))
            / slope_den
        )
        intercept = float(np.mean(prediction) - slope * np.mean(target))
    else:
        slope = float("nan")
        intercept = float("nan")

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
        "r": clean_float(r),
        "nse": clean_float(nse),
        "rmse_mm": rmse,
        "mae_mm": mae,
        "bias_mm": bias,
        "rmse_obs_ge50_mm": clean_float(rmse_ge50),
        "bias_obs_ge80_mm": clean_float(bias_ge80),
        "n_obs_ge50": int(np.sum(ge50)),
        "n_obs_ge80": int(np.sum(ge80)),
        "slope": clean_float(slope),
        "intercept_mm": clean_float(intercept),
        "std_ratio": (
            clean_float(pred_std / target_std)
            if target_std > 1e-12
            else None
        ),
        "target_mean_mm": float(np.mean(target)),
        "prediction_mean_mm": float(np.mean(prediction)),
        "target_std_mm": target_std,
        "prediction_std_mm": pred_std,
        "prediction_min_mm": float(np.min(prediction)),
        "prediction_max_mm": float(np.max(prediction)),
    }


def normalize_station_id(value: Any) -> str:
    text = str(value)
    return text.split(",")[0].strip()


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def detect_cv_source_indices(
    source: pd.DataFrame,
    include_fixed_test: bool = False,
) -> tuple[list[int], list[int], str]:
    split_col = find_column(source, ["split", "Split", "subset"])
    if split_col is None:
        raise RuntimeError(
            "internal_progressive_station.csv 缺少 split 列，"
            f"现有列={source.columns.tolist()}"
        )

    split_norm = source[split_col].astype(str).str.strip().str.lower()
    test_mask = split_norm.str.contains("test", regex=False)

    if include_fixed_test:
        cv_indices = source.index.astype(int).tolist()
        fixed_test_indices: list[int] = []
        expected_cv = 7936
        expected_fixed_test = 0
    else:
        cv_indices = source.index[~test_mask].astype(int).tolist()
        fixed_test_indices = source.index[test_mask].astype(int).tolist()
        expected_cv = 6936
        expected_fixed_test = 1000

    if (
        len(cv_indices) != expected_cv
        or len(fixed_test_indices) != expected_fixed_test
    ):
        counts = split_norm.value_counts(dropna=False).to_dict()
        raise RuntimeError(
            "内部清单数量不符合预期："
            f"CV={len(cv_indices)}, fixed_test={len(fixed_test_indices)}, "
            f"include_fixed_test={include_fixed_test}, "
            f"split_counts={counts}"
        )

    return cv_indices, fixed_test_indices, split_col


def strip_prefix_if_needed(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            return {key[len(prefix):]: value for key, value in state.items()}
    return state


def load_model(
    checkpoint_path: Path,
    c_spatial: int,
    c_point: int,
    device: torch.device,
):
    from models_swe import create_model

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict")
        if state is None:
            state = checkpoint.get("state_dict")
        if state is None and all(torch.is_tensor(v) for v in checkpoint.values()):
            state = checkpoint
        config = checkpoint.get("config", {})
    else:
        raise RuntimeError("checkpoint 不是字典")

    if state is None:
        raise RuntimeError("checkpoint 中找不到 model_state_dict/state_dict")

    state = strip_prefix_if_needed(state)
    d_model = int(config.get("d_model", 256)) if isinstance(config, dict) else 256
    use_wide_branch = bool(config.get("use_wide_branch", False)) if isinstance(config, dict) else False

    model = create_model(
        model_type="full",
        C_spatial=int(c_spatial),
        C_point=int(c_point),
        d_model=d_model,
        use_wide_branch=use_wide_branch,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "模型权重不完全匹配：\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, config


def build_fold_assignment(
    dataset,
    cv_indices: list[int],
    n_splits: int,
    seed: int,
    station_csv: Path,
    manifest_path: Path,
    include_fixed_test: bool = False,
) -> tuple[dict[int, int], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    # seed仅为旧接口兼容；这里不做随机划分。
    del seed
    from balanced_station_cv10 import (
        create_or_load_manifest,
        mapping_for_dataset_indices,
    )

    manifest, balance_summary, _ = create_or_load_manifest(
        station_csv=station_csv,
        manifest_path=manifest_path,
        n_splits=n_splits,
        high_threshold_mm=80.0,
        force_rebuild=False,
        include_fixed_test=include_fixed_test,
    )
    fold_by_index, records = mapping_for_dataset_indices(
        dataset=dataset,
        cv_indices=cv_indices,
        manifest=manifest,
    )

    split_records = []
    for record in records:
        split_records.append({
            "fold": record["fold"],
            "n_train_stations": record["n_train_stations"],
            "n_test_stations": record["n_test_stations"],
            "n_train_samples": record["n_train_samples"],
            "n_test_samples": record["n_test_samples"],
            "train_stations": record["train_stations"],
            "test_stations": record["test_stations"],
        })

    return fold_by_index, split_records, manifest, balance_summary

def get_era5_mm(dataset, meta: dict[str, Any]) -> float:
    date = pd.to_datetime(
        meta.get("feature_date", meta.get("label_date", meta.get("date")))
    ).to_pydatetime()
    row = int(meta["row"])
    col = int(meta["col"])

    if date not in dataset.label_data:
        return float("nan")

    array, nodata = dataset.label_data[date]
    value = float(array[row, col])

    if nodata is not None and value == nodata:
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    return value


def predict_cv_pool(
    dataset,
    model,
    cv_indices: list[int],
    fold_by_index: dict[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label_min_mm: float,
    label_max_mm: float,
) -> pd.DataFrame:
    subset = Subset(dataset, cv_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    rows: list[dict[str, Any]] = []
    expected_order = list(cv_indices)
    cursor = 0

    with torch.inference_mode():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 3:
                raise RuntimeError(f"DataLoader 返回格式异常: {type(batch)}")

            conv = batch[0].to(device, non_blocking=True)
            point = batch[1].to(device, non_blocking=True)

            prediction_norm = model(conv, point)
            prediction_norm = prediction_norm.detach().float().cpu().numpy().reshape(-1)
            prediction_mm = (
                prediction_norm * (label_max_mm - label_min_mm)
                + label_min_mm
            )

            batch_size_actual = len(prediction_mm)

            if len(batch) >= 6:
                returned_indices = batch[5]
                if torch.is_tensor(returned_indices):
                    returned_indices = returned_indices.detach().cpu().numpy().reshape(-1)
                returned_indices = [int(x) for x in returned_indices]
            else:
                returned_indices = expected_order[cursor:cursor + batch_size_actual]

            expected_batch_indices = expected_order[cursor:cursor + batch_size_actual]
            if returned_indices != expected_batch_indices:
                raise RuntimeError(
                    "Dataset __getitem__ 发生索引重定向，拒绝继续："
                    f"expected={expected_batch_indices[:10]}, "
                    f"returned={returned_indices[:10]}"
                )

            for local_i, dataset_idx in enumerate(returned_indices):
                meta = dataset.meta_index[dataset_idx]
                label_date = pd.to_datetime(
                    meta.get("label_date", meta.get("feature_date", meta.get("date")))
                ).strftime("%Y-%m-%d")
                feature_date = pd.to_datetime(
                    meta.get("feature_date", meta.get("label_date", meta.get("date")))
                ).strftime("%Y-%m-%d")

                target_mm = float(meta["swe"])
                era5_mm = get_era5_mm(dataset, meta)

                rows.append({
                    "dataset_index": dataset_idx,
                    "fold": int(fold_by_index[dataset_idx]),
                    "station_id": normalize_station_id(meta.get("station_id", "unknown")),
                    "label_date": label_date,
                    "feature_date": feature_date,
                    "day_gap": int(meta.get("day_gap", 0)),
                    "row": int(meta["row"]),
                    "col": int(meta["col"]),
                    "target_mm": target_mm,
                    "frozen_prediction_mm": float(prediction_mm[local_i]),
                    "era5_land_mm": era5_mm,
                })

            cursor += batch_size_actual
            if cursor % 500 < batch_size_actual:
                print(f"已预测 {cursor:,}/{len(cv_indices):,}")

    frame = pd.DataFrame(rows)
    if len(frame) != len(cv_indices):
        raise RuntimeError(
            f"预测行数异常: expected={len(cv_indices)}, actual={len(frame)}"
        )
    if frame["dataset_index"].duplicated().any():
        raise RuntimeError("预测结果中存在重复 dataset_index")
    if frame["fold"].nunique() != 10:
        raise RuntimeError(f"fold数量异常: {frame['fold'].nunique()}")
    if (frame["feature_date"] != frame["label_date"]).any():
        raise RuntimeError("仍存在 feature_date != label_date")
    if (frame["day_gap"] != 0).any():
        raise RuntimeError("仍存在 day_gap != 0")

    return frame.sort_values(["fold", "station_id", "label_date"]).reset_index(drop=True)


def build_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for fold in range(1, 11):
        fold_frame = predictions[predictions["fold"] == fold]
        n_stations = int(fold_frame["station_id"].nunique())

        for method, column in [
            ("Frozen", "frozen_prediction_mm"),
            ("ERA5-Land", "era5_land_mm"),
        ]:
            metrics = compute_metrics(
                fold_frame["target_mm"],
                fold_frame[column],
            )
            metrics.update({
                "fold": fold,
                "method": method,
                "n_stations": n_stations,
            })
            records.append(metrics)

    return pd.DataFrame(records).sort_values(["method", "fold"])


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> dict[str, Any]:
    metric_columns = [
        "r",
        "nse",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
        "rmse_obs_ge50_mm",
        "bias_obs_ge80_mm",
        "slope",
        "std_ratio",
    ]

    summary: dict[str, Any] = {}
    for method, group in fold_metrics.groupby("method"):
        method_summary: dict[str, Any] = {}
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                continue
            method_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        summary[method] = method_summary
    return summary


def plot_metric_boxplots(fold_metrics: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("r", "R", "fold_boxplot_R.png"),
        ("nse", "NSE", "fold_boxplot_NSE.png"),
        ("rmse_mm", "RMSE (mm)", "fold_boxplot_RMSE.png"),
        ("mae_mm", "MAE (mm)", "fold_boxplot_MAE.png"),
        ("bias_mm", "Bias (mm)", "fold_boxplot_Bias.png"),
        ("slope", "Regression slope", "fold_boxplot_Slope.png"),
        ("bias_obs_ge80_mm", "Bias for obs ≥ 80 mm", "fold_boxplot_HighSWE_Bias.png"),
    ]

    method_order = ["ERA5-Land", "Frozen"]

    for metric, ylabel, filename in metrics:
        data = []
        labels = []
        for method in method_order:
            values = pd.to_numeric(
                fold_metrics.loc[fold_metrics["method"] == method, metric],
                errors="coerce",
            ).dropna().to_numpy()
            if values.size:
                data.append(values)
                labels.append(method)

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 5.5))
        ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=False,
            widths=0.55,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(f"Station-wise 10-fold distribution: {ylabel}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_scatter(
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, Any],
    method: str,
    output_path: Path,
) -> None:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]

    lower = STATION_SCATTER_AXIS_MIN_MM
    upper = STATION_SCATTER_AXIS_MAX_MM

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(target, prediction, s=14, alpha=0.35)
    ax.plot(
        [lower, upper],
        [lower, upper],
        "--",
        linewidth=1.8,
        label="1:1 line",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ticks = np.arange(
        lower,
        upper + STATION_SCATTER_TICK_MM,
        STATION_SCATTER_TICK_MM,
    )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Station SWE (mm)")
    ax.set_ylabel(f"{method} SWE (mm)")
    ax.set_title(f"{method} vs Station SWE — pooled OOF (N={len(target)})")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    text = (
        f"N = {metrics['n_samples']}\n"
        f"R = {metrics['r']:.4f}\n"
        f"NSE = {metrics['nse']:.4f}\n"
        f"RMSE = {metrics['rmse_mm']:.2f} mm\n"
        f"MAE = {metrics['mae_mm']:.2f} mm\n"
        f"Bias = {metrics['bias_mm']:.2f} mm\n"
        f"Slope = {metrics['slope']:.3f}"
    )
    ax.text(
        0.98,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def _draw_fold_scatter_axis(
    ax,
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, Any],
    title: str,
    upper: float,
) -> None:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]

    lower = STATION_SCATTER_AXIS_MIN_MM
    ax.scatter(target, prediction, s=10, alpha=0.38)
    ax.plot(
        [lower, upper],
        [lower, upper],
        '--',
        linewidth=1.2,
        label='1:1',
    )
    if len(target) >= 2 and np.std(target) > 0:
        slope, intercept = np.polyfit(target, prediction, 1)
        xx = np.array([lower, upper], dtype=np.float64)
        ax.plot(xx, intercept + slope * xx, linewidth=1.4, color='red', label='Fit')

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ticks = np.arange(
        lower,
        upper + STATION_SCATTER_TICK_MM,
        STATION_SCATTER_TICK_MM,
    )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.22)
    ax.text(
        0.03, 0.97,
        (
            f"N={metrics['n_samples']}\n"
            f"R={metrics['r']:.3f}\n"
            f"RMSE={metrics['rmse_mm']:.2f}\n"
            f"MAE={metrics['mae_mm']:.2f}\n"
            f"Bias={metrics['bias_mm']:.2f}"
        ),
        transform=ax.transAxes,
        ha='left', va='top', fontsize=8.2,
    )


def plot_fold_scatter_outputs(
    predictions: pd.DataFrame,
    prediction_column: str,
    method_label: str,
    prefix: str,
    output_dir: Path,
) -> None:
    """Save ten individual fold scatters plus one 2x5 panel."""
    fold_dir = output_dir / f'{prefix}_fold_scatter'
    fold_dir.mkdir(parents=True, exist_ok=True)

    upper = STATION_SCATTER_AXIS_MAX_MM

    panel, axes = plt.subplots(2, 5, figsize=(19, 7.6), sharex=True, sharey=True)
    axes = axes.ravel()
    for fold in range(1, 11):
        fold_frame = predictions[predictions['fold'] == fold]
        target = pd.to_numeric(fold_frame['target_mm'], errors='coerce').to_numpy()
        pred = pd.to_numeric(fold_frame[prediction_column], errors='coerce').to_numpy()
        metrics = compute_metrics(target, pred)
        _draw_fold_scatter_axis(axes[fold - 1], target, pred, metrics, f'Fold {fold}', upper)

        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        _draw_fold_scatter_axis(ax, target, pred, metrics, f'{method_label} — Fold {fold}', upper)
        ax.set_xlabel('Station SWE (mm)')
        ax.set_ylabel(f'{method_label} SWE (mm)')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(fold_dir / f'{prefix}_fold_{fold:02d}_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    for row in range(2):
        axes[row * 5].set_ylabel(f'{method_label} SWE (mm)')
    for col in range(5):
        axes[5 + col].set_xlabel('Station SWE (mm)')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        panel.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    panel.text(
        0.995,
        0.012,
        "All x/y axes fixed at 0–400 mm",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="dimgray",
    )
    panel.suptitle(f'{method_label}: balanced station-wise 10-fold held-out scatter', fontsize=15, fontweight='bold')
    panel.tight_layout(rect=(0, 0.04, 1, 0.95))
    panel.savefig(output_dir / f'{prefix}_station_cv10_fold_scatter_panel.png', dpi=300, bbox_inches='tight')
    plt.close(panel)

def main() -> None:
    args = parse_args()
    args.stage_label = args.stage_label.strip().upper()
    if re.fullmatch(r"M[0-9]+", args.stage_label) is None:
        raise ValueError(
            f"--stage-label必须形如M0、M1、...，当前={args.stage_label!r}"
        )

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)

    args.root = args.root.expanduser().resolve()
    args.station_csv = args.station_csv.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.normalization_config = args.normalization_config.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.fold_manifest is None:
        manifest_name = (
            "balanced_station_nested_cv10_all7936_manifest.csv"
            if args.include_fixed_test
            else "balanced_station_cv10_manifest.csv"
        )
        args.fold_manifest = (
            args.root
            / "shared_cache"
            / "progressive_finetune"
            / manifest_name
        )
    args.fold_manifest = args.fold_manifest.expanduser().resolve()

    for required in [
        args.station_csv,
        args.checkpoint,
        args.normalization_config,
        args.root / "data_station_online_swe.py",
        args.root / "models_swe.py",
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            args.root
            / "experiments"
            / f"frozen_{args.stage_label}_station_cv10_{timestamp}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 100)
    print(f"Frozen {args.stage_label}：确定性平衡站点级10折内部评估")
    print("=" * 100)
    print(f"内部清单: {args.station_csv}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"输出目录: {output_dir}")
    print(f"device: {device}")
    print("=" * 100)

    sys.path.insert(0, str(args.root))
    from data_station_online_swe import StationSWEDataset

    source = pd.read_csv(args.station_csv)
    cv_indices, fixed_test_indices, split_col = detect_cv_source_indices(
        source,
        include_fixed_test=args.include_fixed_test,
    )
    print(f"split列: {split_col}")
    print(f"CV池: {len(cv_indices):,}")
    if args.include_fixed_test:
        print("旧固定1000条: 已并回Nested CV池")
    else:
        print(f"旧固定测试（本次不使用）: {len(fixed_test_indices):,}")

    dataset = StationSWEDataset(
        station_csv=args.station_csv,
        year_target=[2015, 2016, 2017, 2018],
        fine_tune_mode=True,
        load_fused_swe=True,
        coordinate_jitter_std=0.0,
        microwave_noise_std=0.0,
        coordinate_mask_prob=0.0,
        use_tta=False,
        cache_dir=args.cache_dir,
        shared_cache_mode=True,
        use_product_correction=False,
        normalization_config_path=args.normalization_config,
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

    if len(dataset) != len(source):
        raise RuntimeError(
            f"Dataset/CSV长度不一致: dataset={len(dataset)}, csv={len(source)}"
        )

    fold_by_index, split_records, fold_manifest, balance_summary = build_fold_assignment(
        dataset=dataset,
        cv_indices=cv_indices,
        n_splits=args.n_splits,
        seed=args.seed,
        station_csv=args.station_csv,
        manifest_path=args.fold_manifest,
        include_fixed_test=args.include_fixed_test,
    )

    fold_manifest.to_csv(
        output_dir / "balanced_station_cv10_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    balance_summary.to_csv(
        output_dir / "balanced_station_cv10_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (output_dir / "station_cv10_splits.json").write_text(
        json.dumps(split_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n十折划分:")
    for record in split_records:
        print(
            f"  Fold {record['fold']:02d}: "
            f"test stations={record['n_test_stations']}, "
            f"test samples={record['n_test_samples']}"
        )

    model, checkpoint_config = load_model(
        checkpoint_path=args.checkpoint,
        c_spatial=int(dataset.C_conv),
        c_point=int(dataset.C_point),
        device=device,
    )

    label_min_mm = float(getattr(dataset, "swe_min", 0.0))
    label_max_mm = float(getattr(dataset, "swe_max", 400.0))
    if not np.isclose(label_min_mm, 0.0) or not np.isclose(label_max_mm, 400.0):
        print(
            "⚠ Dataset SWE范围不是预期[0,400]，"
            f"实际=[{label_min_mm},{label_max_mm}]"
        )

    predictions = predict_cv_pool(
        dataset=dataset,
        model=model,
        cv_indices=cv_indices,
        fold_by_index=fold_by_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        label_min_mm=label_min_mm,
        label_max_mm=label_max_mm,
    )

    predictions.to_csv(
        output_dir / "frozen_station_cv10_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics = build_fold_metrics(predictions)
    fold_metrics.to_csv(
        output_dir / "frozen_station_cv10_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled = {
        "Frozen": compute_metrics(
            predictions["target_mm"],
            predictions["frozen_prediction_mm"],
        ),
        "ERA5-Land": compute_metrics(
            predictions["target_mm"],
            predictions["era5_land_mm"],
        ),
    }

    fold_summary = summarize_fold_metrics(fold_metrics)

    summary = {
        "created_at": datetime.now().isoformat(),
        "stage_label": args.stage_label,
        "protocol": {
            "internal_evaluation": (
                f"station-wise 10-fold; each of {len(cv_indices)} internal "
                "samples is evaluated "
                "exactly once by the unchanged Frozen model"
            ),
            "fixed_internal_1000_merged_into_cv": bool(
                args.include_fixed_test
            ),
            "n_splits": args.n_splits,
            "split_method": "deterministic_balanced_greedy_v1",
            "randomized": False,
            "balance_targets": [
                "total_sample_count",
                "high_swe_sample_count_ge_80mm",
                "station_count_light_weight",
            ],
            "fold_manifest": str(args.fold_manifest),
            "fold_manifest_sha256": sha256_file(args.fold_manifest),
            "legacy_seed_argument_ignored": args.seed,
            "n_cv_samples": len(cv_indices),
            "n_fixed_test_samples_excluded": len(fixed_test_indices),
            "n_unique_cv_stations": int(predictions["station_id"].nunique()),
            "scatter_axis_mm": [
                STATION_SCATTER_AXIS_MIN_MM,
                STATION_SCATTER_AXIS_MAX_MM,
            ],
        },
        "files": {
            "station_csv": str(args.station_csv),
            "station_csv_sha256": sha256_file(args.station_csv),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "normalization_config": str(args.normalization_config),
            "normalization_config_sha256": sha256_file(args.normalization_config),
        },
        "checkpoint_config": checkpoint_config,
        "pooled_oof_metrics": pooled,
        "fold_distribution_summary": fold_summary,
        "metric_definition": {
            "primary_overall": "pooled OOF metrics computed after concatenating all ten held-out folds",
            "fold_distribution": "ten fold-specific values summarized by mean/std/median/boxplot",
            "note": "nonlinear metrics such as R and NSE are not obtained by averaging fold values",
        },
    }

    (output_dir / "frozen_station_cv10_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_metric_boxplots(fold_metrics, output_dir)
    plot_scatter(
        predictions["target_mm"].to_numpy(),
        predictions["frozen_prediction_mm"].to_numpy(),
        pooled["Frozen"],
        f"Frozen {args.stage_label}",
        output_dir / "frozen_station_cv10_pooled_scatter.png",
    )
    plot_scatter(
        predictions["target_mm"].to_numpy(),
        predictions["era5_land_mm"].to_numpy(),
        pooled["ERA5-Land"],
        "ERA5-Land",
        output_dir / "era5_station_cv10_pooled_scatter.png",
    )

    plot_fold_scatter_outputs(
        predictions,
        prediction_column="frozen_prediction_mm",
        method_label=f"Frozen {args.stage_label}",
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

    print("\n" + "=" * 100)
    print("内部Frozen平衡站点10折评估完成")
    print("=" * 100)
    for method in ["ERA5-Land", "Frozen"]:
        item = pooled[method]
        print(
            f"{method:<12s} pooled OOF: "
            f"N={item['n_samples']}, "
            f"R={item['r']:.4f}, NSE={item['nse']:.4f}, "
            f"RMSE={item['rmse_mm']:.2f}, MAE={item['mae_mm']:.2f}, "
            f"Bias={item['bias_mm']:.2f}, slope={item['slope']:.3f}"
        )
    print(f"输出目录: {output_dir}")
    print("=" * 100)

    del model
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
