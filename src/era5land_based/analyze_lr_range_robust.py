# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def round_significant(value: float, digits: int = 1) -> float:
    """将数值保留指定有效数字，例如1.2e-4约为1e-4。"""
    if not np.isfinite(value) or value == 0:
        return float(value)

    places = digits - int(np.floor(np.log10(abs(value)))) - 1
    return float(round(value, places))


def local_linear_slope(
    x: np.ndarray,
    y: np.ndarray,
    center: int,
    half_window: int,
) -> float:
    """
    在center附近窗口内拟合：
        y = slope * x + intercept

    x使用log10(LR)，因此这里得到的是
    loss相对于log10(LR)的持续变化斜率。
    """
    left = center - half_window
    right = center + half_window + 1

    x_window = x[left:right]
    y_window = y[left:right]

    x_centered = x_window - x_window.mean()
    denominator = np.sum(x_centered ** 2)

    if denominator <= 0:
        return np.nan

    slope = np.sum(
        x_centered * (y_window - y_window.mean())
    ) / denominator

    return float(slope)


def robust_lr_analysis(
    history: pd.DataFrame,
    discard_start_fraction: float = 0.15,
    slope_window_fraction: float = 0.10,
) -> dict:
    required_columns = {
        "step",
        "lr",
        "raw_loss",
        "smooth_loss",
    }

    missing = required_columns - set(history.columns)
    if missing:
        raise ValueError(
            f"CSV缺少必要字段: {sorted(missing)}"
        )

    history = history.copy()

    for column in required_columns:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        )

    valid = (
        np.isfinite(history["lr"])
        & np.isfinite(history["smooth_loss"])
        & (history["lr"] > 0)
    )

    history = (
        history.loc[valid]
        .sort_values("lr")
        .reset_index(drop=True)
    )

    n = len(history)

    if n < 50:
        raise ValueError(
            f"有效点数只有{n}，不足以做稳健分析"
        )

    lr = history["lr"].to_numpy(dtype=np.float64)
    raw_loss = history["raw_loss"].to_numpy(
        dtype=np.float64
    )
    smooth_loss = history["smooth_loss"].to_numpy(
        dtype=np.float64
    )
    log_lr = np.log10(lr)

    # --------------------------------------------------------
    # 第二次稳健平滑
    #
    # 第一层：滚动中位数，压制异常batch；
    # 第二层：滚动均值，获得连续趋势。
    # --------------------------------------------------------
    robust_loss = (
        pd.Series(smooth_loss)
        .rolling(
            window=11,
            center=True,
            min_periods=1,
        )
        .median()
        .rolling(
            window=9,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy(dtype=np.float64)
    )

    # --------------------------------------------------------
    # 丢弃开头15%的点。
    #
    # Range Test刚开始时：
    # 1. 第一个batch可能偶然偏高；
    # 2. EMA尚未稳定；
    # 3. 参数几乎没有更新；
    # 因此不能参与最陡下降点识别。
    # --------------------------------------------------------
    search_start = max(
        20,
        int(round(n * discard_start_fraction)),
    )

    if search_start >= n - 20:
        raise ValueError(
            "discard_start_fraction过大，剩余点不足"
        )

    # 原smooth_loss最低点
    minimum_index = search_start + int(
        np.argmin(smooth_loss[search_start:])
    )

    # 稳健趋势最低点
    robust_minimum_index = search_start + int(
        np.argmin(robust_loss[search_start:])
    )

    # --------------------------------------------------------
    # 用约总点数10%的窗口进行局部线性回归。
    #
    # 400点时窗口约41点，覆盖约0.6个LR数量级，
    # 能识别“持续下降”，而不是某两个batch的偶然跳动。
    # --------------------------------------------------------
    slope_window = max(
        21,
        int(round(n * slope_window_fraction)),
    )

    if slope_window % 2 == 0:
        slope_window += 1

    half_window = slope_window // 2

    slopes = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    # 只在：
    #   丢弃起始噪声之后
    #   稳健最低点之前
    # 搜索下降最快区间。
    slope_search_start = max(
        search_start,
        half_window,
    )

    slope_search_end = min(
        robust_minimum_index - half_window,
        n - half_window,
    )

    if slope_search_end <= slope_search_start:
        raise ValueError(
            "最低点之前没有足够窗口用于斜率分析"
        )

    for center in range(
        slope_search_start,
        slope_search_end,
    ):
        slopes[center] = local_linear_slope(
            log_lr,
            robust_loss,
            center,
            half_window,
        )

    valid_slope_indices = np.where(
        np.isfinite(slopes)
    )[0]

    if len(valid_slope_indices) == 0:
        raise RuntimeError(
            "没有得到有效局部斜率"
        )

    steepest_index = valid_slope_indices[
        np.argmin(slopes[valid_slope_indices])
    ]

    minimum_loss_lr = float(lr[minimum_index])
    minimum_loss = float(
        smooth_loss[minimum_index]
    )

    robust_minimum_lr = float(
        lr[robust_minimum_index]
    )
    robust_minimum_loss = float(
        robust_loss[robust_minimum_index]
    )

    steepest_sustained_lr = float(
        lr[steepest_index]
    )
    steepest_sustained_slope = float(
        slopes[steepest_index]
    )

    # --------------------------------------------------------
    # 候选峰值学习率
    #
    # safe_peak:
    #   最低点LR / 10，偏保守。
    #
    # balanced_peak:
    #   最低点LR / 3，仍位于最低点左侧，
    #   更接近工程上的峰值候选。
    # --------------------------------------------------------
    safe_peak_lr = minimum_loss_lr / 10.0
    balanced_peak_lr = minimum_loss_lr / 3.0

    balanced_peak_rounded = round_significant(
        balanced_peak_lr,
        digits=1,
    )

    result = {
        "available": True,
        "n_points": int(n),
        "discard_start_fraction": float(
            discard_start_fraction
        ),
        "discarded_start_points": int(
            search_start
        ),
        "slope_window_points": int(
            slope_window
        ),
        "minimum_loss_lr": minimum_loss_lr,
        "minimum_loss": minimum_loss,
        "robust_minimum_lr": robust_minimum_lr,
        "robust_minimum_loss": robust_minimum_loss,
        "steepest_sustained_lr": (
            steepest_sustained_lr
        ),
        "steepest_sustained_slope": (
            steepest_sustained_slope
        ),
        "safe_peak_lr_minimum_div10": float(
            safe_peak_lr
        ),
        "balanced_peak_lr_minimum_div3": float(
            balanced_peak_lr
        ),
        "balanced_peak_lr_rounded": float(
            balanced_peak_rounded
        ),
        "candidate_peak_lr_lower": float(
            safe_peak_lr
        ),
        "candidate_peak_lr_upper": float(
            balanced_peak_lr
        ),
    }

    history["robust_loss"] = robust_loss
    history["local_slope"] = slopes

    return {
        "summary": result,
        "history": history,
        "indices": {
            "minimum": int(minimum_index),
            "robust_minimum": int(
                robust_minimum_index
            ),
            "steepest": int(steepest_index),
            "search_start": int(search_start),
        },
    }


def make_plot(
    analyzed: dict,
    output_path: Path,
) -> None:
    history = analyzed["history"]
    summary = analyzed["summary"]

    lr = history["lr"].to_numpy()
    raw_loss = history["raw_loss"].to_numpy()
    smooth_loss = history["smooth_loss"].to_numpy()
    robust_loss = history["robust_loss"].to_numpy()

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    axis.plot(
        lr,
        raw_loss,
        alpha=0.20,
        linewidth=0.8,
        label="Raw batch loss",
    )

    axis.plot(
        lr,
        smooth_loss,
        linewidth=1.5,
        label="Original EMA loss",
    )

    axis.plot(
        lr,
        robust_loss,
        linewidth=2.3,
        label="Robust trend",
    )

    axis.axvline(
        summary["steepest_sustained_lr"],
        linestyle="--",
        linewidth=1.5,
        label=(
            "Steepest sustained descent "
            f"{summary['steepest_sustained_lr']:.2e}"
        ),
    )

    axis.axvline(
        summary["minimum_loss_lr"],
        linestyle="--",
        linewidth=1.5,
        label=(
            "Minimum loss "
            f"{summary['minimum_loss_lr']:.2e}"
        ),
    )

    axis.axvline(
        summary["safe_peak_lr_minimum_div10"],
        linestyle=":",
        linewidth=1.5,
        label=(
            "Minimum LR / 10 "
            f"{summary['safe_peak_lr_minimum_div10']:.2e}"
        ),
    )

    axis.axvline(
        summary["balanced_peak_lr_rounded"],
        linestyle="-.",
        linewidth=1.5,
        label=(
            "Balanced peak "
            f"{summary['balanced_peak_lr_rounded']:.2e}"
        ),
    )

    axis.set_xscale("log")
    axis.set_xlabel("Learning rate")
    axis.set_ylabel("Loss")
    axis.set_title(
        "Robust LR Range Test Analysis"
    )
    axis.grid(
        True,
        which="both",
        alpha=0.25,
    )
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="lr_range_history.csv路径",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，默认与CSV同目录",
    )

    parser.add_argument(
        "--discard-start-fraction",
        type=float,
        default=0.15,
        help="丢弃开头点的比例",
    )

    parser.add_argument(
        "--slope-window-fraction",
        type=float,
        default=0.10,
        help="局部斜率窗口占总点数比例",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else csv_path.parent
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = pd.read_csv(csv_path)

    analyzed = robust_lr_analysis(
        history,
        discard_start_fraction=(
            args.discard_start_fraction
        ),
        slope_window_fraction=(
            args.slope_window_fraction
        ),
    )

    summary_path = (
        output_dir
        / "lr_range_summary_robust.json"
    )
    plot_path = (
        output_dir
        / "lr_range_test_robust.png"
    )
    history_path = (
        output_dir
        / "lr_range_history_robust.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            analyzed["summary"],
            file,
            indent=2,
            ensure_ascii=False,
        )

    analyzed["history"].to_csv(
        history_path,
        index=False,
    )

    make_plot(
        analyzed,
        plot_path,
    )

    summary = analyzed["summary"]

    print("=" * 78)
    print("稳健LR Range Test分析完成")
    print("=" * 78)
    print(
        "持续下降最快处LR: "
        f"{summary['steepest_sustained_lr']:.3e}"
    )
    print(
        "原平滑loss最低点LR: "
        f"{summary['minimum_loss_lr']:.3e}"
    )
    print(
        "稳健趋势最低点LR: "
        f"{summary['robust_minimum_lr']:.3e}"
    )
    print(
        "保守峰值候选（最低点/10）: "
        f"{summary['safe_peak_lr_minimum_div10']:.3e}"
    )
    print(
        "平衡峰值候选（最低点/3）: "
        f"{summary['balanced_peak_lr_minimum_div3']:.3e}"
    )
    print(
        "平衡峰值取整值: "
        f"{summary['balanced_peak_lr_rounded']:.3e}"
    )
    print(f"JSON: {summary_path}")
    print(f"曲线: {plot_path}")
    print(f"CSV:  {history_path}")


if __name__ == "__main__":
    main()
