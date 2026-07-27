# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import deque
from pathlib import Path

import numpy as np
import torch


class StabilityMonitor:
    """
    训练初期稳定性监控。

    记录：
    - step loss、EMA loss、滑动窗口 loss CV
    - 裁剪前全局梯度范数
    - 是否触发梯度裁剪
    - 参数更新范数与 update ratio
    - prediction / target mean、std、range
    - AMP 是否跳过 optimizer.step
    - 启发式 STABLE / WARN / CRITICAL

    注意：
    update ratio 是通过 optimizer.step 前后参数差计算的。
    因此前 N 步会增加显存和运行时间；达到 max_steps 后自动停止。
    """

    FIELDS = [
        "global_step",
        "epoch",
        "batch",
        "lr_min",
        "lr_max",

        "loss",
        "loss_ema",
        "loss_roll_mean",
        "loss_roll_std",
        "loss_cv",
        "loss_spike_ratio",

        "grad_norm_before_clip",
        "grad_ema",
        "grad_spike_ratio",
        "clip_threshold",
        "clip_triggered",
        "clip_rate_window",

        "param_norm_before",
        "update_norm",
        "update_ratio",
        "update_ratio_ema",

        "pred_mean_norm",
        "pred_std_norm",
        "pred_min_norm",
        "pred_max_norm",
        "target_mean_norm",
        "target_std_norm",

        "pred_mean_mm",
        "pred_std_mm",
        "target_mean_mm",
        "target_std_mm",

        "pred_std_ratio",
        "pred_target_mean_gap_std",

        "amp_scale_before",
        "amp_scale_after",
        "amp_step_skipped",

        "finite",
        "status",
        "reasons",
    ]

    @staticmethod
    def enabled_from_env() -> bool:
        value = os.environ.get("STABILITY_MONITOR", "0")
        return value.strip().lower() in {
            "1", "true", "yes", "on"
        }

    @classmethod
    def from_env(
        cls,
        save_dir,
        default_prefix: str = "stability",
    ) -> "StabilityMonitor":
        return cls(
            save_dir=save_dir,
            prefix=os.environ.get(
                "STABILITY_PREFIX",
                default_prefix,
            ),
            max_steps=int(
                os.environ.get(
                    "STABILITY_MAX_STEPS",
                    "300",
                )
            ),
            log_every=int(
                os.environ.get(
                    "STABILITY_LOG_EVERY",
                    "1",
                )
            ),
            window=int(
                os.environ.get(
                    "STABILITY_WINDOW",
                    "50",
                )
            ),
            ema_beta=float(
                os.environ.get(
                    "STABILITY_EMA_BETA",
                    "0.98",
                )
            ),
        )

    def __init__(
        self,
        save_dir,
        prefix: str = "stability",
        max_steps: int = 300,
        log_every: int = 1,
        window: int = 50,
        ema_beta: float = 0.98,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prefix = re.sub(
            r"[^0-9A-Za-z._-]+",
            "_",
            str(prefix),
        ).strip("_")

        prefix = prefix or "stability"

        self.csv_path = (
            self.save_dir
            / f"{prefix}_steps.csv"
        )
        self.summary_path = (
            self.save_dir
            / f"{prefix}_summary.json"
        )

        self.max_steps = max(
            1,
            int(max_steps),
        )
        self.log_every = max(
            1,
            int(log_every),
        )
        self.window = max(
            10,
            int(window),
        )
        self.ema_beta = float(ema_beta)

        self.global_step = 0

        self.loss_ema_raw = 0.0
        self.grad_ema_raw = 0.0
        self.update_ema_raw = 0.0

        self.loss_window = deque(
            maxlen=self.window
        )
        self.grad_window = deque(
            maxlen=self.window
        )
        self.update_window = deque(
            maxlen=self.window
        )
        self.clip_window = deque(
            maxlen=self.window
        )

        self.rows = []

        self.status_counts = {
            "STABLE": 0,
            "WARN": 0,
            "CRITICAL": 0,
        }

        self.finished = False

        self.csv_file = self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.FIELDS,
        )
        self.writer.writeheader()
        self.csv_file.flush()

        print("\n" + "=" * 80)
        print("📈 启用训练稳定性监控")
        print(
            f"   max_steps  : "
            f"{self.max_steps}"
        )
        print(
            f"   EMA beta   : "
            f"{self.ema_beta}"
        )
        print(
            f"   window     : "
            f"{self.window}"
        )
        print(
            f"   CSV        : "
            f"{self.csv_path}"
        )
        print(
            f"   Summary    : "
            f"{self.summary_path}"
        )
        print("=" * 80)

    @property
    def active(self) -> bool:
        return (
            not self.finished
            and self.global_step < self.max_steps
        )

    @staticmethod
    def safe_float(value) -> float:
        try:
            value = float(value)
        except Exception:
            return float("nan")

        if not math.isfinite(value):
            return float("nan")

        return value

    @staticmethod
    def global_grad_norm(model) -> float:
        """计算当前裁剪前的全局L2梯度范数。"""

        total_sq = 0.0

        for parameter in model.parameters():
            if (
                parameter.requires_grad
                and parameter.grad is not None
            ):
                grad = (
                    parameter.grad
                    .detach()
                    .float()
                )

                total_sq += float(
                    torch.sum(
                        grad * grad
                    ).item()
                )

        return math.sqrt(total_sq)

    @staticmethod
    def snapshot_parameters(model):
        """
        保存 optimizer.step 前的参数，用来计算实际更新量。

        只保存：
        - requires_grad=True
        - 当前存在梯度
        的参数。
        """

        snapshots = []
        parameter_sq = 0.0

        with torch.no_grad():
            for parameter in model.parameters():
                if (
                    parameter.requires_grad
                    and parameter.grad is not None
                ):
                    before = (
                        parameter
                        .detach()
                        .clone()
                    )

                    snapshots.append(
                        (parameter, before)
                    )

                    before_float = before.float()

                    parameter_sq += float(
                        torch.sum(
                            before_float
                            * before_float
                        ).item()
                    )

        return {
            "items": snapshots,
            "param_norm": math.sqrt(
                parameter_sq
            ),
        }

    @staticmethod
    def compute_update(snapshot):
        """
        update_ratio =
            ||theta_after - theta_before||
            / (||theta_before|| + eps)
        """

        if (
            snapshot is None
            or not snapshot["items"]
        ):
            return (
                float("nan"),
                float("nan"),
                float("nan"),
            )

        update_sq = 0.0

        with torch.no_grad():
            for parameter, before in snapshot["items"]:
                delta = (
                    parameter.detach()
                    - before
                ).float()

                update_sq += float(
                    torch.sum(
                        delta * delta
                    ).item()
                )

        update_norm = math.sqrt(
            update_sq
        )

        param_norm = float(
            snapshot["param_norm"]
        )

        update_ratio = (
            update_norm
            / (param_norm + 1e-12)
        )

        return (
            param_norm,
            update_norm,
            update_ratio,
        )

    def update_ema(
        self,
        value: float,
        state_name: str,
    ) -> float:
        """
        带启动偏差修正的EMA。
        """

        if not math.isfinite(value):
            return float("nan")

        previous = float(
            getattr(self, state_name)
        )

        updated = (
            self.ema_beta * previous
            + (1.0 - self.ema_beta)
            * value
        )

        setattr(
            self,
            state_name,
            updated,
        )

        correction = (
            1.0
            - self.ema_beta
            ** max(1, self.global_step)
        )

        return updated / max(
            correction,
            1e-12,
        )

    @staticmethod
    def window_stats(values):
        array = np.asarray(
            list(values),
            dtype=np.float64,
        )

        array = array[
            np.isfinite(array)
        ]

        if array.size == 0:
            return (
                float("nan"),
                float("nan"),
                float("nan"),
            )

        mean = float(
            array.mean()
        )

        std = float(
            array.std(ddof=0)
        )

        cv = std / (
            abs(mean) + 1e-12
        )

        return mean, std, cv

    def classify(
        self,
        *,
        finite: bool,
        amp_step_skipped: bool,
        loss_spike_ratio: float,
        loss_cv: float,
        grad_spike_ratio: float,
        clip_triggered: bool,
        update_ratio: float,
        pred_std_ratio: float,
        mean_gap_std: float,
    ):
        """
        启发式状态，不是通用科学阈值。

        CRITICAL:
        - NaN/Inf
        - AMP跳过更新
        - 单步参数变化比例 > 1%
        - loss超过EMA的5倍

        WARN:
        - 发生梯度裁剪
        - update ratio > 0.1%
        - loss CV > 0.5
        - 梯度超过EMA的5倍
        - 预测方差明显塌缩或爆炸
        """

        reasons = []
        critical = False
        warning = False

        if not finite:
            critical = True
            reasons.append(
                "nonfinite"
            )

        if amp_step_skipped:
            critical = True
            reasons.append(
                "amp_step_skipped"
            )

        if (
            math.isfinite(update_ratio)
            and update_ratio > 1e-2
        ):
            critical = True
            reasons.append(
                "update_ratio>1e-2"
            )

        # 前10步不采用EMA尖峰判断，
        # 避免启动偏差造成误报。
        if self.global_step >= 10:
            if (
                math.isfinite(
                    loss_spike_ratio
                )
                and loss_spike_ratio > 5.0
            ):
                critical = True
                reasons.append(
                    "loss_spike>5x_ema"
                )

            elif (
                math.isfinite(
                    loss_spike_ratio
                )
                and loss_spike_ratio > 2.0
            ):
                warning = True
                reasons.append(
                    "loss_spike>2x_ema"
                )

            if (
                math.isfinite(loss_cv)
                and loss_cv > 0.50
            ):
                warning = True
                reasons.append(
                    "loss_cv>0.50"
                )

            if (
                math.isfinite(
                    grad_spike_ratio
                )
                and grad_spike_ratio > 5.0
            ):
                warning = True
                reasons.append(
                    "grad_spike>5x_ema"
                )

        if clip_triggered:
            warning = True
            reasons.append(
                "gradient_clipped"
            )

        if (
            math.isfinite(update_ratio)
            and update_ratio > 1e-3
        ):
            warning = True
            reasons.append(
                "update_ratio>1e-3"
            )

        if math.isfinite(
            pred_std_ratio
        ):
            if pred_std_ratio < 0.05:
                warning = True
                reasons.append(
                    "prediction_std_collapse"
                )

            elif pred_std_ratio > 5.0:
                warning = True
                reasons.append(
                    "prediction_std_excessive"
                )

        if (
            math.isfinite(mean_gap_std)
            and mean_gap_std > 5.0
        ):
            warning = True
            reasons.append(
                "prediction_mean_far_from_target"
            )

        if critical:
            return (
                "CRITICAL",
                ";".join(reasons),
            )

        if warning:
            return (
                "WARN",
                ";".join(reasons),
            )

        return "STABLE", ""

    def record(
        self,
        *,
        epoch: int,
        batch: int,
        lr_values,
        loss: float,
        grad_norm: float,
        clip_threshold: float,
        clip_triggered: bool,
        param_snapshot,
        pred,
        target,
        swe_min: float,
        swe_max: float,
        amp_scale_before: float,
        amp_scale_after: float,
        amp_step_skipped: bool,
    ):
        if not self.active:
            return

        self.global_step += 1

        loss = self.safe_float(
            loss
        )

        grad_norm = self.safe_float(
            grad_norm
        )

        (
            param_norm,
            update_norm,
            update_ratio,
        ) = self.compute_update(
            param_snapshot
        )

        pred = (
            pred.detach()
            .float()
            .reshape(-1)
        )

        target = (
            target.detach()
            .float()
            .reshape(-1)
        )

        pred_mean = float(
            pred.mean().item()
        )
        pred_std = float(
            pred.std(
                unbiased=False
            ).item()
        )
        pred_min = float(
            pred.min().item()
        )
        pred_max = float(
            pred.max().item()
        )

        target_mean = float(
            target.mean().item()
        )
        target_std = float(
            target.std(
                unbiased=False
            ).item()
        )

        swe_min = float(
            swe_min
        )
        swe_max = float(
            swe_max
        )

        swe_range = (
            swe_max - swe_min
        )

        pred_mean_mm = (
            pred_mean
            * swe_range
            + swe_min
        )

        pred_std_mm = (
            pred_std
            * abs(swe_range)
        )

        target_mean_mm = (
            target_mean
            * swe_range
            + swe_min
        )

        target_std_mm = (
            target_std
            * abs(swe_range)
        )

        if target_std > 1e-8:
            pred_std_ratio = (
                pred_std
                / target_std
            )

            mean_gap_std = (
                abs(
                    pred_mean
                    - target_mean
                )
                / target_std
            )
        else:
            pred_std_ratio = float("nan")
            mean_gap_std = float("nan")

        loss_ema = self.update_ema(
            loss,
            "loss_ema_raw",
        )

        grad_ema = self.update_ema(
            grad_norm,
            "grad_ema_raw",
        )

        update_ratio_ema = self.update_ema(
            update_ratio,
            "update_ema_raw",
        )

        self.loss_window.append(
            loss
        )

        self.grad_window.append(
            grad_norm
        )

        self.update_window.append(
            update_ratio
        )

        self.clip_window.append(
            1.0
            if clip_triggered
            else 0.0
        )

        (
            loss_roll_mean,
            loss_roll_std,
            loss_cv,
        ) = self.window_stats(
            self.loss_window
        )

        clip_rate_window = float(
            np.mean(
                self.clip_window
            )
        )

        loss_spike_ratio = (
            loss
            / (loss_ema + 1e-12)
        )

        grad_spike_ratio = (
            grad_norm
            / (grad_ema + 1e-12)
        )

        lr_values = [
            float(value)
            for value in lr_values
        ]

        lr_min = min(
            lr_values
        )
        lr_max = max(
            lr_values
        )

        finite_values = [
            loss,
            loss_ema,
            grad_norm,
            param_norm,
            update_norm,
            update_ratio,
            pred_mean,
            pred_std,
            target_mean,
            target_std,
        ]

        finite = all(
            math.isfinite(value)
            for value in finite_values
        )

        status, reasons = self.classify(
            finite=finite,
            amp_step_skipped=amp_step_skipped,
            loss_spike_ratio=loss_spike_ratio,
            loss_cv=loss_cv,
            grad_spike_ratio=grad_spike_ratio,
            clip_triggered=clip_triggered,
            update_ratio=update_ratio,
            pred_std_ratio=pred_std_ratio,
            mean_gap_std=mean_gap_std,
        )

        self.status_counts[
            status
        ] += 1

        row = {
            "global_step": self.global_step,
            "epoch": int(epoch),
            "batch": int(batch),
            "lr_min": lr_min,
            "lr_max": lr_max,

            "loss": loss,
            "loss_ema": loss_ema,
            "loss_roll_mean": loss_roll_mean,
            "loss_roll_std": loss_roll_std,
            "loss_cv": loss_cv,
            "loss_spike_ratio": loss_spike_ratio,

            "grad_norm_before_clip": grad_norm,
            "grad_ema": grad_ema,
            "grad_spike_ratio": grad_spike_ratio,
            "clip_threshold": clip_threshold,
            "clip_triggered": int(
                clip_triggered
            ),
            "clip_rate_window": clip_rate_window,

            "param_norm_before": param_norm,
            "update_norm": update_norm,
            "update_ratio": update_ratio,
            "update_ratio_ema": update_ratio_ema,

            "pred_mean_norm": pred_mean,
            "pred_std_norm": pred_std,
            "pred_min_norm": pred_min,
            "pred_max_norm": pred_max,
            "target_mean_norm": target_mean,
            "target_std_norm": target_std,

            "pred_mean_mm": pred_mean_mm,
            "pred_std_mm": pred_std_mm,
            "target_mean_mm": target_mean_mm,
            "target_std_mm": target_std_mm,

            "pred_std_ratio": pred_std_ratio,
            "pred_target_mean_gap_std": mean_gap_std,

            "amp_scale_before": amp_scale_before,
            "amp_scale_after": amp_scale_after,
            "amp_step_skipped": int(
                amp_step_skipped
            ),

            "finite": int(finite),
            "status": status,
            "reasons": reasons,
        }

        self.rows.append(
            row
        )

        self.writer.writerow(
            row
        )

        self.csv_file.flush()

        if (
            self.global_step
            % self.log_every
            == 0
        ):
            print(
                f"    [STABILITY] "
                f"step={self.global_step:4d} "
                f"| loss={loss:.6f} "
                f"ema={loss_ema:.6f} "
                f"| grad={grad_norm:.3e} "
                f"| clip="
                f"{'Y' if clip_triggered else 'N'} "
                f"| update={update_ratio:.3e} "
                f"| pred="
                f"{pred_mean_mm:.2f}"
                f"±{pred_std_mm:.2f} mm "
                f"| {status}"
                + (
                    f" ({reasons})"
                    if reasons
                    else ""
                )
            )

        if (
            self.global_step % 25 == 0
            or self.global_step
            >= self.max_steps
        ):
            self.save_summary()

        if (
            self.global_step
            >= self.max_steps
        ):
            self.finished = True

            print(
                "\n✅ 稳定性监控达到设定步数。"
            )
            print(
                "   后续训练不再克隆参数，"
                "诊断开销恢复为零。"
            )
            print(
                f"   CSV: "
                f"{self.csv_path}"
            )
            print(
                f"   Summary: "
                f"{self.summary_path}"
            )

    def metric_stats(
        self,
        key: str,
    ):
        values = np.asarray(
            [
                row[key]
                for row in self.rows
            ],
            dtype=np.float64,
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            return {
                "count": 0,
            }

        return {
            "count": int(
                values.size
            ),
            "mean": float(
                values.mean()
            ),
            "median": float(
                np.median(values)
            ),
            "p95": float(
                np.percentile(
                    values,
                    95,
                )
            ),
            "max": float(
                values.max()
            ),
        }

    def save_summary(self):
        if not self.rows:
            return

        n = len(
            self.rows
        )

        clip_rate = float(
            np.mean(
                [
                    row["clip_triggered"]
                    for row in self.rows
                ]
            )
        )

        skipped_rate = float(
            np.mean(
                [
                    row["amp_step_skipped"]
                    for row in self.rows
                ]
            )
        )

        nonfinite_rate = float(
            np.mean(
                [
                    1 - row["finite"]
                    for row in self.rows
                ]
            )
        )

        critical_rate = (
            self.status_counts["CRITICAL"]
            / n
        )

        warn_rate = (
            self.status_counts["WARN"]
            / n
        )

        if (
            critical_rate > 0
            or skipped_rate > 0
            or nonfinite_rate > 0
        ):
            overall = "UNSTABLE"

        elif (
            clip_rate > 0.50
            or warn_rate > 0.50
        ):
            overall = "NEEDS_ATTENTION"

        else:
            overall = "MACRO_STABLE"

        summary = {
            "heuristic_overall_status": overall,

            "note": (
                "该状态是工程诊断启发式，"
                "不是跨模型通用阈值。"
            ),

            "recorded_steps": n,
            "max_steps": self.max_steps,
            "window": self.window,
            "ema_beta": self.ema_beta,

            "status_counts": self.status_counts,
            "critical_rate": critical_rate,
            "warn_rate": warn_rate,
            "clip_rate": clip_rate,
            "amp_step_skipped_rate": skipped_rate,
            "nonfinite_rate": nonfinite_rate,

            "loss": self.metric_stats(
                "loss"
            ),

            "loss_cv": self.metric_stats(
                "loss_cv"
            ),

            "grad_norm_before_clip": self.metric_stats(
                "grad_norm_before_clip"
            ),

            "update_ratio": self.metric_stats(
                "update_ratio"
            ),

            "pred_std_ratio": self.metric_stats(
                "pred_std_ratio"
            ),

            "pred_target_mean_gap_std": self.metric_stats(
                "pred_target_mean_gap_std"
            ),

            "csv_path": str(
                self.csv_path
            ),
        }

        with self.summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )
