#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_NAME = "fine_tune_evaluation_results.json"
PREDICTION_NAME = "test_set_features_complete_with_pretrained.csv"
SUMMARY_NAME = "fine_tune_summary.txt"


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def promote_unique_result(
    output_dir: Path,
    filename: str,
    required: bool = True,
) -> Path | None:
    canonical = output_dir / filename

    if canonical.is_file():
        return canonical

    matches = [
        path
        for path in output_dir.rglob(filename)
        if path.is_file() and path != canonical
    ]

    if len(matches) == 1:
        shutil.copy2(matches[0], canonical)
        print(f"✅ 已归位结果: {matches[0]} -> {canonical}")
        return canonical

    if len(matches) > 1:
        raise RuntimeError(
            f"{output_dir} 内发现多个 {filename}，拒绝猜测:\n"
            + "\n".join(str(path) for path in matches)
        )

    if required:
        raise FileNotFoundError(
            f"评估结束后没有找到文件: {filename}\n"
            f"搜索目录: {output_dir}"
        )

    return None


def promote_plot_files(output_dir: Path) -> list[Path]:
    promoted = []

    for source in sorted(output_dir.rglob("*.png")):
        if source.parent == output_dir:
            promoted.append(source)
            continue

        target = output_dir / source.name

        if target.exists():
            try:
                if filecmp.cmp(source, target, shallow=False):
                    promoted.append(target)
                    continue
            except OSError:
                pass

            target = output_dir / f"{source.parent.name}__{source.name}"

        shutil.copy2(source, target)
        promoted.append(target)

    return sorted(set(promoted))


def compute_metrics(prediction_path: Path) -> dict:
    frame = pd.read_csv(prediction_path)

    required = ["站点SWE_raw", "微调模型预测_raw"]
    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise RuntimeError(
            f"{prediction_path} 缺少预测列: {missing}\n"
            f"现有列: {list(frame.columns)}"
        )

    target = pd.to_numeric(
        frame["站点SWE_raw"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    prediction = pd.to_numeric(
        frame["微调模型预测_raw"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]

    if target.size == 0:
        raise RuntimeError("没有有效预测样本")

    error = prediction - target

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))

    target_std = float(np.std(target))
    prediction_std = float(np.std(prediction))

    if (
        target.size > 1
        and target_std > 1e-12
        and prediction_std > 1e-12
    ):
        r = float(np.corrcoef(target, prediction)[0, 1])
    else:
        r = float("nan")

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(
        np.sum((target - np.mean(target)) ** 2)
    )
    nse = (
        1.0 - ss_res / ss_tot
        if ss_tot > 1e-12
        else float("nan")
    )

    centered_target = target - np.mean(target)
    slope_denominator = float(
        np.sum(centered_target ** 2)
    )

    if slope_denominator > 1e-12:
        slope = float(
            np.sum(
                centered_target
                * (prediction - np.mean(prediction))
            )
            / slope_denominator
        )
        intercept = float(
            np.mean(prediction)
            - slope * np.mean(target)
        )
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

    def clean(value):
        value = float(value)
        return value if np.isfinite(value) else None

    return {
        "n_samples": int(target.size),
        "r": clean(r),
        "nse": clean(nse),
        "rmse_mm": rmse,
        "mae_mm": mae,
        "bias_mm": bias,
        "rmse_obs_ge50_mm": clean(rmse_ge50),
        "bias_obs_ge80_mm": clean(bias_ge80),
        "n_obs_ge50": int(np.sum(ge50)),
        "n_obs_ge80": int(np.sum(ge80)),
        "slope": clean(slope),
        "intercept_mm": clean(intercept),
        "std_ratio": (
            clean(prediction_std / target_std)
            if target_std > 1e-12
            else None
        ),
        "pred_std_mm": prediction_std,
        "target_std_mm": target_std,
        "prediction_min_mm": float(np.min(prediction)),
        "prediction_max_mm": float(np.max(prediction)),
    }


def update_partial_summary(
    split_root: Path,
    stage: int,
    strategy: str,
    split_name: str,
) -> None:
    rows = []

    for fold_dir in sorted(split_root.glob("fold_*")):
        prediction_path = fold_dir / PREDICTION_NAME

        if not prediction_path.is_file():
            continue

        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue

        metrics = compute_metrics(prediction_path)
        metrics.update({
            "fold": fold,
            "stage": stage,
            "strategy": strategy,
            "split": split_name,
        })
        rows.append(metrics)

    if not rows:
        return

    rows = sorted(rows, key=lambda item: item["fold"])
    frame = pd.DataFrame(rows)

    frame.to_csv(
        split_root / "fold_metrics_partial.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_names = [
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

    summary = {
        "stage": stage,
        "strategy": strategy,
        "split": split_name,
        "completed_folds": [
            int(row["fold"]) for row in rows
        ],
        "n_completed_folds": len(rows),
        "metrics": {},
        "note": (
            "This is an interim summary of folds completed so far. "
            "Fixed-test results are evaluation-only and do not select "
            "epochs, folds, or hyperparameters."
        ),
    }

    for metric_name in metric_names:
        values = pd.to_numeric(
            frame[metric_name],
            errors="coerce",
        ).dropna().to_numpy(dtype=np.float64)

        summary["metrics"][metric_name] = {
            "n": int(values.size),
            "mean": (
                float(np.mean(values))
                if values.size
                else None
            ),
            "std": (
                float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0 if values.size == 1 else None
            ),
            "min": (
                float(np.min(values))
                if values.size
                else None
            ),
            "max": (
                float(np.max(values))
                if values.size
                else None
            ),
        }

    (split_root / "fold_summary_partial.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    (split_root / "latest_completed_fold.txt").write_text(
        str(max(summary["completed_folds"])) + "\n",
        encoding="utf-8",
    )


def run_single_evaluation(
    *,
    model_path: Path,
    station_file: Path,
    output_dir: Path,
    evaluation_name: str,
    expected_samples: int,
    split_name: str,
    stage: int,
    strategy: str,
    main_path: Path,
    normalization_path: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_result = output_dir / RESULT_NAME
    canonical_prediction = output_dir / PREDICTION_NAME

    if canonical_result.is_file() and canonical_prediction.is_file():
        print()
        print("✅ 本折评估结果已经完整，跳过重复计算")
        print(f"   {canonical_result}")
        print(f"   {canonical_prediction}")
    else:
        command = [
            sys.executable,
            str(main_path),
            "--mode", "evaluate",
            "--pretrained_model", str(model_path),
            "--model_path", str(model_path),
            "--station_data_path", str(station_file),
            "--save_dir", str(output_dir),
            "--model_type", "full",
            "--batch_size", str(batch_size),
            "--lr", "1e-4",
            "--d_model", "256",
            "--seed", str(seed),
            "--num_workers", str(num_workers),
            "--cv_mode", "station_cv",
            "--normalization_config_path",
            str(normalization_path),
            "--normalization_mode", "load",
            "--fixed_label_min_mm", "0",
            "--fixed_label_max_mm", "400",
            "--coord_jitter_std", "0.02",
            "--microwave_noise_std", "0.01",
            "--coord_mask_prob", "0.2",
            "--val_every", "1",
            "--use_amp",
        ]

        print()
        print("=" * 88)
        print(f"🚀 即时评估: {evaluation_name}")
        print(f"   模型: {model_path}")
        print(f"   数据: {station_file}")
        print(f"   输出: {output_dir}")
        print("=" * 88)

        subprocess.run(command, check=True)

    result_path = promote_unique_result(
        output_dir,
        RESULT_NAME,
        required=True,
    )
    prediction_path = promote_unique_result(
        output_dir,
        PREDICTION_NAME,
        required=True,
    )
    promote_unique_result(
        output_dir,
        SUMMARY_NAME,
        required=False,
    )

    plot_paths = promote_plot_files(output_dir)

    metrics = compute_metrics(prediction_path)

    if metrics["n_samples"] != expected_samples:
        raise RuntimeError(
            f"{split_name}样本数错误: "
            f"actual={metrics['n_samples']}, "
            f"expected={expected_samples}"
        )

    metrics.update({
        "stage": stage,
        "strategy": strategy,
        "split": split_name,
        "model_path": str(model_path),
        "result_path": str(result_path),
        "prediction_path": str(prediction_path),
        "plot_paths": [str(path) for path in plot_paths],
    })

    (output_dir / "immediate_metrics.json").write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    (output_dir / "model_path.txt").write_text(
        str(model_path) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"✅ {evaluation_name} 完成")
    print(
        f"   R={metrics['r']}, "
        f"NSE={metrics['nse']}, "
        f"RMSE={metrics['rmse_mm']:.2f} mm, "
        f"MAE={metrics['mae_mm']:.2f} mm, "
        f"Bias={metrics['bias_mm']:.2f} mm"
    )
    print(
        f"   obs≥50 RMSE="
        f"{metrics['rmse_obs_ge50_mm']} mm"
    )
    print(
        f"   obs≥80 Bias="
        f"{metrics['bias_obs_ge80_mm']} mm"
    )
    print(f"   图件数量: {len(plot_paths)}")
    print(f"   结果目录: {output_dir}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
    )
    parser.add_argument(
        "--fold",
        required=True,
        type=int,
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    stage = int(env_required("SWE_FOLD_EVAL_STAGE"))
    strategy = env_required("SWE_FOLD_EVAL_STRATEGY")

    strategy_dir = Path(
        env_required("SWE_FOLD_EVAL_STRATEGY_DIR")
    ).resolve()
    run_root = Path(
        env_required("SWE_FOLD_EVAL_RUN_ROOT")
    ).resolve()
    main_path = Path(
        env_required("SWE_FOLD_EVAL_MAIN")
    ).resolve()

    internal_data = Path(
        env_required("SWE_FOLD_EVAL_INTERNAL_DATA")
    ).resolve()
    external_data = Path(
        env_required("SWE_FOLD_EVAL_EXTERNAL_DATA")
    ).resolve()
    normalization_path = Path(
        env_required("SWE_FOLD_EVAL_NORMALIZATION")
    ).resolve()

    seed = int(env_required("SWE_FOLD_EVAL_SEED"))
    batch_size = int(
        env_required("SWE_FOLD_EVAL_BATCH_SIZE")
    )
    num_workers = int(
        env_required("SWE_FOLD_EVAL_NUM_WORKERS")
    )
    run_external = (
        env_required("SWE_FOLD_EVAL_RUN_EXTERNAL")
        != "0"
    )

    fold = int(args.fold)
    fold_pad = f"{fold:02d}"

    internal_root = (
        strategy_dir / "internal_test_10fold"
    )
    external_root = (
        strategy_dir / "external_test_10fold"
    )

    internal_dir = internal_root / f"fold_{fold_pad}"
    external_dir = external_root / f"fold_{fold_pad}"

    print()
    print("█" * 88)
    print(
        f"🌟 M{stage} {strategy} Fold {fold} "
        "训练完成，立即执行双测试"
    )
    print("   测试结果仅用于报告，不参与选模")
    print("█" * 88)

    run_single_evaluation(
        model_path=model_path,
        station_file=internal_data,
        output_dir=internal_dir,
        evaluation_name=(
            f"M{stage} {strategy} Fold {fold} "
            "→ 内部1000条固定测试"
        ),
        expected_samples=1000,
        split_name="internal_1000",
        stage=stage,
        strategy=strategy,
        main_path=main_path,
        normalization_path=normalization_path,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    update_partial_summary(
        internal_root,
        stage,
        strategy,
        "internal_1000",
    )

    if run_external:
        isolated_dir = (
            run_root
            / "runtime_inputs"
            / "external_evaluation"
        )
        isolated_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        isolated_file = (
            isolated_dir
            / "external_evaluation_input.csv"
        )

        if not isolated_file.is_file():
            shutil.copy2(
                external_data,
                isolated_file,
            )

        if not filecmp.cmp(
            external_data,
            isolated_file,
            shallow=False,
        ):
            raise RuntimeError(
                "外部评估隔离副本与固定原文件不一致"
            )

        run_single_evaluation(
            model_path=model_path,
            station_file=isolated_file,
            output_dir=external_dir,
            evaluation_name=(
                f"M{stage} {strategy} Fold {fold} "
                "→ 外部987条固定测试"
            ),
            expected_samples=987,
            split_name="external_987",
            stage=stage,
            strategy=strategy,
            main_path=main_path,
            normalization_path=normalization_path,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        update_partial_summary(
            external_root,
            stage,
            strategy,
            "external_987",
        )

    print()
    print("█" * 88)
    print(
        f"✅ Fold {fold} 内部/外部即时测试完成，"
        f"现在可以开始Fold {fold + 1}"
    )
    print("█" * 88)


if __name__ == "__main__":
    main()
