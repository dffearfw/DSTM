#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


STRATEGIES = [
    "frozen",
    "fusion_ft",
    "point_ft",
    "spatial_ft",
    "partial",
    "none",
]

STRATEGY_LABELS = {
    "frozen": "Frozen",
    "fusion_ft": "Fusion-Layer FT",
    "point_ft": "Point-Branch FT",
    "spatial_ft": "Spatial-Branch FT",
    "partial": "Top-Layer FT",
    "none": "Full FT",
}


def load_json(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_row(
    run_dir: Path,
    stage: int,
    strategy: str,
    dataset: str,
    metrics: dict,
    model_path=None,
):
    high = metrics.get("high_swe") or {}

    return {
        "run_dir": str(run_dir),
        "stage": stage,
        "model": f"M{stage}",
        "strategy": strategy,
        "method": STRATEGY_LABELS.get(strategy, strategy),
        "dataset": dataset,
        "n": metrics.get(
            "n",
            metrics.get("num_samples", metrics.get("n_samples")),
        ),
        "nse": metrics.get("nse", metrics.get("r2")),
        "r": metrics.get("r"),
        "rmse_mm": metrics.get("rmse_mm", metrics.get("rmse")),
        "mae_mm": metrics.get("mae_mm", metrics.get("mae")),
        "bias_mm": metrics.get("bias_mm", metrics.get("bias")),
        "alpha": metrics.get("alpha"),
        "beta": metrics.get("beta"),
        "slope": metrics.get("slope"),
        "intercept_mm": metrics.get("intercept_mm", metrics.get("intercept")),
        "pred_mean_mm": metrics.get("prediction_mean_mm", metrics.get("pred_mean")),
        "target_mean_mm": metrics.get("target_mean_mm", metrics.get("target_mean")),
        "high_swe_n": high.get("n"),
        "high_swe_rmse_mm": high.get("rmse"),
        "high_swe_mae_mm": high.get("mae"),
        "high_swe_bias_mm": high.get("bias"),
        "high_swe_pred_mean_mm": high.get("pred_mean"),
        "high_swe_obs_mean_mm": high.get(
            "target_mean",
            high.get("obs_mean"),
        ),
        "model_path": model_path,
    }


def extract_evaluation(path: Path):
    payload = load_json(path)

    if payload is None:
        return None

    if "fixed_test_eval_results" in payload:
        return payload["fixed_test_eval_results"]

    if "main_test_metrics" in payload:
        return payload["main_test_metrics"]

    return payload


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("/root/autodl-tmp/experiments"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/experiments/"
            "progressive_finetune_summary"
        ),
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="同一阶段只保留时间最新的一次运行",
    )

    args = parser.parse_args()

    run_dirs = sorted(
        path
        for path in args.experiments_dir.glob(
            "progressive_M*_finetune_*"
        )
        if path.is_dir()
    )

    if not run_dirs:
        raise SystemExit("没有找到progressive_M*_finetune_*结果目录")

    if args.latest_only:
        latest = {}

        for run_dir in run_dirs:
            match = re.search(
                r"progressive_M(\d+)_finetune_",
                run_dir.name,
            )
            if match:
                latest[int(match.group(1))] = run_dir

        run_dirs = [
            latest[key]
            for key in sorted(latest)
        ]

    rows = []

    for run_dir in run_dirs:
        match = re.search(
            r"progressive_M(\d+)_finetune_",
            run_dir.name,
        )

        if not match:
            continue

        stage = int(match.group(1))

        manifest = load_json(
            run_dir / "stage_run_manifest.json"
        ) or {}

        pretrained_model = manifest.get(
            "pretrained_model"
        )

        frozen_internal = extract_evaluation(
            run_dir
            / "frozen_internal"
            / "fine_tune_evaluation_results.json"
        )

        if frozen_internal:
            rows.append(
                metric_row(
                    run_dir,
                    stage,
                    "frozen",
                    "internal_1000",
                    frozen_internal,
                    pretrained_model,
                )
            )

        frozen_external = extract_evaluation(
            run_dir
            / "frozen_external"
            / "fine_tune_evaluation_results.json"
        )

        if frozen_external:
            rows.append(
                metric_row(
                    run_dir,
                    stage,
                    "frozen",
                    "external_987",
                    frozen_external,
                    pretrained_model,
                )
            )

        for strategy in STRATEGIES:
            if strategy == "frozen":
                continue

            strategy_dir = run_dir / strategy

            if not strategy_dir.exists():
                continue

            cv_candidates = sorted(
                strategy_dir.rglob("cv_station_level_aggregated_results.json")
            )
            cv_payload = load_json(cv_candidates[-1]) if cv_candidates else None
            model_reference = None

            if cv_payload:
                cv_metrics = cv_payload.get("aggregated_metrics")
                model_paths = cv_payload.get("fold_model_paths", {})
                if model_paths:
                    model_reference = "cv10_fold_models:" + str(
                        Path(cv_candidates[-1]).parent
                    )

                if cv_metrics:
                    rows.append(
                        metric_row(
                            run_dir,
                            stage,
                            strategy,
                            "internal_balanced_station_cv10_oof",
                            cv_metrics,
                            model_reference,
                        )
                    )

            ensemble_path = (
                strategy_dir
                / "external_987_once"
                / "external_cv10_ensemble_results.json"
            )
            ensemble_payload = load_json(ensemble_path)
            if ensemble_payload:
                external_metrics = (
                    ensemble_payload.get("metrics", {})
                    .get("CV10 ensemble")
                )
                if external_metrics:
                    rows.append(
                        metric_row(
                            run_dir,
                            stage,
                            strategy,
                            "external_987_once_cv10_ensemble",
                            external_metrics,
                            model_reference,
                        )
                    )

    if not rows:
        raise SystemExit("没有找到可汇总的JSON结果")

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "stage",
            "strategy",
            "dataset",
            "run_dir",
        ]
    ).reset_index(drop=True)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = (
        args.output_dir
        / "progressive_finetune_all_results.csv"
    )

    result.to_csv(
        full_path,
        index=False,
        encoding="utf-8-sig",
    )

    latest_rows = (
        result.sort_values("run_dir")
        .drop_duplicates(
            subset=[
                "stage",
                "strategy",
                "dataset",
            ],
            keep="last",
        )
        .sort_values(
            ["stage", "strategy", "dataset"]
        )
        .reset_index(drop=True)
    )

    latest_path = (
        args.output_dir
        / "progressive_finetune_latest_results.csv"
    )

    latest_rows.to_csv(
        latest_path,
        index=False,
        encoding="utf-8-sig",
    )

    for metric in [
        "nse",
        "r",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
    ]:
        pivot = latest_rows.pivot_table(
            index=["stage", "model"],
            columns=["dataset", "strategy"],
            values=metric,
            aggfunc="first",
        )

        pivot.to_csv(
            args.output_dir
            / f"comparison_{metric}.csv",
            encoding="utf-8-sig",
        )

    print("=" * 78)
    print("✅ 渐进式微调结果汇总完成")
    print(f"结果行数: {len(result):,}")
    print(f"完整结果: {full_path}")
    print(f"各组合最新结果: {latest_path}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 78)

    display_columns = [
        "model",
        "strategy",
        "dataset",
        "n",
        "nse",
        "r",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
    ]

    print(
        latest_rows[display_columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
