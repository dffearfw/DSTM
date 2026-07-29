#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One final external result from the predeclared 10-fold model ensemble.

The external dataset is loaded once. Ten fold checkpoints are applied
sequentially, predictions are averaged, and only the ensemble metric is the
official external result. No fold is selected by external performance.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a 10-fold checkpoint prediction ensemble once on external data"
    )
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--station-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--normalization-config",
        type=Path,
        default=Path(
            "/root/autodl-tmp/shared_cache/progressive_pretrain_normalization.json"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/shared_cache"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--expected-external-count",
        type=int,
        default=987,
        help="Expected number of split=test external rows; mismatch aborts evaluation",
    )
    return parser.parse_args()


def discover_fold_models(checkpoint_dir: Path) -> list[Path]:
    pattern = re.compile(r"^cv_fold_(\d+)_best_model\.pth$")
    found: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("cv_fold_*_best_model.pth"):
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(1)), path.resolve()))
    found.sort(key=lambda item: item[0])

    folds = [fold for fold, _ in found]
    if folds != list(range(1, 11)):
        raise RuntimeError(
            f"必须恰好找到Fold 1-10 checkpoint，实际fold={folds}"
        )
    return [path for _, path in found]


def predict_one_model(
    dataset,
    model,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    label_min_mm: float,
    label_max_mm: float,
) -> np.ndarray:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    predictions: list[np.ndarray] = []
    expected = list(indices)
    cursor = 0

    with torch.inference_mode():
        for batch in loader:
            conv = batch[0].to(device, non_blocking=True)
            point = batch[1].to(device, non_blocking=True)
            pred_norm = model(conv, point).detach().float().cpu().numpy().reshape(-1)
            pred_mm = pred_norm * (label_max_mm - label_min_mm) + label_min_mm

            n_batch = len(pred_mm)
            if len(batch) >= 6:
                returned = batch[5]
                if torch.is_tensor(returned):
                    returned = returned.detach().cpu().numpy().reshape(-1)
                returned = [int(x) for x in returned]
            else:
                returned = expected[cursor:cursor + n_batch]

            expected_batch = expected[cursor:cursor + n_batch]
            if returned != expected_batch:
                raise RuntimeError(
                    "Dataset索引重定向，拒绝外部评估: "
                    f"expected={expected_batch[:10]}, returned={returned[:10]}"
                )
            predictions.append(np.asarray(pred_mm, dtype=np.float64))
            cursor += n_batch

    result = np.concatenate(predictions)
    if len(result) != len(indices):
        raise RuntimeError(
            f"外部预测数量异常: {len(result)}/{len(indices)}"
        )
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)

    args.root = args.root.expanduser().resolve()
    args.station_csv = args.station_csv.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.normalization_config = args.normalization_config.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [
        args.station_csv,
        args.normalization_config,
        args.root / "data_station_online_swe.py",
        args.root / "models_swe.py",
        args.root / "evaluate_frozen_station_cv10.py",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    fold_models = discover_fold_models(args.checkpoint_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    sys.path.insert(0, str(args.root))
    from data_station_online_swe import StationSWEDataset
    from evaluate_frozen_station_cv10 import (
        compute_metrics,
        get_era5_mm,
        load_model,
        normalize_station_id,
        plot_scatter,
        sha256_file,
    )

    print("=" * 100)
    print("外部987条：10-fold模型预声明集成，一次正式结果")
    print("=" * 100)
    print(f"数据: {args.station_csv}")
    print(f"模型目录: {args.checkpoint_dir}")
    print(f"输出: {args.output_dir}")
    print("不按外部性能选择fold；只报告10模型均值预测。")
    print("=" * 100)

    source = pd.read_csv(args.station_csv)
    if "split" not in source.columns:
        raise RuntimeError(
            "外部评估清单缺少split列，无法证明只评估预声明外部样本"
        )

    split_values = source["split"].astype(str).str.strip().str.lower()
    external_source = source.loc[split_values == "test"].copy()
    expected_external_count = int(args.expected_external_count)
    if len(external_source) != expected_external_count:
        raise RuntimeError(
            "外部样本数不符合预声明协议: "
            f"split=test实际={len(external_source)}, "
            f"expected={expected_external_count}, "
            f"完整清单={len(source)}"
        )

    # 只把真正的外部行交给Dataset。不能在完整7923行Dataset上生成预测后
    # 再过滤，因为内部CV池中的样本曾被9/10个fold模型用于训练。
    external_only_csv = args.output_dir / "external_only_input_audit.csv"
    external_source = external_source.reset_index(drop=True)
    external_source.to_csv(
        external_only_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"✅ 外部清单硬筛选: 完整={len(source)}, "
        f"split=test={len(external_source)}"
    )

    dataset = StationSWEDataset(
        station_csv=external_only_csv,
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
    if len(dataset) != len(external_source):
        raise RuntimeError(
            "外部Dataset/CSV长度不一致: "
            f"{len(dataset)}/{len(external_source)}"
        )

    indices = list(range(len(dataset)))
    if len(indices) != expected_external_count:
        raise RuntimeError(
            f"外部推理索引数异常: {len(indices)}/{expected_external_count}"
        )
    label_min_mm = float(getattr(dataset, "swe_min", 0.0))
    label_max_mm = float(getattr(dataset, "swe_max", 400.0))

    fold_predictions: list[np.ndarray] = []
    checkpoint_records: list[dict[str, Any]] = []

    for fold, checkpoint in enumerate(fold_models, start=1):
        print(f"\nFold {fold:02d}/10: {checkpoint.name}")
        model, config = load_model(
            checkpoint_path=checkpoint,
            c_spatial=int(dataset.C_conv),
            c_point=int(dataset.C_point),
            device=device,
        )
        pred = predict_one_model(
            dataset=dataset,
            model=model,
            indices=indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            label_min_mm=label_min_mm,
            label_max_mm=label_max_mm,
        )
        fold_predictions.append(pred)
        checkpoint_records.append(
            {
                "fold": fold,
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "checkpoint_config": config,
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_matrix = np.stack(fold_predictions, axis=0)
    ensemble_prediction = np.mean(prediction_matrix, axis=0)
    ensemble_std = np.std(prediction_matrix, axis=0)

    rows = []
    targets = []
    era5_values = []
    for idx, meta in enumerate(dataset.meta_index):
        label_date = pd.to_datetime(
            meta.get("label_date", meta.get("feature_date", meta.get("date")))
        ).strftime("%Y-%m-%d")
        feature_date = pd.to_datetime(
            meta.get("feature_date", meta.get("label_date", meta.get("date")))
        ).strftime("%Y-%m-%d")
        target = float(meta["swe"])
        era5 = float(get_era5_mm(dataset, meta))
        targets.append(target)
        era5_values.append(era5)
        rows.append(
            {
                "dataset_index": idx,
                "station_id": normalize_station_id(meta.get("station_id", "unknown")),
                "label_date": label_date,
                "feature_date": feature_date,
                "day_gap": int(meta.get("day_gap", 0)),
                "row": int(meta["row"]),
                "col": int(meta["col"]),
                "target_mm": target,
                "ensemble_prediction_mm": float(ensemble_prediction[idx]),
                "fold_prediction_std_mm": float(ensemble_std[idx]),
                "era5_land_mm": era5,
            }
        )

    frame = pd.DataFrame(rows)
    if (frame["feature_date"] != frame["label_date"]).any():
        raise RuntimeError("外部结果存在feature_date != label_date")
    if (frame["day_gap"] != 0).any():
        raise RuntimeError("外部结果存在day_gap != 0")

    metrics = {
        "CV10 ensemble": compute_metrics(targets, ensemble_prediction),
        "ERA5-Land": compute_metrics(targets, era5_values),
    }

    frame.to_csv(
        args.output_dir / "external_cv10_ensemble_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = {
        "created_at": datetime.now().isoformat(),
        "protocol": {
            "official_external_result": "mean_prediction_of_10_fold_models",
            "n_external_evaluations_reported": 1,
            "fold_selection_using_external": False,
            "external_sample_count": int(len(frame)),
        },
        "metrics": metrics,
        "checkpoint_records": checkpoint_records,
        "files": {
            "station_csv": str(args.station_csv),
            "station_csv_sha256": sha256_file(args.station_csv),
            "external_only_csv": str(external_only_csv),
            "external_only_csv_sha256": sha256_file(external_only_csv),
            "normalization_config": str(args.normalization_config),
            "normalization_config_sha256": sha256_file(args.normalization_config),
        },
    }
    (args.output_dir / "external_cv10_ensemble_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_scatter(
        np.asarray(targets),
        ensemble_prediction,
        metrics["CV10 ensemble"],
        "CV10 Ensemble",
        args.output_dir / "external_cv10_ensemble_scatter.png",
    )

    official = metrics["CV10 ensemble"]
    print("\n" + "=" * 100)
    print("✅ 外部一次正式集成结果")
    print(
        f"N={official['n_samples']}, R={official['r']:.4f}, "
        f"NSE={official['nse']:.4f}, RMSE={official['rmse_mm']:.2f}, "
        f"MAE={official['mae_mm']:.2f}, Bias={official['bias_mm']:.2f}"
    )
    print(f"结果目录: {args.output_dir}")
    print("=" * 100)

    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
