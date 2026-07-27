#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--station_data_path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/shared_cache/"
            "progressive_finetune/internal_progressive_station.csv"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/diagnostics/"
            "fold01_true_split_and_duplicates"
        ),
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--swe_min", type=float, default=0.0)
    parser.add_argument("--swe_max", type=float, default=400.0)
    parser.add_argument(
        "--normalization_config_path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/shared_cache/"
            "progressive_pretrain_normalization.json"
        ),
    )
    parser.add_argument(
        "--max_pair_count",
        type=int,
        default=20000,
    )

    return parser.parse_args()


def disable_augmentation(dataset):
    visited = set()

    def visit(obj):
        if obj is None or id(obj) in visited:
            return

        visited.add(id(obj))

        if hasattr(obj, "set_augmentation_mode"):
            obj.set_augmentation_mode(False)
        elif hasattr(obj, "current_augment"):
            obj.current_augment = False

        if isinstance(obj, Subset):
            visit(obj.dataset)
        elif hasattr(obj, "dataset"):
            visit(obj.dataset)

    visit(dataset)


def unwrap_subset(dataset):
    if isinstance(dataset, Subset):
        base, parent_mapping = unwrap_subset(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return base, parent_mapping[indices]

    return dataset, np.arange(len(dataset), dtype=np.int64)


def unpack_batch(batch):
    if len(batch) >= 6:
        conv, point, target, is_zero, fused, sample_idx = batch[:6]
    elif len(batch) == 5:
        conv, point, target, is_zero, fused = batch
        sample_idx = None
    elif len(batch) == 4:
        conv, point, target, is_zero = batch
        fused = None
        sample_idx = None
    elif len(batch) == 3:
        conv, point, target = batch
        is_zero = (target > 0).float()
        fused = None
        sample_idx = None
    else:
        raise RuntimeError(
            f"不支持的 batch 长度: {len(batch)}"
        )

    return (
        conv,
        point,
        target,
        is_zero,
        fused,
        sample_idx,
    )


def tensor_to_numpy(tensor):
    return (
        torch.nan_to_num(
            tensor.detach().cpu(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        .contiguous()
        .numpy()
        .astype(np.float32, copy=False)
    )


def array_hash(array, decimals=None):
    arr = np.asarray(array, dtype=np.float32)

    if decimals is not None:
        arr = np.round(
            arr,
            decimals=decimals,
        ).astype(np.float32)

    arr = np.ascontiguousarray(arr)

    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())
    hasher.update(arr.tobytes())

    return hasher.hexdigest()


def full_input_hash(conv, point, decimals=None):
    hasher = hashlib.sha256()
    hasher.update(
        array_hash(conv, decimals).encode()
    )
    hasher.update(
        array_hash(point, decimals).encode()
    )
    return hasher.hexdigest()


def get_meta(dataset, index):
    if not hasattr(dataset, "meta_index"):
        return {}

    if index < 0 or index >= len(dataset.meta_index):
        return {}

    meta = dataset.meta_index[index]

    if isinstance(meta, dict):
        return dict(meta)

    return {}


def get_date_text(meta):
    value = (
        meta.get("feature_date")
        or meta.get("label_date")
        or meta.get("date")
    )

    if value is None:
        return ""

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def calculate_metrics(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]

    if len(obs) == 0:
        return {
            "N": 0,
            "R": np.nan,
            "NSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "alpha": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "High80_N": 0,
            "High80_Bias": np.nan,
            "High80_RMSE": np.nan,
        }

    error = pred - obs

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    bias = np.mean(error)

    if (
        len(obs) >= 2
        and np.std(obs) > 0
        and np.std(pred) > 0
    ):
        r = np.corrcoef(obs, pred)[0, 1]
    else:
        r = np.nan

    denominator = np.sum(
        (obs - np.mean(obs)) ** 2
    )
    nse = (
        1.0 - np.sum(error ** 2) / denominator
        if denominator > 0
        else np.nan
    )

    alpha = (
        np.std(pred) / np.std(obs)
        if np.std(obs) > 0
        else np.nan
    )

    if len(obs) >= 2 and np.std(obs) > 0:
        slope, intercept = np.polyfit(
            obs,
            pred,
            1,
        )
    else:
        slope = np.nan
        intercept = np.nan

    high_mask = obs >= 80.0

    if np.any(high_mask):
        high_error = (
            pred[high_mask] - obs[high_mask]
        )
        high_n = int(high_mask.sum())
        high_bias = np.mean(high_error)
        high_rmse = np.sqrt(
            np.mean(high_error ** 2)
        )
    else:
        high_n = 0
        high_bias = np.nan
        high_rmse = np.nan

    return {
        "N": int(len(obs)),
        "R": float(r),
        "NSE": float(nse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": float(bias),
        "alpha": float(alpha),
        "slope": float(slope),
        "intercept": float(intercept),
        "High80_N": high_n,
        "High80_Bias": float(high_bias),
        "High80_RMSE": float(high_rmse),
    }


def build_trainer(args):
    sys.path.insert(0, "/root/autodl-tmp")

    from main_tune import SWETrainer

    config = {
        "model_type": "full",
        "epochs": 1,
        "batch_size": args.batch_size,
        "learning_rate": 1e-4,
        "d_model": 256,

        "save_dir": str(args.output_dir.parent),
        "experiment_name": (
            args.output_dir.name + "_loader"
        ),

        "split_method": "temporal",
        "train_year": 2015,
        "val_year": 2016,
        "pretrain_years": [2015, 2016, 2017],

        "fine_tune": True,
        "fine_tune_epochs": 1,
        "fine_tune_lr": 1e-4,

        "freeze_backbone": True,
        "freeze_strategy": "fusion_ft",

        "station_data_path": str(
            args.station_data_path
        ),
        "cv_mode": "station_cv",

        "mixed_mode": False,
        "station_ratio": 1.0,

        "pretrained_model": str(
            args.checkpoint
        ),

        "coord_jitter_std": 0.0,
        "microwave_noise_std": 0.0,
        "coord_mask_prob": 0.0,

        "num_workers": args.num_workers,
        "persistent_workers": False,
        "shared_cache_mode": True,

        "seed": args.seed,

        "normalization_config_path": str(
            args.normalization_config_path
        ),
        "normalization_mode": "load",

        "fixed_label_min_mm": args.swe_min,
        "fixed_label_max_mm": args.swe_max,
    }

    trainer = SWETrainer(config)

    loaded = trainer.load_data(
        fine_tune_mode=True,
        mixed_mode=False,
        station_ratio=1.0,
    )

    if loaded is False or loaded is None:
        raise RuntimeError("load_data() 失败")

    built = trainer.build_model(
        load_pretrained=str(args.checkpoint),
        freeze_backbone=True,
        freeze_strategy="fusion_ft",
    )

    if built is False:
        raise RuntimeError("build_model() 失败")

    trainer.model.eval()

    for parameter in trainer.model.parameters():
        parameter.requires_grad_(False)

    return trainer


def get_station_dataset(trainer):
    if hasattr(trainer, "station_dataset"):
        return trainer.station_dataset

    dataset = trainer.train_loader.dataset

    while isinstance(dataset, Subset):
        dataset = dataset.dataset

    if hasattr(dataset, "station_dataset"):
        return dataset.station_dataset

    return dataset


def reconstruct_fold(trainer, station_dataset, args):
    if hasattr(
        trainer,
        "cv_pool_indices_override",
    ):
        pool = np.asarray(
            trainer.cv_pool_indices_override,
            dtype=np.int64,
        )
    else:
        pool = np.arange(
            len(station_dataset),
            dtype=np.int64,
        )

    pool = pool[
        (pool >= 0)
        & (pool < len(station_dataset))
    ]

    if len(pool) != 6936:
        raise RuntimeError(
            f"CV池不是6936条，而是{len(pool)}条，"
            "停止执行，防止使用错误样本池。"
        )

    if len(np.unique(pool)) != len(pool):
        raise RuntimeError("CV池存在重复索引")

    kfold = KFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    fold_splits = list(kfold.split(pool))

    train_pos, val_pos = fold_splits[
        args.fold - 1
    ]

    train_indices = pool[train_pos]
    val_indices = pool[val_pos]

    if len(train_indices) != 6242:
        raise RuntimeError(
            f"Fold Train应为6242，实际为"
            f"{len(train_indices)}"
        )

    if len(val_indices) != 694:
        raise RuntimeError(
            f"Fold Val应为694，实际为"
            f"{len(val_indices)}"
        )

    if np.intersect1d(
        train_indices,
        val_indices,
    ).size:
        raise RuntimeError("Train/Val索引重叠")

    print("\n真实Fold重建完成:")
    print(f"  CV池:  {len(pool)}")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Fold:  {args.fold}")
    print(f"  Seed:  {args.seed}")

    return train_indices, val_indices


def make_loader(dataset, indices, args):
    visible_dataset = (
        Subset(dataset, indices.tolist())
        if indices is not None
        else dataset
    )

    disable_augmentation(visible_dataset)

    kwargs = {
        "dataset": visible_dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if args.num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def collect_split(
    split_name,
    loader,
    trainer,
    args,
    record_offset,
    tensor_cache,
):
    disable_augmentation(loader.dataset)

    base_dataset, visible_mapping = unwrap_subset(
        loader.dataset
    )

    records = []
    position = 0

    with torch.inference_mode():
        for batch in loader:
            (
                conv,
                point,
                target,
                is_zero,
                fused,
                sample_idx,
            ) = unpack_batch(batch)

            conv = torch.nan_to_num(
                conv,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            point = torch.nan_to_num(
                point,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            target = torch.nan_to_num(
                target,
                nan=0.0,
            ).reshape(-1)
            is_zero = torch.nan_to_num(
                is_zero,
                nan=1.0,
            ).reshape(-1)

            output = trainer.model(
                conv.to(trainer.device),
                point.to(trainer.device),
            )

            if isinstance(output, (tuple, list)):
                output = output[0]

            pred_norm = (
                output.reshape(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            pred_norm = np.maximum(
                pred_norm,
                0.0,
            )

            pred_norm *= (
                is_zero.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            target_norm = (
                target.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            target_mm = (
                target_norm
                * (args.swe_max - args.swe_min)
                + args.swe_min
            )

            pred_mm = (
                pred_norm
                * (args.swe_max - args.swe_min)
                + args.swe_min
            )

            if fused is None:
                era5_mm = np.full(
                    len(target_mm),
                    np.nan,
                )
            else:
                fused_norm = (
                    fused.reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                # 当前数据契约中，第5项为归一化ERA5-Land。
                era5_mm = (
                    fused_norm
                    * (args.swe_max - args.swe_min)
                    + args.swe_min
                )

            conv_np = tensor_to_numpy(conv)
            point_np = tensor_to_numpy(point)

            expected_indices = visible_mapping[
                position:
                position + len(target_mm)
            ]

            # 不依赖batch第6项，直接使用shuffle=False的
            # Subset顺序，避免把source_flag误当sample_idx。
            batch_indices = expected_indices

            for i in range(len(target_mm)):
                dataset_index = int(
                    batch_indices[i]
                )

                meta = get_meta(
                    base_dataset,
                    dataset_index,
                )

                conv_i = conv_np[i]
                point_i = point_np[i]

                record_id = (
                    record_offset
                    + len(records)
                )

                tensor_cache[record_id] = (
                    conv_i.copy(),
                    point_i.copy(),
                )

                records.append(
                    {
                        "record_id": record_id,
                        "split": split_name,
                        "split_position": position + i,
                        "dataset_index": dataset_index,

                        "station_id": str(
                            meta.get(
                                "station_id",
                                "unknown",
                            )
                        ),
                        "date": get_date_text(meta),

                        "row": meta.get("row"),
                        "col": meta.get("col"),

                        "target_mm": float(
                            target_mm[i]
                        ),
                        "model_prediction_mm": float(
                            pred_mm[i]
                        ),
                        "era5_land_mm": float(
                            era5_mm[i]
                        ),

                        "conv_hash_exact": array_hash(
                            conv_i
                        ),
                        "point_hash_exact": array_hash(
                            point_i
                        ),
                        "full_hash_exact": full_input_hash(
                            conv_i,
                            point_i,
                        ),
                        "full_hash_round6": full_input_hash(
                            conv_i,
                            point_i,
                            decimals=6,
                        ),
                    }
                )

            position += len(target_mm)

    if position != len(loader.dataset):
        raise RuntimeError(
            f"{split_name}样本数错位: "
            f"预测={position}, dataset={len(loader.dataset)}"
        )

    print(
        f"  {split_name}: 收集{len(records)}条"
    )

    return records


def make_metric_rows(split_df):
    target = split_df["target_mm"].to_numpy()

    ranges = {
        "ALL": np.ones(
            len(split_df),
            dtype=bool,
        ),
        "LT20": target < 20,
        "20_50": (
            (target >= 20)
            & (target < 50)
        ),
        "50_80": (
            (target >= 50)
            & (target < 80)
        ),
        "GE80": target >= 80,
    }

    rows = []

    for range_name, mask in ranges.items():
        for source, column in {
            "MODEL": "model_prediction_mm",
            "ERA5_LAND": "era5_land_mm",
        }.items():
            metrics = calculate_metrics(
                target[mask],
                split_df.loc[
                    mask,
                    column,
                ].to_numpy(),
            )

            row = {
                "split": split_df[
                    "split"
                ].iloc[0],
                "target_range": range_name,
                "source": source,
            }
            row.update(metrics)
            rows.append(row)

    return rows


def build_duplicate_tables(
    records_df,
    tensor_cache,
    args,
):
    group_rows = []
    member_frames = []

    group_id = 0

    for full_hash, group in records_df.groupby(
        "full_hash_exact",
        sort=False,
    ):
        if len(group) < 2:
            continue

        group_id += 1

        target_range = (
            group["target_mm"].max()
            - group["target_mm"].min()
        )

        group_rows.append(
            {
                "duplicate_group_id": group_id,
                "full_hash_exact": full_hash,
                "count": len(group),

                "splits": "|".join(
                    sorted(
                        group["split"]
                        .astype(str)
                        .unique()
                    )
                ),

                "station_count": (
                    group["station_id"].nunique()
                ),
                "date_count": (
                    group["date"].nunique()
                ),

                "target_min_mm": (
                    group["target_mm"].min()
                ),
                "target_max_mm": (
                    group["target_mm"].max()
                ),
                "target_range_mm": target_range,

                "prediction_range_mm": (
                    group[
                        "model_prediction_mm"
                    ].max()
                    - group[
                        "model_prediction_mm"
                    ].min()
                ),

                "era5_range_mm": (
                    group["era5_land_mm"].max()
                    - group["era5_land_mm"].min()
                ),

                "conflict_gt_1mm": (
                    target_range > 1
                ),
                "conflict_gt_10mm": (
                    target_range > 10
                ),
                "conflict_gt_50mm": (
                    target_range > 50
                ),
            }
        )

        members = group.copy()
        members.insert(
            0,
            "duplicate_group_id",
            group_id,
        )
        member_frames.append(members)

    groups_df = pd.DataFrame(group_rows)

    members_df = (
        pd.concat(
            member_frames,
            ignore_index=True,
        )
        if member_frames
        else pd.DataFrame()
    )

    pair_rows = []

    for station_id, group in records_df.groupby(
        "station_id",
        sort=False,
    ):
        if len(group) < 2:
            continue

        rows = list(
            group.sort_values(
                ["date", "record_id"]
            ).to_dict("records")
        )

        for left, right in itertools.combinations(
            rows,
            2,
        ):
            target_diff = abs(
                left["target_mm"]
                - right["target_mm"]
            )

            if target_diff < 10:
                continue

            pred_diff = abs(
                left["model_prediction_mm"]
                - right["model_prediction_mm"]
            )

            era5_diff = abs(
                left["era5_land_mm"]
                - right["era5_land_mm"]
            )

            same_exact = (
                left["full_hash_exact"]
                == right["full_hash_exact"]
            )

            same_round6 = (
                left["full_hash_round6"]
                == right["full_hash_round6"]
            )

            suspicious = (
                same_exact
                or same_round6
                or pred_diff <= 1e-5
                or era5_diff <= 1e-5
            )

            if not suspicious:
                continue

            left_conv, left_point = tensor_cache[
                int(left["record_id"])
            ]
            right_conv, right_point = tensor_cache[
                int(right["record_id"])
            ]

            conv_diff = np.abs(
                left_conv.astype(np.float64)
                - right_conv.astype(np.float64)
            )

            point_diff = np.abs(
                left_point.astype(np.float64)
                - right_point.astype(np.float64)
            )

            channel_axes = tuple(
                range(1, conv_diff.ndim)
            )

            channel_max = np.max(
                conv_diff,
                axis=channel_axes,
            )

            pair_rows.append(
                {
                    "station_id": station_id,

                    "left_split": left["split"],
                    "right_split": right["split"],

                    "left_dataset_index": (
                        left["dataset_index"]
                    ),
                    "right_dataset_index": (
                        right["dataset_index"]
                    ),

                    "left_date": left["date"],
                    "right_date": right["date"],

                    "left_target_mm": (
                        left["target_mm"]
                    ),
                    "right_target_mm": (
                        right["target_mm"]
                    ),
                    "target_diff_mm": target_diff,

                    "left_prediction_mm": (
                        left[
                            "model_prediction_mm"
                        ]
                    ),
                    "right_prediction_mm": (
                        right[
                            "model_prediction_mm"
                        ]
                    ),
                    "prediction_diff_mm": pred_diff,

                    "left_era5_land_mm": (
                        left["era5_land_mm"]
                    ),
                    "right_era5_land_mm": (
                        right["era5_land_mm"]
                    ),
                    "era5_diff_mm": era5_diff,

                    "same_conv_hash_exact": (
                        left["conv_hash_exact"]
                        == right[
                            "conv_hash_exact"
                        ]
                    ),
                    "same_point_hash_exact": (
                        left[
                            "point_hash_exact"
                        ]
                        == right[
                            "point_hash_exact"
                        ]
                    ),
                    "same_full_hash_exact": (
                        same_exact
                    ),
                    "same_full_hash_round6": (
                        same_round6
                    ),

                    "conv_max_abs_diff": float(
                        np.max(conv_diff)
                    ),
                    "point_max_abs_diff": float(
                        np.max(point_diff)
                    ),

                    "conv_equal_fraction": float(
                        np.mean(conv_diff == 0)
                    ),
                    "point_equal_fraction": float(
                        np.mean(point_diff == 0)
                    ),

                    "conv_channel_max_abs_diff": (
                        json.dumps(
                            channel_max.tolist(),
                            ensure_ascii=False,
                        )
                    ),

                    "point_feature_abs_diff": (
                        json.dumps(
                            point_diff
                            .reshape(-1)
                            .tolist(),
                            ensure_ascii=False,
                        )
                    ),
                }
            )

            if len(pair_rows) >= args.max_pair_count:
                break

        if len(pair_rows) >= args.max_pair_count:
            break

    pairs_df = pd.DataFrame(pair_rows)

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            [
                "same_full_hash_exact",
                "target_diff_mm",
                "prediction_diff_mm",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

    return groups_df, members_df, pairs_df


def main():
    args = parse_args()

    args.checkpoint = (
        args.checkpoint.resolve()
    )
    args.station_data_path = (
        args.station_data_path.resolve()
    )
    args.output_dir = (
        args.output_dir.resolve()
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            args.checkpoint
        )

    if not args.station_data_path.exists():
        raise FileNotFoundError(
            args.station_data_path
        )

    print("=" * 88)
    print("真实Fold + 完整输入重复诊断")
    print("=" * 88)
    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {args.output_dir}")
    print("本脚本不训练、不修改模型、不修改损失。")

    trainer = build_trainer(args)

    station_dataset = get_station_dataset(
        trainer
    )

    disable_augmentation(
        station_dataset
    )

    train_indices, val_indices = (
        reconstruct_fold(
            trainer,
            station_dataset,
            args,
        )
    )

    train_loader = make_loader(
        station_dataset,
        train_indices,
        args,
    )

    val_loader = make_loader(
        station_dataset,
        val_indices,
        args,
    )

    test_dataset = (
        trainer.test_loader.dataset
    )

    disable_augmentation(
        test_dataset
    )

    test_loader = make_loader(
        test_dataset,
        None,
        args,
    )

    if len(test_loader.dataset) != 1000:
        raise RuntimeError(
            f"固定Test应为1000，实际为"
            f"{len(test_loader.dataset)}"
        )

    print("\n严格评估Loader:")
    print(
        f"  Train={len(train_loader.dataset)}"
    )
    print(
        f"  Val={len(val_loader.dataset)}"
    )
    print(
        f"  Test={len(test_loader.dataset)}"
    )
    print(
        "  shuffle=False, drop_last=False, "
        "augmentation=False"
    )

    all_records = []
    tensor_cache = {}

    offset = 0

    for split_name, loader in [
        ("TRAIN_TRUE_FOLD", train_loader),
        ("VAL_TRUE_FOLD", val_loader),
        ("TEST_FIXED", test_loader),
    ]:
        records = collect_split(
            split_name,
            loader,
            trainer,
            args,
            offset,
            tensor_cache,
        )

        all_records.extend(records)
        offset += len(records)

    records_df = pd.DataFrame(
        all_records
    )

    predictions_path = (
        args.output_dir
        / "fold01_true_split_predictions.csv"
    )

    records_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    metric_rows = []

    for _, split_df in records_df.groupby(
        "split",
        sort=False,
    ):
        metric_rows.extend(
            make_metric_rows(split_df)
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_path = (
        args.output_dir
        / "fold01_true_split_model_vs_era5.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    metrics_json_path = (
        args.output_dir
        / "fold01_true_split_model_vs_era5.json"
    )

    metrics_json_path.write_text(
        json.dumps(
            metric_rows,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    (
        groups_df,
        members_df,
        pairs_df,
    ) = build_duplicate_tables(
        records_df,
        tensor_cache,
        args,
    )

    groups_path = (
        args.output_dir
        / "full_input_duplicate_groups.csv"
    )
    members_path = (
        args.output_dir
        / "duplicate_input_members.csv"
    )
    pairs_path = (
        args.output_dir
        / "same_station_temporal_input_audit.csv"
    )

    groups_df.to_csv(
        groups_path,
        index=False,
        encoding="utf-8-sig",
    )

    members_df.to_csv(
        members_path,
        index=False,
        encoding="utf-8-sig",
    )

    pairs_df.to_csv(
        pairs_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "total_records": len(records_df),
        "duplicate_groups": len(groups_df),
        "duplicate_members": len(members_df),

        "conflict_groups_gt_1mm": int(
            (
                groups_df.get(
                    "target_range_mm",
                    pd.Series(dtype=float),
                )
                > 1
            ).sum()
        ),

        "conflict_groups_gt_10mm": int(
            (
                groups_df.get(
                    "target_range_mm",
                    pd.Series(dtype=float),
                )
                > 10
            ).sum()
        ),

        "conflict_groups_gt_50mm": int(
            (
                groups_df.get(
                    "target_range_mm",
                    pd.Series(dtype=float),
                )
                > 50
            ).sum()
        ),

        "suspicious_same_station_pairs": (
            len(pairs_df)
        ),
    }

    summary_path = (
        args.output_dir
        / "duplicate_input_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n完全同样本比较:")
    overall = metrics_df[
        metrics_df["target_range"]
        == "ALL"
    ]

    print(
        overall[
            [
                "split",
                "source",
                "N",
                "R",
                "NSE",
                "RMSE",
                "MAE",
                "Bias",
                "alpha",
                "slope",
                "High80_N",
                "High80_Bias",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n重复输入统计:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    focus_station = "INTGRID_R0254_C0125"

    if not pairs_df.empty:
        focus_df = pairs_df[
            pairs_df["station_id"]
            == focus_station
        ]

        if not focus_df.empty:
            print(
                f"\n重点站点 {focus_station}:"
            )

            print(
                focus_df[
                    [
                        "left_date",
                        "right_date",
                        "left_target_mm",
                        "right_target_mm",
                        "target_diff_mm",
                        "left_prediction_mm",
                        "right_prediction_mm",
                        "prediction_diff_mm",
                        "left_era5_land_mm",
                        "right_era5_land_mm",
                        "era5_diff_mm",
                        "same_full_hash_exact",
                        "same_full_hash_round6",
                        "conv_max_abs_diff",
                        "point_max_abs_diff",
                    ]
                ]
                .head(30)
                .to_string(index=False)
            )

    print("\n输出文件:")
    for path in [
        predictions_path,
        metrics_path,
        metrics_json_path,
        groups_path,
        members_path,
        pairs_path,
        summary_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
