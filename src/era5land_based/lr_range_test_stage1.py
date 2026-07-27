#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from main_tune import SWETrainer
from data_online_era5_swe import SWEDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def parse_batch(batch_data):
    """兼容当前Dataset的3～7元素返回格式。"""
    grid_val_norm = None
    sample_idx = None
    source_flag = None

    if len(batch_data) == 7:
        (
            conv_feats,
            point_feats,
            targets,
            is_zero_mask,
            grid_val_norm,
            sample_idx,
            source_flag,
        ) = batch_data

    elif len(batch_data) == 6:
        (
            conv_feats,
            point_feats,
            targets,
            is_zero_mask,
            grid_val_norm,
            sample_idx,
        ) = batch_data

    elif len(batch_data) == 5:
        (
            conv_feats,
            point_feats,
            targets,
            is_zero_mask,
            grid_val_norm,
        ) = batch_data

    elif len(batch_data) == 4:
        conv_feats, point_feats, targets, is_zero_mask = batch_data

    elif len(batch_data) == 3:
        conv_feats, point_feats, targets = batch_data
        is_zero_mask = torch.where(
            targets > 0,
            torch.ones_like(targets),
            torch.zeros_like(targets),
        )

    else:
        raise RuntimeError(
            f"未知batch长度: {len(batch_data)}"
        )

    return (
        conv_feats,
        point_feats,
        targets,
        is_zero_mask,
        grid_val_norm,
        sample_idx,
        source_flag,
    )


def prepare_point_features(
    point_feats: torch.Tensor,
    expected_dim: int,
) -> torch.Tensor:
    if point_feats.shape[1] < expected_dim:
        padding = torch.zeros(
            point_feats.shape[0],
            expected_dim - point_feats.shape[1],
            device=point_feats.device,
            dtype=point_feats.dtype,
        )
        point_feats = torch.cat(
            [point_feats, padding],
            dim=1,
        )

    elif point_feats.shape[1] > expected_dim:
        point_feats = point_feats[:, :expected_dim]

    return point_feats.contiguous()


def build_dataset(args) -> SWEDataset:
    print("=" * 78)
    print("加载Stage 1固定样本")
    print("=" * 78)

    dataset = SWEDataset(
        region="XINJIANG",
        year_target=[2015, 2016, 2017, 2018],
        patch_size=5,
        min_valid_pixels=100,
        samples_per_day=0,
        clamday_threshold=0.5,
        s1_interp_method="nearest",
        s1_max_gap_days=7,
        smap_interp_method="nearest",
        smap_max_gap_days=7,

        # 与正式Stage脚本一致：不读写Dataset pickle缓存
        cache_dir=None,
        force_reload=False,

        sampling_mode="incremental",
        use_station_guide=False,
        station_guide_file=args.station_guide_file,
        station_csv_dir=Path("/root/ablation"),
        station_neighborhood=0,

        external_station_glob=args.external_station_glob,
        external_station_exclusion_radius=0,
        external_station_strict=True,
        external_station_report_path=args.external_report,

        incremental_manifest_path=args.manifest,
        incremental_stage=1,
        build_incremental_manifest=False,
        incremental_pool_size=152000,
        incremental_stage_sizes=[
            12000,
            20000,
            40000,
            80000,
        ],
        incremental_seed=args.seed,
        incremental_candidate_oversample_factor=3.0,
        incremental_exclude_station_pixels=True,
        incremental_ratio_config=args.ratio_config,
        incremental_fold_block_pixels=0,

        seasonal_min_peak_swe_mm=1.0,
        seasonal_max_swe_mm=400.0,
        seasonal_snow_free_threshold_mm=1.0,
        seasonal_min_warm_snow_free_ratio=0.0,
        seasonal_min_consecutive_snow_free_days=5,
        seasonal_min_snow_year_coverage_ratio=0.90,

        normalization_config_path=args.normalization,
        normalization_mode="load",
        fixed_label_min_mm=0.0,
        fixed_label_max_mm=400.0,

        use_adaptive_supplement=False,
        adaptive_alpha=0.5,
        adaptive_threshold=1.5,
    )

    print(f"Stage 1样本数: {len(dataset):,}")
    print(f"C_conv: {dataset.C_conv}")
    print(f"C_point: {dataset.C_point}")

    if len(dataset) != 12000:
        raise RuntimeError(
            f"Stage 1样本数错误: {len(dataset):,}，预期12,000"
        )

    return dataset


def choose_lr_points(
    lr_values: np.ndarray,
    smooth_losses: np.ndarray,
) -> dict:
    finite = (
        np.isfinite(lr_values)
        & np.isfinite(smooth_losses)
        & (lr_values > 0)
    )

    lrs = lr_values[finite]
    losses = smooth_losses[finite]

    if len(lrs) < 25:
        return {
            "available": False,
            "reason": "有效step不足25，不能可靠自动分析",
        }

    # 忽略最初10步的偏差修正不稳定区
    start_idx = min(10, len(lrs) - 2)

    relative_min_idx = int(
        np.argmin(losses[start_idx:])
    )
    min_idx = start_idx + relative_min_idx

    log_lrs = np.log10(lrs)
    gradients = np.gradient(losses, log_lrs)

    # 最陡下降点只在最低loss之前寻找
    search_end = max(start_idx + 2, min_idx)
    steep_relative_idx = int(
        np.argmin(
            gradients[start_idx:search_end + 1]
        )
    )
    steep_idx = start_idx + steep_relative_idx

    min_loss_lr = float(lrs[min_idx])
    steepest_lr = float(lrs[steep_idx])
    min_loss_div10 = min_loss_lr / 10.0

    return {
        "available": True,
        "minimum_loss_lr": min_loss_lr,
        "minimum_loss": float(losses[min_idx]),
        "steepest_descent_lr": steepest_lr,
        "steepest_gradient": float(
            gradients[steep_idx]
        ),
        "minimum_loss_lr_div10": min_loss_div10,

        # 这里只给诊断区间，不自动修改正式训练
        "candidate_max_lr_lower": float(
            min(steepest_lr, min_loss_div10)
        ),
        "candidate_max_lr_upper": float(
            max(steepest_lr, min_loss_div10)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage0-model",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        default=(
            "/root/autodl-tmp/shared_cache/"
            "incremental_random_pool_152000.csv"
        ),
    )

    parser.add_argument(
        "--normalization",
        default=(
            "/root/autodl-tmp/shared_cache/"
            "progressive_pretrain_normalization.json"
        ),
    )

    parser.add_argument(
        "--ratio-config",
        default=(
            "/root/autodl-tmp/"
            "incremental_swe_ratios.json"
        ),
    )

    parser.add_argument(
        "--station-guide-file",
        default=(
            "/root/ablation/"
            "station_swe_data.xlsx"
        ),
    )

    parser.add_argument(
        "--external-station-glob",
        default=(
            "/root/ablation/"
            "external_test/*.csv"
        ),
    )

    parser.add_argument(
        "--external-report",
        default=(
            "/root/autodl-tmp/shared_cache/"
            "external_station_exclusion_report.csv"
        ),
    )

    parser.add_argument(
        "--start-lr",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--end-lr",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--smooth-beta",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--diverge-factor",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=43,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    for file_path in [
        args.stage0_model,
        args.manifest,
        args.normalization,
        args.ratio_config,
        args.station_guide_file,
        args.external_report,
    ]:
        if not Path(file_path).exists():
            raise FileNotFoundError(file_path)

    if not (
        args.start_lr > 0
        and args.end_lr > args.start_lr
    ):
        raise ValueError(
            "必须满足0 < start_lr < end_lr"
        )

    if args.num_steps < 50:
        raise ValueError(
            "num_steps至少应为50"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(args.seed)

    config = {
        "mode": "lr_range_test",
        "model_type": "full",

        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": 2,
        "persistent_workers": (
            args.num_workers > 0
        ),

        "epochs": 1,
        "learning_rate": args.start_lr,
        "weight_decay": 1e-5,
        "d_model": 256,

        "pretrain_years": [
            2015,
            2016,
            2017,
            2018,
        ],
        "pretrain_samples_per_day": 0,
        "patch_size": 5,
        "min_valid_pixels": 100,
        "clamday_threshold": 0.5,

        "sampling_mode": "incremental",
        "incremental_manifest_path": args.manifest,
        "incremental_stage": 1,
        "incremental_pool_size": 152000,
        "incremental_stage_sizes": [
            12000,
            20000,
            40000,
            80000,
        ],
        "incremental_seed": args.seed,
        "incremental_ratio_config": (
            args.ratio_config
        ),

        "normalization_config_path": (
            args.normalization
        ),
        "normalization_mode": "load",
        "fixed_label_min_mm": 0.0,
        "fixed_label_max_mm": 400.0,

        "pretrained_model": args.stage0_model,

        # 与正式Stage 1保持一致
        "use_amp": True,
        "use_mixup": False,
        "mixed_mode": False,
        "clip_grad": 1.0,

        "save_dir": str(
            output_dir.parent
        ),
        "experiment_name": (
            output_dir.name
        ),
        "seed": args.seed,
    }

    trainer = SWETrainer(config)
    dataset = build_dataset(args)

    trainer.config["C_conv"] = (
        dataset.C_conv
    )
    trainer.config["C_point"] = (
        dataset.C_point
    )

    trainer._bind_swe_scale_from_dataset(
        dataset,
        context="lr_range_test_stage1",
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }

    if args.num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })

    train_loader = DataLoader(
        **loader_kwargs
    )
    trainer.train_loader = train_loader

    print("=" * 78)
    print("构建Stage 1起点模型")
    print(f"Stage 0模型: {args.stage0_model}")
    print("=" * 78)

    ok = trainer.build_model(
        load_pretrained=args.stage0_model,
        freeze_backbone=False,
        freeze_strategy="none",
        is_cv_fold=False,
    )

    if not ok:
        raise RuntimeError(
            "模型构建失败"
        )

    # Range Test期间完全不调用原scheduler
    trainer.scheduler = None

    for group in trainer.optimizer.param_groups:
        group["lr"] = args.start_lr

    trainer.model.train()

    device = trainer.device
    use_amp = (
        bool(trainer.config.get(
            "use_amp",
            False,
        ))
        and device.type == "cuda"
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    lr_multiplier = (
        args.end_lr / args.start_lr
    ) ** (1.0 / max(
        args.num_steps - 1,
        1,
    ))

    iterator = iter(train_loader)

    lr_history = []
    raw_loss_history = []
    smooth_loss_history = []

    moving_average = 0.0
    best_smooth_loss = float("inf")

    expected_point_dim = int(
        trainer.config["C_point"]
    )

    print()
    print("=" * 78)
    print("开始Stage 1 LR Range Test")
    print(
        f"LR: {args.start_lr:.2e} "
        f"→ {args.end_lr:.2e}"
    )
    print(
        f"最多optimizer steps: "
        f"{args.num_steps}"
    )
    print(
        "正式scheduler已禁用；"
        "本次模型不会保存为正式模型"
    )
    print("=" * 78)

    stopped_reason = (
        "completed_all_steps"
    )

    for step in range(args.num_steps):
        current_lr = (
            args.start_lr
            * (lr_multiplier ** step)
        )

        for group in trainer.optimizer.param_groups:
            group["lr"] = current_lr

        try:
            batch_data = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch_data = next(iterator)

        (
            conv_feats,
            point_feats,
            targets,
            is_zero_mask,
            _,
            _,
            _,
        ) = parse_batch(batch_data)

        conv_feats = torch.nan_to_num(
            conv_feats,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        point_feats = torch.nan_to_num(
            point_feats,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        targets = torch.nan_to_num(
            targets,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        is_zero_mask = torch.nan_to_num(
            is_zero_mask,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

        conv_feats = conv_feats.to(
            device,
            non_blocking=True,
        ).contiguous()

        point_feats = point_feats.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        is_zero_mask = is_zero_mask.to(
            device,
            non_blocking=True,
        )

        point_feats = prepare_point_features(
            point_feats,
            expected_point_dim,
        )

        trainer.optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            outputs = trainer.model(
                conv_feats,
                point_feats,
            )

            outputs_flat = outputs.reshape(-1)
            targets_flat = targets.reshape(-1)
            zero_mask_flat = (
                is_zero_mask.reshape(-1)
            )

            if (
                zero_mask_flat.numel()
                == outputs_flat.numel()
            ):
                outputs_flat = (
                    outputs_flat
                    * zero_mask_flat
                )

            loss = trainer.criterion(
                outputs_flat,
                targets_flat,
            )

            if loss.dim() > 0:
                loss = loss.mean()

        raw_loss = float(
            loss.detach().item()
        )

        if not math.isfinite(raw_loss):
            stopped_reason = (
                "non_finite_loss"
            )
            print(
                f"\n🛑 Step {step + 1}: "
                "loss出现NaN/Inf"
            )
            break

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(
                trainer.optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                trainer.optimizer
            )
            scaler.update()

        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(),
                max_norm=1.0,
            )

            trainer.optimizer.step()

        moving_average = (
            args.smooth_beta
            * moving_average
            + (1.0 - args.smooth_beta)
            * raw_loss
        )

        smooth_loss = (
            moving_average
            / (
                1.0
                - args.smooth_beta
                ** (step + 1)
            )
        )

        lr_history.append(current_lr)
        raw_loss_history.append(raw_loss)
        smooth_loss_history.append(
            smooth_loss
        )

        if step >= 10:
            if (
                step >= 20
                and smooth_loss
                > args.diverge_factor
                * best_smooth_loss
            ):
                stopped_reason = (
                    "loss_diverged"
                )
                print(
                    f"\n🛑 Step {step + 1}: "
                    f"平滑loss={smooth_loss:.6f}，"
                    f"超过最佳loss的"
                    f"{args.diverge_factor:.1f}倍"
                )
                break

            best_smooth_loss = min(
                best_smooth_loss,
                smooth_loss,
            )

        if (
            step == 0
            or (step + 1) % 10 == 0
        ):
            print(
                f"Step {step + 1:4d}/"
                f"{args.num_steps} | "
                f"LR={current_lr:.3e} | "
                f"raw={raw_loss:.6f} | "
                f"smooth={smooth_loss:.6f}"
            )

    results = pd.DataFrame({
        "step": np.arange(
            1,
            len(lr_history) + 1,
        ),
        "lr": lr_history,
        "raw_loss": raw_loss_history,
        "smooth_loss": smooth_loss_history,
    })

    csv_path = (
        output_dir
        / "lr_range_history.csv"
    )
    results.to_csv(
        csv_path,
        index=False,
    )

    lr_array = np.asarray(
        lr_history,
        dtype=np.float64,
    )
    smooth_array = np.asarray(
        smooth_loss_history,
        dtype=np.float64,
    )

    analysis = choose_lr_points(
        lr_array,
        smooth_array,
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "stage": 1,
        "stage0_model": args.stage0_model,
        "manifest": args.manifest,
        "stage1_samples": len(dataset),
        "batch_size": args.batch_size,
        "requested_steps": args.num_steps,
        "completed_steps": len(results),
        "start_lr": args.start_lr,
        "end_lr": args.end_lr,
        "smooth_beta": args.smooth_beta,
        "diverge_factor": args.diverge_factor,
        "stopped_reason": stopped_reason,
        "analysis": analysis,
    }

    json_path = (
        output_dir
        / "lr_range_summary.json"
    )
    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    plt.figure(
        figsize=(10, 7)
    )

    if len(results) > 0:
        plt.plot(
            results["lr"],
            results["raw_loss"],
            alpha=0.25,
            label="Raw batch loss",
        )
        plt.plot(
            results["lr"],
            results["smooth_loss"],
            linewidth=2,
            label="Smoothed loss",
        )

    if analysis.get(
        "available",
        False,
    ):
        plt.axvline(
            analysis[
                "steepest_descent_lr"
            ],
            linestyle="--",
            label=(
                "Steepest descent "
                f"{analysis['steepest_descent_lr']:.2e}"
            ),
        )

        plt.axvline(
            analysis[
                "minimum_loss_lr"
            ],
            linestyle=":",
            label=(
                "Minimum loss "
                f"{analysis['minimum_loss_lr']:.2e}"
            ),
        )

        plt.axvline(
            analysis[
                "minimum_loss_lr_div10"
            ],
            linestyle="-.",
            label=(
                "Minimum LR / 10 "
                f"{analysis['minimum_loss_lr_div10']:.2e}"
            ),
        )

    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Training loss")
    plt.title(
        "Stage 1 Learning Rate Range Test"
    )
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    plot_path = (
        output_dir
        / "lr_range_test.png"
    )
    plt.savefig(
        plot_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    print()
    print("=" * 78)
    print("LR Range Test完成")
    print("=" * 78)
    print(
        f"完成steps: {len(results)}"
    )
    print(
        f"停止原因: {stopped_reason}"
    )
    print(f"CSV:  {csv_path}")
    print(f"曲线: {plot_path}")
    print(f"汇总: {json_path}")

    if analysis.get(
        "available",
        False,
    ):
        print()
        print("自动诊断结果：")
        print(
            "  最陡下降处LR: "
            f"{analysis['steepest_descent_lr']:.3e}"
        )
        print(
            "  最低loss处LR: "
            f"{analysis['minimum_loss_lr']:.3e}"
        )
        print(
            "  最低loss处LR / 10: "
            f"{analysis['minimum_loss_lr_div10']:.3e}"
        )
        print(
            "  候选max_lr区间: "
            f"[{analysis['candidate_max_lr_lower']:.3e}, "
            f"{analysis['candidate_max_lr_upper']:.3e}]"
        )

    print()
    print(
        "⚠ 本次参数已经被LR Test临时更新；"
        "不得把本进程模型用于正式训练。"
    )
    print(
        "✅ 正式Stage 1仍应重新加载原Stage 0 final_model.pth。"
    )


if __name__ == "__main__":
    main()
