#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from main_tune import SWETrainer


def move_optimizer_to_device(optimizer, device):
    """确保恢复的AdamW状态与模型位于相同设备。"""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="从final_checkpoint_epochN.pth继续渐进式预训练最终refit"
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target-epochs",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    target_epochs = int(args.target_epochs)

    if not run_dir.is_dir():
        raise FileNotFoundError(f"实验目录不存在: {run_dir}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint不存在: {checkpoint_path}")

    print("=" * 80)
    print("渐进式预训练最终refit断点续跑")
    print(f"实验目录:      {run_dir}")
    print(f"checkpoint:    {checkpoint_path}")
    print(f"目标总轮数:    {target_epochs}")
    print("=" * 80)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = [
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "config",
    ]

    missing = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing:
        raise RuntimeError(
            f"checkpoint缺少断点续跑字段: {missing}"
        )

    completed_epoch = int(checkpoint["epoch"]) + 1
    start_epoch = completed_epoch

    if completed_epoch >= target_epochs:
        raise RuntimeError(
            f"checkpoint已经完成{completed_epoch}轮，"
            f"不需要继续到{target_epochs}轮"
        )

    config = dict(checkpoint["config"])

    # 固定使用原来的实验目录，不能生成新时间戳目录。
    config["save_dir"] = str(run_dir.parent)
    config["experiment_name"] = run_dir.name

    # 十折已经完成，重新构建Dataset时直接跳过这些折。
    config["resume_pretrain_cv"] = True
    config["redraw_completed_cv_plots"] = False

    # 恢复全量refit配置。
    config["final_train_ratio"] = 1.0
    config["final_epochs_mode"] = "fixed"
    config["final_epochs"] = target_epochs
    config["final_scheduler"] = "cosine"

    config["_is_full_refit"] = True
    config["_final_epochs"] = target_epochs
    config["_final_scheduler"] = "cosine"

    seed = int(config.get("seed", 43))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = SWETrainer(config)

    # 强制写回原来的Stage 3目录。
    trainer.save_dir = run_dir
    trainer.save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n1. 恢复固定Stage 3数据集")

    cv_results, full_dataset = trainer.run_pretrain_cv_workflow()

    if cv_results is None or full_dataset is None:
        raise RuntimeError(
            "无法恢复Stage 3十折结果或完整数据集"
        )

    print(f"   数据集样本数: {len(full_dataset):,}")
    print(f"   C_conv:       {full_dataset.C_conv}")
    print(f"   C_point:      {full_dataset.C_point}")

    trainer.config["C_conv"] = full_dataset.C_conv
    trainer.config["C_point"] = full_dataset.C_point

    trainer._bind_swe_scale_from_dataset(
        full_dataset,
        context="resume_progressive_final_refit",
    )

    batch_size = int(
        trainer.config.get("batch_size", 128)
    )
    num_workers = int(
        trainer.config.get("num_workers", 4)
    )

    drop_last = (
        len(full_dataset) % batch_size == 1
    )

    trainer.train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(
            2 if num_workers > 0 else None
        ),
    )

    trainer.val_loader = None

    print("\n2. 重建模型、优化器和Cosine调度器")

    previous_stage_model = trainer.config.get(
        "pretrained_model"
    )

    if not previous_stage_model:
        raise RuntimeError(
            "checkpoint config中没有上一阶段pretrained_model"
        )

    success = trainer.build_model(
        load_pretrained=previous_stage_model,
        freeze_backbone=False,
        is_cv_fold=False,
    )

    if not success:
        raise RuntimeError("模型重建失败")

    print("\n3. 恢复epoch checkpoint状态")

    incompatible = trainer.model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if incompatible.missing_keys:
        raise RuntimeError(
            f"模型缺失参数: {incompatible.missing_keys}"
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"模型存在额外参数: {incompatible.unexpected_keys}"
        )

    trainer.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    trainer.scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    move_optimizer_to_device(
        trainer.optimizer,
        trainer.device,
    )

    trainer.train_history = list(
        checkpoint.get("train_history", [])
    )
    trainer.val_history = list(
        checkpoint.get("val_history", [])
    )
    trainer.lr_history = list(
        checkpoint.get("lr_history", [])
    )
    trainer.fine_tune_history = list(
        checkpoint.get("fine_tune_history", [])
    )

    trainer.model.to(trainer.device)

    print(f"   已完成轮数:       {completed_epoch}")
    print(f"   下一轮:           {start_epoch + 1}")
    print(
        "   当前学习率:       "
        f"{trainer.optimizer.param_groups[0]['lr']:.8e}"
    )
    print(
        "   训练历史长度:     "
        f"{len(trainer.train_history)}"
    )

    if len(trainer.train_history) != completed_epoch:
        print(
            "   ⚠ 训练历史长度与epoch编号不一致，"
            "但模型/优化器/调度器仍可继续"
        )

    resume_record = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "completed_epoch": completed_epoch,
        "resume_start_epoch": start_epoch + 1,
        "target_epochs": target_epochs,
        "previous_stage_model": previous_stage_model,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
    }

    record_path = (
        run_dir
        / "final_refit_resume_record.json"
    )

    record_path.write_text(
        json.dumps(
            resume_record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"   续跑记录:         {record_path}")

    print("\n4. 继续训练")

    last_train_loss = None
    last_lr_before_step = None

    for epoch in range(start_epoch, target_epochs):
        epoch_number = epoch + 1

        print("\n" + "=" * 70)
        print(
            f"Stage 3 refit Epoch "
            f"{epoch_number}/{target_epochs}"
        )
        print("=" * 70)

        train_loss = trainer.train_epoch(
            epoch,
            is_fine_tune=False,
        )

        trainer.train_history.append(
            float(train_loss)
        )

        current_lr = float(
            trainer.optimizer.param_groups[0]["lr"]
        )

        trainer.lr_history.append(current_lr)

        if isinstance(
            trainer.scheduler,
            optim.lr_scheduler.CosineAnnealingLR,
        ):
            trainer.scheduler.step()
        else:
            trainer.scheduler.step(train_loss)

        next_lr = float(
            trainer.optimizer.param_groups[0]["lr"]
        )

        print(f"训练损失:       {train_loss:.8f}")
        print(f"本轮学习率:     {current_lr:.8e}")
        print(f"下一轮学习率:   {next_lr:.8e}")

        # 保持原代码的checkpoint保存规则。
        save_checkpoint_now = (
            epoch_number % 10 == 0
            or epoch >= target_epochs - 5
        )

        if save_checkpoint_now:
            checkpoint_name = (
                f"final_checkpoint_epoch"
                f"{epoch_number}.pth"
            )

            trainer.save_checkpoint(
                checkpoint_name,
                epoch,
                {
                    "loss": float(train_loss),
                    "lr": current_lr,
                    "resumed_from": str(
                        checkpoint_path
                    ),
                },
            )

            print(
                f"✅ 已保存: {checkpoint_name}"
            )

        last_train_loss = float(train_loss)
        last_lr_before_step = current_lr

    if last_train_loss is None:
        raise RuntimeError("没有执行任何续跑epoch")

    final_metrics = {
        "loss": last_train_loss,
        "lr": last_lr_before_step,
        "resumed_from": str(checkpoint_path),
        "resume_start_epoch": start_epoch + 1,
        "target_epochs": target_epochs,
    }

    trainer.save_checkpoint(
        "final_model.pth",
        target_epochs - 1,
        final_metrics,
    )

    trainer.save_checkpoint(
        f"final_full_epoch_{target_epochs}.pth",
        target_epochs - 1,
        final_metrics,
    )

    result_path = (
        run_dir
        / "pretrain_progressive_results.json"
    )

    existing_result = {}

    if result_path.exists():
        try:
            existing_result = json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            existing_result = {}

    existing_result["model_path"] = str(
        run_dir / "final_model.pth"
    )

    existing_result["resume"] = {
        "resumed": True,
        "checkpoint": str(checkpoint_path),
        "completed_epoch_before_resume": (
            completed_epoch
        ),
        "target_epochs": target_epochs,
        "final_train_loss": last_train_loss,
        "final_lr": last_lr_before_step,
        "timestamp": datetime.now().isoformat(),
    }

    result_path.write_text(
        json.dumps(
            existing_result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("✅ Stage 3最终refit续跑完成")
    print(
        f"最终模型: "
        f"{run_dir / 'final_model.pth'}"
    )
    print(
        f"第{target_epochs}轮副本: "
        f"{run_dir / f'final_full_epoch_{target_epochs}.pth'}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
