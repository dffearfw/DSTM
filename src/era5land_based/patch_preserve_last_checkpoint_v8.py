#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(os.environ.get("ROOT", "/root/autodl-tmp")).resolve()
TARGET = ROOT / "main_tune.py"
MARKER = "PRESERVE_LAST_BEFORE_CANONICAL_V8"
POINTER_FILE = ROOT / ".preserve_last_v8_last_backup"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: 期望匹配1次，实际匹配{count}次；拒绝盲改"
        )
    return text.replace(old, new, 1)


def check_running() -> None:
    result = subprocess.run(
        [
            "pgrep",
            "-af",
            r"python.*main_tune\.py.*--mode[ =]fine_tune",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    running = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
        and "patch_preserve_last_checkpoint_v8.py" not in line
    ]
    if running and os.environ.get("FORCE_PATCH_RUNNING", "0") != "1":
        print("❌ 检测到正在运行的微调任务，拒绝修改 main_tune.py：")
        for line in running:
            print(f"   {line}")
        raise SystemExit(2)


def build_patched_text(original: str) -> str:
    required = [
        "FROZEN_RELATIVE_CANONICAL_CHECKPOINT_V1",
        'model_name = "final_fine_tuned_model.pth"',
        'self.save_dir / "best_fine_tuned_model.pth"',
        "selected_is_frozen_fallback",
    ]
    missing = [item for item in required if item not in original]
    if missing:
        raise RuntimeError(
            "当前 main_tune.py 缺少预期结构，拒绝盲改: "
            + ", ".join(missing)
        )

    text = original

    # 1. 必须在载入 canonical 之前保存当时内存中的真实最后训练态。
    canonical_anchor = '''            if fine_tune_mode:
                canonical_path = (
                    self.save_dir / "best_fine_tuned_model.pth"
                )
'''
    canonical_replacement = '''            if fine_tune_mode:
                # PRESERVE_LAST_BEFORE_CANONICAL_V8
                # 此时 self.model 仍是最后一个实际训练 epoch 的权重。
                # 必须先保存，再恢复 Frozen-relative canonical。
                last_trained_model_path = (
                    self.save_dir / "last_trained_model.pth"
                )
                last_trained_metrics = (
                    dict(val_metrics)
                    if isinstance(val_metrics, dict)
                    else {}
                )
                last_trained_metrics.update({
                    "checkpoint_role": (
                        "last_trained_before_canonical_restore"
                    ),
                    "last_completed_epoch_zero_based": int(epoch),
                    "last_completed_epoch_one_based": int(epoch) + 1,
                    "admissible_improvement_found": bool(
                        admissible_improvement_found
                    ),
                    "selected_is_frozen_fallback": bool(
                        not admissible_improvement_found
                    ),
                })
                self.save_checkpoint(
                    "last_trained_model.pth",
                    epoch,
                    last_trained_metrics,
                )

                saved_last_checkpoint = torch.load(
                    last_trained_model_path,
                    map_location="cpu",
                    weights_only=False,
                )
                saved_last_epoch = (
                    int(saved_last_checkpoint.get("epoch", -10**9))
                    if isinstance(saved_last_checkpoint, dict)
                    else -10**9
                )
                del saved_last_checkpoint
                if saved_last_epoch != int(epoch):
                    raise RuntimeError(
                        "last_trained_model.pth 的 epoch 元数据异常: "
                        f"saved={saved_last_epoch}, expected={epoch}"
                    )

                print(
                    "  💾 [PRESERVE-LAST] 已在canonical恢复前保存"
                    "真实最后训练态: "
                    f"{last_trained_model_path} "
                    f"(epoch={epoch + 1})"
                )

                canonical_path = (
                    self.save_dir / "best_fine_tuned_model.pth"
                )
'''
    text = replace_once(
        text,
        canonical_anchor,
        canonical_replacement,
        "在canonical恢复前保存last-trained",
    )

    # 2. 每折开始时清理上一折共享的 last-trained 文件。
    cleanup_anchor = '''                self.save_dir / "final_fine_tuned_model.pth",
                self.save_dir / "best_model.pth",
'''
    cleanup_replacement = '''                self.save_dir / "final_fine_tuned_model.pth",
                self.save_dir / "last_trained_model.pth",
                self.save_dir / "best_model.pth",
'''
    text = replace_once(
        text,
        cleanup_anchor,
        cleanup_replacement,
        "扩展每折共享checkpoint清理列表",
    )

    # 3. 训练返回后立刻复制为折专属文件，避免下一折清理时丢失。
    fold_path_anchor = '''            fold_frozen_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_frozen_baseline.pth"
            )
'''
    fold_path_replacement = '''            fold_frozen_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_frozen_baseline.pth"
            )
            fold_last_trained_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_last_trained.pth"
            )
'''
    text = replace_once(
        text,
        fold_path_anchor,
        fold_path_replacement,
        "定义折专属last-trained路径",
    )

    source_anchor = '''            frozen_source = (
                self.save_dir
                / "best_fine_tuned_frozen_baseline.pth"
            )

            if not canonical_source.exists():
'''
    source_replacement = '''            frozen_source = (
                self.save_dir
                / "best_fine_tuned_frozen_baseline.pth"
            )
            last_trained_source = (
                self.save_dir
                / "last_trained_model.pth"
            )

            if not last_trained_source.exists():
                raise RuntimeError(
                    f"Fold {fold_idx}缺少真实最后训练态: "
                    f"{last_trained_source}"
                )

            if not canonical_source.exists():
'''
    text = replace_once(
        text,
        source_anchor,
        source_replacement,
        "检查共享last-trained来源",
    )

    copy_anchor = '''            shutil.copy2(
                frozen_source,
                fold_frozen_path,
            )
            if global_source.exists():
'''
    copy_replacement = '''            shutil.copy2(
                frozen_source,
                fold_frozen_path,
            )
            shutil.copy2(
                last_trained_source,
                fold_last_trained_path,
            )
            if global_source.exists():
'''
    text = replace_once(
        text,
        copy_anchor,
        copy_replacement,
        "复制折专属last-trained",
    )

    print_anchor = '''            print(
                f"  💾 [FOLD {fold_idx}] Frozen基线审计副本: "
                f"{fold_frozen_path}"
            )
            if global_source.exists():
'''
    print_replacement = '''            print(
                f"  💾 [FOLD {fold_idx}] Frozen基线审计副本: "
                f"{fold_frozen_path}"
            )
            print(
                f"  💾 [FOLD {fold_idx}] 真实最后训练态: "
                f"{fold_last_trained_path}"
            )
            if global_source.exists():
'''
    text = replace_once(
        text,
        print_anchor,
        print_replacement,
        "输出折专属last-trained路径",
    )

    if text.count(MARKER) != 1:
        raise RuntimeError(
            f"补丁标记数量异常: {text.count(MARKER)}"
        )
    return text


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    check_running()
    original = TARGET.read_text(encoding="utf-8")

    if MARKER in original:
        py_compile.compile(str(TARGET), doraise=True)
        print("✅ preserve-last v8 补丁已经存在，未重复修改")
        return

    patched = build_patched_text(original)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT / "backups" / f"preserve_last_v8_before_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_path = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup_path)
    POINTER_FILE.write_text(str(backup_dir) + "\n", encoding="utf-8")

    temp_path = ROOT / ".main_tune_preserve_last_v8.tmp.py"
    temp_path.write_text(patched, encoding="utf-8")

    try:
        py_compile.compile(str(temp_path), doraise=True)
        os.replace(temp_path, TARGET)
        py_compile.compile(str(TARGET), doraise=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        shutil.copy2(backup_path, TARGET)
        raise

    print("✅ preserve-last v8 安装完成")
    print(f"   备份: {backup_path}")
    print(f"   修改: {TARGET}")
    print("   新增共享文件: last_trained_model.pth")
    print("   新增折文件:   cv_fold_XX_last_trained.pth")
    print("   保持不变:     final_fine_tuned_model.pth 仍为canonical")


if __name__ == "__main__":
    main()
