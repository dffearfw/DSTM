#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import py_compile
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path("/root/autodl-tmp")
TARGET = ROOT / "main_tune.py"
MARKER = "DUAL_CHECKPOINT_SELECTION_DIAG_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: 期望匹配1次，实际匹配{count}次"
        )
    return text.replace(old, new, 1)


def insert_after_once(
    text: str,
    anchor: str,
    addition: str,
    label: str,
) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"{label}: 期望匹配1次，实际匹配{count}次"
        )
    return text.replace(anchor, anchor + addition, 1)


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
        and "patch_dual_checkpoint_selection.py" not in line
    ]

    if running and os.environ.get(
        "FORCE_PATCH_RUNNING", "0"
    ) != "1":
        print("❌ 检测到正在运行的微调任务，拒绝修改代码：")
        for line in running:
            print(f"   {line}")
        print("请先停止当前微调。")
        raise SystemExit(2)


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    check_running()

    original = TARGET.read_text(encoding="utf-8")

    if MARKER in original:
        py_compile.compile(str(TARGET), doraise=True)
        print("✅ 双checkpoint补丁已经存在")
        return

    required = [
        "FINETUNE_EPOCH0_BASELINE_V1",
        "FINETUNE_COLLAPSE_GUARD_V1",
        "best_selection_score",
        "rmse_mm",
    ]
    missing = [x for x in required if x not in original]

    if missing:
        raise RuntimeError(
            "当前main_tune.py缺少预期结构，拒绝盲改: "
            + ", ".join(missing)
        )

    text = original

    # ==========================================================
    # 1. 初始化独立的全局RMSE最优状态
    # ==========================================================
    init_anchor = (
        '        best_selection_score = float("inf")\n'
    )

    init_addition = '''        # DUAL_CHECKPOINT_SELECTION_DIAG_V1
        # 复合评分与全局RMSE分别追踪，互不替代。
        best_global_rmse_mm = float("inf")
        best_global_rmse_epoch = None
'''

    text = insert_after_once(
        text,
        init_anchor,
        init_addition,
        "初始化全局RMSE状态",
    )

    # ==========================================================
    # 2. epoch-0基线也参与全局RMSE选择
    # ==========================================================
    baseline_anchor = '''            baseline_collapsed = bool(
                baseline_metrics.get("is_collapsed", False)
            )
'''

    baseline_addition = '''
            baseline_global_rmse_mm = float(
                baseline_metrics.get(
                    "rmse_mm",
                    float("inf"),
                )
            )

            if (
                np.isfinite(baseline_global_rmse_mm)
                and not baseline_collapsed
            ):
                best_global_rmse_mm = baseline_global_rmse_mm
                best_global_rmse_epoch = -1

                self.save_checkpoint(
                    "best_fine_tuned_global_rmse.pth",
                    -1,
                    baseline_metrics,
                )

                print(
                    "  ✅ epoch-0全局RMSE基线已保存: "
                    f"RMSE={best_global_rmse_mm:.2f} mm"
                )
'''

    text = insert_after_once(
        text,
        baseline_anchor,
        baseline_addition,
        "保存epoch-0全局RMSE模型",
    )

    # ==========================================================
    # 3. 每个验证epoch独立更新全局RMSE模型
    # ==========================================================
    epoch_anchor = '''                    is_collapsed = bool(
                        val_metrics.get("is_collapsed", False)
                    )
'''

    epoch_addition = '''
                    # DUAL_CHECKPOINT_SELECTION_DIAG_V1
                    current_global_rmse_mm = float(
                        val_metrics.get(
                            "rmse_mm",
                            float("inf"),
                        )
                    )

                    global_rmse_is_better = bool(
                        np.isfinite(current_global_rmse_mm)
                        and not is_collapsed
                        and current_global_rmse_mm
                        < best_global_rmse_mm
                    )

                    if global_rmse_is_better:
                        best_global_rmse_mm = (
                            current_global_rmse_mm
                        )
                        best_global_rmse_epoch = int(epoch)

                        self.save_checkpoint(
                            "best_fine_tuned_global_rmse.pth",
                            epoch,
                            val_metrics,
                        )

                        print(
                            "\\n💾 保存最低全局RMSE模型: "
                            "best_fine_tuned_global_rmse.pth "
                            f"(Epoch {epoch + 1}, "
                            f"RMSE={best_global_rmse_mm:.2f} mm, "
                            f"R={val_metrics.get('correlation', float('nan')):.4f}, "
                            f"RMSE50={val_metrics.get('rmse_ge50_mm', float('nan')):.2f} mm, "
                            f"Bias80={val_metrics.get('bias_ge80_mm', float('nan')):.2f} mm, "
                            f"slope={val_metrics.get('slope', float('nan')):.3f})"
                        )
'''

    text = insert_after_once(
        text,
        epoch_anchor,
        epoch_addition,
        "逐epoch更新全局RMSE模型",
    )

    # ==========================================================
    # 4. 每折开始前清理上一折的全局RMSE文件
    # ==========================================================
    cleanup_anchor = '''            shared_fold_candidates = [
                self.save_dir / "best_fine_tuned_model.pth",
'''

    if cleanup_anchor in text:
        cleanup_new = '''            shared_fold_candidates = [
                self.save_dir / "best_fine_tuned_model.pth",
                self.save_dir / "best_fine_tuned_global_rmse.pth",
'''
        text = replace_once(
            text,
            cleanup_anchor,
            cleanup_new,
            "扩展每折checkpoint清理列表",
        )
    else:
        train_anchor = '''            # ============ 8.6 训练 ============
            print(f"\\n🚀 [FOLD {fold_idx}] 开始训练...")
'''

        train_new = '''            # DUAL_CHECKPOINT_SELECTION_DIAG_V1
            for stale_name in (
                "best_fine_tuned_model.pth",
                "best_fine_tuned_global_rmse.pth",
                "final_fine_tuned_model.pth",
            ):
                stale_path = self.save_dir / stale_name
                if stale_path.exists():
                    stale_path.unlink()
                    print(
                        f"  🧹 [FOLD {fold_idx}] "
                        f"清除上一折checkpoint: "
                        f"{stale_name}"
                    )

            # ============ 8.6 训练 ============
            print(f"\\n🚀 [FOLD {fold_idx}] 开始训练...")
'''

        text = replace_once(
            text,
            train_anchor,
            train_new,
            "插入每折checkpoint清理",
        )

    # ==========================================================
    # 5. 每折同时保存复合评分与全局RMSE checkpoint
    # ==========================================================
    fold_line_pattern = re.compile(
        r'''
        (?P<indent>[ ]{12})
        \#\ ============\ 8\.7\ 保存当前\ fold\ 模型，
        避免被后续\ fold\ 覆盖\ ============\n
        (?P=indent)
        fold_model_path\ =\ self\.save_dir\ /\ 
        f"cv_fold_\{fold_idx(?::02d)?\}_best_model\.pth"\n
        ''',
        re.VERBOSE,
    )

    matches = list(fold_line_pattern.finditer(text))

    if len(matches) != 1:
        raise RuntimeError(
            "无法唯一定位Fold模型保存代码，"
            f"匹配数={len(matches)}"
        )

    fold_block = '''            # ============ 8.7 保存当前 fold 模型，避免被后续 fold 覆盖 ============
            # DUAL_CHECKPOINT_SELECTION_DIAG_V1
            #
            # 复合评分模型：
            #   cv_fold_XX_best_composite.pth
            #
            # 全局RMSE模型：
            #   cv_fold_XX_best_global_rmse.pth
            #
            # 兼容现有即时测试的canonical路径：
            #   cv_fold_XX_best_model.pth
            # 默认指向全局RMSE模型。
            fold_model_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_best_model.pth"
            )
            fold_global_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_best_global_rmse.pth"
            )
            fold_composite_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_best_composite.pth"
            )

            global_source = (
                self.save_dir
                / "best_fine_tuned_global_rmse.pth"
            )
            composite_source = (
                self.save_dir
                / "best_fine_tuned_model.pth"
            )

            if not global_source.exists():
                raise RuntimeError(
                    f"Fold {fold_idx}缺少全局RMSE模型: "
                    f"{global_source}"
                )

            if not composite_source.exists():
                raise RuntimeError(
                    f"Fold {fold_idx}缺少复合评分模型: "
                    f"{composite_source}"
                )

            # 先保存原始复合评分模型。
            shutil.copy2(
                composite_source,
                fold_composite_path,
            )

            # 保存全局RMSE模型。
            shutil.copy2(
                global_source,
                fold_global_path,
            )

            # 现有即时测试继续读取该文件，
            # 因此canonical路径切换为全局RMSE模型。
            shutil.copy2(
                global_source,
                fold_model_path,
            )

            print(
                f"  💾 [FOLD {fold_idx}] "
                f"复合评分模型: {fold_composite_path}"
            )
            print(
                f"  💾 [FOLD {fold_idx}] "
                f"全局RMSE模型: {fold_global_path}"
            )
            print(
                f"  📌 [FOLD {fold_idx}] "
                f"固定测试默认使用全局RMSE模型: "
                f"{fold_model_path}"
            )

            # 将共享best文件切换为全局RMSE模型。
            # 当前Fold训练已经结束，不影响训练过程或塌缩回滚；
            # 同时保证外层runner最终找到的也是全局RMSE模型。
            shutil.copy2(
                global_source,
                composite_source,
            )
'''

    text = (
        text[:matches[0].start()]
        + fold_block
        + text[matches[0].end():]
    )

    # 原有候选列表也将global RMSE放在第一位。
    candidate_anchor = '''            candidate_model_paths = [
                self.save_dir / "best_fine_tuned_model.pth",
'''

    if candidate_anchor in text:
        candidate_new = '''            candidate_model_paths = [
                self.save_dir / "best_fine_tuned_global_rmse.pth",
                self.save_dir / "best_fine_tuned_model.pth",
'''

        text = replace_once(
            text,
            candidate_anchor,
            candidate_new,
            "调整Fold候选模型优先级",
        )

    if MARKER not in text:
        raise RuntimeError("补丁标记未成功写入")

    # ==========================================================
    # 6. 备份、语法检查、原子写回
    # ==========================================================
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_dir = (
        ROOT
        / "code_backups"
        / f"before_dual_checkpoint_selection_{stamp}"
    )
    backup_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    backup_path = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup_path)

    temp_path = ROOT / "main_tune_dual_checkpoint_tmp.py"
    temp_path.write_text(text, encoding="utf-8")

    try:
        py_compile.compile(
            str(temp_path),
            doraise=True,
        )
        shutil.move(
            str(temp_path),
            str(TARGET),
        )
        py_compile.compile(
            str(TARGET),
            doraise=True,
        )
    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        shutil.copy2(
            backup_path,
            TARGET,
        )
        raise

    print("✅ 双checkpoint诊断补丁安装完成")
    print(f"   备份: {backup_path}")
    print(f"   文件: {TARGET}")
    print()
    print("未修改：")
    print("  - 训练损失")
    print("  - 高值权重")
    print("  - 方差约束")
    print("  - 复合评分公式")
    print("  - 没有增加RMSE硬护栏")
    print()
    print("新增：")
    print("  - best_fine_tuned_global_rmse.pth")
    print("  - cv_fold_XX_best_global_rmse.pth")
    print("  - cv_fold_XX_best_composite.pth")
    print("  - cv_fold_XX_best_model.pth默认指向global RMSE")


if __name__ == "__main__":
    main()
