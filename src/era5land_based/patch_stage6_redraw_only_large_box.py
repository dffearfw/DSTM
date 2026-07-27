#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 /root/autodl-tmp/main_tune.py 安装 Stage 6 单折散点图只重绘补丁。

功能：
1. 新增 --redraw_completed_cv_plots_only：严格只重绘，任何不完整折立即停止，绝不训练。
2. 已完成折加载 best checkpoint 做一次推理，覆盖 density_scatter_chinese_fold_N.png。
3. 指标框、标题、坐标轴、图例和色标字体放大。
4. 保存 pretrain_cv_foldN_predictions.csv，后续可完全脱离模型从 CSV 重画。
"""
from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/root/autodl-tmp"))
TARGET = ROOT / "main_tune.py"
MARKER = "STAGE6_REDRAW_ONLY_LARGE_BOX_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: 期望匹配1次，实际匹配{count}次")
    return text.replace(old, new, 1)


def check_running() -> None:
    result = subprocess.run(
        ["pgrep", "-af", r"python.*main_tune\.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    running = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "patch_stage6_redraw_only_large_box.py" not in line
    ]
    if running and os.environ.get("FORCE_PATCH_RUNNING", "0") != "1":
        print("❌ 检测到 main_tune.py 正在运行，拒绝修改：")
        for line in running:
            print(f"   {line}")
        print("请先停止任务；确需强制修改可设置 FORCE_PATCH_RUNNING=1。")
        raise SystemExit(2)


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    check_running()
    original = TARGET.read_text(encoding="utf-8")

    if MARKER in original:
        py_compile.compile(str(TARGET), doraise=True)
        print("✅ Stage 6 只重绘与大指标框补丁已经存在")
        return

    text = original

    text = replace_once(
        text,
        '''            "resume_pretrain_cv": False,\n            "redraw_completed_cv_plots": False,\n''',
        '''            "resume_pretrain_cv": False,\n            "redraw_completed_cv_plots": False,\n            # STAGE6_REDRAW_ONLY_LARGE_BOX_V1\n            # 严格只重绘：已完成折加载best checkpoint做推理并覆盖图；\n            # 任一折不完整时立即终止，绝不进入训练。\n            "redraw_completed_cv_plots_only": False,\n\n            # 单折密度散点图排版。\n            "density_scatter_fig_width": 12.0,\n            "density_scatter_fig_height": 9.0,\n            "density_scatter_metrics_fontsize": 22,\n            "density_scatter_metrics_pad": 0.90,\n            "density_scatter_metrics_linespacing": 1.25,\n''',
        "新增默认配置",
    )

    text = replace_once(
        text,
        '''        resume_pretrain_cv = bool(self.config.get("resume_pretrain_cv", False))\n        redraw_completed_cv_plots = bool(self.config.get("redraw_completed_cv_plots", False))\n        if resume_pretrain_cv:\n            print("\\n♻️ 已启用预训练CV按折续跑")\n            print(f"   续跑实验目录: {self.save_dir}")\n            if redraw_completed_cv_plots:\n                print("   🎨 已完成折将加载 best checkpoint 重新生成正确 mm 尺度散点图（不训练）")\n''',
        '''        resume_pretrain_cv = bool(self.config.get("resume_pretrain_cv", False))\n        redraw_completed_cv_plots = bool(self.config.get("redraw_completed_cv_plots", False))\n        redraw_completed_cv_plots_only = bool(\n            self.config.get("redraw_completed_cv_plots_only", False)\n        )\n\n        # STAGE6_REDRAW_ONLY_LARGE_BOX_V1\n        # 严格只重绘会自动打开resume和redraw，但绝不允许进入训练分支。\n        if redraw_completed_cv_plots_only:\n            resume_pretrain_cv = True\n            redraw_completed_cv_plots = True\n\n        if resume_pretrain_cv:\n            print("\\n♻️ 已启用预训练CV按折续跑")\n            print(f"   续跑实验目录: {self.save_dir}")\n            if redraw_completed_cv_plots:\n                print("   🎨 已完成折将加载 best checkpoint 重新生成散点图（不训练）")\n            if redraw_completed_cv_plots_only:\n                print("   🔒 严格只重绘模式：任何不完整fold都会立即终止，绝不训练")\n''',
        "加入严格只重绘状态",
    )

    text = replace_once(
        text,
        '''            preds_denorm = preds * (s_max - s_min) + s_min\n            targets_denorm = targets * (s_max - s_min) + s_min\n\n            plot_metrics = self.plot_density_scatter_hardcode(\n''',
        '''            preds_denorm = preds * (s_max - s_min) + s_min\n            targets_denorm = targets * (s_max - s_min) + s_min\n\n            # STAGE6_REDRAW_SAVE_PREDICTIONS_V1\n            # 本次重推理后保存逐样本CSV；以后可完全脱离模型反复重画。\n            prediction_csv = (\n                self.save_dir\n                / f"pretrain_cv_fold{fold_idx}_predictions.csv"\n            )\n            pd.DataFrame({\n                "产品值_mm": np.asarray(targets_denorm).reshape(-1),\n                "预测值_mm": np.asarray(preds_denorm).reshape(-1),\n            }).to_csv(\n                prediction_csv,\n                index=False,\n                encoding="utf-8-sig",\n            )\n            print(f"   💾 Fold {fold_idx}逐样本预测已保存: {prediction_csv}")\n\n            plot_metrics = self.plot_density_scatter_hardcode(\n''',
        "重绘时保存逐样本CSV",
    )

    text = replace_once(
        text,
        '''            fold_is_complete = fold_complete_marker.exists() or legacy_fold_complete\n\n            if resume_pretrain_cv and fold_is_complete:\n''',
        '''            fold_is_complete = fold_complete_marker.exists() or legacy_fold_complete\n\n            # STAGE6_REDRAW_ONLY_NO_TRAIN_GUARD_V1\n            if redraw_completed_cv_plots_only and not fold_is_complete:\n                raise RuntimeError(\n                    f"Fold {fold_idx}未被识别为完整结果，严格只重绘模式拒绝进入训练。\\n"\n                    f"需要完成标记 {fold_complete_marker.name}，或旧版三件套："\n                    f"{fold_best_path.name} + {fold_curve_path.name} + "\n                    f"{fold_scatter_path.name}"\n                )\n\n            if resume_pretrain_cv and fold_is_complete:\n''',
        "禁止只重绘模式进入训练",
    )

    text = replace_once(
        text,
        '''                except Exception as e:\n                    print(f"   ⚠ Fold {fold_idx} 恢复信息读取失败，将重新训练该折: {e}")\n''',
        '''                except Exception as e:\n                    if redraw_completed_cv_plots_only:\n                        raise RuntimeError(\n                            f"Fold {fold_idx}只重绘失败，严格模式禁止回退训练: {e}"\n                        ) from e\n                    print(f"   ⚠ Fold {fold_idx} 恢复信息读取失败，将重新训练该折: {e}")\n''',
        "禁止只重绘失败后回退训练",
    )

    text = replace_once(
        text,
        '''            # ============ 创建图形 ============\n            fig, ax = plt.subplots(figsize=(10, 8))\n''',
        '''            # ============ 创建图形 ============\n            fig_width = float(\n                self.config.get("density_scatter_fig_width", 12.0)\n            )\n            fig_height = float(\n                self.config.get("density_scatter_fig_height", 9.0)\n            )\n            fig, ax = plt.subplots(figsize=(fig_width, fig_height))\n''',
        "放大画布",
    )

    text = replace_once(
        text,
        '''            ax.set_xlabel(\n                x_axis_label,\n                fontsize=16,\n                fontweight="bold"\n            )\n            ax.set_ylabel(\n                "预测值 (mm)",\n                fontsize=16,\n                fontweight="bold"\n            )\n''',
        '''            ax.set_xlabel(\n                x_axis_label,\n                fontsize=20,\n                fontweight="bold"\n            )\n            ax.set_ylabel(\n                "预测值 (mm)",\n                fontsize=20,\n                fontweight="bold"\n            )\n''',
        "放大坐标轴标题",
    )

    text = replace_once(
        text,
        '''            ax.set_title(title, fontsize=18, fontweight="bold")\n            ax.grid(True, alpha=0.3)\n            ax.legend(fontsize=12, loc="lower right")\n''',
        '''            ax.set_title(title, fontsize=24, fontweight="bold")\n            ax.grid(True, alpha=0.3)\n            ax.legend(fontsize=15, loc="lower right")\n            ax.tick_params(axis="both", labelsize=15)\n''',
        "放大标题图例刻度",
    )

    text = replace_once(
        text,
        '''                cbar = plt.colorbar(scatter, ax=ax, pad=0.01)\n                cbar.set_label("点密度", fontsize=14, fontweight="bold")\n''',
        '''                cbar = plt.colorbar(scatter, ax=ax, pad=0.01)\n                cbar.set_label("点密度", fontsize=17, fontweight="bold")\n                cbar.ax.tick_params(labelsize=13)\n''',
        "放大色标",
    )

    text = replace_once(
        text,
        '''            bbox_props = dict(\n                boxstyle="round,pad=0.65",\n                # 背景透明，文字本身仍完全不透明\n                facecolor=(1.0, 1.0, 1.0, 0.45),\n                edgecolor=(0.0, 0.0, 0.0, 0.75),\n                linewidth=1.5,\n            )\n\n            ax.text(\n                0.05,\n                0.95,\n                metrics_text,\n                transform=ax.transAxes,\n                fontsize=16,\n                fontweight="bold",\n                verticalalignment="top",\n                horizontalalignment="left",\n                bbox=bbox_props,\n            )\n''',
        '''            metrics_fontsize = float(\n                self.config.get("density_scatter_metrics_fontsize", 22)\n            )\n            metrics_pad = float(\n                self.config.get("density_scatter_metrics_pad", 0.90)\n            )\n            metrics_linespacing = float(\n                self.config.get("density_scatter_metrics_linespacing", 1.25)\n            )\n\n            bbox_props = dict(\n                boxstyle=f"round,pad={metrics_pad}",\n                facecolor=(1.0, 1.0, 1.0, 0.58),\n                edgecolor=(0.0, 0.0, 0.0, 0.88),\n                linewidth=2.2,\n            )\n\n            ax.text(\n                0.05,\n                0.95,\n                metrics_text,\n                transform=ax.transAxes,\n                fontsize=metrics_fontsize,\n                fontweight="bold",\n                linespacing=metrics_linespacing,\n                verticalalignment="top",\n                horizontalalignment="left",\n                bbox=bbox_props,\n            )\n''',
        "放大指标框",
    )

    text = replace_once(
        text,
        '''    parser.add_argument(\n        "--redraw_completed_cv_plots",\n        action="store_true",\n        help=(\n            "续跑时对已完成折加载 best checkpoint 重新推理并覆盖散点图；"\n            "只重绘、不训练，用于修复旧图错误的 SWE 反归一化尺度。"\n        ),\n    )\n''',
        '''    parser.add_argument(\n        "--redraw_completed_cv_plots",\n        action="store_true",\n        help=(\n            "续跑时对已完成折加载 best checkpoint 重新推理并覆盖散点图；"\n            "只重绘、不训练，用于修复旧图错误的 SWE 反归一化尺度。"\n        ),\n    )\n    parser.add_argument(\n        "--redraw_completed_cv_plots_only",\n        action="store_true",\n        help=(\n            "严格只重绘：自动启用resume和redraw；"\n            "任一fold不完整时立即终止，绝不进入训练。"\n        ),\n    )\n''',
        "新增命令行参数",
    )

    text = replace_once(
        text,
        '''        "resume_pretrain_cv": args.resume_pretrain_cv,\n        "redraw_completed_cv_plots": args.redraw_completed_cv_plots,\n''',
        '''        "resume_pretrain_cv": args.resume_pretrain_cv,\n        "redraw_completed_cv_plots": args.redraw_completed_cv_plots,\n        "redraw_completed_cv_plots_only": (\n            args.redraw_completed_cv_plots_only\n        ),\n''',
        "传递命令行配置",
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT / "code_backups" / f"before_stage6_redraw_patch_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup)

    TARGET.write_text(text, encoding="utf-8")
    py_compile.compile(str(TARGET), doraise=True)

    print("✅ Stage 6 只重绘与大指标框补丁安装完成")
    print(f"   backup: {backup}")
    print(f"   target: {TARGET}")
    print("   指标框字号: 22")
    print("   新参数: --redraw_completed_cv_plots_only")
    print("   安全保证: 不完整fold直接停止，绝不训练")


if __name__ == "__main__":
    main()
