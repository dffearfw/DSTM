#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import py_compile
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp")
TARGET = ROOT / "main_tune.py"

MARKER_CLEAN = "PROGRESSIVE_CV_CLEAR_SHARED_BEST_V1"
MARKER_RELOAD = "PROGRESSIVE_CV_RELOAD_BEST_V1"

ANCHOR_CLEAN = '            # ============ 8.6 训练 ============\n            print(f"\\n🚀 [FOLD {fold_idx}] 开始训练...")\n\n            original_epochs = self.config.get("epochs", 100)\n'
REPLACEMENT_CLEAN = '            # PROGRESSIVE_CV_CLEAR_SHARED_BEST_V1\n            # train()在所有折中共用self.save_dir；如果不清理，\n            # 本折训练失败时可能误复制上一折的best checkpoint。\n            shared_fold_candidates = [\n                self.save_dir / "best_fine_tuned_model.pth",\n                self.save_dir / "final_fine_tuned_model.pth",\n                self.save_dir / "best_model.pth",\n                self.save_dir / "final_model.pth",\n            ]\n            for stale_path in shared_fold_candidates:\n                if stale_path.exists():\n                    stale_path.unlink()\n                    print(\n                        f"  🧹 [FOLD {fold_idx}] 清除上一折共享checkpoint: "\n                        f"{stale_path.name}"\n                    )\n\n            # ============ 8.6 训练 ============\n            print(f"\\n🚀 [FOLD {fold_idx}] 开始训练...")\n\n            original_epochs = self.config.get("epochs", 100)\n'
ANCHOR_RELOAD = '            if copied:\n                fold_model_paths[fold_idx] = str(fold_model_path)\n\n            # ============ 8.8 验证集预测 ============\n            print(f"\\n📊 [FOLD {fold_idx}] 收集验证集预测...")\n\n            self.model.eval()\n'
REPLACEMENT_RELOAD = '            if copied:\n                fold_model_paths[fold_idx] = str(fold_model_path)\n\n                # PROGRESSIVE_CV_RELOAD_BEST_V1\n                # 指标必须由该折“最佳checkpoint”计算，而不是最后epoch模型。\n                try:\n                    best_checkpoint = torch.load(\n                        fold_model_path,\n                        map_location=self.device,\n                        weights_only=False,\n                    )\n\n                    if isinstance(best_checkpoint, dict):\n                        best_state_dict = None\n                        for state_key in (\n                            "model_state_dict",\n                            "state_dict",\n                            "model",\n                        ):\n                            if state_key in best_checkpoint:\n                                best_state_dict = best_checkpoint[state_key]\n                                break\n                    else:\n                        best_state_dict = best_checkpoint\n\n                    if best_state_dict is None:\n                        raise RuntimeError(\n                            f"最佳checkpoint中没有模型权重: {fold_model_path}"\n                        )\n\n                    self.model.load_state_dict(\n                        best_state_dict,\n                        strict=True,\n                    )\n                    self.model.to(self.device)\n                    print(\n                        f"  ✅ [FOLD {fold_idx}] 已重新载入该折最佳模型，"\n                        "后续验证指标基于best checkpoint"\n                    )\n                except Exception as exc:\n                    raise RuntimeError(\n                        f"Fold {fold_idx} 最佳模型重新载入失败: {exc}"\n                    ) from exc\n\n            # ============ 8.8 验证集预测 ============\n            print(f"\\n📊 [FOLD {fold_idx}] 收集验证集预测...")\n\n            self.model.eval()\n'


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    original = TARGET.read_text(encoding="utf-8")
    updated = original
    changes = []

    if MARKER_CLEAN not in updated:
        count = updated.count(ANCHOR_CLEAN)
        if count != 1:
            raise RuntimeError(
                "无法唯一定位十折训练起点；"
                f"匹配数={count}。请确认main_tune.py版本。"
            )
        updated = updated.replace(
            ANCHOR_CLEAN,
            REPLACEMENT_CLEAN,
            1,
        )
        changes.append("clear_shared_best_before_each_fold")

    if MARKER_RELOAD not in updated:
        count = updated.count(ANCHOR_RELOAD)
        if count != 1:
            raise RuntimeError(
                "无法唯一定位最佳模型复制与验证预测代码块；"
                f"匹配数={count}。请确认main_tune.py版本。"
            )
        updated = updated.replace(
            ANCHOR_RELOAD,
            REPLACEMENT_RELOAD,
            1,
        )
        changes.append("reload_fold_best_before_validation")

    if not changes:
        py_compile.compile(str(TARGET), doraise=True)
        print("✅ main_tune.py 已包含最佳checkpoint评估修复")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        ROOT
        / "code_backups"
        / f"before_cv_best_checkpoint_fix_{stamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_path = backup_dir / TARGET.name
    shutil.copy2(TARGET, backup_path)

    TARGET.write_text(updated, encoding="utf-8")
    py_compile.compile(str(TARGET), doraise=True)

    print("✅ main_tune.py 修复完成")
    print(f"   changes={changes}")
    print(f"   backup={backup_path}")
    print(f"   target={TARGET}")


if __name__ == "__main__":
    main()
