#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 SWE 预训练流程改为：
1) incremental_stage=N 读取 manifest 中 stage_id <= N 的累计样本；
2) --from_scratch 时，CV 每折和最终 100% refit 都不加载任何旧模型；
3) 生成独立启动脚本 pretrain_cumulative_scratch.sh。

目标目录固定为 /root/autodl-tmp。
"""
from __future__ import annotations

import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
DATA_FILE = ROOT / 'data_online_era5_swe.py'
MAIN_FILE = ROOT / 'main_tune.py'
RUN_SCRIPT = ROOT / 'pretrain_cumulative_scratch.sh'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            print(f'  ✅ {label}: 已经修改过')
            return text
        raise RuntimeError(f'{label}: 未找到预期代码块，拒绝盲改')
    if count != 1:
        raise RuntimeError(f'{label}: 预期唯一匹配，实际匹配 {count} 次')
    print(f'  ✅ {label}')
    return text.replace(old, new, 1)


def patch_data_file() -> None:
    text = DATA_FILE.read_text(encoding='utf-8')

    text = replace_once(
        text,
        '''        stage_df = df[df["stage_id"].astype(int) == self.incremental_stage].copy()\n        expected = self.incremental_stage_sizes[self.incremental_stage - 1]\n''',
        '''        # [CUMULATIVE-SCRATCH] Stage N 使用 stage_id <= N 的嵌套累计池。\n        stage_df = df[df["stage_id"].astype(int) <= self.incremental_stage].copy()\n        expected = sum(self.incremental_stage_sizes[:self.incremental_stage])\n''',
        'Dataset改为累计池筛选',
    )

    # 改变缓存key，防止误读旧的“单阶段新增包”pickle缓存。
    old_cache = "                'incremental_stage': self.incremental_stage,\n"
    new_cache = (
        "                'incremental_stage': self.incremental_stage,\n"
        "                'incremental_selection_mode': 'cumulative_v1',\n"
    )
    if "'incremental_selection_mode': 'cumulative_v1'" not in text:
        if old_cache not in text:
            raise RuntimeError('Dataset缓存key位置未找到')
        text = text.replace(old_cache, new_cache, 1)
        print('  ✅ Dataset缓存key加入 cumulative_v1')
    else:
        print('  ✅ Dataset缓存key已包含 cumulative_v1')

    text = text.replace(
        'f"incremental_stage_{self.incremental_stage}"',
        'f"incremental_cumulative_stage_{self.incremental_stage}"',
    )
    text = text.replace(
        'print(f"\\n📦 已加载固定增量 Stage {self.incremental_stage}")',
        'print(f"\\n📦 已加载累计样本池 Stage 1-{self.incremental_stage}")',
    )
    text = text.replace(
        'f"增量清单中 Stage {self.incremental_stage} 没有可用样本"',
        'f"累计清单 Stage 1-{self.incremental_stage} 没有可用样本"',
    )
    text = text.replace(
        '"\\n📦 incremental 模式：直接使用清单中的当前新增包；"\n'
        '                "不重新随机、不累计旧阶段样本、不执行 quota/adaptive。"',
        '"\\n📦 cumulative incremental 模式：使用 stage_id <= 当前Stage 的累计池；"\n'
        '                "不重新随机、不执行 quota/adaptive。"',
    )

    DATA_FILE.write_text(text, encoding='utf-8')
    py_compile.compile(str(DATA_FILE), doraise=True)


def patch_main_file() -> None:
    text = MAIN_FILE.read_text(encoding='utf-8')

    old_init = '''        pretrain_init_model = self.config.get("pretrained_model")\n        if resolved_mode == "incremental":\n            if not pretrain_init_model or not os.path.exists(pretrain_init_model):\n                raise FileNotFoundError(\n                    "incremental Stage 1-4 必须通过 --pretrained_model 指定上一阶段 "\n                    "final_model.pth"\n                )\n            print(f"   ✅ 每一折均从同一个上一阶段模型初始化: {pretrain_init_model}")\n        elif pretrain_init_model:\n            print(f"   ℹ 预训练CV将加载初始化权重: {pretrain_init_model}")\n'''

    new_init = '''        # [CUMULATIVE-SCRATCH] --from_scratch 时每折独立随机初始化。\n        is_from_scratch = bool(self.config.get("from_scratch", False))\n        pretrain_init_model = (\n            None if is_from_scratch else self.config.get("pretrained_model")\n        )\n\n        if resolved_mode == "incremental":\n            if is_from_scratch:\n                print("   ✅ 累计池从头训练：每一折均随机初始化，不加载Stage 0或上一阶段权重")\n            else:\n                if not pretrain_init_model or not os.path.exists(pretrain_init_model):\n                    raise FileNotFoundError(\n                        "非from_scratch的incremental训练必须通过 --pretrained_model "\n                        "指定初始化模型"\n                    )\n                print(f"   ✅ 每一折均从同一个模型初始化: {pretrain_init_model}")\n        elif pretrain_init_model:\n            print(f"   ℹ 预训练CV将加载初始化权重: {pretrain_init_model}")\n'''

    text = replace_once(text, old_init, new_init, 'main_tune放行incremental从头CV')

    old_final = '''        # Stage 0从头训练；Stage 1-4重新加载上一阶段正式模型，\n        # 再使用当前新增包的100%样本训练。绝不从某一折checkpoint继续。\n        pretrain_init_model = self.config.get("pretrained_model")\n        print(f"\\n🏗️ 构建全量训练模型...")\n        if pretrain_init_model:\n            print(f"   初始化权重: {pretrain_init_model}")\n        else:\n            print("   初始化权重: 随机初始化（Stage 0）")\n'''

    new_final = '''        # [CUMULATIVE-SCRATCH] 最终100% refit也必须重新随机初始化，\n        # 不从任一CV折、Stage 0或上一累计规模继续。\n        is_from_scratch = bool(self.config.get("from_scratch", False))\n        pretrain_init_model = (\n            None if is_from_scratch else self.config.get("pretrained_model")\n        )\n        print(f"\\n🏗️ 构建全量训练模型...")\n        if pretrain_init_model:\n            print(f"   初始化权重: {pretrain_init_model}")\n        else:\n            print("   初始化权重: 随机初始化（累计池 scratch）")\n'''

    text = replace_once(text, old_final, new_final, '最终100% refit改为随机初始化')

    text = text.replace(
        'print(f"   当前只训练 Stage {self.config.get(\'incremental_stage\', 1)} 新增包")',
        'print(f"   当前累计读取 Stage 1-{self.config.get(\'incremental_stage\', 1)}")',
    )
    text = text.replace(
        'print("   Step 3: 用当前阶段100%新增样本训练正式模型（复用数据集）")',
        'print("   Step 3: 用当前累计池100%样本从头训练正式模型（复用数据集）")',
    )

    # 顺手修复日志表头写“最佳r”却打印r2的问题，不影响训练。
    text = text.replace(
        "{m['best_val_r2']:<10.4f}",
        "{m['best_val_r']:<10.4f}",
        1,
    )

    MAIN_FILE.write_text(text, encoding='utf-8')
    py_compile.compile(str(MAIN_FILE), doraise=True)


def write_run_script() -> None:
    script = r'''#!/bin/bash
set -Eeuo pipefail
set -o pipefail

export PYTHONUNBUFFERED=1

# ============================================================
# 独立随机初始化 + 累计样本池
#
# STAGE=1 -> 12k  = Stage 1
# STAGE=2 -> 32k  = Stage 1+2
# STAGE=3 -> 72k  = Stage 1+2+3
# STAGE=4 -> 152k = Stage 1+2+3+4
#
# 例：
#   STAGE=1 MAX_FOLDS=1 bash pretrain_cumulative_scratch.sh  # 先测Fold1
#   STAGE=1 bash pretrain_cumulative_scratch.sh              # 正式10折+全量refit
# ============================================================

STAGE="${STAGE:-1}"
MAX_FOLDS="${MAX_FOLDS:-10}"
MODE="${MODE:-pretrain_progressive}"

if [[ ! "${STAGE}" =~ ^[1-4]$ ]]; then
    echo "❌ STAGE必须是1、2、3或4，当前=${STAGE}"
    exit 1
fi

case "${STAGE}" in
    1) CUM_SAMPLES=12000;  CUM_TAG="12k" ;;
    2) CUM_SAMPLES=32000;  CUM_TAG="32k" ;;
    3) CUM_SAMPLES=72000;  CUM_TAG="72k" ;;
    4) CUM_SAMPLES=152000; CUM_TAG="152k" ;;
esac

SRC_DIR="/root/autodl-tmp"
SAVE_DIR="/root/autodl-tmp/experiments"
SHARED_CACHE_DIR="/root/autodl-tmp/shared_cache"
MANIFEST_PATH="${MANIFEST_PATH:-${SHARED_CACHE_DIR}/incremental_random_pool_152000.csv}"
NORMALIZATION_CONFIG="${NORMALIZATION_CONFIG:-${SHARED_CACHE_DIR}/progressive_pretrain_normalization.json}"
RATIO_CONFIG="${RATIO_CONFIG:-${SRC_DIR}/incremental_swe_ratios.json}"
STATION_GUIDE_FILE="${STATION_GUIDE_FILE:-${SRC_DIR}/ablation/station_swe_data.xlsx}"
GLACIER_MASK_PATH="${GLACIER_MASK_PATH:-}"

PRETRAIN_YEARS=(2015 2016 2017 2018)
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
D_MODEL="${D_MODEL:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-43}"
VAL_EVERY="${VAL_EVERY:-1}"

export FILTER_GLACIER_SWE_ARTIFACTS=1
export GLACIER_SWE_THRESHOLD_MM=2000
export PRETRAIN_SAMPLES_PER_DAY=0
export USE_TARGET_QUOTA_SAMPLING=0
export USE_QUOTA_SHORTAGE_SUPPLEMENT=0
export STRICT_TARGET_QUOTA=0
export PRECOMPUTE_ALL_SAMPLES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 稳定性监控默认关闭；需要时运行前设 STABILITY_MONITOR=1。
export STABILITY_MONITOR="${STABILITY_MONITOR:-0}"
export STABILITY_MAX_STEPS="${STABILITY_MAX_STEPS:-300}"
export STABILITY_LOG_EVERY="${STABILITY_LOG_EVERY:-1}"
export STABILITY_WINDOW="${STABILITY_WINDOW:-50}"
export STABILITY_EMA_BETA="${STABILITY_EMA_BETA:-0.98}"

cd "${SRC_DIR}"

for path in "${MANIFEST_PATH}" "${NORMALIZATION_CONFIG}"; do
    if [ ! -f "${path}" ]; then
        echo "❌ 必需文件不存在: ${path}"
        exit 1
    fi
done

# 启动前验证累计数量，不重新抽样。
python - "${MANIFEST_PATH}" "${STAGE}" "${CUM_SAMPLES}" <<'PY_CHECK'
import sys
import pandas as pd
path, stage, expected = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
df = pd.read_csv(path)
required = {'stage_id', 'fold_id', 'sample_id'}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f'❌ manifest缺字段: {sorted(missing)}')
sel = df[df['stage_id'].astype(int) <= stage]
if len(sel) != expected:
    raise SystemExit(f'❌ 累计样本数错误: {len(sel)} != {expected}')
if sel['sample_id'].duplicated().any():
    raise SystemExit('❌ 累计池存在重复sample_id')
print(f'✅ 累计池检查通过: Stage 1-{stage}, N={len(sel):,}')
print('   stage分布:', sel.groupby('stage_id').size().to_dict())
print('   fold分布:', sel.groupby('fold_id').size().to_dict())
PY_CHECK

RUN_NAME="scratch_cumulative_${CUM_TAG}_$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="${SAVE_DIR}/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/run.log"
mkdir -p "${RUN_DIR}"

ARGS=(
    --mode "${MODE}"
    --model_type full
    --pretrain_years "${PRETRAIN_YEARS[@]}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --d_model "${D_MODEL}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --val_every "${VAL_EVERY}"
    --disable_pretrain_cv_early_stopping

    --pretrain_samples_per_day 0
    --patch_size 5
    --min_valid_pixels 100
    --clamday_threshold 0.5
    --shared_cache_dir "${SHARED_CACHE_DIR}"
    --disable_dataset_cache

    --sampling_mode incremental
    --incremental_manifest_path "${MANIFEST_PATH}"
    --incremental_stage "${STAGE}"
    --incremental_pool_size 152000
    --incremental_stage_sizes 12000 20000 40000 80000
    --incremental_seed "${SEED}"
    --incremental_fold_block_pixels 0
    --incremental_ratio_config "${RATIO_CONFIG}"

    --station_guide_file "${STATION_GUIDE_FILE}"
    --station_neighborhood 0

    --seasonal_min_peak_swe_mm 1
    --seasonal_max_swe_mm 400
    --seasonal_snow_free_threshold_mm 1
    --seasonal_min_warm_snow_free_ratio 0
    --seasonal_min_consecutive_snow_free_days 5
    --seasonal_min_snow_year_coverage_ratio 0.90

    --normalization_config_path "${NORMALIZATION_CONFIG}"
    --normalization_mode load
    --fixed_label_min_mm 0
    --fixed_label_max_mm 400

    --from_scratch
    --use_amp
    --save_dir "${SAVE_DIR}"
    --exp_name "${RUN_NAME}"

    --final_train_ratio 1.0
    --final_epochs_mode fixed
    --final_epochs "${EPOCHS}"
    --final_scheduler cosine
)

# 兼容你当前较新的main_tune.py：存在参数才加入。
HELP_TEXT="$(python main_tune.py --help 2>&1 || true)"
if grep -q -- '--lr_scheduler' <<<"${HELP_TEXT}"; then
    ARGS+=(--lr_scheduler plateau)
fi
if grep -q -- '--plateau_patience' <<<"${HELP_TEXT}"; then
    ARGS+=(--plateau_patience 8)
fi
if grep -q -- '--plateau_factor' <<<"${HELP_TEXT}"; then
    ARGS+=(--plateau_factor 0.5)
fi
if grep -q -- '--plateau_min_lr' <<<"${HELP_TEXT}"; then
    ARGS+=(--plateau_min_lr 1e-6)
fi
if grep -q -- '--plateau_threshold' <<<"${HELP_TEXT}"; then
    ARGS+=(--plateau_threshold 1e-3)
fi
if grep -q -- '--plateau_cooldown' <<<"${HELP_TEXT}"; then
    ARGS+=(--plateau_cooldown 1)
fi
if grep -q -- '--pretrain_cv_max_folds' <<<"${HELP_TEXT}"; then
    ARGS+=(--pretrain_cv_max_folds "${MAX_FOLDS}")
elif [ "${MAX_FOLDS}" != "10" ]; then
    echo "⚠ 当前main_tune.py不支持 --pretrain_cv_max_folds；MAX_FOLDS=${MAX_FOLDS}不会生效"
fi

if [ -n "${GLACIER_MASK_PATH}" ]; then
    ARGS+=(--incremental_glacier_mask_path "${GLACIER_MASK_PATH}")
fi

export STABILITY_PREFIX="${RUN_NAME}"

echo "============================================================" | tee "${LOG_FILE}"
echo "模式: 每个累计规模独立从头训练" | tee -a "${LOG_FILE}"
echo "累计池: Stage 1-${STAGE}, N=${CUM_SAMPLES}" | tee -a "${LOG_FILE}"
echo "初始化: 随机初始化；不加载Stage0/上一阶段/CV checkpoint" | tee -a "${LOG_FILE}"
echo "CV调度器: ReduceLROnPlateau；无warmup" | tee -a "${LOG_FILE}"
echo "最终100% refit: 随机初始化 + Cosine（因无验证集）" | tee -a "${LOG_FILE}"
echo "运行目录: ${RUN_DIR}" | tee -a "${LOG_FILE}"
printf 'python main_tune.py ' | tee -a "${LOG_FILE}"
printf '%q ' "${ARGS[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

python main_tune.py "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
'''
    RUN_SCRIPT.write_text(script, encoding='utf-8')
    RUN_SCRIPT.chmod(0o755)


def main() -> None:
    for path in (DATA_FILE, MAIN_FILE):
        if not path.exists():
            raise FileNotFoundError(f'找不到正式文件: {path}')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = ROOT / 'code_backups' / f'before_cumulative_scratch_{stamp}'
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(DATA_FILE, backup_dir / DATA_FILE.name)
    shutil.copy2(MAIN_FILE, backup_dir / MAIN_FILE.name)
    if RUN_SCRIPT.exists():
        shutil.copy2(RUN_SCRIPT, backup_dir / RUN_SCRIPT.name)

    print(f'📦 本次唯一备份目录: {backup_dir}')
    patch_data_file()
    patch_main_file()
    write_run_script()

    print('\n✅ 修改完成')
    print(f'   数据集: {DATA_FILE}')
    print(f'   主程序: {MAIN_FILE}')
    print(f'   启动脚本: {RUN_SCRIPT}')
    print('   Python语法检查: 通过')
    print('\n下一步先运行：')
    print('  MODE=pretrain_cv STAGE=1 MAX_FOLDS=1 bash /root/autodl-tmp/pretrain_cumulative_scratch.sh')


if __name__ == '__main__':
    main()
