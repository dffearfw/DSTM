#!/bin/bash
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
STATION_GUIDE_FILE="/root/ablation/station_swe_data.xlsx"
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
