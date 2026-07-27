#!/bin/bash
set -Eeuo pipefail
set -o pipefail

export PYTHONUNBUFFERED=1
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

SRC_DIR="${SRC_DIR:-/root/autodl-tmp}"
SAVE_DIR="${SAVE_DIR:-/root/autodl-tmp/experiments}"
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-/root/autodl-tmp/shared_cache}"

PREVIOUS_MODEL="${PREVIOUS_MODEL:-}"

STATION_GUIDE_FILE="${STATION_GUIDE_FILE:-/root/ablation/station_swe_data.xlsx}"
EXTERNAL_STATION_GLOB="${EXTERNAL_STATION_GLOB:-/root/ablation/external_test/*.csv}"
EXTERNAL_STATION_EXCLUSION_RADIUS="${EXTERNAL_STATION_EXCLUSION_RADIUS:-0}"
EXTERNAL_REPORT="${EXTERNAL_REPORT:-${SHARED_CACHE_DIR}/external_station_exclusion_report.csv}"

STAGE0_MANIFEST="${STAGE0_MANIFEST:-${SHARED_CACHE_DIR}/stage0_station_record_manifest.csv}"

# Stage 5正式训练使用的Stage 1–5总清单
MANIFEST_PATH="${MANIFEST_PATH:-${SHARED_CACHE_DIR}/incremental_random_pool_312000_stage1_5.csv}"

# 原归一化配置建立于旧152k清单，验证时仍使用旧清单作为参照
NORM_REFERENCE_MANIFEST="${NORM_REFERENCE_MANIFEST:-${SHARED_CACHE_DIR}/incremental_random_pool_152000.csv}"
NORMALIZATION_CONFIG="${NORMALIZATION_CONFIG:-${SHARED_CACHE_DIR}/progressive_pretrain_normalization.json}"

RATIO_CONFIG="${RATIO_CONFIG:-${SRC_DIR}/incremental_swe_ratios.json}"

if [ -z "${PREVIOUS_MODEL}" ] || [ ! -f "${PREVIOUS_MODEL}" ]; then
    echo "❌ PREVIOUS_MODEL必须指向Stage 4的final_model.pth"
    echo "当前值: ${PREVIOUS_MODEL:-空}"
    exit 1
fi

for f in \
    "${STATION_GUIDE_FILE}" \
    "${STAGE0_MANIFEST}" \
    "${MANIFEST_PATH}" \
    "${NORM_REFERENCE_MANIFEST}" \
    "${NORMALIZATION_CONFIG}" \
    "${EXTERNAL_REPORT}" \
    "${RATIO_CONFIG}"
do
    if [ ! -f "${f}" ]; then
        echo "❌ 文件不存在: ${f}"
        exit 1
    fi
done

if ! compgen -G "${EXTERNAL_STATION_GLOB}" > /dev/null; then
    echo "❌ 没有匹配到外部测试CSV: ${EXTERNAL_STATION_GLOB}"
    exit 1
fi

cd "${SRC_DIR}"

echo "============================================================"
echo "检查Stage 1–5清单"
echo "============================================================"

python - "${MANIFEST_PATH}" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)

required = {"sample_id", "stage_id", "swe_bin"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"清单缺少字段: {sorted(missing)}")

expected = {
    1: 12000,
    2: 20000,
    3: 40000,
    4: 80000,
    5: 160000,
}

counts = (
    df["stage_id"]
    .astype(int)
    .value_counts()
    .sort_index()
    .to_dict()
)

duplicates = int(df["sample_id"].duplicated().sum())

print("总样本数:", len(df))
print("阶段分布:", counts)
print("sample_id重复数:", duplicates)

if len(df) != 312000:
    raise RuntimeError(f"总样本数错误: {len(df)} != 312000")

if counts != expected:
    raise RuntimeError(f"阶段分布错误: {counts} != {expected}")

if duplicates != 0:
    raise RuntimeError(f"存在重复sample_id: {duplicates}")

print("✅ Stage 1–5清单检查通过")
PY

echo
echo "============================================================"
echo "验证原有统一归一化配置"
echo "============================================================"

python verify_progressive_normalization.py \
    --config "${NORMALIZATION_CONFIG}" \
    --stage0-manifest "${STAGE0_MANIFEST}" \
    --incremental-manifest "${NORM_REFERENCE_MANIFEST}" \
    --label-min 0 \
    --label-max 400

echo
echo "============================================================"
echo "验证Stage 1–5清单的外部测试站点排除"
echo "============================================================"

python verify_external_station_exclusion.py \
    --manifest "${MANIFEST_PATH}" \
    --report "${EXTERNAL_REPORT}" \
    --name "固定312000 Stage 1-5随机池"

RUN_NAME="pretrain_stage5_incremental_$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="${SAVE_DIR}/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/run.log"

mkdir -p "${RUN_DIR}" "${SHARED_CACHE_DIR}"

ARGS=(
    --mode pretrain_progressive
    --model_type full
    --pretrain_years 2015 2016 2017 2018

    --epochs 100
    --batch_size 128
    --lr "${LR:-1e-4}"
    --d_model 256
    --num_workers "${NUM_WORKERS:-4}"
    --seed "${SEED:-43}"

    --val_every 5
    --disable_pretrain_cv_early_stopping

    --pretrain_samples_per_day 0
    --patch_size 5
    --min_valid_pixels 100
    --clamday_threshold 0.5

    --shared_cache_dir "${SHARED_CACHE_DIR}"
    --disable_dataset_cache

    --sampling_mode incremental
    --incremental_manifest_path "${MANIFEST_PATH}"
    --incremental_stage 5
    --incremental_pool_size 312000
    --incremental_stage_sizes 12000 20000 40000 80000 160000
    --incremental_seed "${SEED:-43}"
    --incremental_candidate_oversample_factor "${CANDIDATE_OVERSAMPLE_FACTOR:-3.0}"
    --incremental_fold_block_pixels "${FOLD_BLOCK_PIXELS:-0}"
    --incremental_ratio_config "${RATIO_CONFIG}"

    --station_guide_file "${STATION_GUIDE_FILE}"
    --station_neighborhood 0

    --external_station_glob "${EXTERNAL_STATION_GLOB}"
    --external_station_exclusion_radius "${EXTERNAL_STATION_EXCLUSION_RADIUS}"
    --external_station_strict
    --external_station_report_path "${EXTERNAL_REPORT}"

    --seasonal_min_peak_swe_mm 1
    --seasonal_max_swe_mm 400
    --seasonal_snow_free_threshold_mm 1
    --seasonal_min_warm_snow_free_ratio 0.0
    --seasonal_min_consecutive_snow_free_days 5
    --seasonal_min_snow_year_coverage_ratio 0.90

    --normalization_config_path "${NORMALIZATION_CONFIG}"
    --normalization_mode load
    --fixed_label_min_mm 0
    --fixed_label_max_mm 400

    --pretrained_model "${PREVIOUS_MODEL}"
    --use_amp

    --save_dir "${SAVE_DIR}"
    --exp_name "${RUN_NAME}"

    --final_train_ratio 1.0
    --final_epochs_mode fixed
    --final_epochs 100
    --final_scheduler cosine
)

echo "============================================================" | tee "${LOG_FILE}"
echo "Stage 5增量预训练" | tee -a "${LOG_FILE}"
echo "上一阶段模型: ${PREVIOUS_MODEL}" | tee -a "${LOG_FILE}"
echo "总清单: ${MANIFEST_PATH}" | tee -a "${LOG_FILE}"
echo "当前训练: stage_id=5，共160000条" | tee -a "${LOG_FILE}"
echo "初始学习率: ${LR:-1e-4}" | tee -a "${LOG_FILE}"
echo "统一归一化: ${NORMALIZATION_CONFIG}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

python main_tune.py "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
