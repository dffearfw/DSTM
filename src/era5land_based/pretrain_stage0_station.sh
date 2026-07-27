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
STATION_GUIDE_FILE="${STATION_GUIDE_FILE:-/root/ablation/station_swe_data.xlsx}"
EXTERNAL_STATION_GLOB="${EXTERNAL_STATION_GLOB:-/root/ablation/external_test/*.csv}"
EXTERNAL_STATION_EXCLUSION_RADIUS="${EXTERNAL_STATION_EXCLUSION_RADIUS:-0}"
EXTERNAL_REPORT="${EXTERNAL_REPORT:-${SHARED_CACHE_DIR}/external_station_exclusion_report.csv}"
STAGE0_MANIFEST="${STAGE0_MANIFEST:-${SHARED_CACHE_DIR}/stage0_station_record_manifest.csv}"
INCREMENTAL_MANIFEST="${INCREMENTAL_MANIFEST:-${SHARED_CACHE_DIR}/incremental_random_pool_152000.csv}"
NORMALIZATION_CONFIG="${NORMALIZATION_CONFIG:-${SHARED_CACHE_DIR}/progressive_pretrain_normalization.json}"

for f in "${STATION_GUIDE_FILE}" "${STAGE0_MANIFEST}" "${INCREMENTAL_MANIFEST}" "${NORMALIZATION_CONFIG}" "${EXTERNAL_REPORT}"; do
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
python verify_progressive_normalization.py \
    --config "${NORMALIZATION_CONFIG}" \
    --stage0-manifest "${STAGE0_MANIFEST}" \
    --incremental-manifest "${INCREMENTAL_MANIFEST}" \
    --label-min 0 --label-max 400
python verify_external_station_exclusion.py \
    --manifest "${STAGE0_MANIFEST}" \
    --report "${EXTERNAL_REPORT}" \
    --name "Stage 0清单"

RUN_NAME="pretrain_stage0_station_$(date +'%Y%m%d_%H%M%S')"
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

    --sampling_mode station
    --station_guide_file "${STATION_GUIDE_FILE}"
    --station_neighborhood 0
    --station_samples_per_day -1
    --station_sampling_unit records
    --station_record_dedup grid_date
    --station_record_manifest_path "${STAGE0_MANIFEST}"
    --station_include_zero_target

    --external_station_glob "${EXTERNAL_STATION_GLOB}"
    --external_station_exclusion_radius "${EXTERNAL_STATION_EXCLUSION_RADIUS}"
    --external_station_strict
    --external_station_report_path "${EXTERNAL_REPORT}"

    --normalization_config_path "${NORMALIZATION_CONFIG}"
    --normalization_mode load
    --fixed_label_min_mm 0
    --fixed_label_max_mm 400

    --use_amp
    --save_dir "${SAVE_DIR}"
    --exp_name "${RUN_NAME}"
    --final_train_ratio 1.0
    --final_epochs_mode fixed
    --final_epochs 100
    --final_scheduler cosine
)

echo "============================================================" | tee "${LOG_FILE}"
echo "Stage 0：Excel实际站点-日期；排除外部CSV±${EXTERNAL_STATION_EXCLUSION_RADIUS}格" | tee -a "${LOG_FILE}"
echo "Excel: ${STATION_GUIDE_FILE}" | tee -a "${LOG_FILE}"
echo "外部CSV: ${EXTERNAL_STATION_GLOB}" | tee -a "${LOG_FILE}"
echo "统一归一化: ${NORMALIZATION_CONFIG}" | tee -a "${LOG_FILE}"
echo "运行目录: ${RUN_DIR}" | tee -a "${LOG_FILE}"
printf 'python main_tune.py ' | tee -a "${LOG_FILE}"
printf '%q ' "${ARGS[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

python main_tune.py "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
