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

SRC_DIR=/root/autodl-tmp
SAVE_DIR=/root/autodl-tmp/experiments
SHARED_CACHE_DIR=/root/autodl-tmp/shared_cache

CANDIDATE_MANIFEST="${SHARED_CACHE_DIR}/stage5_candidate_pool_320000_seed1043.csv"

STATION_GUIDE_FILE=/root/ablation/station_swe_data.xlsx
EXTERNAL_STATION_GLOB='/root/ablation/external_test/*.csv'
EXTERNAL_REPORT="${SHARED_CACHE_DIR}/external_station_exclusion_report.csv"
RATIO_CONFIG="${SRC_DIR}/incremental_swe_ratios.json"

cd "${SRC_DIR}"

python main_tune.py \
    --mode build_incremental_manifest \
    --model_type full \
    --pretrain_years 2015 2016 2017 2018 \
    --pretrain_samples_per_day 0 \
    --patch_size 5 \
    --min_valid_pixels 100 \
    --clamday_threshold 0.5 \
    --shared_cache_dir "${SHARED_CACHE_DIR}" \
    --disable_dataset_cache \
    --force_reload \
    --sampling_mode incremental \
    --incremental_manifest_path "${CANDIDATE_MANIFEST}" \
    --incremental_stage 1 \
    --build_incremental_manifest \
    --incremental_pool_size 320000 \
    --incremental_stage_sizes 320000 \
    --incremental_seed 1043 \
    --incremental_candidate_oversample_factor 3.0 \
    --incremental_ratio_config "${RATIO_CONFIG}" \
    --incremental_fold_block_pixels 0 \
    --station_guide_file "${STATION_GUIDE_FILE}" \
    --station_neighborhood 0 \
    --external_station_glob "${EXTERNAL_STATION_GLOB}" \
    --external_station_exclusion_radius 0 \
    --external_station_strict \
    --external_station_report_path "${EXTERNAL_REPORT}" \
    --seasonal_min_peak_swe_mm 1 \
    --seasonal_max_swe_mm 400 \
    --seasonal_snow_free_threshold_mm 1 \
    --seasonal_min_warm_snow_free_ratio 0.0 \
    --seasonal_min_consecutive_snow_free_days 5 \
    --seasonal_min_snow_year_coverage_ratio 0.90 \
    --normalization_mode skip \
    --fixed_label_min_mm 0 \
    --fixed_label_max_mm 400 \
    --batch_size 128 \
    --num_workers 4 \
    --seed 1043 \
    --save_dir "${SAVE_DIR}" \
    --exp_name "prepare_stage5_candidate_$(date +'%Y%m%d_%H%M%S')"

python verify_incremental_manifest.py \
    --manifest "${CANDIDATE_MANIFEST}" \
    --min-annual-max-exclusive 1 \
    --max-annual-max-exclusive 400 \
    --min-post-peak-free-days 5

echo "✅ Stage 5候选池：${CANDIDATE_MANIFEST}"
