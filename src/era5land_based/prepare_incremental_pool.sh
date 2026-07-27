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
MANIFEST_PATH="${MANIFEST_PATH:-${SHARED_CACHE_DIR}/incremental_random_pool_152000.csv}"
RATIO_CONFIG="${RATIO_CONFIG:-${SRC_DIR}/incremental_swe_ratios.json}"
GLACIER_MASK_PATH="${GLACIER_MASK_PATH:-}"
SEED="${SEED:-43}"

for f in "${STATION_GUIDE_FILE}" "${STAGE0_MANIFEST}" "${RATIO_CONFIG}"; do
    if [ ! -f "${f}" ]; then
        echo "❌ 文件不存在: ${f}"
        exit 1
    fi
done
if ! compgen -G "${EXTERNAL_STATION_GLOB}" > /dev/null; then
    echo "❌ 没有匹配到外部测试CSV: ${EXTERNAL_STATION_GLOB}"
    exit 1
fi

mkdir -p "${SAVE_DIR}" "${SHARED_CACHE_DIR}"
cd "${SRC_DIR}"

ARGS=(
    --mode build_incremental_manifest
    --model_type full
    --pretrain_years 2015 2016 2017 2018
    --pretrain_samples_per_day 0
    --patch_size 5
    --min_valid_pixels 100
    --clamday_threshold 0.5
    --shared_cache_dir "${SHARED_CACHE_DIR}"
    --disable_dataset_cache
    --force_reload

    --sampling_mode incremental
    --incremental_manifest_path "${MANIFEST_PATH}"
    --incremental_stage 1
    --incremental_pool_size 152000
    --incremental_stage_sizes 12000 20000 40000 80000
    --incremental_seed "${SEED}"
    --incremental_candidate_oversample_factor "${CANDIDATE_OVERSAMPLE_FACTOR:-30.0}"
    --incremental_ratio_config "${RATIO_CONFIG}"
    --incremental_fold_block_pixels "${FOLD_BLOCK_PIXELS:-0}"

    --station_guide_file "${STATION_GUIDE_FILE}"
    --station_neighborhood 0

    --external_station_glob "${EXTERNAL_STATION_GLOB}"
    --external_station_exclusion_radius "${EXTERNAL_STATION_EXCLUSION_RADIUS}"
    --external_station_strict
    --external_station_report_path "${EXTERNAL_REPORT}"

    --seasonal_min_peak_swe_mm "${SEASONAL_MIN_PEAK_SWE_MM:-1}"
    --seasonal_max_swe_mm "${SEASONAL_MAX_SWE_MM:-400}"
    --seasonal_snow_free_threshold_mm "${SNOW_FREE_THRESHOLD_MM:-1}"
    --seasonal_min_warm_snow_free_ratio 0.0
    --seasonal_min_consecutive_snow_free_days "${MIN_CONSECUTIVE_SNOW_FREE_DAYS:-5}"
    --seasonal_min_snow_year_coverage_ratio "${MIN_SNOW_YEAR_COVERAGE_RATIO:-0.90}"

    --normalization_mode skip
    --fixed_label_min_mm 0
    --fixed_label_max_mm 400

    --batch_size 128
    --num_workers 4
    --seed "${SEED}"
    --save_dir "${SAVE_DIR}"
    --exp_name "prepare_incremental_pool_$(date +'%Y%m%d_%H%M%S')"
)

if [ -n "${GLACIER_MASK_PATH}" ]; then
    ARGS+=(--incremental_glacier_mask_path "${GLACIER_MASK_PATH}")
fi

# 已完成的152000行清单直接复用，避免修复 main_tune.py 后重新扫描全国候选。
MANIFEST_ROWS=0
if [ -f "${MANIFEST_PATH}" ]; then
    MANIFEST_ROWS=$(python - "${MANIFEST_PATH}" <<'PYCOUNT'
import csv
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        n = sum(1 for _ in csv.reader(f)) - 1
    print(max(n, 0))
except Exception:
    print(0)
PYCOUNT
)
fi

if [ "${MANIFEST_ROWS}" -eq 152000 ]; then
    echo "✅ 检测到完整固定池: ${MANIFEST_PATH} (${MANIFEST_ROWS} 行)，本次直接复用，不重建。"
else
    echo "ℹ️ 固定池不存在或不完整: ${MANIFEST_PATH} (${MANIFEST_ROWS} 行)，本次重新构建。"
    ARGS+=(--build_incremental_manifest)
fi

echo "============================================================"
echo "一次性固定152000随机样本（不训练；不足分箱由Python定向穷举补采）"
echo "输出: ${MANIFEST_PATH}"
echo "外部CSV排除: ${EXTERNAL_STATION_GLOB}，半径=${EXTERNAL_STATION_EXCLUSION_RADIUS}格"
echo "初始候选过采倍数: ${CANDIDATE_OVERSAMPLE_FACTOR:-30.0}（不再自动重复整轮运行）"
echo "============================================================"
python main_tune.py "${ARGS[@]}"

python verify_incremental_manifest.py \
    --manifest "${MANIFEST_PATH}" \
    --min-annual-max-exclusive "${SEASONAL_MIN_PEAK_SWE_MM:-1}" \
    --max-annual-max-exclusive "${SEASONAL_MAX_SWE_MM:-400}" \
    --min-post-peak-free-days "${MIN_CONSECUTIVE_SNOW_FREE_DAYS:-5}"

python verify_external_station_exclusion.py \
    --manifest "${MANIFEST_PATH}" \
    --report "${EXTERNAL_REPORT}" \
    --name "固定152000随机池"
