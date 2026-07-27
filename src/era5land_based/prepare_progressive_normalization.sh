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
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-/root/autodl-tmp/shared_cache}"
STATION_GUIDE_FILE="${STATION_GUIDE_FILE:-/root/ablation/station_swe_data.xlsx}"
EXTERNAL_STATION_GLOB="${EXTERNAL_STATION_GLOB:-/root/ablation/external_test/*.csv}"
EXTERNAL_STATION_EXCLUSION_RADIUS="${EXTERNAL_STATION_EXCLUSION_RADIUS:-0}"
EXTERNAL_REPORT="${EXTERNAL_REPORT:-${SHARED_CACHE_DIR}/external_station_exclusion_report.csv}"
STAGE0_MANIFEST="${STAGE0_MANIFEST:-${SHARED_CACHE_DIR}/stage0_station_record_manifest.csv}"
INCREMENTAL_MANIFEST="${INCREMENTAL_MANIFEST:-${SHARED_CACHE_DIR}/incremental_random_pool_152000.csv}"
NORMALIZATION_CONFIG="${NORMALIZATION_CONFIG:-${SHARED_CACHE_DIR}/progressive_pretrain_normalization.json}"

for f in "${STATION_GUIDE_FILE}" "${STAGE0_MANIFEST}" "${INCREMENTAL_MANIFEST}" "${EXTERNAL_REPORT}"; do
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
ARGS=(
    --station-guide-file "${STATION_GUIDE_FILE}"
    --stage0-manifest "${STAGE0_MANIFEST}"
    --incremental-manifest "${INCREMENTAL_MANIFEST}"
    --output "${NORMALIZATION_CONFIG}"
    --shared-cache-dir "${SHARED_CACHE_DIR}"
    --years 2015 2016 2017 2018
    --patch-size 5
    --min-valid-pixels 100
    --clamday-threshold 0.5
    --label-min-mm 0
    --label-max-mm 400
    --quantile-samples "${QUANTILE_SAMPLES:-40000}"
    --seed "${SEED:-43}"
    --external-station-glob "${EXTERNAL_STATION_GLOB}"
    --external-station-exclusion-radius "${EXTERNAL_STATION_EXCLUSION_RADIUS}"
    --external-station-strict
    --external-station-report-path "${EXTERNAL_REPORT}"
)
if [ "${FORCE_RELOAD:-0}" = "1" ]; then
    ARGS+=(--force-reload)
fi

echo "============================================================"
echo "用Stage 0有效清单 + 固定152000随机池生成统一归一化"
echo "输出: ${NORMALIZATION_CONFIG}"
echo "============================================================"
python build_progressive_normalization.py "${ARGS[@]}"

python verify_progressive_normalization.py \
    --config "${NORMALIZATION_CONFIG}" \
    --stage0-manifest "${STAGE0_MANIFEST}" \
    --incremental-manifest "${INCREMENTAL_MANIFEST}" \
    --label-min 0 \
    --label-max 400
