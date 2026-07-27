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
STAGE0_MANIFEST="${STAGE0_MANIFEST:-${SHARED_CACHE_DIR}/stage0_station_record_manifest.csv}"
EXTERNAL_REPORT="${EXTERNAL_REPORT:-${SHARED_CACHE_DIR}/external_station_exclusion_report.csv}"

for f in "${STATION_GUIDE_FILE}"; do
    if [ ! -f "${f}" ]; then
        echo "❌ 文件不存在: ${f}"
        exit 1
    fi
done
if ! compgen -G "${EXTERNAL_STATION_GLOB}" > /dev/null; then
    echo "❌ 没有匹配到外部测试CSV: ${EXTERNAL_STATION_GLOB}"
    exit 1
fi

mkdir -p "${SHARED_CACHE_DIR}"
cd "${SRC_DIR}"

rm -f "${STAGE0_MANIFEST}" \
      "${SHARED_CACHE_DIR}/stage0_station_record_manifest.stats.json" \
      "${EXTERNAL_REPORT}" \
      "${EXTERNAL_REPORT}.meta.json"

echo "============================================================"
echo "生成Stage 0实际站点-日期有效清单（不训练）"
echo "Excel: ${STATION_GUIDE_FILE}"
echo "外部测试CSV: ${EXTERNAL_STATION_GLOB}"
echo "外部排除半径: ${EXTERNAL_STATION_EXCLUSION_RADIUS}格"
echo "输出: ${STAGE0_MANIFEST}"
echo "============================================================"

python build_stage0_manifest.py \
  --station-guide-file "${STATION_GUIDE_FILE}" \
  --shared-cache-dir "${SHARED_CACHE_DIR}" \
  --output "${STAGE0_MANIFEST}" \
  --years 2015 2016 2017 2018 \
  --patch-size 5 \
  --min-valid-pixels 100 \
  --clamday-threshold 0.5 \
  --external-station-glob "${EXTERNAL_STATION_GLOB}" \
  --external-station-exclusion-radius "${EXTERNAL_STATION_EXCLUSION_RADIUS}" \
  --external-station-report-path "${EXTERNAL_REPORT}" \
  --label-min-mm 0 \
  --label-max-mm 400

python verify_external_station_exclusion.py \
  --manifest "${STAGE0_MANIFEST}" \
  --report "${EXTERNAL_REPORT}" \
  --name "Stage 0清单"

wc -l "${STAGE0_MANIFEST}"
cat "${SHARED_CACHE_DIR}/stage0_station_record_manifest.stats.json"
