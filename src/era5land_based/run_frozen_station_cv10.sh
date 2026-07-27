#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -Eeuo pipefail
set -o pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
M0_MODEL="${M0_MODEL:-${ROOT}/experiments/pretrain_stage0_station_20260714_215604/final_model.pth}"
INTERNAL_DATA="${ROOT}/shared_cache/progressive_finetune/internal_progressive_station.csv"
EXTERNAL_SOURCE="${ROOT}/shared_cache/progressive_finetune/external_evaluation_input.csv"
NORMALIZATION="${ROOT}/shared_cache/progressive_pretrain_normalization.json"
FOLD_MANIFEST="${ROOT}/shared_cache/progressive_finetune/balanced_station_cv10_manifest.csv"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
OUT="${OUT:-${ROOT}/experiments/frozen_M0_station_cv10_${TIMESTAMP}}"
LOG="${OUT}/run.log"

mkdir -p "${OUT}"
cd "${ROOT}"

for required in \
    "${ROOT}/balanced_station_cv10.py" \
    "${ROOT}/evaluate_frozen_station_cv10.py" \
    "${ROOT}/main_tune.py" \
    "${ROOT}/data_station_online_swe.py" \
    "${ROOT}/models_swe.py" \
    "${M0_MODEL}" \
    "${INTERNAL_DATA}" \
    "${EXTERNAL_SOURCE}" \
    "${NORMALIZATION}"
do
    [[ -f "${required}" ]] || {
        echo "❌ 缺少文件: ${required}"
        exit 1
    }
done

exec > >(tee -a "${LOG}") 2>&1

echo "================================================================================"
echo "Frozen M0 确定性平衡站点10折内部评估 + 外部987条单次评估"
echo "================================================================================"
echo "M0:       ${M0_MODEL}"
echo "内部CV池: ${INTERNAL_DATA}（只使用split!=test的6936条）"
echo "外部测试: ${EXTERNAL_SOURCE}（只评估一次）"
echo "fold清单:  ${FOLD_MANIFEST}"
echo "输出:     ${OUT}"
echo "================================================================================"

python "${ROOT}/balanced_station_cv10.py" \
    --station-csv "${INTERNAL_DATA}" \
    --output "${FOLD_MANIFEST}" \
    --n-splits 10 \
    --high-threshold-mm 80

python "${ROOT}/evaluate_frozen_station_cv10.py" \
    --root "${ROOT}" \
    --station-csv "${INTERNAL_DATA}" \
    --checkpoint "${M0_MODEL}" \
    --normalization-config "${NORMALIZATION}" \
    --cache-dir "${ROOT}/shared_cache" \
    --output-dir "${OUT}/internal_cv10" \
    --fold-manifest "${FOLD_MANIFEST}" \
    --seed 43 \
    --n-splits 10 \
    --batch-size 128 \
    --num-workers 0

# 内部脚本退出后，其16.8GB缓存被释放；再单独加载外部987条，避免双份缓存并存。
EXTERNAL_RUNTIME_DIR="${OUT}/runtime_inputs/external"
EXTERNAL_RUNTIME_FILE="${EXTERNAL_RUNTIME_DIR}/external_evaluation_input.csv"
mkdir -p "${EXTERNAL_RUNTIME_DIR}"
cp -p "${EXTERNAL_SOURCE}" "${EXTERNAL_RUNTIME_FILE}"

python "${ROOT}/main_tune.py" \
    --mode evaluate \
    --pretrained_model "${M0_MODEL}" \
    --model_path "${M0_MODEL}" \
    --station_data_path "${EXTERNAL_RUNTIME_FILE}" \
    --save_dir "${OUT}/external_987_once" \
    --model_type full \
    --batch_size 32 \
    --num_workers 0 \
    --lr 1e-4 \
    --d_model 256 \
    --seed 43 \
    --cv_mode station_cv \
    --normalization_config_path "${NORMALIZATION}" \
    --normalization_mode load \
    --fixed_label_min_mm 0 \
    --fixed_label_max_mm 400 \
    --coord_jitter_std 0 \
    --microwave_noise_std 0 \
    --coord_mask_prob 0 \
    --val_every 1 \
    --use_amp

echo "================================================================================"
echo "✅ Frozen试验完成"
echo "内部10折: ${OUT}/internal_cv10"
echo "外部一次: ${OUT}/external_987_once"
echo "================================================================================"
