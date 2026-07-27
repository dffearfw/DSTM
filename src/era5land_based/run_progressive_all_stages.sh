#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -Eeuo pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
RUNNER="${ROOT}/run_progressive_finetune_stage.sh"

if [[ ! -x "${RUNNER}" ]]; then
    echo "❌ 找不到可执行脚本: ${RUNNER}"
    exit 1
fi

# M0已有默认路径；M1-M4在对应阶段完成后填写。
M0_MODEL="${M0_MODEL:-/root/autodl-tmp/experiments/pretrain_stage0_station_20260714_215604/final_model.pth}"
M1_MODEL="${M1_MODEL:-}"
M2_MODEL="${M2_MODEL:-}"
M3_MODEL="${M3_MODEL:-}"
M4_MODEL="${M4_MODEL:-}"

declare -A MODELS=(
    [0]="${M0_MODEL}"
    [1]="${M1_MODEL}"
    [2]="${M2_MODEL}"
    [3]="${M3_MODEL}"
    [4]="${M4_MODEL}"
)

# 可用 ONLY_STAGES="0 1" 限制运行阶段。
ONLY_STAGES="${ONLY_STAGES:-0 1 2 3 4}"

for STAGE in ${ONLY_STAGES}; do
    MODEL_PATH="${MODELS[${STAGE}]:-}"

    if [[ -z "${MODEL_PATH}" ]]; then
        echo "⏭️ 跳过M${STAGE}: 尚未设置模型路径"
        continue
    fi

    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "❌ M${STAGE}模型不存在: ${MODEL_PATH}"
        exit 1
    fi

    echo
    echo "################################################################################"
    echo "开始运行M${STAGE}"
    echo "模型: ${MODEL_PATH}"
    echo "################################################################################"

    STAGE="${STAGE}" \
    PRETRAINED_MODEL="${MODEL_PATH}" \
    bash "${RUNNER}"
done
