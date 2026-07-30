#!/usr/bin/env bash
set -euo pipefail

DSTM_ROOT="${DSTM_ROOT:-/root/autodl-tmp}"
cd "${DSTM_ROOT}"
mkdir -p logs experiments

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ALL_STRATEGIES_RUN_ROOT:-${DSTM_ROOT}/experiments/M0_all_strategies_nogate_${RUN_STAMP}}"
LOG_PATH="${ALL_STRATEGIES_LOG_PATH:-${DSTM_ROOT}/logs/M0_all_strategies_nogate_${RUN_STAMP}.log}"

echo "===================================================================================================="
echo "M0全部微调策略临时对比"
echo "策略: fusion_ft point_ft spatial_ft partial none"
echo "Frozen基线: 同时生成"
echo "Frozen-relative gate: 临时关闭（仅本命令）"
echo "Checkpoint候选: 仅微调epoch；epoch-0 Frozen只审计，绝不回退"
echo "塌缩/非有限值保护: 保留"
echo "固定外部987样本: 不评估"
echo "运行目录: ${RUN_ROOT}"
echo "日志: ${LOG_PATH}"
echo "===================================================================================================="

STAGE=0 \
PRETRAINED_MODEL="${DSTM_ROOT}/experiments/pretrain_stage0_station_20260714_215604/final_model.pth" \
STRATEGIES="fusion_ft point_ft spatial_ft partial none" \
RUN_FROZEN=1 \
RUN_EXTERNAL=0 \
INNER_PILOT_ONLY=0 \
NESTED_MAX_FOLDS=10 \
INCLUDE_FIXED_INTERNAL_IN_CV=1 \
DISABLE_FROZEN_RELATIVE_GATE=1 \
FUSION_FT_MAX_UNFROZEN_LAYERS=0 \
FUSION_FT_TREND_EARLY_STOP=1 \
STATION_AUGMENTATION=0 \
COORD_JITTER_STD_DEG=0 \
MICROWAVE_NOISE_STD=0 \
COORD_MASK_PROB=0 \
FINE_TUNE_EPOCHS=80 \
BATCH_SIZE=32 \
NUM_WORKERS=8 \
EVAL_NUM_WORKERS=0 \
RUN_ROOT_OVERRIDE="${RUN_ROOT}" \
bash run_progressive_finetune_stage.sh \
2>&1 | tee -a "${LOG_PATH}"

python "${DSTM_ROOT}/summarize_all_strategies.py" \
    --run-root "${RUN_ROOT}"

echo "===================================================================================================="
echo "✅ M0全部微调策略完成"
echo "结果汇总: ${RUN_ROOT}/all_strategy_comparison"
echo "完整日志: ${LOG_PATH}"
echo "注意: 本次每折canonical只来自微调epoch，不会回退Frozen。"
echo "      省略DISABLE_FROZEN_RELATIVE_GATE或设为0，即恢复原Frozen-relative gate。"
echo "===================================================================================================="
