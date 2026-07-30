#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp
mkdir -p logs

# 唯一实验变量：
#   Top-2: FUSION_FT_MAX_UNFROZEN_LAYERS=2
#   Full-4: FUSION_FT_MAX_UNFROZEN_LAYERS=0
# 本脚本固定为Full-4；其余设置沿用当前Nested 8/1/1正式协议。
STAGE=0 \
PRETRAINED_MODEL=/root/autodl-tmp/experiments/pretrain_stage0_station_20260714_215604/final_model.pth \
STRATEGIES="fusion_ft" \
RUN_FROZEN=0 \
RUN_EXTERNAL=0 \
INNER_PILOT_ONLY=0 \
NESTED_MAX_FOLDS=10 \
INCLUDE_FIXED_INTERNAL_IN_CV=1 \
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
bash run_progressive_finetune_stage.sh \
2>&1 | tee "logs/M0_fusion_full4_nested10_noaug_epoch80_$(date +%Y%m%d_%H%M%S).log"
