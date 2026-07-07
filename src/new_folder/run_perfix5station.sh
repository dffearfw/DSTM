#!/bin/bash

# ============================================
# 十折交叉验证 + 共享缓存 (mixed mode)
# unknown=训练/验证(十折), test=测试(独立)
# mixed mode: 站点实测 + 预训练伪标签回放
# 开启 full_sample_predictions.csv 产品值修正
# 关闭反事实训练
# ============================================

set -o pipefail
export PYTHONUNBUFFERED=1

PRETRAINED_MODEL="/root/autodl-tmp/experiments/swe_full_temporal_20260609_085603/best_model.pth"
STATION_DATA="/root/autodl-tmp/station_prefix_5_(1).csv"
EXPERIMENTS_BASE_DIR="/root/autodl-tmp/experiments"

BATCH_SIZE=32
EPOCHS=50
NUM_WORKERS=8
SEED=66666

# ============ Mixed mode 参数 ============
STATION_RATIO=1.0
PRETRAIN_LOSS_WEIGHT=0.0

# ============ 反事实训练参数（关闭） ============
USE_COUNTERFACTUAL=0
COUNTERFACTUAL_LOSS_WEIGHT=0.0

COUNTERFACTUAL_ARGS=()
if [ "${USE_COUNTERFACTUAL}" -eq 1 ]; then
    COUNTERFACTUAL_ARGS+=(
        --use_counterfactual_prior_loss
        --counterfactual_prior_loss_weight "${COUNTERFACTUAL_LOSS_WEIGHT}"
    )
fi

# ============ 产品值修正 ============
USE_PRODUCT_CORRECTION=1
CORRECTION_FILE="/root/autodl-tmp/full_sample_predictions.csv"
ZERO_CORRECTION_FILE="/root/autodl-tmp/zero_misclassifications.csv"

PRODUCT_CORRECTION_ARGS=()
if [ "${USE_PRODUCT_CORRECTION}" -eq 1 ]; then
    PRODUCT_CORRECTION_ARGS+=(--use_product_correction)
fi

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
RUN_ROOT_DIR="${EXPERIMENTS_BASE_DIR}/partial_productcorr_no_cf_${TIMESTAMP}"
mkdir -p "${RUN_ROOT_DIR}"

LOG_FILE="${RUN_ROOT_DIR}/experiment.log"

echo "==========================================" | tee "${LOG_FILE}"
echo "🚀 十折交叉验证 + 共享缓存" | tee -a "${LOG_FILE}"
echo "   mixed mode: station + pretrain replay + product correction" | tee -a "${LOG_FILE}"
echo "   开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
echo "   实验目录: ${RUN_ROOT_DIR}" | tee -a "${LOG_FILE}"
echo "   数据文件: ${STATION_DATA}" | tee -a "${LOG_FILE}"
echo "   预训练模型: ${PRETRAINED_MODEL}" | tee -a "${LOG_FILE}"
echo "   划分方式: split列(test=测试, unknown=训练/验证)" | tee -a "${LOG_FILE}"
echo "   站点比例: ${STATION_RATIO}" | tee -a "${LOG_FILE}"
echo "   预训练loss权重: ${PRETRAIN_LOSS_WEIGHT}" | tee -a "${LOG_FILE}"
echo "   反事实训练: USE_COUNTERFACTUAL=${USE_COUNTERFACTUAL}, weight=${COUNTERFACTUAL_LOSS_WEIGHT}" | tee -a "${LOG_FILE}"
echo "   产品值修正: ${USE_PRODUCT_CORRECTION}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

# ============ 文件检查 ============
echo "" | tee -a "${LOG_FILE}"
echo "🔧 产品值修正文件检查:" | tee -a "${LOG_FILE}"
echo "   full_sample_predictions: ${CORRECTION_FILE}" | tee -a "${LOG_FILE}"
echo "   zero_misclassifications: ${ZERO_CORRECTION_FILE}" | tee -a "${LOG_FILE}"

if [ "${USE_PRODUCT_CORRECTION}" -eq 1 ]; then
    if [ ! -f "${CORRECTION_FILE}" ]; then
        echo "❌ 找不到 ${CORRECTION_FILE}" | tee -a "${LOG_FILE}"
        exit 1
    else
        echo "✅ 找到 ${CORRECTION_FILE}" | tee -a "${LOG_FILE}"
    fi

    if [ ! -f "${ZERO_CORRECTION_FILE}" ]; then
        echo "⚠️ 找不到 ${ZERO_CORRECTION_FILE}" | tee -a "${LOG_FILE}"
        echo "   如果代码允许缺省 zero_misclassifications.csv，可忽略。" | tee -a "${LOG_FILE}"
    else
        echo "✅ 找到 ${ZERO_CORRECTION_FILE}" | tee -a "${LOG_FILE}"
    fi
fi

# ============ 策略列表 ============
STRATEGIES=(
    "partial"
    "none"
    "spatial_ft"
    "point_ft"
    "last_layer_only"
    "fusion_ft"
)

# ============ 策略配置 ============
declare -A FREEZE_STRATEGY=(
    ["last_layer_only"]="last_layer_only"
    ["fusion_ft"]="fusion_ft"
    ["point_ft"]="point_ft"
    ["spatial_ft"]="spatial_ft"
    ["partial"]="partial"
    ["none"]="none"
)

declare -A LR_HEAD=(
    ["last_layer_only"]="1e-3"
    ["fusion_ft"]="5e-4"
    ["point_ft"]="5e-4"
    ["spatial_ft"]="1e-4"
    ["partial"]="6e-4"
    ["none"]="5e-4"
)

declare -A LR_TRANS=(
    ["last_layer_only"]=""
    ["fusion_ft"]="5e-4"
    ["point_ft"]="5e-4"
    ["spatial_ft"]="1e-4"
    ["partial"]="4e-5"
    ["none"]="3e-5"
)

declare -A LR_ENC=(
    ["last_layer_only"]=""
    ["fusion_ft"]=""
    ["point_ft"]="1e-4"
    ["spatial_ft"]="1e-4"
    ["partial"]="1e-6"
    ["none"]="3e-5"
)

# ============ 划分缓存目录 ============
SPLIT_CACHE_DIR="./split_cache/partial_productcorr_no_cf_${TIMESTAMP}"
mkdir -p "${SPLIT_CACHE_DIR}"
SPLIT_CACHE_FILE="${SPLIT_CACHE_DIR}/shared_partial_productcorr_no_cf_split.pkl"

# ============ 打印数据分布 ============
echo "" | tee -a "${LOG_FILE}"
echo "📊 检查数据分布..." | tee -a "${LOG_FILE}"

python - << EOF 2>&1 | tee -a "${LOG_FILE}"
import pandas as pd

df = pd.read_csv("${STATION_DATA}")

print(f"总行数: {len(df)}")
print("split列分布:")
print(df["split"].value_counts())
print(f"站点数: {df['station_id'].nunique()}")

if "swe" in df.columns:
    print("\\nSWE统计:")
    print(df["swe"].describe())
    print(f"SWE>=20mm: {(df['swe'] >= 20).sum()} ({(df['swe'] >= 20).mean()*100:.2f}%)")
    print(f"SWE>=50mm: {(df['swe'] >= 50).sum()} ({(df['swe'] >= 50).mean()*100:.2f}%)")
    print(f"SWE>=80mm: {(df['swe'] >= 80).sum()} ({(df['swe'] >= 80).mean()*100:.2f}%)")
EOF

echo "==========================================" | tee -a "${LOG_FILE}"
echo "🚀 开始实验" | tee -a "${LOG_FILE}"
echo "   unknown → 训练/验证(十折)" | tee -a "${LOG_FILE}"
echo "   test    → 固定独立测试集" | tee -a "${LOG_FILE}"
echo "   产品值修正参数: ${PRODUCT_CORRECTION_ARGS[*]}" | tee -a "${LOG_FILE}"
echo "   反事实训练参数: ${COUNTERFACTUAL_ARGS[*]}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

# ============ frozen: 预训练模型直接测测试集 ============
echo "" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "📊 [frozen] 预训练模型直接测测试集" | tee -a "${LOG_FILE}"
echo "   开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

FROZEN_DIR="${RUN_ROOT_DIR}/frozen"
mkdir -p "${FROZEN_DIR}"

python main_tune.py \
    --mode evaluate \
    --model_type full \
    --batch_size "${BATCH_SIZE}" \
    --pretrained_model "${PRETRAINED_MODEL}" \
    --station_data_path "${STATION_DATA}" \
    --cv_mode station_cv \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --split_cache_file "${SPLIT_CACHE_FILE}" \
    "${PRODUCT_CORRECTION_ARGS[@]}" \
    --save_dir "${FROZEN_DIR}" 2>&1 | tee -a "${LOG_FILE}"

FROZEN_STATUS=${PIPESTATUS[0]}

if [ "${FROZEN_STATUS}" -eq 0 ]; then
    echo "✅ [frozen] 完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
else
    echo "❌ [frozen] 失败: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
fi

# ============ 循环训练各策略 ============
STRATEGY_COUNT=${#STRATEGIES[@]}
CURRENT=0

for STRATEGY in "${STRATEGIES[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo "" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    echo "📊 [${CURRENT}/${STRATEGY_COUNT}] 策略: ${STRATEGY}" | tee -a "${LOG_FILE}"
    echo "   开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
    echo "   冻结策略: ${FREEZE_STRATEGY[$STRATEGY]}" | tee -a "${LOG_FILE}"
    echo "   站点比例: ${STATION_RATIO}" | tee -a "${LOG_FILE}"
    echo "   预训练loss权重: ${PRETRAIN_LOSS_WEIGHT}" | tee -a "${LOG_FILE}"
    echo "   反事实训练: USE_COUNTERFACTUAL=${USE_COUNTERFACTUAL}, weight=${COUNTERFACTUAL_LOSS_WEIGHT}" | tee -a "${LOG_FILE}"
    echo "   产品值修正参数: ${PRODUCT_CORRECTION_ARGS[*]}" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"

    STRATEGY_DIR="${RUN_ROOT_DIR}/${STRATEGY}"
    mkdir -p "${STRATEGY_DIR}"

    LR_ARGS=()

    if [ -n "${LR_HEAD[$STRATEGY]}" ]; then
        LR_ARGS+=(--lr_head "${LR_HEAD[$STRATEGY]}")
    fi

    if [ -n "${LR_TRANS[$STRATEGY]}" ]; then
        LR_ARGS+=(--lr_transformer "${LR_TRANS[$STRATEGY]}")
    fi

    if [ -n "${LR_ENC[$STRATEGY]}" ]; then
        LR_ARGS+=(--lr_encoder "${LR_ENC[$STRATEGY]}")
    fi

    echo "   🏋️ 训练 mixed mode + product correction" | tee -a "${LOG_FILE}"
    echo "   学习率参数: ${LR_ARGS[*]}" | tee -a "${LOG_FILE}"

    python main_tune.py \
        --mode fine_tune \
        --model_type full \
        --batch_size "${BATCH_SIZE}" \
        --fine_tune_epochs "${EPOCHS}" \
        --freeze_backbone \
        --freeze_strategy "${FREEZE_STRATEGY[$STRATEGY]}" \
        --pretrained_model "${PRETRAINED_MODEL}" \
        --station_data_path "${STATION_DATA}" \
        "${LR_ARGS[@]}" \
        --cv_mode station_cv \
        --mixed_mode \
        --station_ratio "${STATION_RATIO}" \
        --pretrain_loss_weight "${PRETRAIN_LOSS_WEIGHT}" \
        --use_high_swe_weight \
        --pretrain_snow_min_mm 20.0 \
        --quality_threshold 0.83 \
        --snow_quality_threshold 0.60 \
        "${COUNTERFACTUAL_ARGS[@]}" \
        --seed "${SEED}" \
        --num_workers "${NUM_WORKERS}" \
        --split_cache_file "${SPLIT_CACHE_FILE}" \
        "${PRODUCT_CORRECTION_ARGS[@]}" \
        --save_dir "${STRATEGY_DIR}" 2>&1 | tee -a "${LOG_FILE}"

    STRATEGY_STATUS=${PIPESTATUS[0]}

    if [ "${STRATEGY_STATUS}" -eq 0 ]; then
        echo "✅ [${STRATEGY}] 完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
    else
        echo "❌ [${STRATEGY}] 失败: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
    fi

    sleep 3
done

echo "" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "✅ 全部完成!" | tee -a "${LOG_FILE}"
echo "   结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
echo "   结果目录: ${RUN_ROOT_DIR}" | tee -a "${LOG_FILE}"
echo "   日志文件: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "   划分缓存: ${SPLIT_CACHE_FILE}" | tee -a "${LOG_FILE}"
echo "   Mixed模式配置: 站点比例=${STATION_RATIO}, 预训练loss权重=${PRETRAIN_LOSS_WEIGHT}" | tee -a "${LOG_FILE}"
echo "   反事实训练: USE_COUNTERFACTUAL=${USE_COUNTERFACTUAL}, weight=${COUNTERFACTUAL_LOSS_WEIGHT}" | tee -a "${LOG_FILE}"
echo "   产品值修正: USE_PRODUCT_CORRECTION=${USE_PRODUCT_CORRECTION}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"