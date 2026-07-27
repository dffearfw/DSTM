#!/bin/bash
set -Eeuo pipefail
set -o pipefail

export PYTHONUNBUFFERED=1

# ============================================================
# 纯站点位置预训练
# - 只读取 station_swe_data.xlsx
# - 只使用站点所在 ERA5-Land 格点
# - 不扩展邻域
# - 不进行全国随机采样或全国 quota 补采
# ============================================================

# ---------- 数据过滤 ----------
export FILTER_GLACIER_SWE_ARTIFACTS=1
export GLACIER_SWE_THRESHOLD_MM=2000

# station 模式不使用随机采样；这里设为0只是让日志也保持一致。
export PRETRAIN_SAMPLES_PER_DAY=0

# 纯站点模式保留站点位置的自然 SWE 分布。
export USE_TARGET_QUOTA_SAMPLING=0
export USE_QUOTA_SHORTAGE_SUPPLEMENT=0
export STRICT_TARGET_QUOTA=0

# 大样本预计算可能占用大量内存，默认关闭。
export PRECOMPUTE_ALL_SAMPLES=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------- 路径 ----------
SRC_DIR="/root/autodl-tmp"
SAVE_DIR="/root/autodl-tmp/experiments"
SHARED_CACHE_DIR="/root/autodl-tmp/shared_cache"
STATION_GUIDE_FILE="/root/ablation/station_swe_data.xlsx"

# ---------- 采样参数 ----------
SAMPLING_MODE="station"       # random / station / hybrid
STATION_NEIGHBORHOOD=0        # 0 = 仅站点所在格点
STATION_SAMPLES_PER_DAY=-1    # <=0 = 每天使用全部有效站点格点
SAMPLES_PER_DAY=0             # station 模式下不会调用随机采样

# 默认仍过滤站点格点上的 ERA5-Land SWE=0。
# 想保留0值时，在 ARGS 中追加 --station_include_zero_target。

# ---------- 年份与特征 ----------
PRETRAIN_YEARS=(2015 2016 2017 2018)
PATCH_SIZE=5
MIN_VALID_PIXELS=100
CLAMDAY_THRESHOLD=0.5

# ---------- 训练参数 ----------
MODE="pretrain_progressive"
MODEL_TYPE="full"
EPOCHS=100
BATCH_SIZE=128
LR="1e-4"
D_MODEL=256
NUM_WORKERS=4
SEED=43
VAL_EVERY=5

RUN_NAME="pretrain_station_only_$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="${SAVE_DIR}/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/run.log"
mkdir -p "${RUN_DIR}" "${SHARED_CACHE_DIR}"

cd "${SRC_DIR}"

ARGS=(
    --mode "${MODE}"
    --model_type "${MODEL_TYPE}"
    --pretrain_years "${PRETRAIN_YEARS[@]}"

    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --d_model "${D_MODEL}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --val_every "${VAL_EVERY}"

    --pretrain_samples_per_day "${SAMPLES_PER_DAY}"
    --patch_size "${PATCH_SIZE}"
    --min_valid_pixels "${MIN_VALID_PIXELS}"
    --clamday_threshold "${CLAMDAY_THRESHOLD}"
    --shared_cache_dir "${SHARED_CACHE_DIR}"

    --sampling_mode "${SAMPLING_MODE}"
    --station_guide_file "${STATION_GUIDE_FILE}"
    --station_neighborhood "${STATION_NEIGHBORHOOD}"
    --station_samples_per_day "${STATION_SAMPLES_PER_DAY}"

    --use_amp
    --save_dir "${SAVE_DIR}"
    --exp_name "${RUN_NAME}"
)

echo "============================================================" | tee "${LOG_FILE}"
echo "采样模式: ${SAMPLING_MODE}" | tee -a "${LOG_FILE}"
echo "站点文件: ${STATION_GUIDE_FILE}" | tee -a "${LOG_FILE}"
echo "站点邻域: ${STATION_NEIGHBORHOOD}" | tee -a "${LOG_FILE}"
echo "每日站点上限: ${STATION_SAMPLES_PER_DAY}" | tee -a "${LOG_FILE}"
echo "运行目录: ${RUN_DIR}" | tee -a "${LOG_FILE}"
printf 'python main_tune.py ' | tee -a "${LOG_FILE}"
printf '%q ' "${ARGS[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

python main_tune.py "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
