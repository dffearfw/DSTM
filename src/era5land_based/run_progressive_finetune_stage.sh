#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -Eeuo pipefail
set -o pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT="${ROOT:-/root/autodl-tmp}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${ROOT}/experiments}"
MANIFEST_DIR="${MANIFEST_DIR:-${ROOT}/shared_cache/progressive_finetune}"

STAGE="${STAGE:?必须设置 STAGE=0/1/2/3/4}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:?必须设置 PRETRAINED_MODEL=/path/to/final_model.pth}"

case "${STAGE}" in
    0|1|2|3|4)
        ;;
    *)
        echo "❌ STAGE必须是0、1、2、3或4，当前=${STAGE}"
        exit 2
        ;;
esac

MAIN="${ROOT}/main_tune.py"
STATION_MODULE="${ROOT}/data_station_online_swe.py"

INTERNAL_DATA="${MANIFEST_DIR}/internal_progressive_station.csv"
EXTERNAL_DATA="${MANIFEST_DIR}/external_evaluation_input.csv"
MANIFEST_SUMMARY="${MANIFEST_DIR}/progressive_finetune_manifest_summary.json"

NORMALIZATION_CONFIG="${ROOT}/shared_cache/progressive_pretrain_normalization.json"
INCLUDE_FIXED_INTERNAL_IN_CV="${INCLUDE_FIXED_INTERNAL_IN_CV:-1}"
if [[ "${INCLUDE_FIXED_INTERNAL_IN_CV}" == "1" ]]; then
    FOLD_MANIFEST="${MANIFEST_DIR}/balanced_station_nested_cv10_all7936_manifest.csv"
    BALANCED_INCLUDE_ARGS=(--include-fixed-test)
    INTERNAL_SPLIT_DESCRIPTION="7,936条（旧固定1000条已并回）"
    EXPECTED_NESTED_SAMPLES=7936
else
    FOLD_MANIFEST="${MANIFEST_DIR}/balanced_station_cv10_manifest.csv"
    BALANCED_INCLUDE_ARGS=()
    INTERNAL_SPLIT_DESCRIPTION="6,936条（旧固定1000条保持排除）"
    EXPECTED_NESTED_SAMPLES=6936
fi
BALANCED_FOLD_TOOL="${ROOT}/balanced_station_cv10.py"
EXTERNAL_ENSEMBLE_TOOL="${ROOT}/evaluate_cv10_ensemble_external.py"
FROZEN_CV_RUNNER="${ROOT}/run_frozen_station_cv10.sh"

SEED="${SEED:-43}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OVERFIT_TRAIN_ONLY="${OVERFIT_TRAIN_ONLY:-0}"
OVERFIT_RESUME_CHECKPOINT="${OVERFIT_RESUME_CHECKPOINT:-}"
OVERFIT_RESUME_TRANSFORMER_LR="${OVERFIT_RESUME_TRANSFORMER_LR:-}"
INNER_PILOT_ONLY="${INNER_PILOT_ONLY:-0}"
if [[ "${INNER_PILOT_ONLY}" == "1" ]]; then
    NESTED_MAX_FOLDS="${NESTED_MAX_FOLDS:-1}"
else
    NESTED_MAX_FOLDS="${NESTED_MAX_FOLDS:-10}"
fi
FUSION_FT_HEAD_LR="${FUSION_FT_HEAD_LR:-5e-5}"
FUSION_FT_TRANSFORMER_LR="${FUSION_FT_TRANSFORMER_LR:-3e-5}"
FUSION_FT_WARMUP_EPOCHS="${FUSION_FT_WARMUP_EPOCHS:-10}"
FUSION_FT_UNFREEZE_INTERVAL="${FUSION_FT_UNFREEZE_INTERVAL:-1}"
# 0=Full-4 Fusion：最终解冻全部4个Transformer block和共享Fusion参数；
# 正整数N=仅解冻顶部N个Transformer block。
FUSION_FT_MAX_UNFROZEN_LAYERS="${FUSION_FT_MAX_UNFROZEN_LAYERS:-0}"
FUSION_FT_PLATEAU_PATIENCE="${FUSION_FT_PLATEAU_PATIENCE:-8}"
FUSION_FT_MIN_EPOCHS="${FUSION_FT_MIN_EPOCHS:-40}"
FUSION_FT_PATIENCE="${FUSION_FT_PATIENCE:-20}"
# 1=趋势感知早停（默认）；0=恢复旧版“只有完全合格才重置patience”。
FUSION_FT_TREND_EARLY_STOP="${FUSION_FT_TREND_EARLY_STOP:-1}"
# FROZEN_RELATIVE_GATE_ENV_TOGGLE_V1
# 0=原始Frozen-relative准入门槛（默认）；1=临时关闭门槛，
# 在非塌缩微调epoch中按综合selection_score选择。
DISABLE_FROZEN_RELATIVE_GATE="${DISABLE_FROZEN_RELATIVE_GATE:-0}"
if [[ "${OVERFIT_TRAIN_ONLY}" == "1" ]]; then
    FINE_TUNE_EPOCHS="${FINE_TUNE_EPOCHS:-100}"
else
    FINE_TUNE_EPOCHS="${FINE_TUNE_EPOCHS:-80}"
fi
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
# 本轮Full-4对照明确不使用数据增强。保留显式开关仅用于可逆恢复；
# 默认0会同时把三项增强强度归零。
STATION_AUGMENTATION="${STATION_AUGMENTATION:-0}"
COORD_JITTER_STD_DEG="${COORD_JITTER_STD_DEG:-0}"
MICROWAVE_NOISE_STD="${MICROWAVE_NOISE_STD:-0}"
COORD_MASK_PROB="${COORD_MASK_PROB:-0}"

case "${OVERFIT_TRAIN_ONLY}" in
    0|1)
        ;;
    *)
        echo "❌ OVERFIT_TRAIN_ONLY必须是0或1，当前=${OVERFIT_TRAIN_ONLY}"
        exit 2
        ;;
esac

case "${INNER_PILOT_ONLY}" in
    0|1)
        ;;
    *)
        echo "❌ INNER_PILOT_ONLY必须是0或1，当前=${INNER_PILOT_ONLY}"
        exit 2
        ;;
esac

case "${FUSION_FT_TREND_EARLY_STOP}" in
    0|1)
        ;;
    *)
        echo "❌ FUSION_FT_TREND_EARLY_STOP必须是0或1"
        exit 2
        ;;
esac

case "${DISABLE_FROZEN_RELATIVE_GATE}" in
    0|1)
        ;;
    *)
        echo "❌ DISABLE_FROZEN_RELATIVE_GATE必须是0或1"
        exit 2
        ;;
esac

case "${STATION_AUGMENTATION}" in
    0|1)
        ;;
    *)
        echo "❌ STATION_AUGMENTATION必须是0或1"
        exit 2
        ;;
esac

if [[ "${STATION_AUGMENTATION}" == "0" ]]; then
    COORD_JITTER_STD_DEG=0
    MICROWAVE_NOISE_STD=0
    COORD_MASK_PROB=0
fi

TREND_EARLY_STOP_ARGS=()
if [[ "${FUSION_FT_TREND_EARLY_STOP}" == "0" ]]; then
    TREND_EARLY_STOP_ARGS+=(--disable_fusion_ft_trend_early_stopping)
fi

FROZEN_RELATIVE_GATE_ARGS=()
if [[ "${DISABLE_FROZEN_RELATIVE_GATE}" == "1" ]]; then
    FROZEN_RELATIVE_GATE_ARGS+=(--disable_frozen_relative_gate)
fi

case "${INCLUDE_FIXED_INTERNAL_IN_CV}" in
    0|1)
        ;;
    *)
        echo "❌ INCLUDE_FIXED_INTERNAL_IN_CV必须是0或1"
        exit 2
        ;;
esac

if [[ "${OVERFIT_TRAIN_ONLY}" == "1" && "${INNER_PILOT_ONLY}" == "1" ]]; then
    echo "❌ OVERFIT_TRAIN_ONLY与INNER_PILOT_ONLY不能同时为1"
    exit 2
fi

if [[ "${INNER_PILOT_ONLY}" == "1" && "${NESTED_MAX_FOLDS}" != "1" ]]; then
    echo "❌ INNER_PILOT_ONLY=1时NESTED_MAX_FOLDS必须为1"
    exit 2
fi

if [[ "${INNER_PILOT_ONLY}" == "0" && "${NESTED_MAX_FOLDS}" != "10" ]]; then
    echo "❌ 正式Nested OOF要求NESTED_MAX_FOLDS=10"
    exit 2
fi

python - \
    "${FUSION_FT_HEAD_LR}" \
    "${FUSION_FT_TRANSFORMER_LR}" \
    "${FUSION_FT_WARMUP_EPOCHS}" \
    "${FUSION_FT_UNFREEZE_INTERVAL}" \
    "${FUSION_FT_MAX_UNFROZEN_LAYERS}" \
    "${FUSION_FT_PLATEAU_PATIENCE}" \
    "${FUSION_FT_MIN_EPOCHS}" \
    "${FUSION_FT_PATIENCE}" \
    "${COORD_JITTER_STD_DEG}" \
    "${MICROWAVE_NOISE_STD}" \
    "${COORD_MASK_PROB}" <<'PY'
import math
import sys

head_lr = float(sys.argv[1])
transformer_lr = float(sys.argv[2])
integer_values = [int(value) for value in sys.argv[3:9]]
(
    warmup,
    interval,
    max_unfrozen_layers,
    plateau_patience,
    min_epochs,
    patience,
) = integer_values
coordinate_jitter_std_deg = float(sys.argv[9])
microwave_noise_std = float(sys.argv[10])
coordinate_mask_prob = float(sys.argv[11])

if not math.isfinite(head_lr) or head_lr <= 0:
    raise SystemExit("FUSION_FT_HEAD_LR必须是有限正数")
if not math.isfinite(transformer_lr) or transformer_lr <= 0:
    raise SystemExit("FUSION_FT_TRANSFORMER_LR必须是有限正数")
if warmup < 0:
    raise SystemExit("FUSION_FT_WARMUP_EPOCHS不能小于0")
if interval <= 0:
    raise SystemExit("FUSION_FT_UNFREEZE_INTERVAL必须大于0")
if max_unfrozen_layers < 0:
    raise SystemExit("FUSION_FT_MAX_UNFROZEN_LAYERS不能小于0")
if plateau_patience < 0:
    raise SystemExit("FUSION_FT_PLATEAU_PATIENCE不能小于0")
if min_epochs < 0:
    raise SystemExit("FUSION_FT_MIN_EPOCHS不能小于0")
if patience <= 0:
    raise SystemExit("FUSION_FT_PATIENCE必须大于0")
if (
    not math.isfinite(coordinate_jitter_std_deg)
    or coordinate_jitter_std_deg < 0
):
    raise SystemExit("COORD_JITTER_STD_DEG必须是有限非负数")
if (
    not math.isfinite(microwave_noise_std)
    or microwave_noise_std < 0
):
    raise SystemExit("MICROWAVE_NOISE_STD必须是有限非负数")
if (
    not math.isfinite(coordinate_mask_prob)
    or not 0 <= coordinate_mask_prob <= 1
):
    raise SystemExit("COORD_MASK_PROB必须位于[0,1]")
PY

if [[ -n "${OVERFIT_RESUME_CHECKPOINT}" ]]; then
    if [[ "${OVERFIT_TRAIN_ONLY}" != "1" ]]; then
        echo "❌ OVERFIT_RESUME_CHECKPOINT仅允许与OVERFIT_TRAIN_ONLY=1一起使用"
        exit 2
    fi
    if [[ ! -f "${OVERFIT_RESUME_CHECKPOINT}" ]]; then
        echo "❌ 过拟合续训checkpoint不存在: ${OVERFIT_RESUME_CHECKPOINT}"
        exit 1
    fi
fi

if [[ -n "${OVERFIT_RESUME_TRANSFORMER_LR}" ]]; then
    if [[ -z "${OVERFIT_RESUME_CHECKPOINT}" ]]; then
        echo "❌ OVERFIT_RESUME_TRANSFORMER_LR必须与OVERFIT_RESUME_CHECKPOINT一起使用"
        exit 2
    fi
    python - "${OVERFIT_RESUME_TRANSFORMER_LR}" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0:
    raise SystemExit(
        "OVERFIT_RESUME_TRANSFORMER_LR必须是有限正数"
    )
PY
fi

# DETERMINISTIC_TRAINING_V1
# PYTHONHASHSEED必须在Python进程启动前设置；cuBLAS确定性同理。
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN_FROZEN="${RUN_FROZEN:-1}"
RUN_EXTERNAL="${RUN_EXTERNAL:-1}"

# 空格分隔；可临时只跑某几个策略，例如：
# STRATEGIES="partial none"
if [[ "${OVERFIT_TRAIN_ONLY}" == "1" ]]; then
    STRATEGIES="${STRATEGIES:-fusion_ft}"
    RUN_FROZEN=0
    RUN_EXTERNAL=0
    if [[ "${STRATEGIES}" != "fusion_ft" ]]; then
        echo "❌ 训练集过拟合诊断只允许 STRATEGIES=\"fusion_ft\""
        exit 2
    fi
elif [[ "${INNER_PILOT_ONLY}" == "1" ]]; then
    STRATEGIES="${STRATEGIES:-fusion_ft}"
    RUN_FROZEN=0
    RUN_EXTERNAL=0
    if [[ "${STRATEGIES}" != "fusion_ft" ]]; then
        echo "❌ Inner pilot只允许 STRATEGIES=\"fusion_ft\""
        exit 2
    fi
else
    STRATEGIES="${STRATEGIES:-fusion_ft point_ft spatial_ft partial none}"
fi

# 非Fusion策略保留策略级fine_tune_lr；Fusion FT在main_tune内使用
# Head/Transformer独立低学习率、5轮warmup和逐层解冻。
# 同一策略在M0-M4中必须完全一致。
declare -A STRATEGY_LR=(
    ["fusion_ft"]="5e-5"
    ["point_ft"]="1e-4"
    ["spatial_ft"]="1e-4"
    ["partial"]="4e-5"
    ["none"]="3e-5"
)

for required_file in \
    "${MAIN}" \
    "${STATION_MODULE}" \
    "${BALANCED_FOLD_TOOL}" \
    "${EXTERNAL_ENSEMBLE_TOOL}" \
    "${FROZEN_CV_RUNNER}" \
    "${PRETRAINED_MODEL}" \
    "${INTERNAL_DATA}" \
    "${EXTERNAL_DATA}" \
    "${MANIFEST_SUMMARY}" \
    "${NORMALIZATION_CONFIG}"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "❌ 缺少必要文件: ${required_file}"
        exit 1
    fi
done

# 正式微调必须先应用统一归一化补丁。
if ! grep -q "PROGRESSIVE_FINETUNE_NORMALIZATION_V1" "${STATION_MODULE}"; then
    echo "❌ data_station_online_swe.py尚未应用统一归一化补丁"
    echo "   请先运行:"
    echo "   python ${ROOT}/apply_progressive_finetune_normalization.py"
    exit 1
fi

if ! grep -q "PROGRESSIVE_FINETUNE_PASS_NORMALIZATION_V1" "${MAIN}"; then
    echo "❌ main_tune.py尚未传递统一归一化参数"
    echo "   请先运行:"
    echo "   python ${ROOT}/apply_progressive_finetune_normalization.py"
    exit 1
fi

cd "${ROOT}"

echo "🔍 检查固定清单数量与SHA256..."

python - \
    "${MANIFEST_SUMMARY}" \
    "${INTERNAL_DATA}" \
    "${EXTERNAL_DATA}" \
    "${INCLUDE_FIXED_INTERNAL_IN_CV}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
internal_path = Path(sys.argv[2])
external_input_path = Path(sys.argv[3])
include_fixed_internal = sys.argv[4] == "1"

with summary_path.open("r", encoding="utf-8") as f:
    summary = json.load(f)

counts = summary["counts"]

assert counts["internal_cv_samples"] == 6936, counts
assert counts["internal_test_samples"] == 1000, counts
assert (
    counts["internal_cv_samples"]
    + counts["internal_test_samples"]
) == 7936, counts
assert counts["external_aggregated_samples"] == 987, counts

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

expected_internal = summary["files"]["internal_progressive_station.csv"]["sha256"]
expected_external = summary["files"]["external_evaluation_input.csv"]["sha256"]

actual_internal = sha256(internal_path)
actual_external = sha256(external_input_path)

if actual_internal != expected_internal:
    raise RuntimeError(
        "internal_progressive_station.csv的SHA256已变化：\n"
        f"expected={expected_internal}\nactual={actual_internal}"
    )

if actual_external != expected_external:
    raise RuntimeError(
        "external_evaluation_input.csv的SHA256已变化：\n"
        f"expected={expected_external}\nactual={actual_external}"
    )

print("✅ 固定数据清单检查通过")
if include_fixed_internal:
    print(
        "   内部Nested池: "
        f"{counts['internal_cv_samples'] + counts['internal_test_samples']:,}"
    )
    print(
        f"   其中旧固定1000条: "
        f"{counts['internal_test_samples']:,}（并回Nested）"
    )
else:
    print(f"   内部Nested池: {counts['internal_cv_samples']:,}")
    print(
        f"   旧固定内部测试: "
        f"{counts['internal_test_samples']:,}（保持排除）"
    )
print(f"   外部聚合测试: {counts['external_aggregated_samples']:,}")
PY

if [[ -n "${RUN_ROOT_OVERRIDE:-}" ]]; then
    RUN_ROOT="${RUN_ROOT_OVERRIDE}"
    RUN_NAME="$(basename "${RUN_ROOT}")"
    echo "♻ 使用已有运行目录继续执行: ${RUN_ROOT}"
else
    TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
    if [[ "${OVERFIT_TRAIN_ONLY}" == "1" ]]; then
        RUN_NAME="M${STAGE}_fusion_train_overfit_${TIMESTAMP}"
    elif [[ "${INNER_PILOT_ONLY}" == "1" ]]; then
        RUN_NAME="M${STAGE}_fusion_inner_pilot_${TIMESTAMP}"
    else
        RUN_NAME="progressive_M${STAGE}_finetune_${TIMESTAMP}"
    fi
    RUN_ROOT="${EXPERIMENTS_DIR}/${RUN_NAME}"
fi

LOG_FILE="${RUN_ROOT}/run.log"
mkdir -p "${RUN_ROOT}"

# 所有阶段、所有策略应产生完全相同的十折清单。
REFERENCE_SPLIT_JSON="${MANIFEST_DIR}/station_nested_cv_all7936_seed43_reference.json"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================================"
echo "渐进式预训练模型微调与双测试"
echo "================================================================================"
echo "模型阶段:            M${STAGE}"
echo "预训练模型:          ${PRETRAINED_MODEL}"
echo "内部固定数据:        ${INTERNAL_DATA}"
echo "外部固定数据:        ${EXTERNAL_DATA}"
echo "统一归一化:          ${NORMALIZATION_CONFIG}"
echo "内部Nested池:        ${INTERNAL_SPLIT_DESCRIPTION}"
echo "外部测试:            外部CSV全量聚合，987条"
echo "随机种子:            ${SEED}"
echo "确定性模式:          Python/NumPy/PyTorch/CUDA/DataLoader固定；cuDNN deterministic"
echo "微调轮数:            ${FINE_TUNE_EPOCHS}"
echo "批次大小:            ${BATCH_SIZE}"
echo "训练DataLoader workers: ${NUM_WORKERS}"
echo "评估DataLoader workers: ${EVAL_NUM_WORKERS}（串行评估，降低内存）"
echo "策略:                ${STRATEGIES}"
echo "完整训练集过拟合诊断: ${OVERFIT_TRAIN_ONLY}"
echo "单Inner pilot:        ${INNER_PILOT_ONLY}"
echo "Nested运行折数:       ${NESTED_MAX_FOLDS}"
if [[ -n "${OVERFIT_RESUME_CHECKPOINT}" ]]; then
    echo "过拟合断点续训:      ${OVERFIT_RESUME_CHECKPOINT}"
    echo "续训目标总轮数:      ${FINE_TUNE_EPOCHS}"
    if [[ -n "${OVERFIT_RESUME_TRANSFORMER_LR}" ]]; then
        echo "续训Transformer LR:  ${OVERFIT_RESUME_TRANSFORMER_LR}（仅恢复后覆盖）"
    else
        echo "续训Transformer LR:  沿用checkpoint"
    fi
else
    echo "过拟合断点续训:      关闭"
fi
echo "产品值修正:          关闭"
echo "预训练伪标签回放:    关闭"
echo "Fusion FT学习率:      Head=${FUSION_FT_HEAD_LR}, Transformer峰值=${FUSION_FT_TRANSFORMER_LR}"
echo "Fusion FT稳定阶段:    warmup=${FUSION_FT_WARMUP_EPOCHS}轮, 每${FUSION_FT_UNFREEZE_INTERVAL}轮解冻1层"
if [[ "${FUSION_FT_MAX_UNFROZEN_LAYERS}" == "0" ]]; then
    echo "Fusion FT解冻上限:    0（兼容旧行为：全部Transformer层 + 共享Fusion参数）"
else
    echo "Fusion FT解冻上限:    顶部${FUSION_FT_MAX_UNFROZEN_LAYERS}个Transformer block；底层与共享Fusion参数冻结"
fi
echo "Fusion FT调度:        Plateau patience=${FUSION_FT_PLATEAU_PATIENCE}, factor=0.5"
echo "Fusion FT尾部策略:    自然随机采样 + 非累计1.2/1.5/2.0倍权重，无额外high-bias loss"
echo "Fusion FT结构损失:    CCC权重=0.005，旧batch方差惩罚=0"
if [[ "${STATION_AUGMENTATION}" == "1" ]]; then
    echo "站点训练增强:        仅Train启用；坐标抖动=${COORD_JITTER_STD_DEG}°, 微波噪声=${MICROWAVE_NOISE_STD}, 坐标掩码=${COORD_MASK_PROB}"
else
    echo "站点训练增强:        关闭（Inner/Outer始终不增强）"
fi
echo "Fusion FT梯度诊断:    前5轮首batch比较station与加权CCC梯度，不改变训练"
echo "Fusion FT泛化诊断:    固定训练审计集 vs inner-validation；不参与选模"
echo "Fusion FT早停:        最少${FUSION_FT_MIN_EPOCHS}轮，patience=${FUSION_FT_PATIENCE}"
if [[ "${FUSION_FT_TREND_EARLY_STOP}" == "1" ]]; then
    echo "Fusion FT早停依据:    Inner连续gate debt + RMSE/R/slope/std趋势；硬门槛仅负责最终选模"
else
    echo "Fusion FT早停依据:    旧版硬逻辑（仅完全合格checkpoint重置patience）"
fi
if [[ "${DISABLE_FROZEN_RELATIVE_GATE}" == "1" ]]; then
    echo "Checkpoint选模:       ⚠ 临时关闭Frozen-relative gate；非塌缩微调epoch按综合selection_score最低选择"
else
    echo "Checkpoint选模:       Frozen-relative准入后，按综合selection_score最低选择"
fi
echo "运行目录:            ${RUN_ROOT}"
echo "================================================================================"

COMMON_ARGS=(
    --model_type full
    --batch_size "${BATCH_SIZE}"
    --lr 1e-4
    --d_model 256
    --seed "${SEED}"
    --cv_mode station_cv
    --normalization_config_path "${NORMALIZATION_CONFIG}"
    --normalization_mode load
    --fixed_label_min_mm 0
    --fixed_label_max_mm 400
    --coord_jitter_std "${COORD_JITTER_STD_DEG}"
    --microwave_noise_std "${MICROWAVE_NOISE_STD}"
    --coord_mask_prob "${COORD_MASK_PROB}"
    --val_every 1
    --use_amp
)
if [[ "${OVERFIT_TRAIN_ONLY}" == "1" ]]; then
    COMMON_ARGS+=(--train_only_overfit)
    if [[ -n "${OVERFIT_RESUME_CHECKPOINT}" ]]; then
        COMMON_ARGS+=(
            --overfit_resume_checkpoint
            "${OVERFIT_RESUME_CHECKPOINT}"
        )
        if [[ -n "${OVERFIT_RESUME_TRANSFORMER_LR}" ]]; then
            COMMON_ARGS+=(
                --overfit_resume_transformer_lr
                "${OVERFIT_RESUME_TRANSFORMER_LR}"
            )
        fi
    fi
fi
if [[ "${INNER_PILOT_ONLY}" == "1" ]]; then
    COMMON_ARGS+=(
        --inner_pilot_only
        --nested_max_folds "${NESTED_MAX_FOLDS}"
    )
else
    COMMON_ARGS+=(
        --nested_max_folds "${NESTED_MAX_FOLDS}"
    )
fi
if [[ "${INCLUDE_FIXED_INTERNAL_IN_CV}" == "1" ]]; then
    COMMON_ARGS+=(--include_fixed_internal_in_cv)
fi

promote_unique_result() {
    local search_root="$1"
    local filename="$2"
    local required="${3:-1}"
    local canonical="${search_root}/${filename}"

    if [[ -f "${canonical}" ]]; then
        echo "${canonical}"
        return 0
    fi

    local -a matches=()

    mapfile -t matches < <(
        find "${search_root}" \
            -mindepth 1 \
            -maxdepth 5 \
            -type f \
            -name "${filename}" \
            -print 2>/dev/null | sort
    )

    if [[ "${#matches[@]}" -eq 1 ]]; then
        cp -p "${matches[0]}" "${canonical}"

        echo "✅ 已归位嵌套结果:"
        echo "   来源: ${matches[0]}"
        echo "   目标: ${canonical}"

        echo "${canonical}"
        return 0
    fi

    if [[ "${#matches[@]}" -gt 1 ]]; then
        echo "❌ 找到多个同名结果，拒绝猜测:"
        printf '   %s\n' "${matches[@]}"
        exit 1
    fi

    if [[ "${required}" == "1" ]]; then
        echo "❌ 没有找到结果文件:"
        echo "   根目录: ${search_root}"
        echo "   文件名: ${filename}"
        exit 1
    fi

    return 1
}


run_evaluation() {
    local model_path="$1"
    local station_file="$2"
    local output_dir="$3"
    local evaluation_name="$4"
    local result_name="fine_tune_evaluation_results.json"
    local prediction_name="test_set_features_complete_with_pretrained.csv"
    local summary_name="fine_tune_summary.txt"

    # PROGRESSIVE_EXTERNAL_ISOLATED_MANIFEST_V1
    if [[ "${station_file}" == "${EXTERNAL_DATA}" ]]; then
        local isolated_dir="${RUN_ROOT}/runtime_inputs/external_evaluation"
        local isolated_file="${isolated_dir}/external_evaluation_input.csv"

        mkdir -p "${isolated_dir}"

        if [[ ! -f "${isolated_file}" ]]; then
            cp -p "${EXTERNAL_DATA}" "${isolated_file}"
        fi

        if ! cmp -s "${EXTERNAL_DATA}" "${isolated_file}"; then
            echo "❌ 外部评估隔离副本与原文件不一致"
            echo "   原文件: ${EXTERNAL_DATA}"
            echo "   隔离副本: ${isolated_file}"
            exit 1
        fi

        station_file="${isolated_file}"

        echo "✅ 外部评估使用隔离清单:"
        echo "   ${station_file}"
        echo "   该目录不包含内部1000条固定测试文件"
    fi

    mkdir -p "${output_dir}"

    # FOLDWISE_EVALUATION_OUTPUT_V1
    # 聚合不仅需要指标JSON，还需要逐样本预测CSV。
    promote_unique_result "${output_dir}" "${result_name}" 0 >/dev/null || true
    promote_unique_result "${output_dir}" "${prediction_name}" 0 >/dev/null || true
    promote_unique_result "${output_dir}" "${summary_name}" 0 >/dev/null || true

    if [[ -f "${output_dir}/${result_name}" &&
          -f "${output_dir}/${prediction_name}" ]]; then
        echo
        echo "✅ 已有完整评估结果，跳过重复运行:"
        echo "   ${output_dir}/${result_name}"
        echo "   ${output_dir}/${prediction_name}"
        return 0
    fi

    echo
    echo "================================================================================"
    echo "评估: ${evaluation_name}"
    echo "模型: ${model_path}"
    echo "数据: ${station_file}"
    echo "输出: ${output_dir}"
    echo "================================================================================"

    python "${MAIN}" \
        --mode evaluate \
        --pretrained_model "${model_path}" \
        --model_path "${model_path}" \
        --station_data_path "${station_file}" \
        --save_dir "${output_dir}" \
        --num_workers "${EVAL_NUM_WORKERS}" \
        "${COMMON_ARGS[@]}"

    promote_unique_result "${output_dir}" "${result_name}" 1 >/dev/null
    promote_unique_result "${output_dir}" "${prediction_name}" 1 >/dev/null
    promote_unique_result "${output_dir}" "${summary_name}" 0 >/dev/null || true

    printf '%s\n' "${model_path}" > "${output_dir}/model_path.txt"

    echo "✅ 评估结果确认完成:"
    echo "   ${output_dir}/${result_name}"
    echo "   ${output_dir}/${prediction_name}"
}


aggregate_fold_evaluations() {
    local split_root="$1"
    local split_name="$2"
    local expected_samples="$3"
    local strategy="$4"

    # FOLDWISE_DUAL_TEST_AGGREGATION_V1
    python - \
        "${split_root}" \
        "${split_name}" \
        "${expected_samples}" \
        "${STAGE}" \
        "${strategy}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

split_root = Path(sys.argv[1])
split_name = sys.argv[2]
expected_samples = int(sys.argv[3])
stage = int(sys.argv[4])
strategy = sys.argv[5]

prediction_filename = "test_set_features_complete_with_pretrained.csv"

identity_candidates = [
    "样本索引",
    "站点ID",
    "日期",
    "DOY",
    "行列号_row",
    "行列号_col",
    "原始经度",
    "原始纬度",
]

metric_columns = [
    "r",
    "nse",
    "rmse_mm",
    "mae_mm",
    "bias_mm",
    "rmse_obs_ge50_mm",
    "bias_obs_ge80_mm",
    "slope",
    "intercept_mm",
    "std_ratio",
    "pred_std_mm",
]


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def compute_metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)

    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]

    if target.size == 0:
        raise RuntimeError("没有有效目标/预测值")

    error = prediction - target
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))

    target_std = float(np.std(target))
    pred_std = float(np.std(prediction))
    std_ratio = pred_std / target_std if target_std > 1e-12 else float("nan")

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    nse = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    if target.size > 1 and target_std > 1e-12 and pred_std > 1e-12:
        r = float(np.corrcoef(target, prediction)[0, 1])
    else:
        r = float("nan")

    centered_target = target - np.mean(target)
    slope_denom = float(np.sum(centered_target ** 2))
    if slope_denom > 1e-12:
        slope = float(
            np.sum(centered_target * (prediction - np.mean(prediction)))
            / slope_denom
        )
        intercept = float(np.mean(prediction) - slope * np.mean(target))
    else:
        slope = float("nan")
        intercept = float("nan")

    ge50 = target >= 50.0
    ge80 = target >= 80.0

    rmse_ge50 = (
        float(np.sqrt(np.mean(error[ge50] ** 2)))
        if np.any(ge50)
        else float("nan")
    )
    bias_ge80 = (
        float(np.mean(error[ge80]))
        if np.any(ge80)
        else float("nan")
    )

    collapsed = bool(
        pred_std < 1.0
        or (
            np.isfinite(std_ratio)
            and std_ratio < 0.05
            and np.isfinite(slope)
            and abs(slope) < 0.05
        )
    )

    return {
        "n_samples": int(target.size),
        "r": safe_float(r),
        "nse": safe_float(nse),
        "rmse_mm": rmse,
        "mae_mm": mae,
        "bias_mm": bias,
        "rmse_obs_ge50_mm": safe_float(rmse_ge50),
        "bias_obs_ge80_mm": safe_float(bias_ge80),
        "n_obs_ge50": int(np.sum(ge50)),
        "n_obs_ge80": int(np.sum(ge80)),
        "slope": safe_float(slope),
        "intercept_mm": safe_float(intercept),
        "std_ratio": safe_float(std_ratio),
        "pred_std_mm": pred_std,
        "target_std_mm": target_std,
        "collapsed": collapsed,
    }


def canonical_identity(frame, columns):
    if not columns:
        return pd.Series(
            np.arange(len(frame), dtype=np.int64).astype(str),
            index=frame.index,
        )
    values = frame[columns].copy()
    for column in columns:
        values[column] = values[column].fillna("").astype(str)
    return values.agg("||".join, axis=1)


rows = []
long_frames = []
prediction_arrays = []
base_target = None
base_identity = None
base_frame = None
identity_columns = None

for fold in range(1, 11):
    fold_dir = split_root / f"fold_{fold:02d}"
    prediction_path = fold_dir / prediction_filename
    model_path_file = fold_dir / "model_path.txt"

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Fold {fold}缺少逐样本预测CSV: {prediction_path}"
        )

    frame = pd.read_csv(prediction_path)

    required = ["站点SWE_raw", "微调模型预测_raw"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"Fold {fold}预测CSV缺少列: {missing}; "
            f"现有列={list(frame.columns)}"
        )

    if len(frame) != expected_samples:
        raise RuntimeError(
            f"Fold {fold}样本数不一致: "
            f"actual={len(frame)}, expected={expected_samples}"
        )

    target = pd.to_numeric(
        frame["站点SWE_raw"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    prediction = pd.to_numeric(
        frame["微调模型预测_raw"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if identity_columns is None:
        identity_columns = [
            column for column in identity_candidates
            if column in frame.columns
        ]

    identity = canonical_identity(frame, identity_columns)

    if base_target is None:
        base_target = target.copy()
        base_identity = identity.to_numpy(dtype=str)
        base_frame = frame.copy()
    else:
        if not np.array_equal(identity.to_numpy(dtype=str), base_identity):
            raise RuntimeError(
                f"Fold {fold}样本顺序/身份与Fold 1不一致，拒绝直接集成"
            )
        if not np.allclose(
            target,
            base_target,
            equal_nan=True,
            atol=1e-8,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Fold {fold}目标值与Fold 1不一致，拒绝集成"
            )

    metrics = compute_metrics(target, prediction)
    metrics["fold"] = fold
    metrics["stage"] = stage
    metrics["strategy"] = strategy
    metrics["split"] = split_name
    metrics["model_path"] = (
        model_path_file.read_text(encoding="utf-8").strip()
        if model_path_file.is_file()
        else None
    )
    rows.append(metrics)
    prediction_arrays.append(prediction)

    keep_columns = list(identity_columns)
    for column in ["站点SWE_raw", "FusedSWE_raw"]:
        if column in frame.columns and column not in keep_columns:
            keep_columns.append(column)

    long_frame = frame[keep_columns].copy()
    long_frame.insert(0, "fold", fold)
    long_frame["prediction_mm"] = prediction
    long_frames.append(long_frame)

metrics_frame = pd.DataFrame(rows).sort_values("fold")
metrics_frame.to_csv(
    split_root / "fold_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.concat(long_frames, ignore_index=True).to_csv(
    split_root / "predictions_long.csv",
    index=False,
    encoding="utf-8-sig",
)

prediction_matrix = np.vstack(prediction_arrays)
ensemble_prediction = np.nanmean(prediction_matrix, axis=0)
ensemble_std = np.nanstd(prediction_matrix, axis=0)
ensemble_min = np.nanmin(prediction_matrix, axis=0)
ensemble_max = np.nanmax(prediction_matrix, axis=0)

ensemble_metrics = compute_metrics(base_target, ensemble_prediction)
ensemble_metrics.update({
    "stage": stage,
    "strategy": strategy,
    "split": split_name,
    "ensemble": "mean_of_10_fold_models",
    "n_models": 10,
})

ensemble_columns = list(identity_columns)
for column in ["站点SWE_raw", "FusedSWE_raw"]:
    if column in base_frame.columns and column not in ensemble_columns:
        ensemble_columns.append(column)

ensemble_frame = base_frame[ensemble_columns].copy()
ensemble_frame["ensemble_prediction_mean_mm"] = ensemble_prediction
ensemble_frame["ensemble_prediction_std_mm"] = ensemble_std
ensemble_frame["ensemble_prediction_min_mm"] = ensemble_min
ensemble_frame["ensemble_prediction_max_mm"] = ensemble_max
ensemble_frame.to_csv(
    split_root / "ensemble_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

(split_root / "ensemble_metrics.json").write_text(
    json.dumps(
        ensemble_metrics,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)

summary = {
    "stage": stage,
    "strategy": strategy,
    "split": split_name,
    "n_folds": 10,
    "expected_samples_per_fold": expected_samples,
    "fold_selection": "none",
    "test_use": "evaluation_only",
    "metrics": {},
    "collapsed_folds": [
        int(row["fold"])
        for row in rows
        if bool(row.get("collapsed", False))
    ],
    "ensemble_metrics": ensemble_metrics,
}

for metric in metric_columns:
    values = pd.to_numeric(
        metrics_frame[metric],
        errors="coerce",
    ).dropna().to_numpy(dtype=np.float64)

    summary["metrics"][metric] = {
        "n": int(values.size),
        "mean": safe_float(np.mean(values)) if values.size else None,
        "std": (
            safe_float(np.std(values, ddof=1))
            if values.size > 1
            else 0.0 if values.size == 1 else None
        ),
        "min": safe_float(np.min(values)) if values.size else None,
        "max": safe_float(np.max(values)) if values.size else None,
    }

(split_root / "fold_summary.json").write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ),
    encoding="utf-8",
)

pd.DataFrame([
    {"metric": metric, **values}
    for metric, values in summary["metrics"].items()
]).to_csv(
    split_root / "fold_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print()
print("=" * 80)
print(
    f"✅ M{stage} {strategy} | {split_name} "
    "10-fold逐折测试汇总完成"
)
print("=" * 80)
for metric in [
    "r",
    "nse",
    "rmse_mm",
    "mae_mm",
    "bias_mm",
    "rmse_obs_ge50_mm",
    "bias_obs_ge80_mm",
    "slope",
    "std_ratio",
]:
    item = summary["metrics"][metric]
    if item["mean"] is not None:
        print(
            f"{metric:<22s}: "
            f"{item['mean']:.4f} ± {item['std']:.4f}"
        )
print(f"collapsed folds       : {summary['collapsed_folds']}")
print(
    "ensemble RMSE/MAE/R  : "
    f"{ensemble_metrics['rmse_mm']:.4f} / "
    f"{ensemble_metrics['mae_mm']:.4f} / "
    f"{ensemble_metrics['r']}"
)
print(f"结果目录              : {split_root}")
print("=" * 80)
PY
}

# BALANCED_STATION_CV10_V1
# 一套确定性、轻量平衡的站点fold清单供Frozen和所有策略共同复用。
python "${BALANCED_FOLD_TOOL}" \
    --station-csv "${INTERNAL_DATA}" \
    --output "${FOLD_MANIFEST}" \
    --n-splits 10 \
    --high-threshold-mm 80 \
    "${BALANCED_INCLUDE_ARGS[@]}"

echo "✅ 正式内部评估fold清单: ${FOLD_MANIFEST}"

# ----------------------------------------------------------------------
# 1. Frozen：同一平衡站点10折 + 外部一次
# ----------------------------------------------------------------------
if [[ "${RUN_FROZEN}" == "1" ]]; then
    echo
    echo "================================================================================"
    echo "M${STAGE} Frozen：平衡站点10折OOF + 外部一次"
    echo "================================================================================"

    OUT="${RUN_ROOT}/frozen_station_cv10" \
    M0_MODEL="${PRETRAINED_MODEL}" \
    ROOT="${ROOT}" \
    INCLUDE_FIXED_INTERNAL_IN_CV="${INCLUDE_FIXED_INTERNAL_IN_CV}" \
    bash "${FROZEN_CV_RUNNER}"
fi

# ----------------------------------------------------------------------
# 2. 五种微调策略
#    每一折：inner fold选checkpoint -> outer fold仅OOF -> 再进入下一折。
#    外部987条：十折全部结束后只报告一次预声明ensemble结果。
# ----------------------------------------------------------------------
read -r -a STRATEGY_LIST <<< "${STRATEGIES}"

for STRATEGY in "${STRATEGY_LIST[@]}"; do
    if [[ -z "${STRATEGY_LR[${STRATEGY}]+x}" ]]; then
        echo "❌ 未知策略: ${STRATEGY}"
        exit 2
    fi

    FT_LR="${STRATEGY_LR[${STRATEGY}]}"
    STRATEGY_DIR="${RUN_ROOT}/${STRATEGY}"
    mkdir -p "${STRATEGY_DIR}"

    echo
    echo "================================================================================"
    echo "M${STAGE} 微调策略: ${STRATEGY}"
    echo "策略级学习率:      ${FT_LR}"
    if [[ "${OVERFIT_TRAIN_ONLY}" == "1" ]]; then
        echo "诊断模式:          Fold 1的8-fold完整训练池同集过拟合"
        echo "inner/outer:        均不参与"
        echo "调度/早停:         warmup后固定LR；无Plateau、无early stopping"
        echo "正式结果声明:      不是CV、OOF或外部测试结果"
    elif [[ "${INNER_PILOT_ONLY}" == "1" ]]; then
        echo "Pilot模式:          8 folds训练 + 1 Inner选模"
        echo "Outer:              保留但不加载、不预测、不输出OOF"
        echo "正式结果声明:      仅用于冻结训练配置，不是OOF结果"
    else
        echo "内部评估:          Nested站点10折；inner选模，outer仅OOF"
        echo "内部样本:          ${INTERNAL_SPLIT_DESCRIPTION}"
        echo "外部987条:         十折模型预声明ensemble，只报告一次"
    fi
    echo "================================================================================"

    # 只有同时存在10个checkpoint、完整OOF和正确split_method时才允许复用。
    mapfile -t EXISTING_FOLD_MODELS < <(
        find "${STRATEGY_DIR}" \
            -mindepth 2 -maxdepth 4 \
            -type f \
            -name "cv_fold_*_best_model.pth" \
            -print 2>/dev/null | sort -V
    )

    REUSE_CV=0
    if [[ "${INNER_PILOT_ONLY}" == "0" &&
          "${#EXISTING_FOLD_MODELS[@]}" -eq 10 ]]; then
        EXISTING_RUN_DIR="$(dirname "${EXISTING_FOLD_MODELS[0]}")"
        if [[ -f "${EXISTING_RUN_DIR}/station_cv10_oof_predictions.csv" &&
              -f "${EXISTING_RUN_DIR}/cv_station_level_aggregated_results.json" ]]; then
            if python - \
                "${EXISTING_RUN_DIR}/cv_station_level_aggregated_results.json" \
                "${STRATEGY}" \
                "${FUSION_FT_HEAD_LR}" \
                "${FUSION_FT_TRANSFORMER_LR}" \
                "${FUSION_FT_WARMUP_EPOCHS}" \
                "${FUSION_FT_UNFREEZE_INTERVAL}" \
                "${FUSION_FT_MAX_UNFROZEN_LAYERS}" \
                "${FUSION_FT_PLATEAU_PATIENCE}" \
                "${FUSION_FT_MIN_EPOCHS}" \
                "${FUSION_FT_PATIENCE}" \
                "${EXPECTED_NESTED_SAMPLES}" \
                "${INCLUDE_FIXED_INTERNAL_IN_CV}" \
                "${FUSION_FT_TREND_EARLY_STOP}" \
                "${STATION_AUGMENTATION}" \
                "${COORD_JITTER_STD_DEG}" \
                "${MICROWAVE_NOISE_STD}" \
                "${COORD_MASK_PROB}" \
                "${DISABLE_FROZEN_RELATIVE_GATE}" <<'PY_CHECK'
import json, sys
path = sys.argv[1]
strategy = sys.argv[2]
head_lr = float(sys.argv[3])
transformer_lr = float(sys.argv[4])
warmup_epochs = int(sys.argv[5])
unfreeze_interval = int(sys.argv[6])
max_unfrozen_layers = int(sys.argv[7])
plateau_patience = int(sys.argv[8])
min_epochs = int(sys.argv[9])
early_stop_patience = int(sys.argv[10])
expected_nested_samples = int(sys.argv[11])
include_fixed_internal = sys.argv[12] == "1"
trend_early_stop = sys.argv[13] == "1"
station_augmentation = sys.argv[14] == "1"
coordinate_jitter_std_deg = float(sys.argv[15])
microwave_noise_std = float(sys.argv[16])
coordinate_mask_prob = float(sys.argv[17])
disable_frozen_relative_gate = sys.argv[18] == "1"
data = json.load(open(path, encoding="utf-8"))
assert data.get("split_method") == "deterministic_balanced_greedy_v1", data.get("split_method")
assert data.get("randomized") is False
policy = data.get("evaluation_policy", {})
assert policy.get("internal") == "nested_inner_selection_outer_heldout_oof"
assert (
    policy.get("internal_cv_samples_evaluated_once")
    == expected_nested_samples
)
assert policy.get("outer_oof_used_for_checkpoint_selection") is False
assert (
    policy.get("fixed_internal_1000_merged_into_cv")
    is include_fixed_internal
)
expected_checkpoint_gate = (
    "disabled_composite_selection_only"
    if disable_frozen_relative_gate
    else "frozen_relative_non_degradation"
)
assert policy.get("checkpoint_gate") == expected_checkpoint_gate
assert policy.get("admissible_checkpoint_ranking") == "minimum_composite_selection_score"
if strategy == "fusion_ft":
    optimization = data.get("optimization_policy", {})
    assert optimization.get("fusion_ft_progressive_unfreeze") is True
    assert optimization.get("fusion_ft_head_lr") == head_lr
    assert optimization.get("fusion_ft_transformer_lr") == transformer_lr
    assert optimization.get("fusion_ft_warmup_epochs") == warmup_epochs
    assert optimization.get("fusion_ft_unfreeze_interval") == unfreeze_interval
    assert optimization.get("fusion_ft_max_unfrozen_layers") == max_unfrozen_layers
    assert optimization.get("fusion_ft_plateau_patience") == plateau_patience
    assert optimization.get("fusion_ft_sampling") == "natural_random_without_forced_tail_quota"
    assert optimization.get("fusion_ft_tail_weights") == {
        "ge20": 1.2,
        "ge50": 1.5,
        "ge80": 2.0,
    }
    assert optimization.get("fusion_ft_high_bias_weight") == 0.0
    assert optimization.get("fusion_ft_ccc_weight") == 0.005
    assert optimization.get("fusion_ft_variance_weight") == 0.0
    assert optimization.get("fusion_ft_patience") == early_stop_patience
    assert optimization.get("fusion_ft_min_epochs_before_early_stop") == min_epochs
    assert optimization.get("fusion_ft_trend_early_stopping") is trend_early_stop
    augmentation = optimization.get(
        "train_only_station_augmentation",
        {},
    )
    assert augmentation.get("enabled") is station_augmentation
    assert (
        augmentation.get("coordinate_jitter_std_deg")
        == coordinate_jitter_std_deg
    )
    assert (
        augmentation.get("microwave_noise_std")
        == microwave_noise_std
    )
    assert (
        augmentation.get("coordinate_mask_prob")
        == coordinate_mask_prob
    )
    assert augmentation.get("inner_outer_audit_augmented") is False
    assert optimization.get("admissible_checkpoint_ranking") == "minimum_composite_selection_score"
print(
    "✅ 已有结果符合nested OOF + "
    + (
        "无Frozen gate综合选模协议"
        if disable_frozen_relative_gate
        else "Frozen-relative综合选模协议"
    )
)
PY_CHECK
            then
                REUSE_CV=1
            fi
        fi
    fi

    if [[ "${REUSE_CV}" == "1" ]]; then
        echo "✅ ${STRATEGY}平衡站点10折已完成，复用现有OOF结果"
    else
        if [[ "${#EXISTING_FOLD_MODELS[@]}" -gt 0 ]]; then
            echo "⚠ 发现旧协议/不完整fold模型，不会复用；新运行写入新的时间戳目录"
        fi

        # main_tune.py内部现在执行：
        # Fold1 inner选模->Fold1 outer OOF评估->清理->Fold2...
        # 不再等全部fold结束后对固定1000条逐模型测试。
        python "${MAIN}" \
            --mode fine_tune \
            --pretrained_model "${PRETRAINED_MODEL}" \
            --station_data_path "${INTERNAL_DATA}" \
            --save_dir "${STRATEGY_DIR}" \
            --fine_tune_epochs "${FINE_TUNE_EPOCHS}" \
            --fine_tune_lr "${FT_LR}" \
            --freeze_backbone \
            --freeze_strategy "${STRATEGY}" \
            --fusion_ft_head_lr "${FUSION_FT_HEAD_LR}" \
            --fusion_ft_transformer_lr "${FUSION_FT_TRANSFORMER_LR}" \
            --fusion_ft_warmup_epochs "${FUSION_FT_WARMUP_EPOCHS}" \
            --fusion_ft_unfreeze_interval "${FUSION_FT_UNFREEZE_INTERVAL}" \
            --fusion_ft_max_unfrozen_layers "${FUSION_FT_MAX_UNFROZEN_LAYERS}" \
            --fusion_ft_plateau_patience "${FUSION_FT_PLATEAU_PATIENCE}" \
            --fusion_ft_min_epochs_before_early_stop "${FUSION_FT_MIN_EPOCHS}" \
            --fusion_ft_patience "${FUSION_FT_PATIENCE}" \
            "${TREND_EARLY_STOP_ARGS[@]}" \
            "${FROZEN_RELATIVE_GATE_ARGS[@]}" \
            --fusion_ft_ccc_weight 0.005 \
            --fusion_ft_variance_weight 0 \
            --use_high_swe_weight \
            --num_workers "${NUM_WORKERS}" \
            "${COMMON_ARGS[@]}"
    fi

    if [[ "${OVERFIT_TRAIN_ONLY}" == "1" ]]; then
        echo "✅ ${STRATEGY}完整训练集过拟合诊断已完成"
        echo "   该模式不检查fold模型，不运行OOF和外部评估"
        continue
    fi
    if [[ "${INNER_PILOT_ONLY}" == "1" ]]; then
        echo "✅ ${STRATEGY}单Inner pilot已完成"
        echo "   Outer未加载、未预测；不检查10折模型，不运行外部评估"
        continue
    fi

    mapfile -t FOLD_MODELS < <(
        find "${STRATEGY_DIR}" \
            -mindepth 2 -maxdepth 4 \
            -type f \
            -name "cv_fold_*_best_model.pth" \
            -print 2>/dev/null | sort -V
    )

    if [[ "${#FOLD_MODELS[@]}" -ne 10 ]]; then
        echo "❌ ${STRATEGY}的fold-specific best模型数量异常"
        echo "   expected=10, actual=${#FOLD_MODELS[@]}"
        printf '   %s\n' "${FOLD_MODELS[@]}"
        exit 1
    fi

    UNIQUE_MODEL_PARENT_COUNT="$(
        printf '%s\n' "${FOLD_MODELS[@]}" \
            | xargs -n1 dirname \
            | sort -u \
            | wc -l
    )"
    if [[ "${UNIQUE_MODEL_PARENT_COUNT}" -ne 1 ]]; then
        echo "❌ 10个fold模型不在同一个训练运行目录，拒绝混用"
        exit 1
    fi

    FOLD_MODEL_RUN_DIR="$(dirname "${FOLD_MODELS[0]}")"
    printf '%s\n' "${FOLD_MODEL_RUN_DIR}" \
        > "${STRATEGY_DIR}/fold_model_run_dir.txt"

    for required_result in \
        "${FOLD_MODEL_RUN_DIR}/station_cv10_oof_predictions.csv" \
        "${FOLD_MODEL_RUN_DIR}/station_cv10_fold_metrics.csv" \
        "${FOLD_MODEL_RUN_DIR}/cv_station_level_aggregated_results.json" \
        "${FOLD_MODEL_RUN_DIR}/cv_station_level_boxplot.png"
    do
        [[ -f "${required_result}" ]] || {
            echo "❌ 缺少内部站点CV结果: ${required_result}"
            exit 1
        }
    done

    echo "✅ ${STRATEGY}内部OOF完成: ${FOLD_MODEL_RUN_DIR}"

    if [[ "${RUN_EXTERNAL}" == "1" ]]; then
        EXTERNAL_RUNTIME_DIR="${STRATEGY_DIR}/runtime_inputs/external"
        EXTERNAL_RUNTIME_FILE="${EXTERNAL_RUNTIME_DIR}/external_evaluation_input.csv"
        mkdir -p "${EXTERNAL_RUNTIME_DIR}"
        cp -p "${EXTERNAL_DATA}" "${EXTERNAL_RUNTIME_FILE}"

        python "${EXTERNAL_ENSEMBLE_TOOL}" \
            --root "${ROOT}" \
            --station-csv "${EXTERNAL_RUNTIME_FILE}" \
            --checkpoint-dir "${FOLD_MODEL_RUN_DIR}" \
            --output-dir "${STRATEGY_DIR}/external_987_once" \
            --normalization-config "${NORMALIZATION_CONFIG}" \
            --cache-dir "${ROOT}/shared_cache" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${EVAL_NUM_WORKERS}" \
            --expected-external-count 987
    fi
done

if [[ "${OVERFIT_TRAIN_ONLY}" == "1" ]]; then
    echo
    echo "================================================================================"
    echo "✅ M${STAGE}完整训练集过拟合诊断全部完成"
    echo "结果目录: ${RUN_ROOT}"
    echo "日志文件: ${LOG_FILE}"
    echo "⚠ 此结果仅用于拟合能力诊断，不是正式Nested CV/OOF结果"
    echo "================================================================================"
    exit 0
fi

if [[ "${INNER_PILOT_ONLY}" == "1" ]]; then
    echo
    echo "================================================================================"
    echo "✅ M${STAGE}单Inner pilot全部完成"
    echo "结果目录: ${RUN_ROOT}"
    echo "日志文件: ${LOG_FILE}"
    echo "🔒 Outer fold未加载、未预测；没有生成OOF或外部测试结果"
    echo "================================================================================"
    exit 0
fi

python - \
    "${RUN_ROOT}" \
    "${STAGE}" \
    "${PRETRAINED_MODEL}" \
    "${SEED}" \
    "${BATCH_SIZE}" \
    "${FINE_TUNE_EPOCHS}" \
    "${STRATEGIES}" \
    "${INTERNAL_DATA}" \
    "${EXTERNAL_DATA}" \
    "${NORMALIZATION_CONFIG}" \
    "${FUSION_FT_MAX_UNFROZEN_LAYERS}" \
    "${DISABLE_FROZEN_RELATIVE_GATE}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

(
    run_root,
    stage,
    pretrained_model,
    seed,
    batch_size,
    epochs,
    strategies,
    internal_data,
    external_data,
    normalization,
    fusion_ft_max_unfrozen_layers,
    disable_frozen_relative_gate,
) = sys.argv[1:]

payload = {
    "created_at": datetime.now().isoformat(),
    "stage": int(stage),
    "pretrained_model": pretrained_model,
    "seed": int(seed),
    "batch_size": int(batch_size),
    "fine_tune_epochs": int(epochs),
    "strategies": strategies.split(),
    "internal_data": internal_data,
    "external_data": external_data,
    "normalization_config": normalization,
    "fusion_ft_max_unfrozen_layers": int(
        fusion_ft_max_unfrozen_layers
    ),
    "frozen_relative_gate_enabled": (
        disable_frozen_relative_gate != "1"
    ),
    "checkpoint_selection": (
        "minimum_composite_selection_score_without_frozen_gate"
        if disable_frozen_relative_gate == "1"
        else "frozen_relative_gate_then_composite_selection_score"
    ),
    "product_correction": False,
    "mixed_replay": False,
    "fine_tuned_test_policy": (
        "Each fold is evaluated immediately on its own held-out stations; "
        "the ten OOF parts cover the internal CV pool exactly once."
    ),
    "internal_test_definition": (
        "Deterministic balanced station-level Nested 10-fold OOF over all "
        "7936 internal samples; the old 1000-sample holdout is merged back "
        "into the development pool."
    ),
    "external_test_definition": (
        "Fixed 987-sample external set; one official result from the "
        "predeclared mean prediction of all ten fold models."
    ),
}

path = Path(run_root) / "stage_run_manifest.json"
path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"✅ 运行元数据已保存: {path}")
PY

echo
echo "================================================================================"
echo "✅ M${STAGE}平衡站点10折OOF与外部一次集成评估全部完成"
echo "结果目录: ${RUN_ROOT}"
echo "日志文件: ${LOG_FILE}"
echo "================================================================================"
