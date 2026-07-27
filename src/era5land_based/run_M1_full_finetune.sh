#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# M1 完整微调流程
#
# 包含：
#   1. M1 Frozen：内部1000条 + 外部987条
#   2. 5种微调策略，各自进行10折站点CV：
#      fusion_ft
#      point_ft
#      spatial_ft
#      partial
#      none
#   3. 每种策略完成后进行外部987条测试
#
# 用法：
#   bash /root/autodl-tmp/run_M1_full_finetune.sh --check
#   bash /root/autodl-tmp/run_M1_full_finetune.sh
#
# 如自动识别错了，可显式指定：
#   M1_MODEL=/path/to/final_model.pth bash run_M1_full_finetune.sh --check
#   M1_MODEL=/path/to/final_model.pth bash run_M1_full_finetune.sh

set -Eeuo pipefail
set -o pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
EXPERIMENTS="${ROOT}/experiments"
RUNNER="${ROOT}/run_progressive_finetune_stage.sh"
MAIN="${ROOT}/main_tune.py"
STATION="${ROOT}/data_station_online_swe.py"
MANIFEST_DIR="${ROOT}/shared_cache/progressive_finetune"
NORMALIZATION_CONFIG="${ROOT}/shared_cache/progressive_pretrain_normalization.json"

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
elif [[ -n "${1:-}" ]]; then
    echo "❌ 未知参数: $1"
    echo "用法: bash $0 [--check]"
    exit 2
fi

# 完整微调配置
STRATEGIES="${STRATEGIES:-fusion_ft point_ft spatial_ft partial none}"
RUN_FROZEN="${RUN_FROZEN:-1}"
RUN_EXTERNAL="${RUN_EXTERNAL:-1}"
FINE_TUNE_EPOCHS="${FINE_TUNE_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-43}"

for required in \
    "${RUNNER}" \
    "${MAIN}" \
    "${STATION}" \
    "${NORMALIZATION_CONFIG}" \
    "${MANIFEST_DIR}/internal_progressive_station.csv" \
    "${MANIFEST_DIR}/internal_test_approximately_1000.csv" \
    "${MANIFEST_DIR}/external_evaluation_input.csv" \
    "${MANIFEST_DIR}/progressive_finetune_manifest_summary.json"
do
    if [[ ! -f "${required}" ]]; then
        echo "❌ 缺少必要文件: ${required}"
        exit 1
    fi
done

# 关键补丁检查
declare -a REQUIRED_MAIN_MARKERS=(
    "PROGRESSIVE_FINETUNE_PASS_NORMALIZATION_V1"
    "PROGRESSIVE_STABLE_RUNTIME_MANIFEST_V1"
    "PROGRESSIVE_CV_RELOAD_BEST_V1"
)

for marker in "${REQUIRED_MAIN_MARKERS[@]}"; do
    if ! grep -q "${marker}" "${MAIN}"; then
        echo "❌ main_tune.py 缺少补丁标记: ${marker}"
        exit 1
    fi
done

declare -a REQUIRED_STATION_MARKERS=(
    "PROGRESSIVE_FINETUNE_NORMALIZATION_V1"
    "STATION_PRETRAIN_TIME_INTERSECTION_2015_2018_V1"
)

for marker in "${REQUIRED_STATION_MARKERS[@]}"; do
    if ! grep -q "${marker}" "${STATION}"; then
        echo "❌ data_station_online_swe.py 缺少补丁标记: ${marker}"
        exit 1
    fi
done

# 避免并行启动另一套训练
if pgrep -af 'main_tune.py|run_progressive_finetune_stage.sh' \
    | grep -vE 'pgrep|run_M1_full_finetune.sh' >/dev/null
then
    echo "❌ 检测到已有微调/训练进程："
    pgrep -af 'main_tune.py|run_progressive_finetune_stage.sh' || true
    exit 1
fi

# 自动识别M1正式预训练模型
M1_MODEL_RESOLVED="$(
python - "${EXPERIMENTS}" "${M1_MODEL:-}" <<'PY'
import sys
from pathlib import Path
import torch

experiments = Path(sys.argv[1])
override = sys.argv[2].strip()
expected_stage = 1

def validate(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"文件不存在: {path}")

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise RuntimeError("checkpoint不是字典")
    if "model_state_dict" not in ckpt:
        raise RuntimeError("缺少model_state_dict")

    config = ckpt.get("config", {})
    actual_stage = config.get("incremental_stage")

    if actual_stage is None:
        raise RuntimeError("config中缺少incremental_stage")
    if int(actual_stage) != expected_stage:
        raise RuntimeError(
            f"incremental_stage={actual_stage}，预期={expected_stage}"
        )

    return path

if override:
    try:
        print(validate(Path(override)))
    except Exception as exc:
        raise SystemExit(f"❌ 指定M1_MODEL检查失败:\n{exc}")
    raise SystemExit(0)

candidates = []

for path in experiments.rglob("final_model.pth"):
    path_lower = str(path).lower()

    # 排除微调输出目录
    if "_finetune_" in path_lower or "/finetune" in path_lower:
        continue

    try:
        candidates.append(validate(path))
    except Exception:
        continue

candidates = sorted(
    set(candidates),
    key=lambda p: (p.stat().st_mtime, str(p)),
    reverse=True,
)

if not candidates:
    raise SystemExit(
        "❌ 未找到config.incremental_stage=1的正式final_model.pth"
    )

selected = candidates[0]

print(
    f"发现{len(candidates)}个有效M1模型，默认选择最近修改的模型:",
    file=sys.stderr,
)

for index, path in enumerate(candidates, start=1):
    suffix = "  <-- selected" if path == selected else ""
    print(f"  {index}. {path}{suffix}", file=sys.stderr)

print(selected)
PY
)"

if [[ ! -f "${M1_MODEL_RESOLVED}" ]]; then
    echo "❌ M1模型解析失败: ${M1_MODEL_RESOLVED}"
    exit 1
fi

echo
echo "================================================================================"
echo "M1完整微调运行检查"
echo "================================================================================"
echo "M1正式模型:        ${M1_MODEL_RESOLVED}"
echo "微调策略:          ${STRATEGIES}"
echo "Frozen测试:        ${RUN_FROZEN}"
echo "外部987条测试:     ${RUN_EXTERNAL}"
echo "微调轮数:          ${FINE_TUNE_EPOCHS}"
echo "批次大小:          ${BATCH_SIZE}"
echo "workers:           ${NUM_WORKERS}"
echo "随机种子:          ${SEED}"
echo "内部CV池:          ${MANIFEST_DIR}/internal_progressive_station.csv"
echo "内部固定测试:      ${MANIFEST_DIR}/internal_test_approximately_1000.csv"
echo "外部固定测试:      ${MANIFEST_DIR}/external_evaluation_input.csv"
echo "================================================================================"

echo
echo "已有共享站点特征缓存:"
find "${ROOT}/shared_cache" \
    -maxdepth 3 \
    -type f \
    -name 'station_dataset_features_shared_station_features_*.pkl' \
    -printf '  %p (%s bytes)\n' \
    2>/dev/null || true

if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo
    echo "✅ 检查通过，尚未启动M1微调。"
    exit 0
fi

export STAGE=1
export PRETRAINED_MODEL="${M1_MODEL_RESOLVED}"
export STRATEGIES
export RUN_FROZEN
export RUN_EXTERNAL
export FINE_TUNE_EPOCHS
export BATCH_SIZE
export NUM_WORKERS
export SEED

echo
echo "################################################################################"
echo "开始M1完整微调"
echo "################################################################################"

bash "${RUNNER}"

echo
echo "================================================================================"
echo "✅ M1完整微调全部完成"
echo "================================================================================"
