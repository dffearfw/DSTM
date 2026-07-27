#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp"
RUN="${ROOT}/experiments/progressive_M0_finetune_20260721_102153"
NONE_DIR="${RUN}/none"
LAUNCHER="${ROOT}/run_progressive_finetune_stage.sh"
PRETRAINED_MODEL="${ROOT}/experiments/pretrain_stage0_station_20260714_215604/final_model.pth"
REFERENCE_SPLIT="${ROOT}/shared_cache/progressive_finetune/station_cv_fold_splits_seed43_reference.json"

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
fi

echo "================================================================================"
echo "M0 Full FT（none）恢复检查"
echo "================================================================================"
echo "原实验目录:  ${RUN}"
echo "none目录:    ${NONE_DIR}"
echo "Stage0模型:  ${PRETRAINED_MODEL}"
echo "启动脚本:    ${LAUNCHER}"
echo "================================================================================"

for required in \
    "${RUN}" \
    "${NONE_DIR}" \
    "${LAUNCHER}" \
    "${PRETRAINED_MODEL}"
do
    if [[ ! -e "${required}" ]]; then
        echo "❌ 必需路径不存在: ${required}"
        exit 1
    fi
done

if pgrep -af 'main_tune.py|run_progressive_finetune_stage.sh' \
    | grep -vE 'pgrep|resume_M0_none.sh' >/dev/null
then
    echo "❌ 检测到训练进程仍在运行："
    pgrep -af 'main_tune.py|run_progressive_finetune_stage.sh' || true
    exit 1
fi

mapfile -t ACTIVE_RUNS < <(
    find "${NONE_DIR}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name 'swe_*_fine_tune_none_*' \
        -print \
        | sort
)

if [[ "${#ACTIVE_RUNS[@]}" -gt 1 ]]; then
    echo "❌ none目录中发现多个子实验，拒绝自动处理："
    printf '   %s\n' "${ACTIVE_RUNS[@]}"
    exit 1
fi

OLD_RUN=""

if [[ "${#ACTIVE_RUNS[@]}" -eq 1 ]]; then
    OLD_RUN="${ACTIVE_RUNS[0]}"

    echo
    echo "发现中断子实验:"
    echo "  ${OLD_RUN}"

    if find "${OLD_RUN}" \
        -maxdepth 2 \
        -type f \
        -name 'cv_10fold_panel_matrix.png' \
        -print -quit \
        | grep -q .
    then
        echo "❌ 该none子实验已经存在十折面板，可能已经完整结束。"
        exit 1
    fi

    echo
    echo "现有逐折散点图:"
    find "${OLD_RUN}" \
        -maxdepth 1 \
        -type f \
        -name 'fine_tune_scatter_chinese_fold_*.png' \
        -printf '  %f\n' \
        | sort -V

    if [[ -f "${OLD_RUN}/config.json" ]]; then
        python - "${OLD_RUN}/config.json" "${PRETRAINED_MODEL}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
expected_model = str(Path(sys.argv[2]).resolve())

config = json.loads(config_path.read_text(encoding="utf-8"))

checks = {
    "freeze_strategy": "none",
    "cv_mode": "station_cv",
    "seed": 43,
    "batch_size": 32,
    "fine_tune_epochs": 50,
}

errors = []

for key, expected in checks.items():
    actual = config.get(key)
    if actual != expected:
        errors.append(
            f"{key}: 期望={expected!r}, 实际={actual!r}"
        )

actual_model = config.get("pretrained_model")
if actual_model:
    actual_model = str(Path(actual_model).resolve())
    if actual_model != expected_model:
        errors.append(
            "pretrained_model不一致："
            f"\n  期望={expected_model}"
            f"\n  实际={actual_model}"
        )

if errors:
    raise SystemExit(
        "❌ 中断实验配置检查失败：\n- "
        + "\n- ".join(errors)
    )

print("✅ 中断实验配置检查通过")
print("   freeze_strategy: none")
print("   cv_mode:         station_cv")
print("   seed:            43")
print("   batch_size:      32")
print("   epochs:          50")
PY
    fi
else
    echo
    echo "none目录目前没有活动子实验。"
fi

echo
echo "恢复方案:"
echo "  1. 保留 frozen、fusion_ft、point_ft、spatial_ft、partial"
echo "  2. 将中断的none子实验移动到备份目录"
echo "  3. 仅重新运行none策略"
echo "  4. none的十折从Fold 1重新开始"
echo "  5. 使用原运行根目录和原固定数据划分"
echo

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "✅ 检查完成，尚未移动文件，也没有启动训练。"
    exit 0
fi

STAMP="$(date +'%Y%m%d_%H%M%S')"

if [[ -n "${OLD_RUN}" ]]; then
    BACKUP_DIR="${RUN}/interrupted_none_backup_${STAMP}"
    mkdir -p "${BACKUP_DIR}"

    mv "${OLD_RUN}" "${BACKUP_DIR}/"

    echo "✅ 中断none实验已归档:"
    echo "   ${BACKUP_DIR}/$(basename "${OLD_RUN}")"
fi

rm -f "${NONE_DIR}/best_fold_model_path.txt"

echo
echo "================================================================================"
echo "开始重新运行 M0 none（Full FT）"
echo "================================================================================"

export PYTHONUNBUFFERED=1
export ROOT="${ROOT}"
export STAGE=0
export PRETRAINED_MODEL="${PRETRAINED_MODEL}"
export RUN_ROOT_OVERRIDE="${RUN}"
export STRATEGIES="none"

# 已经完成的Frozen不再重复测试
export RUN_FROZEN=0

# none完成后继续执行外部测试
export RUN_EXTERNAL=1

# 明确锁定原实验训练配置
export SEED=43
export FINE_TUNE_EPOCHS=50
export BATCH_SIZE=32
export NUM_WORKERS=8

exec bash "${LAUNCHER}"
