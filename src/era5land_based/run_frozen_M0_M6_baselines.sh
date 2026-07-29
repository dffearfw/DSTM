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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATOR="${SCRIPT_DIR}/evaluate_frozen_station_cv10.py"
BOXPLOT_TOOL="${SCRIPT_DIR}/plot_frozen_M0_M6_boxplots.py"

INTERNAL_DATA="${INTERNAL_DATA:-${ROOT}/shared_cache/progressive_finetune/internal_progressive_station.csv}"
NORMALIZATION="${NORMALIZATION:-${ROOT}/shared_cache/progressive_pretrain_normalization.json}"
FOLD_MANIFEST="${FOLD_MANIFEST:-${ROOT}/shared_cache/progressive_finetune/balanced_station_nested_cv10_all7936_manifest.csv}"
BALANCED_FOLD_TOOL="${BALANCED_FOLD_TOOL:-${ROOT}/balanced_station_cv10.py}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/shared_cache}"

M0_MODEL="${M0_MODEL:-${ROOT}/experiments/pretrain_stage0_station_20260714_215604/final_model.pth}"
M1_MODEL="${M1_MODEL:-${ROOT}/experiments/pretrain_stage1_incremental_20260715_084631/final_model.pth}"
M2_MODEL="${M2_MODEL:-${ROOT}/experiments/pretrain_stage2_incremental_20260715_142142/final_model.pth}"
M3_MODEL="${M3_MODEL:-${ROOT}/experiments/pretrain_stage3_incremental_20260715_195555/final_model.pth}"
M4_MODEL="${M4_MODEL:-${ROOT}/experiments/pretrain_stage4_incremental_20260716_133038/final_model.pth}"
M5_MODEL="${M5_MODEL:-${ROOT}/experiments/pretrain_stage5/final_model.pth}"
M6_MODEL="${M6_MODEL:-${ROOT}/experiments/pretrain_stage6/final_model.pth}"

ONLY_STAGES="${ONLY_STAGES:-0 1 2 3 4 5 6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
OUT="${OUT:-${ROOT}/experiments/frozen_M0_M6_baselines_${TIMESTAMP}}"
LOG="${OUT}/run.log"

declare -A MODELS=(
    [0]="${M0_MODEL}"
    [1]="${M1_MODEL}"
    [2]="${M2_MODEL}"
    [3]="${M3_MODEL}"
    [4]="${M4_MODEL}"
    [5]="${M5_MODEL}"
    [6]="${M6_MODEL}"
)

check_stage_list() {
    local stage
    local -A seen=()
    for stage in ${ONLY_STAGES}; do
        case "${stage}" in
            0|1|2|3|4|5|6)
                ;;
            *)
                echo "❌ ONLY_STAGES只允许0-6，当前包含: ${stage}"
                exit 2
                ;;
        esac
        if [[ -n "${seen[${stage}]:-}" ]]; then
            echo "❌ ONLY_STAGES包含重复阶段: ${stage}"
            exit 2
        fi
        seen["${stage}"]=1
    done
}

preflight() {
    local required
    local stage
    local model

    for required in \
        "${EVALUATOR}" \
        "${BOXPLOT_TOOL}" \
        "${BALANCED_FOLD_TOOL}" \
        "${ROOT}/main_tune.py" \
        "${ROOT}/data_station_online_swe.py" \
        "${ROOT}/models_swe.py" \
        "${INTERNAL_DATA}" \
        "${NORMALIZATION}"
    do
        if [[ ! -f "${required}" ]]; then
            echo "❌ 缺少必要文件: ${required}"
            exit 1
        fi
    done

    for stage in ${ONLY_STAGES}; do
        model="${MODELS[${stage}]:-}"
        if [[ -z "${model}" ]]; then
            echo "❌ M${stage}尚未提供checkpoint路径"
            echo "   请设置 M${stage}_MODEL=/absolute/path/final_model.pth"
            exit 1
        fi
        if [[ ! -f "${model}" ]]; then
            echo "❌ M${stage} checkpoint不存在: ${model}"
            exit 1
        fi
    done

    python -m py_compile "${EVALUATOR}" "${BOXPLOT_TOOL}"
}

check_stage_list
preflight

if [[ "${1:-}" == "--check" ]]; then
    echo "✅ Frozen M0-M6基线预检查通过"
    echo "   本次阶段: ${ONLY_STAGES}"
    for stage in ${ONLY_STAGES}; do
        echo "   M${stage}: ${MODELS[${stage}]}"
    done
    echo "   外部测试: 不读取、不评估"
    exit 0
fi

if [[ "$#" -gt 0 ]]; then
    echo "❌ 未知参数: $*"
    echo "   仅支持无参数运行或 --check"
    exit 2
fi

mkdir -p "${OUT}"
exec > >(tee -a "${LOG}") 2>&1

echo "================================================================================"
echo "Frozen M0-M6统一站点级基线"
echo "================================================================================"
echo "本次阶段:       ${ONLY_STAGES}"
echo "内部CV池:       ${INTERNAL_DATA}"
echo "内部样本口径:   7,936条（旧固定1000条已并回）"
echo "统一归一化:     ${NORMALIZATION}"
echo "统一fold清单:   ${FOLD_MANIFEST}"
echo "输出目录:       ${OUT}"
echo "外部测试:       不读取、不评估"
echo "训练/选模:      无；Frozen只前向预测"
echo "================================================================================"

# 所有阶段只生成并复用这一份确定性站点fold清单。
python "${BALANCED_FOLD_TOOL}" \
    --station-csv "${INTERNAL_DATA}" \
    --output "${FOLD_MANIFEST}" \
    --n-splits 10 \
    --high-threshold-mm 80 \
    --include-fixed-test

for stage in ${ONLY_STAGES}; do
    model="${MODELS[${stage}]}"
    stage_dir="${OUT}/M${stage}/internal_cv10"

    echo
    echo "################################################################################"
    echo "Frozen M${stage}"
    echo "checkpoint: ${model}"
    echo "输出:       ${stage_dir}"
    echo "################################################################################"

    python "${EVALUATOR}" \
        --root "${ROOT}" \
        --station-csv "${INTERNAL_DATA}" \
        --checkpoint "${model}" \
        --stage-label "M${stage}" \
        --normalization-config "${NORMALIZATION}" \
        --cache-dir "${CACHE_DIR}" \
        --output-dir "${stage_dir}" \
        --fold-manifest "${FOLD_MANIFEST}" \
        --seed 43 \
        --n-splits 10 \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --include-fixed-test
done

python - "${OUT}" ${ONLY_STAGES} <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1]).resolve()
requested_stages = [int(value) for value in sys.argv[2:]]
available_stages = set(requested_stages)
for candidate in out.glob("M*/internal_cv10/frozen_station_cv10_summary.json"):
    match = re.fullmatch(r"M([0-9]+)", candidate.parents[1].name)
    if match:
        available_stages.add(int(match.group(1)))
stages = sorted(available_stages)
rows = []

for stage in stages:
    path = out / f"M{stage}" / "internal_cv10" / "frozen_station_cv10_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("stage_label") != f"M{stage}":
        raise RuntimeError(
            f"{path}阶段标签异常: {data.get('stage_label')!r}"
        )
    protocol = data.get("protocol", {})
    if protocol.get("n_cv_samples") != 7936:
        raise RuntimeError(
            f"{path}不是7936条口径: {protocol.get('n_cv_samples')}"
        )
    if protocol.get("fixed_internal_1000_merged_into_cv") is not True:
        raise RuntimeError(f"{path}未将旧固定1000条并回Nested CV")

    metrics = data["pooled_oof_metrics"]["Frozen"]
    files = data["files"]
    rows.append(
        {
            "stage": f"M{stage}",
            "checkpoint": files["checkpoint"],
            "checkpoint_sha256": files["checkpoint_sha256"],
            "n_samples": metrics["n_samples"],
            "r": metrics["r"],
            "nse": metrics["nse"],
            "rmse_mm": metrics["rmse_mm"],
            "mae_mm": metrics["mae_mm"],
            "bias_mm": metrics["bias_mm"],
            "rmse_obs_ge50_mm": metrics["rmse_obs_ge50_mm"],
            "bias_obs_ge80_mm": metrics["bias_obs_ge80_mm"],
            "n_obs_ge50": metrics["n_obs_ge50"],
            "n_obs_ge80": metrics["n_obs_ge80"],
            "slope": metrics["slope"],
            "intercept_mm": metrics["intercept_mm"],
            "std_ratio": metrics["std_ratio"],
            "target_mean_mm": metrics["target_mean_mm"],
            "prediction_mean_mm": metrics["prediction_mean_mm"],
            "target_std_mm": metrics["target_std_mm"],
            "prediction_std_mm": metrics["prediction_std_mm"],
            "prediction_min_mm": metrics["prediction_min_mm"],
            "prediction_max_mm": metrics["prediction_max_mm"],
        }
    )

frame = pd.DataFrame(rows)
frame.to_csv(
    out / "frozen_all_stages_pooled_oof_summary.csv",
    index=False,
    encoding="utf-8-sig",
)
(out / "frozen_all_stages_pooled_oof_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8",
)

print()
print("=" * 100)
print("Frozen各阶段 pooled OOF 汇总")
print("=" * 100)
print(
    frame[
        [
            "stage",
            "r",
            "nse",
            "rmse_mm",
            "mae_mm",
            "bias_mm",
            "slope",
            "std_ratio",
            "rmse_obs_ge50_mm",
            "bias_obs_ge80_mm",
        ]
    ].to_string(index=False)
)
print(f"CSV:  {out / 'frozen_all_stages_pooled_oof_summary.csv'}")
print(f"JSON: {out / 'frozen_all_stages_pooled_oof_summary.json'}")
print("=" * 100)
PY

python "${BOXPLOT_TOOL}" \
    --run-dir "${OUT}" \
    --stages ${ONLY_STAGES}

echo
echo "================================================================================"
echo "✅ Frozen阶段基线全部完成"
echo "输出目录: ${OUT}"
echo "汇总CSV:  ${OUT}/frozen_all_stages_pooled_oof_summary.csv"
echo "四联箱线图: ${OUT}/frozen_M0_M6_fold_distribution_4panel.png"
echo "日志:     ${LOG}"
echo "================================================================================"
