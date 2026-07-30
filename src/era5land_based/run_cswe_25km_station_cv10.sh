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
SCRIPT="${ROOT}/evaluate_cswe_25km_station_cv10.py"
STATION_CSV="${STATION_CSV:-${ROOT}/shared_cache/progressive_finetune/internal_progressive_station.csv}"
PRODUCT_ROOT="${PRODUCT_ROOT:-/root/ablation/cswe}"
INSPECT_FILE="${INSPECT_FILE:-${PRODUCT_ROOT}/F17_SSMIS_SWE_20170827_DAILY_025KM_V1.2.h5}"
SCALE_TO_MM="${SCALE_TO_MM:-1.0}"
CODE_252_POLICY="${CODE_252_POLICY:-zero}"
OOF_PREDICTIONS="${OOF_PREDICTIONS:-}"
SWE_DATASET="${SWE_DATASET:-}"
LATITUDE_DATASET="${LATITUDE_DATASET:-}"
LONGITUDE_DATASET="${LONGITUDE_DATASET:-}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
OUT="${OUT:-${ROOT}/experiments/cswe_25km_station_cv10_${TIMESTAMP}}"
LOG="${OUT}/run.log"

for required in "${SCRIPT}" "${STATION_CSV}" "${PRODUCT_ROOT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "❌ 缺少必要路径: ${required}"
        exit 1
    fi
done

python -m py_compile "${SCRIPT}"
python - <<'PY'
try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "❌ 当前Python缺少h5py，请先执行: pip install h5py"
    ) from exc
print(f"✅ h5py={h5py.__version__}")
try:
    import scipy
    print(f"✅ scipy={scipy.__version__}（二维网格快速最近邻）")
except ImportError:
    print("ℹ 未安装scipy，将使用NumPy分块最近邻；结果相同但预处理较慢")
PY

mkdir -p "${OUT}"
exec > >(tee -a "${LOG}") 2>&1

ARGS=(
    --root "${ROOT}"
    --station-csv "${STATION_CSV}"
    --product-root "${PRODUCT_ROOT}"
    --output-dir "${OUT}"
    --scale-to-mm "${SCALE_TO_MM}"
    --code-252-policy "${CODE_252_POLICY}"
)

if [[ -f "${INSPECT_FILE}" ]]; then
    ARGS+=(--inspect-file "${INSPECT_FILE}")
else
    echo "⚠ 指定审计样例不存在，将自动使用目录中首个HDF5: ${INSPECT_FILE}"
fi
if [[ -n "${OOF_PREDICTIONS}" ]]; then
    ARGS+=(--oof-predictions "${OOF_PREDICTIONS}")
fi
if [[ -n "${SWE_DATASET}" ]]; then
    ARGS+=(--swe-dataset "${SWE_DATASET}")
fi
if [[ -n "${LATITUDE_DATASET}" ]]; then
    ARGS+=(--latitude-dataset "${LATITUDE_DATASET}")
fi
if [[ -n "${LONGITUDE_DATASET}" ]]; then
    ARGS+=(--longitude-dataset "${LONGITUDE_DATASET}")
fi
if [[ "${REQUIRE_COMPLETE}" == "1" ]]; then
    ARGS+=(--require-complete)
fi

echo "===================================================================================================="
echo "Daily snow water equivalent product from 1980 to 2020 over China (25 km)"
echo "确定性平衡站点级10折产品基线"
echo "===================================================================================================="
echo "产品目录: ${PRODUCT_ROOT}"
echo "站点CSV:  ${STATION_CSV}"
if [[ -n "${OOF_PREDICTIONS}" ]]; then
    echo "OOF清单:  ${OOF_PREDICTIONS}"
else
    echo "OOF清单:  自动选择最新有效的Frozen M0 7936样本结果"
fi
echo "输出目录: ${OUT}"
echo "像元编码: 0–240=数值SWE；252=无雪；250/251/253/254/255不作数值"
echo "252策略:  ${CODE_252_POLICY}"
echo "坐标范围: 所有fold固定为0–400 mm"
echo "===================================================================================================="

python "${SCRIPT}" "${ARGS[@]}"

echo
echo "✅ 完成"
echo "图件: ${OUT}/cswe_25km_station_cv10_fold_scatter_panel.png"
echo "指标: ${OUT}/cswe_25km_station_cv10_fold_metrics.csv"
echo "明细: ${OUT}/cswe_25km_station_cv10_oof_values.csv"
echo "编码: ${OUT}/cswe_25km_sampling_audit.json"
echo "结构: ${OUT}/cswe_25km_h5_schema_audit.json"
echo "汇总: ${OUT}/cswe_25km_station_cv10_summary.json"
