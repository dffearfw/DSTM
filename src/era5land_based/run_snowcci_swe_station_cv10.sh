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
SCRIPT="${ROOT}/evaluate_snowcci_swe_station_cv10.py"
STATION_CSV="${STATION_CSV:-${ROOT}/shared_cache/progressive_finetune/internal_progressive_station.csv}"
PRODUCT_ROOT="${PRODUCT_ROOT:-/root/ablation/snowcci}"
SCALE_TO_MM="${SCALE_TO_MM:-1.0}"
OOF_PREDICTIONS="${OOF_PREDICTIONS:-}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
OUT="${OUT:-${ROOT}/experiments/snowcci_swe_station_cv10_${TIMESTAMP}}"
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
    import netCDF4
except ImportError as exc:
    raise SystemExit(
        "❌ 当前Python缺少netCDF4，请先执行: pip install netCDF4"
    ) from exc
print(f"✅ netCDF4={netCDF4.__version__}")
PY

mkdir -p "${OUT}"
exec > >(tee -a "${LOG}") 2>&1

ARGS=(
    --root "${ROOT}"
    --station-csv "${STATION_CSV}"
    --product-root "${PRODUCT_ROOT}"
    --output-dir "${OUT}"
    --scale-to-mm "${SCALE_TO_MM}"
)
if [[ -n "${OOF_PREDICTIONS}" ]]; then
    ARGS+=(--oof-predictions "${OOF_PREDICTIONS}")
fi
if [[ "${REQUIRE_COMPLETE}" == "1" ]]; then
    ARGS+=(--require-complete)
fi

echo "===================================================================================================="
echo "ESA Snow CCI SWE CRDP v4.0"
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
echo "掩膜处理: 不填0，基于有效Snow CCI像元绘图并统计"
echo "===================================================================================================="

python "${SCRIPT}" "${ARGS[@]}"

echo
echo "✅ 完成"
echo "图件: ${OUT}/snowcci_swe_station_cv10_fold_scatter_panel.png"
echo "指标: ${OUT}/snowcci_swe_station_cv10_fold_metrics.csv"
echo "审计: ${OUT}/snowcci_swe_sampling_audit.json"
echo "汇总: ${OUT}/snowcci_swe_station_cv10_summary.json"
