#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import py_compile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp")
MAIN = ROOT / "main_tune.py"
RUNNER = ROOT / "run_progressive_finetune_stage.sh"

MAIN_START = '        # ============================================================\n        # 11. 找最佳折\n'
MAIN_END = '        return agg_results\n'
MAIN_NEW = '        # ============================================================\n        # 11. 固定测试策略：十折逐折评估，不选择单一“最佳折”\n        # ============================================================\n        # FOLDWISE_DUAL_TEST_POLICY_V1\n        #\n        # 不再根据任一测试集或折间验证成绩挑选单一fold。\n        # 每个fold的best checkpoint均保留，由阶段脚本分别在：\n        #   1) 固定内部1000条；\n        #   2) 固定外部987条\n        # 上各评价一次，并汇总mean±std及10-fold ensemble。\n        agg_results["fixed_test_policy"] = {\n            "policy": "evaluate_every_fold",\n            "fold_selection": "disabled",\n            "n_fold_models": int(len(fold_model_paths)),\n            "internal_test": "each_fold_once",\n            "external_test": "each_fold_once",\n            "aggregation": [\n                "fold_mean_std",\n                "fold_min_max",\n                "prediction_ensemble_mean",\n            ],\n            "note": (\n                "No fold is selected using validation, internal-test, or "\n                "external-test performance."\n            ),\n        }\n\n        # 前面已经写过一次聚合JSON；补写测试策略，确保元数据完整。\n        with open(agg_path, "w", encoding="utf-8") as f:\n            json.dump(agg_results, f, indent=2, ensure_ascii=False)\n\n        print(f"\\n{\'=\' * 60}")\n        print("📋 固定测试策略：10个fold模型逐一测试")\n        print("   不选择BEST FOLD；不再仅测试某一个共享模型")\n        print("   内部1000条与外部987条由阶段脚本逐fold评估并汇总")\n        print(f"{\'=\' * 60}")\n\n        return agg_results\n'

RUN_EVAL_NEW = 'run_evaluation() {\n    local model_path="$1"\n    local station_file="$2"\n    local output_dir="$3"\n    local evaluation_name="$4"\n    local result_name="fine_tune_evaluation_results.json"\n    local prediction_name="test_set_features_complete_with_pretrained.csv"\n    local summary_name="fine_tune_summary.txt"\n\n    # PROGRESSIVE_EXTERNAL_ISOLATED_MANIFEST_V1\n    if [[ "${station_file}" == "${EXTERNAL_DATA}" ]]; then\n        local isolated_dir="${RUN_ROOT}/runtime_inputs/external_evaluation"\n        local isolated_file="${isolated_dir}/external_evaluation_input.csv"\n\n        mkdir -p "${isolated_dir}"\n\n        if [[ ! -f "${isolated_file}" ]]; then\n            cp -p "${EXTERNAL_DATA}" "${isolated_file}"\n        fi\n\n        if ! cmp -s "${EXTERNAL_DATA}" "${isolated_file}"; then\n            echo "❌ 外部评估隔离副本与原文件不一致"\n            echo "   原文件: ${EXTERNAL_DATA}"\n            echo "   隔离副本: ${isolated_file}"\n            exit 1\n        fi\n\n        station_file="${isolated_file}"\n\n        echo "✅ 外部评估使用隔离清单:"\n        echo "   ${station_file}"\n        echo "   该目录不包含内部1000条固定测试文件"\n    fi\n\n    mkdir -p "${output_dir}"\n\n    # FOLDWISE_EVALUATION_OUTPUT_V1\n    # 聚合不仅需要指标JSON，还需要逐样本预测CSV。\n    promote_unique_result "${output_dir}" "${result_name}" 0 >/dev/null || true\n    promote_unique_result "${output_dir}" "${prediction_name}" 0 >/dev/null || true\n    promote_unique_result "${output_dir}" "${summary_name}" 0 >/dev/null || true\n\n    if [[ -f "${output_dir}/${result_name}" &&\n          -f "${output_dir}/${prediction_name}" ]]; then\n        echo\n        echo "✅ 已有完整评估结果，跳过重复运行:"\n        echo "   ${output_dir}/${result_name}"\n        echo "   ${output_dir}/${prediction_name}"\n        return 0\n    fi\n\n    echo\n    echo "================================================================================"\n    echo "评估: ${evaluation_name}"\n    echo "模型: ${model_path}"\n    echo "数据: ${station_file}"\n    echo "输出: ${output_dir}"\n    echo "================================================================================"\n\n    python "${MAIN}" \\\n        --mode evaluate \\\n        --pretrained_model "${model_path}" \\\n        --model_path "${model_path}" \\\n        --station_data_path "${station_file}" \\\n        --save_dir "${output_dir}" \\\n        "${COMMON_ARGS[@]}"\n\n    promote_unique_result "${output_dir}" "${result_name}" 1 >/dev/null\n    promote_unique_result "${output_dir}" "${prediction_name}" 1 >/dev/null\n    promote_unique_result "${output_dir}" "${summary_name}" 0 >/dev/null || true\n\n    printf \'%s\\n\' "${model_path}" > "${output_dir}/model_path.txt"\n\n    echo "✅ 评估结果确认完成:"\n    echo "   ${output_dir}/${result_name}"\n    echo "   ${output_dir}/${prediction_name}"\n}\n\n\naggregate_fold_evaluations() {\n    local split_root="$1"\n    local split_name="$2"\n    local expected_samples="$3"\n    local strategy="$4"\n\n    # FOLDWISE_DUAL_TEST_AGGREGATION_V1\n    python - \\\n        "${split_root}" \\\n        "${split_name}" \\\n        "${expected_samples}" \\\n        "${STAGE}" \\\n        "${strategy}" <<\'PY\'\nfrom __future__ import annotations\n\nimport json\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nsplit_root = Path(sys.argv[1])\nsplit_name = sys.argv[2]\nexpected_samples = int(sys.argv[3])\nstage = int(sys.argv[4])\nstrategy = sys.argv[5]\n\nprediction_filename = "test_set_features_complete_with_pretrained.csv"\n\nidentity_candidates = [\n    "样本索引",\n    "站点ID",\n    "日期",\n    "DOY",\n    "行列号_row",\n    "行列号_col",\n    "原始经度",\n    "原始纬度",\n]\n\nmetric_columns = [\n    "r",\n    "nse",\n    "rmse_mm",\n    "mae_mm",\n    "bias_mm",\n    "rmse_obs_ge50_mm",\n    "bias_obs_ge80_mm",\n    "slope",\n    "intercept_mm",\n    "std_ratio",\n    "pred_std_mm",\n]\n\n\ndef safe_float(value):\n    try:\n        value = float(value)\n    except (TypeError, ValueError):\n        return None\n    return value if np.isfinite(value) else None\n\n\ndef compute_metrics(target, prediction):\n    target = np.asarray(target, dtype=np.float64).reshape(-1)\n    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)\n\n    valid = np.isfinite(target) & np.isfinite(prediction)\n    target = target[valid]\n    prediction = prediction[valid]\n\n    if target.size == 0:\n        raise RuntimeError("没有有效目标/预测值")\n\n    error = prediction - target\n    rmse = float(np.sqrt(np.mean(error ** 2)))\n    mae = float(np.mean(np.abs(error)))\n    bias = float(np.mean(error))\n\n    target_std = float(np.std(target))\n    pred_std = float(np.std(prediction))\n    std_ratio = pred_std / target_std if target_std > 1e-12 else float("nan")\n\n    ss_res = float(np.sum(error ** 2))\n    ss_tot = float(np.sum((target - np.mean(target)) ** 2))\n    nse = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")\n\n    if target.size > 1 and target_std > 1e-12 and pred_std > 1e-12:\n        r = float(np.corrcoef(target, prediction)[0, 1])\n    else:\n        r = float("nan")\n\n    centered_target = target - np.mean(target)\n    slope_denom = float(np.sum(centered_target ** 2))\n    if slope_denom > 1e-12:\n        slope = float(\n            np.sum(centered_target * (prediction - np.mean(prediction)))\n            / slope_denom\n        )\n        intercept = float(np.mean(prediction) - slope * np.mean(target))\n    else:\n        slope = float("nan")\n        intercept = float("nan")\n\n    ge50 = target >= 50.0\n    ge80 = target >= 80.0\n\n    rmse_ge50 = (\n        float(np.sqrt(np.mean(error[ge50] ** 2)))\n        if np.any(ge50)\n        else float("nan")\n    )\n    bias_ge80 = (\n        float(np.mean(error[ge80]))\n        if np.any(ge80)\n        else float("nan")\n    )\n\n    collapsed = bool(\n        pred_std < 1.0\n        or (\n            np.isfinite(std_ratio)\n            and std_ratio < 0.05\n            and np.isfinite(slope)\n            and abs(slope) < 0.05\n        )\n    )\n\n    return {\n        "n_samples": int(target.size),\n        "r": safe_float(r),\n        "nse": safe_float(nse),\n        "rmse_mm": rmse,\n        "mae_mm": mae,\n        "bias_mm": bias,\n        "rmse_obs_ge50_mm": safe_float(rmse_ge50),\n        "bias_obs_ge80_mm": safe_float(bias_ge80),\n        "n_obs_ge50": int(np.sum(ge50)),\n        "n_obs_ge80": int(np.sum(ge80)),\n        "slope": safe_float(slope),\n        "intercept_mm": safe_float(intercept),\n        "std_ratio": safe_float(std_ratio),\n        "pred_std_mm": pred_std,\n        "target_std_mm": target_std,\n        "collapsed": collapsed,\n    }\n\n\ndef canonical_identity(frame, columns):\n    if not columns:\n        return pd.Series(\n            np.arange(len(frame), dtype=np.int64).astype(str),\n            index=frame.index,\n        )\n    values = frame[columns].copy()\n    for column in columns:\n        values[column] = values[column].fillna("").astype(str)\n    return values.agg("||".join, axis=1)\n\n\nrows = []\nlong_frames = []\nprediction_arrays = []\nbase_target = None\nbase_identity = None\nbase_frame = None\nidentity_columns = None\n\nfor fold in range(1, 11):\n    fold_dir = split_root / f"fold_{fold:02d}"\n    prediction_path = fold_dir / prediction_filename\n    model_path_file = fold_dir / "model_path.txt"\n\n    if not prediction_path.is_file():\n        raise FileNotFoundError(\n            f"Fold {fold}缺少逐样本预测CSV: {prediction_path}"\n        )\n\n    frame = pd.read_csv(prediction_path)\n\n    required = ["站点SWE_raw", "微调模型预测_raw"]\n    missing = [column for column in required if column not in frame.columns]\n    if missing:\n        raise RuntimeError(\n            f"Fold {fold}预测CSV缺少列: {missing}; "\n            f"现有列={list(frame.columns)}"\n        )\n\n    if len(frame) != expected_samples:\n        raise RuntimeError(\n            f"Fold {fold}样本数不一致: "\n            f"actual={len(frame)}, expected={expected_samples}"\n        )\n\n    target = pd.to_numeric(\n        frame["站点SWE_raw"],\n        errors="coerce",\n    ).to_numpy(dtype=np.float64)\n    prediction = pd.to_numeric(\n        frame["微调模型预测_raw"],\n        errors="coerce",\n    ).to_numpy(dtype=np.float64)\n\n    if identity_columns is None:\n        identity_columns = [\n            column for column in identity_candidates\n            if column in frame.columns\n        ]\n\n    identity = canonical_identity(frame, identity_columns)\n\n    if base_target is None:\n        base_target = target.copy()\n        base_identity = identity.to_numpy(dtype=str)\n        base_frame = frame.copy()\n    else:\n        if not np.array_equal(identity.to_numpy(dtype=str), base_identity):\n            raise RuntimeError(\n                f"Fold {fold}样本顺序/身份与Fold 1不一致，拒绝直接集成"\n            )\n        if not np.allclose(\n            target,\n            base_target,\n            equal_nan=True,\n            atol=1e-8,\n            rtol=0.0,\n        ):\n            raise RuntimeError(\n                f"Fold {fold}目标值与Fold 1不一致，拒绝集成"\n            )\n\n    metrics = compute_metrics(target, prediction)\n    metrics["fold"] = fold\n    metrics["stage"] = stage\n    metrics["strategy"] = strategy\n    metrics["split"] = split_name\n    metrics["model_path"] = (\n        model_path_file.read_text(encoding="utf-8").strip()\n        if model_path_file.is_file()\n        else None\n    )\n    rows.append(metrics)\n    prediction_arrays.append(prediction)\n\n    keep_columns = list(identity_columns)\n    for column in ["站点SWE_raw", "FusedSWE_raw"]:\n        if column in frame.columns and column not in keep_columns:\n            keep_columns.append(column)\n\n    long_frame = frame[keep_columns].copy()\n    long_frame.insert(0, "fold", fold)\n    long_frame["prediction_mm"] = prediction\n    long_frames.append(long_frame)\n\nmetrics_frame = pd.DataFrame(rows).sort_values("fold")\nmetrics_frame.to_csv(\n    split_root / "fold_metrics.csv",\n    index=False,\n    encoding="utf-8-sig",\n)\n\npd.concat(long_frames, ignore_index=True).to_csv(\n    split_root / "predictions_long.csv",\n    index=False,\n    encoding="utf-8-sig",\n)\n\nprediction_matrix = np.vstack(prediction_arrays)\nensemble_prediction = np.nanmean(prediction_matrix, axis=0)\nensemble_std = np.nanstd(prediction_matrix, axis=0)\nensemble_min = np.nanmin(prediction_matrix, axis=0)\nensemble_max = np.nanmax(prediction_matrix, axis=0)\n\nensemble_metrics = compute_metrics(base_target, ensemble_prediction)\nensemble_metrics.update({\n    "stage": stage,\n    "strategy": strategy,\n    "split": split_name,\n    "ensemble": "mean_of_10_fold_models",\n    "n_models": 10,\n})\n\nensemble_columns = list(identity_columns)\nfor column in ["站点SWE_raw", "FusedSWE_raw"]:\n    if column in base_frame.columns and column not in ensemble_columns:\n        ensemble_columns.append(column)\n\nensemble_frame = base_frame[ensemble_columns].copy()\nensemble_frame["ensemble_prediction_mean_mm"] = ensemble_prediction\nensemble_frame["ensemble_prediction_std_mm"] = ensemble_std\nensemble_frame["ensemble_prediction_min_mm"] = ensemble_min\nensemble_frame["ensemble_prediction_max_mm"] = ensemble_max\nensemble_frame.to_csv(\n    split_root / "ensemble_predictions.csv",\n    index=False,\n    encoding="utf-8-sig",\n)\n\n(split_root / "ensemble_metrics.json").write_text(\n    json.dumps(\n        ensemble_metrics,\n        ensure_ascii=False,\n        indent=2,\n        allow_nan=False,\n    ),\n    encoding="utf-8",\n)\n\nsummary = {\n    "stage": stage,\n    "strategy": strategy,\n    "split": split_name,\n    "n_folds": 10,\n    "expected_samples_per_fold": expected_samples,\n    "fold_selection": "none",\n    "test_use": "evaluation_only",\n    "metrics": {},\n    "collapsed_folds": [\n        int(row["fold"])\n        for row in rows\n        if bool(row.get("collapsed", False))\n    ],\n    "ensemble_metrics": ensemble_metrics,\n}\n\nfor metric in metric_columns:\n    values = pd.to_numeric(\n        metrics_frame[metric],\n        errors="coerce",\n    ).dropna().to_numpy(dtype=np.float64)\n\n    summary["metrics"][metric] = {\n        "n": int(values.size),\n        "mean": safe_float(np.mean(values)) if values.size else None,\n        "std": (\n            safe_float(np.std(values, ddof=1))\n            if values.size > 1\n            else 0.0 if values.size == 1 else None\n        ),\n        "min": safe_float(np.min(values)) if values.size else None,\n        "max": safe_float(np.max(values)) if values.size else None,\n    }\n\n(split_root / "fold_summary.json").write_text(\n    json.dumps(\n        summary,\n        ensure_ascii=False,\n        indent=2,\n        allow_nan=False,\n    ),\n    encoding="utf-8",\n)\n\npd.DataFrame([\n    {"metric": metric, **values}\n    for metric, values in summary["metrics"].items()\n]).to_csv(\n    split_root / "fold_summary.csv",\n    index=False,\n    encoding="utf-8-sig",\n)\n\nprint()\nprint("=" * 80)\nprint(\n    f"✅ M{stage} {strategy} | {split_name} "\n    "10-fold逐折测试汇总完成"\n)\nprint("=" * 80)\nfor metric in [\n    "r",\n    "nse",\n    "rmse_mm",\n    "mae_mm",\n    "bias_mm",\n    "rmse_obs_ge50_mm",\n    "bias_obs_ge80_mm",\n    "slope",\n    "std_ratio",\n]:\n    item = summary["metrics"][metric]\n    if item["mean"] is not None:\n        print(\n            f"{metric:<22s}: "\n            f"{item[\'mean\']:.4f} ± {item[\'std\']:.4f}"\n        )\nprint(f"collapsed folds       : {summary[\'collapsed_folds\']}")\nprint(\n    "ensemble RMSE/MAE/R  : "\n    f"{ensemble_metrics[\'rmse_mm\']:.4f} / "\n    f"{ensemble_metrics[\'mae_mm\']:.4f} / "\n    f"{ensemble_metrics[\'r\']}"\n)\nprint(f"结果目录              : {split_root}")\nprint("=" * 80)\nPY\n}'
STRATEGY_NEW = '    # FOLDWISE_DUAL_TEST_POLICY_V1\n    mapfile -t EXISTING_FOLD_MODELS < <(\n        find "${STRATEGY_DIR}" \\\n            -mindepth 2 -maxdepth 4 \\\n            -type f \\\n            -name "cv_fold_*_best_model.pth" \\\n            -print 2>/dev/null | sort -V\n    )\n\n    mapfile -t EXISTING_PANELS < <(\n        find "${STRATEGY_DIR}" \\\n            -mindepth 2 -maxdepth 4 \\\n            -type f \\\n            -name "cv_10fold_panel_matrix.png" \\\n            -print 2>/dev/null | sort\n    )\n\n    if [[ "${#EXISTING_FOLD_MODELS[@]}" -eq 10 &&\n          "${#EXISTING_PANELS[@]}" -ge 1 ]]; then\n        echo "✅ ${STRATEGY}十折训练已经完成，复用10个fold模型"\n    else\n        if [[ "${#EXISTING_FOLD_MODELS[@]}" -gt 0 ]]; then\n            echo "⚠ 当前仅找到 ${#EXISTING_FOLD_MODELS[@]} 个fold模型，将继续/重新运行训练"\n        fi\n\n        python "${MAIN}" \\\n            --mode fine_tune \\\n            --pretrained_model "${PRETRAINED_MODEL}" \\\n            --station_data_path "${INTERNAL_DATA}" \\\n            --save_dir "${STRATEGY_DIR}" \\\n            --fine_tune_epochs "${FINE_TUNE_EPOCHS}" \\\n            --fine_tune_lr "${FT_LR}" \\\n            --freeze_backbone \\\n            --freeze_strategy "${STRATEGY}" \\\n            --use_high_swe_weight \\\n            "${COMMON_ARGS[@]}"\n    fi\n\n    mapfile -t FOLD_MODELS < <(\n        find "${STRATEGY_DIR}" \\\n            -mindepth 2 -maxdepth 4 \\\n            -type f \\\n            -name "cv_fold_*_best_model.pth" \\\n            -print 2>/dev/null | sort -V\n    )\n\n    if [[ "${#FOLD_MODELS[@]}" -ne 10 ]]; then\n        echo "❌ ${STRATEGY}的fold-specific best模型数量异常"\n        echo "   expected=10, actual=${#FOLD_MODELS[@]}"\n        printf \'   %s\\n\' "${FOLD_MODELS[@]}"\n        exit 1\n    fi\n\n    UNIQUE_MODEL_PARENT_COUNT="$(\n        printf \'%s\\n\' "${FOLD_MODELS[@]}" \\\n            | xargs -n1 dirname \\\n            | sort -u \\\n            | wc -l\n    )"\n\n    if [[ "${UNIQUE_MODEL_PARENT_COUNT}" -ne 1 ]]; then\n        echo "❌ 10个fold模型不在同一个训练运行目录，拒绝混用"\n        printf \'   %s\\n\' "${FOLD_MODELS[@]}"\n        exit 1\n    fi\n\n    FOLD_MODEL_RUN_DIR="$(dirname "${FOLD_MODELS[0]}")"\n    printf \'%s\\n\' "${FOLD_MODEL_RUN_DIR}" \\\n        > "${STRATEGY_DIR}/fold_model_run_dir.txt"\n\n    echo "✅ ${STRATEGY} fold模型目录: ${FOLD_MODEL_RUN_DIR}"\n\n    INTERNAL_FOLD_ROOT="${STRATEGY_DIR}/internal_test_10fold"\n    EXTERNAL_FOLD_ROOT="${STRATEGY_DIR}/external_test_10fold"\n\n    mkdir -p "${INTERNAL_FOLD_ROOT}"\n    if [[ "${RUN_EXTERNAL}" == "1" ]]; then\n        mkdir -p "${EXTERNAL_FOLD_ROOT}"\n    fi\n\n    for MODEL_PATH in "${FOLD_MODELS[@]}"; do\n        MODEL_NAME="$(basename "${MODEL_PATH}")"\n\n        if [[ ! "${MODEL_NAME}" =~ ^cv_fold_([0-9]+)_best_model\\.pth$ ]]; then\n            echo "❌ 无法从模型名解析fold编号: ${MODEL_NAME}"\n            exit 1\n        fi\n\n        FOLD_NUM="$((10#${BASH_REMATCH[1]}))"\n        printf -v FOLD_PAD "%02d" "${FOLD_NUM}"\n\n        INTERNAL_FOLD_DIR="${INTERNAL_FOLD_ROOT}/fold_${FOLD_PAD}"\n        mkdir -p "${INTERNAL_FOLD_DIR}"\n        printf \'%s\\n\' "${MODEL_PATH}" > "${INTERNAL_FOLD_DIR}/model_path.txt"\n\n        run_evaluation \\\n            "${MODEL_PATH}" \\\n            "${INTERNAL_DATA}" \\\n            "${INTERNAL_FOLD_DIR}" \\\n            "M${STAGE} ${STRATEGY} Fold ${FOLD_NUM} → 内部1000条固定测试"\n\n        if [[ "${RUN_EXTERNAL}" == "1" ]]; then\n            EXTERNAL_FOLD_DIR="${EXTERNAL_FOLD_ROOT}/fold_${FOLD_PAD}"\n            mkdir -p "${EXTERNAL_FOLD_DIR}"\n            printf \'%s\\n\' "${MODEL_PATH}" > "${EXTERNAL_FOLD_DIR}/model_path.txt"\n\n            run_evaluation \\\n                "${MODEL_PATH}" \\\n                "${EXTERNAL_DATA}" \\\n                "${EXTERNAL_FOLD_DIR}" \\\n                "M${STAGE} ${STRATEGY} Fold ${FOLD_NUM} → 外部987条固定测试"\n        fi\n    done\n\n    aggregate_fold_evaluations \\\n        "${INTERNAL_FOLD_ROOT}" \\\n        "internal_1000" \\\n        1000 \\\n        "${STRATEGY}"\n\n    if [[ "${RUN_EXTERNAL}" == "1" ]]; then\n        aggregate_fold_evaluations \\\n            "${EXTERNAL_FOLD_ROOT}" \\\n            "external_987" \\\n            987 \\\n            "${STRATEGY}"\n    fi\n'

MARKER_MAIN = "FOLDWISE_DUAL_TEST_POLICY_V1"
MARKER_AGG = "FOLDWISE_DUAL_TEST_AGGREGATION_V1"
MARKER_OUTPUT = "FOLDWISE_EVALUATION_OUTPUT_V1"


def replace_range(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"找不到起始标记: {start_marker[:80]}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"找不到结束标记: {end_marker[:80]}")
    end += len(end_marker)
    return text[:start] + replacement + text[end:]


def main() -> None:
    for path in (MAIN, RUNNER):
        if not path.is_file():
            raise FileNotFoundError(path)

    main_text = MAIN.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    changes = []

    if MARKER_MAIN not in main_text:
        main_text = replace_range(
            main_text,
            MAIN_START,
            MAIN_END,
            MAIN_NEW,
        )
        changes.append("disable_single_best_fold_and_single_fold_test")

    if MARKER_AGG not in runner_text or MARKER_OUTPUT not in runner_text:
        fn_start = runner_text.find("run_evaluation() {")
        fn_end_marker = (
            "\n\n# ----------------------------------------------------------------------\n"
            "# 1. Frozen"
        )
        fn_end = runner_text.find(fn_end_marker, fn_start)

        if fn_start < 0 or fn_end < 0:
            raise RuntimeError("无法定位run_evaluation函数")

        runner_text = (
            runner_text[:fn_start]
            + RUN_EVAL_NEW
            + runner_text[fn_end:]
        )
        changes.append("add_fold_evaluation_aggregation")

    # runner中的策略标记与main相同，因此用旧块标记判断。
    old_strategy_marker = "    # PROGRESSIVE_ACTUAL_OUTPUT_V1"
    if old_strategy_marker in runner_text:
        strategy_start = runner_text.find(old_strategy_marker)
        strategy_end_marker = "done\n\npython - \\"
        strategy_end = runner_text.find(
            strategy_end_marker,
            strategy_start,
        )
        if strategy_end < 0:
            raise RuntimeError("无法定位策略循环结束位置")

        runner_text = (
            runner_text[:strategy_start]
            + STRATEGY_NEW
            + runner_text[strategy_end:]
        )
        changes.append("replace_single_model_test_with_10fold_dual_test")

    manifest_anchor = '    "mixed_replay": False,\n'
    if (
        '"fine_tuned_test_policy"' not in runner_text
        and manifest_anchor in runner_text
    ):
        runner_text = runner_text.replace(
            manifest_anchor,
            manifest_anchor
            + '    "fine_tuned_test_policy": (\n'
              '        "Each of the 10 fold-specific best checkpoints is evaluated once "\n'
              '        "on the fixed internal 1000-sample test and once on the fixed "\n'
              '        "external 987-sample test. No fold is selected by test performance."\n'
              '    ),\n',
            1,
        )
        changes.append("write_test_policy_to_manifest")

    if not changes:
        py_compile.compile(str(MAIN), doraise=True)
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        print("✅ 十折逐折双测试补丁已经安装，无需重复修改")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        ROOT
        / "code_backups"
        / f"before_foldwise_dual_test_{stamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(MAIN, backup_dir / MAIN.name)
    shutil.copy2(RUNNER, backup_dir / RUNNER.name)

    MAIN.write_text(main_text, encoding="utf-8")
    RUNNER.write_text(runner_text, encoding="utf-8")
    RUNNER.chmod(0o755)

    py_compile.compile(str(MAIN), doraise=True)
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    print("✅ 十折逐折双测试补丁安装完成")
    print(f"   changes={changes}")
    print(f"   backup={backup_dir}")
    print(f"   main={MAIN}")
    print(f"   runner={RUNNER}")
    print()
    print("测试策略：")
    print("  - 不再选择BEST FOLD")
    print("  - 10个fold模型各测试内部1000条一次")
    print("  - 10个fold模型各测试外部987条一次")
    print("  - 输出mean±std、min/max和10-fold ensemble")


if __name__ == "__main__":
    main()
