#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


OLD_MANIFEST = Path(
    "/root/autodl-tmp/shared_cache/incremental_random_pool_152000.csv"
)

CANDIDATE_MANIFEST = Path(
    "/root/autodl-tmp/shared_cache/"
    "stage5_candidate_pool_320000_seed1043.csv"
)

RATIO_CONFIG = Path(
    "/root/autodl-tmp/incremental_swe_ratios.json"
)

STAGE5_ONLY = Path(
    "/root/autodl-tmp/shared_cache/"
    "incremental_stage5_new_160000.csv"
)

COMBINED_MANIFEST = Path(
    "/root/autodl-tmp/shared_cache/"
    "incremental_random_pool_312000_stage1_5.csv"
)

STAGE5_SIZE = 160000
SELECTION_SEED = 5043
FOLD_SEED = 43 + 1009


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def allocate_counts(
    total: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    names = list(ratios)
    values = np.asarray(
        [float(ratios[name]) for name in names],
        dtype=np.float64,
    )

    values = values / values.sum()
    raw = values * total
    base = np.floor(raw).astype(np.int64)

    remaining = total - int(base.sum())
    fractional = raw - base

    order = np.argsort(-fractional, kind="stable")
    for idx in order[:remaining]:
        base[idx] += 1

    return {
        name: int(count)
        for name, count in zip(names, base)
    }


def main() -> None:
    for path in [
        OLD_MANIFEST,
        CANDIDATE_MANIFEST,
        RATIO_CONFIG,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    old = pd.read_csv(OLD_MANIFEST)
    candidate = pd.read_csv(CANDIDATE_MANIFEST)

    required = {
        "sample_id",
        "date",
        "row",
        "col",
        "swe_bin",
        "stage_id",
        "fold_id",
    }

    for name, df in [
        ("old", old),
        ("candidate", candidate),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"{name}清单缺少字段: {sorted(missing)}"
            )

    old["sample_id"] = old["sample_id"].astype(str)
    candidate["sample_id"] = candidate["sample_id"].astype(str)

    if len(old) != 152000:
        raise RuntimeError(
            f"旧清单不是152000条: {len(old):,}"
        )

    expected_old_counts = {
        1: 12000,
        2: 20000,
        3: 40000,
        4: 80000,
    }

    old_counts = (
        old.groupby(
            old["stage_id"].astype(int)
        ).size().to_dict()
    )

    if old_counts != expected_old_counts:
        raise RuntimeError(
            f"旧Stage 1–4数量异常: {old_counts}"
        )

    if old["sample_id"].duplicated().any():
        raise RuntimeError("旧清单存在重复sample_id")

    if candidate["sample_id"].duplicated().any():
        raise RuntimeError("候选池存在重复sample_id")

    old_ids = set(old["sample_id"])

    before = len(candidate)
    candidate = candidate[
        ~candidate["sample_id"].isin(old_ids)
    ].copy()

    print("=" * 78)
    print("构建Stage 5追加清单")
    print(f"旧Stage 1–4:        {len(old):,}")
    print(f"候选池原始数量:     {before:,}")
    print(f"去除旧样本后:       {len(candidate):,}")
    print(f"候选与旧池重合:     {before-len(candidate):,}")
    print("=" * 78)

    ratios = json.loads(
        RATIO_CONFIG.read_text(encoding="utf-8")
    )

    quotas = allocate_counts(
        STAGE5_SIZE,
        ratios,
    )

    rng = np.random.default_rng(SELECTION_SEED)
    selected_parts = []

    print("\n各SWE箱候选与目标：")

    for bin_name, need in quotas.items():
        pool = candidate[
            candidate["swe_bin"].astype(str)
            == str(bin_name)
        ].copy()

        available = len(pool)

        print(
            f"  {bin_name:>10s}: "
            f"可用={available:>7,}, "
            f"需要={need:>7,}"
        )

        if available < need:
            raise RuntimeError(
                f"SWE箱 {bin_name} 样本不足："
                f"{available:,} < {need:,}。"
                "请扩大候选池。"
            )

        idx = pool.index.to_numpy(copy=True)
        rng.shuffle(idx)

        selected_parts.append(
            pool.loc[idx[:need]].copy()
        )

    stage5 = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    if len(stage5) != STAGE5_SIZE:
        raise RuntimeError(
            f"Stage 5数量错误: {len(stage5):,}"
        )

    if stage5["sample_id"].isin(old_ids).any():
        raise RuntimeError(
            "Stage 5仍然包含旧Stage 1–4样本"
        )

    stage5["stage_id"] = 5
    stage5["source"] = "incremental_random"

    # 按Stage 5每个SWE箱重新固定十折。
    stage5["fold_id"] = 0
    fold_rng = np.random.default_rng(FOLD_SEED)

    for bin_name, group in stage5.groupby(
        "swe_bin",
        sort=True,
    ):
        idx = group.index.to_numpy(copy=True)
        fold_rng.shuffle(idx)

        folds = np.resize(
            np.arange(1, 11, dtype=np.int16),
            len(idx),
        )

        stage5.loc[idx, "fold_id"] = folds

    stage5["fold_id"] = (
        stage5["fold_id"].astype(int)
    )

    stage5 = stage5.sort_values(
        ["stage_id", "swe_bin", "date", "row", "col"]
    ).reset_index(drop=True)

    # 必须保持与旧清单相同的列及列顺序。
    missing_from_candidate = (
        set(old.columns) - set(stage5.columns)
    )

    if missing_from_candidate:
        raise RuntimeError(
            "候选清单缺少旧清单字段: "
            f"{sorted(missing_from_candidate)}"
        )

    stage5 = stage5[old.columns]

    # 原样保留旧清单行，并在末尾追加Stage 5。
    combined = pd.concat(
        [old.copy(), stage5.copy()],
        ignore_index=True,
    )

    if len(combined) != 312000:
        raise RuntimeError(
            f"合并清单数量错误: {len(combined):,}"
        )

    if combined["sample_id"].duplicated().any():
        dup = int(
            combined["sample_id"].duplicated().sum()
        )
        raise RuntimeError(
            f"合并清单重复sample_id: {dup}"
        )

    # 验证合并后前152000行与原清单逐值一致。
    retained = combined.iloc[:len(old)].reset_index(
        drop=True
    )

    old_reset = old.reset_index(drop=True)

    if not retained.equals(old_reset):
        raise RuntimeError(
            "合并清单没有完整保留旧Stage 1–4"
        )

    STAGE5_ONLY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage5.to_csv(
        STAGE5_ONLY,
        index=False,
        encoding="utf-8-sig",
    )

    combined.to_csv(
        COMBINED_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    counts = (
        combined.groupby(
            combined["stage_id"].astype(int)
        ).size().astype(int).to_dict()
    )

    bin_counts = (
        stage5.groupby("swe_bin")
        .size()
        .astype(int)
        .to_dict()
    )

    metadata = {
        "created_at": datetime.now().isoformat(),
        "method": (
            "retain exact original Stage1-4 manifest; "
            "append non-overlapping Stage5"
        ),
        "old_manifest": {
            "path": str(OLD_MANIFEST),
            "sha256": sha256_file(OLD_MANIFEST),
            "rows": int(len(old)),
        },
        "candidate_manifest": {
            "path": str(CANDIDATE_MANIFEST),
            "sha256": sha256_file(
                CANDIDATE_MANIFEST
            ),
            "rows": int(before),
            "nonoverlap_rows": int(len(candidate)),
        },
        "stage5_manifest": {
            "path": str(STAGE5_ONLY),
            "sha256": sha256_file(STAGE5_ONLY),
            "rows": int(len(stage5)),
        },
        "combined_manifest": {
            "path": str(COMBINED_MANIFEST),
            "sha256": sha256_file(
                COMBINED_MANIFEST
            ),
            "rows": int(len(combined)),
        },
        "stage_counts": {
            str(k): int(v)
            for k, v in counts.items()
        },
        "stage5_bin_counts": {
            str(k): int(v)
            for k, v in bin_counts.items()
        },
        "selection_seed": SELECTION_SEED,
        "fold_seed": FOLD_SEED,
    }

    meta_path = COMBINED_MANIFEST.with_suffix(
        COMBINED_MANIFEST.suffix + ".meta.json"
    )

    meta_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("✅ Stage 5清单构建完成")
    print(f"Stage 5单独清单: {STAGE5_ONLY}")
    print(f"Stage 1–5总清单: {COMBINED_MANIFEST}")
    print(f"阶段数量: {counts}")
    print(f"Stage 5分箱: {bin_counts}")
    print(f"元数据: {meta_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
