#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只读扫描 /root/autodl-tmp 中的 Python / Shell 脚本，生成脚本审计报告。

用途：
- 区分预训练、微调、样本准备、验证、补丁、诊断、旧脚本；
- 找出硬编码模型路径、站点路径、manifest、归一化配置；
- 检查错误路径、500 mm旧归一化、Stage范围限制、产品修正/mixed mode；
- 找出内容完全相同的重复脚本；
- 不移动、不删除、不修改任何现有脚本，只写审计报告。

默认跳过：
experiments、shared_cache、code_backups、__pycache__、.git
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SKIP_DIR_NAMES = {
    "experiments",
    "shared_cache",
    "code_backups",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
}

SCRIPT_SUFFIXES = {".py", ".sh"}

WRONG_STATION_PATH = "/root/autodl-tmp/ablation/station_swe_data.xlsx"
RIGHT_STATION_PATH = "/root/ablation/station_swe_data.xlsx"

STRATEGY_NAMES = [
    "frozen",
    "fusion_ft",
    "point_ft",
    "spatial_ft",
    "partial",
    "none",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def iter_scripts(root: Path, recursive: bool) -> list[Path]:
    paths: list[Path] = []

    if recursive:
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name for name in dirnames
                if name not in SKIP_DIR_NAMES
                and not name.startswith(".")
            ]
            base = Path(current_root)
            for filename in filenames:
                path = base / filename
                if path.suffix.lower() in SCRIPT_SUFFIXES:
                    paths.append(path)
    else:
        paths.extend(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in SCRIPT_SUFFIXES
        )

    return sorted(paths, key=lambda p: str(p).lower())


def first_assignment(text: str, variable: str) -> str | None:
    patterns = [
        rf'^\s*(?:export\s+)?{re.escape(variable)}\s*=\s*["\']([^"\']+)["\']',
        rf'^\s*(?:export\s+)?{re.escape(variable)}\s*=\s*([^\s#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def find_paths(text: str) -> list[str]:
    patterns = [
        r'(?:"|\')(/root/[^"\']+)(?:"|\')',
        r'(?:"|\')(/mnt/[^"\']+)(?:"|\')',
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text))
    return sorted(set(found))


def detect_modes(text: str) -> list[str]:
    values = set()

    # Shell: --mode fine_tune
    for match in re.finditer(
        r'--mode(?:\s+|["\']?\s*)["\']?([A-Za-z0-9_]+)',
        text,
    ):
        values.add(match.group(1))

    # Python argparse choices around --mode
    for match in re.finditer(
        r'choices\s*=\s*\[([^\]]+)\]',
        text,
        flags=re.DOTALL,
    ):
        block = match.group(1)
        if any(name in block for name in (
            "fine_tune", "pretrain_cv", "evaluate",
            "pretrain_progressive", "build_incremental_manifest"
        )):
            for item in re.findall(r'["\']([A-Za-z0-9_]+)["\']', block):
                values.add(item)

    return sorted(values)


def detect_stage_values(text: str) -> list[int]:
    values = set()

    for match in re.finditer(r'STAGE\s*=\s*["\']?\$?\{?([0-9]+)', text):
        try:
            values.add(int(match.group(1)))
        except ValueError:
            pass

    for block in re.findall(
        r'choices\s*=\s*\[([0-9,\s]+)\]',
        text,
    ):
        for value in re.findall(r'\d+', block):
            values.add(int(value))

    for block in re.findall(
        r'\^\[([0-9\-]+)\]\$',
        text,
    ):
        for value in re.findall(r'\d+', block):
            values.add(int(value))

    return sorted(values)


def classify(path: Path, text: str) -> list[str]:
    name = path.name.lower()
    lower = text.lower()
    tags: list[str] = []

    if "fine_tune" in lower or "finetune" in name or "perfix" in name:
        tags.append("微调/评估")
    if "pretrain" in name or "pretrain_progressive" in lower:
        tags.append("预训练")
    if (
        "prepare" in name
        or "build_" in name
        or "--mode build_" in lower
    ):
        tags.append("数据/清单准备")
    if "verify" in name or "check_" in name:
        tags.append("验证检查")
    if "diagnos" in name or "test_" in name:
        tags.append("诊断测试")
    if "patch" in name or name.startswith("apply_"):
        tags.append("补丁")
    if "summar" in name or "collect" in name or "aggregate" in name:
        tags.append("结果汇总")
    if "archive" in name:
        tags.append("归档工具")
    if (
        "legacy" in name
        or "old" in name
        or "do_not_run" in lower
        or "旧路线" in text
    ):
        tags.append("旧版/遗留")
    if "incremental" in lower:
        tags.append("增量路线")
    if "cumulative" in lower:
        tags.append("累计从头路线")
    if "stage6" in lower or "stage6" in name:
        tags.append("Stage6")

    return tags or ["未分类"]


def warnings_for(path: Path, text: str) -> list[str]:
    warnings: list[str] = []
    lower = text.lower()

    if WRONG_STATION_PATH in text:
        warnings.append(f"包含错误站点路径: {WRONG_STATION_PATH}")

    if RIGHT_STATION_PATH in text:
        warnings.append(f"包含正确站点路径: {RIGHT_STATION_PATH}")

    if re.search(r'fixed_label_max_mm["\']?\s*[=:]\s*500', text):
        warnings.append("可能仍使用500 mm固定标签上限")
    if re.search(r'--fixed_label_max_mm\s+500\b', text):
        warnings.append("命令行仍传入--fixed_label_max_mm 500")

    if (
        "incremental_stage" in text
        and re.search(r'choices\s*=\s*\[\s*1\s*,\s*2\s*,\s*3\s*,\s*4\s*\]', text)
    ):
        warnings.append("incremental_stage仍被限制为1-4")

    if re.search(
        r'incremental_stage_sizes[^\n]*nargs\s*=\s*4',
        text,
    ):
        warnings.append("incremental_stage_sizes仍固定只能接收4个数")

    if re.search(r'--incremental_stage_sizes\s+12000\s+20000\s+40000\s+80000\b', text):
        warnings.append("只配置了原12k/20k/40k/80k四包")

    if "--use_product_correction" in text or "USE_PRODUCT_CORRECTION=1" in text:
        warnings.append("开启或支持产品值修正，比较实验需确认是否统一")

    if "--mixed_mode" in text or re.search(r'MIXED_MODE\s*=\s*1', text):
        warnings.append("开启或支持mixed mode，需确认是否使用伪标签回放")

    if "PRETRAIN_LOSS_WEIGHT=0.0" in text:
        warnings.append("mixed框架中预训练回放权重为0，实际等价纯站点微调")

    hardcoded = first_assignment(text, "PRETRAINED_MODEL")
    if hardcoded and hardcoded.startswith("/root/"):
        warnings.append(f"硬编码PRETRAINED_MODEL: {hardcoded}")

    station_data = first_assignment(text, "STATION_DATA")
    if station_data:
        warnings.append(f"STATION_DATA: {station_data}")

    if "seed == 42" in text and "time.time()" in text:
        warnings.append("seed=42会转换为动态时间种子")

    if path.name.startswith("pretrain_incremental_stage") and "--pretrained_model" in text:
        warnings.append("属于加载上一阶段权重的增量继承路线")

    if path.name.startswith("pretrain_cumulative_scratch"):
        warnings.append("属于各累计规模独立随机初始化路线")

    return warnings


def inspect_script(root: Path, path: Path) -> dict[str, Any]:
    text = read_text(path)
    stat = path.stat()

    lines = text.count("\n") + (1 if text else 0)
    rel = str(path.relative_to(root))

    info: dict[str, Any] = {
        "path": str(path),
        "relative_path": rel,
        "name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "lines": lines,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
        "tags": classify(path, text),
        "warnings": warnings_for(path, text),
        "modes": detect_modes(text),
        "stage_values": detect_stage_values(text),
        "strategies": [
            strategy for strategy in STRATEGY_NAMES
            if strategy in text
        ],
        "calls_main_tune": "main_tune.py" in text,
        "pretrained_model_assignment": first_assignment(text, "PRETRAINED_MODEL"),
        "previous_model_assignment": first_assignment(text, "PREVIOUS_MODEL"),
        "stage5_model_assignment": first_assignment(text, "STAGE5_MODEL"),
        "station_guide_assignment": first_assignment(text, "STATION_GUIDE_FILE"),
        "station_data_assignment": first_assignment(text, "STATION_DATA"),
        "manifest_assignment": first_assignment(text, "MANIFEST_PATH"),
        "normalization_assignment": first_assignment(text, "NORMALIZATION_CONFIG"),
        "paths_referenced": find_paths(text),
    }

    return info


def format_report(
    root: Path,
    records: list[dict[str, Any]],
    duplicates: dict[str, list[str]],
) -> str:
    lines: list[str] = []
    stamp = datetime.now().isoformat(timespec="seconds")

    lines.append("=" * 100)
    lines.append("AutoDL脚本审计报告")
    lines.append("=" * 100)
    lines.append(f"扫描根目录: {root}")
    lines.append(f"生成时间: {stamp}")
    lines.append(f"脚本数量: {len(records)}")
    lines.append("说明: 只读扫描；未移动、删除或修改任何现有脚本。")
    lines.append("")

    lines.append("=" * 100)
    lines.append("一、根目录脚本清单")
    lines.append("=" * 100)

    root_records = [
        rec for rec in records
        if "/" not in rec["relative_path"]
    ]

    for index, rec in enumerate(root_records, start=1):
        lines.append(
            f"[{index:02d}] {rec['relative_path']} | "
            f"{rec['lines']}行 | {rec['size_bytes']:,} B | "
            f"SHA256={rec['sha256'][:12]}"
        )
        lines.append(f"     类型: {', '.join(rec['tags'])}")
        if rec["modes"]:
            lines.append(f"     mode: {', '.join(rec['modes'])}")
        if rec["stage_values"]:
            lines.append(
                "     检测到Stage值: "
                + ", ".join(map(str, rec["stage_values"]))
            )
        if rec["strategies"]:
            lines.append(
                "     策略: " + ", ".join(rec["strategies"])
            )
        if rec["pretrained_model_assignment"]:
            lines.append(
                "     PRETRAINED_MODEL: "
                + rec["pretrained_model_assignment"]
            )
        if rec["previous_model_assignment"]:
            lines.append(
                "     PREVIOUS_MODEL: "
                + rec["previous_model_assignment"]
            )
        if rec["stage5_model_assignment"]:
            lines.append(
                "     STAGE5_MODEL: "
                + rec["stage5_model_assignment"]
            )
        if rec["station_guide_assignment"]:
            lines.append(
                "     STATION_GUIDE_FILE: "
                + rec["station_guide_assignment"]
            )
        if rec["station_data_assignment"]:
            lines.append(
                "     STATION_DATA: "
                + rec["station_data_assignment"]
            )
        if rec["manifest_assignment"]:
            lines.append(
                "     MANIFEST_PATH: "
                + rec["manifest_assignment"]
            )
        if rec["normalization_assignment"]:
            lines.append(
                "     NORMALIZATION_CONFIG: "
                + rec["normalization_assignment"]
            )
        for warning in rec["warnings"]:
            lines.append(f"     ⚠ {warning}")
        lines.append("")

    nested = [
        rec for rec in records
        if "/" in rec["relative_path"]
    ]
    if nested:
        lines.append("=" * 100)
        lines.append("二、其他未跳过子目录中的脚本")
        lines.append("=" * 100)
        for rec in nested:
            lines.append(
                f"- {rec['relative_path']} | "
                f"{', '.join(rec['tags'])} | "
                f"{rec['lines']}行"
            )
        lines.append("")

    lines.append("=" * 100)
    lines.append("三、按功能分组")
    lines.append("=" * 100)

    tag_map: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        for tag in rec["tags"]:
            tag_map[tag].append(rec["relative_path"])

    for tag in sorted(tag_map):
        lines.append(f"\n[{tag}]")
        for rel in tag_map[tag]:
            lines.append(f"  - {rel}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("四、内容完全相同的重复脚本")
    lines.append("=" * 100)

    if duplicates:
        for digest, paths in duplicates.items():
            lines.append(f"SHA256={digest}")
            for rel in paths:
                lines.append(f"  - {rel}")
    else:
        lines.append("未发现内容完全相同的重复脚本。")

    lines.append("")
    lines.append("=" * 100)
    lines.append("五、重点风险汇总")
    lines.append("=" * 100)

    any_warning = False
    for rec in records:
        for warning in rec["warnings"]:
            any_warning = True
            lines.append(f"- {rec['relative_path']}: {warning}")

    if not any_warning:
        lines.append("未检测到预设规则中的风险。")

    lines.append("")
    lines.append("=" * 100)
    lines.append("六、建议人工重点确认")
    lines.append("=" * 100)
    lines.extend([
        "1. 当前真正使用的预训练入口是哪一个：增量继承、累计从头，还是Stage6扩样。",
        "2. Stage0–4微调脚本是否统一使用同一站点清单、seed、fold和0–400归一化。",
        "3. 微调比较中产品值修正、mixed mode、伪标签回放是否全部统一关闭或统一开启。",
        "4. Stage0–4使用的是final_model.pth还是某一折best.pth，不要混用。",
        "5. 根目录中的旧测试脚本、补丁脚本是否仍可能被误执行。",
    ])

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp"),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描未跳过的子目录；默认只扫根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    paths = iter_scripts(root, recursive=args.recursive)
    records = [inspect_script(root, path) for path in paths]

    by_hash: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        by_hash[rec["sha256"]].append(rec["relative_path"])
    duplicates = {
        digest: rels
        for digest, rels in by_hash.items()
        if len(rels) > 1
    }

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = output_dir / f"script_audit_{stamp}.txt"
    json_path = output_dir / f"script_audit_{stamp}.json"

    report = format_report(root, records, duplicates)
    text_path.write_text(report, encoding="utf-8")

    payload = {
        "root": str(root),
        "generated_at": datetime.now().isoformat(),
        "recursive": bool(args.recursive),
        "skipped_directories": sorted(SKIP_DIR_NAMES),
        "script_count": len(records),
        "records": records,
        "duplicate_groups": duplicates,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest_text = output_dir / "script_audit_latest.txt"
    latest_json = output_dir / "script_audit_latest.json"
    latest_text.write_text(report, encoding="utf-8")
    latest_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(report)
    print("=" * 100)
    print("✅ 扫描完成")
    print(f"文本报告: {text_path}")
    print(f"JSON报告: {json_path}")
    print(f"最新文本: {latest_text}")
    print(f"最新JSON: {latest_json}")


if __name__ == "__main__":
    main()
