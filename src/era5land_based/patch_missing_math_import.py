#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import py_compile
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp")
TARGET = ROOT / "main_tune.py"
MARKER = "MISSING_MATH_IMPORT_FIX_V1"


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    text = TARGET.read_text(encoding="utf-8")

    if "import math" in text:
        py_compile.compile(str(TARGET), doraise=True)
        print("✅ main_tune.py 已包含 import math，无需修改")
        return

    lines = text.splitlines()

    insert_idx = None

    # 优先放在标准库 import 区域中。
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "import json":
            insert_idx = idx + 1
            break

    if insert_idx is None:
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_idx = idx
                break

    if insert_idx is None:
        insert_idx = 0

    lines.insert(insert_idx, "import math")
    lines.insert(insert_idx + 1, f"# {MARKER}")

    updated = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (
        ROOT
        / "code_backups"
        / f"before_missing_math_import_fix_{stamp}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(TARGET, backup_dir / TARGET.name)

    TARGET.write_text(updated, encoding="utf-8")
    py_compile.compile(str(TARGET), doraise=True)

    print("✅ 已修复 NameError: math is not defined")
    print(f"   backup={backup_dir / TARGET.name}")
    print(f"   target={TARGET}")
    print("   added: import math")


if __name__ == "__main__":
    main()
