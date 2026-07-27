#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static checks for the progressive SWE package before copying to AutoDL."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

required = [
    "main_tune.py", "data_online_era5_swe.py", "models_swe.py",
    "data_station_online_swe.py", "build_stage0_manifest.py",
    "build_progressive_normalization.py", "verify_incremental_manifest.py",
    "verify_progressive_normalization.py", "verify_external_station_exclusion.py",
    "diagnose_stage0_station_records.py", "prepare_stage0_manifest.sh",
    "prepare_incremental_pool.sh", "prepare_progressive_normalization.sh",
    "pretrain_stage0_station.sh", "pretrain_incremental_stage.sh",
    "incremental_swe_ratios.json", "README.md",
]
missing = [name for name in required if not (ROOT / name).exists()]
if missing:
    raise SystemExit(f"Missing files: {missing}")

shells = [
    "prepare_stage0_manifest.sh", "prepare_incremental_pool.sh",
    "prepare_progressive_normalization.sh", "pretrain_stage0_station.sh",
    "pretrain_incremental_stage.sh",
]
for name in shells:
    text = (ROOT / name).read_text(encoding="utf-8")
    for needle in [
        "/root/ablation/station_swe_data.xlsx",
        "/root/ablation/external_test/*.csv",
        "EXTERNAL_STATION_EXCLUSION_RADIUS",
    ]:
        if needle not in text:
            raise SystemExit(f"{name} missing {needle}")
    subprocess.run(["bash", "-n", str(ROOT / name)], check=True)

stage0 = (ROOT / "pretrain_stage0_station.sh").read_text(encoding="utf-8")
for needle in [
    "--station_sampling_unit records",
    "--station_record_manifest_path",
    "--normalization_mode load",
    "--fixed_label_max_mm 400",
    "--disable_dataset_cache",
]:
    if needle not in stage0:
        raise SystemExit(f"Stage0 script missing {needle}")

pool = (ROOT / "prepare_incremental_pool.sh").read_text(encoding="utf-8")
for needle in [
    "--incremental_pool_size 152000",
    "--incremental_stage_sizes 12000 20000 40000 80000",
    "--seasonal_min_peak_swe_mm",
    "--seasonal_max_swe_mm",
    "--seasonal_min_consecutive_snow_free_days",
    "--normalization_mode skip",
    "--disable_dataset_cache",
]:
    if needle not in pool:
        raise SystemExit(f"Pool script missing {needle}")

main = (ROOT / "main_tune.py").read_text(encoding="utf-8")
for needle in [
    "--external_station_glob", "--external_station_exclusion_radius",
    "--external_station_strict", "--station_record_manifest_path",
    "--disable_dataset_cache", "choices=['auto', 'create', 'load', 'skip']",
]:
    if needle not in main:
        raise SystemExit(f"main_tune.py missing {needle}")

data = (ROOT / "data_online_era5_swe.py").read_text(encoding="utf-8")
for needle in [
    "def _load_external_station_exclusion", "def _load_station_record_samples",
    "def _load_station_record_samples_from_manifest", "station_sampling_unit=records",
    "self.external_excluded_cells", "clip_then_zscore",
]:
    if needle not in data:
        raise SystemExit(f"data_online_era5_swe.py missing {needle}")

subprocess.run([sys.executable, "-m", "py_compile"] + [str(p) for p in ROOT.glob("*.py")], check=True)
print("OK: package consistency checks passed")
