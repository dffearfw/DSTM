#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PRETRAIN_SAMPLES_PER_DAY", "0")
os.environ.setdefault("USE_TARGET_QUOTA_SAMPLING", "0")
os.environ.setdefault("USE_QUOTA_SHORTAGE_SUPPLEMENT", "0")
os.environ.setdefault("STRICT_TARGET_QUOTA", "0")
os.environ.setdefault("PRECOMPUTE_ALL_SAMPLES", "0")

from data_online_era5_swe import SWEDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-guide-file", required=True)
    ap.add_argument("--shared-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2015, 2016, 2017, 2018])
    ap.add_argument("--patch-size", type=int, default=5)
    ap.add_argument("--min-valid-pixels", type=int, default=100)
    ap.add_argument("--clamday-threshold", type=float, default=0.5)
    ap.add_argument("--external-station-glob", required=True)
    ap.add_argument("--external-station-exclusion-radius", type=int, default=0)
    ap.add_argument("--external-station-report-path", required=True)
    ap.add_argument("--label-min-mm", type=float, default=0.0)
    ap.add_argument("--label-max-mm", type=float, default=400.0)
    args = ap.parse_args()

    station = Path(args.station_guide_file).expanduser().resolve()
    cache_dir = Path(args.shared_cache_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    report = Path(args.external_station_report_path).expanduser().resolve()
    if not station.exists():
        raise FileNotFoundError(station)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = SWEDataset(
        year_target=args.years,
        patch_size=args.patch_size,
        min_valid_pixels=args.min_valid_pixels,
        samples_per_day=0,
        clamday_threshold=args.clamday_threshold,
        cache_dir=None,
        force_reload=True,
        sampling_mode="station",
        station_guide_file=station,
        station_neighborhood=0,
        station_samples_per_day=-1,
        station_filter_zero_target=False,
        station_sampling_unit="records",
        station_record_dedup="grid_date",
        external_station_glob=args.external_station_glob,
        external_station_exclusion_radius=args.external_station_exclusion_radius,
        external_station_strict=True,
        external_station_report_path=report,
        normalization_mode="skip",
        fixed_label_min_mm=args.label_min_mm,
        fixed_label_max_mm=args.label_max_mm,
    )

    stats = cache_dir / "stage0_station_record_manifest.stats.json"
    rows = getattr(dataset, "stage0_manifest_rows", None)
    if rows is None:
        raise RuntimeError("数据集没有保留Stage 0清单行")
    import pandas as pd
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    if not output.exists():
        raise RuntimeError(f"Stage 0清单不存在: {output}")

    payload = dict(getattr(dataset, "station_sample_stats", {}))
    payload["final_dataset_length"] = int(len(dataset))
    payload["external_unique_center_cells"] = int(len(dataset.external_station_centers))
    payload["external_excluded_cells"] = int(len(dataset.external_excluded_cells))
    stats.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 78)
    print(f"✅ Stage 0有效清单: {output}")
    print(f"   最终样本数: {len(dataset):,}")
    print(f"   外部中心格点: {len(dataset.external_station_centers):,}")
    print(f"   外部缓冲排除格点: {len(dataset.external_excluded_cells):,}")
    print(f"   统计: {stats}")
    print("=" * 78)


if __name__ == "__main__":
    main()
