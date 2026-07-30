#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--stage0-manifest', required=True)
    ap.add_argument('--incremental-manifest', required=True)
    ap.add_argument('--label-min', type=float, default=0.0)
    ap.add_argument('--label-max', type=float, default=400.0)
    args = ap.parse_args()

    config = Path(args.config).expanduser().resolve()
    stage0 = Path(args.stage0_manifest).expanduser().resolve()
    incremental = Path(args.incremental_manifest).expanduser().resolve()
    for p in (config, stage0, incremental):
        if not p.exists():
            raise FileNotFoundError(p)
    payload = json.loads(config.read_text(encoding='utf-8'))
    if payload.get('method') != 'clip_then_zscore':
        raise RuntimeError(f"归一化方法不是clip_then_zscore: {payload.get('method')}")
    if float(payload.get('label_min')) != args.label_min:
        raise RuntimeError('label_min不一致')
    if float(payload.get('label_max')) != args.label_max:
        raise RuntimeError('label_max不一致')
    refs = payload.get('reference_manifests', {})
    expected_stage0 = refs.get('stage0', {}).get('sha256')
    expected_inc = refs.get('incremental', {}).get('sha256')
    actual_stage0 = sha256_file(stage0)
    actual_inc = sha256_file(incremental)
    if expected_stage0 != actual_stage0:
        raise RuntimeError('Stage0清单已变化，必须重新生成统一归一化配置')
    if expected_inc != actual_inc:
        raise RuntimeError('152000随机清单已变化，必须重新生成统一归一化配置')
    print('✅ 统一归一化配置、Stage0清单和152000清单哈希一致')
    print(f"   config: {config}")
    print(f"   reference samples: {payload.get('reference_samples')}")


if __name__ == '__main__':
    main()
