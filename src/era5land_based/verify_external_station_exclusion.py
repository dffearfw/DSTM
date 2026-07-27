#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--name', default='manifest')
    args = ap.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if not report.exists():
        raise FileNotFoundError(report)

    m = pd.read_csv(manifest)
    r = pd.read_csv(report)
    for name, df in [('manifest', m), ('report', r)]:
        missing = {'row', 'col'} - set(df.columns)
        if missing:
            raise ValueError(f'{name}缺少字段: {sorted(missing)}')
    keys = set(zip(r['row'].astype(int), r['col'].astype(int)))
    hit = m.apply(lambda x: (int(x['row']), int(x['col'])) in keys, axis=1)
    if hit.any():
        examples = m.loc[hit].head(10).to_dict('records')
        raise RuntimeError(
            f'{args.name}仍有{int(hit.sum())}条样本落入外部测试站点缓冲区，例如: {examples}'
        )
    print(f'✅ {args.name}未落入外部测试站点缓冲区: {len(m):,}条')


if __name__ == '__main__':
    main()
