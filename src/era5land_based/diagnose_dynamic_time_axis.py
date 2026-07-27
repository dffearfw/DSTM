#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, calendar, re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import rasterio


def canon(x):
    return pd.Timestamp(x).normalize().to_pydatetime()


def parse_single(name: str):
    m = re.search(r'CHELSA_(?:pr|sfcWind)_(\d{2})_(\d{2})_(\d{4})', name, re.I)
    if m:
        d, mo, y = map(int, m.groups()); return datetime(y, mo, d)
    m = re.search(r'(\d{2})_(\d{2})_(\d{4})', name)
    if m:
        d, mo, y = map(int, m.groups()); return datetime(y, mo, d)
    m = re.search(r'(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)', name)
    if m:
        y, mo, d = map(int, m.groups()); return datetime(y, mo, d)
    return None


def multiband_dates(path, ds, years):
    descs = list(ds.descriptions or [])
    if descs and any(descs):
        out=[]
        for desc in descs:
            if not desc: out=[]; break
            m=re.search(r'(\d{4})[-_/]?(\d{2})[-_/]?(\d{2})', str(desc))
            if not m: out=[]; break
            try: out.append(datetime(*map(int,m.groups())))
            except ValueError: out=[]; break
        if len(out)==ds.count and all(d.year in years for d in out): return out
    name=path.stem
    m=re.search(r'(?<!\d)(\d{4})[_-]?(\d{2})(?!\d)', name)
    if m:
        y,mo=map(int,m.groups())
        if y in years and 1<=mo<=12 and ds.count<=calendar.monthrange(y,mo)[1]:
            return [datetime(y,mo,d) for d in range(1,ds.count+1)]
    m=re.search(r'(?<!\d)(\d{4})(?!\d)', name)
    if m:
        y=int(m.group(1)); exp=366 if calendar.isleap(y) else 365
        if y in years and ds.count==exp:
            st=datetime(y,1,1); return [st+timedelta(days=i) for i in range(ds.count)]
    return None


def collect(root: Path, var: str, years):
    files=sorted(root.glob('*.tif'))
    dates=set(); failed=[]
    for f in files:
        if not any(str(y) in f.name for y in years):
            continue
        try:
            with rasterio.open(f) as ds:
                if var in {'lst','rh'} or ds.count>1:
                    ds_dates=multiband_dates(f,ds,years)
                    if ds_dates is None:
                        failed.append((f.name,ds.count)); continue
                    dates.update(canon(d) for d in ds_dates)
                else:
                    d=parse_single(f.stem)
                    if d is None: failed.append((f.name,ds.count)); continue
                    if d.year in years: dates.add(canon(d))
        except Exception:
            failed.append((f.name,-1))
    return dates, files, failed


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/root/ablation')
    ap.add_argument('--excel',default='/root/ablation/station_swe_data.xlsx')
    ap.add_argument('--years',nargs='+',type=int,default=[2015,2016,2017,2018])
    args=ap.parse_args(); years=set(args.years); root=Path(args.root)
    paths={'chelsa_sfxwind':root/'sfxwind'/'cn','lst':root/'lst'/'cn','rh':root/'rh'/'cn','pr':root/'pr'/'cn'}
    axes={}
    for var,p in paths.items():
        dates,files,failed=collect(p,var,years); axes[var]=dates
        print(f'{var}: files={len(files)}, dates={len(dates)}')
        if dates: print(f'  range={min(dates):%Y-%m-%d}..{max(dates):%Y-%m-%d}')
        if failed: print(f'  unparsed={len(failed)}, examples={failed[:5]}')
    common=set.intersection(*(axes[v] for v in axes)) if all(axes.values()) else set()
    print(f'COMMON: {len(common)}')
    if common: print(f'  range={min(common):%Y-%m-%d}..{max(common):%Y-%m-%d}')
    df=pd.read_excel(args.excel)
    d=pd.to_datetime(df['date'],errors='coerce').dt.normalize()
    mask=d.dt.year.isin(years)
    matched=sum(canon(x) in common for x in d[mask].dropna())
    print(f'Excel target-year rows={int(mask.sum())}, matched common dynamic dates={matched}')

if __name__=='__main__': main()
