#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
站点SWE vs FusedSWE产品值对比分析（简化版）
error = 产品SWE - 站点SWE（正=高估，负=低估）
abs_error = |产品SWE - 站点SWE|
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import rasterio
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============= 配置 =============
STATION_CSV = Path(r"/root/autodl-tmp/combined_station.csv")
FUSED_SWE_ROOT = Path(r"/root/autodl-tmp/ablation/fusedswe/cn")
OUTPUT_DIR = Path(r"/root/autodl-tmp/station_vs_fused_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_YEARS = [2015, 2016, 2017]

# FusedSWE 网格参数
LON_MIN = 71.974
LON_MAX = 136.724
LAT_MIN = 16.311
LAT_MAX = 54.311
RES = 0.25

H = int((LAT_MAX - LAT_MIN) / RES) + 1
W = int((LON_MAX - LON_MIN) / RES) + 1

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def lonlat_to_pixel(lon, lat):
    col = int((lon - LON_MIN) / RES)
    row = int((LAT_MAX - lat) / RES)
    return row, col


def load_station_data(csv_path):
    print(f"\nLoading station data: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year.isin(TARGET_YEARS)].copy()
    df = df.dropna(subset=['swe', 'longitude', 'latitude'])
    print(f"  Valid station records: {len(df)}")
    return df


def extract_fused_swe_values(df):
    print(f"\nExtracting FusedSWE values...")
    
    date_to_file = {}
    for f in FUSED_SWE_ROOT.glob("*.tif"):
        match = re.search(r'(\d{4})(\d{2})(\d{2})', f.stem)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            date = datetime(year, month, day)
            if year in TARGET_YEARS:
                date_to_file[date] = f
    
    print(f"  Found {len(date_to_file)} date files")
    
    fused_values = []
    success = 0
    
    for idx, row in df.iterrows():
        date = row['date']
        lon = row['longitude']
        lat = row['latitude']
        
        if date not in date_to_file:
            fused_values.append(np.nan)
            continue
        
        r, c = lonlat_to_pixel(lon, lat)
        
        if r < 0 or r >= H or c < 0 or c >= W:
            fused_values.append(np.nan)
            continue
        
        try:
            with rasterio.open(date_to_file[date]) as ds:
                val = ds.read(1)[r, c]
                nodata = ds.nodata
                if nodata is not None and val == nodata:
                    fused_values.append(np.nan)
                else:
                    fused_values.append(float(val))
                    success += 1
        except:
            fused_values.append(np.nan)
    
    df['fused_swe'] = fused_values
    print(f"  Successfully matched: {success}/{len(df)} ({success/len(df)*100:.1f}%)")
    
    return df


def export_zero_misclassifications(df):
    """导出雪区误判为0的站点（站点有雪，产品为0）"""
    zero_misclass = df[(df['swe'] > 0) & (df['fused_swe'] == 0)].copy()
    
    print(f"\n{'='*60}")
    print(f"Snow area misclassified as 0 (Station SWE > 0, FusedSWE = 0)")
    print(f"{'='*60}")
    print(f"  Total: {len(zero_misclass)} samples")
    print(f"  Percentage: {len(zero_misclass)/len(df)*100:.2f}%")
    
    if len(zero_misclass) == 0:
        print("  No such samples")
        return
    
    # 保存为 txt 文件
    txt_path = OUTPUT_DIR / 'zero_misclassifications.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Snow Area Misclassified as 0 (Station SWE > 0, FusedSWE = 0)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total: {len(zero_misclass)} samples\n")
        f.write(f"Percentage: {len(zero_misclass)/len(df)*100:.2f}%\n\n")
        
        f.write(f"{'Index':<8} {'Date':<12} {'Station_ID':<30} {'Lon':<12} {'Lat':<12} {'Station_SWE':<12}\n")
        f.write("-"*90 + "\n")
        
        for i, (idx, row) in enumerate(zero_misclass.iterrows()):
            f.write(f"{i+1:<8} {row['date'].strftime('%Y-%m-%d'):<12} {str(row['station_id'])[:30]:<30} "
                    f"{row['longitude']:<12.2f} {row['latitude']:<12.2f} {row['swe']:<12.2f}\n")
    
    print(f"  Saved to: {txt_path}")
    
    return zero_misclass


def plot_all(valid_df, metrics, valid_no_zero_df, metrics_no_zero):
    """生成所有图表"""
    print(f"\nGenerating plots...")
    
    station_swe = valid_df['swe'].values
    fused_swe = valid_df['fused_swe'].values
    error = valid_df['error'].values          # 有符号误差
    abs_error = valid_df['abs_error'].values  # 绝对误差
    
    station_swe_no_zero = valid_no_zero_df['swe'].values
    fused_swe_no_zero = valid_no_zero_df['fused_swe'].values
    
    # 1. 原始散点图（包含0值误判点）
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(station_swe, fused_swe, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    
    min_val = min(station_swe.min(), fused_swe.min())
    max_val = max(station_swe.max(), fused_swe.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
    
    z = np.polyfit(station_swe, fused_swe, 1)
    ax.plot([min_val, max_val], np.poly1d(z)([min_val, max_val]), 'g-', linewidth=2,
            label=f'Regression: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Station SWE (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('FusedSWE (mm)', fontsize=14, fontweight='bold')
    ax.set_title('Station SWE vs FusedSWE (All samples)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    text = f'N={metrics["n_samples"]}\nRMSE={metrics["rmse"]:.2f}mm\nMAE={metrics["mae"]:.2f}mm\nBias={metrics["bias"]:.2f}mm\nR²={metrics["r2"]:.4f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Scatter plot (all samples)")
    
    # 2. 去除 FusedSWE=0 后的散点图
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(station_swe_no_zero, fused_swe_no_zero, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    
    min_val_no_zero = min(station_swe_no_zero.min(), fused_swe_no_zero.min())
    max_val_no_zero = max(station_swe_no_zero.max(), fused_swe_no_zero.max())
    ax.plot([min_val_no_zero, max_val_no_zero], [min_val_no_zero, max_val_no_zero], 'r--', linewidth=2, label='1:1 Line')
    
    z_no_zero = np.polyfit(station_swe_no_zero, fused_swe_no_zero, 1)
    ax.plot([min_val_no_zero, max_val_no_zero], np.poly1d(z_no_zero)([min_val_no_zero, max_val_no_zero]), 'g-', linewidth=2,
            label=f'Regression: y={z_no_zero[0]:.2f}x+{z_no_zero[1]:.2f}')
    
    ax.set_xlabel('Station SWE (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('FusedSWE (mm)', fontsize=14, fontweight='bold')
    ax.set_title('Station SWE vs FusedSWE (Remove FusedSWE=0)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    text_no_zero = f'N={metrics_no_zero["n_samples"]}\nRMSE={metrics_no_zero["rmse"]:.2f}mm\nMAE={metrics_no_zero["mae"]:.2f}mm\nBias={metrics_no_zero["bias"]:.2f}mm\nR²={metrics_no_zero["r2"]:.4f}'
    ax.text(0.05, 0.95, text_no_zero, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_plot_no_zero.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Scatter plot (remove FusedSWE=0)")
    
    # 3. 直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(station_swe, bins=50, alpha=0.5, label='Station SWE', color='steelblue', edgecolor='black')
    axes[0].hist(fused_swe, bins=50, alpha=0.5, label='FusedSWE', color='orange', edgecolor='black')
    axes[0].set_xlabel('SWE (mm)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 误差分布（error = 产品 - 站点）
    axes[1].hist(error, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[1].axvline(x=metrics['bias'], color='blue', linestyle='-', linewidth=2, label=f'Mean error: {metrics["bias"]:.2f}mm')
    axes[1].set_xlabel('Error (FusedSWE - Station) (mm)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution (Positive = Overestimate)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Histogram")
    
    # 4. 箱线图
    valid_df['year'] = valid_df['date'].dt.year
    years = sorted(valid_df['year'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    station_by_year = [valid_df[valid_df['year'] == y]['swe'].values for y in years]
    fused_by_year = [valid_df[valid_df['year'] == y]['fused_swe'].values for y in years]
    
    positions1 = np.arange(len(years)) * 2 - 0.2
    positions2 = np.arange(len(years)) * 2 + 0.2
    
    bp1 = ax.boxplot(station_by_year, positions=positions1, widths=0.35, patch_artist=True,
                     boxprops=dict(facecolor='steelblue', alpha=0.7))
    bp2 = ax.boxplot(fused_by_year, positions=positions2, widths=0.35, patch_artist=True,
                     boxprops=dict(facecolor='orange', alpha=0.7))
    
    ax.set_xticks(np.arange(len(years)) * 2)
    ax.set_xticklabels(years)
    ax.set_xlabel('Year')
    ax.set_ylabel('SWE (mm)')
    ax.set_title('Yearly SWE Distribution')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Station SWE', 'FusedSWE'])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Boxplot")
    
    # 5. 绝对误差 vs SWE
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(station_swe, abs_error, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    
    bins = np.percentile(station_swe, np.linspace(0, 100, 11))
    bin_centers, bin_means, bin_stds = [], [], []
    
    for i in range(len(bins)-1):
        mask = (station_swe >= bins[i]) & (station_swe < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(abs_error[mask]))
            bin_stds.append(np.std(abs_error[mask]))
    
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='ro-', markersize=8, linewidth=2,
                label='Mean absolute error per bin')
    
    ax.set_xlabel('Station SWE (mm)')
    ax.set_ylabel('Absolute Error (mm)')
    ax.set_title('Absolute Error vs SWE Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'error_vs_swe.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Error vs SWE plot")


def main():
    print("="*70)
    print("Station SWE vs FusedSWE Analysis")
    print("="*70)
    
    print(f"\nGrid parameters:")
    print(f"  Longitude range: {LON_MIN}° ~ {LON_MAX}°")
    print(f"  Latitude range: {LAT_MIN}° ~ {LAT_MAX}°")
    print(f"  Resolution: {RES}°")
    print(f"  Grid size: {H} rows × {W} cols")
    
    # 1. Load station data
    df = load_station_data(STATION_CSV)
    
    # 2. Extract FusedSWE values
    df = extract_fused_swe_values(df)
    
    # 3. 计算误差（产品 - 站点）
    df['error'] = df['fused_swe'] - df['swe']
    df['abs_error'] = np.abs(df['error'])
    
    # 4. Export zero misclassifications
    zero_df = export_zero_misclassifications(df)
    
    # 5. 原始有效数据
    valid_df = df.dropna(subset=['swe', 'fused_swe', 'error']).copy()
    print(f"\nValid comparison samples (all): {len(valid_df)}")
    
    # 6. 去除 FusedSWE=0 后的有效数据
    valid_no_zero_df = valid_df[valid_df['fused_swe'] > 0].copy()
    print(f"Valid comparison samples (remove FusedSWE=0): {len(valid_no_zero_df)}")
    print(f"Removed {len(valid_df) - len(valid_no_zero_df)} samples with FusedSWE=0")
    
    if len(valid_df) == 0:
        print("No valid data")
        return
    
    # 7. 计算原始指标
    station_swe = valid_df['swe'].values
    fused_swe = valid_df['fused_swe'].values
    error = valid_df['error'].values
    
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    bias = np.mean(error)
    ss_res = np.sum(error ** 2)
    ss_tot = np.sum((station_swe - np.mean(station_swe)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    r, p = stats.pearsonr(station_swe, fused_swe)
    
    metrics = {
        'n_samples': len(valid_df),
        'rmse': rmse, 'mae': mae, 'bias': bias, 'r2': r2, 'r': r,
    }
    
    print(f"\n{'='*60}")
    print(f"Statistics Results (All samples)")
    print(f"{'='*60}")
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  Station SWE range: [{station_swe.min():.2f}, {station_swe.max():.2f}] mm")
    print(f"  FusedSWE range: [{fused_swe.min():.2f}, {fused_swe.max():.2f}] mm")
    print(f"  RMSE: {rmse:.2f} mm")
    print(f"  MAE: {mae:.2f} mm")
    print(f"  Bias (product - station): {bias:.2f} mm")
    print(f"  R²: {r2:.4f}")
    print(f"  R: {r:.4f}")
    
    # 8. 计算去除0值后的指标
    if len(valid_no_zero_df) > 0:
        error_no_zero = valid_no_zero_df['error'].values
        station_swe_no_zero = valid_no_zero_df['swe'].values
        fused_swe_no_zero = valid_no_zero_df['fused_swe'].values
        
        rmse_no_zero = np.sqrt(np.mean(error_no_zero ** 2))
        mae_no_zero = np.mean(np.abs(error_no_zero))
        bias_no_zero = np.mean(error_no_zero)
        ss_res_no_zero = np.sum(error_no_zero ** 2)
        ss_tot_no_zero = np.sum((station_swe_no_zero - np.mean(station_swe_no_zero)) ** 2)
        r2_no_zero = 1 - (ss_res_no_zero / ss_tot_no_zero) if ss_tot_no_zero > 0 else 0
        r_no_zero, p_no_zero = stats.pearsonr(station_swe_no_zero, fused_swe_no_zero)
        
        metrics_no_zero = {
            'n_samples': len(valid_no_zero_df),
            'rmse': rmse_no_zero, 'mae': mae_no_zero, 'bias': bias_no_zero,
            'r2': r2_no_zero, 'r': r_no_zero
        }
        
        print(f"\n{'='*60}")
        print(f"Statistics Results (Remove FusedSWE=0)")
        print(f"{'='*60}")
        print(f"  Samples: {metrics_no_zero['n_samples']}")
        print(f"  Station SWE range: [{station_swe_no_zero.min():.2f}, {station_swe_no_zero.max():.2f}] mm")
        print(f"  FusedSWE range: [{fused_swe_no_zero.min():.2f}, {fused_swe_no_zero.max():.2f}] mm")
        print(f"  RMSE: {rmse_no_zero:.2f} mm")
        print(f"  MAE: {mae_no_zero:.2f} mm")
        print(f"  Bias (product - station): {bias_no_zero:.2f} mm")
        print(f"  R²: {r2_no_zero:.4f}")
        print(f"  R: {r_no_zero:.4f}")
        
        # 计算改进幅度
        rmse_improve = (rmse - rmse_no_zero) / rmse * 100
        mae_improve = (mae - mae_no_zero) / mae * 100
        r2_improve = (r2_no_zero - r2) / abs(r2) * 100 if r2 != 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Improvement after removing FusedSWE=0")
        print(f"{'='*60}")
        print(f"  RMSE: {rmse:.2f} → {rmse_no_zero:.2f} mm ({rmse_improve:+.1f}%)")
        print(f"  MAE:  {mae:.2f} → {mae_no_zero:.2f} mm ({mae_improve:+.1f}%)")
        print(f"  R²:   {r2:.4f} → {r2_no_zero:.4f} ({r2_improve:+.1f}%)")
    else:
        metrics_no_zero = metrics
    
    # 9. Generate plots
    plot_all(valid_df, metrics, valid_no_zero_df, metrics_no_zero)
    
    # 10. Save detailed data
    sorted_df = valid_df.sort_values('abs_error', ascending=False)
    
    csv_path = OUTPUT_DIR / 'station_vs_fused_detailed.csv'
    sorted_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  ✓ Detailed data: {csv_path}")
    
    # 11. Print top 20 errors
    print(f"\n{'='*70}")
    print(f"Top 20 Largest Absolute Errors")
    print(f"{'='*70}")
    print(f"{'Date':<12} {'Station ID':<25} {'Lon':<10} {'Lat':<10} {'Station SWE':<12} {'FusedSWE':<12} {'Abs Error':<10}")
    print("-" * 95)
    
    for _, row in sorted_df.head(20).iterrows():
        station_id = str(row['station_id'])[:25]
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} {station_id:<25} "
              f"{row['longitude']:<10.2f} {row['latitude']:<10.2f} "
              f"{row['swe']:<12.2f} {row['fused_swe']:<12.2f} {row['abs_error']:<10.2f}")
    
    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
