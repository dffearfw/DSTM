import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')


def plot_station_doy_swe_precipitation(data_file, station_id, year=None, save_path=None, figsize=(14, 8)):
    """
    绘制站点DOY的SWE和降水量图

    参数:
    data_file: 包含数据的Excel或CSV文件路径
    station_id: 站点ID
    year: 年份，如果不指定则使用所有年份
    save_path: 图片保存路径，如果不指定则显示图片
    figsize: 图表大小
    """

    print(f"正在读取数据文件: {data_file}")

    # 1. 读取数据
    if data_file.endswith('.xlsx') or data_file.endswith('.xls'):
        df = pd.read_excel(data_file)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        raise ValueError("只支持Excel或CSV格式")

    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")

    # 2. 查找列名（大小写不敏感）
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'station' in col_lower:
            col_mapping['station_id'] = col
        elif 'date' in col_lower or 'time' in col_lower:
            col_mapping['date'] = col
        elif 'doy' in col_lower or 'dayofyear' in col_lower:
            col_mapping['DOY'] = col
        elif 'swe' in col_lower:
            col_mapping['swe'] = col
        elif 'precip' in col_lower or '降水' in col_lower:
            col_mapping['precipitation_mm_day'] = col

    print(f"检测到的列映射: {col_mapping}")

    # 重命名列
    df = df.rename(columns=col_mapping)

    # 确保必要的列存在
    required_cols = ['station_id', 'DOY', 'swe', 'precipitation_mm_day']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"警告: 缺少以下列: {missing_cols}")
        print(f"可用列: {df.columns.tolist()}")

        # 尝试使用相似列
        if 'swe' not in df.columns:
            swe_cols = [col for col in df.columns if 'swe' in str(col).lower() or '雪水当量' in str(col)]
            if swe_cols:
                df['swe'] = df[swe_cols[0]]
                print(f"使用 {swe_cols[0]} 作为swe列")

        if 'precipitation_mm_day' not in df.columns:
            precip_cols = [col for col in df.columns if 'precip' in str(col).lower() or '降水' in str(col)]
            if precip_cols:
                df['precipitation_mm_day'] = df[precip_cols[0]]
                print(f"使用 {precip_cols[0]} 作为降水列")

    # 3. 筛选指定站点的数据
    station_data = df[df['station_id'] == station_id].copy()

    if len(station_data) == 0:
        print(f"未找到站点 {station_id} 的数据")
        print(f"可用的站点ID: {df['station_id'].unique()[:10]}")
        return None

    print(f"找到站点 {station_id} 的 {len(station_data)} 条记录")

    # 4. 如果有年份筛选，只保留该年份数据
    if 'date' in df.columns and year is not None:
        station_data['date'] = pd.to_datetime(station_data['date'], errors='coerce')
        station_data = station_data[station_data['date'].dt.year == year].copy()
        print(f"筛选 {year} 年数据: {len(station_data)} 条")

    if len(station_data) == 0:
        print(f"站点 {station_id} 在 {year} 年没有数据")
        return None

    # 5. 确保数据类型正确
    station_data['DOY'] = pd.to_numeric(station_data['DOY'], errors='coerce')
    station_data['swe'] = pd.to_numeric(station_data['swe'], errors='coerce')
    station_data['precipitation_mm_day'] = pd.to_numeric(station_data['precipitation_mm_day'], errors='coerce')

    # 移除无效值
    station_data = station_data.dropna(subset=['DOY', 'swe', 'precipitation_mm_day'])

    if len(station_data) == 0:
        print(f"站点 {station_id} 没有有效的SWE和降水数据")
        return None

    print(f"有效数据: {len(station_data)} 条")
    print(f"DOY范围: {station_data['DOY'].min()} - {station_data['DOY'].max()}")
    print(f"SWE范围: {station_data['swe'].min():.2f} - {station_data['swe'].max():.2f}")
    print(
        f"降水范围: {station_data['precipitation_mm_day'].min():.2f} - {station_data['precipitation_mm_day'].max():.2f}")

    # 6. 创建图表
    fig, ax1 = plt.subplots(figsize=figsize)

    # 设置DOY轴
    doy_values = station_data['DOY'].values
    swe_values = station_data['swe'].values
    precip_values = station_data['precipitation_mm_day'].values

    # 排序数据（按DOY）
    sort_idx = np.argsort(doy_values)
    doy_values = doy_values[sort_idx]
    swe_values = swe_values[sort_idx]
    precip_values = precip_values[sort_idx]

    # 7. 绘制降水量柱状图（左y轴）
    bar_width = 0.8
    bars = ax1.bar(doy_values, precip_values, width=bar_width,
                   color='skyblue', alpha=0.7, label='Precipitation (mm/day)')

    ax1.set_xlabel('Day of Year (DOY)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precipitation (mm/day)', fontsize=12, fontweight='bold', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 设置x轴刻度间距
    doy_min, doy_max = int(doy_values.min()), int(doy_values.max())

    # 计算合适的刻度间距
    doy_range = doy_max - doy_min
    if doy_range <= 30:
        x_ticks = np.arange(doy_min, doy_max + 1, 5)
    elif doy_range <= 100:
        x_ticks = np.arange(doy_min, doy_max + 1, 10)
    elif doy_range <= 200:
        x_ticks = np.arange(doy_min, doy_max + 1, 20)
    else:
        x_ticks = np.arange(doy_min, doy_max + 1, 30)

    ax1.set_xticks(x_ticks)
    ax1.set_xlim(doy_min - 2, doy_max + 2)

    # 8. 创建第二个y轴用于SWE
    ax2 = ax1.twinx()

    # 绘制SWE散点图
    scatter = ax2.scatter(doy_values, swe_values, color='red', s=80,
                          edgecolors='darkred', linewidth=1.5, zorder=5,
                          label='SWE')

    # 可选：添加SWE的连线
    ax2.plot(doy_values, swe_values, color='red', alpha=0.5, linewidth=1, linestyle='-')

    ax2.set_ylabel('Snow Water Equivalent (SWE)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # 9. 添加标题和图例
    if year is not None:
        title = f'Station {station_id} - {year} Year\nSWE and Precipitation vs DOY'
    else:
        title = f'Station {station_id}\nSWE and Precipitation vs DOY'

    plt.title(title, fontsize=14, fontweight='bold', pad=20)

    # 合并图例
    lines_labels1 = [ax1.get_legend_handles_labels()[0][0]]  # 只取柱状图
    lines_labels2 = [ax2.get_legend_handles_labels()[0][0]]  # 只取散点图

    # 创建自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='skyblue', lw=4, label='Precipitation (mm/day)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='SWE', markeredgecolor='darkred')
    ]

    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # 10. 添加统计信息文本框
    stats_text = f"""Statistics:
    DOY Range: {doy_min} - {doy_max}
    SWE Range: {swe_values.min():.2f} - {swe_values.max():.2f}
    SWE Mean: {swe_values.mean():.2f}
    Precipitation Range: {precip_values.min():.2f} - {precip_values.max():.2f}
    Precipitation Total: {precip_values.sum():.2f} mm
    Data Points: {len(station_data)}"""

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # 11. 调整布局
    plt.tight_layout()

    # 12. 保存或显示图片
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()

    # 13. 返回数据用于进一步分析
    return {
        'station_data': station_data,
        'doy_values': doy_values,
        'swe_values': swe_values,
        'precip_values': precip_values,
        'figure': fig
    }


def plot_multiple_stations(data_file, station_ids, year=None, save_dir=None):
    """
    绘制多个站点的对比图
    """

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = {}
    for station_id in station_ids:
        print(f"\n{'=' * 50}")
        print(f"处理站点: {station_id}")

        if save_dir:
            save_path = os.path.join(save_dir, f"station_{station_id}_doy_plot.png")
        else:
            save_path = None

        result = plot_station_doy_swe_precipitation(
            data_file, station_id, year, save_path
        )

        if result is not None:
            results[station_id] = result

    return results


def create_comparison_plot(data_file, station_ids, year=None, save_path=None):
    """
    创建多个站点的对比图（在同一张图上）
    """

    # 读取数据
    if data_file.endswith('.xlsx'):
        df = pd.read_excel(data_file)
    else:
        df = pd.read_csv(data_file)

    # 简化的列名处理
    df = df.rename(columns=lambda x: str(x).lower())

    # 创建图形
    fig, axes = plt.subplots(len(station_ids), 1, figsize=(15, 5 * len(station_ids)))
    if len(station_ids) == 1:
        axes = [axes]

    for idx, station_id in enumerate(station_ids):
        ax = axes[idx]

        # 筛选站点数据
        station_data = df[df['station_id'] == station_id].copy()

        if len(station_data) == 0:
            continue

        if year is not None and 'date' in station_data.columns:
            station_data['date'] = pd.to_datetime(station_data['date'])
            station_data = station_data[station_data['date'].dt.year == year]

        # 确保数据类型
        station_data['doy'] = pd.to_numeric(station_data['doy'], errors='coerce')
        station_data['swe'] = pd.to_numeric(station_data['swe'], errors='coerce')
        station_data['precipitation_mm_day'] = pd.to_numeric(station_data['precipitation_mm_day'], errors='coerce')

        station_data = station_data.dropna(subset=['doy', 'swe', 'precipitation_mm_day'])

        if len(station_data) == 0:
            continue

        # 排序
        station_data = station_data.sort_values('doy')

        # 绘制
        ax2 = ax.twinx()

        # 降水量柱状图
        ax.bar(station_data['doy'], station_data['precipitation_mm_day'],
               color='lightblue', alpha=0.7, width=0.8, label='Precipitation')

        # SWE散点图
        ax2.scatter(station_data['doy'], station_data['swe'],
                    color='red', s=50, label='SWE', zorder=5)
        ax2.plot(station_data['doy'], station_data['swe'],
                 color='red', alpha=0.5, linewidth=1)

        # 设置标签
        ax.set_xlabel('DOY')
        ax.set_ylabel('Precipitation (mm/day)', color='blue')
        ax2.set_ylabel('SWE', color='red')

        # 标题
        ax.set_title(f'Station {station_id}', fontweight='bold')

        # 网格
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图保存到: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # 配置文件路径
    data_file = "你的数据文件.xlsx"  # 修改为你的文件路径
    output_dir = "输出图片"  # 图片输出目录

    # 示例1: 绘制单个站点
    station_id = "54751"  # 你的站点ID，修改为你需要的

    # 绘制图表
    result = plot_station_doy_swe_precipitation(
        data_file=data_file,
        station_id=station_id,
        year=2014,  # 可以指定年份，或者设置为None使用所有年份
        save_path=os.path.join(output_dir, f"station_{station_id}_plot.png"),
        figsize=(16, 9)
    )

    # 如果数据中包含多个年份，可以绘制每个年份的图
    if result is not None:
        # 可以进一步分析数据
        station_data = result['station_data']

        print(f"\n站点 {station_id} 的数据统计:")
        print(station_data[['DOY', 'swe', 'precipitation_mm_day']].describe())

        # 如果有日期信息，可以按月份统计
        if 'date' in station_data.columns:
            station_data['date'] = pd.to_datetime(station_data['date'])
            station_data['month'] = station_data['date'].dt.month

            print(f"\n按月统计:")
            monthly_stats = station_data.groupby('month').agg({
                'swe': ['mean', 'min', 'max', 'count'],
                'precipitation_mm_day': ['mean', 'sum', 'count']
            })
            print(monthly_stats)

    # 示例2: 绘制多个站点
    # station_ids = ["54751", "54752", "54753"]  # 你的站点列表
    # results = plot_multiple_stations(data_file, station_ids, year=2014, save_dir=output_dir)

    # 示例3: 创建对比图
    # create_comparison_plot(data_file, station_ids, year=2014,
    #                       save_path=os.path.join(output_dir, "comparison_plot.png"))