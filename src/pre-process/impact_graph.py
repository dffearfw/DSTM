import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import os
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings('ignore')


def plot_station_with_year_doy(data_file, station_id, start_year=None, end_year=None, save_path=None, figsize=(16, 10)):
    """
    绘制站点SWE和降水量图，横坐标同时显示年份和DOY

    参数:
    data_file: 数据文件路径
    station_id: 站点ID
    start_year: 起始年份（可选）
    end_year: 结束年份（可选）
    save_path: 图片保存路径
    figsize: 图表大小
    """

    print(f"正在读取数据: {data_file}")

    # 读取数据
    if data_file.endswith('.xlsx'):
        df = pd.read_excel(data_file)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        raise ValueError("只支持Excel或CSV格式")

    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")

    # 查找必要的列
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'station' in col_lower:
            col_mapping['station_id'] = col
        elif 'date' in col_lower:
            col_mapping['date'] = col
        elif 'year' in col_lower:
            col_mapping['year'] = col
        elif 'doy' in col_lower:
            col_mapping['doy'] = col
        elif 'swe' in col_lower:
            col_mapping['swe'] = col
        elif 'precip' in col_lower or '降水' in col_lower:
            col_mapping['precipitation'] = col

    print(f"列映射: {col_mapping}")

    # 重命名列
    df = df.rename(columns=col_mapping)

    # 筛选站点数据
    station_data = df[df['station_id'] == station_id].copy()

    if len(station_data) == 0:
        print(f"未找到站点 {station_id} 的数据")
        print(f"可用站点: {df['station_id'].unique()[:10]}")
        return None

    print(f"找到站点 {station_id} 的 {len(station_data)} 条记录")

    # 确保有日期信息
    if 'date' in station_data.columns:
        station_data['date'] = pd.to_datetime(station_data['date'], errors='coerce')
        station_data['year'] = station_data['date'].dt.year
        station_data['month'] = station_data['date'].dt.month
        station_data['day'] = station_data['date'].dt.day
    else:
        # 如果没有日期列，使用year和doy创建
        if 'year' in station_data.columns and 'doy' in station_data.columns:
            station_data['date'] = pd.to_datetime(station_data['year'].astype(str) + ' ' +
                                                  station_data['doy'].astype(str), format='%Y %j', errors='coerce')

    # 确保有DOY列
    if 'doy' not in station_data.columns and 'date' in station_data.columns:
        station_data['doy'] = station_data['date'].dt.dayofyear

    # 筛选年份范围
    if start_year:
        station_data = station_data[station_data['year'] >= start_year].copy()
    if end_year:
        station_data = station_data[station_data['year'] <= end_year].copy()

    # 确保数据类型正确
    station_data['year'] = pd.to_numeric(station_data['year'], errors='coerce')
    station_data['doy'] = pd.to_numeric(station_data['doy'], errors='coerce')
    station_data['swe'] = pd.to_numeric(station_data['swe'], errors='coerce')
    station_data['precipitation'] = pd.to_numeric(station_data['precipitation'], errors='coerce')

    # 移除无效数据
    station_data = station_data.dropna(subset=['year', 'doy', 'swe', 'precipitation'])

    if len(station_data) == 0:
        print(f"站点 {station_id} 没有有效数据")
        return None

    print(f"有效数据: {len(station_data)} 条")
    print(f"年份范围: {station_data['year'].min()} - {station_data['year'].max()}")
    print(f"DOY范围: {station_data['doy'].min()} - {station_data['doy'].max()}")

    # 创建复合横坐标：年份 + DOY/365（用于连续显示）
    station_data = station_data.sort_values(['year', 'doy'])

    # 方法1: 使用连续索引，但在x轴上标记年份和DOY
    station_data['x_index'] = range(len(station_data))

    # 准备x轴标签：每个数据点显示"年-DOY"
    x_labels = []
    for _, row in station_data.iterrows():
        x_labels.append(f"{int(row['year'])}\n{int(row['doy'])}")

    # 方法2: 使用连续数值，其中整数部分表示年份，小数部分表示DOY/365
    station_data['year_doy'] = station_data['year'] + station_data['doy'] / 365.0

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   sharex=True)

    # 主图：SWE和降水量
    ax1_main = ax1

    # 设置x轴（使用连续索引）
    x_positions = station_data['x_index'].values
    swe_values = station_data['swe'].values
    precip_values = station_data['precipitation'].values
    years = station_data['year'].values
    doys = station_data['doy'].values

    # 绘制降水量柱状图
    bars = ax1_main.bar(x_positions, precip_values, width=1.0,
                        color='lightblue', alpha=0.7, label='Precipitation (mm/day)')

    # 绘制SWE散点图和连线
    ax1_secondary = ax1_main.twinx()
    scatter = ax1_secondary.scatter(x_positions, swe_values,
                                    color='red', s=50,
                                    edgecolors='darkred', linewidth=1.5,
                                    zorder=5, label='SWE')
    # 添加SWE连线
    ax1_secondary.plot(x_positions, swe_values, color='red',
                       alpha=0.5, linewidth=1.5, linestyle='-')

    # 设置y轴标签
    ax1_main.set_ylabel('Precipitation (mm/day)', fontsize=12,
                        fontweight='bold', color='blue')
    ax1_main.tick_params(axis='y', labelcolor='blue')

    ax1_secondary.set_ylabel('Snow Water Equivalent (SWE)', fontsize=12,
                             fontweight='bold', color='red')
    ax1_secondary.tick_params(axis='y', labelcolor='red')

    # 设置标题
    title = f'Station {station_id} - SWE and Precipitation'
    if start_year or end_year:
        title += f' ({start_year if start_year else ""}-{end_year if end_year else ""})'
    ax1_main.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # 添加网格
    ax1_main.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 设置x轴刻度
    # 找到每年的第一个数据点作为主要刻度
    unique_years = np.unique(years)
    year_ticks = []
    year_labels = []

    for year in unique_years:
        year_indices = np.where(years == year)[0]
        if len(year_indices) > 0:
            # 使用该年的第一个数据点作为刻度位置
            first_idx = year_indices[0]
            year_ticks.append(x_positions[first_idx])
            year_labels.append(str(int(year)))

    # 设置主要刻度（年份）
    ax1_main.set_xticks(year_ticks)
    ax1_main.set_xticklabels(year_labels, rotation=0, fontsize=11, fontweight='bold')

    # 添加年份分隔线
    for tick in year_ticks:
        ax1_main.axvline(x=tick, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # 创建自定义图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightblue', lw=4, label='Precipitation (mm/day)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='SWE', markeredgecolor='darkred'),
        Line2D([0], [0], color='red', lw=2, label='SWE Trend')
    ]
    ax1_main.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # 第二个子图：显示DOY的详细刻度
    ax2.bar(x_positions, [1] * len(x_positions), width=1.0,  # 创建一个空的柱状图作为基础
            color='lightgray', alpha=0.3)

    # 在第二个子图上显示DOY值
    for i, (x, doy) in enumerate(zip(x_positions, doys)):
        # 每10个DOY显示一次，避免太密集
        if i % 10 == 0 or i == len(x_positions) - 1:
            ax2.text(x, 0.5, str(int(doy)), ha='center', va='center',
                     fontsize=8, rotation=90, color='darkgreen')

    ax2.set_ylabel('DOY', fontsize=10, fontweight='bold', color='darkgreen')
    ax2.set_ylim(0, 1)  # 固定y轴范围
    ax2.set_yticks([])  # 隐藏y轴刻度
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')

    # 同步x轴范围
    ax1_main.set_xlim(x_positions[0] - 1, x_positions[-1] + 1)
    ax2.set_xlim(x_positions[0] - 1, x_positions[-1] + 1)

    # 添加统计信息文本框
    stats_text = f"""Statistics:
    Years: {station_data['year'].min()} - {station_data['year'].max()}
    SWE Range: {swe_values.min():.2f} - {swe_values.max():.2f}
    SWE Mean: {swe_values.mean():.2f}
    Precipitation Range: {precip_values.min():.2f} - {precip_values.max():.2f}
    Data Points: {len(station_data)}"""

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1_main.text(0.02, 0.98, stats_text, transform=ax1_main.transAxes,
                  fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 保存或显示图片
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    else:
        plt.show()

    # 返回数据和图表
    return {
        'station_data': station_data,
        'figure': fig,
        'axes': (ax1_main, ax1_secondary, ax2)
    }


def plot_station_separate_years(data_file, station_id, save_dir=None, figsize=(14, 8)):
    """
    为每个年份单独绘制子图
    """

    # 读取数据
    df = pd.read_excel(data_file) if data_file.endswith('.xlsx') else pd.read_csv(data_file)

    # 简化列名处理
    df = df.rename(columns=lambda x: str(x).lower())

    # 筛选站点数据
    station_data = df[df['station_id'] == station_id].copy()

    if len(station_data) == 0:
        print(f"未找到站点 {station_id}")
        return None

    # 确保有必要的列
    if 'date' in station_data.columns:
        station_data['date'] = pd.to_datetime(station_data['date'], errors='coerce')
        station_data['year'] = station_data['date'].dt.year
        station_data['doy'] = station_data['date'].dt.dayofyear

    # 获取所有年份
    years = sorted(station_data['year'].unique())

    # 计算子图布局
    n_years = len(years)
    n_cols = min(3, n_years)
    n_rows = (n_years + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, year in enumerate(years):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # 筛选该年数据
        year_data = station_data[station_data['year'] == year].copy()
        year_data = year_data.sort_values('doy')

        if len(year_data) == 0:
            ax.text(0.5, 0.5, f"No data for {year}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Year {year} (No Data)", fontsize=10)
            continue

        # 创建双y轴
        ax2 = ax.twinx()

        # 绘制降水量柱状图
        ax.bar(year_data['doy'], year_data['precipitation'],
               color='lightblue', alpha=0.7, width=1.0, label='Precipitation')

        # 绘制SWE散点图和连线
        ax2.scatter(year_data['doy'], year_data['swe'],
                    color='red', s=30, edgecolors='darkred',
                    linewidth=1, label='SWE', zorder=5)
        ax2.plot(year_data['doy'], year_data['swe'],
                 color='red', alpha=0.5, linewidth=1.5)

        # 设置标题和标签
        ax.set_title(f"Year {year}", fontsize=11, fontweight='bold')
        ax.set_xlabel('DOY', fontsize=9)
        ax.set_ylabel('Precipitation (mm/day)', fontsize=9, color='blue')
        ax2.set_ylabel('SWE', fontsize=9, color='red')

        # 设置x轴刻度
        doy_min, doy_max = year_data['doy'].min(), year_data['doy'].max()
        doy_range = doy_max - doy_min

        if doy_range <= 30:
            step = 5
        elif doy_range <= 100:
            step = 10
        else:
            step = 30

        x_ticks = np.arange(max(1, doy_min - doy_min % step), doy_max + step, step)
        ax.set_xticks(x_ticks)

        # 添加图例
        if idx == 0:  # 只在第一个子图添加图例
            lines_labels = [ax.get_legend_handles_labels()[0][0],
                            ax2.get_legend_handles_labels()[0][0]]
            ax.legend(lines_labels, ['Precipitation', 'SWE'],
                      loc='upper right', fontsize=8)

        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')

    # 隐藏多余的子图
    for idx in range(len(years), len(axes)):
        axes[idx].set_visible(False)

    # 设置总标题
    fig.suptitle(f'Station {station_id} - Yearly SWE and Precipitation',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # 保存或显示
    if save_dir:
        save_path = os.path.join(save_dir, f"station_{station_id}_yearly_subplots.png")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多子图保存到: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    # 配置参数
    data_file = "E:/pycharmworkspace/fusing-xgb/src/training/lu_onehot - 副本.xlsx"  # 修改为你的文件路径
    station_id = "55960"  # 你的站点ID
    output_dir = "E:/pycharmworkspace/DSTM/output/plots"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("站点SWE和降水量绘图工具")
    print("=" * 60)

    # 选择绘图方式
    print("\n选择绘图方式:")
    print("1. 单图显示所有年份（带年份和DOY双坐标）")
    print("2. 每年单独子图")

    choice = input("请选择 (1/2, 默认1): ").strip() or "1"

    if choice == "1":
        # 设置年份范围（可选）
        start_year = input("起始年份 (可选，直接回车跳过): ").strip()
        end_year = input("结束年份 (可选，直接回车跳过): ").strip()

        start_year = int(start_year) if start_year and start_year.isdigit() else None
        end_year = int(end_year) if end_year and end_year.isdigit() else None

        # 绘图
        save_path = os.path.join(output_dir, f"station_{station_id}_combined.png")
        result = plot_station_with_year_doy(
            data_file=data_file,
            station_id=station_id,
            start_year=start_year,
            end_year=end_year,
            save_path=save_path,
            figsize=(18, 10)
        )

        if result is not None:
            print(f"\n绘图完成!")
            station_data = result['station_data']
            print(f"数据统计:")
            print(f"年份: {station_data['year'].unique()}")
            print(f"数据点: {len(station_data)}")
            print(f"SWE均值: {station_data['swe'].mean():.2f}")
            print(f"降水量均值: {station_data['precipitation'].mean():.2f}")

    else:
        # 每年单独子图
        save_path = os.path.join(output_dir, f"station_{station_id}_yearly_subplots")
        fig = plot_station_separate_years(
            data_file=data_file,
            station_id=station_id,
            save_dir=save_path,
            figsize=(16, 12)
        )

        if fig is not None:
            print(f"\n多子图绘制完成!")