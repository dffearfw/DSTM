import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import os
import sys
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')


def plot_station_final(data_file, station_id, save_path=None, figsize=(20, 12), show_plot=True):
    """
    最终版：DOY标签黑色，紧贴下沿
    """

    print(f"正在读取数据: {data_file}")

    try:
        # 读取数据
        if data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        elif data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            try:
                df = pd.read_excel(data_file)
            except:
                df = pd.read_csv(data_file)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    print(f"数据形状: {df.shape}")

    # 查找必要的列
    col_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if 'station' in col_lower or '站' in col_lower or 'id' == col_lower:
            col_mapping['station_id'] = col
        elif 'date' in col_lower or '日期' in col_lower:
            col_mapping['date'] = col
        elif 'year' in col_lower or '年' in col_lower:
            col_mapping['year'] = col
        elif 'doy' in col_lower or '年积日' in col_lower:
            col_mapping['doy'] = col
        elif 'swe' in col_lower or '雪水当量' in col_lower:
            col_mapping['swe'] = col
        elif 'precip' in col_lower or '降水' in col_lower:
            col_mapping['precipitation'] = col

    print(f"检测到的列映射: {col_mapping}")

    # 重命名列
    for new_col, old_col in col_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]

    # 筛选站点数据
    if 'station_id' not in df.columns:
        print("错误: 未找到站点ID列")
        return None

    df['station_id_str'] = df['station_id'].astype(str).str.strip()
    station_id_str = str(station_id).strip()

    station_data = df[df['station_id_str'] == station_id_str].copy()

    if len(station_data) == 0:
        print(f"未找到站点 {station_id} 的数据")
        print(f"可用站点ID示例: {df['station_id_str'].unique()[:5]}")
        return None

    print(f"找到站点 {station_id} 的 {len(station_data)} 条记录")

    # 处理数据
    if 'date' in station_data.columns:
        station_data['date'] = pd.to_datetime(station_data['date'], errors='coerce')

    # 确保有年份和DOY
    if 'year' not in station_data.columns:
        if 'date' in station_data.columns and not station_data['date'].isna().all():
            station_data['year'] = station_data['date'].dt.year
        else:
            station_data['year'] = 2000

    if 'doy' not in station_data.columns:
        if 'date' in station_data.columns and not station_data['date'].isna().all():
            station_data['doy'] = station_data['date'].dt.dayofyear
        else:
            station_data['doy'] = np.arange(len(station_data)) % 365 + 1

    # 数据类型转换
    for col in ['year', 'doy', 'swe', 'precipitation']:
        if col in station_data.columns:
            station_data[col] = pd.to_numeric(station_data[col], errors='coerce')

    # 检查数据
    has_swe = 'swe' in station_data.columns and not station_data['swe'].isna().all()
    has_precip = 'precipitation' in station_data.columns and not station_data['precipitation'].isna().all()

    if not has_swe and not has_precip:
        print("错误: 没有数据可绘制")
        return None

    # 按时间排序
    if 'date' in station_data.columns and not station_data['date'].isna().all():
        station_data = station_data.sort_values('date')
    else:
        station_data = station_data.sort_values(['year', 'doy'])

    station_data = station_data.reset_index(drop=True)
    station_data['x_index'] = range(len(station_data))

    # 准备数据
    x_positions = station_data['x_index'].values
    years = station_data['year'].values
    doys = station_data['doy'].values

    if has_swe:
        swe_values = station_data['swe'].values
        valid_swe = swe_values[~np.isnan(swe_values)]

    if has_precip:
        precip_values = station_data['precipitation'].values
        valid_precip = precip_values[~np.isnan(precip_values)]

    # 创建图表
    fig, ax1_main = plt.subplots(figsize=figsize)

    # 绘制降水量
    if has_precip:
        bars = ax1_main.bar(x_positions, precip_values, width=1.0,
                            color='lightblue', alpha=0.7, label='Precipitation (mm/day)')

    # 绘制SWE
    if has_swe:
        ax1_secondary = ax1_main.twinx()
        scatter = ax1_secondary.scatter(x_positions, swe_values,
                                        color='red', s=50,
                                        edgecolors='darkred', linewidth=1.5,
                                        zorder=5, label='SWE (mm/day)')
        ax1_secondary.plot(x_positions, swe_values, color='red',
                           alpha=0.5, linewidth=1.5, linestyle='-')

    # 设置y轴标签
    if has_precip:
        ax1_main.set_ylabel('Precipitation (mm/day)', fontsize=14,
                            fontweight='bold', color='blue')
        ax1_main.tick_params(axis='y', labelcolor='blue')

    if has_swe:
        ax1_secondary.set_ylabel('Snow Water Equivalent - SWE (mm/day)', fontsize=14,
                                 fontweight='bold', color='red')
        ax1_secondary.tick_params(axis='y', labelcolor='red')

    # 设置标题
    title = f'Station {station_id} - SWE and Precipitation ({len(station_data)} records)'
    ax1_main.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 添加网格
    ax1_main.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 设置x轴
    unique_years = np.unique(years)
    print(f"数据涵盖 {len(unique_years)} 年")

    # 年份刻度 - 在年份开始位置
    year_ticks = []
    year_labels = []

    for year in unique_years:
        year_indices = np.where(years == year)[0]
        if len(year_indices) > 0:
            first_idx = year_indices[0]
            year_ticks.append(x_positions[first_idx])
            year_labels.append(str(int(year)))

    # 设置主x轴刻度（年份）
    ax1_main.set_xticks(year_ticks)
    ax1_main.set_xticklabels(year_labels, rotation=0, fontsize=12, fontweight='bold')

    # 添加DOY标签 - 关键修改部分
    # 选择合适的DOY显示点
    if len(x_positions) <= 100:
        # 数据点少，显示更多
        doy_display_step = max(1, len(x_positions) // 20)
    else:
        # 数据点多，显示较少
        doy_display_step = max(1, len(x_positions) // 30)

    doy_indices = list(range(0, len(x_positions), doy_display_step))
    if len(x_positions) - 1 not in doy_indices:
        doy_indices.append(len(x_positions) - 1)

    # 创建次要x轴用于显示DOY标签
    ax_doy = ax1_main.secondary_xaxis('bottom')

    # 设置DOY刻度的位置（使用相同的位置）
    doy_tick_positions = x_positions[doy_indices]
    doy_labels = [str(int(doys[i])) for i in doy_indices]

    ax_doy.set_xticks(doy_tick_positions)
    ax_doy.set_xticklabels(doy_labels, fontsize=9, rotation=0, color='black')

    # 调整DOY标签位置，紧贴下沿
    ax_doy.set_xlabel('DOY', fontsize=10, fontweight='bold', labelpad=5)

    # 隐藏主x轴的标签（年份标签在上面）
    ax1_main.set_xlabel('')

    # 添加年份分隔线
    for tick in year_ticks:
        ax1_main.axvline(x=tick, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # 设置x轴范围
    if len(x_positions) > 0:
        ax1_main.set_xlim(x_positions[0] - 1, x_positions[-1] + 1)
        ax_doy.set_xlim(x_positions[0] - 1, x_positions[-1] + 1)

    # 创建图例 - 放在右上角
    legend_elements = []

    if has_precip:
        legend_elements.append(Patch(facecolor='lightblue', alpha=0.7,
                                     label='Precipitation (mm/day)'))

    if has_swe:
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='red', markersize=10,
                                      label='SWE (mm/day)', markeredgecolor='darkred'))
        legend_elements.append(Line2D([0], [0], color='red', lw=2,
                                      label='SWE Trend'))

    if legend_elements:
        ax1_main.legend(handles=legend_elements, loc='upper right', fontsize=11,
                        framealpha=0.9, fancybox=True)

    # 统计信息 - 放在左上角，位置高一点
    stats_text = f"Station {station_id}\n"
    stats_text += f"Total: {len(station_data)} records\n"

    if 'year' in station_data.columns:
        stats_text += f"Years: {int(station_data['year'].min())}-{int(station_data['year'].max())}\n"

    if has_swe and len(valid_swe) > 0:
        stats_text += f"SWE Range: {valid_swe.min():.1f}-{valid_swe.max():.1f}\n"
        stats_text += f"SWE Mean: {valid_swe.mean():.1f}\n"

    if has_precip and len(valid_precip) > 0:
        stats_text += f"Precip Range: {valid_precip.min():.1f}-{valid_precip.max():.1f}\n"
        stats_text += f"Precip Total: {valid_precip.sum():.0f} mm\n"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    # 位置更高：y=0.97（接近顶部）
    ax1_main.text(0.02, 0.97, stats_text, transform=ax1_main.transAxes,
                  fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 调整子图间距，给DOY标签更多空间
    plt.subplots_adjust(bottom=0.12)

    # 保存图片
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存: {save_path}")
        except Exception as e:
            print(f"保存图片失败: {e}")
            save_path = f"station_{station_id}_final.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        save_path = f"station_{station_id}_final.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"最终图片: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        'station_data': station_data,
        'save_path': save_path,
        'total_records': len(station_data),
        'has_swe': has_swe,
        'has_precip': has_precip
    }


# 更简单的版本，但保证DOY标签显示正确
def plot_station_simple_and_clean(data_file, station_id):
    """
    简洁干净的版本
    """

    print(f"读取数据: {data_file}")

    try:
        if data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        else:
            df = pd.read_csv(data_file)
    except Exception as e:
        print(f"读取失败: {e}")
        return None

    # 查找列
    station_col = None
    swe_col = None
    precip_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if 'station' in col_lower or 'id' in col_lower:
            station_col = col
        elif 'swe' in col_lower:
            swe_col = col
        elif 'precip' in col_lower:
            precip_col = col

    if not station_col:
        print("未找到站点列")
        return None

    # 筛选数据
    station_data = df[df[station_col].astype(str).str.strip() == str(station_id).strip()].copy()

    if len(station_data) == 0:
        print(f"未找到站点 {station_id}")
        return None

    print(f"找到 {len(station_data)} 条记录")

    # 创建图表
    fig = plt.figure(figsize=(18, 10))

    # 创建主坐标轴
    ax1 = plt.gca()

    # x轴位置
    x_pos = np.arange(len(station_data))

    # 绘制降水量
    if precip_col:
        ax1.bar(x_pos, station_data[precip_col],
                color='lightblue', alpha=0.7, width=1.0,
                label='Precipitation (mm/day)')
        ax1.set_ylabel('Precipitation (mm/day)', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

    # 绘制SWE
    if swe_col:
        ax2 = ax1.twinx()
        ax2.scatter(x_pos, station_data[swe_col],
                    color='red', s=60, edgecolor='darkred',
                    linewidth=1.5, zorder=5, label='SWE (mm/day)')
        ax2.plot(x_pos, station_data[swe_col],
                 color='red', alpha=0.5, linewidth=2)
        ax2.set_ylabel('SWE (mm/day)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    # 设置标题
    plt.title(f'Station {station_id} - SWE and Precipitation',
              fontsize=14, fontweight='bold')

    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 设置x轴 - 简单显示数据点序号
    # 年份标签（如果数据中有年份信息）
    if 'year' in station_data.columns:
        # 找到每年开始的位置
        unique_years = station_data['year'].unique()
        year_ticks = []
        year_labels = []

        for year in unique_years:
            indices = station_data[station_data['year'] == year].index
            if len(indices) > 0:
                year_ticks.append(indices[0])
                year_labels.append(str(int(year)))

        if len(year_ticks) > 0:
            ax1.set_xticks(year_ticks)
            ax1.set_xticklabels(year_labels, fontsize=11, fontweight='bold')

            # 添加年份分隔线
            for tick in year_ticks:
                ax1.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)

    # DOY标签 - 在图表底部创建次要x轴
    if 'doy' in station_data.columns:
        # 创建次要x轴显示DOY
        ax_doy = ax1.secondary_xaxis('bottom')

        # 选择一些位置显示DOY
        step = max(1, len(station_data) // 20)
        doy_positions = list(range(0, len(station_data), step))
        if len(station_data) - 1 not in doy_positions:
            doy_positions.append(len(station_data) - 1)

        doy_labels = [str(int(station_data.loc[i, 'doy'])) for i in doy_positions]

        ax_doy.set_xticks(doy_positions)
        ax_doy.set_xticklabels(doy_labels, fontsize=9, color='black')
        ax_doy.set_xlabel('DOY', fontsize=10, labelpad=8)

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    if swe_col:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper right', fontsize=10)

    # 统计信息
    stats_text = f"Records: {len(station_data)}"
    if 'year' in station_data.columns:
        stats_text += f"\nYears: {int(station_data['year'].min())}-{int(station_data['year'].max())}"

    ax1.text(0.02, 0.97, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    # 保存
    save_path = f"station_{station_id}_clean.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片保存到: {save_path}")

    plt.show()

    return station_data


if __name__ == "__main__":
    print("=" * 60)
    print("站点数据绘图工具 - 最终版")
    print("=" * 60)

    # 获取输入
    data_file = input("数据文件路径: ").strip()
    station_id = input("站点ID: ").strip()

    if not data_file or not station_id:
        print("需要文件路径和站点ID")
        sys.exit(1)

    if not os.path.exists(data_file):
        print(f"文件不存在: {data_file}")
        sys.exit(1)

    # 使用最终版
    result = plot_station_final(
        data_file=data_file,
        station_id=station_id,
        save_path=None,
        figsize=(20, 12),
        show_plot=True
    )

    if result:
        print(f"\n绘图完成!")
        print(f"数据点: {result['total_records']}")
        print(f"图片: {result['save_path']}")