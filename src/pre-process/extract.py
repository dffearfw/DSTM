import pandas as pd
import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings


warnings.filterwarnings('ignore')



def extract_chelsa_precipitation(station_file, output_file):
    """
    从CHELSA降水数据提取像元值到Excel表格

    参数:
    station_file: 站点Excel文件路径
    output_file: 输出Excel文件路径
    """

    # 1. 读取站点数据
    print("正在读取站点数据...")
    stations = pd.read_excel(station_file)

    # 查看数据列名
    print(f"数据列名: {stations.columns.tolist()}")
    print(f"总记录数: {len(stations)}")
    print(f"前几行数据:")
    print(stations.head())

    # 2. 重命名列（根据你的示例数据）
    # 你的示例显示有: station_id, date, longitude, latitude
    # 但实际列名可能有空格或格式问题

    # 先找出实际列名
    actual_columns = {}
    for col in stations.columns:
        col_lower = str(col).lower().strip()
        if 'station' in col_lower or '站点' in col_lower:
            actual_columns['station_id'] = col
        elif 'date' in col_lower or '时间' in col_lower or '日期' in col_lower:
            actual_columns['date'] = col
        elif 'lon' in col_lower or 'longitude' in col_lower or '经度' in col_lower:
            actual_columns['longitude'] = col
        elif 'lat' in col_lower or 'latitude' in col_lower or '纬度' in col_lower:
            actual_columns['latitude'] = col

    print(f"检测到的列名映射: {actual_columns}")

    # 如果检测到列名，则重命名
    if len(actual_columns) >= 4:
        stations = stations.rename(columns=actual_columns)
    else:
        # 如果没检测到，使用示例中的列名
        expected_cols = ['station_id', 'date', 'longitude', 'latitude']
        for i, col in enumerate(expected_cols):
            if i < len(stations.columns):
                stations = stations.rename(columns={stations.columns[i]: col})

    print(f"处理后的列名: {stations.columns.tolist()}")

    # 3. 确保数据类型正确
    stations['date'] = pd.to_datetime(stations['date'], errors='coerce')
    stations['longitude'] = pd.to_numeric(stations['longitude'], errors='coerce')
    stations['latitude'] = pd.to_numeric(stations['latitude'], errors='coerce')

    # 移除无效数据
    stations = stations.dropna(subset=['date', 'longitude', 'latitude'])
    print(f"有效记录数: {len(stations)}")

    # 4. 提取像元值
    print("\n开始提取降水数据...")

    precipitation_values = []
    raster_files = []
    years_extracted = []

    for idx, row in stations.iterrows():
        try:
            date_obj = row['date']
            year = date_obj.year
            month = date_obj.month
            day = date_obj.day
            lon = row['longitude']
            lat = row['latitude']

            # 根据年份确定数据目录
            if year == 2014:
                data_dir = r"G:\朱佳腾\降水-CHELSA\2014"
            elif year == 2015:
                data_dir = r"G:\朱佳腾\降水-CHELSA\2015"
            elif year in [2016, 2017]:
                data_dir = r"G:\朱佳腾\降水-CHELSA\2016-2017"
            else:
                print(f"警告: {year}年数据目录未配置，跳过")
                precipitation_values.append(np.nan)
                raster_files.append(None)
                years_extracted.append(None)
                continue

            # 构建文件名（根据你提供的格式）
            # 注意: 01_01_2014 表示 日_月_年
            filename = f"CHELSA_pr_{day:02d}_{month:02d}_{year}_V.2.1.tif"
            file_path = Path(data_dir) / filename

            if not file_path.exists():
                # 尝试其他可能的命名变体
                alt_filename = f"CHELSA_pr_{day}_{month}_{year}_V.2.1.tif"
                alt_path = Path(data_dir) / alt_filename

                if alt_path.exists():
                    file_path = alt_path
                else:
                    print(f"文件不存在: {filename}")
                    precipitation_values.append(np.nan)
                    raster_files.append(None)
                    years_extracted.append(None)
                    continue

            # 打开栅格文件
            with rasterio.open(file_path) as src:
                # 将经纬度转换为行列号
                row_idx, col_idx = src.index(lon, lat)

                # 检查是否在有效范围内
                if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                    # 读取降水值 (kg m-2 day-1 = mm/day)
                    # 根据CHELSA文档，不需要缩放，直接使用
                    value = src.read(1)[row_idx, col_idx]

                    # 检查是否为无效值（CHELSA通常使用-32768作为无效值）
                    if value < -9999:  # 常见的无效值阈值
                        precipitation_values.append(np.nan)
                        print(f"无效值: 站点 {row['station_id']}, 日期 {date_obj.date()}, 值: {value}")
                    else:
                        precipitation_values.append(float(value))

                    raster_files.append(filename)
                    years_extracted.append(year)
                else:
                    print(f"坐标超出范围: 站点 {row['station_id']}, 经纬度({lon}, {lat})")
                    precipitation_values.append(np.nan)
                    raster_files.append(None)
                    years_extracted.append(None)

        except Exception as e:
            print(f"处理站点 {row['station_id']} 时出错: {str(e)}")
            precipitation_values.append(np.nan)
            raster_files.append(None)
            years_extracted.append(None)

    # 5. 添加结果到DataFrame
    stations['precipitation_mm_day'] = precipitation_values
    stations['chelsa_file'] = raster_files
    stations['year'] = years_extracted

    # 6. 保存结果
    stations.to_excel(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")

    # 7. 统计信息
    print("\n=== 提取统计 ===")
    print(f"总记录数: {len(stations)}")
    valid_count = stations['precipitation_mm_day'].count()
    print(f"成功提取数: {valid_count}")
    print(f"缺失数: {len(stations) - valid_count}")

    if valid_count > 0:
        print(f"\n降水量统计:")
        print(f"最小值: {stations['precipitation_mm_day'].min():.2f} mm/day")
        print(f"最大值: {stations['precipitation_mm_day'].max():.2f} mm/day")
        print(f"平均值: {stations['precipitation_mm_day'].mean():.2f} mm/day")
        print(f"中位数: {stations['precipitation_mm_day'].median():.2f} mm/day")

        # 按年份统计
        print(f"\n按年份统计:")
        year_stats = stations.groupby('year')['precipitation_mm_day'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('min', 'min'),
            ('max', 'max')
        ]).round(2)
        print(year_stats)

        # 按站点统计
        print(f"\n按站点统计 (前10个站点):")
        station_stats = stations.groupby('station_id').agg({
            'precipitation_mm_day': ['count', 'mean', 'min', 'max']
        }).round(2)
        print(station_stats.head(10))

    return stations


# 如果需要批量处理多个站点文件
def batch_process_station_files():
    """批量处理多个站点文件"""

    # 站点文件列表
    station_files = [
        "站点数据1.xlsx",
        "站点数据2.xlsx",
        # 添加更多文件
    ]

    results = []

    for station_file in station_files:
        if Path(station_file).exists():
            print(f"\n处理文件: {station_file}")
            output_file = f"结果_{Path(station_file).stem}.xlsx"
            result = extract_chelsa_precipitation(station_file, output_file)
            results.append(result)
        else:
            print(f"文件不存在: {station_file}")

    # 合并所有结果
    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_excel("所有站点_降水提取结果.xlsx", index=False)
        print(f"\n所有结果已合并保存到: 所有站点_降水提取结果.xlsx")
        return combined

    return None


if __name__ == "__main__":
    # 单文件处理
    station_file = "你的站点数据.xlsx"  # 修改为你的文件路径
    output_file = "站点_降水_提取结果.xlsx"

    # 执行提取
    result = extract_chelsa_precipitation(station_file, output_file)

    # 显示前几行结果
    print("\n前10行结果:")
    print(result[['station_id', 'date', 'longitude', 'latitude',
                  'precipitation_mm_day', 'chelsa_file']].head(10))