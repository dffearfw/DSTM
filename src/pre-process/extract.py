import pandas as pd
import rasterio
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def extract_chelsa_precipitation_from_2014(station_file, output_file):
    """
    从2014年开始提取CHELSA降水数据
    """

    # 1. 读取站点数据
    print("正在读取站点数据...")
    stations = pd.read_excel(station_file)

    # 查看数据列名
    print(f"数据列名: {stations.columns.tolist()}")
    print(f"总记录数: {len(stations)}")
    print(f"前几行数据:")
    print(stations.head())

    # 2. 重命名列
    actual_columns = {}
    for col in stations.columns:
        col_lower = str(col).lower().strip()
        if 'station' in col_lower:
            actual_columns['station_id'] = col
        elif 'date' in col_lower:
            actual_columns['date'] = col
        elif 'lon' in col_lower:
            actual_columns['longitude'] = col
        elif 'lat' in col_lower:
            actual_columns['latitude'] = col

    print(f"检测到的列名映射: {actual_columns}")

    # 如果检测到列名，则重命名
    if len(actual_columns) >= 4:
        stations = stations.rename(columns=actual_columns)
    else:
        # 如果没检测到，尝试按位置分配
        if len(stations.columns) >= 4:
            stations.columns = ['station_id', 'date', 'longitude', 'latitude'][:len(stations.columns)]
        else:
            print("错误: 数据列不足4列")
            return None

    print(f"处理后的列名: {stations.columns.tolist()}")

    # 3. 确保数据类型正确
    stations['date'] = pd.to_datetime(stations['date'], errors='coerce')
    stations['longitude'] = pd.to_numeric(stations['longitude'], errors='coerce')
    stations['latitude'] = pd.to_numeric(stations['latitude'], errors='coerce')

    # 4. 筛选2014年及以后的数据
    stations['year'] = stations['date'].dt.year
    stations = stations[stations['year'] >= 2014].copy()

    if len(stations) == 0:
        print("警告: 没有2014年及以后的数据")
        return None

    print(f"2014年及以后有效记录数: {len(stations)}")
    print(f"年份分布:")
    print(stations['year'].value_counts().sort_index())

    # 5. 提取像元值
    print("\n开始提取降水数据...")

    precipitation_values = []
    raster_files = []
    extraction_status = []

    processed_count = 0
    success_count = 0

    for idx, row in stations.iterrows():
        processed_count += 1

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
                # 2018年以后的路径（如果需要）
                data_dir = rf"G:\朱佳腾\降水-CHELSA\{year}"
                if not Path(data_dir).exists():
                    precipitation_values.append(np.nan)
                    raster_files.append(None)
                    extraction_status.append(f"{year}年目录不存在")
                    continue

            # 构建文件名
            filename = f"CHELSA_pr_{day:02d}_{month:02d}_{year}_V.2.1.tif"
            file_path = Path(data_dir) / filename

            if not file_path.exists():
                precipitation_values.append(np.nan)
                raster_files.append(None)
                extraction_status.append("文件不存在")
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count}/{len(stations)} 条记录...")
                continue

            # 打开栅格文件
            with rasterio.open(file_path) as src:
                row_idx, col_idx = src.index(lon, lat)

                if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                    value = src.read(1)[row_idx, col_idx]

                    # 检查无效值
                    if value < -9999:
                        precipitation_values.append(np.nan)
                        raster_files.append(filename)
                        extraction_status.append("无效值")
                    else:
                        precipitation_values.append(float(value))
                        raster_files.append(filename)
                        extraction_status.append("成功")
                        success_count += 1
                else:
                    precipitation_values.append(np.nan)
                    raster_files.append(filename)
                    extraction_status.append("坐标超出范围")

        except Exception as e:
            precipitation_values.append(np.nan)
            raster_files.append(None)
            extraction_status.append(f"错误: {str(e)}")

        # 显示进度
        if processed_count % 100 == 0:
            print(f"进度: {processed_count}/{len(stations)}，成功: {success_count}")

    print(f"处理完成: {processed_count}/{len(stations)}，成功: {success_count}")

    # 6. 添加结果到DataFrame
    stations['precipitation_mm_day'] = precipitation_values
    stations['chelsa_file'] = raster_files
    stations['extraction_status'] = extraction_status

    # 7. 保存结果
    stations.to_excel(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")

    # 8. 详细统计信息
    print("\n=== 详细统计 ===")
    print(f"总处理记录: {len(stations)}")
    print(f"成功提取: {success_count}")
    print(f"成功率: {success_count / len(stations) * 100:.1f}%")

    # 状态统计
    status_stats = stations['extraction_status'].value_counts()
    print(f"\n状态统计:")
    for status, count in status_stats.items():
        print(f"  {status}: {count}")

    # 按年份统计
    if success_count > 0:
        year_success = stations[stations['extraction_status'] == '成功'].groupby('year').size()
        print(f"\n按年份成功提取数:")
        for year, count in year_success.items():
            print(f"  {year}年: {count}")

        # 降水量统计
        valid_precip = stations[stations['extraction_status'] == '成功']['precipitation_mm_day']
        print(f"\n降水量统计 (成功提取的记录):")
        print(f"  最小值: {valid_precip.min():.2f} mm/day")
        print(f"  最大值: {valid_precip.max():.2f} mm/day")
        print(f"  平均值: {valid_precip.mean():.2f} mm/day")
        print(f"  中位数: {valid_precip.median():.2f} mm/day")

        # 按站点统计
        station_stats = stations[stations['extraction_status'] == '成功'].groupby('station_id').agg({
            'precipitation_mm_day': ['count', 'mean', 'min', 'max']
        }).round(2)

        print(f"\n站点统计 (前5个站点):")
        print(station_stats.head())

    return stations


if __name__ == "__main__":
    # 单文件处理
    station_file = "E:/pycharmworkspace/fusing-xgb/src/training/lu_onehot - 副本.xlsx"
    output_file = "E:/pycharmworkspace/DSTM/output/站点_降水_提取结果_2014起.xlsx"

    # 执行提取
    print("开始提取CHELSA降水数据 (2014年起)...")
    result = extract_chelsa_precipitation_from_2014(station_file, output_file)

    if result is not None:
        print("\n前10行结果:")
        print(result[['station_id', 'date', 'year', 'longitude', 'latitude',
                      'precipitation_mm_day', 'chelsa_file', 'extraction_status']].head(10))

        # 保存详细日志
        log_file = output_file.replace('.xlsx', '_log.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("CHELSA降水数据提取日志\n")
            f.write(f"处理时间: {pd.Timestamp.now()}\n")
            f.write(f"总记录数: {len(result)}\n")
            f.write(f"成功提取数: {len(result[result['extraction_status'] == '成功'])}\n")
            f.write(f"成功率: {len(result[result['extraction_status'] == '成功']) / len(result) * 100:.1f}%\n")

            # 状态统计
            f.write("\n状态统计:\n")
            status_stats = result['extraction_status'].value_counts()
            for status, count in status_stats.items():
                f.write(f"  {status}: {count}\n")

        print(f"详细日志保存到: {log_file}")
    else:
        print("提取失败或没有可提取的数据")