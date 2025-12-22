import pandas as pd
import rasterio
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time
import os
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore')


# 全局提取函数
def extract_single_point_worker(args):
    """工作进程函数 - 必须是全局的"""
    idx, date_str, lon, lat = args

    try:
        # 解析日期
        date = pd.to_datetime(date_str)
        year = date.year
        month = date.month
        day = date.day

        # 只处理2014年及以后
        if year < 2014:
            return idx, np.nan, None, "2014年以前跳过"

        # 确定目录
        if year == 2014:
            data_dir = r"G:\朱佳腾\降水-CHELSA\2014"
        elif year == 2015:
            data_dir = r"G:\朱佳腾\降水-CHELSA\2015"
        elif year in [2016, 2017]:
            data_dir = r"G:\朱佳腾\降水-CHELSA\2016-2017"
        else:
            return idx, np.nan, None, f"{year}年目录未配置"

        # 构建文件名
        filename = f"CHELSA_pr_{day:02d}_{month:02d}_{year}_V.2.1.tif"
        file_path = Path(data_dir) / filename

        if not file_path.exists():
            return idx, np.nan, None, "文件不存在"

        # 提取值
        with rasterio.open(file_path) as src:
            row_idx, col_idx = src.index(lon, lat)

            if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                value = src.read(1)[row_idx, col_idx]

                # 检查无效值
                if value <= -32767 or value < -9999:
                    return idx, np.nan, filename, "无效值"
                else:
                    return idx, float(value), filename, "成功"
            else:
                return idx, np.nan, filename, "坐标超出范围"

    except Exception as e:
        error_msg = str(e)
        return idx, np.nan, None, f"错误: {error_msg[:30]}"


def extract_chelsa_balanced(station_file, output_file, num_workers=8, chunk_size=1000):
    """
    平衡版本：多进程但有限制

    参数:
    station_file: 站点Excel文件路径
    output_file: 输出Excel文件路径
    num_workers: 进程数，默认为CPU核心数的一半
    chunk_size: 每批处理的数据量
    """

    print("正在读取站点数据...")
    start_time = time.time()

    # 读取数据
    try:
        stations = pd.read_excel(station_file)
    except Exception as e:
        print(f"读取失败: {e}")
        stations = pd.read_excel(station_file, engine='openpyxl')

    print(f"数据形状: {stations.shape}")

    # 自动识别列
    col_mapping = {}
    for col in stations.columns:
        col_lower = str(col).lower()
        if 'station' in col_lower:
            col_mapping[col] = 'station_id'
        elif 'date' in col_lower:
            col_mapping[col] = 'date'
        elif 'lon' in col_lower:
            col_mapping[col] = 'longitude'
        elif 'lat' in col_lower:
            col_mapping[col] = 'latitude'

    if len(col_mapping) >= 4:
        stations = stations.rename(columns=col_mapping)
        print("自动识别列完成")
    else:
        # 使用前4列
        stations = stations.iloc[:, :4].copy()
        stations.columns = ['station_id', 'date', 'longitude', 'latitude'][:len(stations.columns)]
        print("使用前4列")

    print(f"列名: {stations.columns.tolist()}")

    # 数据清洗
    stations['date'] = pd.to_datetime(stations['date'], errors='coerce')
    stations['longitude'] = pd.to_numeric(stations['longitude'], errors='coerce')
    stations['latitude'] = pd.to_numeric(stations['latitude'], errors='coerce')

    # 移除无效
    original_len = len(stations)
    stations = stations.dropna(subset=['date', 'longitude', 'latitude'])
    stations = stations[stations['date'].dt.year >= 2014]

    print(f"有效数据: {len(stations)}/{original_len}")

    if len(stations) == 0:
        print("没有有效数据")
        return None

    # 准备任务数据（只传必要的数据）
    tasks = []
    for idx, row in stations.iterrows():
        tasks.append((
            idx,
            str(row['date']),  # 转换为字符串，避免传递datetime对象
            float(row['longitude']),
            float(row['latitude'])
        ))

    print(f"准备处理 {len(tasks)} 条记录...")
    print(f"使用 {num_workers} 个进程，每批 {chunk_size} 条")

    # 分批次处理，避免一次加载太多任务
    total_chunks = (len(tasks) + chunk_size - 1) // chunk_size
    all_results = {}

    # 配置进程池
    mp_context = mp.get_context('spawn')  # 使用spawn模式，更稳定

    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(tasks))
        chunk_tasks = tasks[chunk_start:chunk_end]

        print(f"\n处理批次 {chunk_idx + 1}/{total_chunks} ({chunk_start + 1}-{chunk_end})...")

        chunk_results = {}

        # 使用ProcessPoolExecutor，但限制最大任务数
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            # 提交任务
            future_to_idx = {
                executor.submit(extract_single_point_worker, task): task[0]
                for task in chunk_tasks
            }

            # 处理结果
            with tqdm(total=len(chunk_tasks), desc=f"批次{chunk_idx + 1}", leave=False) as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        idx, value, filename, status = future.result(timeout=30)  # 30秒超时
                        chunk_results[idx] = (value, filename, status)
                    except Exception as e:
                        idx = future_to_idx[future]
                        chunk_results[idx] = (np.nan, None, f"超时或错误: {str(e)[:30]}")
                    finally:
                        pbar.update(1)

        # 合并结果
        all_results.update(chunk_results)

        # 每批处理后清理
        del chunk_tasks, chunk_results
        import gc
        gc.collect()

    # 整理结果
    print("\n整理结果...")
    stations['precipitation_mm_day'] = [all_results.get(idx, (np.nan, None, "未处理"))[0] for idx in stations.index]
    stations['chelsa_file'] = [all_results.get(idx, (np.nan, None, "未处理"))[1] for idx in stations.index]
    stations['status'] = [all_results.get(idx, (np.nan, None, "未处理"))[2] for idx in stations.index]

    # 保存结果
    print(f"保存结果到 {output_file}...")
    stations.to_excel(output_file, index=False)

    # 统计
    elapsed = time.time() - start_time
    success = stations[stations['status'] == '成功']['precipitation_mm_day'].count()

    print(f"\n{'=' * 50}")
    print("处理完成!")
    print(f"{'=' * 50}")
    print(f"总耗时: {elapsed:.2f}秒")
    print(f"速度: {len(tasks) / elapsed:.1f}条/秒")
    print(f"总记录: {len(stations)}")
    print(f"成功: {success} ({success / len(stations) * 100:.1f}%)")

    if success > 0:
        valid = stations[stations['status'] == '成功']['precipitation_mm_day']
        print(f"降水量范围: {valid.min():.2f} - {valid.max():.2f} mm/day")
        print(f"平均值: {valid.mean():.2f} mm/day")

    return stations


def extract_chelsa_smart(station_file, output_file, use_workers=None):
    """
    智能版本：根据数据量自动选择进程数
    """

    print("正在读取数据...")
    stations = pd.read_excel(station_file)

    # 简化的列处理
    if len(stations.columns) >= 4:
        stations = stations.iloc[:, :4].copy()
        stations.columns = ['station_id', 'date', 'longitude', 'latitude'][:len(stations.columns)]

    stations['date'] = pd.to_datetime(stations['date'], errors='coerce')
    stations['longitude'] = pd.to_numeric(stations['longitude'], errors='coerce')
    stations['latitude'] = pd.to_numeric(stations['latitude'], errors='coerce')
    stations = stations.dropna()
    stations = stations[stations['date'].dt.year >= 2014]

    data_size = len(stations)
    print(f"需要处理 {data_size} 条记录")

    # 根据数据量自动选择进程数
    if use_workers is None:
        cpu_count = os.cpu_count()
        if data_size < 1000:
            num_workers = 1  # 小数据量用单进程
        elif data_size < 10000:
            num_workers = max(2, cpu_count // 4)  # 中等数据用1/4核心
        elif data_size < 50000:
            num_workers = max(4, cpu_count // 2)  # 大数据用一半核心
        else:
            num_workers = max(8, cpu_count - 4)  # 超大数据用大部分核心
    else:
        num_workers = use_workers

    print(f"自动选择 {num_workers} 个进程")

    # 准备数据
    tasks = []
    for idx, row in stations.iterrows():
        tasks.append((
            idx,
            str(row['date']),
            float(row['longitude']),
            float(row['latitude'])
        ))

    # 处理函数（简单的）
    def process_task_simple(task):
        idx, date_str, lon, lat = task

        try:
            date = pd.to_datetime(date_str)
            year = date.year

            if year < 2014:
                return idx, np.nan

            # 目录
            if year == 2014:
                data_dir = r"G:\朱佳腾\降水-CHELSA\2014"
            elif year == 2015:
                data_dir = r"G:\朱佳腾\降水-CHELSA\2015"
            elif year in [2016, 2017]:
                data_dir = r"G:\朱佳腾\降水-CHELSA\2016-2017"
            else:
                return idx, np.nan

            filename = f"CHELSA_pr_{date.day:02d}_{date.month:02d}_{year}_V.2.1.tif"
            file_path = Path(data_dir) / filename

            if file_path.exists():
                with rasterio.open(file_path) as src:
                    row_idx, col_idx = src.index(lon, lat)

                    if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                        value = src.read(1)[row_idx, col_idx]
                        if value < -9999:
                            return idx, np.nan
                        else:
                            return idx, float(value)

        except:
            pass

        return idx, np.nan

    # 使用多进程但限制并发数
    print("开始提取...")
    results = {}

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=len(tasks), desc="提取进度") as pbar:
                # 分批提交，避免内存问题
                batch_size = 1000
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    futures = [executor.submit(process_task_simple, task) for task in batch]

                    for future in as_completed(futures):
                        idx, value = future.result()
                        results[idx] = value
                        pbar.update(1)

                    # 清理
                    del futures, batch
    else:
        # 单进程
        for task in tqdm(tasks, desc="提取进度"):
            idx, value = process_task_simple(task)
            results[idx] = value

    # 添加结果
    stations['precipitation'] = [results.get(idx, np.nan) for idx in stations.index]
    stations.to_excel(output_file, index=False)

    success = stations['precipitation'].count()
    print(f"\n完成! 成功: {success}/{len(stations)}")

    return stations


# 最快的优化版本
def extract_chelsa_optimized(station_file, output_file, workers=12):
    """
    优化版本：使用固定数量的进程，最快的速度
    """

    print(f"使用优化版本，{workers}个进程...")

    # 快速读取
    stations = pd.read_excel(station_file)
    stations = stations.iloc[:, :4].copy()
    stations.columns = ['station_id', 'date', 'longitude', 'latitude']

    # 快速处理
    stations['date'] = pd.to_datetime(stations['date'], errors='coerce')
    stations['longitude'] = pd.to_numeric(stations['longitude'], errors='coerce')
    stations['latitude'] = pd.to_numeric(stations['latitude'], errors='coerce')
    stations = stations.dropna()
    stations = stations[stations['date'].dt.year >= 2014].reset_index(drop=True)

    print(f"处理 {len(stations)} 条记录...")

    # 准备数据（更高效的方式）
    dates = stations['date'].dt.strftime('%Y-%m-%d').tolist()
    lons = stations['longitude'].tolist()
    lats = stations['latitude'].tolist()
    indices = list(range(len(stations)))

    # 批处理函数
    def process_batch(batch_indices):
        batch_results = []

        for idx in batch_indices:
            try:
                date = pd.to_datetime(dates[idx])
                year = date.year
                month = date.month
                day = date.day
                lon = lons[idx]
                lat = lats[idx]

                if year == 2014:
                    data_dir = r"G:\朱佳腾\降水-CHELSA\2014"
                elif year == 2015:
                    data_dir = r"G:\朱佳腾\降水-CHELSA\2015"
                elif year in [2016, 2017]:
                    data_dir = r"G:\朱佳腾\降水-CHELSA\2016-2017"
                else:
                    batch_results.append((idx, np.nan))
                    continue

                filename = f"CHELSA_pr_{day:02d}_{month:02d}_{year}_V.2.1.tif"
                file_path = Path(data_dir) / filename

                if file_path.exists():
                    with rasterio.open(file_path) as src:
                        row_idx, col_idx = src.index(lon, lat)

                        if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                            value = src.read(1)[row_idx, col_idx]
                            batch_results.append((idx, float(value) if value >= -9999 else np.nan))
                        else:
                            batch_results.append((idx, np.nan))
                else:
                    batch_results.append((idx, np.nan))

            except:
                batch_results.append((idx, np.nan))

        return batch_results

    # 分批次
    batch_size = len(stations) // (workers * 2) + 1
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    print(f"分 {len(batches)} 批处理...")

    # 并行处理
    results = {}

    if workers > 1 and len(batches) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            with tqdm(total=len(stations), desc="提取进度") as pbar:
                for future in as_completed(futures):
                    batch_results = future.result()
                    for idx, value in batch_results:
                        results[idx] = value
                        pbar.update(1)
    else:
        # 单进程
        for batch in tqdm(batches, desc="处理批次"):
            batch_results = process_batch(batch)
            for idx, value in batch_results:
                results[idx] = value

    # 添加结果
    stations['precipitation'] = [results.get(i, np.nan) for i in range(len(stations))]
    stations.to_excel(output_file, index=False)

    success = stations['precipitation'].count()
    print(f"\n完成! 成功提取 {success} 条，速度极快!")

    return stations


if __name__ == "__main__":
    print("=" * 60)
    print("CHELSA降水数据提取 - 平衡速度与稳定性")
    print("=" * 60)

    station_file = "E:/pycharmworkspace/fusing-xgb/src/training/lu_onehot - 副本.xlsx"
    output_file = "E:/pycharmworkspace/DSTM/output/降水_提取_快速版.xlsx"

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cpu_count = os.cpu_count()
    print(f"检测到 {cpu_count} 个CPU核心")

    print("\n选择处理模式:")
    print(f"1. 平衡模式 (使用{cpu_count // 2}个核心)")
    print(f"2. 智能模式 (根据数据量自动调整)")
    print(f"3. 优化模式 (固定12个进程，最快速度)")
    print("4. 测试模式 (只处理前1000条)")

    choice = input("请选择 (1-4, 默认1): ").strip() or "1"

    try:
        start_time = time.time()

        if choice == "1":
            # 平衡模式：使用一半核心
            workers = max(4, cpu_count // 2)
            print(f"\n使用平衡模式，{workers}个进程...")
            result = extract_chelsa_balanced(station_file, output_file, num_workers=workers)

        elif choice == "2":
            # 智能模式
            result = extract_chelsa_smart(station_file, output_file)

        elif choice == "3":
            # 优化模式：固定12个进程（14900K有32线程，12个比较安全）
            result = extract_chelsa_optimized(station_file, output_file, workers=12)

        elif choice == "4":
            # 测试模式
            print("\n测试模式：只处理前1000条...")
            stations = pd.read_excel(station_file)
            stations = stations.iloc[:1000, :4].copy()
            stations.columns = ['station_id', 'date', 'longitude', 'latitude']

            # 单进程快速测试
            precip = []
            for _, row in tqdm(stations.iterrows(), total=len(stations)):
                try:
                    date = pd.to_datetime(row['date'])
                    year = date.year

                    if year < 2014:
                        precip.append(np.nan)
                        continue

                    if year == 2014:
                        data_dir = r"G:\朱佳腾\降水-CHELSA\2014"
                    elif year == 2015:
                        data_dir = r"G:\朱佳腾\降水-CHELSA\2015"
                    elif year in [2016, 2017]:
                        data_dir = r"G:\朱佳腾\降水-CHELSA\2016-2017"
                    else:
                        precip.append(np.nan)
                        continue

                    filename = f"CHELSA_pr_{date.day:02d}_{date.month:02d}_{year}_V.2.1.tif"
                    file_path = Path(data_dir) / filename

                    if file_path.exists():
                        with rasterio.open(file_path) as src:
                            row_idx, col_idx = src.index(row['longitude'], row['latitude'])
                            if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                                value = src.read(1)[row_idx, col_idx]
                                precip.append(float(value) if value >= -9999 else np.nan)
                            else:
                                precip.append(np.nan)
                    else:
                        precip.append(np.nan)

                except:
                    precip.append(np.nan)

            stations['precipitation'] = precip
            test_file = output_file.replace('.xlsx', '_test.xlsx')
            stations.to_excel(test_file, index=False)
            result = stations

        else:
            print("无效选择，使用平衡模式")


            workers = max(4, cpu_count // 2)
            result = extract_chelsa_balanced(station_file, output_file, num_workers=workers)

        if result is not None:
            elapsed = time.time() - start_time
            print(f"\n总耗时: {elapsed:.1f}秒")

            if 'precipitation' in result.columns:
                success = result['precipitation'].count()
            elif 'precipitation_mm_day' in result.columns:
                success = result['precipitation_mm_day'].count()
            else:
                success = 0

            print(f"成功提取: {success}/{len(result)} ({success / len(result) * 100:.1f}%)")
            print(f"前5行结果:")
            print(result.head())

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback

        traceback.print_exc()