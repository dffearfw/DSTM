"""
SMAP L1C数据批量重采样（处理所有有数据的文件）
版本: 7.2 (处理所有文件版)
功能:
  1. 遍历所有SMAP文件，检查是否有SHP范围内的数据
  2. 只处理有数据的文件
  3. 每个文件单独处理输出
"""

import h5py
import numpy as np
import os
import warnings
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RioResampling
from rasterio.mask import mask
from rasterio.transform import from_origin
import tempfile
import shutil
import glob
import geopandas as gpd
from shapely.geometry import mapping
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')

def check_smap_file_has_data(h5_path, shapefile_bounds, buffer_degree=0.5):
    """
    检查SMAP文件在SHP范围内是否有数据
    返回: (has_data, data_count, data_info)
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'Global_Projection' not in f:
                return False, 0, "没有Global_Projection"

            proj = f['Global_Projection']

            # 读取经纬度
            lat = proj['cell_lat'][:].astype(np.float32)
            lon = proj['cell_lon'][:].astype(np.float32)

            # SHP范围（带缓冲）
            min_lon = shapefile_bounds[0] - buffer_degree
            min_lat = shapefile_bounds[1] - buffer_degree
            max_lon = shapefile_bounds[2] + buffer_degree
            max_lat = shapefile_bounds[3] + buffer_degree

            # 检查范围内是否有数据
            in_bounds = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)

            # 检查亮温数据是否有效
            tb_v = None
            for key in ['cell_tb_v_fore', 'cell_tb_v_surface_corrected_fore',
                       'cell_tb_v_aft', 'cell_tb_v_surface_corrected_aft']:
                if key in proj:
                    tb_v = proj[key][:].astype(np.float32)
                    break

            if tb_v is None:
                return False, 0, "没有亮温数据"

            # 有效数据掩膜
            valid_in_bounds = in_bounds & (lat != -9999.0) & (lon != -9999.0) & (tb_v != -9999.0)
            data_count = valid_in_bounds.sum()

            if data_count > 0:
                # 获取范围内数据的统计信息
                lat_in = lat[valid_in_bounds]
                lon_in = lon[valid_in_bounds]
                tb_v_in = tb_v[valid_in_bounds]

                info = f"数据点: {data_count}, 纬度: {lat_in.min():.1f}-{lat_in.max():.1f}°, " \
                       f"经度: {lon_in.min():.1f}-{lon_in.max():.1f}°, " \
                       f"亮温: {tb_v_in.min():.1f}-{tb_v_in.max():.1f}K"
                return True, data_count, info
            else:
                return False, 0, f"范围内有{in_bounds.sum()}个点但无效"

    except Exception as e:
        return False, 0, f"读取错误: {str(e)}"

def extract_smap_data_in_shapefile(h5_path, shapefile_bounds, buffer_degree=0.5, use_corrected=True):
    """
    提取SHP范围内的SMAP数据
    """
    with h5py.File(h5_path, 'r') as f:
        proj = f['Global_Projection']

        # 读取所有数据
        lat = proj['cell_lat'][:].astype(np.float32)
        lon = proj['cell_lon'][:].astype(np.float32)

        # SHP范围（带缓冲）
        min_lon = shapefile_bounds[0] - buffer_degree
        min_lat = shapefile_bounds[1] - buffer_degree
        max_lon = shapefile_bounds[2] + buffer_degree
        max_lat = shapefile_bounds[3] + buffer_degree

        # 范围内的点
        in_bounds = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)

        # 获取亮温数据
        def get_tb_data(pol):
            candidates = []
            if use_corrected:
                candidates.extend([
                    f'cell_tb_{pol}_surface_corrected_fore',
                    f'cell_tb_{pol}_surface_corrected_aft'
                ])
            candidates.extend([
                f'cell_tb_{pol}_fore',
                f'cell_tb_{pol}_aft'
            ])

            data = None
            for key in candidates:
                if key in proj:
                    candidate_data = proj[key][:].astype(np.float32)
                    if data is None:
                        data = candidate_data
                    else:
                        mask_valid = (candidate_data != -9999.0) & (data == -9999.0)
                        data[mask_valid] = candidate_data[mask_valid]
            return data

        tb_v = get_tb_data('v')
        tb_h = get_tb_data('h')

        # 有效数据掩膜
        valid_mask = in_bounds & (lat != -9999.0) & (lon != -9999.0) & \
                    (tb_v != -9999.0) & (tb_h != -9999.0)

        # 提取数据
        filtered_lat = lat[valid_mask]
        filtered_lon = lon[valid_mask]
        filtered_tb_v = tb_v[valid_mask]
        filtered_tb_h = tb_h[valid_mask]

        print(f"  提取到 {len(filtered_lat)} 个有效数据点")
        print(f"  范围: 纬度[{filtered_lat.min():.2f}°, {filtered_lat.max():.2f}°], "
              f"经度[{filtered_lon.min():.2f}°, {filtered_lon.max():.2f}°]")

        return {
            'latitude': filtered_lat,
            'longitude': filtered_lon,
            'tb_v': filtered_tb_v,
            'tb_h': filtered_tb_h,
            'original_shape': lat.shape,
            'filter_mask': valid_mask
        }

def create_grid_from_points(smap_data, target_bounds, target_res):
    """
    从点数据创建网格
    """
    lat = smap_data['latitude']
    lon = smap_data['longitude']

    # 目标范围
    lon_min, lat_min, lon_max, lat_max = target_bounds
    res_lon, res_lat = target_res

    # 创建网格
    grid_lon = np.arange(lon_min, lon_max + res_lon, res_lon)
    grid_lat = np.arange(lat_min, lat_max + res_lat, res_lat)

    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    # 使用scipy插值（如果可用）
    try:
        from scipy.interpolate import griddata
        print("  使用Scipy griddata插值...")

        # 准备数据
        points = np.column_stack([lon, lat])
        values_v = smap_data['tb_v']
        values_h = smap_data['tb_h']

        # 插值
        grid_tb_v = griddata(points, values_v, (grid_lon_mesh, grid_lat_mesh),
                            method='linear', fill_value=-9999.0)
        grid_tb_h = griddata(points, values_h, (grid_lon_mesh, grid_lat_mesh),
                            method='linear', fill_value=-9999.0)

    except ImportError:
        print("  使用简单最近邻插值...")
        grid_height = len(grid_lat)
        grid_width = len(grid_lon)

        grid_tb_v = np.full((grid_height, grid_width), -9999.0, dtype=np.float32)
        grid_tb_h = np.full((grid_height, grid_width), -9999.0, dtype=np.float32)

        for i in range(len(lat)):
            col = int((lon[i] - lon_min) / res_lon)
            row = int((lat_max - lat[i]) / res_lat)

            if 0 <= row < grid_height and 0 <= col < grid_width:
                if grid_tb_v[row, col] == -9999.0:
                    grid_tb_v[row, col] = smap_data['tb_v'][i]
                    grid_tb_h[row, col] = smap_data['tb_h'][i]

    return grid_tb_v, grid_tb_h, grid_lon, grid_lat

def process_single_smap_file(h5_path, shapefile_path, target_tif_path, output_path,
                           use_corrected=True, buffer_degree=0.5):
    """
    处理单个SMAP文件
    """
    print(f"处理文件: {os.path.basename(h5_path)}")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='smap_single_')

    try:
        # 1. 读取SHP获取边界
        gdf = gpd.read_file(shapefile_path)
        shapefile_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        shapes = [mapping(geom) for geom in gdf.geometry]

        print(f"  SHP边界: [{shapefile_bounds[0]:.2f}, {shapefile_bounds[1]:.2f}, "
              f"{shapefile_bounds[2]:.2f}, {shapefile_bounds[3]:.2f}]")

        # 2. 提取SHP范围内的数据
        print("  提取SHP范围内数据...")
        smap_data = extract_smap_data_in_shapefile(
            h5_path, shapefile_bounds, buffer_degree, use_corrected
        )

        if len(smap_data['latitude']) == 0:
            print("  × 错误: 没有提取到有效数据")
            return False

        # 3. 获取目标投影信息
        print("  读取目标投影信息...")
        with rasterio.open(target_tif_path) as src:
            target_crs = src.crs
            target_transform = src.transform
            target_width = src.width
            target_height = src.height
            target_bounds = src.bounds

            # 计算经纬度分辨率和范围
            res_lon = abs(target_transform.a)
            res_lat = abs(target_transform.e)
            lon_min, lat_min, lon_max, lat_max = target_bounds

        print(f"  目标范围: 经度[{lon_min:.2f}, {lon_max:.2f}], "
              f"纬度[{lat_min:.2f}, {lat_max:.2f}]")
        print(f"  目标分辨率: {res_lon:.4f}° x {res_lat:.4f}°")

        # 4. 创建中间网格
        print("  创建网格...")
        intermediate_tif = os.path.join(temp_dir, 'intermediate.tif')

        # 创建目标范围内的网格
        grid_tb_v, grid_tb_h, grid_lon, grid_lat = create_grid_from_points(
            smap_data,
            target_bounds=(lon_min, lat_min, lon_max, lat_max),
            target_res=(res_lon, res_lat)
        )

        # 5. 保存中间文件
        transform = from_origin(
            grid_lon[0] - res_lon/2,
            grid_lat[-1] + res_lat/2,
            res_lon, res_lat
        )

        profile = {
            'driver': 'GTiff',
            'height': len(grid_lat),
            'width': len(grid_lon),
            'count': 2,
            'dtype': np.float32,
            'crs': rasterio.CRS.from_epsg(4326),
            'transform': transform,
            'nodata': -9999.0
        }

        with rasterio.open(intermediate_tif, 'w', **profile) as dst:
            dst.write(grid_tb_v, 1)
            dst.set_band_description(1, 'V_polarization')
            dst.write(grid_tb_h, 2)
            dst.set_band_description(2, 'H_polarization')

        print(f"  中间文件已创建: {intermediate_tif}")

        # 6. 重投影到目标投影
        print("  重投影到目标投影...")
        reprojected_temp = os.path.join(temp_dir, 'reprojected.tif')

        with rasterio.open(intermediate_tif) as src:
            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': target_crs,
                'transform': target_transform,
                'width': target_width,
                'height': target_height,
                'nodata': -9999.0
            })

            with rasterio.open(reprojected_temp, 'w', **dst_profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                        dst_nodata=-9999.0
                    )

        # 7. 应用SHP掩膜
        print("  应用SHP掩膜...")
        with rasterio.open(reprojected_temp) as src:
            out_image, out_transform = mask(
                src,
                shapes,
                crop=False,
                all_touched=True,
                nodata=-9999.0
            )

            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

        print(f"  ✓ 处理完成: {output_path}")

        # 8. 验证结果
        with rasterio.open(output_path) as dst:
            data = dst.read(1)
            valid_data = data[data != -9999.0]

            if len(valid_data) > 0:
                print(f"    有效像元: {len(valid_data)}/{data.size} ({len(valid_data)/data.size*100:.1f}%)")
                print(f"    亮温范围: {valid_data.min():.1f}K 到 {valid_data.max():.1f}K")
            else:
                print(f"    ⚠ 警告: 没有有效像元")

        return True

    except Exception as e:
        print(f"  × 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_all_smap_files(smap_dir, shapefile_path, target_tif_path, output_dir,
                          use_corrected=True, buffer_degree=0.5, max_workers=None):
    """
    处理所有SMAP文件（自动筛选有数据的文件）

    参数:
        smap_dir: SMAP文件目录
        shapefile_path: SHP文件路径
        target_tif_path: 目标投影TIF文件路径
        output_dir: 输出目录
        use_corrected: 是否使用表面校正数据
        buffer_degree: 边界扩展度数
        max_workers: 最大并行处理数（None为串行处理）
    """
    print("=" * 80)
    print("SMAP文件批量处理 - 处理所有有数据的文件")
    print("=" * 80)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有H5文件
    h5_files = sorted(glob.glob(os.path.join(smap_dir, "*.h5")))
    print(f"找到 {len(h5_files)} 个SMAP文件")

    if len(h5_files) == 0:
        print("错误: 没有找到SMAP文件")
        return

    # 读取SHP获取边界（只需一次）
    print("\n读取SHP文件...")
    gdf = gpd.read_file(shapefile_path)
    shapefile_bounds = gdf.total_bounds
    print(f"SHP边界: [{shapefile_bounds[0]:.2f}, {shapefile_bounds[1]:.2f}, "
          f"{shapefile_bounds[2]:.2f}, {shapefile_bounds[3]:.2f}]")

    # 第一步：快速检查所有文件，找出有数据的文件
    print("\n[步骤1/3] 快速检查文件数据...")
    valid_files_info = []
    skip_files = []

    for i, h5_file in enumerate(h5_files):
        filename = os.path.basename(h5_file)
        print(f"{i+1:4d}/{len(h5_files)}: {filename:60s}", end="", flush=True)

        has_data, data_count, info = check_smap_file_has_data(
            h5_file, shapefile_bounds, buffer_degree
        )

        if has_data:
            print(f" ✓ 有数据 ({data_count}点)")
            valid_files_info.append({
                'path': h5_file,
                'filename': filename,
                'data_count': data_count,
                'info': info
            })
        else:
            print(f" × 跳过 ({info})")
            skip_files.append(filename)

    print(f"\n检查完成:")
    print(f"  总文件数: {len(h5_files)}")
    print(f"  有数据文件: {len(valid_files_info)}")
    print(f"  跳过文件: {len(skip_files)}")

    if len(valid_files_info) == 0:
        print("\n错误: 没有找到任何在SHP范围内有数据的文件")
        print("可能的原因:")
        print("  1. 所有文件都不覆盖新疆区域")
        print("  2. SHP边界设置有问题")
        print("  3. 数据质量标记导致数据被过滤")
        return

    # 显示有数据的文件列表
    print(f"\n有数据的文件列表 ({len(valid_files_info)}个):")
    for i, file_info in enumerate(valid_files_info[:20]):  # 只显示前20个
        print(f"  {i+1:3d}. {file_info['filename']:50s} - {file_info['info']}")

    if len(valid_files_info) > 20:
        print(f"  ... 还有 {len(valid_files_info) - 20} 个文件")

    # 第二步：处理有数据的文件
    print(f"\n[步骤2/3] 开始处理 {len(valid_files_info)} 个有数据的文件...")

    success_files = []
    failed_files = []

    for i, file_info in enumerate(valid_files_info):
        print(f"\n{'='*70}")
        print(f"处理文件 {i+1}/{len(valid_files_info)}: {file_info['filename']}")
        print(f"数据信息: {file_info['info']}")
        print('='*70)

        # 创建输出文件名
        base_name = file_info['filename'].replace('.h5', '_resampled.tif')
        output_path = os.path.join(output_dir, base_name)

        # 检查输出文件是否已存在
        if os.path.exists(output_path):
            print(f"  ⚠ 输出文件已存在，跳过: {base_name}")
            success_files.append(file_info['filename'])
            continue

        # 处理单个文件
        success = process_single_smap_file(
            h5_path=file_info['path'],
            shapefile_path=shapefile_path,
            target_tif_path=target_tif_path,
            output_path=output_path,
            use_corrected=use_corrected,
            buffer_degree=buffer_degree
        )

        if success:
            success_files.append(file_info['filename'])
        else:
            failed_files.append(file_info['filename'])

    # 第三步：生成处理报告
    print("\n" + "=" * 80)
    print("处理完成报告")
    print("=" * 80)

    print(f"\n统计信息:")
    print(f"  总SMAP文件数: {len(h5_files)}")
    print(f"  有数据文件数: {len(valid_files_info)} ({len(valid_files_info)/len(h5_files)*100:.1f}%)")
    print(f"  成功处理文件: {len(success_files)}")
    print(f"  处理失败文件: {len(failed_files)}")
    print(f"  跳过文件: {len(skip_files)}")

    # 保存处理日志
    log_file = os.path.join(output_dir, "processing_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("SMAP数据处理日志\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"处理时间: {np.datetime64('now')}\n")
        f.write(f"SMAP目录: {smap_dir}\n")
        f.write(f"SHP文件: {shapefile_path}\n")
        f.write(f"目标TIF: {target_tif_path}\n")
        f.write(f"输出目录: {output_dir}\n\n")

        f.write("统计信息:\n")
        f.write(f"  总SMAP文件数: {len(h5_files)}\n")
        f.write(f"  有数据文件数: {len(valid_files_info)}\n")
        f.write(f"  成功处理文件: {len(success_files)}\n")
        f.write(f"  处理失败文件: {len(failed_files)}\n")
        f.write(f"  跳过文件: {len(skip_files)}\n\n")

        f.write("成功处理文件列表:\n")
        for filename in success_files:
            f.write(f"  ✓ {filename}\n")

        f.write("\n处理失败文件列表:\n")
        for filename in failed_files:
            f.write(f"  ✗ {filename}\n")

        f.write("\n跳过文件列表:\n")
        for filename in skip_files:
            f.write(f"  - {filename}\n")

    print(f"\n详细日志已保存到: {log_file}")

    # 如果有失败的文件，显示建议
    if failed_files:
        print(f"\n警告: 有 {len(failed_files)} 个文件处理失败")
        print("建议检查:")
        print("  1. 文件是否损坏")
        print("  2. 磁盘空间是否足够")
        print("  3. 内存是否足够")

    print("\n" + "=" * 80)
    print("所有文件处理完成!")
    print("=" * 80)

# 简化的主程序
if __name__ == "__main__":
    # 设置路径
    smap_dir = "G:/王扬/smap_data"
    shapefile_path = r"E:\pycharmworkspace\DSTM\src\pre-process\XINGJIANG\XINGJIANG.shp"
    target_tif_path = "G:/王扬/fusedSWE/XINJIANG/XINGJIANG_XGB_SWE_DAILY_025_20150101.tif"
    output_dir = "G:/王扬/smap_resampled_masked"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 检查依赖
    try:
        import h5py, rasterio, geopandas, numpy
        print("依赖检查通过")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install h5py rasterio geopandas numpy scipy")
        exit(1)

    # 检查文件是否存在
    missing_files = []
    if not os.path.exists(smap_dir):
        missing_files.append(f"SMAP目录: {smap_dir}")
    if not os.path.exists(shapefile_path):
        missing_files.append(f"SHP文件: {shapefile_path}")
    if not os.path.exists(target_tif_path):
        missing_files.append(f"目标TIF: {target_tif_path}")

    if missing_files:
        print("错误: 以下文件/目录不存在:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)

    print(f"开始处理...")
    print(f"SMAP目录: {smap_dir}")
    print(f"SHP文件: {shapefile_path}")
    print(f"目标TIF: {target_tif_path}")
    print(f"输出目录: {output_dir}")

    # 处理所有文件
    process_all_smap_files(
        smap_dir=smap_dir,
        shapefile_path=shapefile_path,
        target_tif_path=target_tif_path,
        output_dir=output_dir,
        use_corrected=True,
        buffer_degree=0.5
    )