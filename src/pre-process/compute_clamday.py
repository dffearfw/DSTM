import xarray as xr
import rioxarray as rio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path
import pandas as pd
import warnings
import geopandas as gpd
from shapely.geometry import mapping

warnings.filterwarnings('ignore')

# 设置路径
data_dir = Path(r"G:\王扬\chelsa_sfxwind\2015")
china_shp_path = Path(r"D:\Workstation_share\中国_省\中国_省2.shp")
output_dir = Path(r"G:\王扬\chelsa_sfxwind\15results")
output_dir.mkdir(parents=True, exist_ok=True)


def load_china_shapefile():
    """加载中国shapefile"""
    print(f"加载中国shapefile: {china_shp_path}")

    try:
        # 加载shapefile
        china_gdf = gpd.read_file(china_shp_path)

        print(f"Shapefile信息:")
        print(f"  要素数量: {len(china_gdf)}")
        print(f"  坐标系: {china_gdf.crs}")
        print(f"  字段: {list(china_gdf.columns)}")

        # 查看前几个要素
        print(f"  示例省份:")
        for i, row in china_gdf.head(3).iterrows():
            # 尝试找到名称字段
            name_field = None
            for col in china_gdf.columns:
                if isinstance(row[col], str) and len(row[col]) < 20:
                    name_field = col
                    break
            if name_field:
                print(f"    - {row[name_field]}")

        # 如果shapefile有多个要素（各省），合并为一个中国多边形
        if len(china_gdf) > 1:
            print(f"  合并 {len(china_gdf)} 个要素为中国整体边界...")
            china_union = china_gdf.unary_union
            china_gdf = gpd.GeoDataFrame(geometry=[china_union], crs=china_gdf.crs)

        # 获取边界框
        bounds = china_gdf.total_bounds
        print(f"  中国边界框: {bounds}")
        print(f"  经度范围: {bounds[0]:.2f} - {bounds[2]:.2f}")
        print(f"  纬度范围: {bounds[1]:.2f} - {bounds[3]:.2f}")

        # 确保使用WGS84坐标系
        if china_gdf.crs is None:
            print("  警告: shapefile没有坐标系，假设为WGS84")
            china_gdf = china_gdf.set_crs("EPSG:4326")
        elif str(china_gdf.crs).upper() != "EPSG:4326":
            print(f"  转换坐标系到WGS84 (EPSG:4326)...")
            china_gdf = china_gdf.to_crs("EPSG:4326")
            bounds = china_gdf.total_bounds
            print(f"  转换后边界框: {bounds}")

        return china_gdf

    except Exception as e:
        print(f"加载shapefile失败: {e}")
        return None


def create_china_mask(raster_data, china_gdf):
    """
    创建中国区域的掩膜

    参数:
        raster_data: rioxarray数据
        china_gdf: 中国shapefile的GeoDataFrame

    返回:
        china_mask: 布尔掩膜数组 (True表示中国区域)
        raster_china: 裁剪到中国边界框的数据
    """

    try:
        # 获取数据坐标系
        if raster_data.rio.crs is None:
            print("  警告: 栅格数据没有坐标系，假设为WGS84")
            raster_data = raster_data.rio.write_crs("EPSG:4326")

        # 确保坐标系一致
        if str(raster_data.rio.crs) != str(china_gdf.crs):
            print(f"  转换shapefile坐标系以匹配栅格数据...")
            china_gdf_reprojected = china_gdf.to_crs(raster_data.rio.crs)
        else:
            china_gdf_reprojected = china_gdf

        # 获取中国边界框
        china_bbox = china_gdf_reprojected.total_bounds

        # 先裁剪到中国边界框（提高性能）
        print(f"  裁剪到中国边界框: {china_bbox}")
        raster_clipped = raster_data.rio.clip_box(
            minx=china_bbox[0], miny=china_bbox[1],
            maxx=china_bbox[2], maxy=china_bbox[3]
        )

        # 创建精确的中国区域掩膜
        print("  创建精确中国掩膜...")
        try:
            # 方法1: 使用rioxarray的clip方法
            china_masked = raster_clipped.rio.clip(
                china_gdf_reprojected.geometry,
                all_touched=False,  # 只包括中心点在多边形内的像元
                drop=False
            )

            # 创建掩膜 (非NaN区域为True)
            mask = ~np.isnan(china_masked.values)

        except Exception as e:
            print(f"  精确裁剪失败，使用边界框掩膜: {e}")
            # 方法2: 如果精确裁剪失败，使用边界框内的所有像元
            mask = np.ones_like(raster_clipped.values, dtype=bool)

        print(f"  中国区域像元数: {np.sum(mask):,}")
        print(f"  掩膜形状: {mask.shape}")

        return mask, raster_clipped

    except Exception as e:
        print(f"创建掩膜失败: {e}")
        return None, None


def parse_date_from_filename(filename):
    """从CHELSA文件名解析日期"""
    filename = Path(filename).name
    # 格式: CHELSA_sfcWind_05_05_2016_V.2.1.tif
    parts = filename.split('_')

    # 查找日期部分
    date_parts = []
    for part in parts:
        if part.isdigit():
            date_parts.append(part)

    if len(date_parts) >= 3:
        # 假设格式为: 日_月_年
        day = int(date_parts[0])
        month = int(date_parts[1])
        year = int(date_parts[2])
        try:
            return pd.Timestamp(f"{year:04d}-{month:02d}-{day:02d}")
        except:
            return None

    # 尝试正则表达式
    import re
    date_pattern = r'(\d{2})_(\d{2})_(\d{4})'
    match = re.search(date_pattern, filename)
    if match:
        day, month, year = match.groups()
        try:
            return pd.Timestamp(f"{year}-{month}-{day}")
        except:
            return None

    return None


def get_tif_files_by_year(year=2016):
    """获取指定年份的所有TIF文件"""
    print(f"搜索{year}年的TIF文件...")

    all_tif_files = list(data_dir.glob("*.tif"))
    if len(all_tif_files) == 0:
        print("错误: 未找到任何TIF文件!")
        return []

    # 按年份和日期筛选
    year_files = []
    for file in all_tif_files:
        date = parse_date_from_filename(file)
        if date and date.year == year:
            year_files.append((file, date))
        elif str(year) in file.name:
            year_files.append((file, date))

    # 按日期排序
    year_files.sort(key=lambda x: x[1] if x[1] else pd.Timestamp.min)

    print(f"找到 {len(year_files)} 个{year}年的TIF文件")

    # 显示前几个文件
    for i, (file, date) in enumerate(year_files[:5]):
        date_str = date.strftime('%Y-%m-%d') if date else '未知日期'
        print(f"  {i + 1:2d}. {file.name} ({date_str})")

    if len(year_files) > 5:
        print(f"  ... 还有 {len(year_files) - 5} 个文件")

    return year_files


def calculate_calm_days_with_mask(year=2016, threshold=0.5):
    """
    使用中国掩膜计算逐像元Calm Day频率

    返回:
        calm_frequency: 静风频率数组 (百分比)
        transform: 空间变换参数
        crs: 坐标系
        total_days: 总处理天数
        valid_dates: 有效日期列表
    """

    print(f"\n{'=' * 70}")
    print(f"计算{year}年中国区域Calm Day频率 (阈值: {threshold} m/s)")
    print(f"{'=' * 70}")

    # 1. 加载中国shapefile
    china_gdf = load_china_shapefile()
    if china_gdf is None:
        print("错误: 无法加载中国shapefile!")
        return None, None, None, 0, []

    # 2. 获取TIF文件
    tif_files = get_tif_files_by_year(year)
    if len(tif_files) == 0:
        print("错误: 未找到指定年份的TIF文件!")
        return None, None, None, 0, []

    # 3. 处理第一个文件以获取网格信息
    print(f"\n初始化处理...")
    first_file, first_date = tif_files[0]
    calm_days_count = None
    china_mask = None
    transform = None
    crs = None
    shape_info = None
    valid_dates = []

    # 4. 处理所有TIF文件
    print(f"\n开始处理{len(tif_files)}个文件...")
    total_processed = 0

    for i, (tif_file, date) in enumerate(tif_files):
        date_str = date.strftime('%Y-%m-%d') if date else '未知日期'

        if (i + 1) % 10 == 0 or i == 0 or i == len(tif_files) - 1:
            print(f"  进度: {i + 1}/{len(tif_files)} - {tif_file.name} ({date_str})")

        try:
            # 打开TIF文件
            with rio.open_rasterio(tif_file) as ds:
                # 提取风速数据
                if len(ds.shape) == 3:
                    wind_data = ds.isel(band=0)
                else:
                    wind_data = ds

                # 第一次迭代时初始化
                if calm_days_count is None:
                    print(f"\n初始化数组 (基于文件: {tif_file.name})")

                    # 创建中国区域掩膜
                    china_mask, wind_china = create_china_mask(wind_data, china_gdf)
                    if china_mask is None:
                        print("错误: 无法创建中国掩膜!")
                        continue

                    # 保存网格信息
                    shape_info = wind_china.shape
                    transform = wind_china.rio.transform()
                    crs = wind_china.rio.crs

                    print(f"  网格形状: {shape_info}")
                    print(f"  空间分辨率: {transform[0]:.4f} × {abs(transform[4]):.4f} 度")

                    # 初始化累积数组
                    calm_days_count = np.zeros(shape_info, dtype=np.float32)
                    print(f"  初始化累积数组完成")

                # 对于后续文件，使用相同的掩膜
                else:
                    # 裁剪到中国边界框
                    china_bbox = china_gdf.total_bounds
                    wind_china = wind_data.rio.clip_box(
                        minx=china_bbox[0], miny=china_bbox[1],
                        maxx=china_bbox[2], maxy=china_bbox[3]
                    )

                # 确保数据形状一致
                if wind_china.shape != shape_info:
                    print(f"  警告: 数据形状不匹配 {wind_china.shape} != {shape_info}")
                    # 尝试重新采样（简单起见，这里跳过）
                    continue

                # 计算静风掩膜（只在中国区域内）
                wind_values = wind_china.values
                calm_mask_china = np.zeros_like(wind_values, dtype=np.float32)

                # 只在中国区域内计算
                calm_mask_china[china_mask] = (wind_values[china_mask] < threshold).astype(np.float32)

                # 累加静风天数
                calm_days_count += calm_mask_china
                total_processed += 1

                if date:
                    valid_dates.append(date)

                # 统计信息
                calm_pixels = np.sum(calm_mask_china[china_mask])
                total_china_pixels = np.sum(china_mask)
                calm_ratio = calm_pixels / total_china_pixels * 100 if total_china_pixels > 0 else 0

                if (i + 1) % 20 == 0 or i == len(tif_files) - 1:
                    print(f"    本日静风像元: {calm_pixels:,}/{total_china_pixels:,} ({calm_ratio:.1f}%)")

        except Exception as e:
            print(f"  处理文件 {tif_file.name} 失败: {e}")
            continue

    print(f"\n处理完成，共处理 {total_processed} 天")

    # 5. 计算逐像元静风频率（百分比）
    if total_processed > 0 and calm_days_count is not None:
        calm_frequency = np.full_like(calm_days_count, np.nan, dtype=np.float32)

        # 只在中国区域内计算频率
        calm_frequency[china_mask] = (calm_days_count[china_mask] / total_processed) * 100.0

        # 计算统计信息
        china_pixel_values = calm_frequency[china_mask]
        china_pixel_values = china_pixel_values[~np.isnan(china_pixel_values)]

        if len(china_pixel_values) > 0:
            print(f"\n中国区域统计结果:")
            print(f"  有效像元数: {len(china_pixel_values):,}")
            print(f"  平均静风频率: {np.mean(china_pixel_values):.2f}%")
            print(f"  中位数静风频率: {np.median(china_pixel_values):.2f}%")
            print(f"  最小值: {np.min(china_pixel_values):.2f}%")
            print(f"  最大值: {np.max(china_pixel_values):.2f}%")
            print(f"  标准差: {np.std(china_pixel_values):.2f}%")

            # 频率分布统计
            print(f"\n  静风频率分布:")
            freq_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
            for low, high in freq_ranges:
                count = np.sum((china_pixel_values >= low) & (china_pixel_values < high))
                percentage = count / len(china_pixel_values) * 100
                print(f"    {low:3d}-{high:3d}%: {count:8,d} 像元 ({percentage:6.1f}%)")

        # 日期信息
        if valid_dates:
            valid_dates.sort()
            print(f"\n  日期范围: {valid_dates[0].strftime('%Y-%m-%d')} 到 {valid_dates[-1].strftime('%Y-%m-%d')}")
            print(f"  总天数: {len(valid_dates)}")

        return calm_frequency, transform, crs, total_processed, valid_dates

    else:
        print("错误: 未成功处理任何数据!")
        return None, None, None, 0, []


def create_china_calm_days_map(calm_frequency, transform, crs, year=2016, threshold=0.5, valid_dates=None):
    """
    创建中国区域Calm Day频率分布图
    """

    if calm_frequency is None:
        print("错误: 无数据可绘制!")
        return None

    print(f"\n创建中国区域静风频率分布图...")

    # 创建图形
    fig = plt.figure(figsize=(16, 12))

    # 设置投影
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 根据变换参数创建坐标网格
    height, width = calm_frequency.shape
    dx, rx, x0, ry, dy, y0 = transform[:6]

    # 计算像元中心坐标
    x_coords = np.arange(width) * dx + x0 + dx / 2
    y_coords = np.arange(height) * dy + y0 + dy / 2

    # 创建网格
    X, Y = np.meshgrid(x_coords, y_coords)

    # 绘制数据（只显示中国区域）
    # 创建掩膜用于显示（NaN区域透明）
    calm_frequency_masked = np.ma.array(calm_frequency, mask=np.isnan(calm_frequency))

    # 使用pcolormesh绘制
    im = ax.pcolormesh(X, Y, calm_frequency_masked,
                       cmap='RdYlBu_r',  # 反转的红黄蓝色彩
                       vmin=0, vmax=100,
                       shading='auto',
                       transform=ccrs.PlateCarree())

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        fraction=0.035, pad=0.02, shrink=0.8)
    cbar.set_label(f'Calm Days Frequency (%)',
                   fontsize=13, fontweight='bold')

    # 添加颜色条刻度说明
    cbar.ax.text(1.5, 0.5, f'Wind < {threshold} m/s',
                 transform=cbar.ax.transAxes, fontsize=10,
                 verticalalignment='center', rotation=270)

    # 添加中国边界
    try:
        china_gdf = load_china_shapefile()
        if china_gdf is not None:
            # 绘制中国边界
            china_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5,
                                    transform=ccrs.PlateCarree())

            # 填充中国区域（半透明）
            china_gdf.plot(ax=ax, color='none', edgecolor='black',
                           linewidth=0.5, alpha=0.1, transform=ccrs.PlateCarree())

            print("已添加中国边界")
    except Exception as e:
        print(f"添加中国边界失败: {e}")

    # 添加地理特征
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='darkblue')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle='--', edgecolor='gray', alpha=0.6)
    ax.add_feature(cfeature.LAKES, alpha=0.3, edgecolor='blue', facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.5)

    # 添加网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.6, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # 设置中国区域范围
    ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

    # 添加标题
    title = f'China - Calm Days Frequency Distribution ({year})\n'
    title += f'Wind Speed Threshold: {threshold} m/s | Masked by China Boundary'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # 添加统计信息
    china_pixel_values = calm_frequency[~np.isnan(calm_frequency)]
    if len(china_pixel_values) > 0:
        mean_val = np.mean(china_pixel_values)
        max_val = np.max(china_pixel_values)

        stats_text = f"""
        Pixel Value Meaning:
        • Value = % of days with wind < {threshold} m/s
        • Calculated only within China boundary
        • NaN values: Outside China

        Statistics (China Region):
        • Mean Frequency: {mean_val:.1f}%
        • Maximum Frequency: {max_val:.1f}%
        • Total Pixels: {len(china_pixel_values):,}

        Date Range: {valid_dates[0].strftime('%Y-%m-%d') if valid_dates else 'N/A'}
        to {valid_dates[-1].strftime('%Y-%m-%d') if valid_dates else 'N/A'}
        Total Days: {len(valid_dates) if valid_dates else 'N/A'}
        """

        plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                              edgecolor="gray", alpha=0.9))

    # 添加比例尺和指北针
    try:
        # 比例尺
        scale_lon = 100
        scale_lat = 20
        ax.plot([scale_lon, scale_lon + 10], [scale_lat, scale_lat],
                color='black', linewidth=2, transform=ccrs.PlateCarree())
        ax.text(scale_lon + 5, scale_lat - 0.5, '10° ≈ 1110 km',
                ha='center', va='top', fontsize=9, fontweight='bold',
                transform=ccrs.PlateCarree())

        # 指北针
        north_x, north_y = 132, 52
        ax.plot([north_x, north_x], [north_y, north_y + 0.8], 'k-', linewidth=2,
                transform=ccrs.PlateCarree())
        ax.plot([north_x - 0.15, north_x, north_x + 0.15],
                [north_y + 0.8, north_y + 1.0, north_y + 0.8],
                'k-', linewidth=2, transform=ccrs.PlateCarree())
        ax.text(north_x, north_y + 1.1, 'N', ha='center', va='bottom',
                fontsize=11, fontweight='bold', transform=ccrs.PlateCarree())
    except:
        pass

    # 调整布局
    plt.tight_layout()

    # 保存图像
    output_file = output_dir / f"China_CalmDays_{year}_threshold{threshold}_masked.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"分布图已保存: {output_file}")

    plt.show()

    return fig, ax


def save_masked_calm_frequency_tif(calm_frequency, transform, crs, year=2016, threshold=0.5):
    """
    将掩膜后的静风频率保存为GeoTIFF
    """

    if calm_frequency is None:
        return None

    print(f"\n保存掩膜后的静风频率为GeoTIFF...")

    try:
        # 创建xarray DataArray
        da = xr.DataArray(
            data=calm_frequency[np.newaxis, :, :],  # 添加波段维度
            dims=['band', 'y', 'x'],
            coords={
                'band': [1],
                'y': np.arange(calm_frequency.shape[0]),
                'x': np.arange(calm_frequency.shape[1])
            },
            attrs={
                'long_name': 'Calm days frequency in China',
                'units': 'percent',
                'description': f'Frequency of days with surface wind speed below {threshold} m/s within China boundary',
                'year': str(year),
                'threshold_wind_speed_m/s': str(threshold),
                'data_source': 'CHELSA daily surface wind',
                'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'mask_used': 'China boundary shapefile',
                'note': 'NaN values represent areas outside China'
            }
        )

        # 设置空间参考
        da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
        da.rio.write_crs(crs, inplace=True)
        da.rio.write_transform(transform, inplace=True)

        # 保存为GeoTIFF
        output_tif = output_dir / f"China_CalmDays_Frequency_{year}_threshold{threshold}_masked.tif"

        da.rio.to_raster(
            output_tif,
            driver='GTiff',
            dtype='float32',
            compress='LZW',
            nodata=-9999.0,
            BIGTIFF='IF_SAFER'  # 处理大文件
        )

        print(f"GeoTIFF文件已保存: {output_tif}")
        print(f"  文件大小: {output_tif.stat().st_size / (1024 * 1024):.1f} MB")
        print(f"  数据类型: float32")
        print(f"  无数据值: -9999")
        print(f"  压缩: LZW")

        return output_tif

    except Exception as e:
        print(f"保存GeoTIFF失败: {e}")
        return None


def create_china_statistics_report(calm_frequency, year=2016, threshold=0.5, valid_dates=None):
    """
    创建中国区域详细统计报告
    """

    if calm_frequency is None:
        return None

    print(f"\n生成中国区域详细统计报告...")

    # 提取中国区域数据
    china_data = calm_frequency[~np.isnan(calm_frequency)]

    if len(china_data) == 0:
        print("错误: 没有有效的中国区域数据!")
        return None

    # 创建图形
    fig = plt.figure(figsize=(18, 14))

    # 1. 中国区域频率分布直方图
    ax1 = plt.subplot(3, 3, 1)
    n, bins, patches = ax1.hist(china_data, bins=50, edgecolor='black',
                                alpha=0.7, color='steelblue', density=True)

    # 添加统计线
    mean_val = np.mean(china_data)
    median_val = np.median(china_data)
    std_val = np.std(china_data)

    ax1.axvline(x=mean_val, color='red', linestyle='--',
                linewidth=2, label=f'Mean: {mean_val:.1f}%')
    ax1.axvline(x=median_val, color='green', linestyle='--',
                linewidth=2, label=f'Median: {median_val:.1f}%')
    ax1.axvline(x=mean_val + std_val, color='orange', linestyle=':',
                linewidth=1.5, alpha=0.7)
    ax1.axvline(x=mean_val - std_val, color='orange', linestyle=':',
                linewidth=1.5, alpha=0.7)

    ax1.set_xlabel('Calm Days Frequency (%)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title(f'China Region - Frequency Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. 累积分布函数
    ax2 = plt.subplot(3, 3, 2)
    sorted_data = np.sort(china_data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, cdf, 'b-', linewidth=2)
    ax2.set_xlabel('Calm Days Frequency (%)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 箱线图
    ax3 = plt.subplot(3, 3, 3)
    bp = ax3.boxplot(china_data, vert=False, patch_artist=True,
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'),
                     flierprops=dict(marker='o', color='red', alpha=0.5))
    ax3.set_xlabel('Calm Days Frequency (%)', fontsize=11)
    ax3.set_title('Box Plot', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. 频率分类统计
    ax4 = plt.subplot(3, 3, 4)
    freq_categories = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    freq_bins = [0, 20, 40, 60, 80, 100]

    category_counts = []
    for i in range(len(freq_bins) - 1):
        count = np.sum((china_data >= freq_bins[i]) & (china_data < freq_bins[i + 1]))
        category_counts.append(count)

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(freq_categories)))
    bars = ax4.bar(freq_categories, category_counts, color=colors, edgecolor='black')
    ax4.set_xlabel('Frequency Category', fontsize=11)
    ax4.set_ylabel('Number of Pixels', fontsize=11)
    ax4.set_title('Frequency Category Distribution', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # 在柱子上添加数值
    for bar, count in zip(bars, category_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + max(category_counts) * 0.01,
                 f'{count:,}', ha='center', va='bottom', fontsize=9)

    # 5. 百分比饼图
    ax5 = plt.subplot(3, 3, 5)
    category_percentages = [count / len(china_data) * 100 for count in category_counts]
    explode = [0.05 if pct > 20 else 0 for pct in category_percentages]
    wedges, texts, autotexts = ax5.pie(category_percentages, labels=freq_categories,
                                       autopct='%1.1f%%', startangle=90,
                                       colors=colors, explode=explode,
                                       shadow=True)
    ax5.set_title('Frequency Category Percentage', fontsize=12, fontweight='bold')

    # 6. 统计摘要
    ax6 = plt.subplot(3, 3, (6, 9))
    ax6.axis('off')

    # 计算详细统计
    q1, q3 = np.percentile(china_data, [25, 75])
    iqr = q3 - q1

    stats_text = f"""
    China Region Statistical Summary - {year}
    {'=' * 50}

    Dataset Information:
    • Total Pixels in China: {len(china_data):,}
    • Calm Threshold: {threshold} m/s
    • Analysis Period: {len(valid_dates) if valid_dates else 'N/A'} days
    • Processing Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

    Central Tendency:
    • Mean Frequency: {mean_val:.2f}%
    • Median Frequency: {median_val:.2f}%
    • Mode (approx): {float(bins[np.argmax(n)]):.2f}%

    Dispersion Statistics:
    • Standard Deviation: {std_val:.2f}%
    • Variance: {std_val ** 2:.2f}
    • Coefficient of Variation: {(std_val / mean_val * 100):.2f}%
    • Interquartile Range (IQR): {iqr:.2f}%

    Range Statistics:
    • Minimum: {np.min(china_data):.2f}%
    • Maximum: {np.max(china_data):.2f}%
    • Range: {np.max(china_data) - np.min(china_data):.2f}%

    Percentile Statistics:
    • 5th Percentile: {np.percentile(china_data, 5):.2f}%
    • 25th Percentile (Q1): {q1:.2f}%
    • 75th Percentile (Q3): {q3:.2f}%
    • 95th Percentile: {np.percentile(china_data, 95):.2f}%

    Frequency Categories:
    • 0-20%: {category_counts[0]:,} pixels ({category_percentages[0]:.1f}%)
    • 20-40%: {category_counts[1]:,} pixels ({category_percentages[1]:.1f}%)
    • 40-60%: {category_counts[2]:,} pixels ({category_percentages[2]:.1f}%)
    • 60-80%: {category_counts[3]:,} pixels ({category_percentages[3]:.1f}%)
    • 80-100%: {category_counts[4]:,} pixels ({category_percentages[4]:.1f}%)
    """

    ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan",
                       edgecolor="gray", alpha=0.9))

    plt.suptitle(f'CHELSA Calm Days Frequency - China Region Detailed Statistics ({year})',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # 保存统计图
    stats_file = output_dir / f"China_CalmDays_Statistics_{year}_threshold{threshold}.png"
    plt.savefig(stats_file, dpi=300, bbox_inches='tight')
    print(f"统计报告已保存: {stats_file}")

    plt.show()

    return fig


def main():
    """主函数"""
    print("CHELSA全球TIF风速数据 - 中国区域Calm Day频率分析")
    print("=" * 80)
    print("说明: 使用中国shapefile掩膜，只计算中国区域内的像元")
    print("=" * 80)

    # 检查必要库
    required_libs = ['rioxarray', 'geopandas', 'cartopy']
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✓ {lib}库已安装")
        except ImportError:
            print(f"✗ 需要安装{lib}库!")
            print(f"  请运行: pip install {lib}")
            if lib == 'cartopy':
                print("  注意: cartopy在Windows上可能需要先安装依赖")
                print("  可尝试: conda install -c conda-forge cartopy")

    # 参数设置
    year = 2015
    thresholds = [0.5, 1.0]  # 静风阈值

    all_results = []

    for threshold in thresholds:
        print(f"\n{'=' * 80}")
        print(f"开始分析: 年份={year}, 静风阈值={threshold} m/s")
        print(f"{'=' * 80}")

        # 计算静风频率（使用中国掩膜）
        calm_freq, transform, crs, total_days, valid_dates = calculate_calm_days_with_mask(
            year, threshold
        )

        if calm_freq is not None:
            # 创建分布图
            fig_map = create_china_calm_days_map(calm_freq, transform, crs,
                                                 year, threshold, valid_dates)

            # 保存为GeoTIFF
            tif_file = save_masked_calm_frequency_tif(calm_freq, transform, crs,
                                                      year, threshold)

            # 创建统计报告
            fig_stats = create_china_statistics_report(calm_freq, year,
                                                       threshold, valid_dates)

            all_results.append({
                'year': year,
                'threshold': threshold,
                'total_days': total_days,
                'output_tif': tif_file
            })

            print(f"\n✓ 分析完成: {threshold} m/s 阈值")
        else:
            print(f"\n✗ 分析失败: {threshold} m/s 阈值")

    print(f"\n{'=' * 80}")
    print("所有分析完成!")
    print(f"输出目录: {output_dir}")

    if all_results:
        print("\n生成的文件:")
        for result in all_results:
            if result['output_tif']:
                print(f"  • {result['output_tif'].name}")

    print(f"{'=' * 80}")

    return all_results


if __name__ == "__main__":
    results = main()