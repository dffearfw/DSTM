
import os
import re
import rasterio
from rasterio.mask import mask
import geopandas as gpd


def mask_2016_chelsea_pr(input_dir, output_dir, shp_file):
    """
    专门处理2016年的CHELSA降水数据

    参数:
    - input_dir: 输入目录（包含2016-2017年数据）
    - output_dir: 输出目录（只存放2016年掩膜结果）
    - shp_file: shapefile文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取新疆行政区shp文件
    print("Loading shapefile...")
    gdf = gpd.read_file(shp_file)
    geometry = gdf.geometry.unary_union

    # 正则表达式匹配2016年数据
    # 匹配模式: CHELSA_pr_月_日_2016_V.*.tif
    pattern_2016 = re.compile(r'CHELSA_pr_.*_2016_.*\.tif$')

    # 获取所有TIFF文件
    all_files = os.listdir(input_dir)
    tif_files = [f for f in all_files if f.endswith('.tif')]

    print(f"Found {len(tif_files)} TIFF files in input directory")

    # 筛选2016年文件
    files_2016 = []
    for filename in tif_files:
        # 方法1：使用正则表达式
        if pattern_2016.match(filename):
            files_2016.append(filename)
        # 方法2：简单判断文件名中是否包含2016
        elif '2016' in filename and filename.startswith('CHELSA_pr_'):
            files_2016.append(filename)

    # 去重
    files_2016 = list(set(files_2016))
    files_2016.sort()

    print(f"Found {len(files_2016)} files from 2016")

    if len(files_2016) == 0:
        print("No 2016 files found! Showing all available files:")
        for f in sorted(tif_files)[:20]:  # 只显示前20个文件
            print(f"  - {f}")
        return

    # 处理2016年文件
    processed_count = 0
    for filename in files_2016:
        input_path = os.path.join(input_dir, filename)
        output_filename = f"XINGJIANG_{filename}"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\nProcessing 2016: {filename}")

        try:
            # 读取栅格文件
            with rasterio.open(input_path) as src:
                # 执行掩膜提取
                out_image, out_transform = mask(
                    src,
                    [geometry],
                    crop=True,
                    all_touched=True
                )

                # 获取元数据
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # 写入输出文件
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)

            print(f"✓ Saved: {output_filename}")
            processed_count += 1

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    # 输出总结
    print(f"\n{'=' * 60}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Shapefile: {shp_file}")
    print(f"Total 2016 files found: {len(files_2016)}")
    print(f"Successfully processed: {processed_count}")

    # 显示处理的文件列表
    print(f"\nFiles processed from 2016:")
    for i, filename in enumerate(files_2016, 1):
        print(f"{i:2d}. {filename}")


if __name__ == "__main__":
    # 配置参数 - 根据你的实际路径修改
    input_dir = "G:/朱佳腾/降水-CHELSA/2016-2017"
    output_dir = "G:/王扬/pr_xinjiang/2016"
    shp_file = "XINGJIANG/XINGJIANG.shp"

    # 运行处理
    mask_2016_chelsea_pr(input_dir, output_dir, shp_file)

    print("\nProcessing completed!")