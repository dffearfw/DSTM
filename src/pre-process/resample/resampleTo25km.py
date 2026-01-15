import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import glob


def resample_raster(input_path, output_path, reference_path,
                    resampling_method='bilinear', nodata=None):
    """
    重采样遥感图像到参考图像的分辨率和空间参考

    参数：
    input_path: str - 输入TIFF文件路径
    output_path: str - 输出TIFF文件路径
    reference_path: str - 参考分辨率图像路径
    resampling_method: str - 重采样方法，可选：
        'nearest', 'bilinear', 'cubic', 'cubic_spline',
        'lanczos', 'average', 'mode', 'max', 'min', 'med', 'q1', 'q3'
    nodata: float/int - 设置输出图像的nodata值
    """

    # 定义重采样方法映射
    resampling_methods = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'cubic_spline': Resampling.cubic_spline,
        'lanczos': Resampling.lanczos,
        'average': Resampling.average,
        'mode': Resampling.mode,
        'max': Resampling.max,
        'min': Resampling.min,
        'med': Resampling.med,
        'q1': Resampling.q1,
        'q3': Resampling.q3
    }

    if resampling_method not in resampling_methods:
        raise ValueError(f"不支持的采样方法: {resampling_method}")

    resample_algo = resampling_methods[resampling_method]

    print(f"开始重采样: {os.path.basename(input_path)}")

    try:
        # 打开参考图像获取目标参数
        with rasterio.open(reference_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            ref_bounds = ref.bounds

        # 打开输入图像
        with rasterio.open(input_path) as src:
            # 如果nodata未指定，使用输入图像的nodata值
            if nodata is None:
                nodata = src.nodata

            # 计算重采样后的transform
            transform, width, height = calculate_default_transform(
                src.crs, ref_crs, ref_width, ref_height,
                *ref_bounds
            )

            # 创建输出文件
            output_meta = src.meta.copy()
            output_meta.update({
                'crs': ref_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': nodata
            })

            # 重采样每个波段
            with rasterio.open(output_path, 'w', **output_meta) as dst:
                for i in range(1, src.count + 1):
                    # 创建目标数组
                    destination = np.zeros((height, width), dtype=output_meta['dtype'])

                    # 重采样
                    reproject(
                        source=rasterio.band(src, i),
                        destination=destination,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=ref_crs,
                        resampling=resample_algo,
                        src_nodata=src.nodata,
                        dst_nodata=nodata
                    )

                    # 写入波段
                    dst.write(destination, i)

        print(f"  完成: {os.path.basename(output_path)}")
        return True

    except Exception as e:
        print(f"  错误: {str(e)}")
        return False


def batch_resample_directory(input_dir, output_dir, reference_path,
                             resampling_method='bilinear'):
    """
    批量重采样一个目录下的所有TIFF文件

    参数：
    input_dir: str - 输入目录路径
    output_dir: str - 输出目录路径
    reference_path: str - 参考图像路径
    resampling_method: str - 重采样方法
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有TIFF文件
    tif_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        tif_files.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"找到 {len(tif_files)} 个TIFF文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"参考图像: {reference_path}")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for i, input_file in enumerate(tif_files, 1):
        # 获取文件名
        filename = os.path.basename(input_file)

        # 构建输出路径
        output_file = os.path.join(output_dir, filename)

        print(f"[{i}/{len(tif_files)}] 处理: {filename}")

        # 检查是否已存在
        if os.path.exists(output_file):
            print(f"  警告: 文件已存在，跳过")
            continue

        # 执行重采样
        success = resample_raster(
            input_path=input_file,
            output_path=output_file,
            reference_path=reference_path,
            resampling_method=resampling_method
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    print("-" * 50)
    print(f"处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"总计: {len(tif_files)} 个文件")


def batch_resample_with_suffix(input_dir, output_dir, reference_path,
                               resampling_method='bilinear', suffix='_resampled'):
    """
    批量重采样并添加后缀（避免覆盖）

    参数：
    input_dir: str - 输入目录路径
    output_dir: str - 输出目录路径
    reference_path: str - 参考图像路径
    resampling_method: str - 重采样方法
    suffix: str - 输出文件名后缀
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有TIFF文件
    tif_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        tif_files.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"找到 {len(tif_files)} 个TIFF文件")

    success_count = 0

    for i, input_file in enumerate(tif_files, 1):
        # 获取文件名和扩展名
        filename = os.path.basename(input_file)
        name, ext = os.path.splitext(filename)

        # 构建带后缀的输出文件名
        new_filename = f"{name}{suffix}{ext}"
        output_file = os.path.join(output_dir, new_filename)

        print(f"[{i}/{len(tif_files)}] 处理: {filename} -> {new_filename}")

        # 执行重采样
        success = resample_raster(
            input_path=input_file,
            output_path=output_file,
            reference_path=reference_path,
            resampling_method=resampling_method
        )

        if success:
            success_count += 1

    print(f"处理完成! 成功处理 {success_count} 个文件")


if __name__ == "__main__":
    # ==================== 批量处理示例 ====================
    # 设置路径
    input_directory = "G:/王扬/chelsa_sfxwind/XINJIANG/2016"  # 输入目录
    output_directory = "G:/王扬/chelsa_sfxwind/XINJIANG/resap25km"  # 输出目录
    reference_image = "G:/王扬/fusedSWE/XGB_SWE_DAILY_025/XGB_SWE_DAILY_025_19800101.tif"  # 参考图像

    # 方法1: 直接批量处理
    print("开始批量重采样...")
    batch_resample_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        reference_path=reference_image,
        resampling_method='bilinear'
    )

    # 方法2: 批量处理并添加后缀（推荐，避免覆盖）
    # print("开始批量重采样（带后缀）...")
    # batch_resample_with_suffix(
    #     input_dir=input_directory,
    #     output_dir=output_directory,
    #     reference_path=reference_image,
    #     resampling_method='bilinear',
    #     suffix='_25km'  # 添加后缀，避免同名覆盖
    # )