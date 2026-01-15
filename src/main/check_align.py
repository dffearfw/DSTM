# check_and_align_raster_final_fixed.py
import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os


def align_raster_to_target(src_path, target_bounds, target_transform, target_shape, band_index=0):
    """将源栅格对齐到目标栅格"""
    with rasterio.open(src_path) as src:
        # 1. 读取源数据
        src_data = src.read()
        src_bounds = src.bounds
        src_transform = src.transform

        print(f"\n对齐 {os.path.basename(str(src_path))}:")
        print(f"  源范围: {src_bounds}")
        print(f"  目标范围: {target_bounds}")

        # 2. 创建目标数组
        target_height, target_width = target_shape

        # 如果源数据是多波段，只对齐指定的波段
        if len(src_data.shape) == 3:  # 多波段
            n_bands = src_data.shape[0]
            aligned_data = np.zeros((n_bands, target_height, target_width), dtype=src_data.dtype)
            band_to_use = src_data[band_index]  # 使用指定波段
            print(f"  多波段数据: {n_bands}个波段，使用第{band_index + 1}波段")
        else:  # 单波段
            aligned_data = np.zeros((target_height, target_width), dtype=src_data.dtype)
            band_to_use = src_data

        # 3. 逐像素复制（最精确的方法）
        for row in range(target_height):
            for col in range(target_width):
                # 目标像素中心坐标
                target_x, target_y = target_transform * (col + 0.5, row + 0.5)

                # 在源数据中的像素坐标
                src_row, src_col = src.index(target_x, target_y)

                # 检查是否在源数据范围内
                if (0 <= src_row < src.height and 0 <= src_col < src.width):
                    if len(src_data.shape) == 3:
                        aligned_data[:, row, col] = src_data[:, src_row, src_col]
                    else:
                        aligned_data[row, col] = src_data[src_row, src_col]

        return aligned_data, target_transform


def test_alignment():
    """测试对齐效果"""
    # 请修改为你的实际路径
    label_file = r"G:\王扬\fusedSWE\XINJIANG\2016-12-28.tif"
    lst_file = r"G:\王扬\lst\ERA5_Xinjiang_2016_025deg\ERA5_ST_201505_UTC8_27830m.tif"  # 注意这是5月文件
    chelsa_file = r"G:\王扬\chelsa_sfxwind\XINJIANG\resap25km\XINGJANG_CHELSA_sfcWind_01_01_2015_V.2.1_resampled.tif"

    # 检查文件是否存在
    for file_path in [label_file, lst_file, chelsa_file]:
        if not Path(file_path).exists():
            print(f"❌ 文件不存在: {file_path}")
            return

    # 1. 读取目标网格（标签）
    with rasterio.open(label_file) as target_ds:
        target_bounds = target_ds.bounds
        target_transform = target_ds.transform
        target_shape = target_ds.shape
        target_crs = target_ds.crs

        print("目标网格信息（标签）:")
        print(f"  尺寸: {target_shape}")
        print(f"  范围: {target_bounds}")
        print(f"  变换矩阵: \n{target_transform}")
        print(f"  CRS: {target_crs}")

    # 2. 对齐LST
    print("\n" + "=" * 60)
    print("测试LST对齐")
    print("=" * 60)

    with rasterio.open(lst_file) as lst_ds:
        print(f"LST文件信息:")
        print(f"  波段数: {lst_ds.count}")
        print(f"  尺寸: {lst_ds.width} × {lst_ds.height}")

        # 获取LST的第一天数据（第1波段）
        lst_data_band1 = lst_ds.read(1)  # 第一天

        # 方法1：窗口裁剪
        print("\n方法1：窗口裁剪（基于地理范围）")
        window = lst_ds.window(*target_bounds)
        lst_cropped = lst_ds.read(1, window=window)
        print(f"  裁剪后形状: {lst_cropped.shape}")
        print(f"  期望形状: {target_shape}")

        # 方法2：逐像素对齐（只对齐第一个波段）
        print("\n方法2：逐像素对齐（第1波段）")
        lst_aligned_all, _ = align_raster_to_target(
            lst_file,
            target_bounds,
            target_transform,
            target_shape,
            band_index=0  # 只对齐第1波段
        )

        # 取出对齐后的第1波段
        if len(lst_aligned_all.shape) == 3:
            lst_aligned = lst_aligned_all[0]  # 取第1波段
        else:
            lst_aligned = lst_aligned_all

        print(f"  对齐后形状: {lst_aligned.shape}")

        # 检查对齐效果
        print("\n对齐效果检查:")

        # 检查几个关键点
        test_points = [
            (0, 0),  # 左上角
            (target_shape[0] // 2, target_shape[1] // 2),  # 中心
            (target_shape[0] - 1, target_shape[1] - 1)  # 右下角
        ]

        for row, col in test_points:
            # 目标像素中心坐标
            target_x, target_y = target_transform * (col + 0.5, row + 0.5)

            # 在LST中的原始位置
            lst_row, lst_col = lst_ds.index(target_x, target_y)

            # 获取原始LST值（第1波段）
            if 0 <= lst_row < lst_ds.height and 0 <= lst_col < lst_ds.width:
                lst_value = lst_ds.read(1)[lst_row, lst_col]
            else:
                lst_value = np.nan

            # 获取对齐后的LST值
            aligned_value = lst_aligned[row, col]

            # 处理可能的NaN
            if np.isnan(aligned_value):
                aligned_display = "NaN"
            else:
                aligned_display = f"{float(aligned_value):.2f}"

            print(f"  像素({row},{col}): 原始LST={lst_value:.2f}, 对齐后={aligned_display}, "
                  f"目标位置=({target_x:.3f}, {target_y:.3f})")

    # 3. 对齐CHELSA
    print("\n" + "=" * 60)
    print("测试CHELSA对齐")
    print("=" * 60)

    with rasterio.open(chelsa_file) as chelsa_ds:
        print(f"CHELSA文件信息:")
        print(f"  波段数: {chelsa_ds.count}")

        # 方法2：逐像素对齐
        chelsa_aligned_all, _ = align_raster_to_target(
            chelsa_file,
            target_bounds,
            target_transform,
            target_shape
        )

        if len(chelsa_aligned_all.shape) == 3:
            chelsa_aligned = chelsa_aligned_all[0]
        else:
            chelsa_aligned = chelsa_aligned_all

        print(f"  CHELSA对齐后形状: {chelsa_aligned.shape}")

        # 检查CHELSA对齐效果
        print("\nCHELSA对齐效果检查:")
        for row, col in test_points:
            target_x, target_y = target_transform * (col + 0.5, row + 0.5)
            chelsa_row, chelsa_col = chelsa_ds.index(target_x, target_y)

            if 0 <= chelsa_row < chelsa_ds.height and 0 <= chelsa_col < chelsa_ds.width:
                chelsa_value = chelsa_ds.read(1)[chelsa_row, chelsa_col]
            else:
                chelsa_value = np.nan

            aligned_value = chelsa_aligned[row, col]

            # 处理可能的NaN
            if np.isnan(aligned_value):
                aligned_display = "NaN"
            else:
                aligned_display = f"{float(aligned_value):.2f}"

            print(f"  像素({row},{col}): 原始CHELSA={chelsa_value:.2f}, 对齐后={aligned_display}")

    # 4. 保存对齐结果用于可视化检查
    print("\n" + "=" * 60)
    print("保存对齐结果")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path("alignment_test")
    output_dir.mkdir(exist_ok=True)

    # 保存LST对齐结果（第1波段）
    lst_output = output_dir / "lst_aligned_band1.tif"
    with rasterio.open(
            lst_output, 'w',
            driver='GTiff',
            height=target_shape[0],
            width=target_shape[1],
            count=1,
            dtype=lst_aligned.dtype,
            crs=target_crs,
            transform=target_transform
    ) as dst:
        dst.write(lst_aligned, 1)
    print(f"  LST对齐结果（第1波段）保存到: {lst_output}")

    # 保存CHELSA对齐结果
    chelsa_output = output_dir / "chelsa_aligned.tif"
    with rasterio.open(
            chelsa_output, 'w',
            driver='GTiff',
            height=target_shape[0],
            width=target_shape[1],
            count=1,
            dtype=chelsa_aligned.dtype,
            crs=target_crs,
            transform=target_transform
    ) as dst:
        dst.write(chelsa_aligned, 1)
    print(f"  CHELSA对齐结果保存到: {chelsa_output}")

    # 5. 简单统计
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)

    # 读取标签数据
    with rasterio.open(label_file) as label_ds:
        label_data = label_ds.read(1)

    print(f"标签数据:")
    print(f"  范围: [{np.nanmin(label_data):.2f}, {np.nanmax(label_data):.2f}]")
    print(f"  有效像素: {np.sum(~np.isnan(label_data))}/{label_data.size}")

    print(f"\nLST对齐数据（第1波段）:")
    print(f"  范围: [{np.nanmin(lst_aligned):.2f}, {np.nanmax(lst_aligned):.2f}]")
    print(f"  有效像素: {np.sum(~np.isnan(lst_aligned))}/{lst_aligned.size}")

    print(f"\nCHELSA对齐数据:")
    print(f"  范围: [{np.nanmin(chelsa_aligned):.2f}, {np.nanmax(chelsa_aligned):.2f}]")
    print(f"  有效像素: {np.sum(~np.isnan(chelsa_aligned))}/{chelsa_aligned.size}")

    print("\n✅ 对齐测试完成！")
    print(f"请查看 {output_dir} 目录中的结果文件")


if __name__ == "__main__":
    print("栅格对齐测试")
    print("=" * 60)

    try:
        test_alignment()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()