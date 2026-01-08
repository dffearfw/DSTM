# resample_to_target_resolution.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据目标分辨率重采样动态变量
1. 读取标签数据（ERA5 SWE）获取目标分辨率和CRS
2. 将动态变量重采样到与标签数据相同的分辨率和CRS
3. 输出重采样后的TIFF文件
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling
from pathlib import Path
import glob
import re
from datetime import datetime
import warnings
import sys
import os
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')


# ============= 配置 =============
class ResampleConfig:
    """重采样配置"""

    # 输入配置
    REGION = "XINJIANG"
    YEARS = [2015, 2016]
    VARIABLES = ["chelsa_sfxwind", "pr"]

    # 原始数据根目录
    FEATURE_ROOT = Path(r"G:\王扬")

    # 标签数据路径（用于获取目标分辨率）
    LABEL_ROOT = Path(r"G:\王扬\era5_swe\xinjiang")

    # 输出目录
    OUTPUT_ROOT = Path(r"G:\王扬\target_resolution_resampled")

    # 重采样方法
    RESAMPLING_METHOD = RasterioResampling.bilinear

    # 创建输出目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_var_path(var: str, year: int) -> Path:
        """获取变量原始文件路径"""
        if var == "chelsa_sfxwind":
            return ResampleConfig.FEATURE_ROOT / "chelsa_sfxwind" / ResampleConfig.REGION / str(year)
        elif var == "pr":
            return ResampleConfig.FEATURE_ROOT / "pr_xinjiang" / str(year)
        else:
            raise ValueError(f"未知的动态变量: {var}")

    @staticmethod
    def get_output_path(var: str, year: int) -> Path:
        """获取输出文件路径"""
        output_dir = ResampleConfig.OUTPUT_ROOT / var / ResampleConfig.REGION / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def find_label_file() -> Path:
        """查找标签文件获取目标分辨率"""
        # 查找ERA5 SWE文件
        patterns = [
            "*.tif",
            f"{ResampleConfig.REGION}*.tif",
            "*era5*.tif",
            "*swe*.tif"
        ]

        for pattern in patterns:
            files = list(ResampleConfig.LABEL_ROOT.glob(pattern))
            if files:
                print(f"找到标签文件: {files[0]}")
                return files[0]

        raise FileNotFoundError(f"在 {ResampleConfig.LABEL_ROOT} 中未找到标签文件")


def get_target_resolution(label_path: Path):
    """从标签文件获取目标分辨率、CRS和transform"""
    with rasterio.open(label_path) as src:
        # 获取标签数据的信息
        target_crs = src.crs
        target_transform = src.transform
        target_width = src.width
        target_height = src.height
        target_bounds = src.bounds

        # 计算分辨率（单位：米/度）
        if target_crs.is_geographic:
            # 地理坐标系（经纬度），计算度/像素
            # 近似计算分辨率（简化的，实际更复杂）
            res_x = (target_bounds.right - target_bounds.left) / target_width
            res_y = (target_bounds.top - target_bounds.bottom) / target_height
            resolution = f"约 {res_x:.6f}° × {res_y:.6f}°"
        else:
            # 投影坐标系，直接获取分辨率
            res_x = target_transform.a
            res_y = -target_transform.e  # 注意：e通常是负值
            resolution = f"{abs(res_x):.1f}m × {abs(res_y):.1f}m"

        print(f"目标CRS: {target_crs}")
        print(f"目标分辨率: {resolution}")
        print(f"目标尺寸: {target_width} × {target_height}")
        print(f"目标范围: {target_bounds}")

        return {
            'crs': target_crs,
            'transform': target_transform,
            'width': target_width,
            'height': target_height,
            'bounds': target_bounds,
            'resolution': resolution
        }


def resample_to_target(input_path: Path, target_info: dict, output_path: Path):
    """将单个文件重采样到目标分辨率"""
    try:
        with rasterio.open(input_path) as src:
            # 读取数据
            data = src.read(1)

            # 如果源数据已经是目标CRS和分辨率，直接复制
            if src.crs == target_info['crs'] and src.width == target_info['width'] and src.height == target_info[
                'height']:
                print(f"  {input_path.name}: 已经匹配目标分辨率，直接复制")
                profile = src.profile.copy()
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data, 1)
                return True

            # 计算重采样参数
            transform, width, height = calculate_default_transform(
                src.crs,
                target_info['crs'],
                src.width,
                src.height,
                *src.bounds,
                dst_width=target_info['width'],
                dst_height=target_info['height']
            )

            # 创建输出文件
            profile = src.profile.copy()
            profile.update({
                'crs': target_info['crs'],
                'transform': transform,
                'width': width,
                'height': height,
                'driver': 'GTiff',
                'compress': 'lzw',
                'predictor': 2
            })

            # 重采样
            with rasterio.open(output_path, 'w', **profile) as dst:
                reproject(
                    source=data,
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_info['crs'],
                    resampling=ResampleConfig.RESAMPLING_METHOD
                )

            print(f"  ✓ {input_path.name} -> {output_path.name}")
            return True

    except Exception as e:
        print(f"  ✗ 处理 {input_path.name} 失败: {e}")
        return False


def parse_date_from_filename(filename: str) -> datetime:
    """从文件名解析日期"""
    patterns = [
        r'(\d{2})_(\d{2})_(\d{4})',  # 日_月_年
        r'(\d{4})(\d{2})(\d{2})',  # 年月日
        r'(\d{4})[-_](\d{2})[-_](\d{2})'  # 年-月-日
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if len(groups[0]) == 4:  # 年在前
                    year, month, day = groups
                else:  # 日在前面
                    day, month, year = groups
                return datetime(int(year), int(month), int(day))

    # 默认返回一个日期（如果需要）
    print(f"警告: 无法从 {filename} 解析日期")
    return datetime(2015, 1, 1)


def main():
    """主函数"""
    print("=" * 60)
    print("动态变量重采样到目标分辨率")
    print("=" * 60)

    # 1. 获取目标分辨率信息
    print("\n1. 获取标签数据的分辨率信息...")
    try:
        label_file = ResampleConfig.find_label_file()
        target_info = get_target_resolution(label_file)

        # 保存目标信息
        target_info_path = ResampleConfig.OUTPUT_ROOT / "target_resolution_info.json"
        with open(target_info_path, 'w') as f:
            json.dump({
                'label_file': str(label_file),
                'crs': str(target_info['crs']),
                'width': target_info['width'],
                'height': target_info['height'],
                'resolution': target_info['resolution'],
                'bounds': list(target_info['bounds'])
            }, f, indent=2)
        print(f"目标信息已保存到: {target_info_path}")

    except Exception as e:
        print(f"获取目标分辨率失败: {e}")
        return

    # 2. 处理每个变量
    print("\n2. 开始重采样动态变量...")

    stats = {
        'total_files': 0,
        'successful': 0,
        'failed': 0,
        'skipped': 0
    }

    for var in ResampleConfig.VARIABLES:
        print(f"\n处理变量: {var}")

        for year in ResampleConfig.YEARS:
            print(f"\n  年份: {year}")

            # 获取输入输出路径
            input_dir = ResampleConfig.get_var_path(var, year)
            output_dir = ResampleConfig.get_output_path(var, year)

            if not input_dir.exists():
                print(f"    ✗ 输入目录不存在: {input_dir}")
                stats['skipped'] += 1
                continue

            # 获取所有TIFF文件
            input_files = sorted(list(input_dir.glob("*.tif")))
            if not input_files:
                print(f"    ⚠ 目录中没有TIFF文件: {input_dir}")
                continue

            print(f"    找到 {len(input_files)} 个文件")

            # 处理每个文件
            for input_file in tqdm(input_files, desc=f"    {year}"):
                stats['total_files'] += 1

                # 解析日期
                date_obj = parse_date_from_filename(input_file.name)
                date_str = date_obj.strftime("%Y%m%d")

                # 创建输出文件名
                output_filename = f"{var}_{ResampleConfig.REGION}_{date_str}_resampled.tif"
                output_path = output_dir / output_filename

                # 跳过已存在的文件
                if output_path.exists():
                    stats['skipped'] += 1
                    continue

                # 重采样
                success = resample_to_target(input_file, target_info, output_path)
                if success:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1

    # 3. 打印统计信息
    print("\n" + "=" * 60)
    print("重采样完成！统计信息:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  成功: {stats['successful']}")
    print(f"  失败: {stats['failed']}")
    print(f"  跳过: {stats['skipped']}")

    # 保存统计信息
    stats_path = ResampleConfig.OUTPUT_ROOT / "resampling_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存到: {stats_path}")

    print("\n" + "=" * 60)
    print("注意事项:")
    print("1. 所有动态变量已重采样到与标签数据相同的分辨率和CRS")
    print("2. 输出文件保存在: {ResampleConfig.OUTPUT_ROOT}")
    print("3. 如果数据已经是目标分辨率，会直接复制而不重采样")
    print("=" * 60)


if __name__ == "__main__":
    main()