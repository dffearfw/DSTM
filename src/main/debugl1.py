import h5py
import numpy as np
import pandas as pd


def debug_smap_data(h5_path):
    """深入调试SMAP数据"""

    print("深入调试SMAP数据...")
    with h5py.File(h5_path, 'r') as f:
        proj = f['Global_Projection']

        # 1. 读取所有基本数据
        lat = proj['cell_lat'][:].astype(np.float32)
        lon = proj['cell_lon'][:].astype(np.float32)

        # 2. 检查亮温数据
        tb_v = None
        for key in ['cell_tb_v_fore', 'cell_tb_v_surface_corrected_fore',
                    'cell_tb_v_aft', 'cell_tb_v_surface_corrected_aft']:
            if key in proj:
                tb_v = proj[key][:].astype(np.float32)
                print(f"使用 {key}, 形状: {tb_v.shape}")
                break

        # 3. 新疆范围
        xj_mask = (lat >= 34) & (lat <= 50) & (lon >= 73) & (lon <= 97)

        print(f"\n新疆范围内总点数: {xj_mask.sum()}")

        if xj_mask.sum() > 0:
            print(f"新疆点纬度: {lat[xj_mask][:5]}")
            print(f"新疆点经度: {lon[xj_mask][:5]}")

            if tb_v is not None:
                print(f"新疆点亮温: {tb_v[xj_mask][:5]}")

        # 4. 创建数据分布表
        print("\n全球数据分布（每30度一格）:")
        lat_bins = np.arange(-90, 91, 30)
        lon_bins = np.arange(-180, 181, 30)

        for i in range(len(lat_bins) - 1):
            for j in range(len(lon_bins) - 1):
                mask = (lat >= lat_bins[i]) & (lat < lat_bins[i + 1]) & \
                       (lon >= lon_bins[j]) & (lon < lon_bins[j + 1])
                if mask.sum() > 0:
                    print(f"  区域[{lat_bins[i]:3.0f}°-{lat_bins[i + 1]:3.0f}°, "
                          f"{lon_bins[j]:4.0f}°-{lon_bins[j + 1]:4.0f}°]: "
                          f"{mask.sum():6d} 点")


# 运行调试
debug_smap_data("G:/王扬/smap_data/SMAP_L1C_TB_E_00867_D_20150331T203555_R19240_001.h5")