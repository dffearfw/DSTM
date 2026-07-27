#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path("/root/autodl-tmp")
STATION_FILE = ROOT / "data_station_online_swe.py"
MAIN_FILE = ROOT / "main_tune.py"

STATION_MARKER = "PROGRESSIVE_FINETUNE_NORMALIZATION_V1"
MAIN_MARKER = "PROGRESSIVE_FINETUNE_PASS_NORMALIZATION_V1"


def make_backup(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{timestamp}")
    shutil.copy2(path, backup)
    return backup


def replace_once(
    text: str,
    old: str,
    new: str,
    description: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{description}定位失败：期望出现1次，实际出现{count}次"
        )

    return text.replace(old, new, 1)


def patch_station_dataset(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")

    if STATION_MARKER in text:
        print(f"✅ {path.name} 已包含统一归一化补丁")
        return False

    # ------------------------------------------------------------------
    # 1. 保存统一归一化参数
    # ------------------------------------------------------------------
    init_old = """        self.split_cache_file = split_cache_file
        self.force_recompute_split = force_recompute_split

        # ============ 数据存储初始化 ============
"""

    init_new = """        self.split_cache_file = split_cache_file
        self.force_recompute_split = force_recompute_split

        # PROGRESSIVE_FINETUNE_NORMALIZATION_V1
        # M0-M4微调、内部测试和外部测试统一复用预训练归一化。
        normalization_config_path = kwargs.get(
            "normalization_config_path"
        )
        self.normalization_config_path = (
            Path(normalization_config_path).expanduser()
            if normalization_config_path
            else None
        )

        self.normalization_mode = str(
            kwargs.get("normalization_mode", "auto")
        ).strip().lower()

        if self.normalization_mode not in {
            "auto",
            "load",
            "create",
            "skip",
            "legacy",
        }:
            raise ValueError(
                "StationSWEDataset normalization_mode非法: "
                f"{self.normalization_mode!r}"
            )

        self.fixed_label_min_mm = float(
            kwargs.get("fixed_label_min_mm", 0.0)
        )
        self.fixed_label_max_mm = float(
            kwargs.get("fixed_label_max_mm", 400.0)
        )

        if self.fixed_label_max_mm <= self.fixed_label_min_mm:
            raise ValueError(
                "fixed_label_max_mm必须大于fixed_label_min_mm"
            )

        # ============ 数据存储初始化 ============
"""

    text = replace_once(
        text,
        init_old,
        init_new,
        "StationSWEDataset初始化参数",
    )

    # ------------------------------------------------------------------
    # 2. 禁止正式微调根据当前站点子集重新计算min-max
    # ------------------------------------------------------------------
    compute_old = """        # 7. 计算归一化参数（每次都重新计算）
        self._compute_minmax()

        # 8. 检查SWE差异
"""

    compute_new = """        # 7. 归一化
        # 正式渐进式流程必须加载Stage0-4共用的clip+z-score配置。
        loaded_progressive_norm = (
            self._load_progressive_normalization()
        )

        if not loaded_progressive_norm:
            if self.normalization_mode == "load":
                raise RuntimeError(
                    "normalization_mode=load，"
                    "但统一归一化配置没有成功加载"
                )

            print(
                "   ⚠ 未加载渐进式统一归一化，"
                "使用旧版站点子集min-max"
            )
            self._compute_minmax()
            self.label_min = float(self.swe_min)
            self.label_max = float(self.swe_max)
            self.normalization_method = "minmax"

        # 8. 检查SWE差异
"""

    text = replace_once(
        text,
        compute_old,
        compute_new,
        "归一化初始化调用",
    )

    # ------------------------------------------------------------------
    # 3. 添加统一配置读取函数
    # ------------------------------------------------------------------
    method_anchor = """    def _save_feature_cache(self, cache_path: Path):
"""

    normalization_method = r'''    def _load_progressive_normalization(self) -> bool:
        """加载M0-M4共用的clip+p01/p99+z-score归一化配置。"""
        path = self.normalization_config_path

        should_load = (
            self.normalization_mode == "load"
            or (
                self.normalization_mode == "auto"
                and path is not None
                and path.exists()
            )
        )

        if not should_load:
            return False

        if path is None:
            raise ValueError(
                "normalization_mode=load时必须提供"
                "normalization_config_path"
            )

        if not path.exists():
            raise FileNotFoundError(
                f"统一归一化配置不存在: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        method = str(
            payload.get("method", "")
        ).strip().lower()

        if method != "clip_then_zscore":
            raise ValueError(
                "渐进式微调要求method=clip_then_zscore，"
                f"当前={method!r}"
            )

        expected_dimensions = {
            "C_conv": int(self.C_conv),
            "C_point": int(self.C_point),
            "patch_size": int(self.patch_size),
        }

        for key, expected in expected_dimensions.items():
            actual = int(payload.get(key, -1))

            if actual != expected:
                raise ValueError(
                    f"统一归一化{key}不一致: "
                    f"config={actual}, dataset={expected}"
                )

        array_keys = [
            "conv_clip_low",
            "conv_clip_high",
            "conv_mean",
            "conv_std",
            "point_clip_low",
            "point_clip_high",
            "point_mean",
            "point_std",
        ]

        missing = [
            key
            for key in array_keys
            if key not in payload
        ]

        if missing:
            raise ValueError(
                f"统一归一化配置缺少字段: {missing}"
            )

        for key in array_keys:
            setattr(
                self,
                key,
                np.asarray(
                    payload[key],
                    dtype=np.float32,
                ),
            )

        self.point_transform = list(
            payload.get(
                "point_transform",
                ["zscore"] * self.C_point,
            )
        )

        if len(self.point_transform) != self.C_point:
            raise ValueError(
                "point_transform长度与C_point不一致"
            )

        self.label_min = float(payload["label_min"])
        self.label_max = float(payload["label_max"])

        if (
            abs(
                self.label_min
                - self.fixed_label_min_mm
            )
            > 1e-6
        ):
            raise ValueError(
                "统一标签下限不一致: "
                f"config={self.label_min}, "
                f"requested={self.fixed_label_min_mm}"
            )

        if (
            abs(
                self.label_max
                - self.fixed_label_max_mm
            )
            > 1e-6
        ):
            raise ValueError(
                "统一标签上限不一致: "
                f"config={self.label_max}, "
                f"requested={self.fixed_label_max_mm}"
            )

        self.swe_min = self.label_min
        self.swe_max = self.label_max
        self.normalization_method = "clip_then_zscore"

        print(
            f"   ✅ 已加载渐进式统一归一化: {path}"
        )
        print(
            "      method: clip_then_zscore"
        )
        print(
            f"      C_conv={self.C_conv}, "
            f"C_point={self.C_point}, "
            f"patch={self.patch_size}"
        )
        print(
            f"      SWE范围: "
            f"[{self.label_min}, {self.label_max}] mm"
        )

        return True

'''

    text = replace_once(
        text,
        method_anchor,
        normalization_method + method_anchor,
        "统一归一化函数插入点",
    )

    # ------------------------------------------------------------------
    # 4. 替换__getitem__中的旧min-max标准化
    # ------------------------------------------------------------------
    start_marker = """            # ============================================================
            # 8. 标准化
            # ============================================================
"""

    end_marker = (
        "            return conv_t, point_t, y_t, "
        "is_zero_t, grid_val_norm_t, int(cur_idx)"
    )

    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index)

    if start_index < 0 or end_index < 0:
        raise RuntimeError(
            "无法定位StationSWEDataset标准化代码块"
        )

    new_standardization = r'''            # ============================================================
            # 8. 标准化：与M0-M4预训练严格一致
            # ============================================================
            eps = 1e-6

            try:
                if (
                    getattr(
                        self,
                        "normalization_method",
                        "minmax",
                    )
                    == "clip_then_zscore"
                ):
                    conv_low = torch.from_numpy(
                        self.conv_clip_low
                    ).float().view(-1, 1, 1)

                    conv_high = torch.from_numpy(
                        self.conv_clip_high
                    ).float().view(-1, 1, 1)

                    conv_mean = torch.from_numpy(
                        self.conv_mean
                    ).float().view(-1, 1, 1)

                    conv_std = torch.from_numpy(
                        self.conv_std
                    ).float().view(-1, 1, 1)

                    conv_t = torch.clamp(
                        conv_t,
                        min=conv_low,
                        max=conv_high,
                    )

                    conv_t = (
                        conv_t - conv_mean
                    ) / torch.clamp(
                        conv_std,
                        min=eps,
                    )

                    raw_point = point_t.clone()

                    # 与预训练build_progressive_normalization.py
                    # 中的point_valid_mask保持一致。
                    valid_point = torch.ones(
                        self.C_point,
                        dtype=torch.bool,
                    )

                    if self.C_point >= 15:
                        # Landsat六波段：0表示缺失填充值
                        valid_point[0:6] = (
                            raw_point[0:6] != 0.0
                        )

                        # S1 VV/VH的有效性由coverage判断
                        valid_point[6] = (
                            raw_point[8] > 0.0
                        )
                        valid_point[7] = (
                            raw_point[9] > 0.0
                        )

                        # S1 angle
                        valid_point[10] = (
                            (raw_point[8] > 0.0)
                            | (raw_point[9] > 0.0)
                        )

                        # SMAP TBV/TBH的有效性由mask判断
                        valid_point[11] = (
                            raw_point[13] > 0.0
                        )
                        valid_point[12] = (
                            raw_point[14] > 0.0
                        )

                    point_low = torch.from_numpy(
                        self.point_clip_low
                    ).float()

                    point_high = torch.from_numpy(
                        self.point_clip_high
                    ).float()

                    point_mean = torch.from_numpy(
                        self.point_mean
                    ).float()

                    point_std = torch.from_numpy(
                        self.point_std
                    ).float()

                    for feature_index, transform_name in enumerate(
                        self.point_transform
                    ):
                        if transform_name == "zscore":
                            if valid_point[feature_index]:
                                value = torch.clamp(
                                    raw_point[feature_index],
                                    point_low[feature_index],
                                    point_high[feature_index],
                                )

                                point_t[feature_index] = (
                                    value
                                    - point_mean[feature_index]
                                ) / torch.clamp(
                                    point_std[feature_index],
                                    min=eps,
                                )
                            else:
                                # 缺失连续变量保持0，
                                # 不把缺失填充值变成极端z-score。
                                point_t[feature_index] = 0.0
                        else:
                            # coverage、mask、坐标、DOY保持原语义
                            point_t[feature_index] = (
                                raw_point[feature_index]
                            )

                    y_t = (
                        y_t - self.label_min
                    ) / (
                        self.label_max
                        - self.label_min
                    )

                    grid_val_norm_t = (
                        grid_val_t - self.label_min
                    ) / (
                        self.label_max
                        - self.label_min
                    )

                    grid_val_norm_t = torch.clamp(
                        grid_val_norm_t,
                        0.0,
                        1.0,
                    )

                else:
                    c_min = torch.from_numpy(
                        self.conv_min
                    ).float().view(-1, 1, 1)

                    c_max = torch.from_numpy(
                        self.conv_max
                    ).float().view(-1, 1, 1)

                    p_min = torch.from_numpy(
                        self.point_min
                    ).float()

                    p_max = torch.from_numpy(
                        self.point_max
                    ).float()

                    conv_t = (
                        conv_t - c_min
                    ) / (
                        c_max - c_min + eps
                    )
                    conv_t = torch.clamp(
                        conv_t,
                        0.0,
                        1.0,
                    )

                    point_t = (
                        point_t - p_min
                    ) / (
                        p_max - p_min + eps
                    )
                    point_t = torch.clamp(
                        point_t,
                        0.0,
                        1.0,
                    )

                    y_t = (
                        y_t - self.swe_min
                    ) / (
                        self.swe_max
                        - self.swe_min
                        + eps
                    )
                    y_t = torch.clamp(
                        y_t,
                        0.0,
                        1.0,
                    )

                    grid_val_norm_t = (
                        grid_val_t - self.swe_min
                    ) / (
                        self.swe_max
                        - self.swe_min
                        + eps
                    )
                    grid_val_norm_t = torch.clamp(
                        grid_val_norm_t,
                        0.0,
                        1.0,
                    )

            except Exception as exc:
                self._debug_stats["norm_failed"] += 1

                if not self._debug_logged:
                    print(
                        f"  [DEBUG] 标准化失败: {exc}"
                    )

                cur_idx = (
                    cur_idx + 1
                ) % len(self.meta_index)
                continue

'''

    text = (
        text[:start_index]
        + new_standardization
        + text[end_index:]
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    return True


def patch_main(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")

    if MAIN_MARKER in text:
        print(f"✅ {path.name} 已传递统一归一化参数")
        return False

    old = """                "use_product_correction": self.config.get("use_product_correction", False),
            }
"""

    new = """                "use_product_correction": self.config.get("use_product_correction", False),

                # PROGRESSIVE_FINETUNE_PASS_NORMALIZATION_V1
                "normalization_config_path": self.config.get(
                    "normalization_config_path"
                ),
                "normalization_mode": self.config.get(
                    "normalization_mode",
                    "auto",
                ),
                "fixed_label_min_mm": self.config.get(
                    "fixed_label_min_mm",
                    0.0,
                ),
                "fixed_label_max_mm": self.config.get(
                    "fixed_label_max_mm",
                    400.0,
                ),
            }
"""

    text = replace_once(
        text,
        old,
        new,
        "main_tune站点数据集参数",
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    return True


def verify_normalization_json() -> None:
    path = (
        ROOT
        / "shared_cache"
        / "progressive_pretrain_normalization.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"统一归一化JSON不存在: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    required = [
        "method",
        "C_conv",
        "C_point",
        "patch_size",
        "conv_clip_low",
        "conv_clip_high",
        "conv_mean",
        "conv_std",
        "point_clip_low",
        "point_clip_high",
        "point_mean",
        "point_std",
        "point_transform",
        "label_min",
        "label_max",
    ]

    missing = [
        key
        for key in required
        if key not in payload
    ]

    if missing:
        raise ValueError(
            f"统一归一化JSON缺少字段: {missing}"
        )

    expected = {
        "method": "clip_then_zscore",
        "C_conv": 21,
        "C_point": 18,
        "patch_size": 5,
        "label_min": 0.0,
        "label_max": 400.0,
    }

    for key, expected_value in expected.items():
        actual = payload[key]

        if actual != expected_value:
            raise ValueError(
                f"归一化JSON检查失败: "
                f"{key}={actual!r}, "
                f"期望={expected_value!r}"
            )

    print("✅ 统一归一化JSON检查通过")
    print(
        "   method=clip_then_zscore, "
        "C_conv=21, C_point=18, "
        "patch=5, SWE=[0,400]"
    )


def main() -> None:
    for path in [
        STATION_FILE,
        MAIN_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    verify_normalization_json()

    station_backup = make_backup(
        STATION_FILE
    )
    main_backup = make_backup(
        MAIN_FILE
    )

    print(f"💾 Station备份: {station_backup}")
    print(f"💾 Main备份:    {main_backup}")

    station_changed = patch_station_dataset(
        STATION_FILE
    )
    main_changed = patch_main(
        MAIN_FILE
    )

    py_compile.compile(
        str(STATION_FILE),
        doraise=True,
    )
    py_compile.compile(
        str(MAIN_FILE),
        doraise=True,
    )

    station_text = STATION_FILE.read_text(
        encoding="utf-8"
    )
    main_text = MAIN_FILE.read_text(
        encoding="utf-8"
    )

    if STATION_MARKER not in station_text:
        raise RuntimeError(
            "Station统一归一化标记不存在"
        )

    if MAIN_MARKER not in main_text:
        raise RuntimeError(
            "main_tune参数传递标记不存在"
        )

    print("=" * 78)
    print("✅ 渐进式微调统一归一化补丁完成")
    print(f"   Station changed: {station_changed}")
    print(f"   Main changed:    {main_changed}")
    print("   Python语法检查: 通过")
    print("=" * 78)


if __name__ == "__main__":
    main()
