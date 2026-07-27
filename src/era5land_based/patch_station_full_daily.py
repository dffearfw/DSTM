#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path


NEW_TIME_BLOCK = r'''    def _build_complete_daily_timeline(self):
        """
        根据 self.load_years 构建完整日时间轴。

        2015—2018应得到1461天，不再使用多个动态变量的日期交集
        作为站点样本时间轴。
        """
        years = sorted({
            int(year)
            for year in self.load_years
        })

        if not years:
            raise RuntimeError(
                "load_years为空，无法构建完整日时间轴"
            )

        complete_dates = []

        for year in years:
            current_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)

            while current_date <= end_date:
                complete_dates.append(current_date)
                current_date += timedelta(days=1)

        if len(set(complete_dates)) != len(complete_dates):
            raise RuntimeError(
                "完整日时间轴内部存在重复日期"
            )

        return complete_dates


    def _align_to_timeline(
        self,
        var: str,
        var_data,
        var_dates,
        allow_temporal_fill: bool = False,
    ):
        """
        将单个动态变量对齐到完整日时间轴。

        规则
        ----
        1. LST、RH和PR必须完整覆盖目标时间轴，不允许静默补日期；
        2. 只有chelsa_sfxwind允许对缺失日期进行单变量填补；
        3. 内部缺口使用前后有效日期线性插值；
        4. 时间轴首尾缺口使用最近有效日期；
        5. 最近有效日期距离超过7天时直接报错；
        6. 此函数只改变当前变量，不改变站点样本日期，也不改变
           其他动态变量的日期。
        """
        var_data = np.asarray(
            var_data,
            dtype=np.float32,
        )

        normalized_dates = [
            pd.Timestamp(date_value)
            .normalize()
            .to_pydatetime()
            for date_value in var_dates
        ]

        target_dates = [
            pd.Timestamp(date_value)
            .normalize()
            .to_pydatetime()
            for date_value in self.all_dates
        ]

        if var_data.ndim != 3:
            raise RuntimeError(
                f"{var}必须是(T,H,W)三维数组，"
                f"当前shape={var_data.shape}"
            )

        if var_data.shape[0] != len(normalized_dates):
            raise RuntimeError(
                f"{var}时间维与日期数量不一致："
                f"data={var_data.shape[0]}, "
                f"dates={len(normalized_dates)}"
            )

        if len(set(normalized_dates)) != len(normalized_dates):
            duplicate_count = (
                len(normalized_dates)
                - len(set(normalized_dates))
            )
            raise RuntimeError(
                f"{var}日期标准化后存在"
                f"{duplicate_count}个重复日期"
            )

        date_to_source_index = {
            date_value: index
            for index, date_value
            in enumerate(normalized_dates)
        }

        target_date_set = set(target_dates)

        missing_dates = [
            date_value
            for date_value in target_dates
            if date_value not in date_to_source_index
        ]

        extra_dates = [
            date_value
            for date_value in normalized_dates
            if date_value not in target_date_set
        ]

        if missing_dates and not allow_temporal_fill:
            preview = ", ".join(
                date_value.strftime("%Y-%m-%d")
                for date_value in missing_dates[:10]
            )

            raise RuntimeError(
                f"{var}缺少{len(missing_dates)}个日尺度日期，"
                "该变量禁止静默时间填补。"
                f"前10个缺失日期：{preview}"
            )

        # 日期和顺序已经完全一致时，直接返回原数组，避免额外复制。
        if normalized_dates == target_dates:
            report = {
                "variable": var,
                "source_date_count": len(normalized_dates),
                "target_date_count": len(target_dates),
                "exact_count": len(target_dates),
                "filled_count": 0,
                "linear_count": 0,
                "nearest_count": 0,
                "missing_dates": [],
                "extra_dates": [
                    value.strftime("%Y-%m-%d")
                    for value in extra_dates
                ],
                "max_nearest_gap_days": 0,
                "fill_details": [],
            }

            print(
                f"  {var}: 完整日覆盖，"
                f"{len(target_dates)}天全部精确匹配"
            )

            return var_data, report

        source_dates_sorted = sorted(
            date_to_source_index
        )

        source_ordinals = np.asarray(
            [
                date_value.toordinal()
                for date_value in source_dates_sorted
            ],
            dtype=np.int64,
        )

        target_count = len(target_dates)
        _, height, width = var_data.shape

        aligned_data = np.empty(
            (target_count, height, width),
            dtype=np.float32,
        )

        exact_count = 0
        fill_details = []

        for target_index, target_date in enumerate(
            target_dates
        ):
            source_index = date_to_source_index.get(
                target_date
            )

            if source_index is not None:
                aligned_data[target_index] = (
                    var_data[source_index]
                )
                exact_count += 1
                continue

            insert_position = int(
                np.searchsorted(
                    source_ordinals,
                    target_date.toordinal(),
                    side="left",
                )
            )

            left_date = (
                source_dates_sorted[
                    insert_position - 1
                ]
                if insert_position > 0
                else None
            )

            right_date = (
                source_dates_sorted[
                    insert_position
                ]
                if insert_position
                < len(source_dates_sorted)
                else None
            )

            if left_date is None and right_date is None:
                raise RuntimeError(
                    f"{var}没有任何可用于时间填补的数据"
                )

            left_gap = (
                abs((target_date - left_date).days)
                if left_date is not None
                else None
            )

            right_gap = (
                abs((right_date - target_date).days)
                if right_date is not None
                else None
            )

            available_gaps = [
                gap
                for gap in [left_gap, right_gap]
                if gap is not None
            ]

            nearest_gap = min(available_gaps)

            if (
                nearest_gap
                > DYNAMIC_TEMPORAL_FILL_MAX_GAP_DAYS
            ):
                raise RuntimeError(
                    f"{var}在{target_date:%Y-%m-%d}"
                    "附近缺口过长："
                    f"最近有效日期相距{nearest_gap}天，"
                    "超过允许的"
                    f"{DYNAMIC_TEMPORAL_FILL_MAX_GAP_DAYS}天"
                )

            if (
                left_date is not None
                and right_date is not None
            ):
                left_layer = var_data[
                    date_to_source_index[left_date]
                ]

                right_layer = var_data[
                    date_to_source_index[right_date]
                ]

                total_gap = (
                    right_date - left_date
                ).days

                if total_gap <= 0:
                    raise RuntimeError(
                        f"{var}插值日期顺序异常："
                        f"{left_date} -> {right_date}"
                    )

                weight = (
                    (target_date - left_date).days
                    / total_gap
                )

                left_valid = (
                    np.isfinite(left_layer)
                    & (left_layer != FINAL_NODATA)
                )

                right_valid = (
                    np.isfinite(right_layer)
                    & (right_layer != FINAL_NODATA)
                )

                both_valid = (
                    left_valid & right_valid
                )

                only_left = (
                    left_valid & ~right_valid
                )

                only_right = (
                    right_valid & ~left_valid
                )

                filled_layer = np.full(
                    left_layer.shape,
                    np.nan,
                    dtype=np.float32,
                )

                filled_layer[both_valid] = (
                    left_layer[both_valid]
                    + (
                        right_layer[both_valid]
                        - left_layer[both_valid]
                    )
                    * weight
                )

                filled_layer[only_left] = (
                    left_layer[only_left]
                )

                filled_layer[only_right] = (
                    right_layer[only_right]
                )

                aligned_data[target_index] = (
                    filled_layer
                )

                method = "linear"
                source_dates_text = (
                    f"{left_date:%Y-%m-%d}|"
                    f"{right_date:%Y-%m-%d}"
                )

            else:
                nearest_date = (
                    left_date
                    if left_date is not None
                    else right_date
                )

                aligned_data[target_index] = (
                    var_data[
                        date_to_source_index[
                            nearest_date
                        ]
                    ]
                )

                method = "nearest_edge"
                source_dates_text = (
                    nearest_date.strftime(
                        "%Y-%m-%d"
                    )
                )

            fill_details.append({
                "target_date":
                    target_date.strftime(
                        "%Y-%m-%d"
                    ),
                "method":
                    method,
                "source_dates":
                    source_dates_text,
                "nearest_gap_days":
                    int(nearest_gap),
            })

        linear_count = sum(
            detail["method"] == "linear"
            for detail in fill_details
        )

        nearest_count = sum(
            detail["method"] == "nearest_edge"
            for detail in fill_details
        )

        max_nearest_gap = max(
            (
                detail["nearest_gap_days"]
                for detail in fill_details
            ),
            default=0,
        )

        report = {
            "variable": var,
            "source_date_count":
                len(normalized_dates),
            "target_date_count":
                len(target_dates),
            "exact_count":
                int(exact_count),
            "filled_count":
                int(len(fill_details)),
            "linear_count":
                int(linear_count),
            "nearest_count":
                int(nearest_count),
            "missing_dates": [
                value.strftime("%Y-%m-%d")
                for value in missing_dates
            ],
            "extra_dates": [
                value.strftime("%Y-%m-%d")
                for value in extra_dates
            ],
            "max_nearest_gap_days":
                int(max_nearest_gap),
            "fill_details":
                fill_details,
        }

        print(
            f"  {var}: source={len(normalized_dates)}, "
            f"target={len(target_dates)}, "
            f"exact={exact_count}, "
            f"filled={len(fill_details)}, "
            f"linear={linear_count}, "
            f"nearest_edge={nearest_count}, "
            f"max_gap={max_nearest_gap}天"
        )

        for detail in fill_details[:10]:
            print(
                "     填补 "
                f"{detail['target_date']} <- "
                f"{detail['source_dates']} "
                f"({detail['method']}, "
                f"最近间隔"
                f"{detail['nearest_gap_days']}天)"
            )

        if len(fill_details) > 10:
            print(
                f"     其余"
                f"{len(fill_details) - 10}"
                "个填补日期省略"
            )

        return aligned_data, report


    def _validate_complete_daily_feature_timeline(
        self,
    ):
        """
        检查所有动态变量是否严格采用完整日时间轴。
        """
        expected_dates = (
            self._build_complete_daily_timeline()
        )

        actual_dates = [
            pd.Timestamp(date_value)
            .normalize()
            .to_pydatetime()
            for date_value in self.all_dates
        ]

        if actual_dates != expected_dates:
            raise RuntimeError(
                "动态特征时间轴不是完整日时间轴："
                f"expected={len(expected_dates)}, "
                f"actual={len(actual_dates)}, "
                f"expected_range="
                f"{expected_dates[0]:%Y-%m-%d}—"
                f"{expected_dates[-1]:%Y-%m-%d}, "
                f"actual_range="
                f"{actual_dates[0]:%Y-%m-%d}—"
                f"{actual_dates[-1]:%Y-%m-%d}"
            )

        self.all_dates = actual_dates
        self.date_to_index = {
            date_value: index
            for index, date_value
            in enumerate(actual_dates)
        }

        for var in CONV_VARS:
            if var not in self.conv_dyn_data:
                raise RuntimeError(
                    f"动态特征缺少变量：{var}"
                )

            array = self.conv_dyn_data[var]

            if array.shape[0] != len(expected_dates):
                raise RuntimeError(
                    f"{var}时间维错误："
                    f"{array.shape[0]} "
                    f"!= {len(expected_dates)}"
                )

            report = getattr(
                self,
                "dynamic_time_fill_report",
                {},
            ).get(var, {})

            filled_count = int(
                report.get("filled_count", 0)
            )

            if (
                var != "chelsa_sfxwind"
                and filled_count != 0
            ):
                raise RuntimeError(
                    f"{var}不应进行时间填补，"
                    f"但报告filled_count="
                    f"{filled_count}"
                )

        expected_count = len(expected_dates)

        print(
            "\n✅ 完整日时间轴检查通过："
            f"{expected_count}天，"
            f"{expected_dates[0]:%Y-%m-%d}—"
            f"{expected_dates[-1]:%Y-%m-%d}"
        )

        for var in CONV_VARS:
            report = getattr(
                self,
                "dynamic_time_fill_report",
                {},
            ).get(var, {})

            print(
                f"   {var}: "
                f"shape="
                f"{self.conv_dyn_data[var].shape}, "
                f"filled="
                f"{report.get('filled_count', 0)}"
            )


    def _load_all_features(self):
        """
        加载完整日尺度动态特征。

        不再取四变量日期交集。站点日期始终保留原始观测日期；
        只有缺日的chelsa_sfxwind通道单独做时间填补。
        """
        print("\n加载特征数据...")

        expected_years = [
            2015,
            2016,
            2017,
            2018,
        ]

        actual_years = sorted({
            int(year)
            for year in self.load_years
        })

        if actual_years != expected_years:
            raise RuntimeError(
                "渐进式微调年份必须与预训练一致："
                f"expected={expected_years}, "
                f"actual={actual_years}"
            )

        self.all_dates = (
            self._build_complete_daily_timeline()
        )

        self.date_to_index = {
            date_value: index
            for index, date_value
            in enumerate(self.all_dates)
        }

        print(
            "\n📅 站点微调采用完整日时间轴："
        )

        print(
            f"   日期范围："
            f"{self.all_dates[0]:%Y-%m-%d}—"
            f"{self.all_dates[-1]:%Y-%m-%d}"
        )

        print(
            f"   总天数：{len(self.all_dates)}"
        )

        self.conv_dyn_data = {}
        self.dynamic_time_fill_report = {}

        for var in CONV_VARS:
            print(f"\n加载 {var} 数据...")

            var_data, var_dates = (
                self._load_single_variable(var)
            )

            if (
                var_data is None
                or not var_dates
            ):
                raise RuntimeError(
                    f"必需动态特征{var}加载失败"
                )

            if (
                int(var_data.shape[0])
                != len(var_dates)
            ):
                raise RuntimeError(
                    f"{var}数据长度与日期长度"
                    "不一致："
                    f"data={var_data.shape[0]}, "
                    f"dates={len(var_dates)}"
                )

            normalized_var_dates = [
                pd.Timestamp(date_value)
                .normalize()
                .to_pydatetime()
                for date_value in var_dates
            ]

            if (
                len(set(normalized_var_dates))
                != len(normalized_var_dates)
            ):
                duplicate_count = (
                    len(normalized_var_dates)
                    - len(set(normalized_var_dates))
                )

                raise RuntimeError(
                    f"{var}存在"
                    f"{duplicate_count}个重复日期"
                )

            print(
                f"  {var}原始时间轴："
                f"{len(normalized_var_dates)}天，"
                f"{normalized_var_dates[0]:%Y-%m-%d}—"
                f"{normalized_var_dates[-1]:%Y-%m-%d}"
            )

            aligned_data, fill_report = (
                self._align_to_timeline(
                    var=var,
                    var_data=var_data,
                    var_dates=normalized_var_dates,
                    allow_temporal_fill=(
                        var == "chelsa_sfxwind"
                    ),
                )
            )

            self.conv_dyn_data[var] = (
                aligned_data
            )

            self.dynamic_time_fill_report[var] = (
                fill_report
            )

            del var_data

        self._validate_complete_daily_feature_timeline()

        self._load_static_conv_features()
        self._load_point_features()


'''


def replace_once(
    text: str,
    old: str,
    new: str,
    label: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}替换失败："
            f"预期出现1次，实际出现{count}次"
        )

    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file",
        help="当前实际运行的data_station_online_swe.py",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "输出文件；默认在原文件旁生成"
            "data_station_online_swe_fixed.py"
        ),
    )

    args = parser.parse_args()

    source_path = Path(
        args.input_file
    ).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(
            source_path
        )

    if args.output:
        output_path = Path(
            args.output
        ).expanduser().resolve()
    else:
        output_path = source_path.with_name(
            "data_station_online_swe_fixed.py"
        )

    text = source_path.read_text(
        encoding="utf-8"
    )

    original_text = text
    # 1. timedelta：只修改首次出现的顶层导入
    old_datetime_import = "from datetime import datetime"

    if old_datetime_import not in text:
        raise RuntimeError(
            "找不到from datetime import datetime"
        )

    text = text.replace(
        old_datetime_import,
        "from datetime import datetime, timedelta",
        1,
    )



    # 2. 缓存版本和填补上限
    text = replace_once(
        text,
        (
            'STATION_FEATURE_LOADER_VERSION = '
            '"pretrain_time_intersection_2015_2018_v3"'
        ),
        (
            'STATION_FEATURE_LOADER_VERSION = '
            '"full_daily_wind_only_fill_2015_2018_v4"\n'
            "DYNAMIC_TEMPORAL_FILL_MAX_GAP_DAYS = 7"
        ),
        "站点特征缓存版本",
    )

    # 3. 替换时间轴构建逻辑
    function_pattern = re.compile(
        r"(?ms)"
        r"^    def _load_all_features\(self\):\n"
        r".*?"
        r"(?=^    def _load_single_variable"
        r"\(self, var: str\):)"
    )

    text, replacement_count = (
        function_pattern.subn(
            lambda match: NEW_TIME_BLOCK,
            text,
            count=1,
        )
    )

    if replacement_count != 1:
        raise RuntimeError(
            "_load_all_features时间轴代码块"
            "替换失败"
        )

    # 4. 缓存中保存时间填补报告
    text = replace_once(
        text,
        (
            "optional_attrs = "
            "['ls_data', 'ls_data_default', "
            "'s1_data', 'smap_data', "
            "'all_s1_dates', 'all_smap_dates']"
        ),
        (
            "optional_attrs = "
            "['ls_data', 'ls_data_default', "
            "'s1_data', 'smap_data', "
            "'all_s1_dates', 'all_smap_dates', "
            "'dynamic_time_fill_report']"
        ),
        "缓存可选属性",
    )

    # 5. 缓存恢复后也执行1461天检查
    text = replace_once(
        text,
        (
            '        if not hasattr('
            'self, "ls_data_default"):'
        ),
        (
            "        "
            "self._validate_complete_daily_feature_timeline()\n\n"
            '        if not hasattr('
            'self, "ls_data_default"):'
        ),
        "缓存恢复时间轴检查",
    )

    # 6. 站点日期统一到当天00:00:00
    text = replace_once(
        text,
        (
            "df['date'] = "
            "pd.to_datetime("
            "df['date'], errors='coerce')"
        ),
        (
            "df['date'] = "
            "pd.to_datetime("
            "df['date'], errors='coerce')"
            ".dt.normalize()"
        ),
        "站点日期标准化",
    )

    # 7. 禁止站点日期整体映射到最近共同日期
    old_date_mapping = '''            if dt not in self.date_to_index:
                closest_date = min(self.all_dates, key=lambda d: abs((d - dt).days))
                day_gap = abs((closest_date - dt).days)
                if day_gap > 7:
                    invalid_date_count += 1
                    continue
                feature_date = closest_date
            else:
                feature_date = dt
'''

    new_date_mapping = '''            dt = (
                pd.Timestamp(dt)
                .normalize()
                .to_pydatetime()
            )

            if dt not in self.date_to_index:
                invalid_date_count += 1
                raise RuntimeError(
                    "站点原始观测日期不在完整日时间轴中："
                    f"station={stations}, "
                    f"date={dt:%Y-%m-%d}, "
                    f"row={r}, col={c}"
                )

            # 关键约束：模型特征日期始终等于原始站点观测日期。
            feature_date = dt
'''

    text = replace_once(
        text,
        old_date_mapping,
        new_date_mapping,
        "站点feature_date映射",
    )

    # 8. 元数据明确记录零日期偏移
    old_meta = '''                'feature_date': feature_date,      # 用于提取特征
                'label_date': dt,                  # 原始站点日期（仅用于记录）
                'day_gap': abs((feature_date - dt).days),
'''

    new_meta = '''                'feature_date': feature_date,      # 与原始站点日期严格一致
                'label_date': dt,                  # 原始站点观测日期
                'day_gap': 0,                      # 禁止整样本日期折叠
'''

    text = replace_once(
        text,
        old_meta,
        new_meta,
        "站点meta日期字段",
    )

    # 9. 建完meta_index后强制检查不存在日期折叠
    audit_code = '''        collapsed_date_samples = [
            meta
            for meta in self.meta_index
            if (
                pd.Timestamp(meta["feature_date"]).normalize()
                != pd.Timestamp(meta["label_date"]).normalize()
                or int(meta.get("day_gap", 0)) != 0
            )
        ]

        if collapsed_date_samples:
            example = collapsed_date_samples[0]
            raise RuntimeError(
                "仍检测到站点日期折叠："
                f"count={len(collapsed_date_samples)}, "
                f"example={example}"
            )

        print(
            "  ✅ 站点日期完整性检查通过："
            f"{len(self.meta_index)}个样本全部满足"
            "feature_date == label_date，day_gap=0"
        )

'''

    text = replace_once(
        text,
        '        print(f"\\n数据统计:")',
        (
            audit_code
            + '        print(f"\\n数据统计:")'
        ),
        "站点日期完整性审计",
    )

    if text == original_text:
        raise RuntimeError(
            "文件内容没有发生变化"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_path = source_path.with_suffix(
        source_path.suffix
        + ".before_full_daily_fix.bak"
    )

    if not backup_path.exists():
        shutil.copy2(
            source_path,
            backup_path,
        )

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    py_compile.compile(
        str(output_path),
        doraise=True,
    )

    print("=" * 90)
    print("补丁完成")
    print("=" * 90)
    print(f"原文件：{source_path}")
    print(f"备份：  {backup_path}")
    print(f"新文件：{output_path}")
    print("语法检查：通过")
    print()
    print("关键修改：")
    print(
        "1. 时间轴改为2015-01-01—"
        "2018-12-31完整1461天"
    )
    print(
        "2. 只有chelsa_sfxwind缺日时"
        "进行单变量时间填补"
    )
    print(
        "3. LST、RH、PR缺日直接报错"
    )
    print(
        "4. feature_date严格等于label_date"
    )
    print(
        "5. 新缓存版本避免读取旧1408天缓存"
    )


if __name__ == "__main__":
    main()
