"""
消融实验模块 - 使用特征移除方法
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']


class AblationStudy:
    """消融实验分析器 - 仅使用特征移除方法"""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print("=" * 60)
        print("消融实验分析器初始化（特征移除方法）")
        print("=" * 60)

    def _remove_features(self, conv_data, point_data, feature_type, indices_to_remove):
        """
        移除指定特征

        Args:
            feature_type: 'conv'或'point'
            indices_to_remove: 要移除的特征索引列表
        """
        conv_removed = conv_data.copy()
        point_removed = point_data.copy()

        if feature_type == 'conv':
            # 移除卷积特征通道
            if len(indices_to_remove) > 0:
                # 保留不在移除列表中的通道
                keep_indices = [i for i in range(conv_data.shape[1]) if i not in indices_to_remove]
                conv_removed = conv_removed[:, keep_indices, :, :]
        elif feature_type == 'point':
            # 移除点特征维度
            if len(indices_to_remove) > 0:
                keep_indices = [i for i in range(point_data.shape[1]) if i not in indices_to_remove]
                point_removed = point_removed[:, keep_indices]

        return conv_removed, point_removed

    def _create_model_with_removed_features(self, conv_indices_to_remove, point_indices_to_remove):
        """
        创建移除特征后的新模型

        注意：这里需要根据您实际的模型结构进行调整
        如果是动态创建模型，可以重新构建模型
        如果是固定模型，需要修改输入层
        """
        # 这里需要根据您的模型结构实现
        # 例如，如果模型可以接受不同的输入维度，可以直接使用
        # 如果需要重新构建模型，这里需要实现
        pass

    def compute_performance(self, conv_data, point_data, targets, batch_size=32):
        """计算模型性能"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(conv_data), batch_size):
                batch_end = min(i + batch_size, len(conv_data))

                conv_batch = torch.FloatTensor(conv_data[i:batch_end]).to(self.device)
                point_batch = torch.FloatTensor(point_data[i:batch_end]).to(self.device)

                preds = self.model(conv_batch, point_batch)
                predictions.append(preds.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        # 计算指标
        mse = np.mean((predictions.flatten() - targets.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions.flatten() - targets.flatten()))

        if np.std(targets) > 0:
            r2 = 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
        else:
            r2 = 0

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions
        }

    def run_ablation_study(self, dataloader, n_samples=500, output_dir=None):
        """
        运行消融实验 - 特征移除方法

        Args:
            dataloader: 数据加载器
            n_samples: 使用的样本数
            output_dir: 输出目录
        """
        print("\n" + "=" * 60)
        print(f"运行消融实验（特征移除方法）")
        print("=" * 60)

        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"./ablation_results/remove_{timestamp}")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. 准备数据
            print("\n1. 准备数据...")

            conv_list, point_list, target_list = [], [], []
            sample_count = 0

            for conv_feats, point_feats, targets in dataloader:
                conv_list.append(conv_feats.numpy())
                point_list.append(point_feats.numpy())
                target_list.append(targets.numpy())
                sample_count += conv_feats.shape[0]

                if sample_count >= n_samples:
                    break

            conv_data = np.concatenate(conv_list, axis=0)[:n_samples]
            point_data = np.concatenate(point_list, axis=0)[:n_samples]
            target_data = np.concatenate(target_list, axis=0)[:n_samples]

            print(f"  卷积特征: {conv_data.shape}")
            print(f"  点特征: {point_data.shape}")
            print(f"  目标值: {target_data.shape}")

            # 2. 定义要测试的特征组合
            print("\n2. 定义特征组合...")

            # 推断特征结构
            _, C_conv, H, W = conv_data.shape
            _, C_point = point_data.shape

            # 推断点特征组成
            fixed_features = 7  # S1_VV, S1_VH, SMAP_TBV, SMAP_TBH, lon, lat, doy
            ls_bands = C_point - fixed_features

            # 定义要测试的特征组合
            feature_combinations = [
                # 基准 - 使用所有特征
                {
                    'name': '基准模型（所有特征）',
                    'conv_remove': [],
                    'point_remove': [],
                    'description': '使用所有特征'
                },

                # 移除整组特征
                {
                    'name': '无卷积特征',
                    'conv_remove': list(range(C_conv)),
                    'point_remove': [],
                    'description': '移除所有卷积特征'
                },
                {
                    'name': '无点特征',
                    'conv_remove': [],
                    'point_remove': list(range(C_point)),
                    'description': '移除所有点特征'
                },
                {
                    'name': '无动态卷积特征',
                    'conv_remove': [0, 1, 2],  # 风场、温度、湿度
                    'point_remove': [],
                    'description': '移除动态卷积特征（风场、温度、湿度）'
                },
                {
                    'name': '无静态卷积特征',
                    'conv_remove': [3, 4, 5],  # 积雪日数、DEM均值、DEM标准差
                    'point_remove': [],
                    'description': '移除静态卷积特征（积雪日数、DEM）'
                },

                # 移除微波特征
                {
                    'name': '无微波特征',
                    'conv_remove': [],
                    'point_remove': list(range(ls_bands, ls_bands + 4)),  # 所有微波特征
                    'description': '移除所有微波特征（哨兵1 + SMAP）'
                },
                {
                    'name': '无哨兵1特征',
                    'conv_remove': [],
                    'point_remove': [ls_bands, ls_bands + 1],  # S1_VV, S1_VH
                    'description': '移除哨兵1后向散射特征'
                },
                {
                    'name': '无SMAP特征',
                    'conv_remove': [],
                    'point_remove': [ls_bands + 2, ls_bands + 3],  # SMAP_TBV, SMAP_TBH
                    'description': '移除SMAP亮温特征'
                },

                # 移除其他特征组
                {
                    'name': '无土地覆盖特征',
                    'conv_remove': [],
                    'point_remove': list(range(ls_bands)),  # 所有LS波段
                    'description': '移除土地覆盖特征'
                },
                {
                    'name': '无空间位置特征',
                    'conv_remove': [],
                    'point_remove': [ls_bands + 4, ls_bands + 5],  # lon, lat
                    'description': '移除空间位置特征'
                },
                {
                    'name': '无时间特征',
                    'conv_remove': [],
                    'point_remove': [ls_bands + 6],  # doy
                    'description': '移除时间特征'
                },
            ]

            print(f"  将测试 {len(feature_combinations)} 种特征组合")

            # 3. 运行消融实验
            print("\n3. 运行消融实验...")
            results = []

            for i, combo in enumerate(feature_combinations):
                print(f"  测试 [{i + 1}/{len(feature_combinations)}]: {combo['name']}")

                # 移除特征
                conv_removed, point_removed = self._remove_features(
                    conv_data, point_data,
                    'conv' if len(combo['conv_remove']) > 0 else 'point',
                    combo['conv_remove'] if len(combo['conv_remove']) > 0 else combo['point_remove']
                )

                # 计算性能
                perf = self.compute_performance(conv_removed, point_removed, target_data)

                # 如果是基准模型，保存基准性能
                if i == 0:
                    baseline_perf = perf
                    baseline_mse = perf['mse']

                # 计算性能变化
                mse_change = perf['mse'] - baseline_mse
                mse_change_percent = (mse_change / baseline_mse * 100) if baseline_mse > 0 else 0

                results.append({
                    '组合名称': combo['name'],
                    '描述': combo['description'],
                    '移除卷积特征数': len(combo['conv_remove']),
                    '移除点特征数': len(combo['point_remove']),
                    '移除总特征数': len(combo['conv_remove']) + len(combo['point_remove']),
                    'MSE': perf['mse'],
                    'RMSE': perf['rmse'],
                    'MAE': perf['mae'],
                    'R²': perf['r2'],
                    'MSE变化量': mse_change,
                    'MSE变化百分比': mse_change_percent,
                    '性能影响': '损害' if mse_change_percent > 0 else '改善',
                })

                print(f"    MSE: {perf['mse']:.6f} (变化: {mse_change:+.6f}, {mse_change_percent:+.1f}%)")

            # 4. 创建结果DataFrame
            print("\n4. 分析结果...")
            results_df = pd.DataFrame(results)

            # 按MSE变化百分比排序
            results_df = results_df.sort_values('MSE变化百分比', ascending=False)

            # 5. 保存结果
            print("\n5. 保存结果...")

            # 保存CSV
            csv_path = output_dir / "ablation_results.csv"
            results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  结果表: {csv_path}")

            # 6. 可视化
            print("\n6. 生成可视化图表...")
            self._generate_visualizations(results_df, output_dir, baseline_perf)

            # 7. 生成分析报告
            print("\n7. 生成分析报告...")
            self._generate_analysis_report(results_df, baseline_perf, output_dir)

            print("\n" + "=" * 60)
            print("消融实验完成!")
            print("=" * 60)

            return {
                'results_df': results_df,
                'baseline_perf': baseline_perf,
                'output_dir': output_dir
            }

        except Exception as e:
            print(f"\n✗ 消融实验失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_visualizations(self, results_df, output_dir, baseline_perf):
        """生成可视化图表"""
        try:
            # 排除基准行（第一个）
            plot_df = results_df.iloc[1:].copy() if len(results_df) > 1 else results_df.copy()

            if len(plot_df) == 0:
                print("  警告: 没有可可视化的数据")
                return

            # 图1: MSE变化百分比条形图
            plt.figure(figsize=(14, 10))

            # 按MSE变化排序
            plot_df = plot_df.sort_values('MSE变化百分比', ascending=True)

            # 为不同类型设置颜色
            colors = []
            for name in plot_df['组合名称']:
                if '无卷积' in name:
                    colors.append('steelblue')
                elif '无点' in name:
                    colors.append('forestgreen')
                elif '微波' in name or '哨兵' in name or 'SMAP' in name:
                    colors.append('darkorange')
                else:
                    colors.append('gray')

            bars = plt.barh(range(len(plot_df)), plot_df['MSE变化百分比'],
                            color=colors, alpha=0.8, edgecolor='black')

            # 添加数值标签
            for i, (bar, row) in enumerate(zip(bars, plot_df.iterrows())):
                _, row_data = row
                value = row_data['MSE变化百分比']
                color = 'darkred' if value > 0 else 'darkgreen'
                plt.text(value, i, f' {value:+.1f}%', va='center',
                         fontsize=9, fontweight='bold', color=color)

            plt.yticks(range(len(plot_df)), plot_df['组合名称'], fontsize=10)
            plt.xlabel('MSE变化百分比 (%)', fontsize=14, fontweight='bold')
            plt.title('特征移除对模型性能的影响\n正值表示性能下降（特征重要），负值表示性能改善（特征冗余）',
                      fontsize=16, fontweight='bold', pad=20)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            plot1_path = output_dir / "performance_impact.png"
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  性能影响图: {plot1_path}")

            # 图2: 特征数量 vs 性能影响散点图
            plt.figure(figsize=(12, 8))

            plt.scatter(plot_df['移除总特征数'], plot_df['MSE变化百分比'],
                        s=100, alpha=0.7, c=colors, edgecolor='black')

            # 添加标签
            for _, row in plot_df.iterrows():
                plt.annotate(row['组合名称'][:10],
                             (row['移除总特征数'], row['MSE变化百分比']),
                             fontsize=8, alpha=0.8)

            plt.xlabel('移除的特征数量', fontsize=12)
            plt.ylabel('MSE变化百分比 (%)', fontsize=12)
            plt.title('移除特征数量 vs 性能影响', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot2_path = output_dir / "feature_count_vs_performance.png"
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  特征数量vs性能图: {plot2_path}")

            # 图3: 性能指标对比雷达图
            if len(plot_df) > 3:
                self._create_radar_chart(plot_df, output_dir)

        except Exception as e:
            print(f"  可视化生成失败: {e}")

    def _create_radar_chart(self, plot_df, output_dir):
        """创建雷达图"""
        try:
            # 选择几个关键组合
            key_combinations = ['无卷积特征', '无点特征', '无微波特征', '无SMAP特征']
            selected_df = plot_df[plot_df['组合名称'].isin(key_combinations)]

            if len(selected_df) < 2:
                return

            # 创建雷达图
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))

            # 准备数据
            categories = ['MSE变化%', 'R²', '特征移除数']
            values = []

            for _, row in selected_df.iterrows():
                # 归一化数据
                mse_norm = abs(row['MSE变化百分比']) / 100  # 假设最大变化100%
                r2_norm = max(0, row['R²'])  # R²在0-1之间
                feat_norm = row['移除总特征数'] / 20  # 假设最多移除20个特征

                values.append([mse_norm, r2_norm, feat_norm])

            # 绘制
            for i, (name, vals) in enumerate(zip(selected_df['组合名称'], values)):
                ax.plot(categories, vals, label=name)
                ax.fill(categories, vals, alpha=0.1)

            ax.set_ylim(0, 1)
            ax.set_title('特征组合性能对比雷达图', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')

            plt.tight_layout()
            plot3_path = output_dir / "radar_chart.png"
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  雷达图: {plot3_path}")

        except Exception as e:
            print(f"  雷达图生成失败: {e}")

    def _generate_analysis_report(self, results_df, baseline_perf, output_dir):
        """生成分析报告"""
        try:
            report_path = output_dir / "analysis_report.txt"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SWE模型消融实验分析报告（特征移除方法）\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"基准MSE: {baseline_perf['mse']:.6f}\n")
                f.write(f"基准RMSE: {baseline_perf['rmse']:.6f}\n")
                f.write(f"基准MAE: {baseline_perf['mae']:.6f}\n")
                f.write(f"基准R²: {baseline_perf['r2']:.6f}\n\n")

                if len(results_df) > 1:
                    # 排除基准行
                    analysis_df = results_df.iloc[1:].copy()

                    f.write("核心发现:\n")
                    f.write("-" * 40 + "\n")

                    # 找出最重要的特征组（MSE变化最大的）
                    top_3 = analysis_df.nlargest(3, 'MSE变化百分比')

                    for i, (_, row) in enumerate(top_3.iterrows()):
                        effect = "损害" if row['MSE变化百分比'] > 0 else "改善"
                        f.write(f"{i + 1}. {row['组合名称']} 移除会使MSE{effect} {abs(row['MSE变化百分比']):.1f}%\n")

                    # SMAP特征分析
                    smap_features = analysis_df[analysis_df['组合名称'].str.contains('SMAP')]
                    if len(smap_features) > 0:
                        f.write("\nSMAP特征分析:\n")
                        f.write("-" * 40 + "\n")
                        for _, row in smap_features.iterrows():
                            effect = "损害" if row['MSE变化百分比'] > 0 else "改善"
                            importance = "非常重要" if abs(row['MSE变化百分比']) > 20 else "重要" if abs(
                                row['MSE变化百分比']) > 10 else "一般"
                            f.write(
                                f"{row['组合名称']}: 移除导致MSE{effect} {abs(row['MSE变化百分比']):.1f}% ({importance})\n")

                    f.write("\n详细结果:\n")
                    f.write("-" * 40 + "\n")
                    for _, row in analysis_df.iterrows():
                        direction = "增加" if row['MSE变化百分比'] > 0 else "减少"
                        f.write(f"{row['组合名称']:25s}: MSE{direction} {abs(row['MSE变化百分比']):.1f}% "
                                f"(MSE: {row['MSE']:.6f}, R²: {row['R²']:.3f})\n")

                f.write("\n实验结论:\n")
                f.write("-" * 40 + "\n")
                f.write("1. MSE变化百分比 > 0: 移除特征损害模型性能，说明特征重要\n")
                f.write("2. MSE变化百分比 < 0: 移除特征改善模型性能，说明特征冗余或有害\n")
                f.write("3. 绝对值越大，特征重要性越高\n")
                f.write("4. SMAP特征如果移除导致显著性能下降，说明其对SWE反演很重要\n")

            print(f"  分析报告: {report_path}")

        except Exception as e:
            print(f"  报告生成失败: {e}")


# 使用示例
def run_example():
    """使用示例"""
    # 假设已经有模型和数据
    model = None  # 您的模型
    dataloader = None  # 您的数据加载器

    # 创建消融实验分析器
    analyzer = AblationStudy(model)

    # 运行消融实验
    results = analyzer.run_ablation_study(
        dataloader=dataloader,
        n_samples=500,
        output_dir="./ablation_results"
    )

    return results


if __name__ == "__main__":
    print("消融实验模块（特征移除方法）")
    print("使用方法: from ablation_study import AblationStudy")