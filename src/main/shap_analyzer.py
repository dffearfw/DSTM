"""
SHAP特征重要性分析模块
硬编码：直接生成SHAP分析图
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
import torch
warnings.filterwarnings('ignore')
shap.initjs()  # 初始化SHAP的JS支持


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class SHAPAnalyzer:
    """SHAP分析器 - 硬编码版"""

    def __init__(self, model, feature_names=None):
        """
        初始化SHAP分析器

        Args:
            model: 训练好的PyTorch模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

        print("=" * 60)
        print("SHAP分析器初始化")
        print("=" * 60)

    def prepare_features_from_dataloader(self, dataloader, num_samples=1000):
        """
        从数据加载器中准备特征数据

        Args:
            dataloader: PyTorch DataLoader
            num_samples: 采样的样本数量

        Returns:
            conv_features_array: 卷积特征数组 (N, C, H, W)
            point_features_array: 点特征数组 (N, C_point)
            combined_features_array: 展平后的特征数组 (N, total_features)
        """
        print(f"从数据加载器准备特征数据...")

        conv_features_list = []
        point_features_list = []

        # 收集数据
        batch_count = 0
        for conv_feats, point_feats, _ in dataloader:
            conv_features_list.append(conv_feats.numpy())
            point_features_list.append(point_feats.numpy())
            batch_count += 1

            # 达到样本数量限制时停止
            total_samples = sum([c.shape[0] for c in conv_features_list])
            if total_samples >= num_samples:
                break

        # 合并批次
        conv_features = np.concatenate(conv_features_list, axis=0)[:num_samples]
        point_features = np.concatenate(point_features_list, axis=0)[:num_samples]

        print(f"  卷积特征: {conv_features.shape}")
        print(f"  点特征: {point_features.shape}")

        # 展平卷积特征
        conv_flat = conv_features.reshape(conv_features.shape[0], -1)

        # 合并所有特征
        combined_features = np.concatenate([conv_flat, point_features], axis=1)

        print(f"  合并特征: {combined_features.shape}")
        print(f"  总特征数: {combined_features.shape[1]}")

        return conv_features, point_features, combined_features

    def create_feature_names(self, conv_channels, point_features_dim):
        """
        创建特征名称列表

        Args:
            conv_channels: 卷积特征通道数
            point_features_dim: 点特征维度
        """
        print("创建特征名称...")

        feature_names = []

        # 1. 卷积特征名称
        conv_vars = ["chelsa_sfxwind", "lst", "rh", "clamday", "dem_mean", "dem_std"]

        patch_size = 5  # 假设patch_size=5
        for var_idx, var_name in enumerate(conv_vars):
            for i in range(patch_size):
                for j in range(patch_size):
                    feature_names.append(f"{var_name}_p{i}{j}")

        # 2. 点特征名称
        point_vars = ["ls1", "ls2", "ls3", "lon", "lat", "doy"]  # 假设LS有3个波段

        for i, var_name in enumerate(point_vars[:point_features_dim]):
            feature_names.append(var_name)

        self.feature_names = feature_names
        print(f"  创建了 {len(feature_names)} 个特征名称")

        return feature_names

    def create_shap_explainer(self, background_data, method='deep'):
        """
        创建SHAP解释器

        Args:
            background_data: 背景数据（用于DeepSHAP）
            method: 解释方法 ('deep', 'gradient', 'kernel')
        """
        print(f"创建SHAP解释器 (方法: {method})...")

        if method == 'deep':
            # 使用DeepSHAP（适用于PyTorch模型）
            self.explainer = shap.DeepExplainer(self.model, background_data)
            print("  ✓ 创建DeepExplainer")

        elif method == 'gradient':
            # 使用GradientExplainer
            self.explainer = shap.GradientExplainer(self.model, background_data)
            print("  ✓ 创建GradientExplainer")

        else:
            # 使用KernelSHAP（最通用但较慢）
            self.explainer = shap.KernelExplainer(self.model, background_data)
            print("  ✓ 创建KernelExplainer")

        return self.explainer

    def calculate_shap_values(self, data_to_explain, batch_size=50):
        """
        计算SHAP值

        Args:
            data_to_explain: 要解释的数据
            batch_size: 批量大小

        Returns:
            SHAP值数组
        """
        print("计算SHAP值...")

        if self.explainer is None:
            raise ValueError("请先创建SHAP解释器")

        # 分批计算以避免内存问题
        n_samples = len(data_to_explain)
        shap_values_list = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_data = data_to_explain[i:batch_end]

            print(f"  处理批次 {i // batch_size + 1}/{(n_samples + batch_size - 1) // batch_size}")

            if hasattr(self.explainer, 'shap_values'):
                # DeepExplainer
                batch_shap = self.explainer.shap_values(batch_data)
                # 如果是列表，取第一个元素
                if isinstance(batch_shap, list):
                    batch_shap = batch_shap[0]
            else:
                # KernelExplainer
                batch_shap = self.explainer(batch_data)

            shap_values_list.append(batch_shap)

        # 合并结果
        self.shap_values = np.concatenate(shap_values_list, axis=0)

        print(f"  SHAP值形状: {self.shap_values.shape}")

        return self.shap_values

    def run_variable_level_analysis(self, dataloader, num_samples=300, output_dir=None):
        """
        变量级别的重要性分析 - 只分析原始输入变量
        """
        print("\n" + "=" * 60)
        print("运行变量级别重要性分析")
        print("=" * 60)

        import torch
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
        import matplotlib

        # 设置中文字体
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        # 创建输出目录
        if output_dir is None:
            output_dir = Path("variable_importance")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. 准备数据
            print("准备特征数据...")

            conv_list = []
            point_list = []
            target_list = []

            sample_count = 0
            for conv_feats, point_feats, targets in dataloader:
                conv_list.append(conv_feats.numpy())
                point_list.append(point_feats.numpy())
                target_list.append(targets.numpy())
                sample_count += conv_feats.shape[0]

                if sample_count >= num_samples:
                    break

            # 合并数据
            conv_data = np.concatenate(conv_list, axis=0)[:num_samples]
            point_data = np.concatenate(point_list, axis=0)[:num_samples]
            target_data = np.concatenate(target_list, axis=0)[:num_samples]

            print(f"\n数据统计:")
            print(f"  卷积特征: {conv_data.shape}")  # (N, C_conv, 5, 5)
            print(f"  点特征: {point_data.shape}")  # (N, C_point)
            print(f"  目标值: {target_data.shape}")

            # 2. 定义原始变量（根据你的实际输入）
            # 卷积变量 - 根据你的 CONV_VARS 和 CONV_STATIC_VARS
            conv_variables = [
                "风场(chelsa_sfxwind)",
                "地表温度(lst)",
                "相对湿度(rh)",
                "积雪日数(clamday)",
                "DEM均值",
                "DEM标准差"
            ]

            # 点变量 - 根据你的 POINT_VARS 和添加的特征
            point_variables = [
                "土地覆盖(ls)",
                "经度(lon)",
                "纬度(lat)",
                "年积日(doy)"
            ]

            # 如果point_data有更多维度，添加说明
            if point_data.shape[1] > len(point_variables):
                print(f"  注意: point_data有{point_data.shape[1]}维，但只定义了{len(point_variables)}个变量")
                # 添加额外的点特征
                for i in range(len(point_variables), point_data.shape[1]):
                    point_variables.append(f"点特征_{i + 1}")

            all_variables = conv_variables + point_variables
            print(f"\n分析的变量 ({len(all_variables)}个):")
            for i, var in enumerate(all_variables):
                print(f"  {i + 1:2d}. {var}")

            # 3. 获取设备
            device = next(self.model.parameters()).device

            # 4. 计算基准预测
            print("\n1. 计算基准预测性能...")
            self.model.eval()

            baseline_predictions = []
            batch_size = 32

            with torch.no_grad():
                for i in range(0, len(conv_data), batch_size):
                    batch_end = min(i + batch_size, len(conv_data))

                    conv_batch = torch.FloatTensor(conv_data[i:batch_end]).to(device)
                    point_batch = torch.FloatTensor(point_data[i:batch_end]).to(device)

                    preds = self.model(conv_batch, point_batch)
                    baseline_predictions.append(preds.cpu().numpy())

            baseline_predictions = np.concatenate(baseline_predictions, axis=0)
            baseline_mse = np.mean((baseline_predictions.flatten() - target_data.flatten()) ** 2)
            baseline_rmse = np.sqrt(baseline_mse)

            print(f"  基准MSE: {baseline_mse:.6f}")
            print(f"  基准RMSE: {baseline_rmse:.6f}")
            print(f"  基准MAE: {np.mean(np.abs(baseline_predictions.flatten() - target_data.flatten())):.6f}")

            # 5. 计算变量级别的重要性（使用更可靠的方法）
            print("\n2. 计算变量重要性...")

            n_test_samples = min(100, len(conv_data))
            print(f"  使用 {n_test_samples} 个样本进行重要性评估")

            variable_importance = {}

            # 方法1：逐个变量置零（更稳定）
            print("\n  方法1: 变量置零法")
            print("  " + "-" * 50)

            for var_idx, var_name in enumerate(all_variables):
                print(f"    处理变量: {var_name}")

                if var_idx < len(conv_variables):  # 卷积变量
                    conv_idx = var_idx

                    # 创建置零版本
                    conv_zeroed = conv_data[:n_test_samples].copy()
                    conv_zeroed[:, conv_idx, :, :] = 0  # 将该变量所有像素置零

                    # 预测
                    conv_tensor = torch.FloatTensor(conv_zeroed).to(device)
                    point_tensor = torch.FloatTensor(point_data[:n_test_samples]).to(device)

                    with torch.no_grad():
                        zeroed_preds = self.model(conv_tensor, point_tensor).cpu().numpy()

                    # 计算性能
                    zeroed_mse = np.mean((zeroed_preds.flatten() - target_data[:n_test_samples].flatten()) ** 2)

                    # 重要性 = 性能变化百分比
                    importance = (zeroed_mse - baseline_mse) / baseline_mse * 100

                else:  # 点变量
                    point_idx = var_idx - len(conv_variables)

                    # 创建置零版本
                    point_zeroed = point_data[:n_test_samples].copy()
                    point_zeroed[:, point_idx] = 0  # 将该点特征置零

                    # 预测
                    conv_tensor = torch.FloatTensor(conv_data[:n_test_samples]).to(device)
                    point_tensor = torch.FloatTensor(point_zeroed).to(device)

                    with torch.no_grad():
                        zeroed_preds = self.model(conv_tensor, point_tensor).cpu().numpy()

                    # 计算性能
                    zeroed_mse = np.mean((zeroed_preds.flatten() - target_data[:n_test_samples].flatten()) ** 2)

                    # 重要性 = 性能变化百分比
                    importance = (zeroed_mse - baseline_mse) / baseline_mse * 100

                variable_importance[var_name] = {
                    'importance': importance,
                    'zeroed_mse': zeroed_mse,
                    'zeroed_rmse': np.sqrt(zeroed_mse),
                    'mse_change': zeroed_mse - baseline_mse,
                    'type': '卷积' if var_idx < len(conv_variables) else '点特征'
                }

                print(f"      置零后MSE: {zeroed_mse:.6f} (变化: {zeroed_mse - baseline_mse:+.6f})")
                print(f"      重要性: {importance:+.2f}%")

            # 方法2：使用特征置零的平均绝对误差变化（更直观）
            print("\n  方法2: MAE变化法")
            print("  " + "-" * 50)

            for var_name in all_variables:
                # 已经在方法1中计算了
                pass

            # 6. 创建结果DataFrame
            results = []
            for var_name, metrics in variable_importance.items():
                results.append({
                    '变量': var_name,
                    '类型': metrics['type'],
                    '重要性(%)': metrics['importance'],
                    'MSE变化': metrics['mse_change'],
                    '置零后MSE': metrics['zeroed_mse'],
                    '置零后RMSE': metrics['zeroed_rmse'],
                    '绝对重要性': abs(metrics['importance'])
                })

            importance_df = pd.DataFrame(results)
            importance_df = importance_df.sort_values('绝对重要性', ascending=False)

            # 7. 打印详细结果
            print("\n" + "=" * 80)
            print("变量重要性分析结果")
            print("=" * 80)
            print(f"基准性能: MSE={baseline_mse:.6f}, RMSE={baseline_rmse:.6f}")
            print(f"测试样本数: {n_test_samples}")
            print("-" * 80)

            for idx, row in importance_df.iterrows():
                effect = "损害" if row['重要性(%)'] > 0 else "改善"
                direction = "增加" if row['重要性(%)'] > 0 else "减少"
                print(f"{idx + 1:2d}. [{row['类型']:4s}] {row['变量']:20s}: {row['重要性(%)']:+.2f}% ({effect})")
                print(f"     置零后MSE: {row['置零后MSE']:.6f} ({direction}{abs(row['MSE变化']):.6f})")

            # 8. 可视化
            print("\n3. 生成可视化图表...")

            # 图1：变量重要性条形图
            plt.figure(figsize=(12, 8))

            # 按重要性排序
            plot_df = importance_df.copy()
            plot_df = plot_df.sort_values('重要性(%)', ascending=True)  # 从小到大，便于水平条形图

            colors = ['green' if imp < 0 else 'red' for imp in plot_df['重要性(%)']]
            bars = plt.barh(range(len(plot_df)), plot_df['重要性(%)'], color=colors, alpha=0.7)

            # 添加数值标签
            for i, (bar, imp) in enumerate(zip(bars, plot_df['重要性(%)'])):
                color = 'darkgreen' if imp < 0 else 'darkred'
                plt.text(imp, i, f' {imp:+.1f}%', va='center',
                         fontsize=10, fontweight='bold', color=color)

            plt.yticks(range(len(plot_df)), plot_df['变量'], fontsize=11)
            plt.xlabel('重要性 (%)', fontsize=14, fontweight='bold')
            plt.title('变量重要性分析（置零法）\n正值表示损害模型，负值表示改善模型',
                      fontsize=16, fontweight='bold', pad=20)
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            plt.grid(axis='x', alpha=0.3)

            # 添加图例
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', alpha=0.7, label='正影响（置零损害模型）')
            green_patch = mpatches.Patch(color='green', alpha=0.7, label='负影响（置零改善模型）')
            plt.legend(handles=[red_patch, green_patch], loc='lower right')

            plt.tight_layout()
            importance_plot_path = output_dir / "variable_importance_bar.png"
            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 图2：MSE变化热力图
            plt.figure(figsize=(10, 6))

            # 创建热力图数据
            heatmap_data = importance_df[['变量', 'MSE变化']].copy()
            heatmap_data = heatmap_data.sort_values('MSE变化', ascending=False)

            # 创建颜色映射
            norm = plt.Normalize(heatmap_data['MSE变化'].min(), heatmap_data['MSE变化'].max())
            colors = plt.cm.RdYlGn_r(norm(heatmap_data['MSE变化']))  # 红色表示MSE增加，绿色表示减少

            plt.bar(range(len(heatmap_data)), heatmap_data['MSE变化'], color=colors, alpha=0.7)
            plt.xticks(range(len(heatmap_data)), heatmap_data['变量'], rotation=45, ha='right', fontsize=10)
            plt.ylabel('MSE变化量', fontsize=12, fontweight='bold')
            plt.title('变量置零对MSE的影响', fontsize=14, fontweight='bold')
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            plt.grid(axis='y', alpha=0.3)

            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('MSE变化方向\n(红:增加，绿:减少)', fontsize=10)

            plt.tight_layout()
            mse_plot_path = output_dir / "mse_change_heatmap.png"
            plt.savefig(mse_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 图3：变量类型对比
            plt.figure(figsize=(10, 6))

            conv_vars = importance_df[importance_df['类型'] == '卷积']
            point_vars = importance_df[importance_df['类型'] == '点特征']

            plt.subplot(1, 2, 1)
            conv_colors = ['red' if imp > 0 else 'green' for imp in conv_vars['重要性(%)']]
            plt.bar(range(len(conv_vars)), conv_vars['重要性(%)'], color=conv_colors, alpha=0.7)
            plt.xticks(range(len(conv_vars)), [v.split('(')[0] for v in conv_vars['变量']],
                       rotation=45, ha='right', fontsize=9)
            plt.ylabel('重要性 (%)')
            plt.title('卷积变量重要性', fontsize=12, fontweight='bold')
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            point_colors = ['red' if imp > 0 else 'green' for imp in point_vars['重要性(%)']]
            plt.bar(range(len(point_vars)), point_vars['重要性(%)'], color=point_colors, alpha=0.7)
            plt.xticks(range(len(point_vars)), [v.split('(')[0] for v in point_vars['变量']],
                       rotation=45, ha='right', fontsize=9)
            plt.ylabel('重要性 (%)')
            plt.title('点特征重要性', fontsize=12, fontweight='bold')
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
            plt.grid(True, alpha=0.3)

            plt.suptitle('按类型分组的变量重要性对比', fontsize=14, fontweight='bold')
            plt.tight_layout()

            type_plot_path = output_dir / "importance_by_type.png"
            plt.savefig(type_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 9. 保存结果
            csv_path = output_dir / "variable_importance_results.csv"
            importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 10. 生成分析报告
            report_path = output_dir / "analysis_summary.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SWE模型变量重要性分析报告\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"分析时间: {pd.Timestamp.now()}\n")
                f.write(f"测试样本数: {n_test_samples}\n")
                f.write(f"基准MSE: {baseline_mse:.6f}\n")
                f.write(f"基准RMSE: {baseline_rmse:.6f}\n\n")

                f.write("核心发现:\n")
                f.write("-" * 40 + "\n")

                # 找出最重要的3个变量
                top_3 = importance_df.head(3)
                for idx, row in top_3.iterrows():
                    if row['重要性(%)'] > 0:
                        f.write(f"1. {row['变量']} 是最损害模型的变量，置零会使MSE增加{abs(row['重要性(%)']):.1f}%\n")
                    else:
                        f.write(f"1. {row['变量']} 是最改善模型的变量，置零会使MSE减少{abs(row['重要性(%)']):.1f}%\n")

                f.write("\n详细结果:\n")
                f.write("-" * 40 + "\n")
                for idx, row in importance_df.iterrows():
                    direction = "增加" if row['重要性(%)'] > 0 else "减少"
                    f.write(f"{row['变量']:20s}: 置零会使MSE {direction} {abs(row['重要性(%)']):.1f}% "
                            f"(从{baseline_mse:.6f}到{row['置零后MSE']:.6f})\n")

            print(f"\n✓ 变量级别重要性分析完成!")
            print(f"结果保存在: {output_dir}")
            print(f"\n主要文件:")
            print(f"  {csv_path} - 详细结果表")
            print(f"  {importance_plot_path} - 重要性条形图")
            print(f"  {mse_plot_path} - MSE变化热力图")
            print(f"  {type_plot_path} - 按类型分组图")
            print(f"  {report_path} - 分析报告")

            # 打印关键结论
            print(f"\n关键结论:")
            most_harmful = importance_df.iloc[0]
            most_helpful = importance_df[importance_df['重要性(%)'] < 0]
            if len(most_helpful) > 0:
                most_helpful = most_helpful.iloc[0]  # 负值中绝对值最大的
                print(f"  1. 最损害模型的变量: {most_harmful['变量']} (+{most_harmful['重要性(%)']:.1f}%)")
                print(f"  2. 最改善模型的变量: {most_helpful['变量']} ({most_helpful['重要性(%)']:.1f}%)")
            else:
                print(f"  所有变量置零都会损害模型性能")
                print(f"  最重要的变量: {most_harmful['变量']} (+{most_harmful['重要性(%)']:.1f}%)")

            return {
                'variable_importance': importance_df,
                'baseline_mse': baseline_mse,
                'baseline_rmse': baseline_rmse,
                'n_test_samples': n_test_samples,
                'output_dir': output_dir
            }

        except Exception as e:
            print(f"✗ 变量级别重要性分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_summary(self, data, max_display=20, save_path=None):
        """
        绘制SHAP摘要图（蜜蜂图）

        Args:
            data: 特征数据
            max_display: 显示的最大特征数
            save_path: 保存路径
        """
        print("绘制SHAP摘要图...")

        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")

        # 创建图形
        plt.figure(figsize=(12, 8))

        shap.summary_plot(
            self.shap_values,
            data,
            feature_names=self.feature_names[:data.shape[1]] if self.feature_names else None,
            max_display=max_display,
            plot_type="dot",
            show=False
        )

        plt.title("SHAP特征重要性摘要", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        if save_path is None:
            save_path = "shap_summary.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ SHAP摘要图保存到: {save_path}")

        return save_path

    def plot_bar_chart(self, data, max_display=15, save_path=None):
        """
        绘制SHAP条形图（平均绝对SHAP值）

        Args:
            data: 特征数据
            max_display: 显示的最大特征数
            save_path: 保存路径
        """
        print("绘制SHAP条形图...")

        if self.shap_values is None:
            raise ValueError("请先计算SHAP值")

        # 计算平均绝对SHAP值
        shap_abs_mean = np.abs(self.shap_values).mean(axis=0)

        # 排序并选择前N个特征
        sorted_indices = np.argsort(shap_abs_mean)[::-1][:max_display]

        # 创建DataFrame便于绘图
        if self.feature_names:
            feature_labels = [self.feature_names[i] for i in sorted_indices]
        else:
            feature_labels = [f"Feature {i}" for i in sorted_indices]

        shap_values_sorted = shap_abs_mean[sorted_indices]

        # 创建图形
        plt.figure(figsize=(14, 8))

        # 条形图
        bars = plt.barh(range(len(feature_labels)), shap_values_sorted,
                        color='steelblue', alpha=0.8)

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, shap_values_sorted)):
            plt.text(value, i, f' {value:.4f}',
                     va='center', fontsize=10, fontweight='bold')

        # 设置图形属性
        plt.yticks(range(len(feature_labels)), feature_labels, fontsize=11)
        plt.xlabel('平均绝对SHAP值（特征重要性）', fontsize=14, fontweight='bold')
        plt.title('Top特征重要性（平均绝对SHAP值）', fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()  # 最重要的在顶部

        plt.tight_layout()

        if save_path is None:
            save_path = "shap_bar_chart.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ SHAP条形图保存到: {save_path}")

        return save_path

    def plot_conv_spatial_importance(self, shap_values_conv, save_path=None):
        """
        绘制卷积特征的空间重要性热图

        Args:
            shap_values_conv: 卷积特征的SHAP值 (N, C, H, W)
            save_path: 保存路径
        """
        print("绘制卷积特征空间重要性热图...")

        # 计算每个通道的平均绝对SHAP值
        channel_importance = np.abs(shap_values_conv).mean(axis=(0, 2, 3))

        # 假设通道顺序：wind, lst, rh, clamday, dem_mean, dem_std
        channel_names = ["风场", "地表温度", "相对湿度",
                         "积雪日数", "DEM均值", "DEM标准差"]

        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, (channel_idx, channel_name) in enumerate(zip(range(len(channel_names)), channel_names)):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 计算该通道的空间平均SHAP值
            spatial_shap = np.abs(shap_values_conv[:, channel_idx]).mean(axis=0)

            # 绘制热图
            im = ax.imshow(spatial_shap, cmap='hot_r', aspect='auto')

            # 设置标题
            importance_value = channel_importance[channel_idx]
            ax.set_title(f'{channel_name}\n重要性: {importance_value:.4f}',
                         fontsize=12, fontweight='bold')

            ax.axis('off')

            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.8)

        # 隐藏多余的子图
        for idx in range(len(channel_names), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('卷积特征空间重要性分布', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path is None:
            save_path = "shap_conv_spatial.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 卷积特征空间热图保存到: {save_path}")

        return save_path

    def plot_dependence_plots(self, data, shap_values, top_features=6, save_dir=None):
        """
        绘制SHAP依赖图

        Args:
            data: 特征数据
            shap_values: SHAP值
            top_features: 绘制的顶级特征数量
            save_dir: 保存目录
        """
        print(f"绘制SHAP依赖图 (Top {top_features} 特征)...")

        if self.feature_names is None:
            print("  警告: 没有特征名称，跳过依赖图")
            return

        # 计算特征重要性
        importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[::-1][:top_features]

        if save_dir is None:
            save_dir = Path(".")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, feature_idx in enumerate(top_indices):
            feature_name = self.feature_names[feature_idx]

            plt.figure(figsize=(10, 6))

            shap.dependence_plot(
                feature_idx,
                shap_values,
                data,
                feature_names=self.feature_names,
                show=False
            )

            plt.title(f'SHAP依赖图: {feature_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = save_dir / f"shap_dependence_{feature_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ 依赖图保存到: {save_path}")

    def run_safe_shap_analysis(self, dataloader, num_samples=300, output_dir=None):
        """
        安全的SHAP分析，避免各种问题
        """
        print("\n" + "=" * 60)
        print("运行安全SHAP分析")
        print("=" * 60)

        # 在函数内部导入所有需要的包
        import torch
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path

        # 创建输出目录
        if output_dir is None:
            output_dir = Path("shap_safe_analysis")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. 准备数据
            print("准备特征数据...")

            conv_list = []
            point_list = []
            target_list = []

            sample_count = 0
            for conv_feats, point_feats, targets in dataloader:
                conv_list.append(conv_feats.numpy())
                point_list.append(point_feats.numpy())
                target_list.append(targets.numpy())
                sample_count += conv_feats.shape[0]

                if sample_count >= num_samples:
                    break

            # 合并数据
            conv_data = np.concatenate(conv_list, axis=0)[:num_samples]
            point_data = np.concatenate(point_list, axis=0)[:num_samples]
            target_data = np.concatenate(target_list, axis=0)[:num_samples]

            print(f"  卷积特征: {conv_data.shape}")
            print(f"  点特征: {point_data.shape}")
            print(f"  目标值: {target_data.shape}")

            # 展平卷积特征
            conv_flat = conv_data.reshape(conv_data.shape[0], -1)

            # 计算总特征数
            total_features = conv_flat.shape[1] + point_data.shape[1]
            print(f"  总特征数: {total_features} (卷积: {conv_flat.shape[1]}, 点特征: {point_data.shape[1]})")

            # 2. 创建特征名称（在知道确切数量后）
            feature_names = []

            # 卷积特征名称
            conv_vars = ["风场", "地表温度", "相对湿度", "积雪日数", "DEM均值", "DEM标准差"]
            conv_channels = conv_data.shape[1]  # 应该是6
            patch_size = int(np.sqrt(conv_flat.shape[1] / conv_channels))  # 应该是5

            print(f"  卷积通道数: {conv_channels}, Patch大小: {patch_size}")

            for var_idx in range(conv_channels):
                var_name = conv_vars[var_idx] if var_idx < len(conv_vars) else f"卷积特征_{var_idx}"
                for i in range(patch_size):
                    for j in range(patch_size):
                        feature_names.append(f"{var_name}_({i},{j})")

            # 点特征名称
            point_vars = ["土地覆盖1", "土地覆盖2", "土地覆盖3", "经度", "纬度", "年积日",
                          "附加特征1", "附加特征2", "附加特征3", "附加特征4", "附加特征5"]

            for i in range(point_data.shape[1]):
                if i < len(point_vars):
                    feature_names.append(point_vars[i])
                else:
                    feature_names.append(f"点特征_{i}")

            print(f"  创建的特征名称数量: {len(feature_names)}")
            self.feature_names = feature_names

            # 3. 使用简单的排列重要性（Permutation Importance）
            print("\n计算排列重要性...")

            # 合并所有特征
            all_features = np.concatenate([conv_flat, point_data], axis=1)

            # 获取模型设备
            device = next(self.model.parameters()).device
            print(f"  模型设备: {device}")

            # 计算基准性能
            print("  计算基准性能...")
            baseline_predictions = []

            # 使用小批量进行预测
            batch_size = 32
            for i in range(0, len(all_features), batch_size):
                batch_end = min(i + batch_size, len(all_features))

                # 获取当前批次
                conv_batch = conv_data[i:batch_end]
                point_batch = point_data[i:batch_end]

                # 转换为tensor
                conv_tensor = torch.FloatTensor(conv_batch)
                point_tensor = torch.FloatTensor(point_batch)

                # 移动到设备
                conv_tensor = conv_tensor.to(device)
                point_tensor = point_tensor.to(device)

                # 预测
                self.model.eval()
                with torch.no_grad():
                    batch_pred = self.model(conv_tensor, point_tensor)
                    baseline_predictions.append(batch_pred.cpu().numpy())

            baseline_predictions = np.concatenate(baseline_predictions, axis=0)
            baseline_mse = np.mean((baseline_predictions.flatten() - target_data.flatten()) ** 2)

            print(f"  基准MSE: {baseline_mse:.6f}")
            print(f"  基准RMSE: {np.sqrt(baseline_mse):.6f}")

            # 4. 计算每个特征的重要性
            print("  计算特征重要性...")
            n_features = all_features.shape[1]
            n_samples = min(30, len(all_features))  # 使用少量样本加速

            print(f"  将使用 {n_samples} 个样本计算 {n_features} 个特征的重要性")

            importances = np.zeros(n_features)

            for feat_idx in range(n_features):
                if feat_idx % 5 == 0 or feat_idx == n_features - 1:
                    print(f"    处理特征 {feat_idx + 1}/{n_features}")

                try:
                    # 打乱该特征
                    X_permuted = all_features[:n_samples].copy()
                    original_values = X_permuted[:, feat_idx].copy()
                    np.random.shuffle(X_permuted[:, feat_idx])

                    # 重新组织为卷积和点特征
                    conv_flat_perm = X_permuted[:, :conv_flat.shape[1]]
                    point_perm = X_permuted[:, conv_flat.shape[1]:conv_flat.shape[1] + point_data.shape[1]]

                    # 重塑卷积特征
                    conv_perm = conv_flat_perm.reshape(
                        n_samples,
                        conv_data.shape[1],
                        conv_data.shape[2],
                        conv_data.shape[3]
                    )

                    # 预测
                    conv_tensor = torch.FloatTensor(conv_perm).to(device)
                    point_tensor = torch.FloatTensor(point_perm).to(device)

                    with torch.no_grad():
                        perm_predictions = self.model(conv_tensor, point_tensor).cpu().numpy()

                    # 计算MSE
                    perm_mse = np.mean((perm_predictions.flatten() - target_data[:n_samples].flatten()) ** 2)

                    # 重要性 = MSE增加量
                    importances[feat_idx] = perm_mse - baseline_mse

                except Exception as e:
                    print(f"    特征 {feat_idx} 计算失败: {e}")
                    importances[feat_idx] = 0

            print(f"  重要性计算完成")
            print(f"  重要性范围: [{importances.min():.6f}, {importances.max():.6f}]")

            # 5. 绘制重要性图
            print("\n生成可视化图表...")

            # 确保特征名称和重要性数组长度一致
            if len(self.feature_names) != len(importances):
                print(f"  警告: 特征名称数量 ({len(self.feature_names)}) 与重要性数量 ({len(importances)}) 不匹配")
                print(f"  调整到最小长度")
                min_len = min(len(self.feature_names), len(importances))
                feature_names_adj = self.feature_names[:min_len]
                importances_adj = importances[:min_len]
            else:
                feature_names_adj = self.feature_names
                importances_adj = importances

            # 创建重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names_adj,
                'importance': importances_adj,
                'abs_importance': np.abs(importances_adj)
            })

            # 确保没有NaN值
            importance_df = importance_df.fillna(0)

            # 按绝对重要性排序
            importance_df = importance_df.sort_values('abs_importance', ascending=False)

            # 保存到CSV
            csv_path = output_dir / "feature_importance.csv"
            importance_df.to_csv(csv_path, index=False)

            # 绘制Top 20特征
            plt.figure(figsize=(16, 12))

            top_n = min(20, len(importance_df))
            top_features = importance_df.head(top_n)

            # 创建条形图
            colors = ['red' if imp < 0 else 'green' for imp in top_features['importance']]
            bars = plt.barh(range(top_n), top_features['abs_importance'], color=colors, alpha=0.7)

            # 添加数值标签
            for i, (bar, row) in enumerate(zip(bars, top_features.itertuples())):
                # 显示实际重要性值（可正可负）
                label = f'{row.importance:.6f}'
                color = 'black' if row.abs_importance > 0 else 'gray'
                plt.text(row.abs_importance, i, f' {label}', va='center',
                         fontsize=9, fontweight='bold', color=color)

            plt.yticks(range(top_n), top_features['feature'], fontsize=9)
            plt.xlabel('特征重要性（MSE变化量）', fontsize=14, fontweight='bold')
            plt.title(f'特征重要性排名（排列重要性）\n使用 {n_samples} 个样本评估 {n_features} 个特征',
                      fontsize=16, fontweight='bold', pad=20)
            plt.grid(axis='x', alpha=0.3)
            plt.gca().invert_yaxis()  # 最重要的在顶部

            # 添加图例
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', alpha=0.7, label='负影响（打乱后MSE降低）')
            green_patch = mpatches.Patch(color='green', alpha=0.7, label='正影响（打乱后MSE增加）')
            plt.legend(handles=[red_patch, green_patch], loc='lower right')

            plt.tight_layout()

            importance_plot_path = output_dir / "feature_importance_plot.png"
            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 6. 绘制重要性分布图
            plt.figure(figsize=(14, 6))

            # 正负重要性分开
            positive_imp = importance_df[importance_df['importance'] > 0]['importance']
            negative_imp = importance_df[importance_df['importance'] < 0]['importance']

            plt.subplot(1, 2, 1)
            if len(positive_imp) > 0:
                plt.hist(positive_imp, bins=30, color='green', alpha=0.7, edgecolor='black')
                plt.xlabel('正重要性（损害模型）')
                plt.ylabel('频数')
                plt.title(f'正重要性分布 ({len(positive_imp)}个特征)')
            else:
                plt.text(0.5, 0.5, '没有正重要性特征', ha='center', va='center', fontsize=12)
                plt.title('正重要性分布')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            if len(negative_imp) > 0:
                plt.hist(np.abs(negative_imp), bins=30, color='red', alpha=0.7, edgecolor='black')
                plt.xlabel('负重要性绝对值（改善模型）')
                plt.ylabel('频数')
                plt.title(f'负重要性分布 ({len(negative_imp)}个特征)')
            else:
                plt.text(0.5, 0.5, '没有负重要性特征', ha='center', va='center', fontsize=12)
                plt.title('负重要性分布')
            plt.grid(True, alpha=0.3)

            plt.suptitle('特征重要性分布', fontsize=14, fontweight='bold')
            plt.tight_layout()

            dist_plot_path = output_dir / "importance_distribution.png"
            plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 7. 绘制Top特征详细图
            if len(importance_df) >= 5:
                plt.figure(figsize=(15, 10))

                top_5 = importance_df.head(5)

                for idx, (_, row) in enumerate(top_5.iterrows()):
                    plt.subplot(2, 3, idx + 1)

                    # 分析该特征与预测值的关系
                    feat_idx = feature_names_adj.index(row.feature) if row.feature in feature_names_adj else idx

                    if feat_idx < all_features.shape[1]:
                        # 获取特征值
                        feat_values = all_features[:100, feat_idx]
                        pred_values = baseline_predictions[:100].flatten()

                        plt.scatter(feat_values, pred_values, alpha=0.5, s=20)
                        plt.xlabel(f'{row.feature}\n值')
                        plt.ylabel('预测值')
                        plt.title(f'{row.feature}\n重要性: {row.importance:.6f}')
                        plt.grid(True, alpha=0.3)
                    else:
                        plt.text(0.5, 0.5, '特征索引超出范围', ha='center', va='center')
                        plt.title(f'{row.feature}')

                # 最后一个子图显示重要性排名
                plt.subplot(2, 3, 6)
                importance_df['rank'] = range(1, len(importance_df) + 1)
                plt.plot(importance_df['rank'], importance_df['abs_importance'], 'b-', linewidth=2)
                plt.xlabel('特征排名')
                plt.ylabel('绝对重要性')
                plt.title('重要性随排名下降曲线')
                plt.grid(True, alpha=0.3)

                plt.suptitle('Top 5 最重要特征详细分析', fontsize=16, fontweight='bold')
                plt.tight_layout()

                top_features_path = output_dir / "top_features_analysis.png"
                plt.savefig(top_features_path, dpi=300, bbox_inches='tight')
                plt.close()

            print(f"\n✓ SHAP分析完成!")
            print(f"结果保存在: {output_dir}")
            print(f"  特征重要性表: {csv_path} ({len(importance_df)} 个特征)")
            print(f"  特征重要性图: {importance_plot_path}")
            print(f"  重要性分布图: {dist_plot_path}")

            # 打印Top 10特征
            print("\nTop 10 最重要特征:")
            for idx, row in importance_df.head(10).iterrows():
                effect = "损害模型" if row.importance > 0 else "改善模型"
                importance_str = f"{abs(row.importance):.6f}"
                if row.importance > 0:
                    importance_str = f"+{importance_str}"
                else:
                    importance_str = f"-{importance_str}"

                print(f"  {idx + 1:2d}. {row.feature:35s} : {importance_str} ({effect})")

            # 打印统计信息
            print(f"\n重要性统计:")
            print(f"  正重要性特征数: {len(positive_imp)} ({len(positive_imp) / len(importance_df) * 100:.1f}%)")
            print(f"  负重要性特征数: {len(negative_imp)} ({len(negative_imp) / len(importance_df) * 100:.1f}%)")
            print(f"  零重要性特征数: {len(importance_df) - len(positive_imp) - len(negative_imp)}")
            print(f"  最大正重要性: {positive_imp.max() if len(positive_imp) > 0 else 0:.6f}")
            print(f"  最大负重要性: {negative_imp.min() if len(negative_imp) > 0 else 0:.6f}")

            return {
                'feature_importance': importance_df,
                'importances': importances,
                'output_dir': output_dir,
                'baseline_mse': baseline_mse,
                'n_samples_used': n_samples,
                'n_features': n_features
            }

        except Exception as e:
            print(f"✗ SHAP分析失败: {e}")
            import traceback
            traceback.print_exc()

            # 即使失败也返回一个空结果，避免程序崩溃
            return {
                'feature_importance': pd.DataFrame(),
                'importances': np.array([]),
                'output_dir': output_dir,
                'error': str(e)
            }

    def run_full_analysis(self, dataloader, num_samples=500, output_dir=None):
        """
        运行完整的SHAP分析
        """
        print("\n" + "=" * 60)
        print("开始完整SHAP分析")
        print("=" * 60)

        # 创建输出目录
        if output_dir is None:
            output_dir = Path("shap_analysis")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 准备数据
        conv_features, point_features, combined_features = \
            self.prepare_features_from_dataloader(dataloader, num_samples)

        print(f"[调试] 卷积特征形状: {conv_features.shape}")
        print(f"[调试] 点特征形状: {point_features.shape}")

        # 2. 创建特征名称
        if self.feature_names is None:
            self.create_feature_names(
                conv_channels=conv_features.shape[1],
                point_features_dim=point_features.shape[1]
            )

        # 3. 为SWENet创建包装函数
        def model_wrapper(input_array):
            """
            包装函数，将单个输入数组拆分为conv_feats和point_feats
            """
            import torch
            import numpy as np

            batch_size = input_array.shape[0]

            # 计算卷积特征和点特征的尺寸
            conv_features_flat_size = conv_features.shape[1] * conv_features.shape[2] * conv_features.shape[3]
            point_features_dim = point_features.shape[1]

            print(
                f"[SHAP包装] 输入形状: {input_array.shape}, 卷积展平尺寸: {conv_features_flat_size}, 点特征维度: {point_features_dim}")

            # 拆分特征
            conv_flat = input_array[:, :conv_features_flat_size]
            point_data = input_array[:, conv_features_flat_size:conv_features_flat_size + point_features_dim]

            # 重塑卷积特征
            conv_reshaped = conv_flat.reshape(
                batch_size,
                conv_features.shape[1],  # C
                conv_features.shape[2],  # H
                conv_features.shape[3]  # W
            )

            # 转换为torch tensor
            # 修复：直接使用cpu设备，因为SHAP通常在CPU上运行
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            conv_tensor = torch.FloatTensor(conv_reshaped).to(device)
            point_tensor = torch.FloatTensor(point_data).to(device)

            # 确保模型在正确的设备上
            self.model.to(device)

            # 模型预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(conv_tensor, point_tensor)

            return outputs.cpu().numpy()

        # 4. 准备背景数据（前100个样本）
        background_size = min(100, len(combined_features))
        background_data = combined_features[:background_size]

        print(f"[调试] 背景数据形状: {background_data.shape}")
        print(f"[调试] 准备使用KernelSHAP...")

        # 5. 创建KernelSHAP解释器（使用包装函数）
        try:
            import shap

            # 创建解释器
            self.explainer = shap.KernelExplainer(
                model_wrapper,
                background_data,
                link="identity"
            )
            print("  ✓ 创建KernelExplainer成功")

        except Exception as e:
            print(f"  ✗ 创建解释器失败: {e}")
            return None

        # 6. 计算SHAP值（使用较小的样本）
        print("计算SHAP值...")
        sample_size = min(50, num_samples)  # SHAP计算较慢，先使用少量样本
        sample_data = combined_features[:sample_size]

        try:
            shap_values = self.explainer.shap_values(
                sample_data,
                nsamples=100  # 减少采样数以加快速度
            )

            print(f"  ✓ SHAP值计算完成，形状: {shap_values.shape}")

        except Exception as e:
            print(f"  ✗ 计算SHAP值失败: {e}")
            return None

        # 7. 绘制各种图形
        print("\n生成可视化图表...")

        # 摘要图
        summary_path = output_dir / "shap_summary.png"
        try:
            self.plot_summary(sample_data, max_display=25, save_path=summary_path)
        except Exception as e:
            print(f"  摘要图失败: {e}")

        # 条形图
        bar_path = output_dir / "shap_bar_chart.png"
        try:
            self.plot_bar_chart(sample_data, max_display=20, save_path=bar_path)
        except Exception as e:
            print(f"  条形图失败: {e}")

        # 保存SHAP值
        shap_save_path = output_dir / "shap_values.npy"
        np.save(shap_save_path, shap_values)

        # 保存特征重要性
        importance = np.abs(shap_values).mean(axis=0)
        if len(self.feature_names) >= len(importance):
            feature_names_subset = self.feature_names[:len(importance)]
        else:
            feature_names_subset = [f"Feature_{i}" for i in range(len(importance))]

        importance_df = pd.DataFrame({
            'feature': feature_names_subset,
            'importance': importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)

        importance_path = output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        print("\n" + "=" * 60)
        print("SHAP分析完成!")
        print("=" * 60)
        print(f"输出目录: {output_dir}")
        print(f"主要文件:")
        print(f"  {summary_path}")
        print(f"  {bar_path}")
        print(f"  {importance_path}")
        print(f"  {shap_save_path}")

        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'output_dir': output_dir
        }