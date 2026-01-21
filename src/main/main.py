# train_swe_main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWE反演模型主训练脚本
整合数据加载器和模型，支持不同特征组合的训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import warnings
from shap_analyzer import SHAPAnalyzer
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型
try:
    from models_swe import create_model, SWENet_Full, SWENet_SpatialOnly, SWENet_PointOnly, test_model
    print("✓ 成功导入模型")
except ImportError as e:
    print(f"✗ 导入模型失败: {e}")
    print("请确保 models_swe.py 在相同目录下")
    sys.exit(1)

# 导入数据加载器
try:
    from data_online_era5_swe import build_dataloaders, build_temporal_split_dataloaders, \
        build_spatial_split_dataloaders

    print("✓ 成功导入数据加载器")
except ImportError as e:
    print(f"✗ 导入数据加载器失败: {e}")
    print("请确保 data_online_era5_swe.py 在相同目录下")
    sys.exit(1)


class SWETrainer:
    """SWE模型训练器"""

    def __init__(self, config=None):
        # 默认配置
        self.default_config = {
            # 模型类型
            'model_type': 'full',  # full, spatial_only, point_only

            # 数据参数
            'batch_size': 16,
            'val_ratio': 0.2,
            'num_workers': 0,

            # 训练参数
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'patience': 15,

            # 模型参数（将从数据自动获取）
            'C_conv': None,
            'C_point': None,
            'd_model': 256,

            # 路径设置
            'save_dir': './experiments',
            'experiment_name': None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            # 其他
            'seed': 42,
            'clip_grad': 1.0,
            'save_freq': 10,
        }

        # 更新配置
        if config:
            self.default_config.update(config)
        self.config = self.default_config

        # 设置实验名称
        if self.config['experiment_name'] is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config['experiment_name'] = f"swe_{self.config['model_type']}_{timestamp}"

        # 设置设备
        self.device = torch.device(self.config['device'])
        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # 初始化变量
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None

        # 训练历史
        self.train_history = []
        self.val_history = []
        self.lr_history = []

        # 创建保存目录
        self.save_dir = Path(self.config['save_dir']) / self.config['experiment_name']
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        print(f"实验保存目录: {self.save_dir}")

    def _save_config(self):
        """保存配置到文件"""
        config_path = self.save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"配置已保存到: {config_path}")

    def load_data(self):
        """加载数据"""
        print("\n" + "=" * 60)
        print("加载数据...")
        print("=" * 60)

        try:
            # 根据配置选择数据划分方式
            split_method = self.config.get('split_method', 'random')

            # 准备数据集参数
            dataset_params = {
                'region': 'XINJIANG',
                'year_target': 2016,  # 目标年份
                'patch_size': 5,
                'min_valid_pixels': 100,
                'samples_per_day': 2000,
                'clamday_threshold': 0.5
            }

            if split_method == 'temporal':
                # 按年份划分
                train_year = self.config.get('train_year', 2015)
                val_year = self.config.get('val_year', 2016)

                # 更新 year_target 为验证年份（或最大的年份）
                dataset_params['year_target'] = val_year

                train_loader, val_loader, shapes = build_temporal_split_dataloaders(
                    train_years=[train_year],
                    val_years=[val_year],
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    seed=self.config['seed'],
                    **dataset_params  # 传递数据集参数
                )

            elif split_method == 'spatial':
                # 按空间划分
                spatial_split_ratio = self.config.get('spatial_split_ratio', 0.2)
                split_by = self.config.get('split_by', 'blocks')

                train_loader, val_loader, shapes = build_spatial_split_dataloaders(
                    spatial_split_ratio=spatial_split_ratio,
                    split_by=split_by,
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    seed=self.config['seed'],
                    **dataset_params  # 传递数据集参数
                )

            else:
                # 默认随机划分
                train_loader, val_loader, shapes = build_dataloaders(
                    batch_size=self.config['batch_size'],
                    val_ratio=self.config['val_ratio'],
                    num_workers=self.config['num_workers'],
                    seed=self.config['seed'],
                    **dataset_params  # 传递数据集参数
                )

            # 获取维度信息
            C_conv, C_point = shapes

            # 更新配置
            self.config['C_conv'] = C_conv
            self.config['C_point'] = C_point

            print(f"✓ 数据加载成功!")
            print(f"\n数据维度:")
            print(f"  卷积特征: C_conv={C_conv}")
            print(f"  点特征: C_point={C_point}")
            print(f"\n数据统计:")
            print(f"  训练集: {len(train_loader.dataset)} 个样本")
            print(f"  验证集: {len(val_loader.dataset)} 个样本")
            print(f"  批次大小: {self.config['batch_size']}")

            self.train_loader = train_loader
            self.val_loader = val_loader

            # 测试一个批次
            self._test_data_loading()

            return True

        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _test_data_loading(self):
        """测试数据加载"""
        print(f"\n测试数据加载...")
        try:
            # 获取一个批次
            conv, point, target = next(iter(self.train_loader))

            print(f"  卷积特征: {conv.shape}")
            print(f"  点特征: {point.shape}")
            print(f"  目标值: {target.shape}")

            # 检查数据范围
            print(f"\n  数据范围检查:")
            print(f"    卷积特征: [{conv.min():.3f}, {conv.max():.3f}]")
            print(f"    点特征: [{point.min():.3f}, {point.max():.3f}]")
            print(f"    目标值: [{target.min():.3f}, {target.max():.3f}]")

            # 检查NaN
            print(f"\n  NaN检查:")
            print(f"    卷积特征 NaN: {torch.isnan(conv).any().item()}")
            print(f"    点特征 NaN: {torch.isnan(point).any().item()}")
            print(f"    目标值 NaN: {torch.isnan(target).any().item()}")

            return True

        except Exception as e:
            print(f"✗ 数据测试失败: {e}")
            return False

    def build_model(self):
        """构建模型"""
        print("\n" + "=" * 60)
        print(f"构建模型 ({self.config['model_type']})...")
        print("=" * 60)

        try:
            # 如果C_point为None，尝试从数据集中获取
            if self.config['C_point'] is None:
                print("警告: C_point为None，尝试从数据中推断...")
                # 尝试从数据加载器获取一个样本来推断维度
                try:
                    conv, point, _ = next(iter(self.train_loader))
                    inferred_C_point = point.shape[1]
                    self.config['C_point'] = inferred_C_point
                    print(f"从数据推断 C_point={inferred_C_point}")
                except Exception as e:
                    print(f"无法从数据推断C_point: {e}")
                    # 使用默认值
                    self.config['C_point'] = 10  # 假设有10个点特征
                    print(f"使用默认值 C_point={self.config['C_point']}")

            # 检查维度是否已设置
            if self.config['C_conv'] is None:
                self.config['C_conv'] = 6  # 默认值：3个动态变量 + 3个静态变量

            print(f"最终模型参数: C_conv={self.config['C_conv']}, C_point={self.config['C_point']}")

            # 根据配置创建模型
            self.model = create_model(
                model_type=self.config['model_type'],
                C_spatial=self.config['C_conv'],
                C_point=self.config['C_point'],
                d_model=self.config['d_model']
            )

            # 移动到设备
            self.model.to(self.device)

            # 打印模型信息
            self._print_model_info()

            # 设置损失函数和优化器
            self.criterion = nn.MSELoss()

            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

            # 学习率调度器
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )

            print(f"✓ 模型构建成功!")

            return True

        except Exception as e:
            print(f"✗ 模型构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n模型信息:")
        print(f"  类型: {self.config['model_type']}")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  参数量占比: {trainable_params / total_params * 100:.1f}%")

        # 打印各模块参数量
        print(f"\n各模块参数量:")
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name:20s}: {params:10,} ({params / total_params * 100:5.1f}%)")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, (conv_feats, point_feats, targets) in enumerate(self.train_loader):
            # 移动到设备
            conv_feats = conv_feats.to(self.device)
            point_feats = point_feats.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            outputs = self.model(conv_feats, point_feats)
            loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.config['clip_grad'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['clip_grad']
                )

            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            batch_count += 1

            # 每10个batch打印一次
            if (batch_idx + 1) % 100== 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.6f}")

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        batch_count = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for conv_feats, point_feats, targets in self.val_loader:
                # 移动到设备
                conv_feats = conv_feats.to(self.device)
                point_feats = point_feats.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(conv_feats, point_feats)
                loss = self.criterion(outputs, targets)

                # 记录
                total_loss += loss.item()
                batch_count += 1

                # 收集预测结果
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / batch_count if batch_count > 0 else 0

        # 计算额外指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        metrics = {
            'loss': avg_loss,
            'rmse': np.sqrt(np.mean((all_predictions - all_targets) ** 2)),
            'mae': np.mean(np.abs(all_predictions - all_targets)),
            'correlation': np.corrcoef(all_predictions, all_targets)[0, 1] if len(all_targets) > 1 else 0
        }

        return metrics

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print("开始训练...")
        print("=" * 60)

        # 检查数据和模型
        if self.train_loader is None or self.val_loader is None:
            print("✗ 请先加载数据!")
            return

        if self.model is None:
            print("✗ 请先构建模型!")
            return

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        # 训练循环
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print("-" * 50)

            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_history.append(train_loss)

            # 验证
            val_metrics = self.validate()
            self.val_history.append(val_metrics['loss'])

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)

            # 调整学习率
            self.scheduler.step(val_metrics['loss'])

            # 打印结果
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_metrics['loss']:.6f}")
            print(f"验证RMSE: {val_metrics['rmse']:.6f}")
            print(f"验证MAE:  {val_metrics['mae']:.6f}")
            print(f"验证相关系数: {val_metrics['correlation']:.4f}")
            print(f"学习率: {current_lr:.2e}")

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                patience_counter = 0

                # 保存模型
                self.save_checkpoint(f"best_model.pth", epoch, val_metrics)
                print(f"✓ 保存最佳模型 (epoch {epoch + 1})")
            else:
                patience_counter += 1

            # 定期保存检查点
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch + 1}.pth", epoch, val_metrics)

            # 早停检查
            if patience_counter >= self.config['patience']:
                print(f"\n⚠ 早停触发! 连续{self.config['patience']}轮验证损失未改善")
                break

        print("\n" + "=" * 60)
        print(f"训练完成!")
        print(f"最佳验证损失: {best_val_loss:.6f} (epoch {best_epoch + 1})")
        print("=" * 60)

        # 保存最终模型
        self.save_checkpoint("final_model.pth", best_epoch, {'loss': best_val_loss})

        # 保存训练历史
        self.save_training_history()

        # 绘制训练曲线
        self.plot_training_curves()

        return best_val_loss

    def save_checkpoint(self, filename, epoch, metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),  # 只保存模型状态
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'lr_history': self.lr_history,
            'config': self.config,
            'metrics': metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        save_path = self.save_dir / filename
        # 使用 torch.save 的标准方式
        torch.save(checkpoint, save_path, pickle_protocol=4)
        print(f"✓ 检查点保存到: {save_path}")

    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_loss': self.train_history,
            'val_loss': self.val_history,
            'lr_history': self.lr_history,
            'config': self.config,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

        print(f"训练历史已保存到: {history_path}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. 损失曲线
            ax1 = axes[0, 0]
            epochs = range(1, len(self.train_history) + 1)
            ax1.plot(epochs, self.train_history, 'b-', label='训练损失', linewidth=2)
            ax1.plot(epochs, self.val_history, 'r-', label='验证损失', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss (MSE)', fontsize=12)
            ax1.set_title('训练和验证损失', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)

            # 标记最佳epoch
            best_idx = np.argmin(self.val_history)
            ax1.scatter(best_idx + 1, self.val_history[best_idx], color='red', s=100,
                        zorder=5, label=f'最佳 (Epoch {best_idx + 1})')
            ax1.legend(fontsize=11)

            # 2. 学习率曲线
            ax2 = axes[0, 1]
            ax2.plot(epochs, self.lr_history, 'g-', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('学习率变化', fontsize=14, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

            # 3. 损失对比（对数坐标）
            ax3 = axes[1, 0]
            ax3.plot(epochs, self.train_history, 'b-', label='训练', linewidth=2)
            ax3.plot(epochs, self.val_history, 'r-', label='验证', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Loss (log)', fontsize=12)
            ax3.set_title('损失对比（对数坐标）', fontsize=14, fontweight='bold')
            ax3.set_yscale('log')
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)

            # 4. 损失比率
            ax4 = axes[1, 1]
            if len(self.train_history) > 0 and len(self.val_history) > 0:
                # 计算过拟合比率
                overfit_ratio = [v / t if t > 0 else 0 for t, v in zip(self.train_history, self.val_history)]
                ax4.plot(epochs, overfit_ratio, 'purple', linewidth=2)
                ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Epoch', fontsize=12)
                ax4.set_ylabel('验证/训练损失比率', fontsize=12)
                ax4.set_title('过拟合监测', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)

            plt.suptitle(f'SWE模型训练曲线 - {self.config["model_type"]}', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # 保存图像
            plot_path = self.save_dir / "training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"训练曲线已保存到: {plot_path}")

        except Exception as e:
            print(f"绘制训练曲线失败: {e}")

    def evaluate(self, model_path=None, run_ablation=False, ablation_samples=None,
                 ablation_method='retrain', retrain_epochs=50, run_shap=True):
        """
        评估模型 - 优化版，确保SHAP分析和蜂巢图正确运行

        Args:
            model_path: 模型文件路径
            run_ablation: 是否运行消融实验
            ablation_samples: 消融实验使用的样本数
            ablation_method: 消融实验方法 'retrain'或'zeroing'
            retrain_epochs: 重新训练的轮次
            run_shap: 是否运行SHAP分析（默认True）
        """

        print("\n" + "=" * 60)
        print("评估模型 - 优化版")
        print("=" * 60)

        # 1. 加载模型
        model_loaded = self._load_model_for_evaluation(model_path)
        if not model_loaded:
            print("✗ 模型加载失败，无法进行评估")
            return None

        # 2. 检查数据加载器
        if self.val_loader is None:
            print("✗ 验证数据加载器不存在")
            return None

        # 3. 进行预测
        print("\n1. 进行预测...")
        predictions, targets = self._make_predictions()

        if predictions is None or targets is None:
            print("✗ 预测失败")
            return None

        print(f"  有效预测样本数: {len(predictions):,}")

        # 4. 计算评估指标
        print("\n2. 计算评估指标...")
        eval_results = self._compute_metrics(predictions, targets)

        if eval_results is None:
            print("✗ 指标计算失败")
            return None

        # 5. 保存评估结果
        print("\n3. 保存评估结果...")
        self._save_evaluation_results(eval_results)

        # 6. 绘制图表
        print("\n4. 生成可视化图表...")
        self._generate_plots(predictions, targets, eval_results)

        # 7. 运行SHAP分析（如果启用）
        if run_shap and len(predictions) >= 100:
            print("\n5. 运行SHAP特征重要性分析...")
            shap_results = self._run_shap_with_hexbin(predictions, targets, eval_results)

            if shap_results:
                print(f"  ✓ SHAP分析完成，结果保存在: {shap_results['output_dir']}")
            else:
                print("  ⚠ SHAP分析失败，但继续执行其他评估")

        # 8. 运行消融实验（如果启用）
        if run_ablation:
            print("\n6. 运行消融实验...")
            ablation_results = self._run_ablation_study(
                model_path=model_path,
                ablation_samples=ablation_samples,
                ablation_method=ablation_method,
                retrain_epochs=retrain_epochs
            )

            if ablation_results:
                print(f"  ✓ 消融实验完成，结果保存在: {ablation_results['output_dir']}")

        print("\n" + "=" * 60)
        print("评估完成!")
        print("=" * 60)

        return eval_results

    def plot_debug_analysis(self, predictions, targets):
        """绘制调试分析图，特别关注负相关问题"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. 散点图
            ax1 = axes[0, 0]
            ax1.scatter(targets, predictions, alpha=0.3, s=10)
            ax1.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='1:1线')
            ax1.set_xlabel('目标值')
            ax1.set_ylabel('预测值')
            ax1.set_title(f'散点图 (R={np.corrcoef(predictions, targets)[0, 1]:.4f})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 残差图
            ax2 = axes[0, 1]
            residuals = predictions - targets
            ax2.scatter(targets, residuals, alpha=0.3, s=10)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('目标值')
            ax2.set_ylabel('残差 (预测-目标)')
            ax2.set_title('残差分析')
            ax2.grid(True, alpha=0.3)

            # 3. 预测值和目标值分布
            ax3 = axes[0, 2]
            ax3.hist(predictions, bins=50, alpha=0.5, label='预测值', color='blue')
            ax3.hist(targets, bins=50, alpha=0.5, label='目标值', color='red')
            ax3.set_xlabel('值')
            ax3.set_ylabel('频数')
            ax3.set_title('预测值和目标值分布')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. 分位数图
            ax4 = axes[1, 0]
            sorted_predictions = np.sort(predictions)
            sorted_targets = np.sort(targets)
            ax4.plot(sorted_predictions, sorted_targets, 'o', alpha=0.5, markersize=3)
            ax4.plot([sorted_predictions.min(), sorted_predictions.max()],
                     [sorted_targets.min(), sorted_targets.max()], 'r--')
            ax4.set_xlabel('预测值分位数')
            ax4.set_ylabel('目标值分位数')
            ax4.set_title('分位数-分位数图')
            ax4.grid(True, alpha=0.3)

            # 5. 误差与目标值关系
            ax5 = axes[1, 1]
            abs_errors = np.abs(residuals)
            ax5.scatter(targets, abs_errors, alpha=0.3, s=10)
            ax5.set_xlabel('目标值')
            ax5.set_ylabel('绝对误差')
            ax5.set_title('误差 vs 目标值')
            ax5.grid(True, alpha=0.3)

            # 6. 散点密度图（热力图）
            ax6 = axes[1, 2]
            from scipy.stats import gaussian_kde
            xy = np.vstack([targets, predictions])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            ax6.scatter(targets[idx], predictions[idx], c=z[idx], s=10, alpha=0.6, cmap='viridis')
            ax6.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
            ax6.set_xlabel('目标值')
            ax6.set_ylabel('预测值')
            ax6.set_title('散点密度图')
            ax6.grid(True, alpha=0.3)

            plt.suptitle('模型评估调试分析', fontsize=16, fontweight='bold')
            plt.tight_layout()

            debug_path = self.save_dir / "debug_analysis.png"
            plt.savefig(debug_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ 调试分析图已保存到: {debug_path}")

        except Exception as e:
            print(f"调试分析图失败: {e}")

    def plot_density_scatter_hardcode(self, predictions, targets):
        """
        硬编码的密度散点图，直接调用无需参数
        左上角显示大字体指标
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.colors import LogNorm
            from scipy import stats
            import numpy as np

            # 设置样式
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")

            # 准备数据
            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()

            # 移除NaN
            mask = ~np.isnan(predictions) & ~np.isnan(targets)
            predictions = predictions[mask]
            targets = targets[mask]

            if len(predictions) == 0:
                print("警告：没有有效数据用于绘图")
                return

            # 计算指标
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            bias = np.mean(predictions - targets)

            try:
                r_value, _ = stats.pearsonr(predictions, targets)
            except:
                r_value = np.corrcoef(predictions, targets)[0, 1]

            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 8))

            # 计算密度
            from scipy.stats import gaussian_kde
            xy = np.vstack([targets, predictions])
            z = gaussian_kde(xy)(xy)

            # 排序使高密度点在顶部
            idx = z.argsort()
            x_sorted, y_sorted, z_sorted = targets[idx], predictions[idx], z[idx]

            # 绘制密度散点图
            scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted,
                                 cmap='viridis', s=30, alpha=0.7,
                                 edgecolors='none', norm=LogNorm())

            # 1:1线
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            margin = (max_val - min_val) * 0.05

            ax.plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=3, alpha=0.8, label='1:1线')

            # 回归线
            if len(targets) > 1:
                coeffs = np.polyfit(targets, predictions, 1)
                reg_line = np.poly1d(coeffs)
                x_range = np.linspace(min_val, max_val, 100)
                ax.plot(x_range, reg_line(x_range), 'orange',
                        linewidth=3, alpha=0.8, label=f'回归线')

            # 设置图形属性
            ax.set_xlabel('真实值', fontsize=16, fontweight='bold')
            ax.set_ylabel('预测值', fontsize=16, fontweight='bold')
            ax.set_title('SWE预测结果', fontsize=18, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12, loc='lower right')
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
            cbar.set_label('点密度', fontsize=14, fontweight='bold')

            # 左上角指标框 - 大字体
            metrics_text = (f'MAE = {mae:.4f}\n'
                            f'RMSE = {rmse:.4f}\n'
                            f'R = {r_value:.4f}\n'
                            f'Bias = {bias:.4f}\n'
                            f'N = {len(targets)}')

            # 白色背景，黑色边框
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white",
                              edgecolor="black", alpha=0.95, linewidth=2)

            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    fontsize=14,  # 增大字体
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=bbox_props)

            # 调整布局并保存
            plt.tight_layout()
            plot_path = self.save_dir / "density_scatter_final.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ 密度散点图已保存: {plot_path}")

            return {
                'mae': mae,
                'rmse': rmse,
                'r': r_value,
                'plot_path': str(plot_path)
            }

        except Exception as e:
            print(f"绘图失败: {e}")
            import traceback
            traceback.print_exc()

    def plot_density_scatter(self, predictions, targets,
                             save_path=None, figsize=(12, 10),
                             cmap='viridis', alpha=0.8):
        """
        绘制带密度可视化的预测-真实值散点图

        Args:
            predictions: 预测值数组
            targets: 真实值数组
            save_path: 保存路径，如果为None则使用默认路径
            figsize: 图像大小
            cmap: 颜色映射
            alpha: 透明度
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            from scipy import stats
            import seaborn as sns

            # 设置seaborn样式
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=1.2)

            # 创建图形
            fig, (ax_main, ax_hist_x, ax_hist_y, ax_cbar) = plt.subplots(
                2, 2,
                figsize=figsize,
                gridspec_kw={
                    'height_ratios': [3, 1],
                    'width_ratios': [3, 1],
                    'wspace': 0.05,
                    'hspace': 0.05
                }
            )

            # 移除多余的坐标轴
            ax_cbar.axis('off')

            # 计算统计指标
            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()

            # 移除NaN值
            mask = ~np.isnan(predictions) & ~np.isnan(targets)
            predictions = predictions[mask]
            targets = targets[mask]

            if len(predictions) == 0:
                print("警告: 没有有效数据用于绘图")
                return

            # 计算指标
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))

            try:
                r_value, p_value = stats.pearsonr(predictions, targets)
            except:
                r_value = np.corrcoef(predictions, targets)[0, 1]

            bias = np.mean(predictions - targets)

            # 计算密度
            from scipy.stats import gaussian_kde
            xy = np.vstack([targets, predictions])
            z = gaussian_kde(xy)(xy)

            # 按密度排序，使高密度点最后绘制（显示在顶部）
            idx = z.argsort()
            x_sorted, y_sorted, z_sorted = targets[idx], predictions[idx], z[idx]

            # 主散点图（带密度颜色）
            scatter = ax_main.scatter(
                x_sorted, y_sorted,
                c=z_sorted,
                cmap=cmap,
                s=30,  # 点的大小
                alpha=alpha,
                edgecolors='none',
                norm=LogNorm(vmin=z_sorted.min() + 1, vmax=z_sorted.max())
            )

            # 1:1 参考线
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            ax_main.plot([min_val, max_val], [min_val, max_val],
                         'r--', linewidth=3, alpha=0.8, label='1:1线')

            # 回归线
            if len(targets) > 1:
                coeffs = np.polyfit(targets, predictions, 1)
                reg_line = np.poly1d(coeffs)
                x_range = np.linspace(min_val, max_val, 100)
                ax_main.plot(x_range, reg_line(x_range),
                             'orange', linewidth=3, alpha=0.8,
                             label=f'回归线: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')

            # 设置主图
            ax_main.set_xlabel('真实值 (标准化SWE)', fontsize=16, fontweight='bold', labelpad=10)
            ax_main.set_ylabel('预测值 (标准化SWE)', fontsize=16, fontweight='bold', labelpad=10)
            ax_main.set_title('SWE预测密度散点图', fontsize=18, fontweight='bold', pad=15)
            ax_main.grid(True, alpha=0.3, linestyle='--')
            ax_main.legend(loc='lower right', fontsize=12, framealpha=0.9)

            # 设置坐标轴范围
            margin = (max_val - min_val) * 0.05
            ax_main.set_xlim(min_val - margin, max_val + margin)
            ax_main.set_ylim(min_val - margin, max_val + margin)

            # 添加指标文本框（右上角）
            metrics_text = (f'MAE = {mae:.4f}\n'
                            f'RMSE = {rmse:.4f}\n'
                            f'R = {r_value:.4f}\n'
                            f'Bias = {bias:.4f}\n'
                            f'N = {len(targets):,}')

            bbox_props = dict(boxstyle="round,pad=0.6", facecolor="white",
                              edgecolor="gray", alpha=0.95, linewidth=2)

            ax_main.text(0.98, 0.98, metrics_text,
                         transform=ax_main.transAxes,
                         fontsize=14,
                         fontfamily='monospace',
                         verticalalignment='top',
                         horizontalalignment='right',
                         bbox=bbox_props)

            # 直方图 - X轴（真实值）
            ax_hist_x.hist(targets, bins=50, color='skyblue', alpha=0.7,
                           edgecolor='black', linewidth=0.5)
            ax_hist_x.set_xlabel('真实值分布', fontsize=12, fontweight='bold')
            ax_hist_x.set_ylabel('频数', fontsize=12)
            ax_hist_x.grid(True, alpha=0.3)

            # 直方图 - Y轴（预测值）
            ax_hist_y.hist(predictions, bins=50, orientation='horizontal',
                           color='lightcoral', alpha=0.7,
                           edgecolor='black', linewidth=0.5)
            ax_hist_y.set_xlabel('频数', fontsize=12)
            ax_hist_y.set_ylabel('预测值分布', fontsize=12, fontweight='bold')
            ax_hist_y.grid(True, alpha=0.3)

            # 添加颜色条
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 调整位置
            cbar = plt.colorbar(scatter, cax=cbar_ax)
            cbar.set_label('点密度 (log尺度)', fontsize=14, fontweight='bold', labelpad=15)
            cbar.ax.tick_params(labelsize=12)

            # 添加模型信息标题
            model_info = f"{self.config.get('model_type', 'full').upper()} 模型"
            if self.config.get('split_method'):
                split_method = self.config['split_method']
                if split_method == 'temporal':
                    info = f"训练: {self.config.get('train_year', '')}年 | 验证: {self.config.get('val_year', '')}年"
                else:
                    info = f"{split_method.capitalize()} 划分"
                model_info = f"{model_info} | {info}"

            fig.suptitle(model_info, fontsize=14, fontweight='bold', y=0.98)

            plt.tight_layout()

            # 保存图像
            if save_path is None:
                save_path = self.save_dir / "density_scatter.png"
            else:
                save_path = Path(save_path)

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ 密度散点图已保存到: {save_path}")

            return {
                'mae': mae,
                'rmse': rmse,
                'r': r_value,
                'bias': bias,
                'n_samples': len(targets),
                'plot_path': str(save_path)
            }

        except Exception as e:
            print(f"绘制密度散点图失败: {e}")
            import traceback
            traceback.print_exc()

    def run_variable_level_analysis(self, dataloader, num_samples=300, output_dir=None):
        """
        变量级别的重要性分析 - 完整版，包含所有点特征
        """
        print("\n" + "=" * 60)
        print("运行变量级别重要性分析（包含哨兵1和SMAP）")
        print("=" * 60)

        import torch
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path
        import matplotlib
        from datetime import datetime

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

            # 2. 确定LS波段数（从数据中推断）
            # 点特征的顺序应该是: LS波段 + S1_VV + S1_VH + SMAP_TBV + SMAP_TBH + lon + lat + doy
            total_point_features = point_data.shape[1]
            print(f"\n推断点特征组成:")
            print(f"  总点特征数: {total_point_features}")

            # 固定特征数: S1_VV, S1_VH, SMAP_TBV, SMAP_TBH, lon, lat, doy = 7个
            fixed_features = 7
            ls_bands = total_point_features - fixed_features

            print(f"  推断LS波段数: {ls_bands}")
            print(f"  固定特征: 哨兵1(VV, VH) + SMAP(TBV, TBH) + 空间(经度, 纬度) + 时间(年积日) = 7个")

            # 3. 定义原始变量 - 完整版本
            # 卷积变量
            conv_variables = [
                "风场(chelsa_sfxwind)",
                "地表温度(lst)",
                "相对湿度(rh)",
                "积雪日数(clamday)",
                "DEM均值",
                "DEM标准差"
            ]

            # 点变量 - 根据推断的结构
            point_variables = []

            # LS特征
            for i in range(ls_bands):
                point_variables.append(f"土地覆盖_波段{i + 1}")

            # 微波特征
            point_variables.append("哨兵1_VV后向散射")
            point_variables.append("哨兵1_VH后向散射")
            point_variables.append("SMAP_TBV亮温")
            point_variables.append("SMAP_TBH亮温")

            # 空间和时间特征
            point_variables.append("经度")
            point_variables.append("纬度")
            point_variables.append("年积日")

            all_variables = conv_variables + point_variables

            # 验证变量数量匹配
            expected_total = len(conv_variables) + len(point_variables)
            actual_total = len(conv_variables) + total_point_features

            print(f"\n变量定义验证:")
            print(f"  定义的特征数: {expected_total}")
            print(f"  实际特征数: {actual_total}")

            if expected_total != actual_total:
                print(f"  警告: 特征数量不匹配! 请检查点特征结构")
                # 如果数量不匹配，使用通用命名
                if total_point_features != len(point_variables):
                    point_variables = [f"点特征_{i + 1}" for i in range(total_point_features)]
                    print(f"  已调整为通用命名: {len(point_variables)}个点特征")
                    all_variables = conv_variables + point_variables

            print(f"\n分析的变量 ({len(all_variables)}个):")

            # 分组显示变量
            print("  卷积变量 (6个):")
            for i, var in enumerate(conv_variables):
                print(f"    {i + 1:2d}. {var}")

            print("\n  点特征:")

            # 先创建主要分组
            main_groups = {}

            # 填充分组
            for var in point_variables:
                if "土地覆盖" in var:
                    if "土地覆盖" not in main_groups:
                        main_groups["土地覆盖"] = []
                    main_groups["土地覆盖"].append(var)
                elif "哨兵" in var or "SMAP" in var:
                    if "微波遥感" not in main_groups:
                        main_groups["微波遥感"] = []
                    main_groups["微波遥感"].append(var)
                elif "经度" in var or "纬度" in var:
                    if "空间位置" not in main_groups:
                        main_groups["空间位置"] = []
                    main_groups["空间位置"].append(var)
                elif "年积日" in var:
                    if "时间信息" not in main_groups:
                        main_groups["时间信息"] = []
                    main_groups["时间信息"].append(var)
                else:
                    if "其他" not in main_groups:
                        main_groups["其他"] = []
                    main_groups["其他"].append(var)

            idx_offset = len(conv_variables) + 1
            for group_name, features in main_groups.items():
                if features:
                    print(f"    {group_name} ({len(features)}个):")
                    for i, var in enumerate(features):
                        var_idx = point_variables.index(var) + idx_offset
                        print(f"      {var_idx:2d}. {var}")

            # 4. 获取设备
            device = next(self.model.parameters()).device
            print(f"\n设备: {device}")

            # 5. 计算基准预测
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
            baseline_mae = np.mean(np.abs(baseline_predictions.flatten() - target_data.flatten()))

            print(f"  基准MSE: {baseline_mse:.6f}")
            print(f"  基准RMSE: {baseline_rmse:.6f}")
            print(f"  基准MAE: {baseline_mae:.6f}")

            # 6. 计算变量级别的重要性
            print("\n2. 计算变量重要性...")

            n_test_samples = min(100, len(conv_data))
            print(f"  使用 {n_test_samples} 个样本进行重要性评估")

            variable_importance = {}

            # 准备基础数据
            conv_base = conv_data[:n_test_samples].copy()
            point_base = point_data[:n_test_samples].copy()
            target_base = target_data[:n_test_samples].copy()

            print("\n  方法: 变量置零法")
            print("  " + "-" * 50)

            for var_idx, var_name in enumerate(all_variables):
                print(f"    处理变量: {var_name}")

                start_time = datetime.now()

                if var_idx < len(conv_variables):  # 卷积变量
                    conv_idx = var_idx

                    # 创建置零版本
                    conv_zeroed = conv_base.copy()
                    conv_zeroed[:, conv_idx, :, :] = 0  # 将该变量所有像素置零

                    # 预测
                    conv_tensor = torch.FloatTensor(conv_zeroed).to(device)
                    point_tensor = torch.FloatTensor(point_base).to(device)

                    with torch.no_grad():
                        zeroed_preds = self.model(conv_tensor, point_tensor).cpu().numpy()

                    # 计算性能
                    zeroed_mse = np.mean((zeroed_preds.flatten() - target_base.flatten()) ** 2)

                else:  # 点变量
                    point_idx = var_idx - len(conv_variables)

                    # 创建置零版本
                    point_zeroed = point_base.copy()
                    point_zeroed[:, point_idx] = 0  # 将该点特征置零

                    # 预测
                    conv_tensor = torch.FloatTensor(conv_base).to(device)
                    point_tensor = torch.FloatTensor(point_zeroed).to(device)

                    with torch.no_grad():
                        zeroed_preds = self.model(conv_tensor, point_tensor).cpu().numpy()

                    # 计算性能
                    zeroed_mse = np.mean((zeroed_preds.flatten() - target_base.flatten()) ** 2)

                # 重要性 = 性能变化百分比
                importance = (zeroed_mse - baseline_mse) / baseline_mse * 100

                # 确定变量类型
                if var_idx < len(conv_variables):
                    var_type = '卷积'
                elif "土地覆盖" in var_name:
                    var_type = '土地覆盖'
                elif "哨兵" in var_name or "SMAP" in var_name:
                    var_type = '微波遥感'
                elif "经度" in var_name or "纬度" in var_name:
                    var_type = '空间位置'
                elif "年积日" in var_name:
                    var_type = '时间信息'
                else:
                    var_type = '点特征'

                variable_importance[var_name] = {
                    'importance': importance,
                    'zeroed_mse': zeroed_mse,
                    'zeroed_rmse': np.sqrt(zeroed_mse),
                    'mse_change': zeroed_mse - baseline_mse,
                    'type': var_type
                }

                elapsed_time = (datetime.now() - start_time).total_seconds()
                print(f"      置零后MSE: {zeroed_mse:.6f} (变化: {zeroed_mse - baseline_mse:+.6f})")
                print(f"      重要性: {importance:+.2f}% | 耗时: {elapsed_time:.2f}秒")

            # 7. 创建结果DataFrame
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

            # 8. 打印详细结果
            print("\n" + "=" * 80)
            print("变量重要性分析结果")
            print("=" * 80)
            print(f"基准性能: MSE={baseline_mse:.6f}, RMSE={baseline_rmse:.6f}, MAE={baseline_mae:.6f}")
            print(f"测试样本数: {n_test_samples}")
            print(f"总变量数: {len(all_variables)} (卷积: {len(conv_variables)}, 点特征: {len(point_variables)})")
            print("-" * 80)

            # 按类型分组显示
            type_groups = importance_df.groupby('类型')
            print("\n按类型分组的结果:")
            for type_name, group_df in type_groups:
                print(f"\n  {type_name} ({len(group_df)}个变量):")
                for idx, row in group_df.iterrows():
                    effect = "损害" if row['重要性(%)'] > 0 else "改善"
                    sign = "+" if row['重要性(%)'] > 0 else ""
                    print(f"    {row['变量']:25s}: {sign}{row['重要性(%)']:+.1f}% ({effect})")

            # 9. 可视化
            print("\n3. 生成可视化图表...")

            # 图1：总体变量重要性条形图
            plt.figure(figsize=(14, 10))

            # 选择Top 20最重要的变量
            top_n = min(20, len(importance_df))
            top_df = importance_df.head(top_n).copy()
            top_df = top_df.sort_values('重要性(%)', ascending=True)

            # 为不同类型设置不同颜色
            color_map = {
                '卷积': 'steelblue',
                '土地覆盖': 'forestgreen',
                '微波遥感': 'darkorange',
                '空间位置': 'purple',
                '时间信息': 'brown',
                '点特征': 'gray'
            }

            colors = [color_map.get(row['类型'], 'gray') for _, row in top_df.iterrows()]
            bars = plt.barh(range(top_n), top_df['重要性(%)'], color=colors, alpha=0.8, edgecolor='black')

            # 添加数值标签
            for i, (bar, row) in enumerate(zip(bars, top_df.itertuples())):
                color = 'black' if abs(row.重要性) > 1 else 'gray'
                plt.text(row.重要性, i, f' {row.重要性:+.1f}%', va='center',
                         fontsize=9, fontweight='bold', color=color)

            plt.yticks(range(top_n), top_df['变量'], fontsize=10)
            plt.xlabel('重要性 (%)', fontsize=14, fontweight='bold')
            plt.title(f'Top {top_n} 变量重要性分析\n正值表示损害模型，负值表示改善模型',
                      fontsize=16, fontweight='bold', pad=20)
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            plt.grid(axis='x', alpha=0.3)

            # 添加图例
            from matplotlib.patches import Patch
            legend_patches = [Patch(color=color, label=type_name)
                              for type_name, color in color_map.items()
                              if type_name in importance_df['类型'].unique()]
            plt.legend(handles=legend_patches, loc='lower right', fontsize=10)

            plt.tight_layout()
            importance_plot_path = output_dir / "variable_importance_top20.png"
            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 图2：微波特征特别分析
            microwave_features = importance_df[importance_df['类型'] == '微波遥感']
            if len(microwave_features) > 0:
                plt.figure(figsize=(12, 6))

                microwave_df = microwave_features.copy().sort_values('重要性(%)', ascending=True)

                # 区分哨兵1和SMAP
                colors = []
                for var in microwave_df['变量']:
                    if '哨兵' in var:
                        colors.append('royalblue')
                    elif 'SMAP' in var:
                        colors.append('coral')
                    else:
                        colors.append('gray')

                bars = plt.barh(range(len(microwave_df)), microwave_df['重要性(%)'],
                                color=colors, alpha=0.8, edgecolor='black')

                for i, (bar, row) in enumerate(zip(bars, microwave_df.itertuples())):
                    plt.text(row.重要性, i, f' {row.重要性:+.1f}%', va='center',
                             fontsize=10, fontweight='bold', color='black')

                plt.yticks(range(len(microwave_df)), microwave_df['变量'], fontsize=11)
                plt.xlabel('重要性 (%)', fontsize=12, fontweight='bold')
                plt.title('微波遥感特征重要性分析', fontsize=14, fontweight='bold')
                plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                plt.grid(axis='x', alpha=0.3)

                # 添加图例
                s1_patch = Patch(color='royalblue', label='哨兵1后向散射')
                smap_patch = Patch(color='coral', label='SMAP亮温')
                plt.legend(handles=[s1_patch, smap_patch], loc='lower right')

                plt.tight_layout()
                microwave_plot_path = output_dir / "microwave_features_importance.png"
                plt.savefig(microwave_plot_path, dpi=300, bbox_inches='tight')
                plt.close()

            # 图3：按类型分组的重要性箱线图
            plt.figure(figsize=(12, 8))

            type_data = []
            type_labels = []
            for type_name, group_df in importance_df.groupby('类型'):
                if len(group_df) > 1:  # 只有至少2个变量才绘制箱线图
                    type_data.append(group_df['重要性(%)'].values)
                    type_labels.append(f"{type_name}\n({len(group_df)})")

            if type_data:
                bp = plt.boxplot(type_data, labels=type_labels, patch_artist=True)

                # 设置颜色
                colors = plt.cm.Set3(np.linspace(0, 1, len(type_data)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.ylabel('重要性 (%)', fontsize=12, fontweight='bold')
                plt.title('各类型变量重要性分布', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')

                plt.tight_layout()
                boxplot_path = output_dir / "importance_by_type_boxplot.png"
                plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
                plt.close()

            # 10. 保存结果
            csv_path = output_dir / "variable_importance_results.csv"
            importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 11. 生成详细分析报告
            report_path = output_dir / "analysis_summary.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SWE模型变量重要性分析报告（包含哨兵1和SMAP）\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"测试样本数: {n_test_samples}\n")
                f.write(f"基准MSE: {baseline_mse:.6f}\n")
                f.write(f"基准RMSE: {baseline_rmse:.6f}\n")
                f.write(f"基准MAE: {baseline_mae:.6f}\n\n")

                f.write("变量统计:\n")
                f.write("-" * 40 + "\n")
                for type_name, group_df in importance_df.groupby('类型'):
                    avg_importance = group_df['重要性(%)'].mean()
                    f.write(f"{type_name:10s}: {len(group_df):2d}个变量，平均重要性: {avg_importance:+.1f}%\n")

                f.write("\n核心发现:\n")
                f.write("-" * 40 + "\n")

                # Top 3最重要的变量
                top_3 = importance_df.head(3)
                for i, (_, row) in enumerate(top_3.iterrows()):
                    if row['重要性(%)'] > 0:
                        f.write(
                            f"{i + 1}. {row['变量']} 是最损害模型的变量，置零会使MSE增加{abs(row['重要性(%)']):.1f}%\n")
                    else:
                        f.write(
                            f"{i + 1}. {row['变量']} 是最改善模型的变量，置零会使MSE减少{abs(row['重要性(%)']):.1f}%\n")

                # 微波特征分析
                microwave_df = importance_df[importance_df['类型'] == '微波遥感']
                if len(microwave_df) > 0:
                    f.write("\n微波特征表现:\n")
                    f.write("-" * 40 + "\n")
                    for _, row in microwave_df.iterrows():
                        effect = "损害" if row['重要性(%)'] > 0 else "改善"
                        f.write(f"{row['变量']}: {row['重要性(%)']:+.1f}% ({effect})\n")

                f.write("\n详细结果:\n")
                f.write("-" * 40 + "\n")
                for _, row in importance_df.iterrows():
                    direction = "增加" if row['重要性(%)'] > 0 else "减少"
                    f.write(f"{row['变量']:25s}: 置零会使MSE {direction} {abs(row['重要性(%)']):.1f}%\n")

            print(f"\n✓ 变量级别重要性分析完成!")
            print(f"结果保存在: {output_dir}")
            print(f"\n主要文件:")
            print(f"  {csv_path} - 详细结果表 ({len(importance_df)}个变量)")
            print(f"  {importance_plot_path} - Top 20变量重要性图")
            if len(microwave_features) > 0:
                print(f"  {microwave_plot_path} - 微波特征特别分析")
            print(f"  {report_path} - 详细分析报告")

            # 打印关键结论
            print(f"\n关键结论:")

            # 最损害模型的变量
            most_harmful = importance_df.iloc[0]
            print(f"  1. 最损害模型的变量: {most_harmful['变量']} (+{most_harmful['重要性(%)']:.1f}%)")

            # 最改善模型的变量
            most_helpful = importance_df[importance_df['重要性(%)'] < 0]
            if len(most_helpful) > 0:
                most_helpful = most_helpful.iloc[0]
                print(f"  2. 最改善模型的变量: {most_helpful['变量']} ({most_helpful['重要性(%)']:.1f}%)")

            # 微波特征总结
            if len(microwave_features) > 0:
                print(f"\n  微波特征总结:")
                avg_microwave = microwave_features['重要性(%)'].mean()
                overall_effect = "正影响" if avg_microwave > 0 else "负影响"
                print(f"    平均重要性: {avg_microwave:+.1f}% ({overall_effect})")

                # SMAP特别分析
                smap_features = microwave_features[microwave_features['变量'].str.contains('SMAP')]
                if len(smap_features) > 0:
                    avg_smap = smap_features['重要性(%)'].mean()
                    print(f"    SMAP平均: {avg_smap:+.1f}%")
                    for _, row in smap_features.iterrows():
                        effect = "损害" if row['重要性(%)'] > 0 else "改善"
                        print(f"      {row['变量']}: {row['重要性(%)']:+.1f}% ({effect})")

            return {
                'variable_importance': importance_df,
                'baseline_mse': baseline_mse,
                'baseline_rmse': baseline_rmse,
                'baseline_mae': baseline_mae,
                'n_test_samples': n_test_samples,
                'output_dir': output_dir
            }

        except Exception as e:
            print(f"✗ 变量级别重要性分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_shap_analysis(self, num_samples=500):
        """
        运行特征重要性分析 - 变量级别
        """
        print("\n" + "=" * 60)
        print("运行变量级别重要性分析")
        print("=" * 60)

        try:
            # 导入分析器
            try:
                from shap_analyzer import SHAPAnalyzer
            except ImportError:
                print("✗ 无法导入分析器")
                return None

            # 确保模型和数据加载器存在
            if self.model is None:
                print("✗ 模型未加载")
                return None

            if self.val_loader is None:
                print("✗ 验证数据加载器不存在")
                return None

            # 创建输出目录
            output_dir = self.save_dir / "variable_importance"
            output_dir.mkdir(exist_ok=True)

            # 创建分析器
            analyzer = SHAPAnalyzer(self.model)

            # 运行变量级别分析
            results = analyzer.run_variable_level_analysis(
                dataloader=self.val_loader,
                num_samples=min(num_samples, 300),
                output_dir=output_dir
            )

            return results

        except Exception as e:
            print(f"✗ 分析失败: {e}")
            return None

    def plot_predictions(self, predictions, targets):
        """绘制简单预测散点图"""
        print(f"\n[强制调试] plot_predictions 被调用!")
        print(f"预测值长度: {len(predictions) if hasattr(predictions, '__len__') else '无'}")
        print(f"目标值长度: {len(targets) if hasattr(targets, '__len__') else '无'}")

        # 检查数据是否有效
        if predictions is None or targets is None:
            print("[错误] 预测值或目标值为 None!")
            return

        if len(predictions) == 0 or len(targets) == 0:
            print("[错误] 预测值或目标值为空!")
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()

            mask = ~np.isnan(predictions) & ~np.isnan(targets)
            predictions = predictions[mask]
            targets = targets[mask]

            plt.figure(figsize=(10, 8))

            # 散点图
            plt.scatter(targets, predictions, alpha=0.6, s=20,
                        color='blue', edgecolors='black', linewidth=0.5)

            # 1:1线
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val],
                     'r--', linewidth=2, label='1:1线')

            # 计算指标
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            bias = np.mean(predictions - targets)

            try:
                from scipy import stats
                r_value, _ = stats.pearsonr(predictions, targets)
            except:
                r_value = np.corrcoef(predictions, targets)[0, 1]

            # 添加指标文本
            text_str = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR = {r_value:.4f}\nBias = {bias:.4f}'
            plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            plt.xlabel('真实值', fontsize=14)
            plt.ylabel('预测值', fontsize=14)
            plt.title('SWE预测结果 - 简单散点图', fontsize=16, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            save_path = self.save_dir / "simple_scatter.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ 简单散点图已保存到: {save_path}")

        except Exception as e:
            print(f"绘制简单散点图失败: {e}")

    def run_ablation_experiment(self, model_path=None, use_all_samples=True, max_samples=None,
                                output_dir=None, ablation_method='retrain', retrain_epochs=50):
        """
        运行消融实验

        Args:
            model_path: 模型文件路径（可选）
            use_all_samples: 是否使用全部验证集样本（推荐True）
            max_samples: 最大样本数（当use_all_samples=False时使用）
            output_dir: 输出目录（可选）
            ablation_method: 消融实验方法 'retrain'（重新训练）或 'zeroing'（特征置零）
            retrain_epochs: 重新训练的轮次（仅当ablation_method='retrain'时有效）
        """
        print("\n" + "=" * 60)
        print(f"运行消融实验（{ablation_method}方法）")
        print("=" * 60)

        try:
            # 导入消融实验模块
            try:
                from ablation_study import AblationStudy
            except ImportError as e:
                print(f"✗ 导入消融实验模块失败: {e}")
                print("请确保 ablation_study.py 在相同目录下")
                return None

            # 如果提供了模型路径，加载该模型
            if model_path is not None and os.path.exists(model_path):
                print(f"加载指定模型: {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    model_state_dict = checkpoint.get('model_state_dict', {})

                    if model_state_dict:
                        self.model.load_state_dict(model_state_dict)
                        print("✓ 模型权重加载成功")

                        # 打印模型信息
                        epoch = checkpoint.get('epoch', '未知')
                        val_loss = checkpoint.get('metrics', {}).get('loss', '未知')
                        print(f"  训练轮次: {epoch}")
                        print(f"  验证损失: {val_loss:.6f}")
                except Exception as e:
                    print(f"模型加载失败，使用当前模型: {e}")

            # 检查数据加载器
            if self.val_loader is None:
                print("✗ 验证数据加载器不存在")
                return None

            # 获取数据集信息
            total_val_samples = len(self.val_loader.dataset)
            print(f"验证集总样本数: {total_val_samples:,}")

            # 检查训练数据（仅重新训练方法需要）
            if ablation_method == 'retrain':
                if self.train_loader is None:
                    print("✗ 重新训练方法需要训练数据加载器，但train_loader不存在")
                    return None
                total_train_samples = len(self.train_loader.dataset)
                print(f"训练集总样本数: {total_train_samples:,}")

            # 决定使用多少评估样本
            if use_all_samples:
                n_samples = min(total_val_samples, 10000)  # 限制最大评估样本数，避免内存问题
                print(f"使用 {n_samples:,} 个样本进行评估（最多10,000个）")
            elif max_samples:
                n_samples = min(max_samples, total_val_samples)
                print(f"使用 {n_samples:,} 个样本进行评估（最多{max_samples}个）")
            else:
                n_samples = min(5000, total_val_samples)  # 默认使用5000个样本
                print(f"使用 {n_samples:,} 个样本进行评估（默认）")

            if n_samples < 100:
                print(f"⚠ 警告: 评估样本数较少 ({n_samples})，结果可能不够可靠")

            # 创建消融实验分析器
            print("创建消融实验分析器...")
            analyzer = AblationStudy(self.model, self.device)

            # 设置输出目录
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                method_name = "retrain" if ablation_method == 'retrain' else "zeroing"
                ablation_output_dir = self.save_dir / f"ablation_{method_name}_{timestamp}"
            else:
                ablation_output_dir = Path(output_dir)

            ablation_output_dir.mkdir(parents=True, exist_ok=True)

            # 保存实验配置
            config = {
                'ablation_method': ablation_method,
                'retrain_epochs': retrain_epochs if ablation_method == 'retrain' else None,
                'evaluation_samples': n_samples,
                'total_train_samples': total_train_samples if ablation_method == 'retrain' else None,
                'total_val_samples': total_val_samples,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(ablation_output_dir / "ablation_config.json", 'w') as f:
                json.dump(config, f, indent=2)

            # 运行消融实验
            print(f"\n开始消融实验...")
            print(f"  方法: {ablation_method}")
            if ablation_method == 'retrain':
                print(f"  重新训练轮次: {retrain_epochs}")
            print(f"  评估样本数: {n_samples:,}")

            start_time = datetime.now()

            if ablation_method == 'retrain':
                # 重新训练方法
                results = analyzer.run_ablation_study(
                    dataloader=self.val_loader,
                    n_samples=n_samples,
                    output_dir=ablation_output_dir,
                    train_loader=self.train_loader,  # 传递训练数据
                    val_loader=self.val_loader,  # 传递验证数据（用于重新训练时的验证）
                    epochs=retrain_epochs  # 重新训练轮次
                )
            else:
                # 特征置零方法（原方法）
                results = analyzer.run_ablation_study(
                    dataloader=self.val_loader,
                    n_samples=n_samples,
                    output_dir=ablation_output_dir
                )

            elapsed_time = (datetime.now() - start_time).total_seconds()
            print(f"\n消融实验总耗时: {elapsed_time:.1f} 秒 ({elapsed_time / 60:.1f} 分钟)")

            if results:
                print(f"\n✓ 消融实验完成!")
                print(f"结果保存在: {results['output_dir']}")

                # 打印关键结论
                self._print_ablation_summary(results)

                # 保存额外信息
                results['config'] = config
                results['elapsed_time'] = elapsed_time

                # 保存完整结果
                results_path = ablation_output_dir / "full_results.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"完整结果已保存: {results_path}")

            return results

        except Exception as e:
            print(f"✗ 消融实验失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _print_ablation_summary(self, results):
        """打印消融实验摘要"""
        try:
            results_df = results['results_df']

            if len(results_df) > 1:
                print(f"\n消融实验关键发现:")
                print("-" * 50)

                # 找出MSE变化最大的前3个（损害性能）
                sorted_df = results_df[results_df['组合名称'] != '基准模型（所有特征）'].copy()
                if len(sorted_df) > 0:
                    sorted_df = sorted_df.sort_values('MSE变化百分比', ascending=False)

                    print("1. 最重要的特征组（移除后损害最大）:")
                    for i, (_, row) in enumerate(sorted_df.head(3).iterrows()):
                        if row['MSE变化百分比'] > 0:
                            print(f"   {row['组合名称']}: +{row['MSE变化百分比']:.1f}%")
                            print(f"     说明: {row.get('描述', '')}")

                    # 找出可能冗余的特征（改善性能）
                    improving_features = sorted_df[sorted_df['MSE变化百分比'] < 0]
                    if len(improving_features) > 0:
                        print(f"\n2. 可能冗余的特征组（移除后改善性能）:")
                        for i, (_, row) in enumerate(improving_features.head(3).iterrows()):
                            print(f"   {row['组合名称']}: {row['MSE变化百分比']:.1f}%")
                            print(f"     说明: {row.get('描述', '')}")

                    # SMAP特征特别分析
                    smap_features = sorted_df[sorted_df['组合名称'].str.contains('SMAP')]
                    if len(smap_features) > 0:
                        print(f"\n3. SMAP特征重要性分析:")
                        for _, row in smap_features.iterrows():
                            effect = "很重要" if row['MSE变化百分比'] > 10 else "重要" if row[
                                                                                              'MSE变化百分比'] > 5 else "一般"
                            if row['MSE变化百分比'] > 0:
                                print(f"   {row['组合名称']}: +{row['MSE变化百分比']:.1f}% ({effect})")
                            else:
                                print(f"   {row['组合名称']}: {row['MSE变化百分比']:.1f}% (可能冗余)")

                    print(f"\n统计信息:")
                    print(f"  测试特征组合数: {len(sorted_df)}")
                    print(f"  平均MSE变化: {sorted_df['MSE变化百分比'].mean():+.1f}%")
                    print(f"  最大损害: {sorted_df['MSE变化百分比'].max():+.1f}%")
                    print(f"  最大改善: {sorted_df['MSE变化百分比'].min():+.1f}%")

        except Exception as e:
            print(f"打印摘要失败: {e}")




if __name__ == "__main__":
    """主函数"""
    print("=" * 70)
    print("SWE反演模型训练系统")
    print("=" * 70)

    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(description='训练SWE反演模型')

    # 模式选择
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'test', 'all', 'ablation'],
                        help='运行模式: train=训练, evaluate=评估, test=测试, all=完整流程, ablation=消融实验')

    # 模型类型
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'spatial_only', 'point_only'],
                        help='模型类型')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=3, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')

    # 数据划分方式
    parser.add_argument('--split_method', type=str, default='temporal',
                        choices=['random', 'temporal', 'spatial'],
                        help='数据划分方法')

    parser.add_argument('--train_year', type=int, default=2015, help='训练年份')
    parser.add_argument('--val_year', type=int, default=2016, help='验证年份')

    # 路径参数
    parser.add_argument('--save_dir', type=str, default='./experiments', help='保存目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')

    # 消融实验参数
    parser.add_argument('--run_ablation', action='store_true',
                        help='运行消融实验（在评估时运行）')
    parser.add_argument('--ablation_samples', type=int, default=None,
                        help='消融实验使用的样本数（None表示使用全部验证集）')
    parser.add_argument('--ablation_max_samples', type=int, default=None,
                        help='消融实验最大样本数')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型文件路径（用于消融实验）')

    parser.add_argument('--ablation_method', type=str, default='retrain',
                        choices=['retrain', 'zeroing'],
                        help='消融实验方法: retrain=重新训练, zeroing=特征置零')

    parser.add_argument('--retrain_epochs', type=int, default=50,
                        help='重新训练的轮次（仅当ablation_method=retrain时有效）')

    args = parser.parse_args()

    # 创建基础配置
    config = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'd_model': args.d_model,
        'save_dir': args.save_dir,
        'experiment_name': args.exp_name,

        # 数据划分配置
        'split_method': args.split_method,
        'train_year': args.train_year,
        'val_year': args.val_year,

        # 其他固定配置
        'val_ratio': 0.2,
        'num_workers': 0,
        'weight_decay': 1e-5,
        'patience': 15,
        'seed': 42,
        'clip_grad': 1.0,
        'save_freq': 10,
    }

    # 设置实验名称
    if config['experiment_name'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment_name'] = f"swe_{args.model_type}_{args.split_method}_{timestamp}"

    print(f"\n完整配置:")
    for key, value in config.items():
        if key not in ['experiment_name']:
            print(f"  {key:20s}: {value}")

    print(f"  实验名称: {config['experiment_name']}")

    # 创建训练器
    trainer = SWETrainer(config)

    # 根据模式执行
    if args.mode == 'train':
        # 只训练
        if trainer.load_data():
            if trainer.build_model():
                trainer.train()


    elif args.mode == 'evaluate':
        # 只评估
        if trainer.load_data():
            if trainer.build_model():
                trainer.evaluate(
                    model_path=args.model_path,
                    run_ablation=args.run_ablation,
                    ablation_samples=args.ablation_samples,
                    ablation_method=args.ablation_method,
                    retrain_epochs=args.retrain_epochs
                )



    elif args.mode == 'test':
        # 测试模式
        print("\n测试模式...")
        print("1. 测试模型结构...")
        try:
            test_model()
        except Exception as e:
            print(f"模型测试失败: {e}")
            print("继续...")

        print("\n2. 测试数据加载...")
        if trainer.load_data():
            print("\n3. 测试模型构建...")
            if trainer.build_model():
                print("\n✓ 所有测试通过!")



    elif args.mode == 'all':
        # 完整流程
        if trainer.load_data():
            if trainer.build_model():
                trainer.train()
                trainer.evaluate(
                    model_path=args.model_path,
                    run_ablation=args.run_ablation,
                    ablation_samples=args.ablation_samples,
                    ablation_method=args.ablation_method,
                    retrain_epochs=args.retrain_epochs
                )



    elif args.mode == 'ablation':
        # 消融实验模式
        print("\n消融实验模式...")

        # 如果提供了模型路径，直接使用
        model_path_to_use = args.model_path

        # 如果没有提供模型路径，尝试使用最佳模型
        if model_path_to_use is None:
            # 检查是否有训练过的模型
            possible_paths = [
                trainer.save_dir / "best_model.pth",
                trainer.save_dir / "final_model.pth",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path_to_use = path
                    print(f"找到模型文件: {model_path_to_use}")
                    break

        if model_path_to_use is None:
            print("警告: 没有找到现有模型，将使用随机初始化的模型")

        # 决定使用多少样本
        if args.ablation_samples is not None:
            use_all_samples = False
            max_samples = args.ablation_samples
        elif args.ablation_max_samples is not None:
            use_all_samples = False
            max_samples = args.ablation_max_samples
        else:
            # 默认使用全部样本
            use_all_samples = True
            max_samples = None

        # 运行消融实验
        if trainer.load_data():
            if trainer.build_model():
                ablation_results = trainer.run_ablation_experiment(
                    model_path=model_path_to_use,
                    use_all_samples=use_all_samples,
                    max_samples=max_samples,
                    output_dir=trainer.save_dir / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    ablation_method=args.ablation_method,  # 从命令行参数获取
                    retrain_epochs=args.retrain_epochs  # 从命令行参数获取
                )

                if ablation_results:
                    print("\n✓ 消融实验完成!")
                else:
                    print("\n✗ 消融实验失败")

    print("\n" + "=" * 70)
    print("程序执行完成!")
    print("=" * 70)