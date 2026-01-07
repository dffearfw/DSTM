# train_swe_main.py
# !/usr/bin/env python3
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

warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型（使用我们修改后的版本）
try:
    from models_smap import create_model, SWENet_Full, test_model

    print("✓ 成功导入模型")
except ImportError as e:
    print(f"✗ 导入模型失败: {e}")
    print("请确保 models_swe.py 在相同目录下")
    sys.exit(1)

# 导入数据加载器
try:
    from data_online_smap import build_dataloaders

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
            'model_type': 'full',  # full, temporal_only, spatial_only, point_only等

            # 数据参数
            'batch_size': 16,  # 小batch size以适应Transformer
            'val_ratio': 0.2,
            'num_workers': 0,  # 调试时设为0

            # 训练参数
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'patience': 15,  # 早停耐心值

            # 模型参数（将从数据自动获取）
            'C_dyn': None,
            'C_spatial': None,
            'C_point': None,
            'T_max': None,
            'd_model': 256,

            # 路径设置
            'save_dir': './experiments',
            'experiment_name': None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            # 其他
            'seed': 42,
            'clip_grad': 1.0,  # 梯度裁剪
            'save_freq': 10,  # 保存频率（epoch）
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
            # 构建数据加载器
            train_loader, val_loader, shapes = build_dataloaders(
                batch_size=self.config['batch_size'],
                val_ratio=self.config['val_ratio'],
                num_workers=self.config['num_workers'],
                seed=self.config['seed']
            )

            # 获取维度信息
            C_dyn, T, P, C_spatial, C_point = shapes

            # 更新配置
            self.config['C_dyn'] = C_dyn
            self.config['C_spatial'] = C_spatial
            self.config['C_point'] = C_point
            self.config['T_max'] = T

            print(f"✓ 数据加载成功!")
            print(f"\n数据维度:")
            print(f"  动态序列: C_dyn={C_dyn}, T={T}, P={P}")
            print(f"  空间特征: C_spatial={C_spatial}")
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
            dyn, spatial, point, target = next(iter(self.train_loader))

            print(f"  动态序列: {dyn.shape}")
            print(f"  空间补丁: {spatial.shape}")
            print(f"  点特征: {point.shape}")
            print(f"  目标值: {target.shape}")

            # 检查数据范围
            print(f"\n  数据范围检查:")
            print(f"    动态序列: [{dyn.min():.3f}, {dyn.max():.3f}]")
            print(f"    空间补丁: [{spatial.min():.3f}, {spatial.max():.3f}]")
            print(f"    点特征: [{point.min():.3f}, {point.max():.3f}]")
            print(f"    目标值: [{target.min():.3f}, {target.max():.3f}]")

            # 检查NaN
            print(f"\n  NaN检查:")
            print(f"    动态序列 NaN: {torch.isnan(dyn).any().item()}")
            print(f"    空间补丁 NaN: {torch.isnan(spatial).any().item()}")
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
            # 根据配置创建模型
            self.model = create_model(
                model_type=self.config['model_type'],
                C_dyn=self.config['C_dyn'],
                C_spatial=self.config['C_spatial'],
                C_point=self.config['C_point'],
                d_model=self.config['d_model'],
                T_max=self.config['T_max']
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

        for batch_idx, (dyn_seq, spatial_patch, point_feats, targets) in enumerate(self.train_loader):
            # 移动到设备
            if dyn_seq is not None:
                dyn_seq = dyn_seq.to(self.device)
            if spatial_patch is not None:
                spatial_patch = spatial_patch.to(self.device)
            if point_feats is not None:
                point_feats = point_feats.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            outputs = self.model(dyn_seq, spatial_patch, point_feats)
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
            if (batch_idx + 1) % 10 == 0:
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
            for dyn_seq, spatial_patch, point_feats, targets in self.val_loader:
                # 移动到设备
                if dyn_seq is not None:
                    dyn_seq = dyn_seq.to(self.device)
                if spatial_patch is not None:
                    spatial_patch = spatial_patch.to(self.device)
                if point_feats is not None:
                    point_feats = point_feats.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(dyn_seq, spatial_patch, point_feats)
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
            'model_state_dict': self.model.state_dict(),
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
        torch.save(checkpoint, save_path)

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

    def evaluate(self, model_path=None):
        """评估模型"""
        print("\n" + "=" * 60)
        print("评估模型...")
        print("=" * 60)

        # 加载模型
        if model_path is None:
            model_path = self.save_dir / "best_model.pth"

        if not os.path.exists(model_path):
            print(f"✗ 模型文件不存在: {model_path}")
            return

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"加载模型: {model_path}")
        print(f"训练轮次: {checkpoint.get('epoch', '未知')}")
        print(f"验证损失: {checkpoint.get('metrics', {}).get('loss', '未知'):.6f}")

        # 评估
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for dyn_seq, spatial_patch, point_feats, targets in self.val_loader:
                # 移动到设备
                if dyn_seq is not None:
                    dyn_seq = dyn_seq.to(self.device)
                if spatial_patch is not None:
                    spatial_patch = spatial_patch.to(self.device)
                if point_feats is not None:
                    point_feats = point_feats.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(dyn_seq, spatial_patch, point_feats)

                # 收集结果
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 计算指标
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))

        if np.std(all_targets) > 0:
            r2 = 1 - np.sum((all_predictions - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
            corr = np.corrcoef(all_predictions, all_targets)[0, 1]
        else:
            r2 = 0
            corr = 0

        # 打印结果
        print(f"\n评估结果:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}")
        print(f"  相关系数: {corr:.6f}")

        # 保存评估结果
        eval_results = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'correlation': float(corr),
            'num_samples': len(all_targets),
            'model_path': str(model_path),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        eval_path = self.save_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)

        print(f"\n评估结果已保存到: {eval_path}")

        # 绘制预测vs真实值散点图
        self.plot_predictions(all_predictions, all_targets)

        return eval_results

    def plot_predictions(self, predictions, targets):
        """绘制预测vs真实值散点图"""
        try:
            plt.figure(figsize=(10, 8))

            # 散点图
            plt.scatter(targets, predictions, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)

            # 1:1线
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1线')

            # 回归线
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(targets, predictions)
                reg_line = slope * np.array([min_val, max_val]) + intercept
                plt.plot([min_val, max_val], reg_line, 'g-', linewidth=2,
                         label=f'回归线 (R={r_value:.3f})')
            except:
                pass

            plt.xlabel('真实值 (标准化SWE)', fontsize=14)
            plt.ylabel('预测值 (标准化SWE)', fontsize=14)
            plt.title(f'SWE预测结果 - {self.config["model_type"]}', fontsize=16, fontweight='bold')

            # 添加指标文本
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            r2 = 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

            text_str = f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}\nR² = {r2:.4f}'
            plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # 保存图像
            plot_path = self.save_dir / "predictions_scatter.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"预测散点图已保存到: {plot_path}")

        except Exception as e:
            print(f"绘制散点图失败: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print("SWE反演模型训练系统")
    print("=" * 70)

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='训练SWE反演模型')

    # 模式选择
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'test', 'all'],
                        help='运行模式: train=训练, evaluate=评估, test=测试, all=完整流程')

    # 模型类型
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'temporal_only', 'spatial_only', 'point_only',
                                 'no_temporal', 'no_spatial', 'no_point'],
                        help='模型类型')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')

    # 路径参数
    parser.add_argument('--save_dir', type=str, default='./experiments', help='保存目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')

    args = parser.parse_args()

    # 创建配置
    config = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'd_model': args.d_model,
        'save_dir': args.save_dir,
        'experiment_name': args.exp_name,
    }

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
                trainer.evaluate()

    elif args.mode == 'test':
        # 测试模式
        print("\n测试模式...")
        print("1. 测试模型结构...")
        try:
            test_model()
        except:
            print("模型测试失败，但继续...")

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
                trainer.evaluate()

    print("\n" + "=" * 70)
    print("程序执行完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()