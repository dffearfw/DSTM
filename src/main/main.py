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

    def evaluate(self, model_path=None):
        """评估模型 - 增加密度图"""
        print("\n" + "=" * 60)
        print("评估模型...")
        print("=" * 60)

        # [调试信息]
        print("[调试] self.model 存在:", self.model is not None)
        print("[调试] self.val_loader 存在:", self.val_loader is not None)
        if self.val_loader is not None:
            print("[调试] 验证集样本数:", len(self.val_loader.dataset))

        # 加载模型
        if model_path is None:
            model_path = self.save_dir / "best_model.pth"

        if not os.path.exists(model_path):
            print(f"✗ 模型文件不存在: {model_path}")
            print("使用当前模型进行评估")
            # 如果没有保存的模型，使用当前模型进行评估
            if self.model is None:
                print("✗ 模型未构建，无法评估")
                return None
            else:
                print("✓ 使用当前内存中的模型进行评估")
                use_current_model = True
        else:
            print(f"✓ 找到模型文件: {model_path}")
            use_current_model = False

            try:
                # 尝试方法1：使用 weights_only=False（PyTorch 2.6+）
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                print("✓ 使用 weights_only=False 成功加载模型")
            except Exception as e1:
                print(f"方法1失败: {e1}")
                try:
                    # 尝试方法2：添加安全全局变量
                    import torch.serialization
                    import numpy.core.multiarray
                    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                    print("✓ 使用安全全局变量成功加载模型")
                except Exception as e2:
                    print(f"方法2失败: {e2}")
                    try:
                        # 尝试方法3：只加载模型权重（最安全）
                        checkpoint = torch.load(model_path, map_location=self.device)
                        print("✓ 使用传统方法成功加载模型")
                    except Exception as e3:
                        print(f"所有加载方法都失败: {e3}")
                        print("使用当前模型进行评估")
                        use_current_model = True

            if not use_current_model:
                try:
                    # 只加载模型状态字典
                    model_state_dict = checkpoint['model_state_dict']

                    # 检查模型架构是否匹配
                    current_model_keys = set(self.model.state_dict().keys())
                    loaded_model_keys = set(model_state_dict.keys())

                    if current_model_keys == loaded_model_keys:
                        self.model.load_state_dict(model_state_dict)
                        print("✓ 模型权重加载成功")
                        print(f"  训练轮次: {checkpoint.get('epoch', '未知')}")
                        print(f"  最佳验证损失: {checkpoint.get('metrics', {}).get('loss', '未知'):.6f}")
                    else:
                        print("⚠ 模型架构不匹配，使用当前模型")
                        print(f"  当前模型键: {len(current_model_keys)}")
                        print(f"  加载模型键: {len(loaded_model_keys)}")
                        print(f"  差异: {current_model_keys - loaded_model_keys}")
                        use_current_model = True
                except Exception as e:
                    print(f"模型权重加载失败: {e}")
                    use_current_model = True

        # 检查数据加载器
        if self.val_loader is None:
            print("✗ 验证数据加载器不存在")
            return None

        # 评估
        self.model.eval()

        all_predictions = []
        all_targets = []

        print("\n正在进行预测...")
        batch_count = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (conv_feats, point_feats, targets) in enumerate(self.val_loader):
                # 移动到设备
                conv_feats = conv_feats.to(self.device)
                point_feats = point_feats.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(conv_feats, point_feats)

                # 收集结果
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

                batch_count += 1
                total_samples += len(outputs)

                # 显示进度
                if (batch_idx + 1) % 10 == 0:
                    print(f"  已处理 {batch_idx + 1} 个批次, {total_samples} 个样本")

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        print(f"\n预测完成，共 {len(all_predictions)} 个样本")

        # [调试] 检查数据范围
        print("[调试] 预测值统计:")
        print(f"  范围: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
        print(f"  均值: {all_predictions.mean():.6f}, 标准差: {all_predictions.std():.6f}")

        print("[调试] 目标值统计:")
        print(f"  范围: [{all_targets.min():.6f}, {all_targets.max():.6f}]")
        print(f"  均值: {all_targets.mean():.6f}, 标准差: {all_targets.std():.6f}")

        # 移除NaN值
        mask = ~np.isnan(all_predictions) & ~np.isnan(all_targets)
        valid_predictions = all_predictions[mask]
        valid_targets = all_targets[mask]

        print(
            f"[调试] 有效样本数: {len(valid_predictions)} (移除 {len(all_predictions) - len(valid_predictions)} 个NaN)")

        if len(valid_predictions) == 0:
            print("✗ 没有有效数据用于评估")
            return None

        # 计算指标
        mse = np.mean((valid_predictions - valid_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(valid_predictions - valid_targets))
        bias = np.mean(valid_predictions - valid_targets)

        # 计算R²和相关系数
        if np.std(valid_targets) > 0:
            r2 = 1 - np.sum((valid_predictions - valid_targets) ** 2) / np.sum(
                (valid_targets - np.mean(valid_targets)) ** 2)
            try:
                from scipy import stats
                r_value, p_value = stats.pearsonr(valid_predictions, valid_targets)
            except Exception as e:
                print(f"[警告] scipy.pearsonr 失败: {e}")
                r_value = np.corrcoef(valid_predictions, valid_targets)[0, 1]
                p_value = None
        else:
            r2 = 0
            r_value = 0
            p_value = None

        # 打印结果
        print(f"\n评估结果:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  Bias: {bias:.6f}")
        print(f"  R²:   {r2:.6f}")
        print(f"  R:    {r_value:.6f}")
        if p_value is not None:
            print(f"  p值:  {p_value:.6f}")
        print(f"  有效样本数: {len(valid_predictions):,}")

        # 保存评估结果
        eval_results = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'bias': float(bias),
            'r2': float(r2),
            'r': float(r_value),
            'p_value': float(p_value) if p_value is not None else None,
            'num_samples': len(valid_predictions),
            'model_source': 'current_model' if use_current_model else 'loaded_from_file',
            'model_path': str(model_path) if not use_current_model else 'current_in_memory',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        eval_path = self.save_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)

        print(f"\n评估结果已保存到: {eval_path}")

        # 绘制图表
        print("\n1. 生成简单散点图...")
        try:
            self.plot_predictions(valid_predictions, valid_targets)
        except Exception as e:
            print(f"简单散点图失败: {e}")

        print("\n2. 生成密度散点图...")
        try:
            density_results = self.plot_density_scatter_hardcode(valid_predictions, valid_targets)
        except Exception as e:
            print(f"密度散点图失败: {e}")

        # 绘制训练曲线（如果训练过且有历史数据）
        if hasattr(self, 'train_history') and len(self.train_history) > 0:
            print("\n3. 生成训练曲线...")
            try:
                self.plot_training_curves()
            except Exception as e:
                print(f"训练曲线失败: {e}")

        # 运行SHAP分析（如果样本足够）
        if len(valid_predictions) >= 100:
            print("\n4. 运行SHAP分析...")
            try:
                self.run_shap_analysis(num_samples=min(300, len(valid_predictions)))
            except Exception as e:
                print(f"SHAP分析失败: {e}")
        else:
            print("\n4. 跳过SHAP分析（样本数不足）")

        # 绘制调试图（特别关注相关系数）
        if r_value < 0:  # 如果相关系数是负的
            print(f"\n⚠ 警告: 相关系数为负 (R={r_value:.4f})")
            print("绘制详细调试图...")
            self.plot_debug_analysis(valid_predictions, valid_targets)

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
                            choices=['train', 'evaluate', 'test', 'all'],
                            help='运行模式: train=训练, evaluate=评估, test=测试, all=完整流程')

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

        # 创建训练器（只创建一个！）
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
                    # 需要先训练或加载模型
                    # 如果没有训练过的模型，先训练一下
                    print("警告: 没有现有模型，将使用随机初始化的模型进行评估")
                    trainer.evaluate()

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
                    trainer.evaluate()

        print("\n" + "=" * 70)
        print("程序执行完成!")
        print("=" * 70)