# CV10_FOLD_SCATTER_PANEL_V1
# CV10_EPOCH_DIAGNOSTICS_V1
# main_tune.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_tune.py

[CONTRACT]
    SWE 预训练 / 微调 / 交叉验证主入口脚本。

当前主流程：
    - pretrain_progressive:
        先进行预训练十折诊断，再用该阶段100%样本训练阶段正式模型。
    - fine_tune / station_cv:
        使用站点实测 SWE 对预训练模型进行微调和策略比较。

[COMPAT] 兼容性原则：
    1. 旧实验、旧缓存、旧权重可能仍然包含不同的 C_point / C_conv。
       因此不要在核心训练逻辑里盲目硬编码特征维度。
    2. 新增配置项必须提供 default，并从 config.get(...) 读取。
       这样旧命令、旧 JSON、旧 checkpoint 仍能运行。
    3. 历史字段 r2 当前内部仍可能存在，但它实际按 1-SSE/SST 计算，
       展示层不要标成 R²；主图统一使用 r / RMSE / MAE / Bias。
    4. 旧训练器 SWEFullDatasetTrainer 保留用于回看历史实验，
       当前正式 pretrain_progressive 主流程走 SWETrainer。
    5. 对外接口尽量不改函数名、参数名、返回字段名。
       如果必须改，先保留旧字段作为 alias，再逐步迁移。

[DEFAULT-2026] 当前正式预训练推荐：
    MODE=pretrain_progressive
    EPOCHS=90
    BATCH_SIZE=128
    VAL_EVERY=5
    PRECOMPUTE_ALL_SAMPLES=1

[DIAG] profile_timing 只用于短跑性能诊断，正式长跑默认关闭。
"""
import math
# TOP_LEVEL_MATH_IMPORT_FIX_V1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from tqdm import tqdm
import gc
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
import hashlib
from collections import defaultdict, Counter
from scipy import stats
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances
import traceback
import matplotlib
import matplotlib.font_manager as fm
import platform
import random
import shutil
from stability_monitor import StabilityMonitor
warnings.filterwarnings("ignore")


def _seed_dataloader_worker(worker_id):
    """让每个DataLoader worker中的Python/NumPy随机状态可复现。"""
    del worker_id
    worker_seed = int(torch.initial_seed() % (2 ** 32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _dataloader_seed_kwargs(seed):
    """为每个DataLoader创建独立且固定的随机数generator。"""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return {
        "worker_init_fn": _seed_dataloader_worker,
        "generator": generator,
    }


def _configure_reproducibility(seed, *, verbose=True):
    """
    固定Python、NumPy、PyTorch和CUDA随机状态，并启用确定性后端。

    同一软件/硬件环境、同一checkpoint和同一fold下应产生一致结果。
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except TypeError:
        # 兼容没有warn_only参数的旧PyTorch。
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

    if verbose:
        print(
            "✅ 确定性模式: "
            f"seed={seed}, cuDNN benchmark=False, "
            "cuDNN deterministic=True, TF32=False"
        )


# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from lora_swe import convert_to_lora
    LORA_MODULE_AVAILABLE = True
    print("✓ 成功导入LoRA模块")
except ImportError as e:
    LORA_MODULE_AVAILABLE = False
    print(f"⚠ LoRA模块不可用: {e}")

# 导入模型
try:
    from models_swe import (
        create_model,
        SWENet_Full,
        SWENet_SpatialOnly,
        SWENet_PointOnly,
        test_model,
    )

    print("✓ 成功导入模型")
except ImportError as e:
    print(f"✗ 导入模型失败: {e}")
    print("请确保 models_swe.py 在相同目录下")
    sys.exit(1)

# 导入数据加载器
try:
    from data_online_era5_swe import (
        build_dataloaders,
        build_temporal_split_dataloaders,
        build_spatial_split_dataloaders,
        build_spatial_grid_cv_indices, 
    )

    print("✓ 成功导入数据加载器")
except ImportError as e:
    print(f"✗ 导入数据加载器失败: {e}")
    print("请确保 data_online_era5_swe.py 在相同目录下")
    sys.exit(1)

# 尝试导入站点数据加载器（用于微调）
try:
    from data_station_online_swe import build_station_dataloaders_swe

    STATION_MODULE_AVAILABLE = True
    print("✓ 成功导入站点数据加载器")
except ImportError as e:
    STATION_MODULE_AVAILABLE = False
    print(f"⚠ 站点数据加载器不可用: {e}")
    print("微调功能将不可用")



# SWE_STRATIFIED_BATCH_SAMPLER_V1
class SWEStratifiedBatchSampler:
    """
    按站点SWE分层组装每个训练batch，确保中高SWE样本持续出现。

    默认四层：
        <=20 mm, (20,50] mm, (50,80) mm, >=80 mm

    默认比例：
        55%, 25%, 12.5%, 7.5%

    batch_size=32时通常约为：18 / 8 / 4 / 2。
    少数层样本不足时允许循环重采样；每个epoch的batch数仍按原训练集大小确定。
    """

    def __init__(
        self,
        targets_mm,
        batch_size,
        seed=43,
        drop_last=True,
        bin_edges=(20.0, 50.0, 80.0),
        bin_fractions=(0.55, 0.25, 0.125, 0.075),
    ):
        self.targets_mm = np.asarray(targets_mm, dtype=np.float64).reshape(-1)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.bin_edges = tuple(float(v) for v in bin_edges)
        self.bin_fractions = np.asarray(bin_fractions, dtype=np.float64)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError(f"batch_size必须>0，当前={self.batch_size}")
        if self.targets_mm.size == 0:
            raise ValueError("targets_mm为空，无法构建分层batch")
        if len(self.bin_edges) != 3 or self.bin_fractions.size != 4:
            raise ValueError("当前实现要求3个阈值和4个bin比例")
        if np.any(self.bin_fractions < 0) or self.bin_fractions.sum() <= 0:
            raise ValueError("bin_fractions必须为非负且总和>0")

        self.bin_fractions = self.bin_fractions / self.bin_fractions.sum()

        e0, e1, e2 = self.bin_edges
        self.bin_indices = [
            np.where(self.targets_mm <= e0)[0],
            np.where((self.targets_mm > e0) & (self.targets_mm <= e1))[0],
            np.where((self.targets_mm > e1) & (self.targets_mm < e2))[0],
            np.where(self.targets_mm >= e2)[0],
        ]

        raw_quota = self.bin_fractions * self.batch_size
        quotas = np.floor(raw_quota).astype(int)
        remainder = self.batch_size - int(quotas.sum())
        if remainder > 0:
            order = np.argsort(-(raw_quota - quotas))
            for idx in order[:remainder]:
                quotas[idx] += 1

        # 有>=80 mm样本且batch足够大时，至少保证2个高值样本。
        if self.bin_indices[3].size > 0 and self.batch_size >= 16 and quotas[3] < 2:
            donor = int(np.argmax(quotas[:3]))
            transfer = min(2 - quotas[3], max(0, quotas[donor] - 1))
            quotas[donor] -= transfer
            quotas[3] += transfer

        # 空bin的配额转移给当前样本最多的非空bin。
        for idx, pool in enumerate(self.bin_indices):
            if pool.size == 0 and quotas[idx] > 0:
                nonempty = [j for j, p in enumerate(self.bin_indices) if p.size > 0 and j != idx]
                if not nonempty:
                    raise ValueError("所有SWE分层均为空")
                receiver = max(nonempty, key=lambda j: self.bin_indices[j].size)
                quotas[receiver] += quotas[idx]
                quotas[idx] = 0

        if int(quotas.sum()) != self.batch_size:
            raise RuntimeError(
                f"分层batch配额和错误: {quotas.tolist()} != batch_size={self.batch_size}"
            )

        self.quotas = quotas.astype(int)
        if self.drop_last:
            self.num_batches = self.targets_mm.size // self.batch_size
        else:
            self.num_batches = int(np.ceil(self.targets_mm.size / self.batch_size))
        self.num_batches = max(1, int(self.num_batches))

    def __len__(self):
        return self.num_batches

    @staticmethod
    def _draw_from_pool(pool, quota, rng, state):
        if quota <= 0:
            return []
        if pool.size == 0:
            return []

        selected = []
        while len(selected) < quota:
            order, cursor = state
            if order is None or cursor >= len(order):
                order = rng.permutation(pool)
                cursor = 0

            take = min(quota - len(selected), len(order) - cursor)
            selected.extend(order[cursor:cursor + take].tolist())
            cursor += take
            state[0] = order
            state[1] = cursor

        return selected

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        states = [[None, 0] for _ in self.bin_indices]

        for _ in range(self.num_batches):
            batch = []
            for pool, quota, state in zip(self.bin_indices, self.quotas, states):
                batch.extend(self._draw_from_pool(pool, int(quota), rng, state))

            rng.shuffle(batch)
            yield batch

    def describe(self):
        names = ["<=20", "20-50", "50-80", ">=80"]
        return {
            name: {
                "available": int(pool.size),
                "per_batch": int(quota),
            }
            for name, pool, quota in zip(names, self.bin_indices, self.quotas)
        }

    
class MixUp:
    """MixUp 数据增强"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        print(f"  MixUp初始化: alpha={alpha}")
    
    def __call__(self, conv, point, target):
        """
        对批次数据应用 MixUp
        返回: 混合后的 conv, point, target, lambda, index
        """
        batch_size = conv.size(0)
        
        # 随机打乱索引
        index = torch.randperm(batch_size).to(conv.device)
        
        # 从 Beta 分布采样 lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # 混合卷积特征 (7, 5, 5)
        mixed_conv = lam * conv + (1 - lam) * conv[index]
        
        # 混合点特征 (13)
        mixed_point = lam * point + (1 - lam) * point[index]
        
        # 混合标签
        mixed_target = lam * target + (1 - lam) * target[index]
        
        return mixed_conv, mixed_point, mixed_target, lam, index


class MixUpWrapper:
    """MixUp 包装器，支持预热和概率"""
    def __init__(self, config):
        self.config = config
        self.enabled = config.get('use_mixup', False)
        self.alpha = config.get('mixup_alpha', 0.2)
        self.prob = config.get('mixup_prob', 0.5)
        self.warmup = config.get('mixup_warmup', 5)
        
        if self.enabled:
            self.mixup = MixUp(alpha=self.alpha)
            print(f"  MixUp增强: alpha={self.alpha}, prob={self.prob}, warmup={self.warmup}")
    
    def __call__(self, conv, point, target, epoch):
        """
        应用 MixUp
        返回: 混合后的数据，以及 lam 和 index（用于损失计算）
        """
        if not self.enabled:
            return conv, point, target, 1.0, None
        
        # 预热期不用 MixUp
        if epoch < self.warmup:
            return conv, point, target, 1.0, None
        
        # 按概率应用
        if random.random() > self.prob:
            return conv, point, target, 1.0, None
        
        # 应用 MixUp
        return self.mixup(conv, point, target)


def mixup_criterion(criterion, pred, target_a, target_b, lam):
    """
    MixUp 损失函数
    loss = lam * loss(pred, target_a) + (1-lam) * loss(pred, target_b)
    """
    return lam * criterion(pred, target_a) + (1 - lam) * criterion(pred, target_b)
    
    
    
    
# ============================================================
# [LEGACY] SWEFullDatasetTrainer
# ============================================================
# 早期全样本训练器：无验证集划分，主要用于历史实验/调试。
#
# [COMPAT]
#   保留该类是为了旧实验脚本仍可运行。
#   当前正式 pretrain_progressive / fine_tune 主流程不走这里。
#
# [DANGER]
#   文件中存在两个 train_epoch()：
#       1. SWEFullDatasetTrainer.train_epoch()  (行 ~560)
#       2. SWETrainer.train_epoch()            (行 ~3530)
#   当前正式流程使用第二个。
#   训练加速、AMP、val_every、best model 等逻辑应优先改 SWETrainer。
# ============================================================
class SWEFullDatasetTrainer:
    """[LEGACY] 使用全样本训练SWE模型（无验证集划分）"""

    def __init__(self, config=None):
        # 默认配置
        self.default_config = {
            # 模型类型
            "model_type": "full",
            # 训练参数
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            # 模型参数
            "d_model": 256,
            "C_conv": None,
            "C_point": None,
            # 路径设置
            "save_dir": "./full_training",
            "experiment_name": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            # 训练监控
            "save_freq": 10,
            "clip_grad": 1.0,
            "patience": 20,  # 基于训练损失早停
        }

        # 更新配置
        if config:
            self.default_config.update(config)
        self.config = self.default_config

        # 设置实验名称
        if self.config["experiment_name"] is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config["experiment_name"] = (
                f"swe_full_{self.config['model_type']}_{timestamp}"
            )

        # 设置设备
        self.device = torch.device(self.config["device"])
        print(f"使用设备: {self.device}")

        # 初始化变量
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.full_loader = None

        # 训练历史
        self.train_history = []
        self.lr_history = []

        # 创建保存目录
        self.save_dir = Path(self.config["save_dir"]) / self.config["experiment_name"]
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        print(f"全样本训练保存目录: {self.save_dir}")

    def _save_config(self):
        """保存配置到文件"""
        config_path = self.save_dir / "full_training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"配置已保存到: {config_path}")

    def build_full_dataloader(self):
        """构建全样本数据加载器（无验证集划分）"""
        print("\n" + "=" * 60)
        print("构建全样本数据加载器...")
        print("=" * 60)

        try:
            # 导入数据集
            from data_online_era5_swe import SWEDataset

            # 创建完整数据集（不划分验证集）
            dataset = SWEDataset(
                region="XINJIANG",
                year_target=2016,
                patch_size=5,
                min_valid_pixels=100,
                samples_per_day=2000,
                clamday_threshold=0.5,
            )

            # 创建全样本数据加载器
            self.full_loader = DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=10,
                pin_memory=True,
                drop_last=False,  # 保留所有样本
            )

            # 获取特征维度
            self.config["C_conv"] = dataset.C_conv
            self.config["C_point"] = dataset.C_point

            print(f"✓ 全样本数据加载器构建成功!")
            print(f"\n数据统计:")
            print(f"  总样本数: {len(dataset):,}")
            print(f"  卷积特征维度: C_conv={dataset.C_conv}")
            print(f"  点特征维度: C_point={dataset.C_point}")
            print(f"  批次大小: {self.config['batch_size']}")
            print(f"  总批次: {len(self.full_loader)}")

            # 测试一个批次
            self._test_dataloader()

            return True

        except Exception as e:
            print(f"✗ 构建全样本数据加载器失败: {e}")

            traceback.print_exc()
            return False

    def _test_dataloader(self):
        """测试数据加载器"""
        print(f"\n测试数据加载...")
        try:
            conv, point, target = next(iter(self.full_loader))

            print(f"  卷积特征: {conv.shape}")
            print(f"  点特征: {point.shape}")
            print(f"  目标值: {target.shape}")

            # 检查数据范围
            print(f"\n  数据范围检查:")
            print(f"    卷积特征: [{conv.min():.3f}, {conv.max():.3f}]")
            print(f"    点特征: [{point.min():.3f}, {point.max():.3f}]")
            print(f"    目标值: [{target.min():.3f}, {target.max():.3f}]")

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
            # 检查维度是否已设置
            if self.config["C_conv"] is None or self.config["C_point"] is None:
                print("警告: 特征维度未设置，尝试从数据中推断...")
                if self.full_loader is not None:
                    conv, point, _ = next(iter(self.full_loader))
                    self.config["C_conv"] = conv.shape[1]
                    self.config["C_point"] = point.shape[1]
                else:
                    self.config["C_conv"] = 7 # 默认值
                    self.config["C_point"] = 10  # 默认值

            print(
                f"模型参数: C_conv={self.config['C_conv']}, C_point={self.config['C_point']}"
            )

            # 创建模型
            self.model = create_model(
                model_type=self.config["model_type"],
                C_spatial=self.config["C_conv"],
                C_point=self.config["C_point"],
                d_model=self.config["d_model"],
                use_wide_branch=False,
            )

            # ============ 关键修复：解决高值平线问题 ============
            print("🔧 初始化输出层，解决高值平线问题")
            
            # 找到输出层并修改其初始化
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.out_features == 1:
                    # 这是最终的输出层
                    print(f"  找到输出层: {name}")
                    
                    # 1. 增大权重初始范围
                    if hasattr(module, 'weight'):
                        nn.init.xavier_uniform_(module.weight, gain=2.0)
                        print(f"    权重初始化: gain=2.0")
                    
                    # 2. 将偏置初始化为正值（关键！）
                    if hasattr(module, 'bias') and module.bias is not None:
                        # 根据您的数据，SWE归一化后高值大约0.6-1.0
                        # 将偏置初始化为中高值（0.4）
                        nn.init.constant_(module.bias, 0.4)
                        print(f"    偏置初始化为: 0.4")
                        
                        # 验证
                        print(f"    验证: 偏置值 = {module.bias.data.item():.4f}")
            
            # 验证模型初始化效果
            print("\n  验证初始化效果:")
            self.model.eval()
            with torch.no_grad():
                # 创建测试输入
                batch_size = 2
                test_conv = torch.randn(batch_size, self.config["C_conv"], 5, 5)
                test_point = torch.randn(batch_size, self.config["C_point"])
                
                # 测试零输入（应该输出偏置值）
                zero_output = self.model(
                    torch.zeros_like(test_conv),
                    torch.zeros_like(test_point)
                )
                print(f"    零输入预测均值: {zero_output.mean().item():.4f}")
                
                # 测试随机输入
                rand_output = self.model(test_conv, test_point)
                print(f"    随机输入预测范围: [{rand_output.min().item():.4f}, {rand_output.max().item():.4f}]")
            
            self.model.train()  # 恢复训练模式
            
            # 移到设备
            self.model.to(self.device)

            # 打印模型信息
            self._print_model_info()

            # 设置损失函数和优化器
            self.criterion = nn.SmoothL1Loss(beta=0.02, reduction="mean")

            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )

            # 学习率调度器（基于训练损失）
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
            )

            print(f"✓ 模型构建成功!")

            return True

        except Exception as e:
            print(f"✗ 模型构建失败: {e}")

            traceback.print_exc()
            return False

    def _print_model_info(self, freeze_backbone=False, use_lora=False):
        """打印模型信息 - 修正文字输出"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params * 100

        print(f"\n模型信息:")
        print(f"  类型: {self.config['model_type']}")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  可训练比例: {trainable_ratio:.2f}%")

        # ============ 修正：根据实际可训练比例判断状态 ============
        if use_lora:
            print(f"  模式: LoRA微调")
        elif freeze_backbone and trainable_ratio < 50:
            print(f"  模式: 冻结主干 (可训练比例 {trainable_ratio:.1f}%)")
        elif freeze_backbone and trainable_ratio > 80:
            print(f"  ⚠️ 警告: 冻结主干模式但可训练比例过高 ({trainable_ratio:.1f}%)")
            print(f"     冻结可能未生效，请检查配置")
        else:
            print(f"  模式: 全部可训练")

    def train_full_dataset(self):
        """使用全样本训练模型"""
        print("\n" + "=" * 60)
        print("开始全样本训练...")
        print("=" * 60)

        # 检查数据加载器和模型
        if self.full_loader is None:
            print("✗ 请先构建全样本数据加载器!")
            return None

        if self.model is None:
            print("✗ 请先构建模型!")
            return None

        best_train_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        # 训练循环
        for epoch in range(self.config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print("-" * 50)

            # 训练一个epoch
            train_loss = self._train_epoch(epoch)
            self.train_history.append(train_loss)

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)

            # 调整学习率（基于训练损失）
            self.scheduler.step(train_loss)

            # 打印结果
            print(f"训练损失: {train_loss:.6f}")
            print(f"学习率: {current_lr:.2e}")

            # 保存最佳模型
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch
                patience_counter = 0

                self._save_checkpoint(f"best_full_model.pth", epoch, train_loss)
                print(f"✓ 保存最佳模型 (epoch {epoch + 1})")
            else:
                patience_counter += 1

            # 定期保存检查点
            if (epoch + 1) % self.config["save_freq"] == 0:
                self._save_checkpoint(
                    f"checkpoint_epoch{epoch + 1}.pth", epoch, train_loss
                )

            # 早停检查（基于训练损失）
            if patience_counter >= self.config["patience"]:
                print(f"\n⚠ 早停触发! 连续{self.config['patience']}轮训练损失未改善")
                break

        print("\n" + "=" * 60)
        print(f"全样本训练完成!")
        print(f"最佳训练损失: {best_train_loss:.6f} (epoch {best_epoch + 1})")
        print(f"总训练轮次: {epoch + 1}")
        print("=" * 60)

        # 保存最终模型
        self._save_checkpoint("final_full_model.pth", best_epoch, best_train_loss)

        # 保存训练历史
        self._save_training_history()

        # 绘制训练曲线
        self._plot_training_curves()

        return best_train_loss

    def train_epoch(self, epoch, is_fine_tune=False):
        """训练一个epoch - 支持混合精度加速"""
        # ============ 修复：这个if语句的缩进错误 ============
        if epoch == 0:
            print("\n【数据完整性检查】")
            for batch in self.train_loader:
                conv, point, target, mask = batch
                print(f"  conv stats:")
                print(f"    shape: {conv.shape}")
                print(f"    dtype: {conv.dtype}")
                print(f"    range: [{conv.min():.4f}, {conv.max():.4f}]")
                print(f"    mean: {conv.mean():.4f} ± {conv.std():.4f}")
                print(f"    has nan: {torch.isnan(conv).any()}")
                print(f"    has inf: {torch.isinf(conv).any()}")
                print(f"    has 0: {(conv==0).sum().item()} zeros")
                break
        
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        # ============ CUDA错误检查 ============
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if epoch == 0:
                print(f"  显存使用: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        
        # ============ 混合精度初始化 ============
        use_amp = self.config.get("use_amp", False)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # ============ 模式判断 ============
        use_lora = is_fine_tune and self.config.get("use_lora", False)
        use_traditional_fine_tune = is_fine_tune and not use_lora
        
        if use_lora:
            print(f"  【LoRA训练模式】Epoch {epoch+1}")
            lora_grad_norms = []
            lora_weight_changes = []
        elif use_traditional_fine_tune:
            print(f"  【传统微调模式】Epoch {epoch+1}")
        
        # ============ 微调专用：初始化SmoothL1Loss ============
        if is_fine_tune:
            smooth_l1_criterion = nn.SmoothL1Loss(beta=0.01, reduction='mean')
            print(f"    使用SmoothL1Loss (beta=0.01)")
            
            prediction_stats = {
                'pred_means': [],
                'pred_stds': [],
                'target_means': [],
                'target_stds': [],
                'zero_pred_stats': []
            }
        
        # 如果使用channels last格式，转换模型
        if self.config.get("channels_last", False) and not hasattr(self, '_channels_last_set'):
            self.model = self.model.to(memory_format=torch.channels_last)
            self._channels_last_set = True
        
        # ============ batch 时间诊断 ============
        data_time_total = 0.0
        gpu_time_total = 0.0
        last_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            data_time = time.time() - last_time
            data_time_total += data_time
            step_start = time.time()

            try:
                # ============ 处理不同数量的返回值 ============
                if len(batch_data) == 4:
                    conv_feats, point_feats, targets, is_zero_mask = batch_data
                elif len(batch_data) == 3:
                    conv_feats, point_feats, targets = batch_data
                    is_zero_mask = torch.where(targets > 0, 
                                             torch.ones_like(targets), 
                                             torch.zeros_like(targets))
                else:
                    print(f"  批次 {batch_idx+1}: 数据格式错误，跳过")
                    continue
                
                # ============ 详细的输入检查 ============
                if epoch == 0 and batch_idx == 0:
                    print(f"\n【详细输入检查 - 第一个批次】")
                    print(f"  conv_feats:")
                    print(f"    shape: {conv_feats.shape}")
                    print(f"    dtype: {conv_feats.dtype}")
                    print(f"    device: {conv_feats.device}")
                    print(f"    range: [{conv_feats.min():.4f}, {conv_feats.max():.4f}]")
                    print(f"    mean: {conv_feats.mean():.4f} ± {conv_feats.std():.4f}")
                    print(f"    has nan: {torch.isnan(conv_feats).any()}")
                    print(f"    has inf: {torch.isinf(conv_feats).any()}")
                    
                    # 检查每个通道
                    for c in range(conv_feats.shape[1]):
                        channel = conv_feats[:, c, :, :]
                        print(f"    channel {c}: range [{channel.min():.4f}, {channel.max():.4f}], "
                              f"mean={channel.mean():.4f}±{channel.std():.4f}")
                    
                    print(f"\n  point_feats:")
                    print(f"    shape: {point_feats.shape}")
                    print(f"    dtype: {point_feats.dtype}")
                    print(f"    device: {point_feats.device}")
                    print(f"    range: [{point_feats.min():.4f}, {point_feats.max():.4f}]")
                    print(f"    mean: {point_feats.mean():.4f} ± {point_feats.std():.4f}")
                    print(f"    has nan: {torch.isnan(point_feats).any()}")
                    print(f"    has inf: {torch.isinf(point_feats).any()}")
                    
                    # 检查点特征的每个维度
                    for f in range(point_feats.shape[1]):
                        feature = point_feats[:, f]
                        print(f"    feature {f}: range [{feature.min():.4f}, {feature.max():.4f}], "
                              f"mean={feature.mean():.4f}±{feature.std():.4f}")
                    
                    print(f"\n  targets:")
                    print(f"    shape: {targets.shape}")
                    print(f"    dtype: {targets.dtype}")
                    print(f"    range: [{targets.min():.4f}, {targets.max():.4f}]")
                    print(f"    mean: {targets.mean():.4f} ± {targets.std():.4f}")
                    print(f"    has nan: {torch.isnan(targets).any()}")
                    print(f"    has inf: {torch.isinf(targets).any()}")
                
                # ============ 检查输入是否有NaN/Inf ============
                if torch.isnan(conv_feats).any():
                    print(f"  批次 {batch_idx+1}: 卷积特征包含NaN，修复中...")
                    conv_feats = torch.nan_to_num(conv_feats, nan=0.0)
                if torch.isinf(conv_feats).any():
                    print(f"  批次 {batch_idx+1}: 卷积特征包含Inf，修复中...")
                    conv_feats = torch.nan_to_num(conv_feats, posinf=1.0, neginf=-1.0)
                
                if torch.isnan(point_feats).any():
                    print(f"  批次 {batch_idx+1}: 点特征包含NaN，修复中...")
                    point_feats = torch.nan_to_num(point_feats, nan=0.0)
                if torch.isinf(point_feats).any():
                    print(f"  批次 {batch_idx+1}: 点特征包含Inf，修复中...")
                    point_feats = torch.nan_to_num(point_feats, posinf=1.0, neginf=-1.0)
                
                if torch.isnan(targets).any():
                    print(f"  批次 {batch_idx+1}: 目标值包含NaN，修复中...")
                    targets = torch.nan_to_num(targets, nan=0.0)
                if torch.isinf(targets).any():
                    print(f"  批次 {batch_idx+1}: 目标值包含Inf，修复中...")
                    targets = torch.nan_to_num(targets, posinf=1.0, neginf=-1.0)
                
                # ============ 微调专用：特征增强 ============
                if is_fine_tune:
                    original_batch_size = len(targets)
                    if epoch < 10 and original_batch_size > 1:
                        noise_scale = 0.002 * (1.0 - epoch/10)
                        point_feats = point_feats + torch.randn_like(point_feats) * noise_scale
                    
                    if epoch < 8 and batch_idx % 3 == 0 and original_batch_size > 2:
                        lam = np.random.beta(0.15, 0.15)
                        idx = torch.randperm(original_batch_size)
                        point_feats = lam * point_feats + (1 - lam) * point_feats[idx]
                        if conv_feats is not None and conv_feats.numel() > 0:
                            conv_feats = lam * conv_feats + (1 - lam) * conv_feats[idx]
                        targets = lam * targets + (1 - lam) * targets[idx]
                        is_zero_mask = torch.where(targets > 0, 
                                                 torch.ones_like(targets), 
                                                 torch.zeros_like(targets))
                
                # 移动到设备
                conv_feats = conv_feats.to(self.device, non_blocking=True)
                point_feats = point_feats.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # ============ 确保 is_zero_mask 维度正确 ============
                if is_zero_mask is not None:
                    is_zero_mask = is_zero_mask.to(self.device, non_blocking=True)
                    if is_zero_mask.dim() > 1:
                        is_zero_mask = is_zero_mask.squeeze()
                    if is_zero_mask.dtype != torch.float32:
                        is_zero_mask = is_zero_mask.float()
                else:
                    # 如果 is_zero_mask 为 None，创建一个默认的
                    is_zero_mask = torch.ones_like(targets)
                
                # 确保张量是连续的
                conv_feats = conv_feats.contiguous()
                point_feats = point_feats.contiguous()
                
                # 如果使用channels last格式，转换卷积特征
                if self.config.get("channels_last", False):
                    conv_feats = conv_feats.to(memory_format=torch.channels_last)
                                

                # ============ 修复点特征维度 ============
                expected_point_dim = self.config.get("C_point", 18)
                if point_feats.shape[1] != expected_point_dim:
                    print(f"  批次 {batch_idx+1}: 点特征维度 {point_feats.shape[1]} != {expected_point_dim}")
                    if point_feats.shape[1] < expected_point_dim:
                        padding = torch.zeros(point_feats.shape[0], expected_point_dim - point_feats.shape[1], device=point_feats.device)
                        point_feats = torch.cat([point_feats, padding], dim=1)
                    else:
                        point_feats = point_feats[:, :expected_point_dim]
                
                
                # ============ 前向传播前的最终检查 ============
                if epoch == 0 and batch_idx == 0:
                    print(f"\n【前向传播前最终检查】")
                    print(f"  conv_feats device: {conv_feats.device}")
                    print(f"  point_feats device: {point_feats.device}")
                    print(f"  model device: {next(self.model.parameters()).device}")
                    
                    # 检查模型参数是否有NaN
                    for name, param in self.model.named_parameters():
                        if torch.isnan(param).any():
                            print(f"  ⚠️ 模型参数 {name} 包含NaN!")
                        if torch.isinf(param).any():
                            print(f"  ⚠️ 模型参数 {name} 包含Inf!")
                
                # ============ 混合精度前向传播 ============
                gpu_start = time.time()
                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        # 测试用小批次进行前向传播
                        if epoch == 0 and batch_idx == 0:
                            print(f"\n【测试前向传播 - 第一个批次的前2个样本】")
                            test_conv = conv_feats[:2]
                            test_point = point_feats[:2]
                            test_outputs = self.model(test_conv, test_point)
                            print(f"  测试输出 shape: {test_outputs.shape}")
                            print(f"  测试输出 range: [{test_outputs.min():.4f}, {test_outputs.max():.4f}]")
                            print(f"  测试输出 mean: {test_outputs.mean():.4f} ± {test_outputs.std():.4f}")
                            print(f"  测试输出 has nan: {torch.isnan(test_outputs).any()}")
                            print(f"  测试输出 has inf: {torch.isinf(test_outputs).any()}")
                        
                        # 完整前向传播
                        outputs = self.model(conv_feats, point_feats)
                        
                        # 检查输出
                        if torch.isnan(outputs).any():
                            print(f"  批次 {batch_idx+1}: 模型输出包含NaN，修复中...")
                            outputs = torch.nan_to_num(outputs, nan=0.0)
                        if torch.isinf(outputs).any():
                            print(f"  批次 {batch_idx+1}: 模型输出包含Inf，修复中...")
                            outputs = torch.nan_to_num(outputs, posinf=1.0, neginf=-1.0)
                        
                        # ============ 安全地处理 is_zero_mask ============
                        try:
                            # 预训练阶段不要强制把 target=0 的预测改成0，否则这些样本没有真实梯度。
                            # 站点微调阶段如果存在0值，再启用这个约束。
                            if is_fine_tune and is_zero_mask is not None and is_zero_mask.numel() == targets.numel():
                                zero_mask = (is_zero_mask == 0)
                                if torch.any(zero_mask):
                                    zero_indices = zero_mask.nonzero(as_tuple=True)[0]
                                    if len(zero_indices) > 0:
                                        # 检查强制置0前的值
                                        if epoch == 0 and batch_idx == 0 and len(zero_indices) > 0:
                                            print(f"\n  【target=0样本检查】")
                                            print(f"    强制置0前: {outputs[zero_indices].detach().cpu().numpy()}")
                                        
                                        outputs[zero_indices] = 0.0
                                        
                                        if is_fine_tune:
                                            prediction_stats['zero_pred_stats'].append({
                                                'count': len(zero_indices),
                                                'mean_abs': outputs[zero_indices].abs().mean().item(),
                                                'std': outputs[zero_indices].std().item()
                                            })
                                        
                                        if batch_idx % 20 == 0:
                                            mode = "微调" if is_fine_tune else "预训练"
                                            zero_pred_avg = outputs[zero_indices].abs().mean().item()
                                            print(f"    {mode}批次 {batch_idx+1}: 强制{len(zero_indices)}个target=0样本预测值为0, 平均={zero_pred_avg:.6f}")
                        except Exception as e:
                            if batch_idx % 20 == 0:
                                print(f"    is_zero_mask处理跳过: {e}")
                        
                        # ============ 损失计算 ============
                        if is_fine_tune:
                            outputs_flat = outputs.reshape(-1)
                            targets_flat = targets.reshape(-1)

                            swe_min = getattr(self, "swe_min", 0.0)
                            swe_max = getattr(self, "swe_max", 170.0)

                            target_mm = targets_flat * (swe_max - swe_min) + swe_min

                            loss_each = F.smooth_l1_loss(
                                outputs_flat,
                                targets_flat,
                                beta=0.01,
                                reduction="none"
                            )

                            weights = torch.ones_like(targets_flat)

                            # 高 SWE 加权：先别太猛，避免过拟合
                            weights = weights + 1.0 * (target_mm >= 20.0).float()
                            weights = weights + 2.0 * (target_mm >= 50.0).float()
                            weights = weights + 3.0 * (target_mm >= 80.0).float()

                            loss = (loss_each * weights).sum() / (weights.sum() + 1e-8)

                            # 保留轻量方差约束，防止输出塌成窄带
                            if epoch < 15:
                                target_var = targets_flat.var()
                                pred_var = outputs_flat.var()
                                if target_var > 1e-6 and pred_var / target_var < 0.5:
                                    variance_loss = torch.relu(0.5 * target_var - pred_var)
                                    loss = loss + 0.02 * variance_loss
                        else:
                            # 预训练分支必须先 flatten，避免 outputs=[B,1] 与 targets=[B] 广播成 [B,B]
                            outputs_flat = outputs.reshape(-1)
                            targets_flat = targets.reshape(-1)

                            loss_each = F.smooth_l1_loss(
                                outputs_flat,
                                targets_flat,
                                beta=0.02,
                                reduction="none"
                            )

                            # 轻量高 SWE 加权：只在归一化 loss 上加权，避免低值样本完全支配
                            swe_min = getattr(self, "swe_min", 0.0)
                            swe_max = getattr(self, "swe_max", 1.0)
                            target_mm = targets_flat * (swe_max - swe_min) + swe_min

                            weights = torch.ones_like(targets_flat)
                            weights = weights + 0.5 * (target_mm >= 20.0).float()
                            weights = weights + 1.0 * (target_mm >= 50.0).float()
                            weights = weights + 2.0 * (target_mm >= 80.0).float()
                            weights = weights + 3.0 * (target_mm >= 200.0).float()

                            loss = (loss_each * weights).sum() / (weights.sum() + 1e-8)

                            # 防止输出塌成窄带：前15轮给轻量方差约束
                            if epoch < 15:
                                target_var = targets_flat.var()
                                pred_var = outputs_flat.var()
                                if target_var > 1e-6 and pred_var / target_var < 0.5:
                                    variance_loss = torch.relu(0.5 * target_var - pred_var)
                                    loss = loss + 0.02 * variance_loss
                        
                        # 检查损失
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"  批次 {batch_idx+1}: 损失为 {loss.item()}，跳过")
                            continue
                        
                except RuntimeError as e:
                    print(f"\n❌ 前向传播失败: {e}")
                    print(f"  conv_feats shape: {conv_feats.shape}")
                    print(f"  point_feats shape: {point_feats.shape}")
                    print(f"  conv_feats dtype: {conv_feats.dtype}")
                    print(f"  point_feats dtype: {point_feats.dtype}")
                    print(f"  conv_feats device: {conv_feats.device}")
                    print(f"  point_feats device: {point_feats.device}")
                    
                    # 尝试逐层调试
                    print("\n【尝试逐层调试】")
                    x = conv_feats[:1]
                    p = point_feats[:1]
                    
                    # 检查spatial_encoder
                    try:
                        print("  spatial_encoder...")
                        spatial_out = self.model.spatial_encoder(x)
                        print(f"    output shape: {spatial_out.shape}")
                        print(f"    range: [{spatial_out.min():.4f}, {spatial_out.max():.4f}]")
                    except Exception as e2:
                        print(f"    spatial_encoder failed: {e2}")
                    
                    # 检查point_encoder
                    try:
                        print("  point_encoder...")
                        point_out = self.model.point_encoder(p)
                        print(f"    output shape: {point_out.shape}")
                        print(f"    range: [{point_out.min():.4f}, {point_out.max():.4f}]")
                    except Exception as e2:
                        print(f"    point_encoder failed: {e2}")
                    
                    raise e
                
                # ============ 微调专用：预测分析 ============
                if is_fine_tune and batch_idx % 20 == 0:
                    prediction_stats['pred_means'].append(outputs.mean().item())
                    prediction_stats['pred_stds'].append(outputs.std().item())
                    prediction_stats['target_means'].append(targets.mean().item())
                    prediction_stats['target_stds'].append(targets.std().item())
                    
                    unique_vals = torch.unique(outputs.round(decimals=3))
                    if len(unique_vals) < 5:
                        print(f"      ⚠ 批次 {batch_idx+1}: 预测值种类较少 ({len(unique_vals)}种)")
                
                # ============ 反向传播（使用scaler） ============
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # ============ 梯度监控 ============
                if is_fine_tune:
                    if use_lora and (batch_idx == 0 or batch_idx % 10 == 0):
                        batch_lora_grads = []
                        for name, param in self.model.named_parameters():
                            if ('lora_A' in name or 'lora_B' in name) and param.grad is not None:
                                batch_lora_grads.append(param.grad.norm().item())
                        if batch_lora_grads:
                            lora_grad_norms.append(np.mean(batch_lora_grads))
                    
                    elif use_traditional_fine_tune and (batch_idx == 0 or batch_idx % 10 == 0):
                        batch_gradients = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None and param.requires_grad:
                                batch_gradients.append(param.grad.norm().item())
                        if batch_gradients and batch_idx % 20 == 0:
                            print(f"      梯度平均范数: {np.mean(batch_gradients):.6e}")
                
                # ============ 梯度裁剪（需要先unscale） ============
                if self.config["clip_grad"] > 0:
                    scaler.unscale_(self.optimizer)
                    if use_lora:
                        clip_value = self.config["clip_grad"] * 2.0
                    else:
                        clip_value = self.config["clip_grad"]
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                
                # ============ 更新参数 ============
                scaler.step(self.optimizer)
                scaler.update()

                # [DIAG] profile_timing 只用于短跑性能诊断。
                # True 时会调用 torch.cuda.synchronize()，使 gpu_time 统计更准确。
                #
                # [DANGER]
                #   synchronize 会破坏 CUDA 异步执行，正式训练会变慢。
                #   正式 10 折 / 95% 最终训练时必须保持 False。
                profile_timing = bool(self.config.get("profile_timing", False))
                if profile_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
                gpu_time = time.time() - gpu_start
                gpu_time_total += gpu_time
                last_time = time.time()
                step_time = time.time() - step_start

                # 记录损失
                total_loss += loss.item()
                batch_count += 1
                
                # 每10个batch打印一次
                if (batch_idx + 1) % 10 == 0:
                    mode = "LoRA微调" if use_lora else ("传统微调" if use_traditional_fine_tune else "预训练")
                    print(f"    {mode}批次 {batch_idx + 1}/{len(self.train_loader)} | 损失: {loss.item():.6f}")

                # 每50个batch打印时间诊断
                if profile_timing and (batch_idx + 1) % 50 == 0:
                    print(
                        f"    batch {batch_idx+1}: "
                        f"data_time={data_time_total/(batch_idx+1):.4f}s, "
                        f"gpu_time={gpu_time_total/(batch_idx+1):.4f}s, "
                        f"step_time={step_time:.4f}s"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  批次 {batch_idx+1}: CUDA显存不足，跳过")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"  批次 {batch_idx+1}: 发生错误: {e}")
                    print(f"  conv_feats shape: {conv_feats.shape if 'conv_feats' in locals() else 'N/A'}")
                    print(f"  point_feats shape: {point_feats.shape if 'point_feats' in locals() else 'N/A'}")
                    torch.cuda.empty_cache()
                    continue
        
        # ============ 微调专用：本轮总结 ============
        if is_fine_tune and batch_count > 0:
            print(f"\n  【微调分析】Epoch {epoch} 总结:")
            
            if prediction_stats['pred_means']:
                avg_pred_mean = np.mean(prediction_stats['pred_means'])
                avg_pred_std = np.mean(prediction_stats['pred_stds'])
                avg_target_mean = np.mean(prediction_stats['target_means'])
                avg_target_std = np.mean(prediction_stats['target_stds'])
                
                print(f"    预测统计: {avg_pred_mean:.4f}±{avg_pred_std:.4f}")
                print(f"    目标统计: {avg_target_mean:.4f}±{avg_target_std:.4f}")
                
                if avg_pred_std < 0.05 and avg_target_std > 0.1:
                    print(f"    ⚠ 警告: 预测方差过小 (可能平线)")
            
            if prediction_stats['zero_pred_stats']:
                total_zero_count = sum(stat['count'] for stat in prediction_stats['zero_pred_stats'])
                avg_zero_mean_abs = np.mean([stat['mean_abs'] for stat in prediction_stats['zero_pred_stats']])
                print(f"\n    【约束效果】target=0样本总结:")
                print(f"      总样本数: {total_zero_count}")
                print(f"      预测绝对值均值: {avg_zero_mean_abs:.6f}")
        
        # ============ LoRA梯度历史记录 ============
        if use_lora and lora_grad_norms:
            if not hasattr(self, 'lora_grad_history'):
                self.lora_grad_history = []
            self.lora_grad_history.append({
                'epoch': epoch,
                'avg_grad_norm': np.mean(lora_grad_norms),
                'max_grad_norm': max(lora_grad_norms),
                'min_grad_norm': min(lora_grad_norms),
                'num_samples': len(lora_grad_norms),
            })
            print(f"    【LoRA梯度】Epoch {epoch}: 平均={np.mean(lora_grad_norms):.6e}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    def _save_checkpoint(self, filename, epoch, loss):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "train_history": self.train_history,
            "lr_history": self.lr_history,
            "val_history_metrics": list(
                getattr(self, "val_history_metrics", [])
            ),
            "random_state": (
                self._capture_overfit_resume_random_state()
                if bool(
                    self.config.get("train_only_overfit", False)
                )
                and hasattr(
                    self,
                    "_capture_overfit_resume_random_state",
                )
                else None
            ),
            "config": self.config,
            "train_loss": loss,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path, pickle_protocol=4)
        print(f"✓ 检查点保存到: {save_path}")

    def _save_training_history(self):
        """保存训练历史"""
        history = {
            "train_loss": self.train_history,
            "lr_history": self.lr_history,
            "config": self.config,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        history_path = self.save_dir / "full_training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

        print(f"训练历史已保存到: {history_path}")

    def _plot_training_curves(self):
        """绘制训练曲线"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # 1. 损失曲线
            epochs = range(1, len(self.train_history) + 1)
            ax1.plot(epochs, self.train_history, "b-", linewidth=2)
            ax1.set_xlabel("Epoch", fontsize=12)
            ax1.set_ylabel("Training Loss (MSE)", fontsize=12)
            ax1.set_title("全样本训练损失曲线", fontsize=14, fontweight="bold")
            ax1.grid(True, alpha=0.3)

            # 标记最佳epoch
            best_idx = np.argmin(self.train_history)
            ax1.scatter(
                best_idx + 1,
                self.train_history[best_idx],
                color="red",
                s=100,
                zorder=5,
                label=f"最佳 (Epoch {best_idx + 1})",
            )
            ax1.legend(fontsize=11)

            # 2. 学习率曲线
            ax2.plot(epochs, self.lr_history, "g-", linewidth=2)
            ax2.set_xlabel("Epoch", fontsize=12)
            ax2.set_ylabel("Learning Rate", fontsize=12)
            ax2.set_title("学习率变化", fontsize=14, fontweight="bold")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

            plt.suptitle(
                f'全样本SWE模型训练曲线 - {self.config["model_type"]}',
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()

            plot_path = self.save_dir / "full_training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"训练曲线已保存到: {plot_path}")

        except Exception as e:
            print(f"绘制训练曲线失败: {e}")

    def evaluate_on_test_set(self, test_dataloader):
        """在测试集上评估模型"""
        if self.model is None:
            print("✗ 模型未加载")
            return None

        self.model.eval()
        total_loss = 0
        batch_count = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for conv_feats, point_feats, targets in test_dataloader:
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

        if not is_fine_tune:
            print(f"\n  【预训练验证分布】:")
            print(f"    预测值范围: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
            print(f"    预测均值: {all_predictions.mean():.6f} ± {all_predictions.std():.6f}")
            print(f"    目标均值: {all_targets.mean():.6f} ± {all_targets.std():.6f}")
            print(f"    target=0样本数: {np.sum(all_is_zero == 0)} ({np.sum(all_is_zero == 0)/len(all_is_zero)*100:.2f}%)")

        metrics = {
            "loss": avg_loss,
            "rmse": np.sqrt(np.mean((all_predictions - all_targets) ** 2)),
            "mae": np.mean(np.abs(all_predictions - all_targets)),
            "correlation": (
                np.corrcoef(all_predictions, all_targets)[0, 1]
                if len(all_targets) > 1
                else 0
            ),
            "n_samples": len(all_predictions),
        }

        print(f"\n测试集评估结果:")
        print(f"  损失: {metrics['loss']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  相关系数: {metrics['correlation']:.4f}")
        print(f"  样本数: {metrics['n_samples']:,}")

        # 保存评估结果
        eval_results = {
            "metrics": metrics,
            "config": self.config,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        eval_path = self.save_dir / "test_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)

        print(f"评估结果已保存到: {eval_path}")

        return metrics


# ============================================================
# [MAIN] SWETrainer
# ============================================================
# 当前主训练器。
#
# [CONTRACT]
#   负责以下正式流程：
#       - pretrain_cv
#       - pretrain_progressive
#       - fine_tune
#       - station_cv
#       - mixed_mode
#
# [COMPAT]
#   本类需要兼容：
#       - 旧 checkpoint 中可能没有新增 config 字段
#       - 旧 dataset 返回 3/4/5/6/7 个 batch 元素
#       - 旧模型可能 C_point 不同
#       - 旧指标字段可能仍叫 r2
#
# [DANGER]
#   修改 train_epoch / validate / run_pretrain_cv_workflow /
#   run_pretrain_progressive_from_cv 前，先确认当前 mode 实际走哪条分支。
# ============================================================
class SWETrainer:
    """[MAIN] SWE模型训练器，支持微调功能"""

    def __init__(self, config=None):
        # 默认配置
        self.default_config = {
            # 模型类型
            "model_type": "full",

            # 数据参数 - 针对12核CPU优化
            "batch_size": 64,
            "val_ratio": 0.2,
            "num_workers": 10,
            "prefetch_factor": 2,
            "persistent_workers": True,

            # 训练参数
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "patience": 25,

            # ============ 学习率调度 ============
            "lr_scheduler": "plateau",
            "warmup_start_lr": 1e-5,
            "min_lr": 1e-6,
            "warmup_ratio": 0.05,
            "pretrain_cv_max_folds": 10,

            # [RESUME] 预训练十折断点续跑：只跳过已经完整结束的折。
            # 识别标准不是“存在 best.pth” alone，而是完成标记，
            # 或旧实验同时存在 best.pth + 本折曲线 + 本折散点图。
            "resume_pretrain_cv": False,
            "redraw_completed_cv_plots": False,

            # 微调参数
            "fine_tune": False,
            "fine_tune_epochs": 50,
            "fine_tune_lr": 5e-5,
            "freeze_backbone": True,
            "station_data_path": None,

            "lambda_elastic": 0.1,
            "residual_injection": False,

            # 模型参数
            "C_conv": None,
            "C_point": None,
            "d_model": 256,

            # 路径设置
            "save_dir": "./experiments",
            "experiment_name": None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",

            # 其他
            "seed": 43,
            "clip_grad": 1.0,
            "save_freq": 10,

            # PyTorch 2.1.0 优化选项
            "use_amp": True,
            "gradient_accumulation_steps": 1,
            "pin_memory": True,
            "channels_last": True,
            "compile_model": False,

            # ============ Mixed fine-tuning ============
            "mixed_mode": False,
            "station_ratio": 1.0,
            
            "use_product_correction": False,

            # ============ Mixed mode loss 控制 ============
            "pretrain_loss_weight": 0.0,  # ← 改为 0.0
            "use_high_swe_weight": True,

            # SIMPLIFIED_NSE_LOGCOSH_V1
            # 逐样本基础损失：MSE + Log-Cosh；再加高值Bias和轻量方差约束。
            "finetune_mse_weight": 0.7,
            "finetune_logcosh_weight": 0.3,
            "finetune_high_bias_weight": 0.10,
            "finetune_variance_weight": 0.05,

            # SWE_NONNEGATIVE_PHYSICAL_CONSTRAINT_V1
            # 训练时惩罚负SWE；验证/测试时统一截断到物理下界0。
            "finetune_nonnegative_penalty_weight": 2.0,
            "finetune_target_std_ratio": 0.70,
            "finetune_high_bias_threshold_mm": 80.0,
            "finetune_high_bias_min_count": 2,

            # 每个训练batch的SWE分层采样。
            "use_swe_stratified_batches": True,
            "swe_batch_bin_edges_mm": [20.0, 50.0, 80.0],
            "swe_batch_bin_fractions": [0.55, 0.25, 0.125, 0.075],

            # checkpoint复合评分（越小越好）。
            "checkpoint_w_global_rmse": 1.0,
            "checkpoint_w_rmse_ge50": 0.50,
            "checkpoint_w_abs_bias_ge80": 0.30,
            "checkpoint_w_slope": 0.10,
            "checkpoint_w_std_ratio": 0.10,
            "checkpoint_w_correlation": 0.05,

            # FROZEN_RELATIVE_CHECKPOINT_GATE_V1
            # 正式checkpoint必须相对epoch-0 Frozen至少改善RMSE，并且其他
            # 核心指标不得超过数值容差退化。通过门槛的checkpoint再按
            # 综合selection_score排名，避免最低RMSE重新选择压缩预测范围。
            "checkpoint_require_frozen_dominance": True,
            "checkpoint_min_rmse_improvement_mm": 0.01,
            "checkpoint_mae_tolerance_mm": 0.05,
            "checkpoint_nse_tolerance": 1e-4,
            "checkpoint_r_tolerance": 1e-4,
            "checkpoint_abs_bias_tolerance_mm": 0.10,
            "checkpoint_shape_tolerance": 0.02,
            "checkpoint_tail_rmse_tolerance_mm": 0.10,
            "checkpoint_tail_abs_bias_tolerance_mm": 0.50,
            "checkpoint_tail_min_n_ge50": 10,
            "checkpoint_tail_min_n_ge80": 5,
            # 外层fold只负责OOF报告；相邻的另一个fold负责选epoch。
            "nested_station_cv": True,
            "nested_inner_fold_offset": 1,

            # FUSION_FT_STABLE_PROGRESSIVE_V1
            # Fusion FT不再从第一个optimizer step就更新整个4层Transformer。
            # epoch 1只训练回归Head；之后每轮从顶层向底层解冻1层。
            # FUSION_FT_INNER_PILOT_SCHEDULE_V1
            # 训练集拟合诊断表明2e-6长期学习过慢；正式训练改为10轮
            # 线性warmup至3e-5。前5轮仍从顶层向底层逐层解冻，
            # 因而底层刚解冻时不会直接承受峰值学习率。
            "fusion_ft_progressive_unfreeze": True,
            "fusion_ft_head_lr": 5e-5,
            "fusion_ft_transformer_lr": 3e-5,
            "fusion_ft_warmup_epochs": 10,
            "fusion_ft_unfreeze_interval": 1,
            # 0表示保持旧行为：最终解冻全部Transformer层和共享Fusion参数。
            # 正整数N表示只解冻最顶部N个Transformer block；其余层以及
            # CLS/位置编码/输入Norm等共享Fusion参数保持冻结。
            "fusion_ft_max_unfrozen_layers": 0,
            "fusion_ft_plateau_patience": 8,
            # Fusion FT只保留一种温和的高SWE强调：
            # 自然随机采样 + 非累计1.2/1.5/2.0倍权重。
            # 不再叠加强制高值batch和额外high-bias损失。
            "fusion_ft_use_stratified_batches": False,
            "fusion_ft_swe_weight_ge20": 1.2,
            "fusion_ft_swe_weight_ge50": 1.5,
            "fusion_ft_swe_weight_ge80": 2.0,
            "fusion_ft_high_bias_weight": 0.0,
            # FUSION_FT_CCC_V1
            # 只增加一个结构监督：CCC同时约束相关性、均值和预测方差。
            # 替代Fusion FT原先按随机batch计算、经常为0的方差惩罚。
            "fusion_ft_ccc_weight": 0.005,
            "fusion_ft_variance_weight": 0.0,
            # 仅诊断，不改变loss或optimizer：在前5个渐进解冻epoch的
            # 首个batch比较站点基础损失与加权CCC损失的真实参数梯度。
            "fusion_ft_gradient_diagnostic_epochs": 5,
            # 80是建议最大轮数；warmup期间不累计早停。每折至少训练
            # 40轮，之后连续20轮没有新的合格综合最优checkpoint才早停。
            "fusion_ft_patience": 20,
            "fusion_ft_min_epochs_before_early_stop": 40,
            # Inner pilot只运行一个8/1/1划分：outer保留但完全不加载、
            # 不预测、不输出OOF。配置冻结后再从预训练模型启动正式10折。
            "inner_pilot_only": False,
            "nested_max_folds": 10,

            # SWE_FINETUNE_V2_FINAL_GUARD
            "collapse_std_ratio_threshold": 0.05,
            "collapse_pred_std_mm_threshold": 1.0,
            "collapse_max_rollbacks": 1,
            "collapse_early_stop_patience": 2,

            # ============ 预训练样本筛选 ============
            "pretrain_snow_min_mm": 20.0,
            "quality_threshold": 0.83,
            "snow_quality_threshold": 0.60,

            # ============ NSE-oriented loss / prior diagnosis ============
            # Clean-18D 不包含显式产品先验列。只有未来明确把先验加入 point_feats
            # 并配置 physical_prior_col 时，才允许运行 prior ablation。
            "physical_prior_col": None,
            "use_nse_oriented_loss": False,
            "enable_prior_ablation": False,

            # ============ internal augmentation ============
            "use_internal_mix_aug": False,

            # ============ Prior Dropout（暂时关闭） ============
            "use_prior_dropout": False,
            "prior_dropout_p": 0.0,

            # ============ Counterfactual Prior Loss ============
            "use_counterfactual_prior_loss": False,
            "counterfactual_prior_loss_weight": 0.0,
        }
        # 更新配置
        if config:
            self.default_config.update(config)
        self.config = self.default_config

        # 设置实验名称
        if self.config["experiment_name"] is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config["experiment_name"] = (
                f"swe_{self.config['model_type']}_{timestamp}"
            )

        # 设置设备
        self.device = torch.device(self.config["device"])
        print(f"使用设备: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # 初始化变量
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scheduler_step_per_batch = False
        self.criterion = None
        self.train_loader = None
        self.train_audit_loader = None
        self.val_loader = None
        self.test_loader = None  # 用于微调的测试集

        # 训练历史
        self.train_history = []
        self.val_history = []
        self.lr_history = []
        self.fine_tune_history = []  # 微调历史
        
        self.mixup_wrapper = None 

        self.gradient_history = []  # 梯度范数历史
        self.weight_change_history = []  # 权重变化历史
        self.gradient_stats = {  # 梯度统计
            'mean': [],
            'std': [],
            'max': [],
            'min': []
        }
        self.param_groups_stats = []  # 参数组统计

        # 创建保存目录
        self.save_dir = Path(self.config["save_dir"]) / self.config["experiment_name"]
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self._save_config()

        print(f"实验保存目录: {self.save_dir}")

    def _save_config(self):
        """保存配置到文件"""
        config_path = self.save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"配置已保存到: {config_path}")

    def _compute_simplified_nse_logcosh_loss(
        self,
        outputs_flat,
        targets_flat,
        source_flag=None,
    ):
        """
        简化NSE-oriented微调目标：

            weighted [a*MSE + b*LogCosh]
            + lambda_high * high-SWE mean-bias^2
            + lambda_var  * variance-collapse penalty

        所有逐样本误差都在模型的归一化尺度计算；mm只用于划分SWE阈值。
        """
        outputs_flat = outputs_flat.reshape(-1)
        targets_flat = targets_flat.reshape(-1)
        error = outputs_flat - targets_flat

        mse_each = error.pow(2)
        log_cosh_each = torch.logaddexp(error, -error) - math.log(2.0)

        mse_weight = float(self.config.get("finetune_mse_weight", 0.7))
        logcosh_weight = float(self.config.get("finetune_logcosh_weight", 0.3))
        weight_sum = max(mse_weight + logcosh_weight, 1e-8)
        mse_weight /= weight_sum
        logcosh_weight /= weight_sum

        loss_each = mse_weight * mse_each + logcosh_weight * log_cosh_each

        swe_min = float(getattr(self, "swe_min", 0.0))
        swe_max = float(getattr(self, "swe_max", 400.0))
        swe_range = max(swe_max - swe_min, 1e-8)
        target_mm = targets_flat * swe_range + swe_min

        weights = torch.ones_like(loss_each)
        is_stable_fusion_ft = (
            str(
                self.config.get("freeze_strategy", "")
            ).lower() == "fusion_ft"
            and bool(
                self.config.get(
                    "fusion_ft_progressive_unfreeze",
                    True,
                )
            )
        )
        if bool(self.config.get("use_high_swe_weight", True)):
            if is_stable_fusion_ft:
                # FUSION_FT_SINGLE_MILD_TAIL_WEIGHT_V1
                # 非累计分段权重，避免旧2/4/8倍权重再与高值重采样叠加。
                weight_ge20 = float(
                    self.config.get(
                        "fusion_ft_swe_weight_ge20",
                        1.2,
                    )
                )
                weight_ge50 = float(
                    self.config.get(
                        "fusion_ft_swe_weight_ge50",
                        1.5,
                    )
                )
                weight_ge80 = float(
                    self.config.get(
                        "fusion_ft_swe_weight_ge80",
                        2.0,
                    )
                )
                weights = torch.where(
                    target_mm >= 20.0,
                    torch.full_like(weights, weight_ge20),
                    weights,
                )
                weights = torch.where(
                    target_mm >= 50.0,
                    torch.full_like(weights, weight_ge50),
                    weights,
                )
                weights = torch.where(
                    target_mm >= 80.0,
                    torch.full_like(weights, weight_ge80),
                    weights,
                )
            else:
                weights = weights + 1.0 * (target_mm >= 20.0).float()
                weights = weights + 2.0 * (target_mm >= 50.0).float()
                weights = weights + 4.0 * (target_mm >= 80.0).float()

        station_mask = torch.ones_like(targets_flat, dtype=torch.bool)
        if source_flag is not None:
            source_flag = source_flag.reshape(-1).long()
            if source_flag.numel() == targets_flat.numel():
                station_mask = source_flag == 0
                pretrain_mask = source_flag == 1
                pretrain_loss_weight = float(
                    self.config.get("pretrain_loss_weight", 0.0)
                )
                weights = torch.where(
                    pretrain_mask,
                    torch.full_like(weights, pretrain_loss_weight),
                    weights,
                )

        weights = weights / weights.mean().clamp_min(1e-6)
        base_loss = (loss_each * weights).mean()

        high_threshold = float(
            self.config.get("finetune_high_bias_threshold_mm", 80.0)
        )
        high_min_count = int(
            self.config.get("finetune_high_bias_min_count", 2)
        )
        high_mask = station_mask & (target_mm >= high_threshold)

        if int(high_mask.sum().item()) >= high_min_count:
            high_bias_loss = error[high_mask].mean().pow(2)
        else:
            high_bias_loss = torch.zeros((), device=error.device, dtype=error.dtype)

        metric_mask = station_mask
        if int(metric_mask.sum().item()) >= 2:
            metric_targets = targets_flat[metric_mask]
            metric_outputs = outputs_flat[metric_mask]
            target_std = metric_targets.std(unbiased=False)
            pred_std = metric_outputs.std(unbiased=False)
            target_std_ratio = float(
                self.config.get("finetune_target_std_ratio", 0.70)
            )
            if float(target_std.detach().item()) > 1e-8:
                std_ratio = pred_std / target_std.clamp_min(1e-8)
                # SWE_FINETUNE_V2_STD_RATIO_LOSS
                variance_loss = torch.relu(
                    torch.as_tensor(
                        target_std_ratio,
                        device=error.device,
                        dtype=error.dtype,
                    ) - std_ratio
                ).pow(2)

                # FUSION_FT_CCC_V1
                # Lin's concordance correlation coefficient:
                # 同时惩罚低相关、均值偏差和预测范围压缩。
                target_mean = metric_targets.mean()
                pred_mean = metric_outputs.mean()
                target_centered = metric_targets - target_mean
                pred_centered = metric_outputs - pred_mean
                covariance = (
                    target_centered * pred_centered
                ).mean()
                target_variance = target_centered.pow(2).mean()
                pred_variance = pred_centered.pow(2).mean()
                ccc_denominator = (
                    target_variance
                    + pred_variance
                    + (target_mean - pred_mean).pow(2)
                ).clamp_min(1e-8)
                ccc = 2.0 * covariance / ccc_denominator
                ccc_loss = 1.0 - ccc
            else:
                std_ratio = torch.ones((), device=error.device, dtype=error.dtype)
                variance_loss = torch.zeros((), device=error.device, dtype=error.dtype)
                ccc = torch.zeros((), device=error.device, dtype=error.dtype)
                ccc_loss = torch.zeros((), device=error.device, dtype=error.dtype)
        else:
            target_std = torch.zeros((), device=error.device, dtype=error.dtype)
            pred_std = torch.zeros((), device=error.device, dtype=error.dtype)
            std_ratio = torch.zeros((), device=error.device, dtype=error.dtype)
            variance_loss = torch.zeros((), device=error.device, dtype=error.dtype)
            ccc = torch.zeros((), device=error.device, dtype=error.dtype)
            ccc_loss = torch.zeros((), device=error.device, dtype=error.dtype)

        # SWE_NONNEGATIVE_PHYSICAL_CONSTRAINT_V1
        # 只根据模型输出本身施加物理约束，不读取真实标签是否为0。
        # 在训练阶段不直接clamp，避免负区间梯度被截断。
        negative_mask = station_mask

        if int(negative_mask.sum().item()) == 0:
            negative_mask = torch.ones_like(
                outputs_flat,
                dtype=torch.bool,
            )

        negative_part = torch.relu(
            -outputs_flat[negative_mask]
        )

        if negative_part.numel() > 0:
            nonnegative_loss = negative_part.pow(2).mean()
        else:
            nonnegative_loss = torch.zeros(
                (),
                device=error.device,
                dtype=error.dtype,
            )

        nonnegative_weight = float(
            self.config.get(
                "finetune_nonnegative_penalty_weight",
                2.0,
            )
        )

        if is_stable_fusion_ft:
            high_bias_weight = float(
                self.config.get(
                    "fusion_ft_high_bias_weight",
                    0.0,
                )
            )
            variance_weight = float(
                self.config.get(
                    "fusion_ft_variance_weight",
                    0.0,
                )
            )
            ccc_weight = float(
                self.config.get(
                    "fusion_ft_ccc_weight",
                    0.005,
                )
            )
        else:
            high_bias_weight = float(
                self.config.get("finetune_high_bias_weight", 0.10)
            )
            variance_weight = float(
                self.config.get("finetune_variance_weight", 0.05)
            )
            ccc_weight = 0.0

        loss = (
            base_loss
            + high_bias_weight * high_bias_loss
            + variance_weight * variance_loss
            + ccc_weight * ccc_loss
            + nonnegative_weight * nonnegative_loss
        )

        diagnostics = {
            "base_loss": base_loss.detach(),
            "high_bias_loss": high_bias_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "ccc": ccc.detach(),
            "ccc_loss": ccc_loss.detach(),
            "ccc_weight": ccc_weight,
            "variance_weight": variance_weight,
            # 仅供首batch梯度诊断；不要detach，否则无法比较真实梯度。
            "_base_loss_tensor": base_loss,
            "_weighted_ccc_loss_tensor": ccc_weight * ccc_loss,
            "nonnegative_loss": nonnegative_loss.detach(),
            "negative_count": int(
                (outputs_flat[negative_mask] < 0).sum().item()
            ),
            "target_std": target_std.detach(),
            "pred_std": pred_std.detach(),
            "std_ratio": std_ratio.detach(),
            "high_count": int(high_mask.sum().item()),
        }
        return loss, diagnostics

    def _make_swe_stratified_train_loader(
        self,
        dataset,
        targets_mm,
        *,
        context="fine_tune",
    ):
        """创建保证每批含中高SWE样本的训练DataLoader。"""
        use_stratified = bool(
            self.config.get("use_swe_stratified_batches", True)
        )
        if (
            str(
                self.config.get("freeze_strategy", "")
            ).lower() == "fusion_ft"
            and bool(
                self.config.get(
                    "fusion_ft_progressive_unfreeze",
                    True,
                )
            )
        ):
            use_stratified = bool(
                self.config.get(
                    "fusion_ft_use_stratified_batches",
                    False,
                )
            )

        loader_seed = int(
            self.config.get(
                "_active_dataloader_seed",
                self.config.get("seed", 43),
            )
        )
        common_kwargs = {
            "num_workers": self.config.get("num_workers", 8),
            "pin_memory": True,
            **_dataloader_seed_kwargs(loader_seed),
        }
        drop_last = not bool(
            self.config.get("train_only_overfit", False)
        )

        if not use_stratified:
            if str(
                self.config.get("freeze_strategy", "")
            ).lower() == "fusion_ft":
                print(
                    f"  ✅ [{context}] Fusion FT使用自然随机采样，"
                    "不再强制每batch重复高SWE样本"
                )
            return DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                drop_last=drop_last,
                **common_kwargs,
            )

        sampler = SWEStratifiedBatchSampler(
            targets_mm=targets_mm,
            batch_size=self.config["batch_size"],
            seed=loader_seed,
            drop_last=drop_last,
            bin_edges=self.config.get(
                "swe_batch_bin_edges_mm",
                [20.0, 50.0, 80.0],
            ),
            bin_fractions=self.config.get(
                "swe_batch_bin_fractions",
                [0.55, 0.25, 0.125, 0.075],
            ),
        )

        print(f"  ✅ [{context}] 启用SWE分层batch采样")
        print(f"     每batch配额/可用样本: {sampler.describe()}")
        print(f"     每epoch batch数: {len(sampler)}")

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            **common_kwargs,
        )

    @staticmethod
    def _safe_regression_slope(targets, predictions):
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if targets.size < 2:
            return float("nan")
        centered = targets - targets.mean()
        denom = float(np.sum(centered ** 2))
        if denom <= 1e-12:
            return float("nan")
        return float(np.sum(centered * (predictions - predictions.mean())) / denom)

    def _compute_finetune_selection_metrics(self, predictions_norm, targets_norm):
        """计算每epoch高值诊断和checkpoint复合评分。"""
        predictions_norm = np.asarray(predictions_norm, dtype=np.float64).reshape(-1)
        targets_norm = np.asarray(targets_norm, dtype=np.float64).reshape(-1)

        swe_min = float(getattr(self, "swe_min", 0.0))
        swe_max = float(getattr(self, "swe_max", 400.0))
        swe_range = max(swe_max - swe_min, 1e-8)

        predictions_mm = predictions_norm * swe_range + swe_min
        targets_mm = targets_norm * swe_range + swe_min
        error_norm = predictions_norm - targets_norm
        error_mm = predictions_mm - targets_mm

        rmse_norm = float(np.sqrt(np.mean(error_norm ** 2)))
        rmse_mm = float(np.sqrt(np.mean(error_mm ** 2)))
        mae_mm = float(np.mean(np.abs(error_mm)))
        bias_mm = float(np.mean(error_mm))

        mask50 = targets_mm >= 50.0
        mask80 = targets_mm >= 80.0

        if np.any(mask50):
            rmse_ge50_norm = float(np.sqrt(np.mean(error_norm[mask50] ** 2)))
            rmse_ge50_mm = float(np.sqrt(np.mean(error_mm[mask50] ** 2)))
        else:
            rmse_ge50_norm = rmse_norm
            rmse_ge50_mm = float("nan")

        if np.any(mask80):
            bias_ge80_norm = float(np.mean(error_norm[mask80]))
            bias_ge80_mm = float(np.mean(error_mm[mask80]))
        else:
            bias_ge80_norm = 0.0
            bias_ge80_mm = float("nan")

        slope = self._safe_regression_slope(targets_mm, predictions_mm)
        target_std_mm = float(np.std(targets_mm))
        pred_std_mm = float(np.std(predictions_mm))
        std_ratio = pred_std_mm / target_std_mm if target_std_mm > 1e-8 else float("nan")

        if targets_norm.size > 1 and np.std(targets_norm) > 1e-12 and np.std(predictions_norm) > 1e-12:
            correlation = float(np.corrcoef(predictions_norm, targets_norm)[0, 1])
        else:
            correlation = 0.0

        # TRAIN_VS_INNER_AUDIT_V1
        # 这里计算的是整个评估集合上的CCC，而不是训练时的随机batch CCC。
        # 它只用于诊断，不参与checkpoint评分或Frozen-relative硬门槛。
        target_mean = float(np.mean(targets_norm))
        pred_mean = float(np.mean(predictions_norm))
        target_centered = targets_norm - target_mean
        pred_centered = predictions_norm - pred_mean
        target_variance = float(np.mean(target_centered ** 2))
        pred_variance = float(np.mean(pred_centered ** 2))
        covariance = float(np.mean(target_centered * pred_centered))
        ccc_denominator = (
            target_variance
            + pred_variance
            + (target_mean - pred_mean) ** 2
        )
        if ccc_denominator > 1e-12:
            ccc = float(2.0 * covariance / ccc_denominator)
        else:
            ccc = float("nan")

        slope_penalty = min(abs(1.0 - slope), 2.0) if np.isfinite(slope) else 2.0
        std_penalty = min(abs(1.0 - std_ratio), 2.0) if np.isfinite(std_ratio) else 2.0
        corr_penalty = 1.0 - float(np.clip(correlation, -1.0, 1.0))

        selection_score = (
            float(self.config.get("checkpoint_w_global_rmse", 1.0)) * rmse_norm
            + float(self.config.get("checkpoint_w_rmse_ge50", 0.50)) * rmse_ge50_norm
            + float(self.config.get("checkpoint_w_abs_bias_ge80", 0.30)) * abs(bias_ge80_norm)
            + float(self.config.get("checkpoint_w_slope", 0.10)) * slope_penalty
            + float(self.config.get("checkpoint_w_std_ratio", 0.10)) * std_penalty
            + float(self.config.get("checkpoint_w_correlation", 0.05)) * corr_penalty
        )

        # FINETUNE_COLLAPSE_METRIC_V1
        # 只识别“近常数输出”，不把普通的低斜率模型误判为塌缩。
        collapse_std_ratio = float(
            self.config.get("collapse_std_ratio_threshold", 0.05)
        )
        collapse_pred_std_mm = float(
            self.config.get("collapse_pred_std_mm_threshold", 1.0)
        )
        is_collapsed = bool(
            (not np.isfinite(pred_std_mm))
            or pred_std_mm < collapse_pred_std_mm
            or (
                np.isfinite(std_ratio)
                and std_ratio < collapse_std_ratio
                and abs(float(slope)) < 0.05
            )
        )

        return {
            "rmse_mm": rmse_mm,
            "mae_mm": mae_mm,
            "bias_mm": bias_mm,
            "rmse_ge50_mm": rmse_ge50_mm,
            "bias_ge80_mm": bias_ge80_mm,
            "n_ge50": int(mask50.sum()),
            "n_ge80": int(mask80.sum()),
            "slope": float(slope),
            "pred_std_mm": pred_std_mm,
            "target_std_mm": target_std_mm,
            "std_ratio": float(std_ratio),
            "ccc": float(ccc),
            "selection_score": float(selection_score),
            "is_collapsed": bool(is_collapsed),
        }

    def _run_train_vs_inner_audit(self, inner_metrics, epoch_label):
        """
        在固定训练审计子集上做纯前向评估，并与inner-validation并排输出。

        该结果不传给scheduler、checkpoint选择或early stopping，只回答：
        训练样本上的结构指标是否改善，以及这种改善能否迁移到未见站点。
        """
        audit_loader = getattr(self, "train_audit_loader", None)
        if audit_loader is None:
            return None

        audit_metrics = self.validate(
            dataloader=audit_loader,
            is_fine_tune=True,
            report_name="固定训练审计集",
        )

        print(
            f"\n  🔬 [Train-vs-Inner] {epoch_label}\n"
            "     Train-audit: "
            f"R={audit_metrics.get('correlation', float('nan')):.4f}, "
            f"CCC={audit_metrics.get('ccc', float('nan')):.4f}, "
            f"slope={audit_metrics.get('slope', float('nan')):.4f}, "
            f"std_ratio={audit_metrics.get('std_ratio', float('nan')):.4f}\n"
            "     Inner-val:   "
            f"R={inner_metrics.get('correlation', float('nan')):.4f}, "
            f"CCC={inner_metrics.get('ccc', float('nan')):.4f}, "
            f"slope={inner_metrics.get('slope', float('nan')):.4f}, "
            f"std_ratio={inner_metrics.get('std_ratio', float('nan')):.4f}"
        )
        return audit_metrics

    def _assess_frozen_relative_checkpoint(self, candidate, frozen_baseline):
        """
        判断微调epoch能否成为正式checkpoint。

        规则：
        1. 全局RMSE必须严格优于同一选模集上的epoch-0 Frozen；
        2. MAE、NSE、R、|Bias|以及分布形态不得超过容差退化；
        3. 高SWE样本数达到最低门槛时，尾部指标同样不得退化。

        外部测试集绝不参与此函数。
        """
        if not bool(
            self.config.get("checkpoint_require_frozen_dominance", True)
        ):
            score = float(candidate.get("selection_score", float("inf")))
            return {
                "admissible": bool(np.isfinite(score)),
                "control_score": score,
                "violations": [],
                "checked_metrics": {},
            }

        violations = []
        checked = {}

        def finite_value(metrics, key):
            value = float(metrics.get(key, float("nan")))
            return value if np.isfinite(value) else None

        def require_minimize(key, tolerance, *, strict_improvement=0.0):
            current = finite_value(candidate, key)
            baseline = finite_value(frozen_baseline, key)
            checked[key] = {"candidate": current, "frozen": baseline}
            if current is None or baseline is None:
                violations.append(f"{key}:non_finite")
                return
            limit = baseline + float(tolerance) - float(strict_improvement)
            if current > limit:
                violations.append(
                    f"{key}:{current:.6f}>{limit:.6f}"
                )

        def require_maximize(key, tolerance):
            current = finite_value(candidate, key)
            baseline = finite_value(frozen_baseline, key)
            checked[key] = {"candidate": current, "frozen": baseline}
            if current is None or baseline is None:
                violations.append(f"{key}:non_finite")
                return
            limit = baseline - float(tolerance)
            if current < limit:
                violations.append(
                    f"{key}:{current:.6f}<{limit:.6f}"
                )

        def require_abs_minimize(key, tolerance):
            current = finite_value(candidate, key)
            baseline = finite_value(frozen_baseline, key)
            checked[f"abs_{key}"] = {
                "candidate": abs(current) if current is not None else None,
                "frozen": abs(baseline) if baseline is not None else None,
            }
            if current is None or baseline is None:
                violations.append(f"abs_{key}:non_finite")
                return
            limit = abs(baseline) + float(tolerance)
            if abs(current) > limit:
                violations.append(
                    f"abs_{key}:{abs(current):.6f}>{limit:.6f}"
                )

        min_rmse_gain = float(
            self.config.get("checkpoint_min_rmse_improvement_mm", 0.01)
        )
        require_minimize(
            "rmse_mm",
            tolerance=0.0,
            strict_improvement=min_rmse_gain,
        )
        require_minimize(
            "mae_mm",
            self.config.get("checkpoint_mae_tolerance_mm", 0.05),
        )
        require_maximize(
            "r2",
            self.config.get("checkpoint_nse_tolerance", 1e-4),
        )
        require_maximize(
            "correlation",
            self.config.get("checkpoint_r_tolerance", 1e-4),
        )
        require_abs_minimize(
            "bias_mm",
            self.config.get("checkpoint_abs_bias_tolerance_mm", 0.10),
        )

        shape_tolerance = float(
            self.config.get("checkpoint_shape_tolerance", 0.02)
        )
        for key in ("slope", "std_ratio"):
            current = finite_value(candidate, key)
            baseline = finite_value(frozen_baseline, key)
            checked[f"distance_to_one_{key}"] = {
                "candidate": (
                    abs(1.0 - current) if current is not None else None
                ),
                "frozen": (
                    abs(1.0 - baseline) if baseline is not None else None
                ),
            }
            if current is None or baseline is None:
                violations.append(f"{key}:non_finite")
            elif (
                abs(1.0 - current)
                > abs(1.0 - baseline) + shape_tolerance
            ):
                violations.append(
                    f"{key}_distance_to_one:"
                    f"{abs(1.0-current):.6f}>"
                    f"{abs(1.0-baseline)+shape_tolerance:.6f}"
                )

        n_ge50 = min(
            int(candidate.get("n_ge50", 0)),
            int(frozen_baseline.get("n_ge50", 0)),
        )
        if n_ge50 >= int(
            self.config.get("checkpoint_tail_min_n_ge50", 10)
        ):
            require_minimize(
                "rmse_ge50_mm",
                self.config.get(
                    "checkpoint_tail_rmse_tolerance_mm",
                    0.10,
                ),
            )
        else:
            checked["rmse_ge50_mm"] = {
                "skipped": True,
                "n": n_ge50,
            }

        n_ge80 = min(
            int(candidate.get("n_ge80", 0)),
            int(frozen_baseline.get("n_ge80", 0)),
        )
        if n_ge80 >= int(
            self.config.get("checkpoint_tail_min_n_ge80", 5)
        ):
            require_abs_minimize(
                "bias_ge80_mm",
                self.config.get(
                    "checkpoint_tail_abs_bias_tolerance_mm",
                    0.50,
                ),
            )
        else:
            checked["abs_bias_ge80_mm"] = {
                "skipped": True,
                "n": n_ge80,
            }

        candidate_rmse = finite_value(candidate, "rmse_mm")
        candidate_selection_score = finite_value(
            candidate,
            "selection_score",
        )
        baseline_selection_score = finite_value(
            frozen_baseline,
            "selection_score",
        )
        admissible = len(violations) == 0

        if admissible and candidate_selection_score is not None:
            control_score = candidate_selection_score
        else:
            reference = (
                baseline_selection_score
                if baseline_selection_score is not None
                else 1e6
            )
            # 未通过硬门槛的epoch一定排在Frozen之后；违反项目越多越差。
            control_score = reference + 1000.0 + 100.0 * len(violations)

        return {
            "admissible": bool(admissible),
            "control_score": float(control_score),
            "candidate_rmse_mm": candidate_rmse,
            "candidate_selection_score": candidate_selection_score,
            "violations": violations,
            "checked_metrics": checked,
        }

    def setup_chinese_fonts(self):
        """设置中文字体，解决乱码问题"""

        
        try:
            # 检查系统类型
            system = platform.system()
            
            if system == 'Linux':
                print("Setting up Chinese fonts on Linux...")
                
                # 直接指定文泉驿字体路径
                wqy_paths = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                    '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
                    '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
                ]
                
                for font_path in wqy_paths:
                    if os.path.exists(font_path):
                        print(f"Found Chinese font: {font_path}")
                        # 添加字体
                        fm.fontManager.addfont(font_path)
                        # 获取字体名称
                        font_prop = fm.FontProperties(fname=font_path)
                        font_name = font_prop.get_name()
                        
                        # 更新matplotlib配置
                        matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
                        matplotlib.rcParams['axes.unicode_minus'] = False
                        
                        print(f"Successfully set Chinese font: {font_name}")
                        return True
                
                # 如果找不到，使用英文
                print("Chinese fonts not found, using English fonts")
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
                matplotlib.rcParams['axes.unicode_minus'] = False
                
            else:
                # Windows或其他系统
                matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
                matplotlib.rcParams['axes.unicode_minus'] = False
                
        except Exception as e:
            print(f"Error setting up Chinese fonts: {e}")
            # 回退到英文
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
            matplotlib.rcParams['axes.unicode_minus'] = False
        
        return False


    # ============================================================
    # [ROUTER] load_data()
    # ============================================================
    # 根据 fine_tune_mode / mixed_mode / split_method 路由到不同数据加载逻辑。
    #
    # [CONTRACT]
    #   - mixed_mode=True:
    #       站点样本 + 预训练伪标签样本混合。
    #       如果 station_csv 有 split 列，split='test' 必须只作为固定测试集。
    #
    #   - fine_tune_mode=True:
    #       只使用站点实测 SWE。
    #       station_cv 时，split!='test' 作为 CV 池，split='test' 固定独立测试。
    #
    #   - 默认预训练：
    #       使用 data_online_era5_swe.py 构建 ERA5-Land SWE 伪标签样本。
    #
    # [DANGER]
    #   固定 test 站点不能进入 train/val/mixed/pretrain_aux。
    #   否则站点泄漏，测试结果会虚高。
    # ============================================================
    def load_data(self, fine_tune_mode=False, mixed_mode=False, station_ratio=0.5):
        """
        加载数据 - 支持：
        1. 普通预训练模式
        2. 纯站点微调模式
        3. mixed mode：站点样本 + 预训练样本回放约束

        关键逻辑：
        - 如果 station_csv 中存在 split 列，且 cv_mode == 'station_cv':
            split == 'test'       -> 固定独立测试集，永远不参与训练/验证/mixed
            split != 'test'       -> 训练/验证池，用于 station_cv 十折
        - mixed mode 下：
            只在 split != 'test' 的训练/验证池中加入预训练样本
            test_loader 永远由 split == 'test' 的 StationSWEDataset 构建
        """

        print("\n" + "=" * 60)

        # ============ 判断模式 ============
        if mixed_mode:
            print("加载混合数据（站点 + 预训练样本）...")
            use_mixed_mode = True
            use_fine_tune_mode = False
        elif fine_tune_mode or (hasattr(self, "config") and self.config.get("fine_tune", False)):
            print("加载微调数据（站点数据）...")
            use_mixed_mode = False
            use_fine_tune_mode = True
        else:
            print("加载预训练数据...")
            use_mixed_mode = False
            use_fine_tune_mode = False

        print("=" * 60)

        # ============ 共享缓存目录 ============
        shared_cache_dir = self.config.get("shared_cache_dir", "/root/autodl-tmp/shared_cache")
        cache_dir = Path(shared_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 共享缓存目录: {cache_dir}")

        # ============ 站点引导采样配置 ============
        use_station_guide = self.config.get("use_station_guide", False)
        station_neighborhood = self.config.get("station_neighborhood", 3)
        station_samples_per_day = self.config.get("station_samples_per_day", 2000)
        station_sampling_unit = self.config.get(
            "station_sampling_unit", "positions_all_dates"
        )
        station_record_dedup = self.config.get("station_record_dedup", "grid_date")
        station_date_column = self.config.get("station_date_column")

        # ============ 划分缓存配置 ============
        split_cache_file = self.config.get("split_cache_file", None)
        force_recompute_split = self.config.get("force_recompute_split", False)

        if split_cache_file:
            print(f"📦 划分缓存文件: {split_cache_file}")
            if force_recompute_split:
                print("   ⚠️ 强制重新计算划分模式")

        if use_station_guide:
            print("\n📍 站点引导采样: 已启用")
            print(f"   邻域半径: {station_neighborhood} ({station_neighborhood * 2 + 1}x{station_neighborhood * 2 + 1})")
            print(f"   每日站点样本上限: {station_samples_per_day}")

        # ============ 通用 dataset 参数 ============
        def make_station_dataset_params():
            return {
                "region": "CHINA",
                "year_target": [2015, 2016, 2017, 2018],
                "patch_size": 5,
                "clamday_threshold": 0.5,
                "s1_interp_method": "nearest",
                "s1_max_gap_days": 7,
                "s1_nodata_value": -9999.0,
                "smap_interp_method": "nearest",
                "smap_max_gap_days": 7,
                "smap_nodata_value": -9999.0,
                "cache_dir": cache_dir,
                "use_station_guide": use_station_guide,
                "station_neighborhood": station_neighborhood,
                "station_samples_per_day": station_samples_per_day,
                "split_cache_file": split_cache_file,
                "force_recompute_split": force_recompute_split,
                # PROGRESSIVE_STATION_SHARED_FEATURE_CACHE_V1
                "shared_cache_mode": (
                    self.config.get("shared_cache_mode", False)
                    or self.config.get("cv_mode") == "station_cv"
                ),
                "use_product_correction": self.config.get("use_product_correction", False),

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

        def make_pretrain_dataset_params():
            return {
                "region": "CHINA",
                "year_target": 2016,
                "patch_size": 5,
                "min_valid_pixels": 100,
                "samples_per_day": 10000,
                "clamday_threshold": 0.5,
                "s1_interp_method": "nearest",
                "s1_max_gap_days": 7,
                "s1_nodata_value": -9999.0,
                "smap_interp_method": "nearest",
                "smap_max_gap_days": 7,
                "smap_nodata_value": -9999.0,
                "cache_dir": cache_dir,
                "use_station_guide": use_station_guide,
                "station_neighborhood": station_neighborhood,
                "station_samples_per_day": station_samples_per_day,
                "sampling_mode": self.config.get("sampling_mode", "auto"),
                "station_guide_file": self.config.get("station_guide_file"),
            "station_record_manifest_path": self.config.get("station_record_manifest_path"),
                "station_sampling_unit": station_sampling_unit,
                "station_record_dedup": station_record_dedup,
                "station_date_column": station_date_column,
                "station_filter_zero_target": self.config.get(
                    "station_filter_zero_target", True
                ),
                "use_adaptive_supplement": self.config.get("use_adaptive_supplement", False),
                "adaptive_alpha": self.config.get("adaptive_alpha", 0.5),
                "adaptive_threshold": self.config.get("adaptive_threshold", 1.5),
                "pretrain_snow_priority_ratio": self.config.get("pretrain_snow_priority_ratio", 1.0),
            }

        # ============ 读取 / 合并站点文件 ============
        def resolve_station_data_source(station_data_path):
            station_data_path = Path(station_data_path)

            if station_data_path.is_dir():
                print(f"  检测到目录路径: {station_data_path}")

                target_files = [
                    "station_swe_data.xlsx",
                    "station_swe_data.xls",
                    "station_swe_data.csv",
                    "long_comb.csv",
                    "long_comb2.csv",
                    "long_comb3.csv",
                    "one_record.csv",
                    "*.xlsx",
                    "*.xls",
                    "*.csv",
                ]

                found_files = []

                for filename_pattern in target_files:
                    if "*" in filename_pattern:
                        pattern_files = list(station_data_path.glob(filename_pattern))
                        pattern_files = [f for f in pattern_files if f not in found_files]
                        found_files.extend(pattern_files)
                        if pattern_files:
                            print(f"  找到 {len(pattern_files)} 个 {filename_pattern} 文件")
                    else:
                        file_path = station_data_path / filename_pattern
                        if file_path.exists():
                            found_files.append(file_path)
                            print(f"  找到数据文件: {filename_pattern}")

                found_files = list(set(found_files))

                if not found_files:
                    print("  ✗ 目录中没有找到任何数据文件")
                    return None

                if len(found_files) == 1:
                    print(f"  使用单个数据文件: {found_files[0].name}")
                    return found_files[0]

                print("\n  发现多个数据文件，正在合并...")
                all_dfs = []

                for file_path in found_files:
                    try:
                        print(f"    正在读取: {file_path.name}...")
                        file_ext = file_path.suffix.lower()

                        if file_ext in [".xlsx", ".xls"]:
                            df = pd.read_excel(file_path, engine="openpyxl")
                        elif file_ext == ".csv":
                            try:
                                df = pd.read_csv(file_path, encoding="utf-8")
                            except UnicodeDecodeError:
                                try:
                                    df = pd.read_csv(file_path, encoding="gbk")
                                except UnicodeDecodeError:
                                    df = pd.read_csv(file_path, encoding="latin1")
                        else:
                            continue

                        column_mapping = {
                            "longtitude": "longitude",
                            "lon": "longitude",
                            "lng": "longitude",
                            "long": "longitude",
                            "latitude": "latitude",
                            "lat": "latitude",
                            "swe": "swe",
                            "swedepth": "swe",
                            "swe_depth": "swe",
                            "swe_mm": "swe",
                            "swe_value": "swe",
                            "value": "swe",
                            "date": "date",
                            "time": "date",
                            "datetime": "date",
                            "station_id": "station_id",
                            "station": "station_id",
                            "id": "station_id",
                            "stationid": "station_id",
                            "site_id": "station_id",
                        }

                        df = df.rename(columns=lambda x: column_mapping.get(str(x).strip().lower(), x))

                        required_cols = ["station_id", "date", "swe", "longitude", "latitude"]
                        missing_cols = [col for col in required_cols if col not in df.columns]

                        if missing_cols:
                            print(f"      跳过: 缺少列 {missing_cols}")
                            continue

                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df = df.dropna(subset=["date"])
                        df = df[df["date"].dt.year.isin([2015, 2016, 2017])].copy()

                        if len(df) == 0:
                            print("      跳过: 无2015-2017年数据")
                            continue

                        df = df.dropna(subset=["swe"])
                        df = df[df["swe"] >= 0]
                        df = df.dropna(subset=["longitude", "latitude"])

                        # 中国范围粗过滤，防止异常经纬度
                        df = df[
                            (df["longitude"] >= 73) &
                            (df["longitude"] <= 135) &
                            (df["latitude"] >= 18) &
                            (df["latitude"] <= 54)
                        ].copy()

                        if len(df) == 0:
                            print("      跳过: 中国范围内无有效数据")
                            continue

                        df["data_source"] = file_path.name
                        all_dfs.append(df)
                        print(f"      成功: {len(df)} 行")

                    except Exception as e:
                        print(f"      读取 {file_path.name} 失败: {e}")
                        continue

                if not all_dfs:
                    print("  ✗ 没有有效数据可合并")
                    return None

                combined_df = pd.concat(all_dfs, ignore_index=True)

                before_dedup = len(combined_df)
                combined_df = combined_df.drop_duplicates(
                    subset=["station_id", "date", "longitude", "latitude"]
                )
                after_dedup = len(combined_df)
                print(f"    去重: {before_dedup} -> {after_dedup}")

                combined_df = combined_df.dropna(subset=["swe", "date", "longitude", "latitude"])
                combined_df = combined_df.sort_values(
                    by=["station_id", "date", "longitude", "latitude"],
                    ignore_index=True
                )

                temp_dir = self.save_dir / "temp_data"
                temp_dir.mkdir(parents=True, exist_ok=True)

                file_list_str = ",".join(sorted([str(f) for f in found_files]))
                file_hash = hashlib.md5(file_list_str.encode()).hexdigest()[:12]

                combined_file = temp_dir / f"combined_station_data_{file_hash}.csv"
                combined_df.to_csv(combined_file, index=False, encoding="utf-8")

                print(f"    创建临时文件: {combined_file}")
                return combined_file

            else:
                if not station_data_path.exists():
                    print(f"✗ 指定的文件不存在: {station_data_path}")
                    return None

                return station_data_path

        # ============ 保存 split 信息 ============
        def save_basic_split_info(split_records, filename_prefix="split_info"):
            try:
                splits_dir = self.save_dir / "splits"
                splits_dir.mkdir(parents=True, exist_ok=True)

                df_splits = pd.DataFrame(split_records)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = splits_dir / f"{filename_prefix}_{timestamp}.csv"
                latest_path = splits_dir / f"{filename_prefix}_latest.csv"

                df_splits.to_csv(csv_path, index=False, encoding="utf-8-sig")
                df_splits.to_csv(latest_path, index=False, encoding="utf-8-sig")

                print(f"\n💾 划分信息已保存: {csv_path}")
                print(f"   最新版本: {latest_path}")

            except Exception as e:
                print(f"  ⚠ 保存划分信息失败: {e}")
                traceback.print_exc()

        try:
            # ============================================================
            # 1. Mixed mode：站点 + 预训练样本
            # ============================================================
            if use_mixed_mode:
                if not STATION_MODULE_AVAILABLE:
                    print("✗ 站点数据模块不可用，无法进行混合训练")
                    return False

                station_data_path = self.config.get("station_data_path")
                if not station_data_path:
                    print("✗ 未指定站点数据路径，请设置 station_data_path 参数")
                    return False

                main_data_source = resolve_station_data_source(station_data_path)
                if main_data_source is None:
                    return False

                print(f"\n  最终站点数据源: {main_data_source}")

                dataset_params = make_station_dataset_params()

                try:
                    from data_station_online_swe import (
                        build_mixed_dataloaders,
                        StationSWEDataset,
                    )
                    print("  ✓ 成功导入 mixed 数据集模块")
                except ImportError as e:
                    print(f"✗ mixed 数据集模块未找到: {e}")
                    return False

                df_check = pd.read_csv(main_data_source, nrows=5)
                has_split_col = "split" in df_check.columns
                cv_mode = self.config.get("cv_mode", "standard")

                # ============ mixed + split列 + station_cv ============
                if has_split_col and cv_mode == "station_cv":
                    print("\n   ✅ mixed_mode 检测到 split 列 + station_cv")
                    print("      split='test' → 固定独立测试集，不参与 mixed 训练")
                    print("      split!='test' → 训练/验证池，用于 station_cv 十折 + 预训练样本混合")

                    df_full = pd.read_csv(main_data_source)
                    df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
                    df_full = df_full.dropna(subset=["date"])

                    print("\n   📊 split 列分布:")
                    for split_name, count in df_full["split"].value_counts().items():
                        print(f"      {split_name}: {count} 条记录")

                    df_test = df_full[df_full["split"] == "test"].copy()
                    df_train_pool = df_full[df_full["split"] != "test"].copy()

                    if len(df_test) == 0:
                        print("   ✗ split='test' 为空，无法构建固定独立测试集")
                        return False

                    if len(df_train_pool) == 0:
                        print("   ✗ split!='test' 训练/验证池为空")
                        return False

                    print("\n   📊 mixed 数据划分:")
                    print(f"      训练/验证池: {len(df_train_pool)} 条, {df_train_pool['station_id'].nunique()} 站点")
                    print(f"      固定测试集: {len(df_test)} 条, {df_test['station_id'].nunique()} 站点")

                    def print_swe_stats(name, df):
                        if "swe" not in df.columns or len(df) == 0:
                            return
                        high_n = int((df["swe"] >= 80).sum())
                        high_ratio = float((df["swe"] >= 80).mean() * 100)
                        print(
                            f"      {name}: mean={df['swe'].mean():.2f}, "
                            f"max={df['swe'].max():.2f}, "
                            f"obs>=80: {high_n} ({high_ratio:.2f}%)"
                        )

                    print_swe_stats("train_pool", df_train_pool)
                    print_swe_stats("fixed_test", df_test)

                    temp_dir = self.save_dir / "temp_data"
                    temp_dir.mkdir(parents=True, exist_ok=True)

                    train_pool_file = temp_dir / "mixed_train_pool.csv"
                    test_file = temp_dir / "mixed_test_split.csv"

                    df_train_pool.to_csv(train_pool_file, index=False)
                    df_test.to_csv(test_file, index=False)

                    # 只在 train_pool 上构建 mixed dataset
                    print("\n   🔧 构建 mixed 训练/验证池数据集...")
                    train_loader, val_loader, internal_test_loader, shapes, splits_info = build_mixed_dataloaders(
                        station_csv=train_pool_file,
                        batch_size=self.config["batch_size"],
                        station_ratio=station_ratio,
                        val_ratio=self.config.get("val_ratio", 0.2),
                        test_ratio=0.1,
                        num_workers=self.config.get("num_workers", 10),
                        prefetch_factor=self.config.get("prefetch_factor", 4),
                        persistent_workers=self.config.get("persistent_workers", True),
                        seed=self.config["seed"],
                        **dataset_params,
                    )

                    # 固定测试集：只用 split='test' 的站点实测样本
                    print("\n   🔧 构建固定独立测试集 DataLoader...")
                    dataset_test = StationSWEDataset(
                        station_csv=test_file,
                        fine_tune_mode=True,
                        **dataset_params,
                    )

                    test_loader = DataLoader(
                        dataset_test,
                        batch_size=self.config.get("batch_size", 32),
                        shuffle=False,
                        num_workers=self.config.get("num_workers", 10),
                        pin_memory=True,
                    )

                    # 获取 MixedFineTuneDataset
                    if hasattr(train_loader.dataset, "dataset"):
                        mixed_dataset = train_loader.dataset.dataset
                    else:
                        mixed_dataset = train_loader.dataset

                    # 剥壳找到 mixed_dataset
                    depth = 0
                    while hasattr(mixed_dataset, "dataset") and not hasattr(mixed_dataset, "station_dataset"):
                        mixed_dataset = mixed_dataset.dataset
                        depth += 1

                    if not hasattr(mixed_dataset, "station_dataset"):
                        print("   ✗ mixed_dataset 中找不到 station_dataset")
                        return False

                    self.mixed_dataset = mixed_dataset
                    self.station_dataset = mixed_dataset.station_dataset
                    self.pretrain_dataset = mixed_dataset.pretrain_dataset
                    self.pretrain_indices = (
                        mixed_dataset.selected_pretrain
                        if hasattr(mixed_dataset, "selected_pretrain")
                        else []
                    )

                    # 关键：station_cv 的候选池应该是整个 train_pool 的 station_dataset
                    self.cv_pool_indices_override = list(range(len(self.station_dataset)))

                    # 关键：预训练辅助样本固定加入每折训练集
                    self.pretrain_aux_indices_override = [
                        len(self.station_dataset) + i
                        for i in range(len(self.pretrain_indices))
                    ]

                    print("\n   ✅ mixed station_cv 加载修正完成:")
                    print(f"      station_dataset 样本数: {len(self.station_dataset)}")
                    print(f"      pretrain selected: {len(self.pretrain_indices)}")
                    print(f"      CV pool override: {len(self.cv_pool_indices_override)}")
                    print(f"      pretrain aux override: {len(self.pretrain_aux_indices_override)}")
                    print(f"      fixed test samples: {len(dataset_test)}")

                    # self loader
                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    self.test_loader = test_loader

                    splits_info["has_split_column"] = True
                    splits_info["cv_mode"] = "station_cv"
                    splits_info["fixed_test_samples"] = len(df_test)
                    splits_info["fixed_test_stations"] = int(df_test["station_id"].nunique())
                    splits_info["train_pool_samples"] = len(df_train_pool)
                    splits_info["train_pool_stations"] = int(df_train_pool["station_id"].nunique())
                    splits_info["mixed_mode_fixed_test"] = True

                    self.splits_info = splits_info

                    # 保存简化 split 记录
                    split_records = []

                    for _, row in df_train_pool.iterrows():
                        split_records.append({
                            "split": "cv_pool",
                            "station_id": row.get("station_id", "unknown"),
                            "date": row.get("date", "unknown"),
                            "swe": row.get("swe", np.nan),
                            "longitude": row.get("longitude", np.nan),
                            "latitude": row.get("latitude", np.nan),
                        })

                    for _, row in df_test.iterrows():
                        split_records.append({
                            "split": "fixed_test",
                            "station_id": row.get("station_id", "unknown"),
                            "date": row.get("date", "unknown"),
                            "swe": row.get("swe", np.nan),
                            "longitude": row.get("longitude", np.nan),
                            "latitude": row.get("latitude", np.nan),
                        })

                    # 预训练样本记录
                    for idx in self.pretrain_indices:
                        split_records.append({
                            "split": "pretrain_aux",
                            "station_id": "PRETRAIN",
                            "date": "unknown",
                            "swe": np.nan,
                            "longitude": np.nan,
                            "latitude": np.nan,
                            "pretrain_index": idx,
                        })

                    save_basic_split_info(split_records, filename_prefix="mixed_split_info")

                # ============ mixed 但没有 split/station_cv ============
                else:
                    print(f"\n  正在构建普通 mixed 数据加载器 (站点比例={station_ratio * 100:.0f}%)...")
                    train_loader, val_loader, test_loader, shapes, splits_info = build_mixed_dataloaders(
                        station_csv=main_data_source,
                        batch_size=self.config["batch_size"],
                        station_ratio=station_ratio,
                        val_ratio=self.config.get("val_ratio", 0.2),
                        test_ratio=0.1,
                        num_workers=self.config.get("num_workers", 10),
                        prefetch_factor=self.config.get("prefetch_factor", 4),
                        persistent_workers=self.config.get("persistent_workers", True),
                        seed=self.config["seed"],
                        **dataset_params,
                    )

                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    self.test_loader = test_loader
                    self.splits_info = splits_info

                    if hasattr(train_loader.dataset, "dataset"):
                        mixed_dataset = train_loader.dataset.dataset
                    else:
                        mixed_dataset = train_loader.dataset

                    while hasattr(mixed_dataset, "dataset") and not hasattr(mixed_dataset, "station_dataset"):
                        mixed_dataset = mixed_dataset.dataset

                    if hasattr(mixed_dataset, "station_dataset"):
                        self.mixed_dataset = mixed_dataset
                        self.station_dataset = mixed_dataset.station_dataset
                        self.pretrain_dataset = mixed_dataset.pretrain_dataset
                        self.pretrain_indices = (
                            mixed_dataset.selected_pretrain
                            if hasattr(mixed_dataset, "selected_pretrain")
                            else []
                        )

            # ============================================================
            # 2. 纯站点微调模式
            # ============================================================
            elif use_fine_tune_mode:
                if not STATION_MODULE_AVAILABLE:
                    print("✗ 站点数据模块不可用，无法进行微调")
                    return False

                station_data_path = self.config.get("station_data_path")
                if not station_data_path:
                    print("✗ 未指定站点数据路径，请设置 station_data_path 参数")
                    return False

                main_data_source = resolve_station_data_source(station_data_path)
                if main_data_source is None:
                    return False

                print(f"\n  最终数据源: {main_data_source}")

                dataset_params = make_station_dataset_params()

                from data_station_online_swe import (
                    build_station_dataloaders_swe,
                    StationSWEDataset,
                )

                df_check = pd.read_csv(main_data_source, nrows=5)
                has_split_col = "split" in df_check.columns
                cv_mode = self.config.get("cv_mode", "standard")

                # ============ split列 + station_cv ============
                if has_split_col and cv_mode == "station_cv":
                    include_fixed_internal_in_cv = bool(
                        self.config.get(
                            "include_fixed_internal_in_cv",
                            False,
                        )
                    )
                    print("\n   ✅ 检测到 split 列，且 cv_mode='station_cv'")
                    if include_fixed_internal_in_cv:
                        print(
                            "      正式新协议：全部内部样本参与Nested CV"
                        )
                        print(
                            "      原split='test'的旧1000条已并回开发池；"
                            "不再构建内部固定测试Loader"
                        )
                    else:
                        print("      split='test' → 固定测试集，不参与 CV")
                        print("      split!='test' → 训练/验证池，参与 station_cv 十折")

                    df_full = pd.read_csv(main_data_source)
                    df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
                    df_full = df_full.dropna(subset=["date"])

                    print("\n   📊 split 列分布:")
                    for split_name, count in df_full["split"].value_counts().items():
                        print(f"      {split_name}: {count} 条记录")

                    if include_fixed_internal_in_cv:
                        df_test = df_full.iloc[0:0].copy()
                        df_train_pool = df_full.copy()
                    else:
                        df_test = df_full[df_full["split"] == "test"].copy()
                        df_train_pool = df_full[df_full["split"] != "test"].copy()

                    if (
                        not include_fixed_internal_in_cv
                        and len(df_test) == 0
                    ):
                        print("   ✗ split='test' 为空")
                        return False

                    if len(df_train_pool) == 0:
                        print("   ✗ split!='test' 训练/验证池为空")
                        return False

                    print("\n   📊 数据划分:")
                    print(f"      训练/验证池: {len(df_train_pool)} 条, {df_train_pool['station_id'].nunique()} 站点")
                    if include_fixed_internal_in_cv:
                        print("      内部固定测试集: 0 条（旧1000条已并回）")
                    else:
                        print(f"      固定测试集: {len(df_test)} 条, {df_test['station_id'].nunique()} 站点")

                    # PROGRESSIVE_STABLE_RUNTIME_MANIFEST_V1
                    # 固定清单不能放在单次实验temp_data中。
                    manifest_root = Path(main_data_source).resolve().parent
                    fixed_train_pool = manifest_root / "internal_cv_pool.csv"
                    fixed_test_file = manifest_root / "internal_test_approximately_1000.csv"

                    if include_fixed_internal_in_cv:
                        stable_dir = self.save_dir / "runtime_inputs"
                        stable_dir.mkdir(parents=True, exist_ok=True)
                        train_pool_file = (
                            stable_dir
                            / "internal_nested_cv_all_7936.csv"
                        )
                        runtime_train_pool = df_train_pool.copy()
                        runtime_train_pool["split"] = "cv"
                        runtime_train_pool.to_csv(
                            train_pool_file,
                            index=False,
                            encoding="utf-8-sig",
                        )
                        test_file = None
                    elif fixed_train_pool.exists() and fixed_test_file.exists():
                        fixed_train_rows = len(pd.read_csv(fixed_train_pool))
                        fixed_test_rows = len(pd.read_csv(fixed_test_file))
                        if fixed_train_rows != len(df_train_pool):
                            raise RuntimeError(
                                "固定internal_cv_pool.csv行数不一致: "
                                f"file={fixed_train_rows}, expected={len(df_train_pool)}"
                            )
                        if fixed_test_rows != len(df_test):
                            raise RuntimeError(
                                "固定internal_test_approximately_1000.csv行数不一致: "
                                f"file={fixed_test_rows}, expected={len(df_test)}"
                            )
                        train_pool_file = fixed_train_pool
                        test_file = fixed_test_file
                    else:
                        stable_dir = manifest_root / "_station_cv_runtime"
                        stable_dir.mkdir(parents=True, exist_ok=True)
                        train_pool_file = stable_dir / "train_pool.csv"
                        test_file = stable_dir / "test_split.csv"
                        df_train_pool.to_csv(train_pool_file, index=False)
                        df_test.to_csv(test_file, index=False)

                    print(f"      固定训练池: {train_pool_file}")
                    if test_file is not None:
                        print(f"      固定测试集: {test_file}")

                    print("\n   🔧 构建训练/验证池 DataLoader...")
                    train_loader, val_loader, internal_test_loader, shapes, splits_info = build_station_dataloaders_swe(
                        station_csv=train_pool_file,
                        batch_size=self.config["batch_size"],
                        val_ratio=self.config.get("val_ratio", 0.2),
                        test_ratio=0.1,
                        num_workers=self.config.get("num_workers", 10),
                        prefetch_factor=self.config.get("prefetch_factor", 4),
                        persistent_workers=self.config.get("persistent_workers", True),
                        seed=self.config["seed"],
                        fine_tune_mode=True,
                        **dataset_params,
                    )

                    if include_fixed_internal_in_cv:
                        print(
                            "\n   🔧 跳过内部固定测试DataLoader："
                            "全部7936条均参加Nested轮转"
                        )
                        test_loader = None
                    else:
                        print("\n   🔧 构建固定独立测试集 DataLoader...")
                        dataset_test = StationSWEDataset(
                            station_csv=test_file,
                            fine_tune_mode=True,
                            **dataset_params,
                        )

                        test_loader = DataLoader(
                            dataset_test,
                            batch_size=self.config.get("batch_size", 32),
                            shuffle=False,
                            num_workers=self.config.get("num_workers", 10),
                            pin_memory=True,
                        )

                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    self.test_loader = test_loader

                    # 找到底层 station_dataset
                    if hasattr(train_loader.dataset, "dataset"):
                        station_ds = train_loader.dataset.dataset
                    else:
                        station_ds = train_loader.dataset

                    self.station_dataset = station_ds
                    self.cv_pool_indices_override = list(range(len(station_ds)))
                    self.pretrain_aux_indices_override = []

                    splits_info["has_split_column"] = True
                    splits_info["cv_mode"] = "station_cv"
                    splits_info["fixed_test_samples"] = len(df_test)
                    splits_info["fixed_test_stations"] = int(df_test["station_id"].nunique())
                    splits_info["fixed_internal_1000_merged_into_cv"] = (
                        include_fixed_internal_in_cv
                    )
                    splits_info["train_pool_samples"] = len(df_train_pool)
                    splits_info["train_pool_stations"] = int(df_train_pool["station_id"].nunique())
                    self.splits_info = splits_info

                    print("\n✅ 按 split 列 + station_cv 模式加载完成")
                    print(f"   CV池: {len(df_train_pool)} 样本, {df_train_pool['station_id'].nunique()} 站点")
                    if include_fixed_internal_in_cv:
                        print("   内部固定测试集: 无（旧1000条已并回CV）")
                    else:
                        print(f"   固定测试集: {len(df_test)} 样本, {df_test['station_id'].nunique()} 站点")

                # ============ split列但不是 station_cv ============
                elif has_split_col:
                    print(f"\n   ✅ 检测到 split 列，按列划分 (cv_mode={cv_mode})")

                    df_full = pd.read_csv(main_data_source)
                    df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
                    df_full = df_full.dropna(subset=["date"])

                    df_train_val = df_full[df_full["split"] != "test"].copy()
                    df_test = df_full[df_full["split"] == "test"].copy()

                    if len(df_train_val) == 0 or len(df_test) == 0:
                        print("   ✗ train_val 或 test 为空")
                        return False

                    val_ratio = self.config.get("val_ratio", 0.2)

                    train_idx, val_idx = train_test_split(
                        df_train_val.index,
                        test_size=val_ratio,
                        random_state=self.config.get("seed", 42),
                    )

                    df_train = df_train_val.loc[train_idx].copy()
                    df_val = df_train_val.loc[val_idx].copy()

                    temp_dir = self.save_dir / "temp_data"
                    temp_dir.mkdir(parents=True, exist_ok=True)

                    train_file = temp_dir / "train_split.csv"
                    val_file = temp_dir / "val_split.csv"
                    test_file = temp_dir / "test_split.csv"

                    df_train.to_csv(train_file, index=False)
                    df_val.to_csv(val_file, index=False)
                    df_test.to_csv(test_file, index=False)

                    dataset_train = StationSWEDataset(
                        station_csv=train_file,
                        fine_tune_mode=True,
                        **dataset_params,
                    )
                    dataset_val = StationSWEDataset(
                        station_csv=val_file,
                        fine_tune_mode=True,
                        **dataset_params,
                    )
                    dataset_test = StationSWEDataset(
                        station_csv=test_file,
                        fine_tune_mode=True,
                        **dataset_params,
                    )

                    train_targets_mm = [
                        float(meta["swe"])
                        for meta in dataset_train.meta_index
                    ]
                    self.train_loader = self._make_swe_stratified_train_loader(
                        dataset_train,
                        train_targets_mm,
                        context="custom split",
                    )
                    self.val_loader = DataLoader(
                        dataset_val,
                        batch_size=self.config["batch_size"],
                        shuffle=False,
                        num_workers=self.config.get("num_workers", 10),
                        pin_memory=True,
                    )
                    self.test_loader = DataLoader(
                        dataset_test,
                        batch_size=self.config.get("batch_size", 32),
                        shuffle=False,
                        num_workers=self.config.get("num_workers", 10),
                        pin_memory=True,
                    )

                    shapes = (dataset_train.C_conv, dataset_train.C_point)

                    self.splits_info = {
                        "split_method": "custom_split_column",
                        "train_samples": len(df_train),
                        "val_samples": len(df_val),
                        "test_samples": len(df_test),
                        "train_stations": int(df_train["station_id"].nunique()),
                        "val_stations": int(df_val["station_id"].nunique()),
                        "test_stations": int(df_test["station_id"].nunique()),
                        "has_split_column": True,
                    }

                    print("\n✅ 按 split 列划分完成")
                    print(f"   训练集: {len(df_train)} 样本, {df_train['station_id'].nunique()} 站点")
                    print(f"   验证集: {len(df_val)} 样本, {df_val['station_id'].nunique()} 站点")
                    print(f"   测试集: {len(df_test)} 样本, {df_test['station_id'].nunique()} 站点")

                # ============ 无 split 列，使用原有 build_station_dataloaders_swe ============
                else:
                    print("\n   ℹ️ 未检测到 split 列，使用原有站点划分逻辑")

                    train_loader, val_loader, test_loader, shapes, splits_info = build_station_dataloaders_swe(
                        station_csv=main_data_source,
                        batch_size=self.config["batch_size"],
                        val_ratio=self.config.get("val_ratio", 0.2),
                        test_ratio=0.1,
                        num_workers=self.config.get("num_workers", 10),
                        prefetch_factor=self.config.get("prefetch_factor", 4),
                        persistent_workers=self.config.get("persistent_workers", True),
                        seed=self.config["seed"],
                        fine_tune_mode=True,
                        **dataset_params,
                    )

                    self.train_loader = train_loader
                    self.val_loader = val_loader
                    self.test_loader = test_loader
                    self.splits_info = splits_info

            # ============================================================
            # 3. 普通预训练模式
            # ============================================================
            else:
                split_method = self.config.get("split_method", "random")
                dataset_params = make_pretrain_dataset_params()

                if split_method == "temporal":
                    train_year = self.config.get("train_year", 2015)
                    val_year = self.config.get("val_year", 2016)
                    dataset_params["year_target"] = val_year

                    train_loader, val_loader, shapes = build_temporal_split_dataloaders(
                        train_years=[train_year],
                        val_years=[val_year],
                        batch_size=self.config["batch_size"],
                        num_workers=self.config.get("num_workers", 10),
                        prefetch_factor=self.config.get("prefetch_factor", 4),
                        persistent_workers=self.config.get("persistent_workers", True),
                        seed=self.config["seed"],
                        **dataset_params,
                    )

                elif split_method == "spatial":
                    spatial_split_ratio = self.config.get("spatial_split_ratio", 0.2)
                    split_by = self.config.get("split_by", "blocks")

                    train_loader, val_loader, shapes = build_spatial_split_dataloaders(
                        spatial_split_ratio=spatial_split_ratio,
                        split_by=split_by,
                        batch_size=self.config["batch_size"],
                        num_workers=self.config.get("num_workers", 10),
                        prefetch_factor=self.config.get("prefetch_factor", 4),
                        persistent_workers=self.config.get("persistent_workers", True),
                        seed=self.config["seed"],
                        **dataset_params,
                    )

                else:
                    train_loader, val_loader, shapes = build_dataloaders(
                        batch_size=self.config["batch_size"],
                        val_ratio=self.config["val_ratio"],
                        num_workers=self.config.get("num_workers", 10),
                        prefetch_factor=self.config.get("prefetch_factor", 4),
                        persistent_workers=self.config.get("persistent_workers", True),
                        seed=self.config["seed"],
                        **dataset_params,
                    )

                self.train_loader = train_loader
                self.val_loader = val_loader
                self.test_loader = None

            # ============================================================
            # 通用收尾：维度、SWE范围、统计、测试 batch
            # ============================================================

            # ============ 获取维度 ============
            if "shapes" in locals():
                C_conv, C_point = shapes
            elif hasattr(self, "train_loader") and hasattr(self.train_loader.dataset, "C_conv"):
                C_conv = self.train_loader.dataset.C_conv
                C_point = self.train_loader.dataset.C_point
            else:
                # 尝试剥壳
                try:
                    tmp_ds = self.train_loader.dataset
                    while hasattr(tmp_ds, "dataset"):
                        tmp_ds = tmp_ds.dataset

                    if hasattr(tmp_ds, "C_conv") and hasattr(tmp_ds, "C_point"):
                        C_conv = tmp_ds.C_conv
                        C_point = tmp_ds.C_point
                    elif hasattr(tmp_ds, "station_dataset"):
                        C_conv = tmp_ds.station_dataset.C_conv
                        C_point = tmp_ds.station_dataset.C_point
                    else:
                        C_conv = self.config.get("C_conv", 21)
                        C_point = self.config.get("C_point", 18)
                except Exception:
                    C_conv = self.config.get("C_conv", 21)
                    C_point = self.config.get("C_point", 18)

            self.config["C_conv"] = C_conv
            self.config["C_point"] = C_point

            print("✓ 数据加载成功!")
            print("\n数据维度:")
            print(f"  卷积特征: C_conv={C_conv}")
            print(f"  点特征: C_point={C_point}")

            # ============ 获取 SWE 范围 ============
            try:
                actual_dataset = self.train_loader.dataset

                while hasattr(actual_dataset, "dataset"):
                    actual_dataset = actual_dataset.dataset

                if hasattr(actual_dataset, "station_dataset"):
                    actual_dataset_for_swe = actual_dataset.station_dataset
                else:
                    actual_dataset_for_swe = actual_dataset

                self.swe_min = actual_dataset_for_swe.swe_min
                self.swe_max = actual_dataset_for_swe.swe_max
                print(f"✓ 成功捕获数据集真实SWE范围: [{self.swe_min:.2f}, {self.swe_max:.2f}]")

            except Exception as e:
                print(f"⚠ 捕获SWE范围失败(非关键错误): {e}")

            # ============ 打印统计 ============
            if use_mixed_mode:
                print("\n混合数据统计:")
                print(f"  训练集 loader dataset: {len(self.train_loader.dataset)} 个样本")
                print(f"  验证集 loader dataset: {len(self.val_loader.dataset)} 个样本")
                print(f"  测试集 loader dataset: {len(self.test_loader.dataset) if self.test_loader is not None else 0} 个样本")

                if hasattr(self, "station_dataset"):
                    print(f"  station_dataset: {len(self.station_dataset)} 个样本")

                if hasattr(self, "pretrain_indices"):
                    print(f"  selected pretrain: {len(self.pretrain_indices)} 个样本")

            elif use_fine_tune_mode:
                print("\n站点数据统计:")
                print(f"  训练集: {len(self.train_loader.dataset)} 个样本")
                print(f"  验证集: {len(self.val_loader.dataset)} 个样本")
                print(f"  测试集: {len(self.test_loader.dataset) if self.test_loader is not None else 0} 个样本")

                if hasattr(self, "splits_info") and self.splits_info:
                    print(f"  训练站点数: {self.splits_info.get('train_stations', self.splits_info.get('train_pool_stations', 'N/A'))}")
                    print(f"  验证站点数: {self.splits_info.get('val_stations', 'N/A')}")
                    actual_test_station_ids = (
                        self._get_actual_loader_station_ids(
                            self.test_loader
                        )
                        if self.test_loader is not None
                        else []
                    )

                    print(
                        "  测试站点数(实际test_loader): "
                        f"{len(actual_test_station_ids)}"
                    )
                    print(
                        "  测试站点ID(实际test_loader): "
                        f"{actual_test_station_ids}"
                    )

            else:
                print("\n预训练数据统计:")
                print(f"  训练集: {len(self.train_loader.dataset)} 个样本")
                print(f"  验证集: {len(self.val_loader.dataset)} 个样本")

            print(f"  批次大小: {self.config['batch_size']}")
            print(f"  数据加载线程: {self.config.get('num_workers', 10)}")
            print(f"  预加载因子: {self.config.get('prefetch_factor', 4)}")

            # ============ 测试一个 batch ============
            if self.train_loader is not None:
                try:
                    print("\n测试数据加载...")
                    batch_data = next(iter(self.train_loader))

                    print(f"  batch 返回元素数: {len(batch_data)}")

                    conv = batch_data[0]
                    point = batch_data[1]
                    target = batch_data[2]

                    print(f"  卷积特征: {conv.shape}")
                    print(f"  点特征: {point.shape}")
                    print(f"  目标值: {target.shape}")

                    if len(batch_data) == 6:
                        sample_idx = batch_data[5]

                        print(
                            "  sample_idx 检查"
                            "（纯站点6元素batch）:"
                        )
                        print(f"    前10个: {sample_idx[:10]}")
                        print(
                            f"    范围: "
                            f"[{sample_idx.min().item()}, "
                            f"{sample_idx.max().item()}]"
                        )
                        print(
                            "    当前batch不包含source_flag，"
                            "不统计station/pretrain数量"
                        )

                    elif len(batch_data) >= 7:
                        detected_source_flag = None
                        detected_sample_idx = None

                        for extra in batch_data[5:]:
                            if not torch.is_tensor(extra):
                                continue

                            flat = extra.detach().reshape(-1)

                            if flat.numel() == 0:
                                continue

                            unique_values = set(
                                flat.cpu().numpy().tolist()
                            )

                            if unique_values.issubset(
                                {0, 1, 0.0, 1.0}
                            ):
                                if detected_source_flag is None:
                                    detected_source_flag = extra
                            elif detected_sample_idx is None:
                                detected_sample_idx = extra

                        if detected_sample_idx is not None:
                            print("  sample_idx 检查:")
                            print(
                                f"    前10个: "
                                f"{detected_sample_idx[:10]}"
                            )

                        if detected_source_flag is not None:
                            source_flag = detected_source_flag
                            print("  source_flag 检查:")
                            print(
                                f"    前10个: "
                                f"{source_flag[:10]}"
                            )
                            print(
                                "    station样本数(source=0): "
                                f"{(source_flag == 0).sum().item()}"
                            )
                            print(
                                "    pretrain样本数(source=1): "
                                f"{(source_flag == 1).sum().item()}"
                            )
                        else:
                            print(
                                "  额外字段中未检测到"
                                "0/1型source_flag"
                            )

                    print("\n  数据范围检查:")
                    print(f"    卷积特征: [{conv.min():.3f}, {conv.max():.3f}]")
                    print(f"    点特征: [{point.min():.3f}, {point.max():.3f}]")
                    print(f"    目标值: [{target.min():.3f}, {target.max():.3f}]")

                    if torch.isnan(conv).any():
                        print(f"    ⚠ 卷积特征包含NaN: {torch.isnan(conv).sum().item()}个")
                    if torch.isnan(point).any():
                        print(f"    ⚠ 点特征包含NaN: {torch.isnan(point).sum().item()}个")
                    if torch.isnan(target).any():
                        print(f"    ⚠ 目标值包含NaN: {torch.isnan(target).sum().item()}个")

                except Exception as e:
                    print(f"  数据测试失败: {e}")
                    traceback.print_exc()

            return True

        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            traceback.print_exc()
            return False
        
    # EVAL_DIAGNOSTIC_LOG_FIX_V2
    def _get_actual_loader_station_ids(self, loader):
        """从实际loader的meta_index提取站点ID。"""
        if loader is None:
            return []

        ds = loader.dataset

        if hasattr(ds, "dataset") and hasattr(ds, "indices"):
            base_ds = ds.dataset
            indices = list(ds.indices)
        else:
            base_ds = ds
            indices = list(range(len(ds)))

        if hasattr(base_ds, "station_dataset"):
            base_ds = base_ds.station_dataset

        meta_index = getattr(base_ds, "meta_index", None)
        station_ids = set()

        if meta_index is not None:
            for idx in indices:
                idx = int(idx)

                if idx < 0 or idx >= len(meta_index):
                    continue

                meta = meta_index[idx]

                if not isinstance(meta, dict):
                    continue

                station_value = (
                    meta.get("station_id")
                    or meta.get("site_id")
                    or meta.get("station")
                )

                if station_value is None:
                    continue

                for station_id in str(station_value).split(","):
                    station_id = station_id.strip()
                    if station_id:
                        station_ids.add(station_id)

        return sorted(station_ids)


    def _load_model_for_evaluation(self, model_path):
        """加载模型用于评估"""
        print(f"  加载模型: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # 找到模型权重
            model_state_dict = None
            for key in ["model_state_dict", "state_dict", "model"]:
                if key in checkpoint:
                    model_state_dict = checkpoint[key]
                    print(f"  找到权重键: {key}")
                    break

            if model_state_dict is None:
                print("  警告: 未找到模型权重，使用随机初始化")
                return False

            # 🔥 先创建模型（不加载预训练权重，避免 build_model 的冻结逻辑）
            from models_swe import create_model
            self.model = create_model(
                model_type=self.config["model_type"],
                C_spatial=self.config.get("C_conv", 21),
                C_point=self.config.get("C_point", 18),
                d_model=self.config.get("d_model", 256),
                use_wide_branch=False,
            )

            # 加载权重
            self.model.load_state_dict(model_state_dict, strict=False)
            print(f"  ✓ 模型权重加载成功")

            # 🔥 评估模式：强制冻结所有参数
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            self.model.to(self.device)

            return True

        except Exception as e:
            print(f"  加载模型失败: {e}")
            return False
    
    def _test_data_loading(self, fine_tune_mode=False):
        """测试数据加载"""
        print(f"\n测试数据加载...")
        try:
            # 获取一个批次
            for i, (conv, point, target) in enumerate(self.train_loader):
                print(f"\n批次 {i+1}:")
                print(f"  卷积特征: {conv.shape}")
                print(f"  点特征: {point.shape}")
                print(f"  目标值: {target.shape}")

                # 检查每个样本的通道数
                for j in range(conv.shape[0]):
                    if conv[j].shape[0] != 6:
                        print(f"  ⚠ 样本 {j}: 只有 {conv[j].shape[0]} 个通道，不是6个!")
                        print(f"    可能是第 {j} 个样本的特征缺失")

                # 只测试第一个批次
                if i == 0:
                    break

            return True

        except Exception as e:
            print(f"✗ 数据测试失败: {e}")
            return False

    def _verify_freezing_detailed(self):
        """详细验证冻结状态"""
        total_params = 0
        trainable_params = 0
        module_stats = {}

        print("  各层冻结状态:")
        for name, param in self.model.named_parameters():
            total_params += param.numel()

            # 获取模块名
            parts = name.split(".")
            if len(parts) >= 2:
                module_name = f"{parts[0]}.{parts[1]}"
            else:
                module_name = parts[0]

            if module_name not in module_stats:
                module_stats[module_name] = {"total": 0, "trainable": 0}

            module_stats[module_name]["total"] += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()
                module_stats[module_name]["trainable"] += param.numel()
                status = "✓ 可训练"
            else:
                status = "✗ 冻结"

            # 只显示每个模块的第一层
            if "." in name and name.split(".")[-1] in ["weight", "bias"]:
                if param.numel() < 100:  # 小参数层详细显示
                    print(f"    {name:50s} {status:12s} shape={param.shape}")

        print(f"\n  总体统计:")
        print(f"    总参数: {total_params:,}")
        print(f"    可训练参数: {trainable_params:,}")
        print(f"    冻结参数: {total_params - trainable_params:,}")
        print(f"    可训练比例: {trainable_params/total_params*100:.1f}%")

        print(f"\n  模块统计:")
        for module, stats in module_stats.items():
            trainable_ratio = (
                stats["trainable"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            status = (
                "全部冻结"
                if trainable_ratio == 0
                else (
                    "全部可训练"
                    if trainable_ratio == 100
                    else f"{trainable_ratio:.1f}%可训练"
                )
            )
            print(f"    {module:20s}: {stats['total']:10,} 参数 ({status})")

    def _verify_optimizer(self):
        """验证优化器配置"""
        print(f"  优化器: {type(self.optimizer).__name__}")
        print(f"  参数组数: {len(self.optimizer.param_groups)}")

        for i, group in enumerate(self.optimizer.param_groups):
            print(f"\n  参数组 {i+1}:")
            print(f"    学习率: {group['lr']:.2e}")
            print(f"    权重衰减: {group['weight_decay']:.2e}")

            # 计算该组的参数数量
            param_count = sum(p.numel() for p in group["params"])
            print(f"    参数数量: {param_count:,}")

            # 显示前几个参数
            if len(group["params"]) > 0:
                first_param = group["params"][0]
                print(
                    f"    示例参数: shape={first_param.shape}, requires_grad={first_param.requires_grad}"
                )

    def _verify_optimizer_with_gradients(self):
        """验证优化器配置并监控梯度"""
        print(f"  优化器: {type(self.optimizer).__name__}")
        print(f"  参数组数: {len(self.optimizer.param_groups)}")
        
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"\n  参数组 {i+1}:")
            print(f"    学习率: {group['lr']:.2e}")
            print(f"    权重衰减: {group['weight_decay']:.2e}")
            
            # 计算该组的参数数量
            param_count = sum(p.numel() for p in group["params"])
            print(f"    参数数量: {param_count:,}")
            
            # 检查参数是否需要梯度
            trainable_params = sum(p.numel() for p in group["params"] if p.requires_grad)
            frozen_params = param_count - trainable_params
            print(f"    可训练参数: {trainable_params:,}")
            print(f"    冻结参数: {frozen_params:,}")
            
            # 显示参数基本信息 - 简化版
            print(f"    参数示例 (前3个):")
            for j, param in enumerate(group["params"][:3]):
                # 直接显示基本信息
                print(f"      参数 {j}: shape={param.shape}, "
                      f"requires_grad={param.requires_grad}")
                
    def _test_forward_pass(self):
        """测试前向传播"""
        try:
            self.model.eval()

            # 取一个批次
            conv_batch, point_batch, target_batch = next(iter(self.train_loader))
            batch_size = len(target_batch)

            # 移动到设备
            conv_batch = conv_batch.to(self.device)
            point_batch = point_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            with torch.no_grad():
                # 前向传播
                outputs = self.model(conv_batch, point_batch)

                # 计算初始损失
                initial_loss = self.criterion(outputs, target_batch)

                # 计算相关系数
                outputs_np = outputs.cpu().numpy()
                targets_np = target_batch.cpu().numpy()

                if len(outputs_np) > 1:
                    correlation = np.corrcoef(outputs_np, targets_np)[0, 1]
                else:
                    correlation = 0

            print(f"  前向传播测试:")
            print(f"    批次大小: {batch_size}")
            print(f"    输出范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
            print(f"    输出均值: {outputs.mean():.4f} ± {outputs.std():.4f}")
            print(f"    目标范围: [{target_batch.min():.4f}, {target_batch.max():.4f}]")
            print(f"    目标均值: {target_batch.mean():.4f} ± {target_batch.std():.4f}")
            print(f"    初始损失: {initial_loss.item():.6f}")
            print(f"    初始相关性: {correlation:.4f}")

            # 检查输出方差
            output_var = outputs.var().item()
            target_var = target_batch.var().item()
            var_ratio = output_var / target_var if target_var > 0 else 0

            print(f"    输出方差: {output_var:.6f}")
            print(f"    目标方差: {target_var:.6f}")
            print(f"    方差比例: {var_ratio:.3f}")

            if var_ratio < 0.1:
                print(f"    ⚠ 警告: 输出方差过小，可能过平滑!")
            elif var_ratio > 10:
                print(f"    ⚠ 警告: 输出方差过大!")

        except Exception as e:
            print(f"  ⚠ 前向传播测试失败: {e}")

    def _log_data_statistics(self):
        """记录数据统计"""
        try:
            # 训练集统计
            train_targets = []
            for _, _, targets in self.train_loader:
                train_targets.extend(targets.numpy())

            train_targets = np.array(train_targets)

            print(f"  训练集统计:")
            print(f"    样本数: {len(train_targets):,}")
            print(
                f"    目标范围: [{train_targets.min():.4f}, {train_targets.max():.4f}]"
            )
            print(
                f"    目标均值: {train_targets.mean():.4f} ± {train_targets.std():.4f}"
            )

            # 验证集统计
            if hasattr(self, "val_loader") and self.val_loader is not None:
                val_targets = []
                for _, _, targets in self.val_loader:
                    val_targets.extend(targets.numpy())

                val_targets = np.array(val_targets)

                print(f"\n  验证集统计:")
                print(f"    样本数: {len(val_targets):,}")
                print(
                    f"    目标范围: [{val_targets.min():.4f}, {val_targets.max():.4f}]"
                )
                print(
                    f"    目标均值: {val_targets.mean():.4f} ± {val_targets.std():.4f}"
                )

        except Exception as e:
            print(f"  ⚠ 数据统计失败: {e}")

    def _fusion_ft_progressive_enabled(self):
        """仅对正式Fusion FT启用渐进式解冻。"""
        return bool(
            self.config.get("fusion_ft_progressive_unfreeze", True)
        ) and str(
            self.config.get("freeze_strategy", "")
        ).lower() == "fusion_ft"

    @staticmethod
    def _fusion_ft_parameter_group(name):
        """返回Fusion参数所属的渐进式分组。"""
        name_lower = str(name).lower()
        if "fusion_transformer" not in name_lower:
            return None
        if ".head." in name_lower:
            return ("head", None)

        layer_prefix = "fusion_transformer.encoder.layers."
        if layer_prefix in name_lower:
            suffix = name_lower.split(layer_prefix, 1)[1]
            layer_text = suffix.split(".", 1)[0]
            try:
                return ("transformer_layer", int(layer_text))
            except ValueError:
                pass

        # CLS token、位置编码、输入LayerNorm等共享Fusion参数。
        return ("transformer_shared", None)

    def _configure_fusion_ft_progressive_optimizer(self):
        """
        为Fusion FT创建包含全部Fusion参数的优化器。

        尚未解冻的参数也预先放入optimizer group，但requires_grad=False，
        因而不会产生梯度或更新。这样逐层解冻时无需重建Adam状态。
        """
        self._fusion_ft_progressive_optimizer = False
        if not self._fusion_ft_progressive_enabled():
            return False

        grouped = defaultdict(list)
        for name, param in self.model.named_parameters():
            group_info = self._fusion_ft_parameter_group(name)
            if group_info is not None:
                grouped[group_info].append(param)
            elif "spatial_encoder" in name.lower() or "point_encoder" in name.lower():
                param.requires_grad = False

        layer_indices = sorted(
            group_key[1]
            for group_key in grouped
            if group_key[0] == "transformer_layer"
        )
        if not grouped.get(("head", None)):
            raise RuntimeError("Fusion FT渐进式解冻未找到回归Head参数")
        if not layer_indices:
            raise RuntimeError("Fusion FT渐进式解冻未找到Transformer层")

        max_unfrozen_layers = int(
            self.config.get("fusion_ft_max_unfrozen_layers", 0)
        )
        if max_unfrozen_layers < 0:
            raise ValueError(
                "fusion_ft_max_unfrozen_layers不能小于0"
            )
        if max_unfrozen_layers > len(layer_indices):
            raise ValueError(
                "fusion_ft_max_unfrozen_layers超过实际Transformer层数: "
                f"requested={max_unfrozen_layers}, "
                f"available={len(layer_indices)}"
            )

        limited_scope = 0 < max_unfrozen_layers < len(layer_indices)
        if max_unfrozen_layers == 0:
            selected_layer_indices = list(layer_indices)
        else:
            selected_layer_indices = list(
                layer_indices[-max_unfrozen_layers:]
            )
        frozen_layer_indices = [
            layer_index
            for layer_index in layer_indices
            if layer_index not in selected_layer_indices
        ]

        # build_model的fusion_ft旧逻辑会先解冻全部Fusion参数。
        # 在构造受限优化器前必须重新冻结所有Fusion组，随后只激活
        # Head和明确选中的顶部Transformer层。
        for params in grouped.values():
            for param in params:
                param.requires_grad = False

        interval = max(
            1,
            int(self.config.get("fusion_ft_unfreeze_interval", 1)),
        )
        max_layer = max(selected_layer_indices)
        unlock_epoch_by_group = {
            ("head", None): 0,
        }
        for layer_index in selected_layer_indices:
            # epoch使用0-based：epoch 0只训练Head；epoch 1解冻最顶层。
            unlock_epoch_by_group[("transformer_layer", layer_index)] = (
                1 + (max_layer - layer_index) * interval
            )

        # 旧的全层模式保持原行为：CLS/位置编码/输入Norm与最底层
        # Transformer同时解冻。受限模式则冻结这些共享参数，保证
        # 实验变量严格等于“Head + 顶部N个block”。
        full_unfreeze_epoch = max(
            unlock_epoch_by_group.values()
        )
        if not limited_scope:
            unlock_epoch_by_group[("transformer_shared", None)] = (
                full_unfreeze_epoch
            )

        head_lr = float(
            self.config.get("fusion_ft_head_lr", 5e-5)
        )
        transformer_lr = float(
            self.config.get("fusion_ft_transformer_lr", 3e-5)
        )
        if head_lr <= 0 or transformer_lr <= 0:
            raise ValueError(
                "Fusion FT学习率必须为正数: "
                f"head={head_lr}, transformer={transformer_lr}"
            )

        param_groups = []
        ordered_keys = [("head", None)]
        ordered_keys.extend(
            ("transformer_layer", layer_index)
            for layer_index in sorted(
                selected_layer_indices,
                reverse=True,
            )
        )
        if (
            not limited_scope
            and grouped.get(("transformer_shared", None))
        ):
            ordered_keys.append(("transformer_shared", None))

        schedule_rows = []
        for group_key in ordered_keys:
            params = grouped.get(group_key, [])
            if not params:
                continue

            unlock_epoch = int(unlock_epoch_by_group[group_key])
            target_lr = (
                head_lr
                if group_key[0] == "head"
                else transformer_lr
            )
            for param in params:
                param.requires_grad = unlock_epoch == 0

            if group_key[0] == "transformer_layer":
                group_name = f"transformer_layer_{group_key[1]}"
            else:
                group_name = group_key[0]

            param_groups.append({
                "params": params,
                "lr": 0.0,
                "target_lr": target_lr,
                "unlock_epoch": unlock_epoch,
                "group_name": group_name,
                "activation_lr_initialized": False,
            })
            schedule_rows.append(
                (group_name, unlock_epoch, target_lr, sum(p.numel() for p in params))
            )

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=1e-4,
        )
        self._fusion_ft_full_unfreeze_epoch = int(full_unfreeze_epoch)
        self._fusion_ft_selected_layer_indices = tuple(
            selected_layer_indices
        )
        self._fusion_ft_frozen_layer_indices = tuple(
            frozen_layer_indices
        )
        self._fusion_ft_progressive_optimizer = True
        self._fusion_ft_last_logged_epoch = None

        total_params = sum(p.numel() for p in self.model.parameters())
        initial_trainable = sum(
            p.numel() for p in self.model.parameters()
            if p.requires_grad
        )
        print("  ✅ Fusion FT稳定化优化器:")
        if limited_scope:
            print(
                "     解冻范围: Head + 顶部"
                f"{len(selected_layer_indices)}个Transformer block "
                f"(layers={selected_layer_indices}); "
                f"冻结layers={frozen_layer_indices}及共享Fusion参数"
            )
        else:
            print(
                "     解冻范围: 旧版完整Fusion "
                f"(全部{len(selected_layer_indices)}个Transformer block"
                " + 共享Fusion参数)"
            )
        print(
            f"     初始仅Head可训练: {initial_trainable:,}/{total_params:,} "
            f"({100.0 * initial_trainable / max(total_params, 1):.2f}%)"
        )
        for group_name, unlock_epoch, target_lr, param_count in schedule_rows:
            print(
                f"     {group_name:<22s} "
                f"从epoch {unlock_epoch + 1}解冻, "
                f"peak_lr={target_lr:.2e}, params={param_count:,}"
            )
        print(
            f"     warmup={int(self.config.get('fusion_ft_warmup_epochs', 10))} epochs; "
            f"epoch {full_unfreeze_epoch + 1}起设定范围全部可训练"
        )
        return True

    def _apply_fusion_ft_progressive_stage(self, epoch):
        """在每个epoch开始前按既定顺序切换requires_grad。"""
        if not bool(
            getattr(self, "_fusion_ft_progressive_optimizer", False)
        ):
            return

        active_groups = []
        trainable_params = 0
        total_params = sum(p.numel() for p in self.model.parameters())
        for group in self.optimizer.param_groups:
            unlock_epoch = int(group.get("unlock_epoch", 0))
            active = int(epoch) >= unlock_epoch
            for param in group["params"]:
                param.requires_grad = active
                if active:
                    trainable_params += param.numel()
            if active:
                active_groups.append(str(group.get("group_name", "group")))

        if self._fusion_ft_last_logged_epoch != int(epoch):
            print(
                f"  🧊 Fusion FT渐进解冻 epoch {epoch + 1}: "
                f"{', '.join(active_groups)}"
            )
            print(
                f"     可训练参数={trainable_params:,}/{total_params:,} "
                f"({100.0 * trainable_params / max(total_params, 1):.2f}%)"
            )
            self._fusion_ft_last_logged_epoch = int(epoch)

    def _apply_fusion_ft_warmup_lr(self, epoch, batch_idx, steps_per_epoch):
        """按optimizer step线性升至各参数组的峰值LR。"""
        if not bool(
            getattr(self, "_fusion_ft_progressive_optimizer", False)
        ):
            return

        warmup_epochs = max(
            0,
            int(self.config.get("fusion_ft_warmup_epochs", 10)),
        )
        if warmup_epochs == 0:
            warmup_factor = 1.0
            warmup_finished = True
        else:
            total_warmup_steps = max(1, warmup_epochs * steps_per_epoch)
            current_step = epoch * steps_per_epoch + batch_idx + 1
            warmup_factor = min(
                1.0,
                current_step / total_warmup_steps,
            )
            warmup_finished = current_step >= total_warmup_steps

        for group in self.optimizer.param_groups:
            target_lr = float(group.get("target_lr", group["lr"]))
            if epoch < int(group.get("unlock_epoch", 0)):
                group["lr"] = 0.0
            elif not warmup_finished:
                group["lr"] = target_lr * warmup_factor
            elif not bool(
                group.get("activation_lr_initialized", False)
            ):
                # warmup结束或该层首次解冻时只设置一次峰值LR。
                # 后续不得覆盖Plateau衰减或collapse rollback的LR。
                group["lr"] = target_lr
                group["activation_lr_initialized"] = True

        if batch_idx == 0:
            active_lr_text = ", ".join(
                f"{group.get('group_name', 'group')}={float(group['lr']):.2e}"
                for group in self.optimizer.param_groups
                if epoch >= int(group.get("unlock_epoch", 0))
            )
            print(
                f"     当前epoch首个step学习率: {active_lr_text}"
            )

    def _fusion_ft_schedule_is_protected(self, epoch):
        """渐进解冻或warmup未完成时，不累计早停且不触发Plateau。"""
        if not bool(
            getattr(self, "_fusion_ft_progressive_optimizer", False)
        ):
            return False

        full_unfreeze_epoch = int(
            getattr(self, "_fusion_ft_full_unfreeze_epoch", 0)
        )
        warmup_epochs = max(
            0,
            int(self.config.get("fusion_ft_warmup_epochs", 10)),
        )
        protected_until_exclusive = max(
            full_unfreeze_epoch + 1,
            warmup_epochs,
        )
        return int(epoch) < protected_until_exclusive

    def _diagnose_fusion_ft_loss_gradients(
        self,
        base_loss,
        weighted_ccc_loss,
    ):
        """
        比较站点基础损失与加权CCC损失对当前可训练参数的真实梯度。

        使用torch.autograd.grad，不写入param.grad，不执行optimizer.step，
        因而只增加诊断开销，不改变后续正式反向传播和参数更新。
        """
        trainable_params = [
            param
            for param in self.model.parameters()
            if param.requires_grad
        ]

        result = {
            "base_grad_norm": float("nan"),
            "weighted_ccc_grad_norm": float("nan"),
            "ccc_to_base_grad_ratio": float("nan"),
            "gradient_cosine": float("nan"),
            "trainable_tensor_count": len(trainable_params),
            "status": "not_run",
        }

        if not trainable_params:
            result["status"] = "no_trainable_parameters"
            return result

        if (
            not torch.is_tensor(base_loss)
            or not base_loss.requires_grad
        ):
            result["status"] = "base_loss_has_no_gradient"
            return result

        if (
            not torch.is_tensor(weighted_ccc_loss)
            or not weighted_ccc_loss.requires_grad
        ):
            result.update(
                {
                    "weighted_ccc_grad_norm": 0.0,
                    "ccc_to_base_grad_ratio": 0.0,
                    "gradient_cosine": 0.0,
                    "status": "ccc_loss_has_no_gradient",
                }
            )
            return result

        try:
            base_grads = torch.autograd.grad(
                base_loss,
                trainable_params,
                retain_graph=True,
                allow_unused=True,
            )
            ccc_grads = torch.autograd.grad(
                weighted_ccc_loss,
                trainable_params,
                retain_graph=True,
                allow_unused=True,
            )

            device = base_loss.device
            base_sq = torch.zeros(
                (),
                device=device,
                dtype=torch.float32,
            )
            ccc_sq = torch.zeros(
                (),
                device=device,
                dtype=torch.float32,
            )
            dot = torch.zeros(
                (),
                device=device,
                dtype=torch.float32,
            )

            for base_grad, ccc_grad in zip(
                base_grads,
                ccc_grads,
            ):
                if base_grad is not None:
                    base_grad_work = base_grad.detach().to(
                        dtype=torch.float32
                    )
                    base_sq = (
                        base_sq
                        + base_grad_work.pow(2).sum()
                    )
                else:
                    base_grad_work = None

                if ccc_grad is not None:
                    ccc_grad_work = ccc_grad.detach().to(
                        dtype=torch.float32
                    )
                    ccc_sq = (
                        ccc_sq
                        + ccc_grad_work.pow(2).sum()
                    )
                else:
                    ccc_grad_work = None

                if (
                    base_grad_work is not None
                    and ccc_grad_work is not None
                ):
                    dot = dot + (
                        base_grad_work * ccc_grad_work
                    ).sum()

            base_norm = torch.sqrt(base_sq)
            ccc_norm = torch.sqrt(ccc_sq)
            epsilon = torch.as_tensor(
                1e-30,
                device=device,
                dtype=torch.float32,
            )
            ratio = ccc_norm / torch.clamp(
                base_norm,
                min=epsilon,
            )
            cosine = dot / torch.clamp(
                base_norm * ccc_norm,
                min=epsilon,
            )

            result.update(
                {
                    "base_grad_norm": float(
                        base_norm.detach().item()
                    ),
                    "weighted_ccc_grad_norm": float(
                        ccc_norm.detach().item()
                    ),
                    "ccc_to_base_grad_ratio": float(
                        ratio.detach().item()
                    ),
                    "gradient_cosine": float(
                        cosine.detach().item()
                    ),
                    "status": "ok",
                }
            )
        except RuntimeError as exc:
            result["status"] = (
                f"autograd_error:{type(exc).__name__}:{exc}"
            )

        return result



    # ============================================================
    # [ROUTER] build_model()
    # ============================================================
    # 根据配置构建不同模型/微调形式：
    #
    #   1. 普通 SWENet
    #   2. ResidualInjectionSWENet
    #   3. ResidualSWENet
    #   4. SpatiallyGatedSWENet
    #   5. LoRA 微调模型
    #
    # FIVE_FINETUNE_STRATEGIES_V1
    # [CONTRACT] freeze_strategy:
    #   fusion_ft  -> Fusion Layer（Transformer + 内含回归Head）
    #   point_ft   -> Point Encoder + Fusion Layer
    #   spatial_ft -> Spatial Encoder + Fusion Layer
    #   partial    -> 两个Encoder顶层 + Fusion Layer（展示名Top-Layer FT）
    #   none       -> 全部解冻（展示名Full FT）
    #
    # [COMPAT]
    #   旧 checkpoint 可能 C_point 不同。
    #   加载预训练权重时允许部分层跳过或扩展。
    #
    # [DANGER]
    #   修改冻结策略后必须检查 trainable 参数比例。
    #   否则可能以为在微调，实际全模型都在训。
    # ============================================================
    def build_model(self, load_pretrained=None, freeze_backbone=True, freeze_strategy='fusion_ft', use_residual=False, is_cv_fold=False):
        """
        构建模型，支持残差注入、LoRA微调、门控模型

        Args:
            load_pretrained: 预训练模型路径
            freeze_backbone: 是否冻结主干
            freeze_strategy: 冻结策略
            use_residual: 是否使用残差学习
            is_cv_fold: 是否为交叉验证折（True时减少打印，避免输出过多）
        """

        # ============ 🔥 添加调试打印 ============
        print(f"\n[DEBUG build_model] 收到参数:")
        print(f"  freeze_strategy = {freeze_strategy}")
        print(f"  freeze_backbone = {freeze_backbone}")
        print(f"  load_pretrained = {load_pretrained}")
        print(f"  config.get('freeze_strategy') = {self.config.get('freeze_strategy')}")

        # 🔥 确保 config 中的值被更新
        self.config['freeze_strategy'] = freeze_strategy
        print(f"  ✅ 已更新 config['freeze_strategy'] = {freeze_strategy}")
        # =====================================

        # ============ 交叉验证时减少打印 ============
        if not is_cv_fold:
            print("\n" + "=" * 70)
            print(f"构建模型 ({self.config['model_type']})...")
        else:
            print(f"\n🏗️ 构建模型...")

        # 检查是否使用LoRA
        use_lora = self.config.get("use_lora", False) and load_pretrained is not None
        lora_config = self.config.get("lora_config", {})

        # 检查是否为残差注入模式
        use_residual_injection = self.config.get("residual_injection", False)

        # 渐进式增量预训练：加载上一阶段权重，但仍属于预训练，
        # 不是站点微调。全参数训练，使用预训练学习率和优化器。
        is_incremental_pretrain = bool(load_pretrained) and (
            str(self.config.get("sampling_mode", "")).lower() == "incremental"
        )

        # 🔥 检查是否为 evaluate 模式
        is_evaluate_mode = self.config.get('mode') == 'evaluate'

        if not is_cv_fold:
            if use_lora:
                print(f"【LoRA微调模式】启用")
                print(f"LoRA配置: {lora_config}")
            elif use_residual_injection:
                print(f"【残差注入模式】启用 (Parallel Residual Adapter)")
            elif use_residual:
                print(f"【残差学习模式】启用")
            elif is_evaluate_mode:
                print(f"【评估模式】只推理，不训练")
            elif is_incremental_pretrain:
                print(f"【渐进式增量预训练】加载上一阶段权重，全参数继续预训练")
            else:
                print(f"【普通训练模式】")

        try:
            # ============ 干净版 C_point ============
            if self.config.get("C_point") is None:
                self.config["C_point"] = 18
            if not is_cv_fold:
                print(f"【干净版】C_point = {self.config['C_point']}")

            if self.config["C_conv"] is None:
                self.config["C_conv"] = 7

            if not is_cv_fold:
                print(f"最终模型参数: C_conv={self.config['C_conv']}, C_point={self.config['C_point']}")

            # ============ 残差注入模式（新架构） ============
            if use_residual_injection:
                try:
                    from models_swe import ResidualInjectionSWENet

                    if not is_cv_fold:
                        print("\n【残差注入模型】创建 ResidualInjectionSWENet...")

                    # 创建基础骨干网络（预训练模型）
                    base_model = create_model(
                        model_type=self.config["model_type"],
                        C_spatial=self.config["C_conv"],
                        C_point=self.config["C_point"],
                        d_model=self.config["d_model"],
                        use_wide_branch=False,
                    )

                    # 创建残差注入模型
                    self.model = ResidualInjectionSWENet(
                        pretrained_model=base_model,
                        C_point=self.config["C_point"],
                        d_model=self.config["d_model"]
                    )
                    if not is_cv_fold:
                        print("✓ 残差注入模型构建成功")

                    # 加载预训练权重
                    if load_pretrained and os.path.exists(load_pretrained):
                        if not is_cv_fold:
                            print(f"\n【残差模式特权加载】正在加载预训练权重...")
                            print(f"  预训练权重路径: {load_pretrained}")

                        self._load_pretrained_weights(self.model.backbone, load_pretrained)
                        # self._init_new_pointencoder_weights(self.model.backbone)

                        if not is_cv_fold:
                            print("  ✅ 残差模式权重加载完成")
                    else:
                        if not is_cv_fold:
                            print("  ⚠ 未提供预训练权重，backbone将使用随机初始化")

                except Exception as e:
                    print(f"✗ 残差注入模型构建失败: {e}")
                    traceback.print_exc()
                    return False

            # ============ 残差学习模式（原架构） ============
            elif use_residual:
                try:
                    from models_swe import ResidualSWENet

                    # 创建基础模型
                    base_model = create_model(
                        model_type=self.config["model_type"],
                        C_spatial=self.config["C_conv"],
                        C_point=self.config["C_point"],
                        d_model=self.config["d_model"],
                        use_wide_branch=False,
                    )

                    # 先加载预训练权重到基础模型
                    if load_pretrained and os.path.exists(load_pretrained):
                        self._load_pretrained_weights(base_model, load_pretrained)
                        # self._init_new_pointencoder_weights(base_model)

                    # 创建残差模型
                    self.model = ResidualSWENet(
                        pretrained_model=base_model,
                        C_point=self.config["C_point"],
                        d_model=self.config["d_model"]
                    )
                    if not is_cv_fold:
                        print("✓ 残差模型构建成功")

                except Exception as e:
                    print(f"✗ 残差模型构建失败: {e}")
                    return False

            # ============ 门控模型 ============
            elif self.config.get("use_gate", False):
                try:
                    from models_swe import SpatiallyGatedSWENet

                    if not is_cv_fold:
                        print("\n【门控融合模型】启用 SpatiallyGatedSWENet")

                    # 创建基础模型
                    base_model = create_model(
                        model_type=self.config["model_type"],
                        C_spatial=self.config["C_conv"],
                        C_point=self.config["C_point"],
                        d_model=self.config["d_model"],
                        use_wide_branch=False,
                    )

                    # 确保预训练模型已加载（作为专家A）
                    if not (load_pretrained and os.path.exists(load_pretrained)):
                        print("✗ 门控模型需要预训练模型作为专家A，请提供预训练权重")
                        return False

                    # 加载预训练权重到基础模型
                    self._load_pretrained_weights(base_model, load_pretrained)
                    # self._init_new_pointencoder_weights(base_model)

                    # 创建门控模型
                    self.model = SpatiallyGatedSWENet(
                        pretrained_model=base_model,
                        C_point=self.config["C_point"],
                        d_model=self.config["d_model"]
                    )
                    if not is_cv_fold:
                        print("✓ 门控模型构建成功")

                except Exception as e:
                    print(f"✗ 门控模型构建失败: {e}")
                    traceback.print_exc()
                    return False

            # ============ 普通模式 ============
            else:
                # 普通模式：直接使用基础模型
                self.model = create_model(
                    model_type=self.config["model_type"],
                    C_spatial=self.config["C_conv"],
                    C_point=self.config["C_point"],
                    d_model=self.config["d_model"],
                    use_wide_branch=False,
                )

                # 加载预训练权重（如果需要）
                if load_pretrained and os.path.exists(load_pretrained):
                    self._load_pretrained_weights(self.model, load_pretrained)
                    # self._init_new_pointencoder_weights(self.model)

            # 移动到设备
            self.model.to(self.device)

            # ============ 🔥 评估模式：强制冻结所有参数 ============
            if is_evaluate_mode:
                print("\n🔒 [EVALUATE MODE] 强制冻结所有参数（只推理，不训练）")
                for param in self.model.parameters():
                    param.requires_grad = False
                # 设置模型为 eval 模式
                self.model.eval()
                print("   ✅ 所有参数已冻结，模型只用于推理")
                # 跳过后续的冻结策略和优化器配置
                # 直接打印信息并返回
                if not is_cv_fold:
                    total_params = sum(p.numel() for p in self.model.parameters())
                    print(f"\n模型信息:")
                    print(f"  类型: {self.config['model_type']}")
                    print(f"  总参数: {total_params:,}")
                    print(f"  可训练参数: 0 (评估模式，全部冻结)")
                    print(f"\n✅ 模型加载成功! (评估模式)")
                return True

            # ============ 3. 执行参数冻结（非评估模式） ============
            if load_pretrained and freeze_backbone:
                if not is_cv_fold:
                    print(f"\n🛡️ 执行微调冻结策略: {freeze_strategy}")

                # 第一步：先冻结所有，再根据策略解冻
                for param in self.model.parameters():
                    param.requires_grad = False

                if freeze_strategy == 'fusion_ft':
                    # 输出Head属于Fusion Layer，不再作为独立微调策略。
                    trainable_keywords = ['transformer', 'head', 'regression', 'output', 'fc', 'correction']
                    for name, param in self.model.named_parameters():
                        if 'spatial_encoder' in name.lower() or 'point_encoder' in name.lower():
                            param.requires_grad = False
                        elif any(k in name.lower() for k in trainable_keywords):
                            param.requires_grad = True
                            if not is_cv_fold:
                                print(f"  🔥 已激活(解冻): {name}")

                elif freeze_strategy == 'point_ft':
                    if not is_cv_fold:
                        print("  🎯 Point-Branch FT")

                    for name, param in self.model.named_parameters():
                        name_lower = name.lower()

                        # Point Encoder
                        is_point = 'point_encoder' in name_lower

                        # Fusion Transformer
                        is_transformer = 'transformer' in name_lower

                        # Head
                        is_head = any(
                            k in name_lower
                            for k in [
                                'head',
                                'regression',
                                'output',
                                'fc',
                                'correction'
                            ]
                        )

                        if is_point or is_transformer or is_head:
                            param.requires_grad = True

                            if not is_cv_fold:
                                print(f"  🔥 [Point-Branch FT]: {name}")

                        else:
                            param.requires_grad = False

                elif freeze_strategy == 'spatial_ft':
                    if not is_cv_fold:
                        print("  🎯 Spatial-Branch FT")

                    for name, param in self.model.named_parameters():
                        name_lower = name.lower()

                        # Spatial Encoder
                        is_spatial = 'spatial_encoder' in name_lower

                        # Fusion Transformer
                        is_transformer = 'transformer' in name_lower

                        # Head
                        is_head = any(
                            k in name_lower
                            for k in [
                                'head',
                                'regression',
                                'output',
                                'fc',
                                'correction'
                            ]
                        )

                        if is_spatial or is_transformer or is_head:
                            param.requires_grad = True

                            if not is_cv_fold:
                                print(f"  🔥 [Spatial-Branch FT]: {name}")

                        else:
                            param.requires_grad = False

                elif freeze_strategy == 'partial':
                    if not is_cv_fold:
                        print("  🏗️ 模式：Top-Layer FT（编码器顶层 + Fusion Layer）")

                    high_level_keywords = ['transformer', 'head', 'regression', 'output', 'fc', 'correction']
                    encoder_last_keywords = [
                        'spatial_encoder.blocks.5',
                        'spatial_encoder.se_module',
                        'spatial_encoder.global_attention',
                        'spatial_encoder.final_proj',
                        'point_encoder.mlp.6'           
                    ]

                    for name, param in self.model.named_parameters():
                        is_high_level = any(k in name.lower() for k in high_level_keywords)
                        is_encoder_last = any(k in name.lower() for k in encoder_last_keywords)

                        if is_high_level or is_encoder_last:
                            param.requires_grad = True
                            if not is_cv_fold:
                                print(f"  🔥 [Top-Layer FT]: {name}")
                        else:
                            param.requires_grad = False

                elif freeze_strategy == 'none':
                    for param in self.model.parameters():
                        param.requires_grad = True
                    if not is_cv_fold:
                        print("  🔥 全部参数已解冻")

                else:
                    # 🔥 未知策略，打印警告并解冻所有
                    print(f"  ⚠️ 未知冻结策略: {freeze_strategy}，将解冻所有参数")
                    for param in self.model.parameters():
                        param.requires_grad = True

            # ============ 如果是残差模式，二次确认 ============
            if use_residual_injection:
                if not is_cv_fold:
                    print("\n  🔧 [残差模式特权检查]")

                if hasattr(self.model, 'backbone'):
                    for param in self.model.backbone.parameters():
                        param.requires_grad = False
                    self.model.backbone.eval()
                    if not is_cv_fold:
                        print("    ❄️ 主干(Backbone)已强制锁死")

                if hasattr(self.model, 'correction_mlp'):
                    for param in self.model.correction_mlp.parameters():
                        param.requires_grad = True
                    if not is_cv_fold:
                        mlp_params = sum(p.numel() for p in self.model.correction_mlp.parameters())
                        print(f"    🔥 纠偏模块(Correction MLP)已强制激活, 参数量: {mlp_params:,}")

                if not is_cv_fold:
                    trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    frozen_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
                    print(f"    📊 最终参数统计: 可训练={trainable_count:,}, 冻结={frozen_count:,}")

            # ============ LoRA微调模式 ============
            if use_lora and LORA_MODULE_AVAILABLE and not use_residual_injection:
                if not is_cv_fold:
                    print(f"\n【LoRA微调模式】应用LoRA到模型...")

                conversion_stats = convert_to_lora(
                    model=self.model,
                    target_modules=lora_config.get('target_modules', ['linear', 'conv']),
                    r=lora_config.get('r', 8),
                    lora_alpha=lora_config.get('lora_alpha', 16.0),
                    lora_dropout=lora_config.get('lora_dropout', 0.0),
                    verbose=not is_cv_fold
                )

                if not is_cv_fold:
                    print(f"  LoRA转换完成，转换层数: {conversion_stats['total']}")

                # LoRA模式：只训练LoRA参数
                for name, param in self.model.named_parameters():
                    if 'lora_A' not in name and 'lora_B' not in name:
                        param.requires_grad = False

            # ============ 损失函数定义 ============
            class BaseCombinedLoss(nn.Module):
                def __init__(self, alpha=0.6, beta=0.3, gamma=0.01):
                    super().__init__()
                    self.alpha = alpha
                    self.beta = beta
                    self.gamma = gamma

                def forward_without_reduction(self, outputs, targets, is_zero_mask=None):
                    mse_losses = (outputs - targets) ** 2
                    l1_losses = torch.abs(outputs - targets)
                    base_losses = self.alpha * mse_losses + self.beta * l1_losses
                    if is_zero_mask is not None:
                        zero_mask = (is_zero_mask == 0)
                        if torch.any(zero_mask):
                            base_losses[zero_mask] += self.gamma * (outputs[zero_mask] ** 2)
                    return base_losses

                def forward(self, outputs, targets, is_zero_mask=None):
                    losses = self.forward_without_reduction(outputs, targets, is_zero_mask)
                    return torch.mean(losses)

            class AdaptiveWeightedLoss(nn.Module):
                def __init__(self, base_loss_fn, weight_power=3.0):
                    super().__init__()
                    self.base_loss_fn = base_loss_fn
                    self.weight_power = weight_power

                def forward(self, outputs, targets, is_zero_mask=None):
                    base_weights = torch.abs(targets) ** self.weight_power
                    high_mask = targets > 0.3
                    extra_weights = torch.ones_like(targets)
                    extra_weights[high_mask] = 5.0
                    weights = base_weights * extra_weights
                    weights = weights + 0.5
                    weights = weights / weights.mean()
                    individual_losses = self.base_loss_fn.forward_without_reduction(outputs, targets, is_zero_mask)
                    weighted_loss = torch.mean(individual_losses * weights)
                    return weighted_loss

            # 设置损失函数。增量阶段虽然加载权重，但仍走预训练损失语义。
            if load_pretrained and not is_incremental_pretrain:
                self.criterion = BaseCombinedLoss(alpha=0.6, beta=0.3, gamma=0.01)
                if not is_cv_fold:
                    print(f"  损失函数: BaseCombinedLoss (微调模式)")
            else:
                base_loss = BaseCombinedLoss(alpha=0.6, beta=0.3, gamma=0.01)
                self.criterion = AdaptiveWeightedLoss(base_loss, weight_power=1.8)
                if not is_cv_fold:
                    mode_text = "增量预训练" if is_incremental_pretrain else "从头预训练"
                    print(f"  损失函数: AdaptiveWeightedLoss (weight_power=1.8, {mode_text})")

            # ============ 优化器配置 ============
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

            if not trainable_params:
                print("警告: 没有可训练参数！检查冻结策略")
                return False

            # 根据模式设置学习率
            if use_residual_injection:
                lr = self.config.get("fine_tune_lr", 1e-4)
                weight_decay = 0.01
                self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
                if not is_cv_fold:
                    print(f"  残差注入优化器: AdamW, lr={lr:.2e}, weight_decay={weight_decay}")

            elif use_lora and LORA_MODULE_AVAILABLE:
                lr = self.config.get("fine_tune_lr", self.config["learning_rate"] * 0.1)
                self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-6)
                if not is_cv_fold:
                    print(f"  LoRA优化器: AdamW, lr={lr:.2e}")

            elif is_incremental_pretrain:
                # 加载上一阶段模型后继续预训练：所有可训练参数使用统一预训练学习率。
                lr = self.config["learning_rate"]
                self.optimizer = optim.AdamW(
                    trainable_params,
                    lr=lr,
                    weight_decay=self.config["weight_decay"],
                )
                if not is_cv_fold:
                    print(f"  增量预训练优化器: AdamW, lr={lr:.2e}")

            elif load_pretrained:
                # 传统微调：使用分层学习率
                lr_head = self.config.get("lr_head", 5e-4)
                lr_transformer = self.config.get("lr_transformer", 2e-5)
                lr_encoder = self.config.get("lr_encoder", 5e-5)

                head_params = []
                transformer_params = []
                encoder_params = []
                assigned_params = set()

                for name, param in self.model.named_parameters():
                    if param.requires_grad and param not in assigned_params:
                        name_lower = name.lower()
                        if any(k in name_lower for k in ['head', 'regression', 'output', 'fc', 'correction']):
                            head_params.append(param)
                            assigned_params.add(param)
                        elif 'transformer' in name_lower:
                            transformer_params.append(param)
                            assigned_params.add(param)
                        else:
                            encoder_params.append(param)
                            assigned_params.add(param)

                param_groups = []
                if head_params:
                    param_groups.append({'params': head_params, 'lr': lr_head})
                if transformer_params:
                    param_groups.append({'params': transformer_params, 'lr': lr_transformer})
                if encoder_params:
                    param_groups.append({'params': encoder_params, 'lr': lr_encoder})

                self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
                if not is_cv_fold:
                    print(f"  【分层学习率优化器】共 {len(param_groups)} 个参数组")

            else:
                lr = self.config["learning_rate"]
                self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=self.config["weight_decay"])
                if not is_cv_fold:
                    print(f"  训练优化器: AdamW, lr={lr:.2e}")

            # ============ 学习率调度器 ============
            # 每次build_model都必须重置，避免上一折残留。
            self.scheduler_step_per_batch = False

            is_full_refit_mode = bool(
                self.config.get("_is_full_refit", False)
            )

            if is_full_refit_mode:
                scheduler_mode = str(
                    self.config.get("_final_scheduler", "cosine")
                ).lower()
            else:
                scheduler_mode = str(
                    self.config.get("lr_scheduler", "plateau")
                ).lower()

            if scheduler_mode == "warmup_cosine":
                if self.train_loader is None:
                    raise RuntimeError(
                        "创建warmup+cosine前，self.train_loader尚未设置"
                    )

                if is_full_refit_mode:
                    schedule_epochs = int(
                        self.config.get(
                            "_final_epochs",
                            self.config.get("epochs", 100),
                        )
                    )
                else:
                    schedule_epochs = int(
                        self.config.get(
                            "pretrain_cv_epochs",
                            self.config.get("epochs", 100),
                        )
                    )

                steps_per_epoch = len(self.train_loader)
                total_steps = max(
                    1,
                    schedule_epochs * steps_per_epoch,
                )

                warmup_ratio = float(
                    self.config.get("warmup_ratio", 0.05)
                )
                warmup_steps = max(
                    1,
                    int(total_steps * warmup_ratio),
                )

                max_lr = float(
                    self.config.get("learning_rate", 1e-4)
                )
                start_lr = float(
                    self.config.get("warmup_start_lr", 1e-5)
                )
                min_lr = float(
                    self.config.get("min_lr", 1e-6)
                )

                if not (0.0 < min_lr <= start_lr <= max_lr):
                    raise ValueError(
                        "warmup+cosine要求 "
                        f"0 < min_lr <= start_lr <= max_lr，当前为 "
                        f"{min_lr}, {start_lr}, {max_lr}"
                    )

                start_factor = start_lr / max_lr
                min_factor = min_lr / max_lr

                def warmup_cosine_lambda(step):
                    step = min(max(int(step), 0), total_steps)

                    if step < warmup_steps:
                        progress = step / max(1, warmup_steps)
                        return (
                            start_factor
                            + (1.0 - start_factor) * progress
                        )

                    cosine_steps = max(
                        1,
                        total_steps - warmup_steps,
                    )
                    progress = (
                        step - warmup_steps
                    ) / cosine_steps
                    progress = min(max(progress, 0.0), 1.0)

                    cosine_value = 0.5 * (
                        1.0 + np.cos(np.pi * progress)
                    )

                    return (
                        min_factor
                        + (1.0 - min_factor) * cosine_value
                    )

                # optimizer的base_lr仍为max_lr；
                # LambdaLR在step=0时把实际LR设为start_lr。
                self.scheduler = optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=warmup_cosine_lambda,
                )
                self.scheduler_step_per_batch = True

                print(
                    "  Warmup+Cosine调度器:"
                    f" start_lr={start_lr:.2e},"
                    f" max_lr={max_lr:.2e},"
                    f" min_lr={min_lr:.2e}"
                )
                print(
                    f"    epochs={schedule_epochs},"
                    f" steps/epoch={steps_per_epoch},"
                    f" total_steps={total_steps},"
                    f" warmup_steps={warmup_steps}"
                    f" ({warmup_ratio * 100:.1f}%)"
                )
                print(
                    "    当前实际学习率: "
                    f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

            elif (
                is_full_refit_mode
                and scheduler_mode == "cosine"
            ):
                final_epochs = int(
                    self.config.get("_final_epochs", 100)
                )
                self.scheduler = (
                    optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=final_epochs,
                        eta_min=1e-6,
                    )
                )
                print(
                    "  全量refit调度器: "
                    f"CosineAnnealingLR "
                    f"(T_max={final_epochs}, eta_min=1e-6)"
                )

            elif is_incremental_pretrain:
                self.scheduler = (
                    optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode="min",
                        factor=0.5,
                        patience=10,
                        min_lr=1e-7,
                    )
                )
                if not is_cv_fold:
                    print(
                        "  增量预训练调度器: "
                        "ReduceLROnPlateau (patience=10)"
                    )

            elif load_pretrained:
                self.scheduler = (
                    optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode="min",
                        factor=0.5,
                        patience=10,
                        min_lr=1e-7,
                    )
                )
                if not is_cv_fold:
                    print(
                        "  微调调度器: "
                        "ReduceLROnPlateau (patience=10)"
                    )

            else:
                self.scheduler = (
                    optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer,
                        mode="min",
                        factor=0.5,
                        patience=10,
                    )
                )
                if not is_cv_fold:
                    print(
                        "  训练调度器: ReduceLROnPlateau"
                    )

            # 打印模型信息（交叉验证时简化）
            if not is_cv_fold:
                self._print_model_info(freeze_backbone, use_lora)
                print(f"\n✅ 模型构建成功!")
                if use_residual_injection:
                    print(f"  模式: 残差注入 (Parallel Residual Adapter)")
                elif use_lora:
                    print(f"  模式: LoRA微调")
                elif use_residual:
                    print(f"  模式: 残差学习")
                elif self.config.get("use_gate", False):
                    print(f"  模式: 门控融合")
                elif is_incremental_pretrain:
                    print(f"  模式: 渐进式增量预训练（全参数）")
                elif load_pretrained:
                    print(f"  模式: 传统微调（分层学习率）")
                else:
                    print(f"  模式: 从头训练")
            else:
                # 交叉验证时简化输出
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"  ✅ 模型构建完成 (可训练: {trainable:,}/{total:,})")

            return True

        except Exception as e:
            print(f"\n❌ 模型构建失败: {e}")
            traceback.print_exc()
            return False
    
    def save_lora_weights(self, filename="lora_weights.pth"):
        """仅保存LoRA权重"""
        lora_state_dict = {}
        
        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_state_dict[name] = param.data.clone()
        
        if lora_state_dict:
            save_path = self.save_dir / filename
            torch.save(lora_state_dict, save_path)
            print(f"✓ LoRA权重已保存到: {save_path}")
            return True
        else:
            print("⚠ 没有找到LoRA权重")
            return False
    
    def load_lora_weights(self, filepath):
        """加载LoRA权重"""
        try:
            lora_weights = torch.load(filepath, map_location=self.device)
            
            for name, param in self.model.named_parameters():
                if name in lora_weights:
                    param.data.copy_(lora_weights[name])
            
            print(f"✓ LoRA权重已加载: {filepath}")
            return True
        except Exception as e:
            print(f"✗ 加载LoRA权重失败: {e}")
            return False
        

        
    def _load_pretrained_weights(self, model, pretrained_path):
        """加载预训练权重，处理维度不匹配（支持13/15维 → 16维扩展）"""
        try:
            print(f"加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)

            # 获取模型权重
            if isinstance(checkpoint, dict):
                model_state_dict = None
                for key in ["model_state_dict", "state_dict", "model"]:
                    if key in checkpoint:
                        model_state_dict = checkpoint[key]
                        print(f"找到模型权重键: {key}")
                        break

                if model_state_dict is None:
                    print("警告: 未找到模型权重，使用随机初始化")
                    return False
            else:
                model_state_dict = checkpoint

            # 获取当前模型的 state_dict
            current_state_dict = model.state_dict()

            # 手动处理维度不匹配的层
            matched_count = 0
            mismatched_count = 0
            extended_count = 0  # 新增：记录成功扩展的层数

            for name, param in model_state_dict.items():
                if name in current_state_dict:
                    if param.shape == current_state_dict[name].shape:
                        # 形状匹配，直接加载
                        current_state_dict[name].copy_(param)
                        matched_count += 1
                    else:
                        # 检查是否为 PointEncoder 第一层（输入维度扩展）
                        if 'point_encoder' in name and 'weight' in name:
                            old_dim = param.shape[1]
                            new_dim = current_state_dict[name].shape[1]

                            # 支持 13→16 或 15→16 的扩展
                            if old_dim < new_dim and param.shape[0] == current_state_dict[name].shape[0]:
                                print(f"  🔧 扩展 PointEncoder 权重: {name} ({old_dim} → {new_dim})")

                                # 保留前 old_dim 维的预训练权重
                                current_state_dict[name][:, :old_dim].copy_(param)

                                # 新增维度用很小的值初始化（0.01倍的标准初始化）
                                with torch.no_grad():
                                    # 计算新增部分的均值，用于平滑过渡
                                    old_weight_mean = param.abs().mean().item()
                                    init_scale = old_weight_mean * 0.1 if old_weight_mean > 0 else 0.01
                                    current_state_dict[name][:, old_dim:] = current_state_dict[name][:, old_dim:] * init_scale

                                extended_count += 1
                                matched_count += 1
                                print(f"    ✓ 已保留前{old_dim}维预训练权重，新增{new_dim - old_dim}维已初始化(scale={init_scale:.4f})")
                            else:
                                mismatched_count += 1
                                print(f"  ⚠ 跳过不匹配的层: {name}")
                                print(f"    预训练权重形状: {param.shape}")
                                print(f"    当前模型形状: {current_state_dict[name].shape}")

                        elif 'point_encoder' in name and 'bias' in name:
                            # bias 维度也可能不匹配，但通常输出维度不变
                            if param.shape[0] == current_state_dict[name].shape[0]:
                                current_state_dict[name].copy_(param)
                                matched_count += 1
                            else:
                                mismatched_count += 1
                                print(f"  ⚠ 跳过不匹配的 bias 层: {name}")
                        else:
                            # 其他不匹配的层，跳过
                            mismatched_count += 1
                            print(f"  ⚠ 跳过不匹配的层: {name}")
                            print(f"    预训练权重形状: {param.shape}")
                            print(f"    当前模型形状: {current_state_dict[name].shape}")

            # 加载处理后的权重
            model.load_state_dict(current_state_dict, strict=False)

            print(f"✓ 预训练权重加载成功")
            print(f"  匹配层数: {matched_count}")
            print(f"  跳过层数: {mismatched_count}")
            if extended_count > 0:
                print(f"  扩展层数: {extended_count}")

            # 特别提示 PointEncoder 维度变化
            if extended_count > 0:
                print(f"\n  【提示】点特征维度已从预训练模型扩展到16维")
                print(f"  - 前13/15维保留预训练权重")
                print(f"  - 新增维度已用较小的值初始化，平滑过渡")
            elif mismatched_count > 0:
                print(f"\n  【提示】存在维度不匹配的层，已跳过")
                print(f"  其他层权重已从预训练模型迁移")

            return True

        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            traceback.print_exc()
            return False


    def _interpolate_nan_patch(self, patch: np.ndarray) -> np.ndarray:
        """使用scipy插值填充NaN - 从data_station_online_swe.py复制过来的函数"""
        if not np.isnan(patch).any():
            return patch
        
        try:
            
            # 获取网格坐标
            x = np.arange(patch.shape[1])
            y = np.arange(patch.shape[0])
            xx, yy = np.meshgrid(x, y)
            
            # 有效点
            valid_mask = ~np.isnan(patch)
            if not valid_mask.any():
                # 全部是NaN，填充为0
                return np.zeros_like(patch)
            
            # 如果有效点太少，直接使用均值填充
            if np.sum(valid_mask) < 3:  # 少于3个有效点
                mean_value = np.nanmean(patch)
                if np.isnan(mean_value):
                    mean_value = 0.0
                
                result = patch.copy()
                result[np.isnan(result)] = mean_value
                return result
            
            # 获取有效点的坐标和值
            valid_points = np.column_stack([xx[valid_mask], yy[valid_mask]])
            valid_values = patch[valid_mask]
            
            # 检查维度 - 如果所有点都在一条直线上，使用最近邻插值
            unique_x = np.unique(valid_points[:, 0])
            unique_y = np.unique(valid_points[:, 1])
            
            if len(unique_x) == 1 or len(unique_y) == 1:
                # 点都在同一行或同一列上，使用最近邻插值
                interpolation_method = 'nearest'
            else:
                interpolation_method = 'linear'
            
            # 无效点
            invalid_mask = np.isnan(patch)
            if not invalid_mask.any():
                return patch  # 没有无效点，直接返回
            
            invalid_points = np.column_stack([xx[invalid_mask], yy[invalid_mask]])
            
            try:
                # 根据情况选择插值方法
                interpolated = griddata(valid_points, valid_values, invalid_points, 
                                        method=interpolation_method, fill_value=0.0)
                
                # 创建结果
                result = patch.copy()
                result[invalid_mask] = interpolated
                return result
                
            except Exception as e:
                # 如果插值失败，使用简单的均值填充
                print(f"警告: {interpolation_method}插值失败 ({e})，使用均值填充")
                
                # 计算局部均值
                result = patch.copy()
                
                # 对于每个NaN像素，使用其3x3邻域的均值
                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        if np.isnan(result[i, j]):
                            # 获取3x3邻域
                            i_min = max(0, i-1)
                            i_max = min(result.shape[0], i+2)
                            j_min = max(0, j-1)
                            j_max = min(result.shape[1], j+2)
                            
                            neighborhood = result[i_min:i_max, j_min:j_max]
                            valid_neighbors = neighborhood[~np.isnan(neighborhood)]
                            
                            if len(valid_neighbors) > 0:
                                result[i, j] = np.mean(valid_neighbors)
                            else:
                                result[i, j] = 0.0
                
                return result
        except Exception as e:
            print(f"插值函数调用失败: {e}")
            # 如果所有方法都失败，返回0填充的patch
            return np.nan_to_num(patch, nan=0.0)
        
    def _freeze_backbone_layers(self):
        """冻结主干网络层"""
        # 根据模型类型冻结不同部分
        model_type = self.config["model_type"]

        if model_type == "full":
            # 冻结卷积部分和融合部分
            for name, param in self.model.named_parameters():
                if "spatial_net" in name or "fusion_net" in name:
                    param.requires_grad = False
                    print(f"  冻结: {name}")
            print("✓ 冻结了spatial_net和fusion_net部分")

        elif model_type == "spatial_only":
            # 冻结整个spatial_net
            for name, param in self.model.named_parameters():
                if "spatial_net" in name:
                    param.requires_grad = False
                    print(f"  冻结: {name}")
            print("✓ 冻结了spatial_net部分")

        elif model_type == "point_only":
            # 冻结point_net
            for name, param in self.model.named_parameters():
                if "point_net" in name:
                    param.requires_grad = False
                    print(f"  冻结: {name}")
            print("✓ 冻结了point_net部分")





    def _print_model_info(self, freeze_backbone=False, use_lora=False):
        """打印模型信息，支持LoRA"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        print(f"\n模型信息:")
        print(f"  类型: {self.config['model_type']}")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  参数量占比: {trainable_params / total_params * 100:.2f}%")
        
        if use_lora:
            # 统计LoRA参数
            lora_params = 0
            for name, param in self.model.named_parameters():
                if ('lora_A' in name or 'lora_B' in name) and param.requires_grad:
                    lora_params += param.numel()
            
            if lora_params > 0:
                print(f"  LoRA参数: {lora_params:,} ({lora_params/total_params*100:.2f}%)")
        
        if freeze_backbone and not use_lora:
            print(f"  主干网络已冻结")
        
        # 打印各模块参数量
        print(f"\n各模块参数量:")
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                status = "LoRA" if use_lora and trainable > 0 else "冻结" if trainable == 0 else "可训练"
                print(f"  {name:20s}: {params:10,} ({trainable:10,} {status})")
            
    def train_epoch(self, epoch, is_fine_tune=False):
        """
        训练一个 epoch。

        支持：
        1. 普通预训练
        2. 纯站点微调
        3. mixed mode: station + pretrain pseudo-label
           - source_flag = 0: 站点实测样本
           - source_flag = 1: 预训练伪标签样本
        4. mixed mode 下预训练样本使用小权重 loss
        5. 可选站点高 SWE 加权
        """

        self.model.train()

        # FINETUNE_FROZEN_MODULES_EVAL_V1
        # requires_grad=False 并不会自动关闭 Dropout 或 BatchNorm 的状态更新。
        # 对完全冻结的子模块强制 eval，防止不同fold的冻结表征发生漂移。
        if is_fine_tune:
            frozen_eval_count = 0
            for module_name, module in self.model.named_modules():
                if module_name == "":
                    continue
                module_params = list(module.parameters(recurse=True))
                if module_params and not any(p.requires_grad for p in module_params):
                    module.eval()
                    frozen_eval_count += 1
            if epoch == 0:
                print(
                    f"  ✅ 冻结模块保持eval: {frozen_eval_count} 个子模块 "
                    "(关闭其Dropout/BN更新)"
                )

        total_loss = 0.0
        batch_count = 0

        is_mixed_mode = bool(self.config.get("mixed_mode", False))

        # ============ AMP / 时间诊断 ============
        use_amp = bool(self.config.get("use_amp", False)) and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        data_time_total = 0.0
        gpu_time_total = 0.0
        last_time = time.time()

        if epoch == 0:
            print(f"  AMP enabled: {use_amp}")

        # ============ 初始化 MixUp 包装器 ============
        if (
            not is_mixed_mode
            and not hasattr(self, "mixup_wrapper")
            and self.config.get("use_mixup", False)
        ):
            try:
                from main_tune import MixUpWrapper
                self.mixup_wrapper = MixUpWrapper(self.config)
                print(
                    f"  MixUp已启用: "
                    f"alpha={self.config.get('mixup_alpha', 0.2)}, "
                    f"prob={self.config.get('mixup_prob', 0.5)}"
                )
            except Exception as e:
                print(f"  ⚠ MixUp 初始化失败，关闭 MixUp: {e}")
                self.mixup_wrapper = None

        # ============ 课程学习配置 ============
        use_curriculum = (
            self.config.get("use_curriculum", False)
            and is_fine_tune
            and (not is_mixed_mode)
            and hasattr(self, "sample_difficulties")
        )

        total_epochs = self.config.get("fine_tune_epochs", self.config.get("epochs", 100))
        progress = epoch / total_epochs if total_epochs > 0 else 1.0

        if use_curriculum:
            curriculum_start = self.config.get("curriculum_start", 0.3)
            threshold = curriculum_start + (1.0 - curriculum_start) * progress

            if epoch % 5 == 0:
                print(f"  📚 课程学习: 进度={progress:.2f}, 阈值={threshold:.3f}")
        else:
            threshold = None

        # ============ CUDA 检查 ============
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if epoch == 0:
                print(f"  显存使用: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # ============ 模式判定 ============
        use_lora = is_fine_tune and self.config.get("use_lora", False)
        is_residual_model = is_fine_tune and self.config.get("residual_injection", False)
        is_gate_model = is_fine_tune and self.config.get("use_gate", False)
        use_traditional_fine_tune = is_fine_tune and not (
            use_lora or is_residual_model or is_gate_model
        )

        if is_fine_tune:
            if use_lora:
                print(f"  【LoRA微调模式】Epoch {epoch + 1}")
            elif is_residual_model:
                print(f"  【残差注入模式】Epoch {epoch + 1}")
                print(f"    当前弹性系数 λ = {self.config.get('lambda_elastic', 0.1)}")
            elif is_gate_model:
                print(f"  【门控融合模式】Epoch {epoch + 1}")
            elif use_traditional_fine_tune:
                if is_mixed_mode:
                    print(f"  【传统微调 + Mixed Mode】Epoch {epoch + 1}")
                else:
                    print(f"  【传统微调模式】Epoch {epoch + 1}")

        # ============ 损失函数 ============
        if is_fine_tune:
            if epoch == 0:
                print("    NSE-oriented loss: 0.8*MSE + 0.2*Huber + bias/var constraints")
                if is_mixed_mode:
                    print(
                        f"    mixed source-aware loss: "
                        f"pretrain_loss_weight={self.config.get('pretrain_loss_weight', 0.0)}"
                    )
        else:
            loss_fn = self.criterion

        # ============ Epoch 0 数据检查 ============
        if epoch == 0:
            print("\n【数据完整性检查】")
            try:
                batch = next(iter(self.train_loader))

                if isinstance(batch, (list, tuple)):
                    conv = batch[0]
                    point = batch[1]
                    target = batch[2]
                    # FINETUNE_BATCH_DEBUG_SOURCE_FLAG_V1
                    sample_idx_debug = batch[5] if len(batch) >= 6 else None
                    source_flag_debug = batch[6] if len(batch) >= 7 else None

                    print(f"  batch 元素数量: {len(batch)}")
                    print("  conv stats:")
                    print(f"    shape: {conv.shape}")
                    print(f"    dtype: {conv.dtype}")
                    print(f"    range: [{conv.min():.4f}, {conv.max():.4f}]")
                    print(f"    mean: {conv.mean():.4f} ± {conv.std():.4f}")
                    print(f"    has nan: {torch.isnan(conv).any()}")
                    print(f"    has inf: {torch.isinf(conv).any()}")

                    print(f"  point shape: {point.shape}")
                    print(f"  target shape: {target.shape}")

                    if source_flag_debug is not None:
                        source_flag_debug = source_flag_debug.reshape(-1).long()
                        print(f"  source_flag 前10个: {source_flag_debug[:10]}")
                        print(f"  station样本数: {(source_flag_debug == 0).sum().item()}")
                        print(f"  pretrain样本数: {(source_flag_debug == 1).sum().item()}")
                    elif sample_idx_debug is not None:
                        sample_idx_debug = sample_idx_debug.reshape(-1).long()
                        print(f"  sample_idx 前10个: {sample_idx_debug[:10]}")
                        print("  当前为纯站点6元素batch，不包含source_flag")
                else:
                    print("  ⚠ batch 格式异常，跳过检查")

            except Exception as e:
                print(f"  ⚠ 数据完整性检查失败: {e}")

        # ============================================================
        # 主训练循环
        # ============================================================
        for batch_idx, batch_data in enumerate(self.train_loader):
            if is_fine_tune:
                self._apply_fusion_ft_warmup_lr(
                    epoch,
                    batch_idx,
                    max(1, len(self.train_loader)),
                )

            data_time = time.time() - last_time
            data_time_total += data_time
            step_start = time.time()

            # 🔥 每个 batch 重置
            prior_norm_for_weight = None
            outputs_cf_flat = None  # 反事实输出

            try:
                # [COMPAT] Dataset 返回值兼容层
                # 历史版本和不同数据模式返回的 batch 长度不同：
                #
                #   3: conv, point, target
                #   4: conv, point, target, is_zero_mask
                #   5: conv, point, target, is_zero_mask, grid_val_norm
                #   6: conv, point, target, is_zero_mask, grid_val_norm, sample_idx
                #   7: conv, point, target, is_zero_mask, grid_val_norm, sample_idx, source_flag
                #
                # 不要轻易删除旧格式支持，否则旧缓存/旧 Dataset/旧 mixed mode 可能直接崩。
                # ============ 解析 batch ============
                grid_val_norm = None
                sample_idx = None
                source_flag = None

                if len(batch_data) == 7:
                    conv_feats, point_feats, targets, is_zero_mask, grid_val_norm, sample_idx, source_flag = batch_data
                elif len(batch_data) == 6:
                    conv_feats, point_feats, targets, is_zero_mask, grid_val_norm, sample_idx = batch_data
                elif len(batch_data) == 5:
                    conv_feats, point_feats, targets, is_zero_mask, grid_val_norm = batch_data
                elif len(batch_data) == 4:
                    conv_feats, point_feats, targets, is_zero_mask = batch_data
                elif len(batch_data) == 3:
                    conv_feats, point_feats, targets = batch_data
                    is_zero_mask = torch.where(
                        targets > 0,
                        torch.ones_like(targets),
                        torch.zeros_like(targets)
                    )
                else:
                    print(f"  批次 {batch_idx + 1}: 数据格式错误，长度={len(batch_data)}，跳过")
                    continue

                # ============ mixed mode 下关闭 MixUp ============
                original_targets = None
                mixup_lam = 1.0
                mixup_index = None

                if (
                    not is_mixed_mode
                    and hasattr(self, "mixup_wrapper")
                    and self.mixup_wrapper is not None
                ):
                    original_targets = targets.clone()
                    conv_feats, point_feats, targets, mixup_lam, mixup_index = \
                        self.mixup_wrapper(conv_feats, point_feats, targets, epoch)

                # ============ NaN / Inf 处理 ============
                conv_feats = torch.nan_to_num(conv_feats, nan=0.0, posinf=0.0, neginf=0.0)
                point_feats = torch.nan_to_num(point_feats, nan=0.0, posinf=0.0, neginf=0.0)
                targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
                is_zero_mask = torch.nan_to_num(is_zero_mask, nan=1.0, posinf=1.0, neginf=1.0)

                if grid_val_norm is not None:
                    grid_val_norm = torch.nan_to_num(grid_val_norm, nan=0.0, posinf=0.0, neginf=0.0)

                # ============ 移动到设备 ============
                conv_feats = conv_feats.to(self.device, non_blocking=True)
                point_feats = point_feats.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                is_zero_mask = is_zero_mask.to(self.device, non_blocking=True)

                if grid_val_norm is not None:
                    grid_val_norm = grid_val_norm.to(self.device, non_blocking=True)

                if source_flag is not None:
                    source_flag = source_flag.to(self.device, non_blocking=True)

                conv_feats = conv_feats.contiguous()
                point_feats = point_feats.contiguous()

                # [DEFAULT-2026] 当前 Clean-18D 默认点特征维度为 18。
                # [COMPAT] 这里是兜底，不是强制设计。
                #          优先使用 config["C_point"] / dataset.C_point。
                #          只有在旧缓存或旧模型缺少维度信息时才回退到 18。
                expected_point_dim = int(self.config.get("C_point", 18))

                if point_feats.shape[1] != expected_point_dim:
                    if point_feats.shape[1] < expected_point_dim:
                        padding = torch.zeros(
                            point_feats.shape[0],
                            expected_point_dim - point_feats.shape[1],
                            device=point_feats.device,
                            dtype=point_feats.dtype
                        )
                        point_feats = torch.cat([point_feats, padding], dim=1)
                    else:
                        point_feats = point_feats[:, :18]

                # ============ 保存原始 point_feats（用于反事实 loss） ============
                point_feats_original = point_feats.clone() if point_feats is not None else None

                # ============ 🔥 Prior Dropout (关闭，p=0.0) ============
                prior_col = self.config.get("physical_prior_col", None)

                if is_fine_tune and self.config.get("use_prior_dropout", False):
                    if isinstance(prior_col, int) and point_feats is not None and point_feats.dim() == 2 and point_feats.shape[1] > prior_col:
                        p = float(self.config.get("prior_dropout_p", 0.0))
                        prior_norm_for_weight = point_feats[:, prior_col].detach().clone()
                        drop_mask = torch.rand(point_feats.shape[0], device=point_feats.device) < p
                        if drop_mask.any():
                            point_feats = point_feats.clone()
                            point_feats[drop_mask, prior_col] = 0.0

                # ============ 正常前向传播 ============
                gpu_start = time.time()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if is_residual_model:
                        outputs, delta_y = self.model(conv_feats, point_feats, grid_val_norm)
                    elif is_gate_model:
                        outputs, y_pre, y_fine, alpha = self.model(conv_feats, point_feats)
                    else:
                        outputs = self.model(conv_feats, point_feats)

                    # [DANGER] 必须 flatten。
                    # outputs 常见形状是 [B,1]，targets 常见形状是 [B]。
                    # 如果不 flatten，PyTorch 会广播成 [B,B]，loss 会错误但不一定报错。
                    # ============ shape 统一 ============
                    outputs_flat = outputs.reshape(-1)
                    targets_flat = targets.reshape(-1)

                    # ============ target=0 样本硬约束 ============
                    zero_mask_flat = is_zero_mask.reshape(-1).to(outputs_flat.device)

                    if zero_mask_flat.numel() == outputs_flat.numel():
                        outputs_flat = outputs_flat * zero_mask_flat
                        if is_residual_model:
                            delta_y = delta_y.reshape(-1) * zero_mask_flat
                    else:
                        if batch_idx == 0:
                            print(
                                f"  ⚠ is_zero_mask数量({zero_mask_flat.numel()}) "
                                f"!= outputs数量({outputs_flat.numel()})，跳过硬零约束"
                            )

                    # ========================================================
                    # 损失计算
                    # ========================================================
                    if is_fine_tune:
                        # SIMPLIFIED_NSE_LOGCOSH_V1
                        loss, loss_diag = self._compute_simplified_nse_logcosh_loss(
                            outputs_flat,
                            targets_flat,
                            source_flag=source_flag,
                        )

                        if epoch == 0 and batch_idx == 0:
                            print("    简化NSE-oriented微调损失已启用:")
                            print(
                                f"      base={float(loss_diag['base_loss']):.6f}, "
                                f"high_bias={float(loss_diag['high_bias_loss']):.6f}, "
                                f"variance={float(loss_diag['variance_loss']):.6f}, "
                                f"ccc={float(loss_diag['ccc']):.6f}, "
                                f"ccc_loss={float(loss_diag['ccc_loss']):.6f}, "
                                f"high_count={loss_diag['high_count']}"
                            )
                            if self._fusion_ft_progressive_enabled():
                                print(
                                    "      Fusion FT温和尾部权重: "
                                    f">=20×{float(self.config.get('fusion_ft_swe_weight_ge20', 1.2)):.1f}, "
                                    f">=50×{float(self.config.get('fusion_ft_swe_weight_ge50', 1.5)):.1f}, "
                                    f">=80×{float(self.config.get('fusion_ft_swe_weight_ge80', 2.0)):.1f}, "
                                    f"high_bias_weight={float(self.config.get('fusion_ft_high_bias_weight', 0.0)):.1f}"
                                )
                                print(
                                    "      Fusion FT结构损失: "
                                    f"ccc_weight={float(loss_diag['ccc_weight']):.4f}, "
                                    f"variance_weight={float(loss_diag['variance_weight']):.1f}"
                                )

                    else:
                        # ============ 预训练模式 ============
                        if mixup_index is not None and mixup_lam != 1.0 and original_targets is not None:
                            target_b = original_targets[mixup_index].to(self.device)
                            target_b = target_b.reshape(-1)

                            loss_a = self.criterion(outputs_flat, targets_flat)
                            loss_b = self.criterion(outputs_flat, target_b)

                            loss = mixup_lam * loss_a + (1.0 - mixup_lam) * loss_b
                        else:
                            loss = self.criterion(outputs_flat, targets_flat)

                    # ============ loss 检查 ============
                    if loss.dim() > 0:
                        loss = loss.mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  批次 {batch_idx + 1}: 损失为 {loss.item()}，跳过")
                    continue

                gradient_diag_epochs = int(
                    self.config.get(
                        "fusion_ft_gradient_diagnostic_epochs",
                        5,
                    )
                )
                if (
                    is_fine_tune
                    and self._fusion_ft_progressive_enabled()
                    and batch_idx == 0
                    and epoch < gradient_diag_epochs
                ):
                    gradient_diag = (
                        self._diagnose_fusion_ft_loss_gradients(
                            loss_diag["_base_loss_tensor"],
                            loss_diag[
                                "_weighted_ccc_loss_tensor"
                            ],
                        )
                    )
                    print(
                        "    [CCC梯度诊断] "
                        f"Epoch {epoch + 1}, "
                        f"status={gradient_diag['status']}"
                    )
                    print(
                        "      ||grad_station||="
                        f"{gradient_diag['base_grad_norm']:.6e}, "
                        "||grad_weighted_CCC||="
                        f"{gradient_diag['weighted_ccc_grad_norm']:.6e}"
                    )
                    print(
                        "      CCC/station梯度范数比="
                        f"{gradient_diag['ccc_to_base_grad_ratio']:.6f}, "
                        "梯度cosine="
                        f"{gradient_diag['gradient_cosine']:.6f}, "
                        "可训练参数张量数="
                        f"{gradient_diag['trainable_tensor_count']}"
                    )

               # ============ 反向传播 + 训练稳定性监控 ============
                self.optimizer.zero_grad(set_to_none=True)

                clip_grad = float(
                    self.config.get(
                        "clip_grad",
                        1.0,
                    )
                )

                clip_value = (
                    clip_grad * 2.0
                    if use_lora
                    else clip_grad
                )

                # --------------------------------------------------
                # 只在环境变量 STABILITY_MONITOR=1 时初始化。
                # 达到 max_steps 后 monitor.active 会自动变成 False。
                # --------------------------------------------------
                stability_monitor = getattr(
                    self,
                    "_stability_monitor",
                    None,
                )

                if (
                    stability_monitor is None
                    and StabilityMonitor.enabled_from_env()
                ):
                    exp_name = self.config.get(
                        "experiment_name",
                        self.config.get(
                            "exp_name",
                            "run",
                        ),
                    )

                    self._stability_monitor = (
                        StabilityMonitor.from_env(
                            save_dir=self.save_dir,
                            default_prefix=(
                                f"stability_{exp_name}"
                            ),
                        )
                    )

                    stability_monitor = (
                        self._stability_monitor
                    )

                monitor_active = (
                    stability_monitor is not None
                    and stability_monitor.active
                )

                grad_norm_value = float("nan")
                clip_triggered = False
                param_snapshot = None

                amp_scale_before = float("nan")
                amp_scale_after = float("nan")
                amp_step_skipped = False

                if use_amp:
                    scaler.scale(
                        loss
                    ).backward()

                    # 梯度统计和裁剪都必须基于unscale后的真实梯度。
                    if (
                        clip_grad > 0
                        or monitor_active
                    ):
                        scaler.unscale_(
                            self.optimizer
                        )

                    if clip_grad > 0:
                        # 返回值就是“裁剪前”的全局梯度范数。
                        grad_norm_tensor = (
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                clip_value,
                            )
                        )

                        grad_norm_value = float(
                            grad_norm_tensor
                            .detach()
                            .item()
                        )

                        clip_triggered = (
                            grad_norm_value
                            > clip_value
                        )

                    elif monitor_active:
                        grad_norm_value = (
                            StabilityMonitor
                            .global_grad_norm(
                                self.model
                            )
                        )

                    # optimizer.step前记录参数。
                    if monitor_active:
                        param_snapshot = (
                            StabilityMonitor
                            .snapshot_parameters(
                                self.model
                            )
                        )

                    amp_scale_before = float(
                        scaler.get_scale()
                    )

                    scaler.step(
                        self.optimizer
                    )

                    scaler.update()

                    amp_scale_after = float(
                        scaler.get_scale()
                    )

                    # 检测到Inf/NaN时，GradScaler会跳过step并降低scale。
                    amp_step_skipped = (
                        amp_scale_after
                        < amp_scale_before
                    )

                else:
                    loss.backward()

                    if clip_grad > 0:
                        grad_norm_tensor = (
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                clip_value,
                            )
                        )

                        grad_norm_value = float(
                            grad_norm_tensor
                            .detach()
                            .item()
                        )

                        clip_triggered = (
                            grad_norm_value
                            > clip_value
                        )

                    elif monitor_active:
                        grad_norm_value = (
                            StabilityMonitor
                            .global_grad_norm(
                                self.model
                            )
                        )

                    if monitor_active:
                        param_snapshot = (
                            StabilityMonitor
                            .snapshot_parameters(
                                self.model
                            )
                        )

                    self.optimizer.step()

                # optimizer.step之后计算实际参数变化。
                if monitor_active:
                    stability_monitor.record(
                        epoch=epoch + 1,
                        batch=batch_idx + 1,

                        lr_values=[
                            group.get(
                                "lr",
                                float("nan"),
                            )
                            for group
                            in self.optimizer.param_groups
                        ],

                        loss=float(
                            loss.detach().item()
                        ),

                        grad_norm=grad_norm_value,

                        clip_threshold=(
                            clip_value
                            if clip_grad > 0
                            else float("nan")
                        ),

                        clip_triggered=clip_triggered,
                        param_snapshot=param_snapshot,

                        pred=outputs_flat.detach(),
                        target=targets_flat.detach(),

                        swe_min=getattr(
                            self,
                            "swe_min",
                            0.0,
                        ),

                        swe_max=getattr(
                            self,
                            "swe_max",
                            400.0,
                        ),

                        amp_scale_before=amp_scale_before,
                        amp_scale_after=amp_scale_after,
                        amp_step_skipped=amp_step_skipped,
                    )

                # Warmup+Cosine按optimizer step更新。
                # Plateau和旧Cosine仍在epoch末更新。
                if (
                    self.scheduler is not None
                    and getattr(
                        self,
                        "scheduler_step_per_batch",
                        False,
                    )
                ):
                    self.scheduler.step()

                # ============ 时间诊断 ============
                # [DIAG] profile_timing 只用于短跑性能诊断。
                # True 时会调用 torch.cuda.synchronize()，使 gpu_time 统计更准确。
                #
                # [DANGER]
                #   synchronize 会破坏 CUDA 异步执行，正式训练会变慢。
                #   正式 10 折 / 95% 最终训练时必须保持 False。
                profile_timing = bool(self.config.get("profile_timing", False))
                if profile_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()

                gpu_time = time.time() - gpu_start
                gpu_time_total += gpu_time
                last_time = time.time()
                step_time = time.time() - step_start

                total_loss += float(loss.item())
                batch_count += 1

                if (batch_idx + 1) % 10 == 0:
                    if use_lora:
                        mode = "LoRA"
                    elif is_residual_model:
                        mode = "残差"
                    elif is_gate_model:
                        mode = "门控"
                    elif is_mixed_mode and is_fine_tune:
                        mode = "Mixed微调"
                    elif use_traditional_fine_tune:
                        mode = "微调"
                    else:
                        mode = "预训练"

                    print(
                        f"    {mode}批次 {batch_idx + 1}/{len(self.train_loader)} "
                        f"| 损失: {loss.item():.6f}"
                    )

                # 每50个batch打印时间诊断
                if profile_timing and (batch_idx + 1) % 50 == 0:
                    print(
                        f"    batch {batch_idx+1}: "
                        f"data_time={data_time_total/(batch_idx+1):.4f}s, "
                        f"gpu_time={gpu_time_total/(batch_idx+1):.4f}s, "
                        f"step_time={step_time:.4f}s"
                    )
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  批次 {batch_idx + 1}: CUDA显存不足，跳过")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return avg_loss
    
    def _mixed_epoch_diagnostic(self, epoch):
        """Mixed mode 每 epoch 诊断：样本分布、loss 分解、SWE 分布"""
        
        print(f"\n{'='*60}")
        print(f"【Mixed Epoch 诊断】Epoch {epoch + 1}")
        print(f"{'='*60}")
        
        # 收集训练集前若干 batch 的统计
        station_count = 0
        pretrain_count = 0
        station_loss_sum = 0.0
        pretrain_loss_sum = 0.0
        station_weight_sum = 0.0
        pretrain_weight_sum = 0.0
        station_swe_list = []
        station_pred_list = []
        max_batches = min(20, len(self.train_loader))
        
        swe_min = getattr(self, "swe_min", 0.0)
        swe_max = getattr(self, "swe_max", 200.0)
        
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch_data in enumerate(self.train_loader):
                if batch_idx >= max_batches:
                    break
                
                if len(batch_data) < 6:
                    continue
                conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe, source_flag = batch_data
                
                source_flag = source_flag.reshape(-1).long()
                station_mask = source_flag == 0
                pretrain_mask = source_flag == 1
                
                station_count += station_mask.sum().item()
                pretrain_count += pretrain_mask.sum().item()
                
                # 反归一化 target
                targets_mm = targets.reshape(-1) * (swe_max - swe_min) + swe_min
                station_swe = targets_mm[station_mask]
                station_swe_list.extend(station_swe.cpu().numpy().tolist())
                
                # 获取预测
                conv_feats = conv_feats.to(self.device)
                point_feats = point_feats.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(conv_feats, point_feats)
                preds_mm = outputs.reshape(-1) * (swe_max - swe_min) + swe_min
                station_pred = preds_mm[station_mask]
                station_pred_list.extend(station_pred.cpu().numpy().tolist())
            
            self.model.train()
        
        # 样本来源
        total = station_count + pretrain_count
        if total > 0:
            print(f"\n样本来源:")
            print(f"  station: {station_count}")
            print(f"  pretrain: {pretrain_count}")
            print(f"  实际比例: station={station_count/total*100:.1f}%, pretrain={pretrain_count/total*100:.1f}%")
        
        # Station SWE 分布
        if station_swe_list:
            station_swe_arr = np.array(station_swe_list)
            station_pred_arr = np.array(station_pred_list)
            print(f"\nstation SWE 分布:")
            for thresh in [30, 60, 80, 120]:
                n = int((station_swe_arr >= thresh).sum())
                print(f"  >= {thresh} mm: {n} ({n/len(station_swe_arr)*100:.1f}%)")
            
            print(f"\nstation target/pred mean:")
            print(f"  target: {station_swe_arr.mean():.2f} mm")
            print(f"  pred:   {station_pred_arr.mean():.2f} mm")
        
        pretrain_loss_weight = float(self.config.get("pretrain_loss_weight", 0.2))
        print(f"\nloss 配置:")
        print(f"  pretrain_loss_weight: {pretrain_loss_weight}")
        print(f"  use_high_swe_weight: {self.config.get('use_high_swe_weight', False)}")
        print(f"{'='*60}\n")
    
    def _analyze_bad_test_samples(self, all_test_predictions, all_test_targets):
        """分析测试集中 obs>=80 且 pred<40 的坏样本"""
        
        preds = np.array(all_test_predictions)
        targets = np.array(all_test_targets)
        
        bad_mask = (targets >= 80) & (preds < 40)
        bad_count = int(bad_mask.sum())
        total_high = int((targets >= 80).sum())
        
        print(f"\n{'='*60}")
        print(f"🔍 坏样本分析: obs>=80 且 pred<40")
        print(f"{'='*60}")
        print(f"  obs>=80 总样本数: {total_high}")
        print(f"  obs>=80 & pred<40: {bad_count} ({bad_count/max(1,total_high)*100:.1f}%)")
        
        if bad_count == 0:
            print(f"  ✅ 没有严重低估的坏样本")
            print(f"{'='*60}\n")
            return
        
        print(f"\n  坏样本详情 (前 30 个):")
        print(f"  {'station_id':<20} {'date':<12} {'obs':>8} {'pred':>8} {'fold':>5}")
        print(f"  {'-'*60}")
        
        bad_indices = np.where(bad_mask)[0]
        meta_list = getattr(self, '_test_meta_list', [])
        
        for rank, bi in enumerate(bad_indices[:30]):
            obs_val = float(targets[bi])
            pred_val = float(preds[bi])
            
            if bi < len(meta_list):
                m = meta_list[bi]
                sid = m.get('station_id', '?')[:18]
                date = m.get('date', '?')[:10]
                fold = m.get('fold_id', '?')
            else:
                sid = '?'
                date = '?'
                fold = '?'
            
            print(f"  {sid:<20} {date:<12} {obs_val:>8.2f} {pred_val:>8.2f} {str(fold):>5}")
        
        if bad_count > 30:
            print(f"  ... 还有 {bad_count-30} 个")
        
        # 保存到 CSV
        if meta_list:
            try:
                bad_meta = [meta_list[i] for i in bad_indices if i < len(meta_list)]
                df_bad = pd.DataFrame(bad_meta)
                save_path = self.save_dir / "bad_test_samples.csv"
                df_bad.to_csv(save_path, index=False, encoding='utf-8')
                print(f"\n  💾 坏样本已保存到: {save_path}")
            except Exception as e:
                print(f"\n  ⚠ 保存 CSV 失败: {e}")
        
        print(f"{'='*60}\n")
        
    def validate(
        self,
        dataloader=None,
        is_fine_tune=False,
        report_name=None,
    ):
        """验证方法 - 适配残差注入、门控、普通模型"""
        if dataloader is None:
            dataloader = self.val_loader

        if dataloader is None:
            print("⚠ 验证集为空")
            return {
                "loss": float('nan'),
                "rmse": 0,
                "mae": 0,
                "correlation": 0,
                "r2": 0,
                "n_samples": 0
            }

        validation_name = (
            str(report_name)
            if report_name
            else f"{'微调' if is_fine_tune else '训练'}验证"
        )
        print(f"\n【{validation_name}】开始...")

        self.model.eval()
        total_loss = 0
        batch_count = 0

        all_predictions = []
        all_targets = []
        all_is_zero = []

        # ============ 微调专用：初始化SmoothL1Loss ============
        if is_fine_tune:
            smooth_l1_criterion = nn.SmoothL1Loss(beta=0.01, reduction='mean')

        # 调试：获取期望的点特征维度（从模型或配置）
        expected_point_dim = self.config.get("C_point", 18)
        print(f"  期望点特征总维度: {expected_point_dim} (无Wide分支、无产品值)")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # [COMPAT] 验证集 batch 返回值兼容层
                # 不同 Dataset / 历史缓存可能返回：
                #   3: conv, point, target
                #   4: conv, point, target, is_zero_mask
                #   5: conv, point, target, is_zero_mask, raw_fused_swe
                #   6: conv, point, target, is_zero_mask, raw_fused_swe, extra
                #
                # [DANGER]
                #   validate() 和 train_epoch() 的 batch 解析规则要保持一致。
                #   否则训练能跑，验证可能 silently 用错字段。
                # ============ 1. 解析批次数据 ============
                if len(batch_data) == 6:
                    # 适配最新 Dataset 返回格式（6个值）                
                    conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe, _ = batch_data
                elif len(batch_data) == 5:
                    conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe = batch_data
                elif len(batch_data) == 4:
                    conv_feats, point_feats, targets, is_zero_mask = batch_data
                    raw_fused_swe = None
                elif len(batch_data) == 3:
                    conv_feats, point_feats, targets = batch_data
                    is_zero_mask = torch.where(targets > 0, 
                                             torch.ones_like(targets), 
                                             torch.zeros_like(targets))
                    raw_fused_swe = None
                else:
                    print(f"  ⚠️ 警告: 收到未知长度的数据包 ({len(batch_data)})，跳过")
                    continue

                # ============ 调试：第一个 batch 打印详细信息 ============
                if batch_idx == 0:
                    print("\n========== 验证集第一个 batch 调试 ==========")
                    print(f"  batch_data 长度: {len(batch_data)}")
                    print(f"  conv_feats shape: {conv_feats.shape}")
                    print(f"  point_feats shape: {point_feats.shape}")
                    print(f"  targets shape: {targets.shape}")
                    print(f"  targets 前5个值: {targets[:5]}")
                    print(f"  targets 是否有 NaN: {torch.isnan(targets).any()}")
                    print(f"  is_zero_mask 前5个: {is_zero_mask[:5] if is_zero_mask is not None else 'None'}")
                    if raw_fused_swe is not None:
                        print(f"  raw_fused_swe shape: {raw_fused_swe.shape}")
                    print(f"  实际 point_feats 维度: {point_feats.shape[1]}, 期望: {expected_point_dim}")

                # ============ 处理NaN ============
                nan_conv = torch.isnan(conv_feats).any().item()
                nan_point = torch.isnan(point_feats).any().item()
                nan_target = torch.isnan(targets).any().item()

                if nan_conv or nan_point or nan_target:
                    if nan_point:
                        point_feats = torch.nan_to_num(point_feats, nan=0.0)
                    if nan_target:
                        targets = torch.nan_to_num(targets, nan=0.0)
                        is_zero_mask = torch.where(targets > 0, 
                                                 torch.ones_like(targets), 
                                                 torch.zeros_like(targets))
                    if nan_conv:
                        conv_np = conv_feats.cpu().numpy()
                        conv_interp = np.zeros_like(conv_np)
                        for b in range(conv_np.shape[0]):
                            for c in range(conv_np.shape[1]):
                                patch = conv_np[b, c]
                                if np.isnan(patch).any():
                                    conv_interp[b, c] = self._interpolate_nan_patch(patch)
                                else:
                                    conv_interp[b, c] = patch
                        conv_feats = torch.from_numpy(conv_interp).to(self.device)

                if raw_fused_swe is not None and torch.isnan(raw_fused_swe).any():
                    raw_fused_swe = torch.nan_to_num(raw_fused_swe, nan=0.0)

                conv_feats = torch.nan_to_num(conv_feats, nan=0.0)
                point_feats = torch.nan_to_num(point_feats, nan=0.0)
                targets = torch.nan_to_num(targets, nan=0.0)
                is_zero_mask = torch.nan_to_num(is_zero_mask, nan=1.0)

                # ============ 修复点特征维度 ============
                if point_feats.shape[1] != expected_point_dim:
                    if point_feats.shape[1] < expected_point_dim:
                        padding = torch.zeros(point_feats.shape[0], 
                                             expected_point_dim - point_feats.shape[1], 
                                             device=point_feats.device)
                        point_feats = torch.cat([point_feats, padding], dim=1)
                    else:
                        point_feats = point_feats[:, :expected_point_dim]

                # 移动到设备
                conv_feats = conv_feats.to(self.device)
                point_feats = point_feats.to(self.device)
                targets = targets.to(self.device)
                is_zero_mask = is_zero_mask.to(self.device)
                if raw_fused_swe is not None:
                    raw_fused_swe = raw_fused_swe.to(self.device)

                # ============ 2. 前向传播（适配残差注入、门控、普通模型） ============
                if hasattr(self.model, 'correction_mlp'):
                    outputs, _ = self.model(conv_feats, point_feats, raw_fused_swe)
                elif hasattr(self.model, 'pretrained_model') and hasattr(self.model, 'fine_tune_net'):
                    outputs, _, _, _ = self.model(conv_feats, point_feats)
                else:
                    outputs = self.model(conv_feats, point_feats)

                # SWE_NONNEGATIVE_PHYSICAL_CONSTRAINT_V1
                # raw输出用于计算训练目标，物理输出用于验证指标与选模。
                outputs_raw_for_loss = outputs
                outputs = torch.clamp(
                    outputs_raw_for_loss,
                    min=0.0,
                )

                if batch_idx == 0:
                    raw_negative_count = int(
                        (outputs_raw_for_loss.reshape(-1) < 0)
                        .sum()
                        .item()
                    )
                    print(
                        f"  非负约束: raw负值数={raw_negative_count}, "
                        f"physical最小值={outputs.min().item():.6f}"
                    )

                # ============ 调试：第一个 batch 的模型输出 ============
                if batch_idx == 0:
                    print(f"  模型输出前5个值: {outputs[:5]}")
                    print(f"  模型输出是否有 NaN: {torch.isnan(outputs).any()}")

                # SWE_NONNEGATIVE_PHYSICAL_CONSTRAINT_V1
                # 验证阶段不读取target零值掩膜修改预测。
                # ============ 4. 损失计算 ============
                if is_fine_tune:
                    loss, _ = self._compute_simplified_nse_logcosh_loss(
                        outputs_raw_for_loss.reshape(-1),
                        targets.reshape(-1),
                        source_flag=None,
                    )
                else:
                    # 预训练验证同样必须 flatten，避免 [B,1] vs [B] 广播成 [B,B]
                    loss = F.smooth_l1_loss(
                        outputs.reshape(-1),
                        targets.reshape(-1),
                        beta=0.02,
                        reduction="mean"
                    )

                total_loss += loss.item()
                batch_count += 1

                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_is_zero.extend(is_zero_mask.cpu().numpy().flatten())

        # ============ 计算指标 ============
        if batch_count == 0 or len(all_predictions) == 0:
            print("⚠ 验证集没有有效数据")
            return {
                "loss": float('nan'),
                "rmse": 0,
                "mae": 0,
                "correlation": 0,
                "r2": 0,
                "n_samples": 0
            }

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_is_zero = np.array(all_is_zero)

        valid_mask = ~np.isnan(all_predictions) & ~np.isnan(all_targets)
        all_predictions = all_predictions[valid_mask]
        all_targets = all_targets[valid_mask]
        all_is_zero = all_is_zero[valid_mask]

        avg_loss = total_loss / batch_count

        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        mae = np.mean(np.abs(all_predictions - all_targets))

        if len(all_targets) > 1:
            try:
                correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
            except:
                correlation = 0
        else:
            correlation = 0

        # [METRIC-COMPAT]
        # 历史代码中字段名仍叫 r2。
        # 但这里的计算是：
        #     1 - SSE / SST
        # 在水文评价语境下更接近 NSE (Nash-Sutcliffe Efficiency)。
        #
        # [DANGER]
        #   不要在论文图、表、标题中把这个字段标为 R²。
        #   主展示指标统一使用：
        #       r, RMSE, MAE, Bias
        #
        # [COMPAT]
        #   暂时保留 metrics["r2"] 是为了兼容旧 checkpoint / old summary JSON。
        #   新增展示层应优先读取 metrics["correlation"]。
        if len(all_targets) > 1:
            ss_res = np.sum((all_predictions - all_targets) ** 2)
            ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
            if ss_tot > 0:
                r2 = 1 - (ss_res / ss_tot)
            else:
                r2 = 0
        else:
            r2 = 0

        # FINETUNE_HIGH_VALUE_METRICS_V1
        finetune_extra_metrics = {}
        if is_fine_tune:
            finetune_extra_metrics = self._compute_finetune_selection_metrics(
                all_predictions,
                all_targets,
            )

        # ============ 微调专用：详细分析 ============
        if is_fine_tune:
            print(f"\n  【{validation_name}分析】:")
            print(f"    损失: {avg_loss:.6f}")
            print(f"    RMSE: {rmse:.6f}")
            print(f"    MAE:  {mae:.6f}")
            print(f"    相关系数: {correlation:.4f}")
            print(f"    全集合CCC: {finetune_extra_metrics['ccc']:.4f}")
            print(f"    NSE:   {r2:.4f}")
            print(f"    RMSE(mm): {finetune_extra_metrics['rmse_mm']:.2f}")
            print(
                f"    obs>=50 RMSE(mm): "
                f"{finetune_extra_metrics['rmse_ge50_mm']:.2f} "
                f"(N={finetune_extra_metrics['n_ge50']})"
            )
            print(
                f"    obs>=80 Bias(mm): "
                f"{finetune_extra_metrics['bias_ge80_mm']:.2f} "
                f"(N={finetune_extra_metrics['n_ge80']})"
            )
            print(f"    回归斜率: {finetune_extra_metrics['slope']:.4f}")
            print(
                f"    std(pred)/std(obs): "
                f"{finetune_extra_metrics['std_ratio']:.4f}"
            )
            print(
                f"    checkpoint selection score: "
                f"{finetune_extra_metrics['selection_score']:.6f}"
            )

            zero_count = np.sum(all_is_zero == 0)
            pos_count = len(all_is_zero) - zero_count
            print(f"\n    【约束效果】验证集:")
            print(f"      target=0样本数: {zero_count} ({zero_count/len(all_is_zero)*100:.1f}%)")
            print(f"      target>0样本数: {pos_count} ({pos_count/len(all_is_zero)*100:.1f}%)")

            if zero_count > 0:
                zero_mask = (all_is_zero == 0)
                zero_predictions = all_predictions[zero_mask]
                zero_abs_mean = np.mean(np.abs(zero_predictions))
                zero_std = np.std(zero_predictions)
                print(f"      target=0样本预测统计:")
                print(f"        绝对均值: {zero_abs_mean:.6f}")
                print(f"        标准差: {zero_std:.6f}")
                if zero_abs_mean < 0.001:
                    print(f"        ✅ 约束效果: 优秀")
                elif zero_abs_mean < 0.01:
                    print(f"        ✓ 约束效果: 良好")
                elif zero_abs_mean < 0.05:
                    print(f"        ⚠ 约束效果: 一般")
                else:
                    print(f"        ❌ 约束效果: 需要改进")

            unique_preds = np.unique(np.round(all_predictions, 3))
            print(f"\n    【预测分布】:")
            print(f"      预测值范围: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
            print(f"      预测唯一值数量: {len(unique_preds)}")
            print(f"      预测均值: {all_predictions.mean():.4f} ± {all_predictions.std():.4f}")
            print(f"      目标均值: {all_targets.mean():.4f} ± {all_targets.std():.4f}")

            if len(unique_preds) < 10 and len(all_targets) > 50:
                print(f"      ⚠ 警告: 预测值种类过少 ({len(unique_preds)}种)")
                print(f"        前10个唯一值: {unique_preds[:10]}")

        if not is_fine_tune:
            print(f"\n  【预训练验证分布】:")
            print(f"    预测值范围: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
            print(f"    预测均值: {all_predictions.mean():.6f} ± {all_predictions.std():.6f}")
            print(f"    目标均值: {all_targets.mean():.6f} ± {all_targets.std():.6f}")
            print(f"    target=0样本数: {np.sum(all_is_zero == 0)} ({np.sum(all_is_zero == 0)/len(all_is_zero)*100:.2f}%)")

        metrics = {
            "loss": avg_loss,
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "r2": r2,
            "n_samples": len(all_predictions),
        }
        if is_fine_tune:
            metrics.update(finetune_extra_metrics)

        return metrics

    def verify_all_splits_independence(self):
        """验证训练集、验证集、测试集三者之间的站点独立性"""
        print("\n" + "="*70)
        print("🔍 验证训练集、验证集、测试集站点独立性")
        print("="*70)

        def extract_stations_from_loader(loader, name):
            """从 DataLoader 中提取站点ID"""
            stations = set()
            if loader is None:
                print(f"  {name}: 无数据")
                return stations

            try:
                dataset = loader.dataset
                # 处理 Subset 包装
                if hasattr(dataset, 'dataset'):
                    base_ds = dataset.dataset
                    indices = dataset.indices
                else:
                    base_ds = dataset
                    indices = range(len(dataset))

                # 获取站点数据集
                if hasattr(base_ds, 'station_dataset'):
                    station_ds = base_ds.station_dataset
                elif hasattr(base_ds, 'meta_index'):
                    station_ds = base_ds
                else:
                    print(f"  {name}: 无法获取站点数据集")
                    return stations

                # 提取站点ID
                for idx in indices:
                    if idx < len(station_ds.meta_index):
                        station_id = station_ds.meta_index[idx]['station_id']
                        # 处理可能的逗号分隔（多个站点共享同一像素）
                        if ',' in str(station_id):
                            for sid in str(station_id).split(','):
                                stations.add(sid.strip())
                        else:
                            stations.add(str(station_id))

                print(f"  {name}: {len(stations)} 个站点, {len(indices)} 个样本")

            except Exception as e:
                print(f"  {name}: 提取失败 - {e}")

            return stations

        # 提取三个数据集的站点
        train_stations = extract_stations_from_loader(self.train_loader, "训练集")
        val_stations = extract_stations_from_loader(self.val_loader, "验证集")
        test_stations = extract_stations_from_loader(self.test_loader, "测试集")

        print("\n" + "-"*50)
        print("📊 站点统计:")
        print(f"  训练集站点数: {len(train_stations)}")
        print(f"  验证集站点数: {len(val_stations)}")
        print(f"  测试集站点数: {len(test_stations)}")
        print(f"  总唯一站点数: {len(train_stations | val_stations | test_stations)}")

        # 检查两两重叠
        print("\n" + "-"*50)
        print("🔍 站点重叠检查:")

        # 训练集 vs 验证集
        train_val_overlap = train_stations & val_stations
        if train_val_overlap:
            print(f"  ⚠️ 训练集 ∩ 验证集: {len(train_val_overlap)} 个重叠站点")
            print(f"     重叠站点: {list(train_val_overlap)[:10]}")
            if len(train_val_overlap) > 10:
                print(f"     ... 还有 {len(train_val_overlap)-10} 个")
        else:
            print(f"  ✅ 训练集 ∩ 验证集: 无重叠")

        # 训练集 vs 测试集
        train_test_overlap = train_stations & test_stations
        if train_test_overlap:
            print(f"  ⚠️ 训练集 ∩ 测试集: {len(train_test_overlap)} 个重叠站点")
            print(f"     重叠站点: {list(train_test_overlap)[:10]}")
            if len(train_test_overlap) > 10:
                print(f"     ... 还有 {len(train_test_overlap)-10} 个")
        else:
            print(f"  ✅ 训练集 ∩ 测试集: 无重叠")

        # 验证集 vs 测试集
        val_test_overlap = val_stations & test_stations
        if val_test_overlap:
            print(f"  ⚠️ 验证集 ∩ 测试集: {len(val_test_overlap)} 个重叠站点")
            print(f"     重叠站点: {list(val_test_overlap)[:10]}")
            if len(val_test_overlap) > 10:
                print(f"     ... 还有 {len(val_test_overlap)-10} 个")
        else:
            print(f"  ✅ 验证集 ∩ 测试集: 无重叠")

        # 三者的共同重叠
        three_way_overlap = train_stations & val_stations & test_stations
        if three_way_overlap:
            print(f"  ⚠️ 训练集 ∩ 验证集 ∩ 测试集: {len(three_way_overlap)} 个重叠站点")

        # 打印测试集完整列表
        print("\n" + "-"*50)
        print(f"📋 测试集站点列表（共 {len(test_stations)} 个）:")
        for i, sid in enumerate(sorted(test_stations)[:30]):
            print(f"  {i+1}. {sid}")
        if len(test_stations) > 30:
            print(f"  ... 还有 {len(test_stations)-30} 个")

        # 打印验证集完整列表
        print("\n" + "-"*50)
        print(f"📋 验证集站点列表（共 {len(val_stations)} 个）:")
        for i, sid in enumerate(sorted(val_stations)[:30]):
            print(f"  {i+1}. {sid}")
        if len(val_stations) > 30:
            print(f"  ... 还有 {len(val_stations)-30} 个")

        # 总结
        print("\n" + "="*70)
        is_fully_independent = (len(train_val_overlap) == 0 and 
                               len(train_test_overlap) == 0 and 
                               len(val_test_overlap) == 0)

        if is_fully_independent:
            print("✅ 结论: 训练集、验证集、测试集三者站点完全独立！")
            print("   评估结果可信，反映了模型的真实泛化能力。")
        else:
            print("⚠️ 结论: 存在站点重叠，评估结果可能过于乐观！")
            print("   请检查数据划分逻辑。")
        print("="*70)

        return {
            'train_stations': train_stations,
            'val_stations': val_stations,
            'test_stations': test_stations,
            'train_val_overlap': train_val_overlap,
            'train_test_overlap': train_test_overlap,
            'val_test_overlap': val_test_overlap,
            'is_fully_independent': is_fully_independent
        }


    def inspect_all_samples(self, num_samples_per_loader=100):
        """
        在训练开始前检查训练集、验证集、测试集的样本质量和多样性
        方便后续对照散点图
        """
        print("\n" + "="*90)
        print("🔍【所有数据集样本质量检查】- 对照散点图用")
        print("="*90)

        # 检查三个数据集
        datasets_to_check = {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader if hasattr(self, 'test_loader') else None
        }

        all_results = {}

        for dataset_name, loader in datasets_to_check.items():
            if loader is None:
                print(f"\n⚠ {dataset_name} 数据集为空，跳过")
                continue

            print(f"\n{'='*60}")
            print(f"📊 【{dataset_name.upper()}集】样本分析")
            print(f"{'='*60}")

            # 收集样本
            all_conv = []
            all_point = []
            all_targets = []
            all_is_zero = []

            for i, batch in enumerate(loader):
                if len(all_targets) >= num_samples_per_loader:
                    break

                if len(batch) == 4:
                    conv, point, targets, is_zero = batch
                else:
                    # 自动将多余的变量装入 _ 列表中
                    conv, point, targets, *_ = batch
                    # 🔥 修复：is_zero 应该是 target > 0 时为 1，target == 0 时为 0
                    is_zero = (targets > 0).float()

                all_conv.append(conv.numpy())
                all_point.append(point.numpy())
                all_targets.append(targets.numpy())
                all_is_zero.append(is_zero.numpy())

            # 合并样本
            if not all_targets:
                print(f"  ⚠ 没有收集到样本")
                continue

            conv_samples = np.concatenate(all_conv, axis=0)[:num_samples_per_loader]
            point_samples = np.concatenate(all_point, axis=0)[:num_samples_per_loader]
            target_samples = np.concatenate(all_targets, axis=0)[:num_samples_per_loader]
            is_zero_samples = np.concatenate(all_is_zero, axis=0)[:num_samples_per_loader]

            print(f"\n📈 分析了 {len(target_samples)} 个样本")

            # ============ 1. 目标值分析 ============
            print("\n【1. 目标值分布】")
            unique_targets = np.unique(target_samples.round(4))
            print(f"  范围: [{target_samples.min():.4f}, {target_samples.max():.4f}]")
            print(f"  均值: {target_samples.mean():.4f} ± {target_samples.std():.4f}")
            print(f"  中位数: {np.median(target_samples):.4f}")
            print(f"  唯一值数量: {len(unique_targets)}")
            print(f"  唯一值比例: {len(unique_targets)/len(target_samples)*100:.1f}%")

            # 🔥 修复：target=0 和 target>0 的统计
            zero_count = np.sum(target_samples == 0)
            pos_count = len(target_samples) - zero_count
            print(f"  target=0样本: {zero_count} ({zero_count/len(target_samples)*100:.1f}%)")
            print(f"  target>0样本: {pos_count} ({pos_count/len(target_samples)*100:.1f}%)")

            if pos_count > 0:
                pos_targets = target_samples[target_samples > 0]
                print(f"  target>0范围: [{pos_targets.min():.4f}, {pos_targets.max():.4f}]")
                print(f"  target>0均值: {pos_targets.mean():.4f} ± {pos_targets.std():.4f}")

            # ============ 2. 目标值分布详情 ============
            print("\n【2. 目标值分布详情】")
            bins = np.linspace(0, 1, 11)
            hist, bin_edges = np.histogram(target_samples, bins=bins)
            for i in range(len(hist)):
                if hist[i] > 0:
                    print(f"  {bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}: {hist[i]}个 ({hist[i]/len(target_samples)*100:.1f}%)")

            # ============ 3. 卷积特征分析 ============
            print("\n【3. 卷积特征】")
            n_samples, C, H, W = conv_samples.shape
            print(f"  形状: {n_samples}×{C}×{H}×{W}")

            # 动态生成卷积特征名称
            conv_names = []
            conv_names.extend(['chelsa_sfxwind', 'lst', 'rh', 'pr'])
            conv_names.append('clamday')
            dem_count = len(self.dem_data) if hasattr(self, 'dem_data') and self.dem_data else 0
            for i in range(dem_count):
                conv_names.append(f'dem_band_{i+1}')
            while len(conv_names) < C:
                conv_names.append(f'feature_{len(conv_names)+1}')

            for c in range(C):
                name = conv_names[c] if c < len(conv_names) else f'feature_{c+1}'
                channel_data = conv_samples[:, c, :, :].reshape(n_samples, -1)
                channel_mean = channel_data.mean()
                channel_std = channel_data.std()
                channel_min = channel_data.min()
                channel_max = channel_data.max()

                sample_means = channel_data.mean(axis=1)
                sample_stds = channel_data.std(axis=1)

                nan_count = np.isnan(channel_data).sum()
                nan_ratio = nan_count / channel_data.size

                print(f"\n  通道 {c+1} ({name}):")
                print(f"    全局范围: [{channel_min:.4f}, {channel_max:.4f}]")
                print(f"    全局均值: {channel_mean:.4f} ± {channel_std:.4f}")
                print(f"    NaN比例: {nan_ratio:.1%}")
                print(f"    样本均值范围: [{sample_means.min():.4f}, {sample_means.max():.4f}]")
                print(f"    样本均值标准差: {sample_means.std():.4f} (样本间差异)")
                print(f"    样本内标准差范围: [{sample_stds.min():.4f}, {sample_stds.max():.4f}]")

                constant_samples = np.sum(sample_stds < 1e-6)
                if constant_samples > 0:
                    print(f"    ⚠ 常数样本: {constant_samples}个 ({constant_samples/n_samples*100:.1f}%)")

            # ============ 4. 点特征分析 ============
            print("\n【4. 点特征】")
            print(f"  形状: {point_samples.shape}")

            point_names = [
                'LS1', 'LS2', 'LS3', 'LS4', 'LS5', 'LS6',
                'S1_VV', 'S1_VH', 'SMAP_TBV', 'SMAP_TBH',
                'lon_norm', 'lat_norm', 'doy_norm'
            ]

            while len(point_names) < point_samples.shape[1]:
                point_names.append(f'feature_{len(point_names)+1}')

            for f in range(point_samples.shape[1]):
                feature_data = point_samples[:, f]
                unique_values = np.unique(feature_data.round(4))
                nan_count = np.isnan(feature_data).sum()

                print(f"\n  特征 {f+1} ({point_names[f]}):")
                print(f"    范围: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
                print(f"    均值: {feature_data.mean():.4f} ± {feature_data.std():.4f}")
                print(f"    NaN数量: {nan_count} ({nan_count/len(feature_data)*100:.1f}%)")
                print(f"    唯一值数量: {len(unique_values)}")

                if len(unique_values) < 5:
                    print(f"    ⚠ 可能为常数或离散值: {unique_values[:10]}")

            # ============ 5. 微调关键特征分析（微波特征） ============
            print("\n【5. 微调关键特征分析（微波特征）】")

            if point_samples.shape[1] >= 10:
                microwave_features = point_samples[:, 6:10]
                feature_names = ['S1_VV', 'S1_VH', 'SMAP_TBV', 'SMAP_TBH']

                for i, feat_name in enumerate(feature_names):
                    if i < microwave_features.shape[1]:
                        feat_data = microwave_features[:, i]

                        print(f"\n  {feat_name}:")
                        print(f"    范围: [{feat_data.min():.2f}, {feat_data.max():.2f}]")
                        print(f"    均值: {feat_data.mean():.2f} ± {feat_data.std():.2f}")

                        if feat_name.startswith('S1'):
                            reasonable_min, reasonable_max = -30, 10
                            default_val = 0.0
                        else:
                            reasonable_min, reasonable_max = 150, 350
                            default_val = 250.0

                        outliers = np.sum((feat_data < reasonable_min) | (feat_data > reasonable_max))
                        if outliers > 0:
                            print(f"    ⚠ 异常值: {outliers}个 ({outliers/len(feat_data)*100:.1f}%)")
                            print(f"      合理范围应为 [{reasonable_min}, {reasonable_max}]")

                        default_count = np.sum(np.abs(feat_data - default_val) < 1e-4)
                        if default_count > 0:
                            default_ratio = default_count / len(feat_data)
                            print(f"    ⚠ 默认值({default_val})比例: {default_ratio:.1%}")

            # ============ 6. LS特征分析 ============
            print("\n【6. LS特征分析】")

            if point_samples.shape[1] >= 6:
                ls_features = point_samples[:, 0:6]
                unique_combinations = len(np.unique(ls_features, axis=0))
                print(f"  LS特征唯一组合数: {unique_combinations} / {n_samples} ({unique_combinations/n_samples*100:.1f}%)")

                for i in range(6):
                    if i < ls_features.shape[1]:
                        feat_data = ls_features[:, i]
                        unique_vals = np.unique(feat_data.round(4))
                        print(f"  LS波段{i+1}: 唯一值 {len(unique_vals)}个, 范围 [{feat_data.min():.4f}, {feat_data.max():.4f}]")

            # ============ 7. 时空特征分析 ============
            print("\n【7. 时空特征分析】")

            if point_samples.shape[1] >= 13:
                lon_feat = point_samples[:, 10]
                lat_feat = point_samples[:, 11]
                doy_feat = point_samples[:, 12]

                print(f"  经度归一化: 范围 [{lon_feat.min():.4f}, {lon_feat.max():.4f}], 唯一值 {len(np.unique(lon_feat.round(4)))}个")
                print(f"  纬度归一化: 范围 [{lat_feat.min():.4f}, {lat_feat.max():.4f}], 唯一值 {len(np.unique(lat_feat.round(4)))}个")
                print(f"  DOY归一化:  范围 [{doy_feat.min():.4f}, {doy_feat.max():.4f}], 唯一值 {len(np.unique(doy_feat.round(4)))}个")

            # ============ 8. 样本相似性分析 ============
            print("\n【8. 样本相似性分析】")

            try:
                if len(point_samples) > 1:
                    if np.isnan(point_samples).any():
                        print(f"  ⚠ 点特征包含NaN，使用非NaN样本计算相似性")
                        valid_samples_mask = ~np.isnan(point_samples).any(axis=1)
                        valid_point_samples = point_samples[valid_samples_mask]

                        if len(valid_point_samples) > 1:
                            print(f"  有效样本数: {len(valid_point_samples)}/{len(point_samples)}")
                            distances = euclidean_distances(valid_point_samples)
                            mask = ~np.eye(len(valid_point_samples), dtype=bool)
                            pairwise_distances = distances[mask]

                            print(f"  点特征样本间距离:")
                            print(f"    最小距离: {pairwise_distances.min():.4f}")
                            print(f"    最大距离: {pairwise_distances.max():.4f}")
                            print(f"    平均距离: {pairwise_distances.mean():.4f} ± {pairwise_distances.std():.4f}")

                            similar_pairs = np.sum(pairwise_distances < 1e-4)
                            if similar_pairs > 0:
                                print(f"    ⚠ 极度相似样本对: {similar_pairs}对")
                        else:
                            print(f"  有效样本不足，无法计算相似性")
                    else:
                        distances = euclidean_distances(point_samples)
                        mask = ~np.eye(len(point_samples), dtype=bool)
                        pairwise_distances = distances[mask]

                        print(f"  点特征样本间距离:")
                        print(f"    最小距离: {pairwise_distances.min():.4f}")
                        print(f"    最大距离: {pairwise_distances.max():.4f}")
                        print(f"    平均距离: {pairwise_distances.mean():.4f} ± {pairwise_distances.std():.4f}")

                        similar_pairs = np.sum(pairwise_distances < 1e-4)
                        if similar_pairs > 0:
                            print(f"    ⚠ 极度相似样本对: {similar_pairs}对")
                else:
                    print(f"  样本数不足，无法计算相似性")

            except ImportError:
                print("  sklearn not available, skipping similarity analysis")
            except Exception as e:
                print(f"  相似性分析失败: {e}")

            # ============ 9. 特征-目标相关性 ============
            print("\n【9. 特征-目标相关性】")

            correlations = []
            for f in range(point_samples.shape[1]):
                try:
                    corr = np.corrcoef(point_samples[:, f], target_samples)[0, 1]
                    if not np.isnan(corr) and not np.isinf(corr):
                        correlations.append(corr)
                        if abs(corr) > 0.2:
                            print(f"  特征 {f+1} ({point_names[f]}): 与目标值相关性 = {corr:.4f}")
                except:
                    pass

            if correlations:
                print(f"  平均绝对相关性: {np.mean(np.abs(correlations)):.4f}")
                print(f"  最大相关性: {np.max(np.abs(correlations)):.4f}")

            # ============ 10. 保存结果 ============
            all_results[dataset_name] = {
                'targets': target_samples,
                'is_zero': is_zero_samples,
                'n_samples': len(target_samples),
                'target_stats': {
                    'min': float(target_samples.min()),
                    'max': float(target_samples.max()),
                    'mean': float(target_samples.mean()),
                    'std': float(target_samples.std()),
                    'zero_ratio': float(zero_count/len(target_samples)),
                    'unique_ratio': float(len(unique_targets)/len(target_samples))
                },
                'conv_stats': {
                    'shape': conv_samples.shape,
                    'nan_ratios': [float(np.isnan(conv_samples[:, c, :, :]).sum() / conv_samples[:, c, :, :].size) 
                                  for c in range(C)]
                },
                'point_stats': {
                    'shape': point_samples.shape,
                    'means': [float(point_samples[:, f].mean()) for f in range(point_samples.shape[1])],
                    'stds': [float(point_samples[:, f].std()) for f in range(point_samples.shape[1])]
                }
            }

            print(f"\n【{dataset_name.upper()}集总结】")
            print(f"  样本数: {len(target_samples)}")
            print(f"  目标值范围: [{target_samples.min():.4f}, {target_samples.max():.4f}]")
            print(f"  目标值多样性: {len(unique_targets)}/{len(target_samples)} = {len(unique_targets)/len(target_samples)*100:.1f}%")
            print(f"  target=0比例: {zero_count/len(target_samples)*100:.1f}%")

        # 打印三个数据集的对比
        print("\n" + "="*90)
        print("📊 【三个数据集对比】- 对照散点图用")
        print("="*90)

        if len(all_results) >= 1:
            print("\n数据集      样本数   目标值范围          均值±标准差    唯一值%    zero%")
            print("-" * 70)
            for name, results in all_results.items():
                stats = results['target_stats']
                print(f"{name.upper():8s} {results['n_samples']:4d}    "
                      f"[{stats['min']:.3f}, {stats['max']:.3f}]  "
                      f"{stats['mean']:.3f}±{stats['std']:.3f}  "
                      f"{stats['unique_ratio']*100:5.1f}%    "
                      f"{stats['zero_ratio']*100:5.1f}%")

        # 绘制对比直方图
        try:
            n_datasets = len(all_results)
            if n_datasets > 0:
                fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4))
                if n_datasets == 1:
                    axes = [axes]

                for i, (name, results) in enumerate(all_results.items()):
                    ax = axes[i]
                    ax.hist(results['targets'], bins=20, alpha=0.7, edgecolor='black', color='steelblue')
                    ax.set_xlabel('目标值 (归一化)', fontsize=12)
                    ax.set_ylabel('频数', fontsize=12)
                    ax.set_title(f'{name.upper()}集分布', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)

                    stats = results['target_stats']
                    text = (f"N={results['n_samples']}\n"
                           f"Range=[{stats['min']:.2f},{stats['max']:.2f}]\n"
                           f"Mean={stats['mean']:.2f}±{stats['std']:.2f}\n"
                           f"Zero={stats['zero_ratio']*100:.0f}%")
                    ax.text(0.95, 0.95, text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10)

                plt.tight_layout()
                plot_path = self.save_dir / "dataset_targets_comparison.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\n✅ 数据集对比图已保存: {plot_path}")

        except Exception as e:
            print(f"绘图失败: {e}")

        print("\n" + "="*90)
        print("✅ 样本检查完成，可以对照后续的散点图了！")
        print("="*90)

        return all_results
    
    
    def _capture_overfit_resume_random_state(self):
        """保存可恢复的随机状态；旧checkpoint没有这些字段时仍保持兼容。"""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "train_loader_generator": None,
        }
        loader_generator = getattr(
            getattr(self, "train_loader", None),
            "generator",
            None,
        )
        if loader_generator is not None:
            state["train_loader_generator"] = loader_generator.get_state()
        return state

    def _restore_overfit_resume_random_state(self, state):
        """恢复checkpoint中存在的随机状态，返回是否完整恢复。"""
        if not isinstance(state, dict):
            return False

        required = ("python", "numpy", "torch_cpu")
        if not all(key in state for key in required):
            return False

        def _as_cpu_byte_tensor(value):
            """
            torch.load(map_location=self.device)会把checkpoint中的RNG状态
            一并映射到CUDA；PyTorch的set_rng_state接口仍要求CPU
            ByteTensor。这里同时兼容Tensor、NumPy数组和普通列表。
            """
            if torch.is_tensor(value):
                return (
                    value.detach()
                    .to(device="cpu", dtype=torch.uint8)
                    .contiguous()
                )
            if isinstance(value, np.ndarray):
                return (
                    torch.as_tensor(value, dtype=torch.uint8)
                    .clone()
                    .contiguous()
                )
            return torch.tensor(
                value,
                dtype=torch.uint8,
                device="cpu",
            ).contiguous()

        fully_restored = True
        try:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(
                _as_cpu_byte_tensor(state["torch_cpu"])
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            print(
                "  ⚠ Python/NumPy/CPU RNG状态恢复失败，"
                f"将继续续训: {exc}"
            )
            fully_restored = False

        cuda_state = state.get("torch_cuda")
        if torch.cuda.is_available() and cuda_state is not None:
            try:
                if torch.is_tensor(cuda_state) or isinstance(
                    cuda_state,
                    np.ndarray,
                ):
                    cuda_states = [cuda_state]
                else:
                    cuda_states = list(cuda_state)

                device_count = torch.cuda.device_count()
                if len(cuda_states) != device_count:
                    print(
                        "  ⚠ checkpoint CUDA RNG设备数"
                        f"({len(cuda_states)})与当前设备数"
                        f"({device_count})不同；恢复可匹配设备后继续"
                    )
                    fully_restored = False

                for device_index, rng_state in enumerate(
                    cuda_states[:device_count]
                ):
                    torch.cuda.set_rng_state(
                        _as_cpu_byte_tensor(rng_state),
                        device=device_index,
                    )
            except (TypeError, ValueError, RuntimeError) as exc:
                print(
                    "  ⚠ CUDA RNG状态恢复失败，模型与AdamW仍将"
                    f"正常续训: {exc}"
                )
                fully_restored = False

        loader_generator_state = state.get("train_loader_generator")
        loader_generator = getattr(
            getattr(self, "train_loader", None),
            "generator",
            None,
        )
        if (
            loader_generator is not None
            and loader_generator_state is not None
        ):
            try:
                loader_generator.set_state(
                    _as_cpu_byte_tensor(loader_generator_state)
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                print(
                    "  ⚠ DataLoader generator状态恢复失败，"
                    f"将继续续训: {exc}"
                )
                fully_restored = False
        return fully_restored

    @staticmethod
    def _resume_values_match(current, previous):
        if isinstance(current, (float, np.floating)) or isinstance(
            previous,
            (float, np.floating),
        ):
            try:
                return bool(
                    np.isclose(
                        float(current),
                        float(previous),
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
            except (TypeError, ValueError):
                return False
        return current == previous

    def _load_train_only_overfit_resume(self):
        """
        恢复训练集过拟合诊断。

        fine_tune_epochs表示续训完成后的总epoch数，而不是新增epoch数。
        """
        resume_value = self.config.get(
            "overfit_resume_checkpoint",
        )
        if not resume_value:
            return None
        if not bool(self.config.get("train_only_overfit", False)):
            raise ValueError(
                "overfit_resume_checkpoint仅允许用于train_only_overfit"
            )
        if str(self.config.get("freeze_strategy", "")).lower() != "fusion_ft":
            raise ValueError(
                "训练集过拟合断点续训仅支持fusion_ft"
            )

        resume_path = Path(resume_value).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"过拟合续训checkpoint不存在: {resume_path}"
            )

        checkpoint = torch.load(
            resume_path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"续训checkpoint不是完整训练checkpoint: {resume_path}"
            )

        previous_config = checkpoint.get("config", {})
        if not bool(previous_config.get("train_only_overfit", False)):
            raise RuntimeError(
                "续训checkpoint不是train_only_overfit产生的文件"
            )

        compatibility_keys = (
            "freeze_strategy",
            "batch_size",
            "seed",
            "fusion_ft_progressive_unfreeze",
            "fusion_ft_head_lr",
            "fusion_ft_transformer_lr",
            "fusion_ft_warmup_epochs",
            "fusion_ft_unfreeze_interval",
            "fusion_ft_max_unfrozen_layers",
            "fusion_ft_ccc_weight",
            "fusion_ft_variance_weight",
            "normalization_config_path",
            "station_data_path",
        )
        mismatches = []
        for key in compatibility_keys:
            if key == "fusion_ft_max_unfrozen_layers":
                # 旧checkpoint没有该字段时等价于0（完整Fusion解冻）。
                current_value = int(self.config.get(key, 0))
                previous_value = int(previous_config.get(key, 0))
                if current_value != previous_value:
                    mismatches.append(
                        f"{key}: current={current_value!r}, "
                        f"checkpoint={previous_value!r}"
                    )
                continue
            if key not in previous_config or key not in self.config:
                continue
            current_value = self.config.get(key)
            previous_value = previous_config.get(key)
            if not self._resume_values_match(
                current_value,
                previous_value,
            ):
                mismatches.append(
                    f"{key}: current={current_value!r}, "
                    f"checkpoint={previous_value!r}"
                )
        if mismatches:
            raise RuntimeError(
                "续训配置与checkpoint不一致，拒绝混合实验:\n  "
                + "\n  ".join(mismatches)
            )

        completed_epochs = int(checkpoint.get("epoch", -1)) + 1
        target_epochs = int(
            self.config.get(
                "fine_tune_epochs",
                self.config.get("epochs", 0),
            )
        )
        if completed_epochs <= 0:
            raise RuntimeError(
                f"续训checkpoint的epoch无效: {completed_epochs}"
            )
        if target_epochs <= completed_epochs:
            raise ValueError(
                "FINE_TUNE_EPOCHS表示续训后的总轮数，必须大于checkpoint"
                f"已完成轮数: target={target_epochs}, "
                f"completed={completed_epochs}"
            )

        model_state = checkpoint.get("model_state_dict")
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if model_state is None or optimizer_state is None:
            raise RuntimeError(
                "续训checkpoint缺少model_state_dict或"
                "optimizer_state_dict"
            )

        self.model.load_state_dict(model_state, strict=True)
        try:
            self.optimizer.load_state_dict(optimizer_state)
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "AdamW参数组与续训checkpoint不一致；"
                "请确认仍使用相同Fusion FT配置"
            ) from exc

        transformer_lr_override = self.config.get(
            "overfit_resume_transformer_lr"
        )
        transformer_override_rows = []
        if transformer_lr_override is not None:
            transformer_lr_override = float(
                transformer_lr_override
            )
            if (
                not np.isfinite(transformer_lr_override)
                or transformer_lr_override <= 0
            ):
                raise ValueError(
                    "overfit_resume_transformer_lr必须是有限正数"
                )

            warmup_epochs = int(
                self.config.get(
                    "fusion_ft_warmup_epochs",
                    5,
                )
            )
            for group in self.optimizer.param_groups:
                group_name = str(
                    group.get("group_name", "")
                )
                if not group_name.startswith("transformer"):
                    continue

                old_lr = float(group.get("lr", 0.0))
                unlock_epoch = int(
                    group.get("unlock_epoch", 0)
                )
                is_active = completed_epochs > unlock_epoch
                group["target_lr"] = transformer_lr_override
                group["lr"] = (
                    transformer_lr_override
                    if is_active
                    else 0.0
                )
                group["activation_lr_initialized"] = bool(
                    is_active
                    and completed_epochs >= warmup_epochs
                )
                transformer_override_rows.append(
                    (group_name, old_lr, float(group["lr"]))
                )

            if not transformer_override_rows:
                raise RuntimeError(
                    "续训学习率覆盖未找到Transformer参数组"
                )
            self.config[
                "_effective_overfit_transformer_lr"
            ] = transformer_lr_override

        self.train_history = list(
            checkpoint.get("train_history", [])
        )
        self.val_history = list(
            checkpoint.get("val_history", [])
        )
        self.lr_history = list(
            checkpoint.get("lr_history", [])
        )
        self.fine_tune_history = list(
            checkpoint.get("fine_tune_history", [])
        )

        metrics_history = checkpoint.get("val_history_metrics")
        history_source = "checkpoint"
        if not isinstance(metrics_history, list):
            metrics_history = None

        # 兼容本次已经运行的旧版checkpoint：逐轮指标CSV在checkpoint同目录。
        if metrics_history is None:
            csv_candidates = []
            current_parent = resume_path.parent
            for _ in range(4):
                csv_candidates.append(
                    current_parent
                    / "train_only_overfit_epoch_metrics.csv"
                )
                current_parent = current_parent.parent

            metrics_csv = next(
                (
                    path for path in csv_candidates
                    if path.is_file()
                ),
                None,
            )
            if metrics_csv is not None:
                metrics_frame = pd.read_csv(metrics_csv)
                metrics_history = (
                    metrics_frame
                    .replace({np.nan: None})
                    .to_dict(orient="records")
                )
                history_source = str(metrics_csv)
            else:
                metrics_history = []
                history_source = "unavailable"

        if len(metrics_history) > completed_epochs:
            metrics_history = metrics_history[:completed_epochs]
        self.val_history_metrics = list(metrics_history)

        valid_history_rows = []
        for index, metrics in enumerate(self.val_history_metrics):
            try:
                rmse_mm = float(metrics.get("rmse_mm", float("nan")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(rmse_mm):
                epoch_value = metrics.get("epoch", index + 1)
                try:
                    epoch_index = int(epoch_value) - 1
                except (TypeError, ValueError):
                    epoch_index = index
                valid_history_rows.append(
                    (rmse_mm, epoch_index, metrics)
                )

        if valid_history_rows:
            best_rmse_mm, best_epoch, best_metrics = min(
                valid_history_rows,
                key=lambda row: row[0],
            )
        else:
            final_metrics = checkpoint.get("metrics", {})
            best_rmse_mm = float(
                final_metrics.get("rmse_mm", float("inf"))
            )
            best_epoch = completed_epochs - 1
            best_metrics = final_metrics

        previous_best_path = (
            resume_path.parent
            / "best_train_overfit_model.pth"
        )
        current_best_path = (
            self.save_dir
            / "best_train_overfit_model.pth"
        )
        if previous_best_path.is_file():
            if previous_best_path.resolve() != current_best_path.resolve():
                shutil.copy2(previous_best_path, current_best_path)
            print(
                "  ✅ 已继承历史最低RMSE checkpoint: "
                f"{current_best_path}"
            )
        elif np.isfinite(best_rmse_mm):
            print(
                "  ⚠ 未找到历史best_train_overfit_model.pth；"
                "续训仅继承最佳指标，出现新低后会创建新best"
            )

        random_state_restored = (
            self._restore_overfit_resume_random_state(
                checkpoint.get("random_state")
            )
        )
        if not random_state_restored:
            print(
                "  ⚠ 该checkpoint由旧版脚本生成，未包含完整随机状态；"
                "模型与AdamW会真实续接，但Epoch 101的shuffle序列"
                "不能保证与一次性跑200轮逐位相同"
            )

        self.config["_overfit_resume_completed_epochs"] = (
            completed_epochs
        )
        self.config["_overfit_resume_source"] = str(resume_path)

        print("\n" + "=" * 70)
        print("♻ 训练集过拟合断点恢复完成")
        print(f"  checkpoint:       {resume_path}")
        print(f"  已完成轮数:       {completed_epochs}")
        print(f"  目标总轮数:       {target_epochs}")
        print(f"  本次新增轮数:     {target_epochs - completed_epochs}")
        print("  模型参数:         已恢复")
        print("  AdamW状态:        已恢复")
        print("  warmup/渐进解冻:  不重新执行")
        if transformer_override_rows:
            print(
                "  Transformer LR:   仅本次诊断覆盖为 "
                f"{transformer_lr_override:.2e}"
            )
            for group_name, old_lr, new_lr in transformer_override_rows:
                print(
                    f"    {group_name:<20s} "
                    f"{old_lr:.2e} -> {new_lr:.2e}"
                )
        resumed_lr_text = ", ".join(
            f"{group.get('group_name', f'group{index}')}="
            f"{float(group['lr']):.2e}"
            for index, group in enumerate(
                self.optimizer.param_groups,
                start=1,
            )
        )
        print(f"  恢复后学习率:     {resumed_lr_text}")
        print(f"  历史逐轮指标:     {history_source}")
        print(
            f"  历史最低RMSE:     {best_rmse_mm:.2f} mm "
            f"(Epoch {best_epoch + 1})"
        )
        print(
            "  随机状态:         "
            + (
                "完整恢复"
                if random_state_restored
                else "旧版兼容模式"
            )
        )
        print("=" * 70)

        return {
            "start_epoch": completed_epochs,
            "best_global_rmse_mm": best_rmse_mm,
            "best_global_rmse_epoch": best_epoch,
            "best_metrics": best_metrics,
            "random_state_restored": random_state_restored,
            "resume_path": str(resume_path),
        }

    def train(self, fine_tune_mode=False, is_cv_sub_run=False, is_full_refit=False):
        """主训练循环 - 完整版本

        is_full_refit: 全量 refit 模式——无验证集、无早停、
                       使用 CosineAnnealingLR 或训练loss调度、
                       保存 final_checkpoint_epoch{N}.pth 和 final_model.pth
        """
        is_train_only_overfit = bool(
            self.config.get("train_only_overfit", False)
        )

        if fine_tune_mode and not is_cv_sub_run:
            cv_mode = self.config.get('cv_mode', 'station_cv')

            if cv_mode == 'station_full_cv':
                return self.run_station_full_cv()
            elif cv_mode == 'station_cv':
                return self.run_cv_workflow_by_station()  # ← 改成这个
            else:
                # standard 模式，继续执行后面的训练代码
                pass

        # ============ 下面的代码保持不变，用于 standard 模式 ============

        # ============ 交叉验证时减少打印 ============
        verbose = not is_cv_sub_run

        if verbose:
            print("\n" + "=" * 70)

        if fine_tune_mode:
            if verbose:
                print("🚀 开始微调训练")
                epochs = self.config.get("fine_tune_epochs", self.config["epochs"])
                print(f"\n📋 微调配置:")
                print(f"  预训练模型: {self.config.get('pretrained_model', '未指定')}")
                print(f"  冻结主干: {self.config.get('freeze_backbone', True)}")
                print(f"  微调轮次: {epochs}")
                print(f"  微调学习率: {self.config.get('fine_tune_lr', 1e-5):.2e}")
                print(f"  批次大小: {self.config['batch_size']}")
                print(f"  早停耐心: {self.config['patience']}")
                print(f"  课程学习: {self.config.get('use_curriculum', False)}")
        else:
            if verbose:
                print("🚀 开始训练")
                epochs = self.config["epochs"]
                print(f"\n📋 训练配置:")
                print(f"  训练轮次: {epochs}")
                print(f"  学习率: {self.config['learning_rate']:.2e}")
                print(f"  批次大小: {self.config['batch_size']}")
                print(f"  权重衰减: {self.config['weight_decay']:.2e}")

        if verbose:
            print("=" * 70)

        # ============ 样本统计（交叉验证时简化） ============
        if fine_tune_mode and hasattr(self, 'train_loader') and verbose:
            print("\n📊 【混合模式样本统计】")
            if hasattr(self.train_loader.dataset, 'dataset'):
                dataset = self.train_loader.dataset.dataset
            else:
                dataset = self.train_loader.dataset

            if hasattr(dataset, 'station_indices') and hasattr(dataset, 'pretrain_indices'):
                print(f"  总训练样本: {len(self.train_loader.dataset)}")
                print(f"    ├─ 站点样本: {len(dataset.station_indices)}")
                print(f"    └─ 预训练样本: {len(dataset.pretrain_indices)}")
                if hasattr(self, 'val_loader'):
                    print(f"\n  验证集样本: {len(self.val_loader.dataset)} (仅站点)")
                if hasattr(self, 'test_loader'):
                    print(f"  测试集样本: {len(self.test_loader.dataset)} (仅站点)")
            else:
                print(f"\n  训练集样本: {len(self.train_loader.dataset)}")
                print(f"  验证集样本: {len(self.val_loader.dataset)}")
                if hasattr(self, 'test_loader'):
                    print(f"  测试集样本: {len(self.test_loader.dataset)}")

        # ============ 微调时检查样本（交叉验证时跳过） ============
        if fine_tune_mode and verbose and not is_cv_sub_run:
            print("\n【微调前样本检查】")
            self.inspect_all_samples(num_samples_per_loader=200)
            if hasattr(self, 'train_loader') and hasattr(self, 'val_loader') and hasattr(self, 'test_loader'):
                self.verify_all_splits_independence()

        if self.train_loader is None or (not is_full_refit and self.val_loader is None):
            print("❌ 请先加载数据!")
            return
        if self.model is None:
            print("❌ 请先构建模型!")
            return

        # ============ 训练前模型验证和修复（交叉验证时简化） ============
        if verbose:
            print("\n" + "=" * 70)
            print("🔧【训练前模型验证和修复】")
            print("=" * 70)

        self.model = self.model.to(self.device)
        if verbose:
            print(f"  模型已移动到设备: {self.device}")

        if verbose:
            print("\n1. 检查模型参数...")
        param_issues = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                if verbose:
                    print(f"  ⚠️ 参数 {name} 包含NaN，正在重置...")
                param_issues = True
                if 'weight' in name:
                    if 'conv' in name or 'fc' in name or 'linear' in name:
                        nn.init.kaiming_normal_(param)
                    else:
                        nn.init.normal_(param, mean=0, std=0.02)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            if torch.isinf(param).any():
                if verbose:
                    print(f"  ⚠️ 参数 {name} 包含Inf，正在重置...")
                param_issues = True
                param.data = torch.where(torch.isinf(param), torch.zeros_like(param), param)
        if verbose and not param_issues:
            print("  ✓ 所有参数正常")

        if verbose:
            print("\n2. 检查BatchNorm层...")
        bn_issues = False
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.running_mean is not None:
                    if torch.isnan(module.running_mean).any() or torch.isinf(module.running_mean).any():
                        if verbose:
                            print(f"  ⚠️ BatchNorm {name} 的running_mean包含非法值，重置...")
                        bn_issues = True
                        module.running_mean = torch.zeros_like(module.running_mean)
                    if torch.isnan(module.running_var).any() or torch.isinf(module.running_var).any():
                        if verbose:
                            print(f"  ⚠️ BatchNorm {name} 的running_var包含非法值，重置...")
                        bn_issues = True
                        module.running_var = torch.ones_like(module.running_var)
        if verbose and not bn_issues:
            print("  ✓ 所有BatchNorm层正常")

        # 微调前验证检查（交叉验证时简化）
        if fine_tune_mode and verbose and not is_cv_sub_run:
            print("\n" + "=" * 70)
            print("🔍 微调前验证检查")
            print("=" * 70)
            if self.config.get("freeze_backbone", True):
                print("\n1. 冻结状态验证:")
                self._verify_freezing_detailed()
            else:
                print("\n1. 冻结状态: 全部解冻")
            print("\n2. 优化器验证:")
            self._verify_optimizer_with_gradients()
            print("\n3. 前向传播测试:")
            self._test_forward_pass()
            print("\n4. 数据统计:")
            self._log_data_statistics()
            print("=" * 70)
            print("✅ 验证检查完成，开始训练...")
            print("=" * 70)
        elif fine_tune_mode and is_cv_sub_run:
            # 交叉验证时快速检查
            if verbose:
                print("  🔍 快速检查通过")

        # ============ 课程学习初始化（交叉验证时跳过） ============
        use_curriculum = self.config.get('use_curriculum', False)

        if use_curriculum and fine_tune_mode and not is_cv_sub_run:
            # ... 课程学习初始化代码保持不变 ...
            pass  # 为了简洁，这里省略，实际使用时保留原有代码

        # 3. 训练前热身检查（交叉验证时简化）
        if verbose:
            print(f"\n🔥 训练前热身检查...")

        conv_batch = None
        point_batch = None
        target_batch = None

        try:
            first_batch = next(iter(self.train_loader))
            if len(first_batch) == 6:
                conv_batch, point_batch, target_batch, _, _, _ = first_batch
            elif len(first_batch) == 5:
                conv_batch, point_batch, target_batch, _, _ = first_batch
            elif len(first_batch) == 4:
                conv_batch, point_batch, target_batch, _ = first_batch
            elif len(first_batch) == 3:
                conv_batch, point_batch, target_batch = first_batch
            else:
                conv_batch, point_batch, target_batch = first_batch[0], first_batch[1], first_batch[2]

            if verbose:
                print(f"  第一个批次形状:")
                print(f"    卷积特征: {conv_batch.shape}")
                print(f"    点特征: {point_batch.shape}")
                print(f"    目标值: {target_batch.shape}")
                print(f"  数据范围:")
                print(f"    卷积特征: [{conv_batch.min():.4f}, {conv_batch.max():.4f}]")
                print(f"    点特征: [{point_batch.min():.4f}, {point_batch.max():.4f}]")
                print(f"    目标值: [{target_batch.min():.4f}, {target_batch.max():.4f}]")
        except StopIteration:
            print(f"  ⚠ 训练集为空!")
            return
        except Exception as e:
            if verbose:
                print(f"  ⚠ 批次检查失败: {e}")

        if verbose:
            print("\n【用第一个批次测试前向传播】")
        if conv_batch is not None and point_batch is not None:
            try:
                test_conv = conv_batch[:2].to(self.device)
                test_point = point_batch[:2].to(self.device)
                test_target = target_batch[:2].to(self.device) if target_batch is not None else None
                with torch.no_grad():
                    test_output = self.model(test_conv, test_point)
                    if verbose:
                        print(f"  ✓ 前向传播成功")
                        print(f"    输出shape: {test_output.shape}")
                        print(f"    输出范围: [{test_output.min():.4f}, {test_output.max():.4f}]")
                        print(f"    输出均值: {test_output.mean():.4f} ± {test_output.std():.4f}")
                        if test_target is not None:
                            test_loss = self.criterion(test_output.reshape(-1), test_target.reshape(-1))
                            print(f"    损失值: {test_loss.item():.6f}")
            except Exception as e:
                if verbose:
                    print(f"  ❌ GPU前向传播测试失败: {e}")

        overfit_resume_state = None
        if is_train_only_overfit:
            overfit_resume_state = (
                self._load_train_only_overfit_resume()
            )
        start_epoch = (
            int(overfit_resume_state["start_epoch"])
            if overfit_resume_state is not None
            else 0
        )

        # 4. 训练准备
        best_val_loss = float("inf")
        best_val_r2 = -float("inf")
        best_selection_score = float("inf")
        # DUAL_CHECKPOINT_SELECTION_DIAG_V1
        # 复合评分与全局RMSE分别追踪，互不替代。
        best_global_rmse_mm = float("inf")
        best_global_rmse_epoch = None
        patience_counter = 0
        best_epoch = 0
        best_val_r = -float("inf")
        collapse_rollbacks = 0
        collapse_streak = 0
        frozen_baseline_metrics = None
        best_admissible_rmse_mm = float("inf")
        best_admissible_selection_score = float("inf")
        admissible_improvement_found = False
        best_checkpoint_assessment = None

        if overfit_resume_state is not None:
            best_global_rmse_mm = float(
                overfit_resume_state["best_global_rmse_mm"]
            )
            best_global_rmse_epoch = int(
                overfit_resume_state["best_global_rmse_epoch"]
            )
            best_epoch = best_global_rmse_epoch
            previous_best_metrics = (
                overfit_resume_state.get("best_metrics") or {}
            )
            best_val_loss = float(
                previous_best_metrics.get("loss", float("inf"))
            )
            best_val_r2 = float(
                previous_best_metrics.get("r2", -float("inf"))
            )
            best_val_r = float(
                previous_best_metrics.get("correlation", -float("inf"))
            )

        # FINETUNE_EPOCH0_BASELINE_V1
        # 训练前先把“未微调预训练模型”作为候选checkpoint。
        # 如果某个策略的所有训练epoch都退化，最终自动退回该基线，
        # 而不是保存一个常数预测模型。
        if (
            fine_tune_mode
            and not is_full_refit
            and overfit_resume_state is None
        ):
            print("\n🧭 评估 epoch-0 微调基线（尚未更新参数）...")
            baseline_metrics = self.validate(is_fine_tune=True)
            self._run_train_vs_inner_audit(
                baseline_metrics,
                "Epoch 0 Frozen",
            )
            frozen_baseline_metrics = baseline_metrics.copy()
            baseline_score = float(
                baseline_metrics.get("selection_score", float("inf"))
            )
            baseline_collapsed = bool(
                baseline_metrics.get("is_collapsed", False)
            )

            baseline_global_rmse_mm = float(
                baseline_metrics.get(
                    "rmse_mm",
                    float("inf"),
                )
            )

            if (
                np.isfinite(baseline_global_rmse_mm)
                and not baseline_collapsed
            ):
                best_global_rmse_mm = baseline_global_rmse_mm
                best_global_rmse_epoch = -1

                self.save_checkpoint(
                    "best_fine_tuned_global_rmse.pth",
                    -1,
                    baseline_metrics,
                )
                self.save_checkpoint(
                    "best_fine_tuned_frozen_baseline.pth",
                    -1,
                    baseline_metrics,
                )
                if is_train_only_overfit:
                    self.save_checkpoint(
                        "best_train_overfit_model.pth",
                        -1,
                        baseline_metrics,
                    )

                print(
                    "  ✅ epoch-0全局RMSE基线已保存: "
                    f"RMSE={best_global_rmse_mm:.2f} mm"
                )

            if np.isfinite(baseline_score) and not baseline_collapsed:
                best_admissible_rmse_mm = baseline_global_rmse_mm
                best_selection_score = baseline_score
                best_val_loss = float(baseline_metrics["loss"])
                best_val_r2 = float(
                    baseline_metrics.get("r2", -float("inf"))
                )
                best_val_r = float(
                    baseline_metrics.get("correlation", 0.0)
                )
                best_epoch = -1
                self.save_checkpoint(
                    "best_fine_tuned_model.pth",
                    -1,
                    baseline_metrics,
                )
                print(
                    f"  ✅ epoch-0 Frozen安全回退已保存: "
                    f"RMSE={best_admissible_rmse_mm:.2f} mm, "
                    f"r={best_val_r:.4f}, "
                    f"slope={baseline_metrics.get('slope', float('nan')):.3f}, "
                    f"std_ratio={baseline_metrics.get('std_ratio', float('nan')):.3f}"
                )
            else:
                raise RuntimeError(
                    "epoch-0 Frozen基线无效或塌缩，无法执行"
                    "Frozen-relative checkpoint选择；"
                    "请检查预训练checkpoint与特征归一化"
                )

        start_time = datetime.now()

        if verbose:
            print(f"\n⏱ 训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if fine_tune_mode and verbose and not is_cv_sub_run:
            self.gradient_history = []
            self.weight_change_history = []
            initial_weights = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    initial_weights[name] = param.data.clone().cpu()
            print(f"  【梯度监控】保存了 {len(initial_weights)} 个可训练参数的初始权重")

        # 5. 训练循环
        epochs = self.config.get("fine_tune_epochs", self.config["epochs"]) if fine_tune_mode else self.config["epochs"]

        if start_epoch >= epochs:
            raise ValueError(
                f"续训起始epoch={start_epoch}不小于目标总轮数={epochs}"
            )

        for epoch in range(start_epoch, epochs):
            mode = "微调" if fine_tune_mode else "训练"
            epoch_start_time = datetime.now()

            if verbose:
                print(f"\n" + "=" * 60)
                print(f"📊 {mode} Epoch {epoch + 1}/{epochs}")
                print(f"⏰ 开始时间: {epoch_start_time.strftime('%H:%M:%S')}")
                print("-" * 60)
            elif (epoch + 1) % 5 == 0:  # 交叉验证时每5轮打印一次
                print(f"  Epoch {epoch + 1}/{epochs}")

            if fine_tune_mode:
                self._apply_fusion_ft_progressive_stage(epoch)

            train_loss = self.train_epoch(epoch, is_fine_tune=fine_tune_mode)

            # ============ Mixed mode 诊断日志 ============
            if self.config.get("mixed_mode", False) and fine_tune_mode:
                self._mixed_epoch_diagnostic(epoch)

            if fine_tune_mode:
                self.fine_tune_history.append(train_loss)
            else:
                self.train_history.append(train_loss)

            # [SPEED] val_every 控制验证频率。
            # 大样本预训练中，每个 epoch 都完整验证会明显拖慢训练。
            #
            # [CONTRACT]
            #   need_validate=True 时：
            #       - 执行 validate()
            #       - 允许 scheduler.step()
            #       - 允许更新 best model
            #       - 允许 early stopping 计数
            #
            #   need_validate=False 时：
            #       - 不执行 validate()
            #       - 不更新 scheduler
            #       - 不保存 best model
            #       - 不增加 patience_counter
            #
            # [DANGER]
            #   如果跳过验证的 epoch 仍然参与 best / early stopping，
            #   会用旧 val_loss 或假 val_metrics 覆盖真正最佳模型。
            # ============ 验证（按 val_every 跳间隔） ============
            val_every = int(self.config.get("val_every", 1))

            if is_full_refit:
                # 全量 refit 模式：无验证集，不执行 validate()
                need_validate = False
                val_metrics = {
                    "loss": train_loss,     # 用训练 loss 占位（仅用于日志）
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "correlation": float("nan"),
                    "r2": float("nan"),
                    "n_samples": 0,
                }
            else:
                need_validate = (
                    (epoch + 1) % val_every == 0
                    or epoch == epochs - 1
                    or epoch >= epochs - 10
                )

                if need_validate:
                    val_metrics = self.validate(is_fine_tune=fine_tune_mode)
                    if fine_tune_mode and not is_train_only_overfit:
                        self._run_train_vs_inner_audit(
                            val_metrics,
                            f"Epoch {epoch + 1}",
                        )
                else:
                    prev = self.val_history_metrics[-1] if hasattr(self, "val_history_metrics") and self.val_history_metrics else {}
                    val_metrics = {
                        "loss": self.val_history[-1] if self.val_history else float("inf"),
                        "rmse": float("nan"),
                        "mae": float("nan"),
                        "correlation": prev.get("correlation", 0.0),
                        "r2": prev.get("r2", 0.0),
                        "n_samples": 0,
                    }

            current_checkpoint_assessment = None
            if (
                fine_tune_mode
                and not is_full_refit
                and not is_train_only_overfit
                and need_validate
                and frozen_baseline_metrics is not None
            ):
                current_checkpoint_assessment = (
                    self._assess_frozen_relative_checkpoint(
                        val_metrics,
                        frozen_baseline_metrics,
                    )
                )
                val_metrics["checkpoint_admissible"] = bool(
                    current_checkpoint_assessment["admissible"]
                )
                val_metrics["checkpoint_control_score"] = float(
                    current_checkpoint_assessment["control_score"]
                )
                val_metrics["checkpoint_violations"] = list(
                    current_checkpoint_assessment["violations"]
                )

            if not is_full_refit:
                self.val_history.append(val_metrics["loss"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history.append(current_lr)

            # 学习率调度
            # warmup+cosine已经在每个optimizer step后更新，
            # 这里不能再次step。
            if (
                self.scheduler is not None
                and not getattr(
                    self,
                    "scheduler_step_per_batch",
                    False,
                )
            ):
                if self._fusion_ft_schedule_is_protected(epoch):
                    # 逐层解冻期间由显式warmup控制LR；此时Plateau不应把
                    # 尚未完整训练的阶段误判为性能停滞。
                    pass
                elif is_full_refit:
                    _is_cosine = isinstance(
                        self.scheduler,
                        optim.lr_scheduler.CosineAnnealingLR,
                    )
                    if _is_cosine:
                        self.scheduler.step()
                    else:
                        self.scheduler.step(train_loss)
                elif need_validate:
                    scheduler_metric = (
                        val_metrics.get(
                            "checkpoint_control_score",
                            val_metrics.get(
                                "selection_score",
                                val_metrics["loss"],
                            ),
                        )
                        if fine_tune_mode
                        else val_metrics["loss"]
                    )
                    self.scheduler.step(scheduler_metric)

            epoch_end_time = datetime.now()
            epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()

            if verbose:
                print(f"\n📈 Epoch {epoch + 1} 结果:")
                print(f"  {mode}损失:  {train_loss:.6f}")
                if is_full_refit:
                    print("  验证指标:  全量refit模式，无验证集")
                else:
                    print(f"  验证损失:  {val_metrics['loss']:.6f}")
                    print(f"  验证RMSE:  {val_metrics['rmse']:.6f}")
                    print(f"  验证MAE:   {val_metrics['mae']:.6f}")
                    print(f"  验证相关系数: {val_metrics['correlation']:.4f}")
                    if fine_tune_mode:
                        print(f"  验证CCC:      {val_metrics.get('ccc', float('nan')):.4f}")
                        print(f"  obs>=50 RMSE: {val_metrics.get('rmse_ge50_mm', float('nan')):.2f} mm")
                        print(f"  obs>=80 Bias: {val_metrics.get('bias_ge80_mm', float('nan')):.2f} mm")
                        print(f"  回归斜率:    {val_metrics.get('slope', float('nan')):.4f}")
                        print(f"  Std Ratio:   {val_metrics.get('std_ratio', float('nan')):.4f}")
                        print(f"  选模评分:     {val_metrics.get('selection_score', float('nan')):.6f}")
                print(f"  学习率:    {current_lr:.2e}")
                print(f"  耗时:      {epoch_duration:.1f}秒")
            elif (epoch + 1) % 5 == 0:
                # 交叉验证时简化输出
                print(
                    f"    损失: {train_loss:.6f}, "
                    f"Val Loss: {val_metrics['loss']:.6f}, "
                    f"r: {val_metrics.get('correlation', 0):.4f}"
                )

            if epoch >= 1 and verbose:
                if len(self.train_history) >= 2:
                    train_loss_change = self.train_history[-2] - train_loss
                    if train_loss_change > 0:
                        print(f"  📉 训练损失下降: {train_loss_change:.6f}")
                    elif train_loss_change < 0:
                        print(f"  📈 训练损失上升: {-train_loss_change:.6f}")
                if (not is_full_refit) and len(self.val_history) >= 2:
                    val_loss_change = self.val_history[-2] - val_metrics["loss"]
                    if val_loss_change > 0:
                        print(f"  📉 验证损失下降: {val_loss_change:.6f}")
                    elif val_loss_change < 0:
                        print(f"  📈 验证损失上升: {-val_loss_change:.6f}")

            if not is_full_refit:
                if not hasattr(self, 'val_history_metrics'):
                    self.val_history_metrics = []
                self.val_history_metrics.append(val_metrics.copy())

            # 只有真正验证过的 epoch 才允许更新 best model / early stopping
            # 全量 refit 模式：跳过 best model 和 early stopping
            if is_full_refit:
                # 全量 refit：每 10 轮保存一次 checkpoint，最后 5 轮每轮保存
                save_checkpoint_now = (
                    (epoch + 1) % 10 == 0
                    or epoch >= epochs - 5
                )
                if save_checkpoint_now:
                    ckpt_name = f"final_checkpoint_epoch{epoch + 1}.pth"
                    ckpt_metrics = {"loss": train_loss, "lr": current_lr}
                    self.save_checkpoint(ckpt_name, epoch, ckpt_metrics)
                    if verbose:
                        print(f"  💾 保存refit检查点: {ckpt_name}")

                # 追踪训练 loss 最低的 epoch（不称其为 "best"，仅记录）
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_epoch = epoch
                    if verbose:
                        print(f"  📉 训练损失新低: {best_val_loss:.6f} (epoch {epoch+1})")

            elif need_validate:
                if is_train_only_overfit:
                    current_train_rmse_mm = float(
                        val_metrics.get("rmse_mm", float("inf"))
                    )
                    train_rmse_is_better = bool(
                        np.isfinite(current_train_rmse_mm)
                        and current_train_rmse_mm < best_global_rmse_mm
                    )
                    if train_rmse_is_better:
                        best_global_rmse_mm = current_train_rmse_mm
                        best_global_rmse_epoch = int(epoch)
                        best_val_loss = float(val_metrics["loss"])
                        best_val_r2 = float(
                            val_metrics.get("r2", -float("inf"))
                        )
                        best_val_r = float(
                            val_metrics.get("correlation", 0.0)
                        )
                        best_epoch = int(epoch)
                        self.save_checkpoint(
                            "best_train_overfit_model.pth",
                            epoch,
                            val_metrics,
                        )
                        print(
                            "\n💾 保存训练集过拟合最佳模型: "
                            "best_train_overfit_model.pth "
                            f"(Epoch {epoch + 1}, "
                            f"RMSE={current_train_rmse_mm:.2f} mm, "
                            f"R={best_val_r:.4f}, "
                            f"CCC={val_metrics.get('ccc', float('nan')):.4f}, "
                            f"slope={val_metrics.get('slope', float('nan')):.4f}, "
                            f"std_ratio={val_metrics.get('std_ratio', float('nan')):.4f})"
                        )
                    patience_counter = 0
                elif fine_tune_mode:
                    # FINETUNE_COLLAPSE_GUARD_V1
                    is_collapsed = bool(
                        val_metrics.get("is_collapsed", False)
                    )
                    checkpoint_assessment = (
                        current_checkpoint_assessment
                        or self._assess_frozen_relative_checkpoint(
                            val_metrics,
                            frozen_baseline_metrics,
                        )
                    )

                    # 仍保存“无约束最低RMSE”用于诊断，但它不再能成为
                    # cv_fold_XX_best_model.pth，除非同时通过Frozen硬门槛。
                    current_global_rmse_mm = float(
                        val_metrics.get(
                            "rmse_mm",
                            float("inf"),
                        )
                    )

                    global_rmse_is_better = bool(
                        np.isfinite(current_global_rmse_mm)
                        and not is_collapsed
                        and current_global_rmse_mm
                        < best_global_rmse_mm
                    )

                    if global_rmse_is_better:
                        best_global_rmse_mm = (
                            current_global_rmse_mm
                        )
                        best_global_rmse_epoch = int(epoch)

                        self.save_checkpoint(
                            "best_fine_tuned_global_rmse.pth",
                            epoch,
                            val_metrics,
                        )

                        print(
                            "\n💾 保存最低全局RMSE模型: "
                            "best_fine_tuned_global_rmse.pth "
                            f"(Epoch {epoch + 1}, "
                            f"RMSE={best_global_rmse_mm:.2f} mm, "
                            f"R={val_metrics.get('correlation', float('nan')):.4f}, "
                            f"RMSE50={val_metrics.get('rmse_ge50_mm', float('nan')):.2f} mm, "
                            f"Bias80={val_metrics.get('bias_ge80_mm', float('nan')):.2f} mm, "
                            f"slope={val_metrics.get('slope', float('nan')):.3f})"
                        )

                    is_admissible = bool(
                        checkpoint_assessment["admissible"]
                        and not is_collapsed
                    )
                    current_selection_score = float(
                        val_metrics.get(
                            "selection_score",
                            float("inf"),
                        )
                    )
                    is_better = bool(
                        is_admissible
                        and np.isfinite(current_selection_score)
                        and current_selection_score
                        < best_admissible_selection_score
                    )

                    if is_collapsed:
                        collapse_streak += 1
                        patience_counter += 1
                        print(
                            f"\n🚨 Epoch {epoch + 1} 检测到近常数预测，"
                            "禁止成为best checkpoint"
                        )
                        print(
                            f"   pred_std={val_metrics.get('pred_std_mm', float('nan')):.4f} mm, "
                            f"std_ratio={val_metrics.get('std_ratio', float('nan')):.5f}, "
                            f"slope={val_metrics.get('slope', float('nan')):.5f}, "
                            f"r={val_metrics.get('correlation', float('nan')):.5f}"
                        )

                        max_rollbacks = int(
                            self.config.get("collapse_max_rollbacks", 1)
                        )
                        best_path = self.save_dir / "best_fine_tuned_model.pth"

                        if collapse_rollbacks < max_rollbacks and best_path.exists():
                            checkpoint = torch.load(
                                best_path,
                                map_location=self.device,
                                weights_only=False,
                            )
                            state_dict = (
                                checkpoint.get("model_state_dict")
                                if isinstance(checkpoint, dict)
                                else checkpoint
                            )
                            if state_dict is not None:
                                self.model.load_state_dict(state_dict, strict=True)
                                # Adam动量可能保留导致再次冲坏，必须清空。
                                self.optimizer.state.clear()
                                for group in self.optimizer.param_groups:
                                    group["lr"] = max(
                                        float(group["lr"]) * 0.2,
                                        1e-7,
                                    )
                                collapse_rollbacks += 1
                                print(
                                    f"   ↩ 已回滚到当前best并清空优化器状态；"
                                    f"学习率×0.2（第{collapse_rollbacks}/{max_rollbacks}次）"
                                )

                        collapse_limit = int(
                            self.config.get("collapse_early_stop_patience", 2)
                        )
                        if collapse_streak >= collapse_limit:
                            patience_counter = max(
                                patience_counter,
                                int(self.config.get("patience", 10)),
                            )
                            print(
                                f"   🛑 连续{collapse_streak}个验证epoch塌缩，"
                                "本折将在当前epoch后早停并恢复历史best"
                            )
                    elif is_better:
                        collapse_streak = 0
                        best_admissible_rmse_mm = current_global_rmse_mm
                        best_admissible_selection_score = (
                            current_selection_score
                        )
                        best_selection_score = current_selection_score
                        best_val_loss = float(val_metrics["loss"])
                        best_val_r2 = float(val_metrics.get("r2", -float("inf")))
                        best_val_r = float(val_metrics.get("correlation", 0.0))
                        best_epoch = epoch
                        patience_counter = 0
                        admissible_improvement_found = True
                        best_checkpoint_assessment = checkpoint_assessment
                        model_name = "best_fine_tuned_model.pth"
                        self.save_checkpoint(model_name, epoch, val_metrics)
                        print(
                            f"\n💾 保存Frozen-relative合格模型: {model_name} "
                            f"(Epoch {epoch + 1}, "
                            f"RMSE={best_admissible_rmse_mm:.2f} mm, "
                            f"MAE={val_metrics.get('mae_mm', float('nan')):.2f} mm, "
                            f"R={val_metrics.get('correlation', float('nan')):.4f}, "
                            f"NSE={val_metrics.get('r2', float('nan')):.4f}, "
                            f"RMSE50={val_metrics.get('rmse_ge50_mm', float('nan')):.2f} mm, "
                            f"Bias80={val_metrics.get('bias_ge80_mm', float('nan')):.2f} mm, "
                            f"slope={val_metrics.get('slope', float('nan')):.3f}, "
                            f"std_ratio={val_metrics.get('std_ratio', float('nan')):.3f})"
                        )
                        print(
                            "   综合selection_score="
                            f"{best_admissible_selection_score:.6f}"
                        )
                    else:
                        collapse_streak = 0
                        patience_counter += 1
                        if verbose or (epoch + 1) % 5 == 0:
                            violations = checkpoint_assessment.get(
                                "violations",
                                [],
                            )
                            violation_text = (
                                "; ".join(violations[:4])
                                if violations
                                else (
                                    "通过门槛但综合selection_score"
                                    "未优于当前best"
                                )
                            )
                            print(
                                f"\n⏳ 连续 {patience_counter} 轮没有新的"
                                "Frozen-relative合格checkpoint\n"
                                f"   {violation_text}"
                            )
                else:
                    is_best_by_loss = val_metrics["loss"] < best_val_loss
                    is_best_by_r2 = 'r2' in val_metrics and val_metrics['r2'] > best_val_r2

                    if is_best_by_loss or is_best_by_r2:
                        if is_best_by_loss:
                            best_val_loss = val_metrics["loss"]
                            if verbose:
                                print(f"  🎉 新的最佳验证损失: {best_val_loss:.6f}")
                        if is_best_by_r2:
                            best_val_r2 = val_metrics['r2']
                            best_val_r = val_metrics.get('correlation', 0)
                            if verbose:
                                print(f"  🎉 新的最佳 r: {best_val_r:.4f}")
                        best_epoch = epoch
                        patience_counter = 0
                        model_name = "best_model.pth"
                        self.save_checkpoint(model_name, epoch, val_metrics)
                        if verbose:
                            print(f"\n💾 保存最佳{mode}模型: {model_name} (Epoch {epoch+1}, val_loss={val_metrics['loss']:.6f})")
                    else:
                        patience_counter += 1
                        if verbose:
                            print(f"\n⏳ 连续 {patience_counter} 轮未改善最佳指标")
            else:
                # 跳过验证的 epoch 不参与 best / patience
                if verbose:
                    print("  ⏭ 本 epoch 未验证，不更新 best model / early stopping")

            # 标准模式的定期 checkpoint（全量 refit 已在上面处理）
            if not is_full_refit:
                if (epoch + 1) % self.config["save_freq"] == 0 and verbose:
                    checkpoint_name = f"checkpoint_{mode}_epoch{epoch + 1}.pth" if fine_tune_mode else f"checkpoint_epoch{epoch + 1}.pth"
                    self.save_checkpoint(checkpoint_name, epoch, val_metrics)
                    print(f"  💾 保存检查点: {checkpoint_name}")

            # 课程学习难度更新（交叉验证时跳过）
            if use_curriculum and fine_tune_mode and not is_cv_sub_run:
                # ... 课程学习更新代码保持不变 ...
                pass

            if (
                fine_tune_mode
                and self._fusion_ft_schedule_is_protected(epoch)
                and collapse_streak == 0
            ):
                # Head-only/部分Transformer阶段是预声明的稳定化过程，
                # 不应消耗完整Fusion训练的early-stopping耐心。
                patience_counter = 0

            # 全量 refit 无早停；标准模式按 patience 早停。
            # Fusion FT先完成足够轮数的稳定训练，不能再在约15轮时结束。
            min_epochs_before_early_stop = 0
            if (
                fine_tune_mode
                and self._fusion_ft_progressive_enabled()
            ):
                min_epochs_before_early_stop = int(
                    self.config.get(
                        "fusion_ft_min_epochs_before_early_stop",
                        40,
                    )
                )

            early_stop_allowed = (
                epoch + 1 >= min_epochs_before_early_stop
            )
            if (
                not is_full_refit
                and not is_train_only_overfit
                and early_stop_allowed
                and patience_counter >= self.config["patience"]
            ):
                if verbose:
                    print(f"\n" + "!" * 60)
                    print(f"🛑 早停触发! 连续 {self.config['patience']} 轮验证指标未改善")
                    print(f"最佳验证损失: {best_val_loss:.6f}, 最佳NSE: {best_val_r2:.4f}, 最佳轮次: {best_epoch + 1}")
                    print("!" * 60)
                break

            if verbose:
                progress = (epoch + 1) / epochs * 100
                print(f"\n📊 进度: {progress:.1f}% ({epoch + 1}/{epochs})")

        # 6. 训练完成
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        session_epochs_ran = epoch - start_epoch + 1

        if verbose:
            print("\n" + "=" * 70)
            mode = "微调" if fine_tune_mode else "训练"
            print(f"✅ {mode}完成!")
            print("=" * 70)
            print(f"\n🎯 训练总结:")
            print(f"  总轮次: {epoch + 1}")
            if start_epoch > 0:
                print(f"  本次续训轮次: {session_epochs_ran}")
            if is_full_refit:
                print(f"  最终训练损失: {train_loss:.6f}")
                print(f"  训练损失新低: {best_val_loss:.6f} (epoch {best_epoch + 1})")
            else:
                print(f"  最佳轮次: {best_epoch + 1}")
                print(f"  最佳验证损失: {best_val_loss:.6f}")

            if not is_full_refit and hasattr(self, 'val_history_metrics') and len(self.val_history_metrics) > 0:
                r_values = [m.get('correlation', 0) for m in self.val_history_metrics]
                if any(r > 0 for r in r_values):
                    max_r = max(r_values)
                    print(f"  最佳验证 r: {max_r:.4f}")
                    avg_r = sum(r_values) / len(r_values)
                    print(f"  平均验证 r: {avg_r:.4f}")
                    print(f"  r 标准差: {np.std(r_values):.4f}")

            print(f"  总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
            print(
                f"  本次平均每轮耗时: "
                f"{total_duration/max(session_epochs_ran, 1):.1f}秒"
            )

            if len(self.train_history) > 0:
                print(f"\n📉 训练损失历史:")
                print(f"  起始损失: {self.train_history[0]:.6f}")
                print(f"  最终损失: {self.train_history[-1]:.6f}")
                print(f"  总下降: {self.train_history[0] - self.train_history[-1]:.6f}")
                if not is_full_refit and len(self.val_history) > 0:
                    print(f"\n📊 验证损失历史:")
                    print(f"  起始损失: {self.val_history[0]:.6f}")
                    print(f"  最终损失: {self.val_history[-1]:.6f}")
                    print(f"  总下降: {self.val_history[0] - self.val_history[-1]:.6f}")
        else:
            # 交叉验证时简化输出
            if fine_tune_mode and not is_train_only_overfit:
                print(
                    f"  ✅ 训练完成: best_admissible_rmse="
                    f"{best_admissible_rmse_mm:.2f} mm, "
                    f"admissible_improvement_found="
                    f"{admissible_improvement_found}, "
                    f"best_nse={best_val_r2:.4f}"
                )
            elif is_train_only_overfit:
                print(
                    "  ✅ 完整训练集过拟合测试完成: "
                    f"epochs={epoch + 1}, "
                    f"session_epochs={session_epochs_ran}, "
                    f"best_train_rmse={best_global_rmse_mm:.2f} mm, "
                    f"best_epoch={best_epoch + 1}"
                )
            else:
                print(f"  ✅ 训练完成: best_val_loss={best_val_loss:.6f}, best_r2={best_val_r2:.4f}")

        # 保存最终模型
        if is_full_refit or is_train_only_overfit:
            # 全量 refit: 保存最后一轮模型作为 final_model.pth
            model_name = (
                "train_overfit_final_model.pth"
                if is_train_only_overfit
                else "final_model.pth"
            )
            final_metrics = (
                val_metrics.copy()
                if is_train_only_overfit
                else {"loss": train_loss, "lr": current_lr}
            )
            final_metrics["lr"] = current_lr
            self.save_checkpoint(model_name, epoch, final_metrics)
            # 同时保存带 epoch 号的副本
            epoch_model_name = (
                f"train_overfit_final_epoch_{epoch + 1}.pth"
                if is_train_only_overfit
                else f"final_full_epoch_{epoch + 1}.pth"
            )
            self.save_checkpoint(epoch_model_name, epoch, final_metrics)
        else:
            model_name = "final_fine_tuned_model.pth" if fine_tune_mode else "final_model.pth"
            final_metrics = {"loss": best_val_loss}
            if hasattr(self, 'val_history_metrics'):
                final_metrics['r2'] = best_val_r2
            if fine_tune_mode:
                # PRESERVE_LAST_BEFORE_CANONICAL_V8
                # 此时 self.model 仍是最后一个实际训练 epoch 的权重。
                # 必须先保存，再恢复 Frozen-relative canonical。
                last_trained_model_path = (
                    self.save_dir / "last_trained_model.pth"
                )
                last_trained_metrics = (
                    dict(val_metrics)
                    if isinstance(val_metrics, dict)
                    else {}
                )
                last_trained_metrics.update({
                    "checkpoint_role": (
                        "last_trained_before_canonical_restore"
                    ),
                    "last_completed_epoch_zero_based": int(epoch),
                    "last_completed_epoch_one_based": int(epoch) + 1,
                    "admissible_improvement_found": bool(
                        admissible_improvement_found
                    ),
                    "selected_is_frozen_fallback": bool(
                        not admissible_improvement_found
                    ),
                })
                self.save_checkpoint(
                    "last_trained_model.pth",
                    epoch,
                    last_trained_metrics,
                )

                saved_last_checkpoint = torch.load(
                    last_trained_model_path,
                    map_location="cpu",
                    weights_only=False,
                )
                saved_last_epoch = (
                    int(saved_last_checkpoint.get("epoch", -10**9))
                    if isinstance(saved_last_checkpoint, dict)
                    else -10**9
                )
                del saved_last_checkpoint
                if saved_last_epoch != int(epoch):
                    raise RuntimeError(
                        "last_trained_model.pth 的 epoch 元数据异常: "
                        f"saved={saved_last_epoch}, expected={epoch}"
                    )

                print(
                    "  💾 [PRESERVE-LAST] 已在canonical恢复前保存"
                    "真实最后训练态: "
                    f"{last_trained_model_path} "
                    f"(epoch={epoch + 1})"
                )

                canonical_path = (
                    self.save_dir / "best_fine_tuned_model.pth"
                )
                if not canonical_path.exists():
                    raise RuntimeError(
                        "训练结束但缺少Frozen-relative canonical checkpoint: "
                        f"{canonical_path}"
                    )
                canonical_checkpoint = torch.load(
                    canonical_path,
                    map_location=self.device,
                    weights_only=False,
                )
                canonical_state = (
                    canonical_checkpoint.get("model_state_dict")
                    if isinstance(canonical_checkpoint, dict)
                    else canonical_checkpoint
                )
                if canonical_state is None:
                    raise RuntimeError(
                        f"canonical checkpoint缺少model_state_dict: "
                        f"{canonical_path}"
                    )
                self.model.load_state_dict(
                    canonical_state,
                    strict=True,
                )
            self.save_checkpoint(model_name, best_epoch, final_metrics)

        if verbose:
            print(f"\n💾 保存最终模型: {model_name}")

        # 以下分析只在非CV子运行且 verbose 时执行
        if verbose and not is_cv_sub_run:
            self.save_training_history(fine_tune_mode)
            self.plot_training_curves_with_r2(fine_tune_mode)

            if fine_tune_mode:
                self.plot_gradient_monitoring(fine_tune_mode)

            # 随机森林对比实验
            if fine_tune_mode:
                print("\n" + "=" * 70)
                print("🌲 运行随机森林对比实验（与深度学习模型对比）")
                print("=" * 70)
                try:
                    if hasattr(self, 'test_loader') and self.test_loader is not None:
                        self.run_rf_baseline()
                        self.analyze_test_set_features(
                            test_loader=self.test_loader,
                            pretrained_model_path=self.config.get('pretrained_model')
                        )
                    else:
                        print("⚠ 没有测试集，使用验证集作为测试集")
                        self.test_loader = self.val_loader
                        self.run_rf_baseline()
                        self.analyze_test_set_features(
                            test_loader=self.test_loader,
                            pretrained_model_path=self.config.get('pretrained_model')
                        )
                except Exception as e:
                    print(f"⚠ 随机森林对比实验失败: {e}")

            # 预训练模式生成散点图（仅当有验证集时）
            if not fine_tune_mode and not is_full_refit:
                print("\n" + "=" * 70)
                print("📊 生成预训练验证集散点图")
                print("=" * 70)
                try:
                    best_model_path = self.save_dir / "best_model.pth"
                    if os.path.exists(best_model_path):
                        checkpoint = torch.load(best_model_path, map_location=self.device)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    predictions, targets, _ = self._make_predictions(self.val_loader)
                    if predictions is not None and len(predictions) > 0:
                        swe_min = getattr(self, 'swe_min', 0.0)
                        swe_max = getattr(self, 'swe_max', 200.0)
                        pred_denorm = predictions * (swe_max - swe_min) + swe_min
                        target_denorm = targets * (swe_max - swe_min) + swe_min
                        self.plot_density_scatter_hardcode(pred_denorm, target_denorm, is_fine_tune=False)
                except Exception as e:
                    print(f"  生成散点图失败: {e}")

            # 微调模式生成散点图
            if fine_tune_mode:
                print("\n" + "=" * 70)
                print("📊 生成微调/混合模式散点图")
                print("=" * 70)
                try:
                    print("\n1. 生成所有微调样本散点图...")
                    self.plot_all_finetune_samples()
                    print("\n2. 生成测试集密度散点图...")
                    if hasattr(self, 'test_loader') and self.test_loader is not None:
                        predictions, targets, _ = self._make_predictions(self.test_loader)
                        if predictions is not None and len(predictions) > 0:
                            swe_min = getattr(self, 'swe_min', 0.0)
                            swe_max = getattr(self, 'swe_max', 200.0)
                            pred_denorm = predictions * (swe_max - swe_min) + swe_min
                            target_denorm = targets * (swe_max - swe_min) + swe_min
                            self.plot_density_scatter_hardcode(pred_denorm, target_denorm, is_fine_tune=True)
                except Exception as e:
                    print(f"  ✗ 生成散点图失败: {e}")

            # 训练完成后诊断 PointEncoder 权重
            if fine_tune_mode and not is_cv_sub_run:
                print("\n" + "=" * 70)
                print("🔍 训练完成 - PointEncoder 权重诊断")
                print("=" * 70)
                self._diagnose_point_encoder_weights()

        return {
            "best_val_loss": best_val_loss,
            "best_val_r2": best_val_r2 if hasattr(self, 'val_history_metrics') else 0,
            "best_selection_score": best_selection_score if fine_tune_mode else None,
            "best_epoch": best_epoch,
            "total_epochs": epoch + 1,
            "admissible_improvement_found": (
                bool(admissible_improvement_found)
                if fine_tune_mode and not is_train_only_overfit
                else None
            ),
            "selected_is_frozen_fallback": (
                bool(not admissible_improvement_found)
                if fine_tune_mode and not is_train_only_overfit
                else None
            ),
            "best_checkpoint_assessment": (
                best_checkpoint_assessment
                if fine_tune_mode and not is_train_only_overfit
                else None
            ),
            "train_only_overfit": is_train_only_overfit,
        }
    
    def run_pretrain_spatial_cv(self):
        """
        支持空间独立性的预训练十折验证
        确保训练集和验证集的空间 Patch 完全不重叠
        """
        print(f"\n{'█'*80}")
        print("🌟 空间网格十折交叉验证 (Spatial Grid CV)")
        print("   特点：训练集与验证集空间隔离，避免邻近自相关")
        print(f"{'█'*80}")

        # 1. 导入必要模块
        from data_online_era5_swe import SWEDataset, build_spatial_grid_cv_indices

        # 2. 预训练配置
        self.config["use_wide_branch"] = False
        print(f"\n📋 预训练模式配置:")
        print(f"   use_wide_branch = {self.config['use_wide_branch']}")

        # 3. 构建完整数据集
        print("\n正在加载完整栅格数据集...")

        dataset_kwargs = {
            "region": self.config.get("region", "XINJIANG"),
            "year_target": self.config.get("year_target", 2016),
            "patch_size": self.config.get("patch_size", 5),
            "min_valid_pixels": self.config.get("min_valid_pixels", 100),
            "samples_per_day": self.config.get("samples_per_day", 5000),
            "clamday_threshold": self.config.get("clamday_threshold", 0.5),
            "s1_interp_method": self.config.get("s1_interp_method", "nearest"),
            "s1_max_gap_days": self.config.get("s1_max_gap_days", 7),
            "smap_interp_method": self.config.get("smap_interp_method", "nearest"),
            "smap_max_gap_days": self.config.get("smap_max_gap_days", 7),
            "sampling_mode": self.config.get("sampling_mode", "auto"),
            "use_station_guide": self.config.get("use_station_guide", False),
            "station_guide_file": self.config.get("station_guide_file"),
            "station_record_manifest_path": self.config.get("station_record_manifest_path"),
            "station_csv_dir": self.config.get("station_csv_dir", "/root/ablation"),
            "station_neighborhood": self.config.get("station_neighborhood", 3),
            "station_samples_per_day": self.config.get("station_samples_per_day", 2000),
            "station_filter_zero_target": self.config.get("station_filter_zero_target", True),
            "station_sampling_unit": self.config.get(
                "station_sampling_unit", "positions_all_dates"
            ),
            "station_record_dedup": self.config.get("station_record_dedup", "grid_date"),
            "station_date_column": self.config.get("station_date_column"),
            "external_station_glob": self.config.get("external_station_glob"),
            "external_station_exclusion_radius": self.config.get(
                "external_station_exclusion_radius", 0
            ),
            "external_station_strict": self.config.get("external_station_strict", False),
            "external_station_report_path": self.config.get("external_station_report_path"),
        }

        full_dataset = SWEDataset(**dataset_kwargs)

        # 同步维度信息
        self.config["C_conv"] = full_dataset.C_conv
        self.config["C_point"] = full_dataset.C_point

        print(f"✅ 数据集加载完成")
        print(f"   总样本数: {len(full_dataset):,}")
        print(f"   卷积特征维度: {self.config['C_conv']}")
        print(f"   点特征维度: {self.config['C_point']}")

        # 4. 获取空间网格划分
        lon_step = self.config.get("spatial_grid_lon_step", 1.0)
        lat_step = self.config.get("spatial_grid_lat_step", 1.0)
        min_samples_per_grid = self.config.get("min_samples_per_grid", 10)

        all_folds = build_spatial_grid_cv_indices(
            full_dataset,
            lon_step=lon_step,
            lat_step=lat_step,
            n_splits=10,
            seed=self.config.get("seed", 42),
            min_samples_per_grid=min_samples_per_grid
        )

        # ============ 验证划分的互斥性 ============
        print(f"\n{'='*70}")
        print("🔍 验证十折划分的互斥性")
        print(f"{'='*70}")

        all_val_indices = []
        fold_val_samples = []

        for fold_idx, (train_indices, val_indices) in enumerate(all_folds):
            fold_val_samples.append(len(val_indices))
            all_val_indices.extend(val_indices)
            print(f"  折{fold_idx+1}: 训练样本 {len(train_indices):,} | 验证样本 {len(val_indices):,}")

        # 检查样本是否重复
        counter = Counter(all_val_indices)
        duplicates = [idx for idx, count in counter.items() if count > 1]

        if len(duplicates) == 0:
            print(f"\n✅ 验证通过: 无样本重复出现在多个验证集中")
        else:
            print(f"\n❌ 验证失败: 发现 {len(duplicates)} 个样本重复")

        # 5. 存储每折结果
        all_fold_results = []
        fold_histories = []

        # 6. 十折循环
        for fold_idx, (train_indices, val_indices) in enumerate(all_folds):
            fold_num = fold_idx + 1

            print(f"\n\n{'='*60}")
            print(f"🟢 空间网格 FOLD {fold_num} / 10")
            print(f"   训练样本: {len(train_indices):,}")
            print(f"   验证样本: {len(val_indices):,}")
            print(f"{'='*60}")

            # 6.1 创建本折 DataLoader
            self.train_loader = DataLoader(
                Subset(full_dataset, train_indices),
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True,
                drop_last=True
            )

            self.val_loader = DataLoader(
                Subset(full_dataset, val_indices),
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True
            )

            # 6.2 构建模型
            print(f"\n  🏗️ 创建新模型实例...")
            success = self.build_model(
                load_pretrained=None,
                freeze_backbone=False,
                is_cv_fold=True
            )

            if not success:
                print(f"  ❌ 模型构建失败，跳过折 {fold_num}")
                continue

            # 🔥 强制关闭 Wide & Deep 并重建 head 层（修复设备问题）
            if hasattr(self.model, 'fusion_transformer'):
                # 关闭标志位
                self.model.fusion_transformer.use_wide_branch = False

                # 重新创建 head 层（去掉 Wide 分支的额外输入）
                d_model = self.model.fusion_transformer.d_model
                dropout = 0.1

                new_head = nn.Sequential(
                    nn.Linear(d_model, 128),
                    nn.Dropout(dropout),
                    nn.Linear(128, 64),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1)
                )

                # 🔥 关键：将新创建的 head 层移动到正确的设备
                new_head = new_head.to(self.device)

                # 替换原有的 head
                self.model.fusion_transformer.head = new_head

                print(f"  🔧 强制关闭 Wide & Deep 分支")
                print(f"     head 输入维度: {d_model} (原为 {d_model+1})")
                print(f"     head 已移动到设备: {self.device}")

            # 6.3 训练本折
            print(f"  🚀 开始训练...")

            best_val_loss = float('inf')
            best_val_r2 = -float('inf')
            best_val_r = -float('inf')
            best_epoch = 0
            patience_counter = 0

            fold_train_losses = []
            fold_val_losses = []
            fold_val_r2 = []
            fold_val_r = []

            epochs = self.config.get("pretrain_cv_epochs", self.config.get("epochs", 100))
            val_every = int(self.config.get("val_every", 1))

            for epoch in range(epochs):
                # 训练
                train_loss = self.train_epoch(epoch, is_fine_tune=False)
                fold_train_losses.append(train_loss)

                # 验证（按 val_every 跳间隔）
                need_validate = (
                    (epoch + 1) % val_every == 0
                    or epoch == epochs - 1
                    or epoch >= epochs - 10
                )

                if need_validate:
                    val_metrics = self.validate(self.val_loader, is_fine_tune=False)
                else:
                    val_metrics = {
                        "loss": fold_val_losses[-1] if fold_val_losses else float("inf"),
                        "rmse": float("nan"),
                        "mae": float("nan"),
                        "correlation": fold_val_r[-1] if fold_val_r else 0.0,
                        "r2": fold_val_r2[-1] if fold_val_r2 else 0.0,
                        "n_samples": 0,
                    }

                fold_val_losses.append(val_metrics["loss"])
                fold_val_r2.append(val_metrics.get("r2", 0))
                fold_val_r.append(val_metrics.get('correlation', 0))

                # 学习率调度
                current_lr = self.optimizer.param_groups[0]["lr"]

                if (
                    need_validate
                    and not getattr(
                        self,
                        "scheduler_step_per_batch",
                        False,
                    )
                ):
                    self.scheduler.step(
                        val_metrics["loss"]
                    )

                # 打印进度
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_metrics['loss']:.6f} | "
                          f"Val r: {val_metrics.get('correlation', 0):.4f} | "
                          f"LR: {current_lr:.2e}")

                # 只有真正验证过的 epoch 才允许更新 best / early stopping
                if need_validate:
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        best_val_r2 = val_metrics.get("r2", 0)
                        best_val_r = val_metrics.get('correlation', 0)
                        best_epoch = epoch
                        patience_counter = 0

                        self.save_checkpoint(
                            f"spatial_cv_fold{fold_num}_best.pth",
                            epoch,
                            val_metrics
                        )
                    else:
                        patience_counter += 1

                    # 早停
                    patience = self.config.get("patience", 10)
                    if patience_counter >= patience:
                        print(f"    🛑 早停触发！最佳 Epoch: {best_epoch + 1}")
                        break
                # else: 跳过验证的 epoch 不参与 best / patience

            # 6.4 记录本折结果
            fold_result = {
                'fold': fold_num,
                'best_epoch': best_epoch + 1,
                'best_val_loss': best_val_loss,
                'best_val_r2': best_val_r2,
                'best_val_r': best_val_r,
                'n_train_samples': len(train_indices),
                'n_val_samples': len(val_indices),
                'train_losses': fold_train_losses,
                'val_losses': fold_val_losses,
                'val_r2': fold_val_r2
            }
            all_fold_results.append(fold_result)
            fold_histories.append({
                'train_losses': fold_train_losses,
                'val_losses': fold_val_losses,
                'val_r': fold_val_r,
                'val_r2': fold_val_r2
            })

            print(f"\n  📊 [FOLD {fold_num}] 总结:")
            print(f"     最佳验证损失: {best_val_loss:.6f}")
            print(f"     最佳验证 r: {best_val_r:.4f}")
            print(f"     最佳轮次: {best_epoch + 1}")

            # ============ 🆕 每折结束后绘制本折的散点图 ============
            print(f"\n  📈 [FOLD {fold_num}] 绘制本折验证集散点图...")

            # 对验证集进行预测并绘图
            self.model.eval()
            preds, targets, _ = self._make_predictions(self.val_loader)

            if preds is not None and len(preds) > 0:
                # [DANGER] 不允许回退到 200 mm；必须继承 Dataset 的真实标签范围。
                s_min, s_max = self._require_swe_scale(context=f"pretrain_cv_fold_{fold_num}")
                preds_denorm = preds * (s_max - s_min) + s_min
                targets_denorm = targets * (s_max - s_min) + s_min

                # 绘制本折散点图
                self.plot_density_scatter_hardcode(
                    preds_denorm, targets_denorm,
                    is_fine_tune=False,  # 预训练模式
                    fold_index=fold_num
                )
            else:
                print(f"      ⚠️ 无法绘制散点图：无有效预测结果")

            # ============ 🆕 每折结束后绘制本折的训练曲线 ============
            print(f"  📈 [FOLD {fold_num}] 绘制本折训练曲线...")
            self._plot_single_fold_curve(fold_num, fold_train_losses, fold_val_losses, fold_val_r)
            # ====================================================

            # 6.5 清理显存
            del self.model
            del self.optimizer
            del self.scheduler
            try:
                del self.train_loader
                del self.val_loader
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()

        # 7. 汇总十折结果
        print(f"\n\n{'█'*80}")
        print("🏆 空间网格十折交叉验证完成！")
        print(f"{'█'*80}")

        if all_fold_results:
            best_losses = [r['best_val_loss'] for r in all_fold_results]
            best_rs = [r['best_val_r'] for r in all_fold_results]

            print(f"\n📊 十折统计结果:")
            print(f"  验证损失 (Loss): {np.mean(best_losses):.6f} ± {np.std(best_losses):.6f}")
            print(f"  验证 r:         {np.mean(best_rs):.4f} ± {np.std(best_rs):.4f}")

            # 8. 保存结果
            cv_results = {
                'cv_type': 'spatial_grid_cv',
                'grid_config': {'lon_step': lon_step, 'lat_step': lat_step},
                'n_folds': 10,
                'fold_results': all_fold_results,
                'summary': {
                    'loss_mean': float(np.mean(best_losses)),
                    'loss_std': float(np.std(best_losses)),
                    'r_mean': float(np.mean(best_rs)),
                    'r_std': float(np.std(best_rs)),
                }
            }

            save_path = self.save_dir / "spatial_cv_results.json"
            with open(save_path, 'w') as f:
                json.dump(cv_results, f, indent=2, default=str)
            print(f"\n💾 十折结果已保存: {save_path}")

            # ============ 9. 绘制汇总可视化（所有折结束后） ============
            print(f"\n📊 绘制十折汇总可视化图表...")

            # 绘制汇总训练曲线（所有折叠加）
            self._plot_spatial_cv_curves(fold_histories)

            # 绘制箱线图
            self._plot_spatial_cv_boxplot(best_losses, best_rs)
            # ====================================================

        else:
            print("\n❌ 没有成功的折次")
            cv_results = None

        return cv_results


    def _plot_single_fold_curve(self, fold_num, train_losses, val_losses, val_r):
        """绘制单折的训练曲线"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 左图：损失曲线
            ax1 = axes[0]
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
            ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='验证损失')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Fold {fold_num} - 损失曲线')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 右图：r 曲线
            ax2 = axes[1]
            ax2.plot(epochs, val_r, 'g-', linewidth=2, label='验证 r')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('r')
            ax2.set_title(f'Fold {fold_num} - r 曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.1, 1.1)

            plt.tight_layout()

            # 保存到实验目录
            fold_dir = self.save_dir / "cv_folds"
            fold_dir.mkdir(parents=True, exist_ok=True)
            save_path = fold_dir / f"fold_{fold_num}_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      ✓ 本折训练曲线已保存: {save_path}")

        except Exception as e:
            print(f"      ⚠ 绘制本折曲线失败: {e}")


    def _plot_spatial_cv_curves(self, fold_histories):
        """绘制空间网格CV的训练曲线"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 左图：损失曲线
            ax1 = axes[0]
            for i, history in enumerate(fold_histories):
                epochs = range(1, len(history['train_losses']) + 1)
                ax1.plot(epochs, history['train_losses'], 'b-', alpha=0.2, linewidth=0.5)
                ax1.plot(epochs, history['val_losses'], 'r-', alpha=0.2, linewidth=0.5)

            # 平均曲线
            max_epochs = max(len(h['train_losses']) for h in fold_histories)
            avg_train = []
            avg_val = []
            for e in range(max_epochs):
                train_vals = [h['train_losses'][e] for h in fold_histories if e < len(h['train_losses'])]
                val_vals = [h['val_losses'][e] for h in fold_histories if e < len(h['val_losses'])]
                if train_vals:
                    avg_train.append(np.mean(train_vals))
                if val_vals:
                    avg_val.append(np.mean(val_vals))

            ax1.plot(range(1, len(avg_train) + 1), avg_train, 'b-', linewidth=2, label='平均训练损失')
            ax1.plot(range(1, len(avg_val) + 1), avg_val, 'r-', linewidth=2, label='平均验证损失')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('空间网格十折CV - 损失曲线')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 右图：r 曲线
            ax2 = axes[1]
            for i, history in enumerate(fold_histories):
                epochs = range(1, len(history['val_r']) + 1)
                ax2.plot(epochs, history['val_r'], 'g-', alpha=0.2, linewidth=0.5)

            # 平均 r
            avg_r = []
            for e in range(max_epochs):
                r_vals = [h['val_r'][e] for h in fold_histories if e < len(h['val_r'])]
                if r_vals:
                    avg_r.append(np.mean(r_vals))

            ax2.plot(range(1, len(avg_r) + 1), avg_r, 'g-', linewidth=2, label='平均验证 r')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('r')
            ax2.set_title('空间网格十折CV - r 曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.save_dir / "spatial_cv_training_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 十折训练曲线已保存: {save_path}")

        except Exception as e:
            print(f"⚠ 绘制曲线失败: {e}")

    def _plot_spatial_cv_boxplot(self, best_losses, best_rs):
        """绘制空间网格CV的箱线图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 左图：损失
            ax1 = axes[0]
            bp1 = ax1.boxplot(best_losses, patch_artist=True)
            bp1['boxes'][0].set_facecolor('lightcoral')
            ax1.set_ylabel('Validation Loss')
            ax1.set_title('空间网格十折 - 验证损失分布')
            ax1.grid(True, alpha=0.3)
            ax1.scatter(1, np.mean(best_losses), color='blue', s=50, zorder=5,
                       label=f'Mean: {np.mean(best_losses):.4f}')
            ax1.legend()

            # 右图：r
            ax2 = axes[1]
            bp2 = ax2.boxplot(best_rs, patch_artist=True)
            bp2['boxes'][0].set_facecolor('lightgreen')
            ax2.set_ylabel('r')
            ax2.set_title('空间网格十折 - 验证 r 分布')
            ax2.grid(True, alpha=0.3)
            ax2.scatter(1, np.mean(best_rs), color='blue', s=50, zorder=5,
                       label=f'Mean: {np.mean(best_rs):.4f}')
            ax2.legend()

            plt.tight_layout()
            save_path = self.save_dir / "spatial_cv_boxplot.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 十折箱线图已保存: {save_path}")

        except Exception as e:
            print(f"⚠ 绘制箱线图失败: {e}")
    
    def _diagnose_point_encoder_weights(self):
        """诊断 PointEncoder 第一层的权重分布"""
        # 查找 point_encoder 的第一层线性层
        target_layer = None
        layer_name = None
        for name, module in self.model.named_modules():
            if "point_encoder" in name and isinstance(module, nn.Linear):
                if target_layer is None:  # 取第一层
                    target_layer = module
                    layer_name = name
                    break

        if target_layer is None:
            print("❌ 未找到 point_encoder 层")
            return

        # 获取权重
        weight = target_layer.weight.data.cpu().numpy()
        print(f"\n找到目标层: {layer_name}")
        print(f"权重矩阵形状: {weight.shape} (输出维度, 输入维度)")

        input_dim = weight.shape[1]

        # 特征名称（16维）
        feature_names = [
            'LS1', 'LS2', 'LS3', 'LS4', 'LS5', 'LS6',
            'S1_VV', 'S1_VH', 'SMAP_TBV', 'SMAP_TBH',
            'lon_norm', 'lat_norm', 'doy_norm'
        ]

        print(f"\n【各输入维度的权重统计】")
        print(f"{'维度':<6} {'特征':<20} {'权重均值':<12} {'权重绝对值均值':<12}")
        print("-" * 55)

        for i in range(min(input_dim, len(feature_names))):
            col_weights = weight[:, i]
            mean_val = np.mean(col_weights)
            abs_mean = np.mean(np.abs(col_weights))
            marker = " 🔥新增" if i >= 13 else ""
            print(f"{i:<6} {feature_names[i]:<20} {mean_val:<12.6f} {abs_mean:<12.6f}{marker}")

        # 特别关注新增的3个维度（13,14,15）
        print("\n" + "=" * 70)
        print("🎯 重点关注：新增维度（降水累积 + 原产品值）")
        print("=" * 70)

        for i in [13, 14, 15]:
            if i < input_dim:
                col_weights = weight[:, i]
                mean_val = np.mean(col_weights)
                abs_mean = np.mean(np.abs(col_weights))
                zero_ratio = np.sum(np.abs(col_weights) < 1e-6) / len(col_weights)

                if abs_mean > 0.01:
                    status = "✅ 已学习"
                elif abs_mean > 0.001:
                    status = "⚠️ 正在学习（偏小）"
                else:
                    status = "❌ 几乎为零"

                print(f"\n特征 {i} ({feature_names[i]}):")
                print(f"  权重均值: {mean_val:.8f}")
                print(f"  权重绝对值均值: {abs_mean:.8f}")
                print(f"  接近零的比例: {zero_ratio*100:.1f}%")
                print(f"  学习状态: {status}")

        # 保存权重到文件
        save_path = self.save_dir / "point_encoder_weights_final.npz"
        np.savez(save_path, weight=weight, feature_names=feature_names[:input_dim])
        print(f"\n💾 权重已保存到: {save_path}")
        
    # ============================================================
    # [CONTRACT] SWE 标签反归一化范围
    # ============================================================
    def _bind_swe_scale_from_dataset(self, dataset, context=""):
        """
        从当前 Dataset 绑定 SWE 的真实 Min-Max 归一化范围。

        [COMPAT]
        新版 Dataset 暴露 swe_min/swe_max；旧版可能只有
        label_min/label_max，因此这里同时兼容两套字段。

        [CONTRACT]
        当前渐进式预训练使用固定的季节性积雪标签范围，默认 0–400 mm。
        该范围由 fixed_label_min_mm / fixed_label_max_mm 决定。
        GLACIER_SWE_THRESHOLD_MM=2000 仅用于过滤冰川或极端伪值，
        不能再被当作标签归一化上限。
        """
        min_candidates = [
            getattr(dataset, "swe_min", None),
            getattr(dataset, "label_min", None),
        ]
        max_candidates = [
            getattr(dataset, "swe_max", None),
            getattr(dataset, "label_max", None),
        ]

        swe_min = next((float(v) for v in min_candidates if v is not None), None)
        swe_max = next((float(v) for v in max_candidates if v is not None), None)

        if swe_min is None or swe_max is None:
            raise RuntimeError(
                "Dataset 缺少 swe_min/swe_max 和 label_min/label_max，"
                "禁止使用默认值进行反归一化。"
            )
        if not np.isfinite(swe_min) or not np.isfinite(swe_max) or swe_max <= swe_min:
            raise RuntimeError(
                f"Dataset SWE 范围非法: swe_min={swe_min}, swe_max={swe_max}"
            )

        self.swe_min = swe_min
        self.swe_max = swe_max
        self.config["swe_min"] = swe_min
        self.config["swe_max"] = swe_max

        tag = f" ({context})" if context else ""
        print(f"\n✅ 已绑定 Dataset SWE 反归一化范围{tag}:")
        print(f"   swe_min = {self.swe_min:.6f} mm")
        print(f"   swe_max = {self.swe_max:.6f} mm")
        print(f"   range   = {self.swe_max - self.swe_min:.6f} mm")

        if "pretrain" in context.lower():
            # 当前正式预训练采用固定季节性积雪范围（默认 0–400 mm）。
            # 2000 mm 只是冰川/极端伪值过滤阈值，不是归一化上限。
            expected_min = float(self.config.get("fixed_label_min_mm", self.swe_min))
            expected_max = float(self.config.get("fixed_label_max_mm", self.swe_max))

            if not np.isclose(self.swe_min, expected_min, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    f"预训练 swe_min 与配置不一致：Dataset={self.swe_min:.6f} mm，"
                    f"fixed_label_min_mm={expected_min:.6f} mm。"
                )
            if not np.isclose(self.swe_max, expected_max, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    f"预训练 swe_max 与配置不一致：Dataset={self.swe_max:.6f} mm，"
                    f"fixed_label_max_mm={expected_max:.6f} mm。"
                    "请检查启动参数、归一化配置和 Dataset 是否使用同一标签范围。"
                )

            print(
                f"   ✅ 预训练 SWE 范围与固定配置一致: "
                f"[{expected_min:.1f}, {expected_max:.1f}] mm"
            )

        return self.swe_min, self.swe_max

    def _require_swe_scale(self, context=""):
        """返回已绑定的 SWE 范围；缺失时直接报错，不允许静默回退。"""
        if not hasattr(self, "swe_min") or not hasattr(self, "swe_max"):
            suffix = f" ({context})" if context else ""
            raise RuntimeError(
                f"缺少 SWE 反归一化范围{suffix}。"
                "必须先调用 _bind_swe_scale_from_dataset(dataset)。"
            )

        swe_min = float(self.swe_min)
        swe_max = float(self.swe_max)
        if not np.isfinite(swe_min) or not np.isfinite(swe_max) or swe_max <= swe_min:
            raise RuntimeError(
                f"SWE 反归一化范围非法: swe_min={swe_min}, swe_max={swe_max}"
            )
        return swe_min, swe_max

    # ============================================================
    # [MODE] pretrain_cv
    # ============================================================
    # 预训练十折交叉验证。
    #
    # [CONTRACT]
    #   只用于诊断当前预训练配置的稳定性。
    #   每折保存 best model 和 fold metrics。
    #   返回 cv_results, full_dataset，供 pretrain_progressive 的 Step 3 复用。
    #
    # [SPEED]
    #   支持 val_every，跳过验证的 epoch 不参与：
    #       - scheduler.step
    #       - best model
    #       - early stopping
    #
    # [DANGER]
    #   这里是 pretrain_progressive 的 Step 1。
    #   改这里会影响最终预训练流程，不只是单独 CV。
    # ============================================================
    def run_pretrain_cv_workflow(self, manifest_only=False):
        """
        预训练阶段的十折交叉验证编排器（增强版）
        支持多年份数据训练
        包含：每折保存最佳模型、详细指标记录、损失曲线绘制、验证集散点图
        支持共享缓存目录，跨实验复用
        支持站点引导采样
        支持自适应修正
        """
        print(f"\n{'█'*80}")
        print("🌟 启动预训练十折交叉验证 (10-Fold CV)")
        print(f"   支持多年份数据训练")
        print(f"{'█'*80}")

        # 1. 创建完整数据集（支持多年份）
        from data_online_era5_swe import SWEDataset

        print("\n正在加载完整栅格数据集...")

        # ========== 获取训练年份 ==========
        pretrain_years = self.config.get("pretrain_years", [2015, 2016, 2017])
        if isinstance(pretrain_years, int):
            pretrain_years = [pretrain_years]

        print(f"  📅 训练年份: {pretrain_years}")

        # 🔥 关键修复：直接传入多年份列表
        year_target = pretrain_years

        # ========== 采样参数 ==========
        samples_per_day = int(os.environ.get("PRETRAIN_SAMPLES_PER_DAY", self.config.get("pretrain_samples_per_day", 20000)))
        print(f"  📊 每日采样数: {samples_per_day}")

        # ========== 采样来源配置 ==========
        sampling_mode = self.config.get("sampling_mode", "auto")
        use_station_guide = self.config.get("use_station_guide", False)
        station_guide_file = self.config.get("station_guide_file")
        station_neighborhood = self.config.get("station_neighborhood", 3)
        station_samples_per_day = self.config.get("station_samples_per_day", 2000)
        station_filter_zero_target = self.config.get("station_filter_zero_target", True)
        station_sampling_unit = self.config.get(
            "station_sampling_unit", "positions_all_dates"
        )
        station_record_dedup = self.config.get("station_record_dedup", "grid_date")
        station_date_column = self.config.get("station_date_column")
        station_record_manifest_path = self.config.get("station_record_manifest_path")
        external_station_glob = self.config.get("external_station_glob")
        external_station_exclusion_radius = self.config.get(
            "external_station_exclusion_radius", 0
        )
        external_station_strict = self.config.get("external_station_strict", False)
        external_station_report_path = self.config.get("external_station_report_path")

        resolved_mode = sampling_mode
        if resolved_mode == "auto":
            resolved_mode = "hybrid" if use_station_guide else "random"

        print(f"\n📍 预训练采样模式: {resolved_mode}")
        if resolved_mode == "incremental":
            print(f"   固定清单: {self.config.get('incremental_manifest_path')}")
            print(f"   当前累计读取 Stage 1-{self.config.get('incremental_stage', 1)}")
            print(f"   上一阶段模型: {self.config.get('pretrained_model')}")
        if resolved_mode in {"station", "hybrid"}:
            print(f"   站点文件: {station_guide_file or self.config.get('station_csv_dir')}")
            print(f"   邻域半径: {station_neighborhood} ({station_neighborhood*2+1}x{station_neighborhood*2+1})")
            limit_text = "全部" if station_samples_per_day <= 0 else str(station_samples_per_day)
            print(f"   每日站点样本上限: {limit_text}")
            print(f"   站点采样单位: {station_sampling_unit}")
            if station_sampling_unit == "records":
                print(f"   实际记录去重: {station_record_dedup}")
                print(f"   日期列: {station_date_column or '自动识别'}")
            print(f"   过滤 ERA5-Land SWE=0: {station_filter_zero_target}")

        # ============ 🔥 获取自适应修正配置 ============
        use_adaptive_supplement = self.config.get("use_adaptive_supplement", False)
        adaptive_alpha = self.config.get("adaptive_alpha", 0.5)
        adaptive_threshold = self.config.get("adaptive_threshold", 1.5)

        print(f"\n🔥 自适应修正配置:")
        print(f"   use_adaptive_supplement: {use_adaptive_supplement}")
        print(f"   adaptive_alpha: {adaptive_alpha}")
        print(f"   adaptive_threshold: {adaptive_threshold}")

        # ========== 🔥 共享缓存目录配置 ==========
        # 优先级: 命令行参数 > 配置 > 默认路径
        shared_cache_dir = self.config.get("shared_cache_dir")

        if shared_cache_dir is None:
            # 使用默认共享目录
            shared_cache_dir = Path("/root/autodl-tmp/shared_cache")
        else:
            shared_cache_dir = Path(shared_cache_dir)

        shared_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 共享缓存目录: {shared_cache_dir}")
        print(f"💡 缓存将跨实验复用，大幅提升加载速度")
        # ==========================================

        # 清单准备阶段可禁用Dataset缓存，避免把完整栅格数组再次写入磁盘。
        disable_dataset_cache = bool(self.config.get("disable_dataset_cache", False))
        cache_dir = None if disable_dataset_cache else shared_cache_dir
        if disable_dataset_cache:
            print("   ⏭ disable_dataset_cache=True：本次不读写Dataset pickle缓存")

        # 是否强制重新加载
        force_reload = self.config.get("force_reload", False)
        if force_reload:
            print(f"⚠️ 强制重新加载模式: 将忽略现有缓存")

        # 创建数据集（传递站点引导参数 + 自适应修正参数）
        full_dataset = SWEDataset(
            region=self.config.get("region", "XINJIANG"),
            year_target=year_target,
            patch_size=self.config.get("patch_size", 5),
            min_valid_pixels=self.config.get("min_valid_pixels", 100),
            samples_per_day=samples_per_day,
            clamday_threshold=self.config.get("clamday_threshold", 0.5),
            s1_interp_method=self.config.get("s1_interp_method", "nearest"),
            s1_max_gap_days=self.config.get("s1_max_gap_days", 7),
            smap_interp_method=self.config.get("smap_interp_method", "nearest"),
            smap_max_gap_days=self.config.get("smap_max_gap_days", 7),
            cache_dir=cache_dir,                     # 🔥 使用共享缓存
            force_reload=force_reload,               # 🔥 是否强制重新加载
            # ============ 采样来源参数 ============
            sampling_mode=sampling_mode,
            use_station_guide=use_station_guide,  # 旧命令兼容
            station_guide_file=station_guide_file,
            station_csv_dir=self.config.get("station_csv_dir", Path("/root/ablation")),
            station_neighborhood=station_neighborhood,
            station_samples_per_day=station_samples_per_day,
            station_filter_zero_target=station_filter_zero_target,
            station_sampling_unit=station_sampling_unit,
            station_record_dedup=station_record_dedup,
            station_date_column=station_date_column,
            station_record_manifest_path=station_record_manifest_path,
            external_station_glob=external_station_glob,
            external_station_exclusion_radius=external_station_exclusion_radius,
            external_station_strict=external_station_strict,
            external_station_report_path=external_station_report_path,
            # ============ 固定增量随机池 ============
            incremental_manifest_path=self.config.get("incremental_manifest_path"),
            incremental_stage=self.config.get("incremental_stage", 1),
            build_incremental_manifest=self.config.get("build_incremental_manifest", False),
            incremental_pool_size=self.config.get("incremental_pool_size", 152000),
            incremental_stage_sizes=self.config.get(
                "incremental_stage_sizes", [12000, 20000, 40000, 80000]
            ),
            incremental_seed=self.config.get("incremental_seed", self.config.get("seed", 43)),
            incremental_candidate_oversample_factor=self.config.get(
                "incremental_candidate_oversample_factor", 3.0
            ),
            incremental_exclude_station_pixels=self.config.get(
                "incremental_exclude_station_pixels", True
            ),
            incremental_ratio_config=self.config.get("incremental_ratio_config"),
            incremental_glacier_mask_path=self.config.get("incremental_glacier_mask_path"),
            incremental_fold_block_pixels=self.config.get("incremental_fold_block_pixels", 0),
            # 季节性判据只在 incremental manifest 构建时使用。
            seasonal_min_peak_swe_mm=self.config.get("seasonal_min_peak_swe_mm", 1.0),
            seasonal_max_swe_mm=self.config.get("seasonal_max_swe_mm", 400.0),
            seasonal_snow_free_threshold_mm=self.config.get(
                "seasonal_snow_free_threshold_mm", 1.0
            ),
            seasonal_min_warm_snow_free_ratio=self.config.get(
                "seasonal_min_warm_snow_free_ratio", 0.0
            ),
            seasonal_min_consecutive_snow_free_days=self.config.get(
                "seasonal_min_consecutive_snow_free_days", 5
            ),
            seasonal_min_snow_year_coverage_ratio=self.config.get(
                "seasonal_min_snow_year_coverage_ratio", 0.90
            ),
            normalization_config_path=self.config.get("normalization_config_path"),
            normalization_mode=self.config.get("normalization_mode", "auto"),
            fixed_label_min_mm=self.config.get("fixed_label_min_mm", 0.0),
            fixed_label_max_mm=self.config.get("fixed_label_max_mm", 400.0),
            # ============ 🔥 添加自适应修正参数 ============
            use_adaptive_supplement=use_adaptive_supplement,
            adaptive_alpha=adaptive_alpha,
            adaptive_threshold=adaptive_threshold,
        )

        # 同步维度信息到 config
        self.config["C_conv"] = full_dataset.C_conv
        self.config["C_point"] = full_dataset.C_point
        # [CONTRACT] 预训练绘图、mm指标和阈值诊断必须使用 Dataset 的真实范围。
        self._bind_swe_scale_from_dataset(full_dataset, context="pretrain_cv")

        total_samples = len(full_dataset)
        indices = np.arange(total_samples)

        # 🔍 调试信息：检查标签年份分布
        print(f"\n📅 标签数据年份分布:")
        year_counts = {}
        for dt in full_dataset.label_data.keys():
            year_counts[dt.year] = year_counts.get(dt.year, 0) + 1
        for year, count in sorted(year_counts.items()):
            print(f"    {year}年: {count} 个文件")

        print(f"\n✅ 数据集加载完成")
        print(f"   总样本数: {total_samples:,}")
        print(f"   卷积特征维度: {self.config['C_conv']}")
        print(f"   点特征维度: {self.config['C_point']}")
        if full_dataset.use_station_guide:
            print(f"   站点像元数: {len(full_dataset.station_pixels):,}")

        if manifest_only:
            print("\n✅ 固定152000增量清单准备完成；按要求不启动十折训练")
            return {
                "manifest_only": True,
                "manifest_path": str(getattr(full_dataset, "incremental_manifest_path", "")),
                "stage_loaded": int(getattr(full_dataset, "incremental_stage", 1)),
                "stage_samples": int(len(full_dataset)),
            }, full_dataset

        # 2. 初始化十折划分。incremental 清单自带固定 fold_id；
        # 其他模式保持旧KFold逻辑。
        sample_fold_ids = getattr(full_dataset, "sample_fold_ids", None)
        if (
            resolved_mode == "incremental"
            and sample_fold_ids is not None
            and len(sample_fold_ids) == total_samples
        ):
            cv_splits = []
            for fold_id in range(1, 11):
                val_idx = np.flatnonzero(sample_fold_ids == fold_id)
                train_idx = np.flatnonzero(sample_fold_ids != fold_id)
                if len(val_idx) == 0 or len(train_idx) == 0:
                    raise RuntimeError(
                        f"固定增量清单 Fold {fold_id} 为空；请检查 fold 分配"
                    )
                cv_splits.append((train_idx, val_idx))
            print("   ✅ 使用manifest中的固定fold_id")
        else:
            kf = KFold(
                n_splits=10, shuffle=True,
                random_state=self.config.get('seed', 42)
            )
            cv_splits = list(kf.split(indices))

        # [CUMULATIVE-SCRATCH] --from_scratch 时每折独立随机初始化。
        is_from_scratch = bool(self.config.get("from_scratch", False))
        pretrain_init_model = (
            None if is_from_scratch else self.config.get("pretrained_model")
        )

        if resolved_mode == "incremental":
            if is_from_scratch:
                print("   ✅ 累计池从头训练：每一折均随机初始化，不加载Stage 0或上一阶段权重")
            else:
                if not pretrain_init_model or not os.path.exists(pretrain_init_model):
                    raise FileNotFoundError(
                        "非from_scratch的incremental训练必须通过 --pretrained_model "
                        "指定初始化模型"
                    )
                print(f"   ✅ 每一折均从同一个模型初始化: {pretrain_init_model}")
        elif pretrain_init_model:
            print(f"   ℹ 预训练CV将加载初始化权重: {pretrain_init_model}")

        all_fold_metrics = []
        fold_histories = []

        # ============================================================
        # [RESUME] 预训练CV按折续跑
        # ============================================================
        # [CONTRACT]
        #   - 已完整结束的折直接跳过，不重新训练。
        #   - 只有 best.pth 但没有完成标记/曲线/散点图时，不视为完成；
        #     这通常意味着该折在训练中途被中断，应从该折重新开始。
        #   - 跳过折时会从 checkpoint 或 result.json 恢复本折汇总指标，
        #     确保十折最终统计仍包含此前完成的折。
        #
        # [DANGER]
        #   必须复用相同数据缓存、seed、总样本数和KFold参数。
        #   数据集顺序变化时，旧 Fold 1/2 与新 Fold 1/2 不再对应，不能续跑。
        resume_pretrain_cv = bool(self.config.get("resume_pretrain_cv", False))
        redraw_completed_cv_plots = bool(self.config.get("redraw_completed_cv_plots", False))
        if resume_pretrain_cv:
            print("\n♻️ 已启用预训练CV按折续跑")
            print(f"   续跑实验目录: {self.save_dir}")
            if redraw_completed_cv_plots:
                print("   🎨 已完成折将加载 best checkpoint 重新生成正确 mm 尺度散点图（不训练）")

        def _to_builtin(value):
            """把 numpy / torch 标量转换为 JSON 可写的 Python 标量。"""
            if isinstance(value, np.generic):
                return value.item()
            if torch.is_tensor(value) and value.numel() == 1:
                return value.detach().cpu().item()
            return value

        def _read_fold_result_for_resume(fold_idx, result_path, checkpoint_path):
            """
            [COMPAT] 优先读取新版本每折 result.json；
            对已经跑完但没有 result.json 的旧折，从 best checkpoint 恢复核心指标。
            """
            if result_path.exists():
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                result["resumed"] = True
                return result

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            metrics = checkpoint.get("metrics", {}) or {}
            metrics = {k: _to_builtin(v) for k, v in metrics.items()}

            return {
                "fold": int(fold_idx),
                "best_epoch": int(checkpoint.get("epoch", -1)) + 1,
                "best_val_loss": float(metrics.get("loss", float("inf"))),
                "best_val_r2": float(metrics.get("r2", 0.0) or 0.0),
                "best_val_r": float(metrics.get("correlation", 0.0) or 0.0),
                "train_losses": [],
                "val_losses": [],
                "final_val_metrics": metrics,
                "resumed": True,
                "history_available": False,
            }

        def _get_raw_swe_values(dataset, subset_indices):
            """
            [DIAG] 根据 meta_index 直接读取未归一化 ERA5-Land SWE。
            兼容 tuple(date,row,col,source) 和旧 dict 元数据格式。
            """
            values = []
            for sample_idx in subset_indices:
                meta = dataset.meta_index[int(sample_idx)]
                if isinstance(meta, dict):
                    date_dt = meta.get("date")
                    row = meta.get("row")
                    col = meta.get("col")
                else:
                    date_dt, row, col = meta[:3]

                label_entry = dataset.label_data[date_dt]
                label_arr = label_entry[0] if isinstance(label_entry, tuple) else label_entry
                value = float(label_arr[int(row), int(col)])
                if np.isfinite(value):
                    values.append(value)

            return np.asarray(values, dtype=np.float32)

        def _save_and_print_fold_distribution(fold_idx, val_idx):
            """
            [DIAG] 验证每折是否继承 33% / 33% / 34% 的总体采样分布。
            该检查在跳过旧折之前也会执行，因此不用重训 Fold 1/2。
            """
            values = _get_raw_swe_values(full_dataset, val_idx)
            n = int(values.size)
            if n == 0:
                print(f"   ⚠ Fold {fold_idx} 无可用原始 SWE，跳过分布检查")
                return None

            low_n = int(np.sum(values <= 5.0))
            mid_n = int(np.sum((values > 5.0) & (values <= 30.0)))
            high_n = int(np.sum(values > 30.0))
            over80_n = int(np.sum(values > 80.0))
            over120_n = int(np.sum(values > 120.0))

            result = {
                "fold": int(fold_idx),
                "n": n,
                "le_5": {"count": low_n, "ratio": low_n / n},
                "gt_5_le_30": {"count": mid_n, "ratio": mid_n / n},
                "gt_30": {"count": high_n, "ratio": high_n / n},
                "gt_80_count": over80_n,
                "gt_120_count": over120_n,
                "min_mm": float(np.min(values)),
                "max_mm": float(np.max(values)),
                "mean_mm": float(np.mean(values)),
            }

            print(f"\n📊 Fold {fold_idx} 验证集原始 SWE 分布检查:")
            print(f"   有效样本:          {n:,}")
            print(f"   SWE ≤ 5 mm:        {low_n:,} ({low_n / n * 100:.2f}%)")
            print(f"   5 < SWE ≤ 30 mm:   {mid_n:,} ({mid_n / n * 100:.2f}%)")
            print(f"   SWE > 30 mm:       {high_n:,} ({high_n / n * 100:.2f}%)")
            print(f"   SWE > 80 mm:       {over80_n:,}")
            print(f"   SWE > 120 mm:      {over120_n:,}")
            print(f"   SWE范围:           [{values.min():.2f}, {values.max():.2f}] mm")

            dist_path = self.save_dir / f"pretrain_cv_fold{fold_idx}_val_distribution.json"
            with open(dist_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   💾 分布检查已保存: {dist_path}")
            return result

        def _redraw_completed_fold_scatter(fold_idx, val_idx, checkpoint_path):
            """
            [DIAG] 已完成折只重新推理并覆盖散点图，不更新任何模型参数。
            """
            print(f"\n🎨 Fold {fold_idx}: 重新生成正确 mm 尺度散点图（不训练）")

            num_workers = int(self.config.get("num_workers", 4))
            self.val_loader = DataLoader(
                Subset(full_dataset, val_idx),
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )

            self.model = create_model(
                model_type=self.config["model_type"],
                C_spatial=self.config["C_conv"],
                C_point=self.config["C_point"],
                d_model=self.config["d_model"],
                use_wide_branch=False,
            ).to(self.device)

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"   ⚠ missing_keys: {len(incompatible.missing_keys)}")
            if incompatible.unexpected_keys:
                print(f"   ⚠ unexpected_keys: {len(incompatible.unexpected_keys)}")

            self.model.eval()
            preds, targets, _ = self._make_predictions(self.val_loader)
            if preds is None or len(preds) == 0:
                raise RuntimeError(f"Fold {fold_idx} 重绘时没有获得有效预测")

            s_min, s_max = self._require_swe_scale(context=f"pretrain_cv_fold_{fold_idx}_redraw")
            preds_denorm = preds * (s_max - s_min) + s_min
            targets_denorm = targets * (s_max - s_min) + s_min

            plot_metrics = self.plot_density_scatter_hardcode(
                preds_denorm,
                targets_denorm,
                is_fine_tune=False,
                fold_index=fold_idx,
            )

            scale_meta = {
                "fold": int(fold_idx),
                "checkpoint": str(checkpoint_path),
                "swe_min": float(s_min),
                "swe_max": float(s_max),
                "n_predictions": int(len(preds_denorm)),
                "n_targets": int(len(targets_denorm)),
                "target_le_5": int(np.sum(targets_denorm <= 5.0)),
                "target_gt_5_le_30": int(np.sum((targets_denorm > 5.0) & (targets_denorm <= 30.0))),
                "target_gt_30": int(np.sum(targets_denorm > 30.0)),
                "target_max_mm": float(np.max(targets_denorm)),
                "plot_metrics": plot_metrics,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            meta_path = self.save_dir / f"pretrain_cv_fold{fold_idx}_scatter_scale.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(scale_meta, f, indent=2, ensure_ascii=False)
            print(f"   💾 散点图尺度元数据: {meta_path}")

            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 3. 十折循环
        # [FIX] 即使续跑时十折全部跳过，汇总结果仍需要epochs。
        epochs = self.config.get(
            "pretrain_cv_epochs",
            self.config.get("epochs", 100)
        )

        max_cv_folds = int(
            self.config.get("pretrain_cv_max_folds", 10)
        )
        max_cv_folds = min(
            10,
            max(1, max_cv_folds),
        )

        print(
            f"   本次最多运行CV折数: {max_cv_folds}"
        )

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            fold_idx = fold + 1

            if fold_idx > max_cv_folds:
                print(
                    f"✅ 已完成指定的{max_cv_folds}折，停止CV"
                )
                break

            # 每折先检查验证集目标分布；该操作不需要构建模型。
            fold_distribution = _save_and_print_fold_distribution(fold_idx, val_idx)

            fold_best_path = self.save_dir / f"pretrain_cv_fold{fold_idx}_best.pth"
            fold_result_path = self.save_dir / f"pretrain_cv_fold{fold_idx}_result.json"
            fold_complete_marker = self.save_dir / f"pretrain_cv_fold{fold_idx}.complete.json"
            fold_curve_path = self.save_dir / "cv_folds" / f"fold_{fold_idx}_curves.png"
            fold_scatter_path = self.save_dir / f"density_scatter_chinese_fold_{fold_idx}.png"

            # [COMPAT] 旧版没有 complete.json。
            # 对旧实验，best checkpoint + 曲线 + 散点图同时存在，才认定该折完整结束。
            legacy_fold_complete = (
                fold_best_path.exists()
                and fold_curve_path.exists()
                and fold_scatter_path.exists()
            )
            fold_is_complete = fold_complete_marker.exists() or legacy_fold_complete

            if resume_pretrain_cv and fold_is_complete:
                try:
                    resumed_result = _read_fold_result_for_resume(
                        fold_idx, fold_result_path, fold_best_path
                    )
                    resumed_result["val_distribution"] = fold_distribution
                    all_fold_metrics.append(resumed_result)

                    # 新版 result.json 中若有完整历史，可继续纳入总训练曲线。
                    if resumed_result.get("train_losses"):
                        fold_histories.append({
                            "train_losses": resumed_result.get("train_losses", []),
                            "val_losses": resumed_result.get("val_losses", []),
                            "val_r": resumed_result.get("val_r_history", []),
                            "val_r2": resumed_result.get("val_r2_history", []),
                        })

                    # 给旧折补写完成标记，之后续跑识别更稳。
                    if not fold_complete_marker.exists():
                        marker = {
                            "fold": int(fold_idx),
                            "status": "complete",
                            "recovered_from_legacy_files": True,
                            "checkpoint": str(fold_best_path),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        with open(fold_complete_marker, "w", encoding="utf-8") as f:
                            json.dump(marker, f, indent=2, ensure_ascii=False)

                    print(f"\n⏭️ Fold {fold_idx} 已完整完成，续跑时跳过训练")
                    print(f"   checkpoint: {fold_best_path}")
                    print(f"   best epoch: {resumed_result.get('best_epoch', 'N/A')}")
                    print(f"   best val loss: {resumed_result.get('best_val_loss', 'N/A')}")
                    print(f"   best val r: {resumed_result.get('best_val_r', 'N/A')}")

                    if redraw_completed_cv_plots:
                        _redraw_completed_fold_scatter(fold_idx, val_idx, fold_best_path)

                    continue
                except Exception as e:
                    print(f"   ⚠ Fold {fold_idx} 恢复信息读取失败，将重新训练该折: {e}")

            print(f"\n\n{'='*60}")
            print(f"🟢 预训练 FOLD {fold_idx} / 10")
            print(f"   训练样本: {len(train_idx):,}")
            print(f"   验证样本: {len(val_idx):,}")
            print(f"{'='*60}")

            # 3.1 创建本折的 DataLoader
            self.train_loader = DataLoader(
                Subset(full_dataset, train_idx),
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.config.get('num_workers', 8) > 0
            )

            self.val_loader = DataLoader(
                Subset(full_dataset, val_idx),
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True,
                persistent_workers=self.config.get('num_workers', 8) > 0
            )

            # 3.2 全新初始化模型（关键！）
            print(f"  -> 创建新模型实例...")

            # 获取模型配置
            freeze_backbone = self.config.get('freeze_backbone', False)
            freeze_strategy = self.config.get('freeze_strategy', 'fusion_ft')
            use_residual_injection = self.config.get('residual_injection', False)

            success = self.build_model(
                # Stage 0 为None；Stage 1-4每折都重新加载同一个上一阶段正式模型。
                load_pretrained=pretrain_init_model,
                freeze_backbone=False,
                freeze_strategy=freeze_strategy,
                use_residual=use_residual_injection,
                is_cv_fold=True            # 减少打印输出
            )

            if not success:
                print(f"  ❌ 模型构建失败，跳过折 {fold_idx}")
                continue

            # 3.3 训练本折
            print(f"  -> 开始训练...")

            best_val_loss = float('inf')
            best_val_r2 = -float('inf')
            best_val_r = -float('inf')
            best_epoch = 0
            patience_counter = 0

            # 记录本折的训练历史
            fold_train_losses = []
            fold_val_losses = []
            fold_val_metrics = []

            epochs = self.config.get("pretrain_cv_epochs", self.config.get("epochs", 100))
            val_every = int(self.config.get("val_every", 1))

            for epoch in range(epochs):
                # 训练一个 epoch
                train_loss = self.train_epoch(epoch, is_fine_tune=False)
                fold_train_losses.append(train_loss)

                # 验证（按 val_every 跳间隔）
                need_validate = (
                    (epoch + 1) % val_every == 0
                    or epoch == epochs - 1
                    or epoch >= epochs - 10
                )

                if need_validate:
                    val_metrics = self.validate(self.val_loader, is_fine_tune=False)
                else:
                    prev = fold_val_metrics[-1] if fold_val_metrics else {}
                    val_metrics = {
                        "loss": fold_val_losses[-1] if fold_val_losses else float("inf"),
                        "rmse": float("nan"),
                        "mae": float("nan"),
                        "correlation": prev.get("correlation", 0.0),
                        "r2": prev.get("r2", 0.0),
                        "n_samples": 0,
                    }

                fold_val_losses.append(val_metrics["loss"])
                fold_val_metrics.append(val_metrics)

                # 学习率调度
                current_lr = self.optimizer.param_groups[0]["lr"]

                if (
                    need_validate
                    and not getattr(
                        self,
                        "scheduler_step_per_batch",
                        False,
                    )
                ):
                    self.scheduler.step(
                        val_metrics["loss"]
                    )

                # 打印进度
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_metrics['loss']:.6f} | "
                          f"Val r: {val_metrics.get('correlation', 0):.4f} | "
                          f"LR: {current_lr:.2e}")

                # 只有真正验证过的 epoch 才允许更新 best / early stopping
                if need_validate:
                    is_better = val_metrics["loss"] < best_val_loss

                    if is_better:
                        best_val_loss = val_metrics["loss"]
                        best_val_r2 = val_metrics.get('r2', 0)
                        best_val_r = val_metrics.get('correlation', 0)
                        best_epoch = epoch
                        patience_counter = 0

                        # 保存本折最佳模型
                        self.save_checkpoint(
                            f"pretrain_cv_fold{fold_idx}_best.pth",
                            epoch,
                            val_metrics
                        )
                    else:
                        patience_counter += 1

                    # 早停检查
                    patience = self.config.get("patience", 15)
                    disable_cv_early_stop = bool(
                        self.config.get("disable_pretrain_cv_early_stopping", False)
                    )
                    if (not disable_cv_early_stop) and patience_counter >= patience:
                        print(f"    🛑 早停触发！最佳 Epoch: {best_epoch + 1}, "
                              f"Best Val Loss: {best_val_loss:.6f}")
                        break
                # else: 跳过验证的 epoch 不参与 best / patience

            # ============ 绘制本折训练曲线 ============
            print(f"\n  📈 [FOLD {fold_idx}] 绘制训练曲线...")
            self._plot_single_fold_curve(
                fold_idx, 
                fold_train_losses, 
                fold_val_losses, 
                [m.get('correlation', 0) for m in fold_val_metrics]
            )

            # ============ 绘制本折验证集散点图 ============
            print(f"  📊 [FOLD {fold_idx}] 绘制验证集散点图...")
            self.model.eval()
            preds, targets, _ = self._make_predictions(self.val_loader)

            if preds is not None and len(preds) > 0:
                # [DANGER] 不允许回退到 200 mm；必须继承 Dataset 的真实标签范围。
                s_min, s_max = self._require_swe_scale(context=f"pretrain_cv_fold_{fold_idx}")
                preds_denorm = preds * (s_max - s_min) + s_min
                targets_denorm = targets * (s_max - s_min) + s_min

                # 绘制本折散点图
                self.plot_density_scatter_hardcode(
                    preds_denorm, targets_denorm,
                    is_fine_tune=False,  # 预训练模式
                    fold_index=fold_idx
                )
            else:
                print(f"      ⚠️ 无法绘制散点图：无有效预测结果")

            # 3.4 记录本折最终指标
            fold_result = {
                'fold': fold_idx,
                'best_epoch': best_epoch + 1,
                'best_val_loss': best_val_loss,
                'best_val_r2': best_val_r2,
                'best_val_r': best_val_r,
                'train_losses': fold_train_losses,
                'val_losses': fold_val_losses,
                'final_val_metrics': val_metrics
            }
            all_fold_metrics.append(fold_result)
            val_r_history = [m.get('correlation', 0) for m in fold_val_metrics]
            val_r2_history = [m.get('r2', 0) for m in fold_val_metrics]
            fold_histories.append({
                'train_losses': fold_train_losses,
                'val_losses': fold_val_losses,
                'val_r': val_r_history,
                'val_r2': val_r2_history
            })

            # [RESUME] 每折结束立即保存独立结果和完成标记。
            # 下次运行时只跳过有完成标记的折，不会误把中途 checkpoint 当成完整折。
            fold_result['val_r_history'] = val_r_history
            fold_result['val_r2_history'] = val_r2_history
            fold_result['val_distribution'] = fold_distribution

            with open(fold_result_path, 'w', encoding='utf-8') as f:
                json.dump(fold_result, f, indent=2, ensure_ascii=False, default=str)

            complete_info = {
                'fold': int(fold_idx),
                'status': 'complete',
                'best_checkpoint': str(fold_best_path),
                'result_json': str(fold_result_path),
                'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'seed': int(self.config.get('seed', 42)),
                'total_samples': int(total_samples),
            }
            with open(fold_complete_marker, 'w', encoding='utf-8') as f:
                json.dump(complete_info, f, indent=2, ensure_ascii=False)

            progress_path = self.save_dir / 'pretrain_cv_progress.json'
            completed_folds = sorted([int(m['fold']) for m in all_fold_metrics])
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'completed_folds': completed_folds,
                    'total_folds': 10,
                    'seed': int(self.config.get('seed', 42)),
                    'total_samples': int(total_samples),
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }, f, indent=2, ensure_ascii=False)

            print(f"  ✅ Fold {fold_idx} 完成标记已保存: {fold_complete_marker}")

            print(f"\n  📊 [FOLD {fold_idx}] 总结:")
            print(f"     最佳验证损失: {best_val_loss:.6f}")
            print(f"     最佳验证 r: {best_val_r:.4f}")
            print(f"     最佳轮次: {best_epoch + 1}")

            # 3.5 清理显存（重要！防止显存溢出）
            del self.model
            del self.optimizer
            del self.scheduler
            try:
                del self.train_loader
                del self.val_loader
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()

        # 4. 汇总十折结果
        print(f"\n\n{'█'*80}")
        print("🏆 预训练十折交叉验证完成！")
        print(f"{'█'*80}")

        # 提取指标
        best_losses = [m['best_val_loss'] for m in all_fold_metrics]
        best_rs = [m['best_val_r'] for m in all_fold_metrics]

        print(f"\n📊 十折统计结果:")
        print(f"  验证损失 (Loss):")
        print(f"    均值: {np.mean(best_losses):.6f}")
        print(f"    标准差: {np.std(best_losses):.6f}")
        print(f"    最小值: {np.min(best_losses):.6f}")
        print(f"    最大值: {np.max(best_losses):.6f}")

        print(f"\n  验证 r:")
        print(f"    均值: {np.mean(best_rs):.4f}")
        print(f"    标准差: {np.std(best_rs):.4f}")
        print(f"    最小值: {np.min(best_rs):.4f}")
        print(f"    最大值: {np.max(best_rs):.4f}")

        # 各折详细结果
        print(f"\n📋 各折详细结果:")
        print(f"{'折数':<6} {'最佳轮次':<8} {'最佳损失':<12} {'最佳r':<10}")
        print("-" * 40)
        for m in all_fold_metrics:
            print(f"{m['fold']:<6} {m['best_epoch']:<8} {m['best_val_loss']:<12.6f} {m['best_val_r']:<10.4f}")

        # 5. 保存结果
        cv_results = {
            'n_folds': 10,
            'seed': self.config.get('seed', 42),
            'pretrain_years': pretrain_years,
            'samples_per_day': samples_per_day,
            'sampling_mode': full_dataset.sampling_mode,
            'use_station_guide': full_dataset.use_station_guide,
            'station_guide_file': str(full_dataset.station_guide_file) if full_dataset.station_guide_file else None,
            'station_neighborhood': station_neighborhood,
            'station_samples_per_day': station_samples_per_day,
            'station_filter_zero_target': station_filter_zero_target,
            'station_sampling_unit': station_sampling_unit,
            'station_record_dedup': station_record_dedup,
            'station_date_column': station_date_column,
            'use_adaptive_supplement': use_adaptive_supplement,
            'adaptive_alpha': adaptive_alpha,
            'adaptive_threshold': adaptive_threshold,
            'shared_cache_dir': str(shared_cache_dir),
            'total_samples': total_samples,
            'fold_metrics': all_fold_metrics,
            'summary': {
                'loss_mean': float(np.mean(best_losses)),
                'loss_std': float(np.std(best_losses)),
                'loss_min': float(np.min(best_losses)),
                'loss_max': float(np.max(best_losses)),
                'r_mean': float(np.mean(best_rs)),
                'r_std': float(np.std(best_rs)),
                'r_min': float(np.min(best_rs)),
                'r_max': float(np.max(best_rs)),
            },
            'config': {
                'batch_size': self.config['batch_size'],
                'learning_rate': self.config['learning_rate'],
                'd_model': self.config['d_model'],
                'epochs_per_fold': epochs,
            }
        }

        save_path = self.save_dir / "pretrain_cv_results.json"
        with open(save_path, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        print(f"\n💾 十折结果已保存: {save_path}")

        # 6. 绘制十折训练曲线
        self._plot_cv_training_curves(fold_histories)

        # 7. 绘制十折箱线图
        self._plot_cv_boxplot(best_losses, best_rs)

        # 8. 打印缓存使用信息
        print(f"\n📦 缓存信息:")
        print(f"   缓存目录: {shared_cache_dir}")
        print(f"   下次运行相同参数的实验将自动使用缓存")
        print(f"   如需强制重新加载，设置 force_reload=True")

        return cv_results, full_dataset 
    
    # ============================================================
    # [MODE] pretrain_progressive
    # ============================================================
    # 当前正式预训练推荐流程。
    #
    # [CONTRACT]
    #   Step 1: 运行 pretrain_cv，得到十折诊断结果。
    #   Step 2: 汇总十折 r / RMSE / MAE / loss。
    #   Step 3: 用该阶段100%样本训练 final_model.pth。
    #
    # [COMPAT]
    #   十折结果只用于诊断，不再用旧的 R²/NSE 阈值阻止最终训练。
    #   这样避免 AutoDL 后台运行时卡在 input()。
    #
    # [DANGER]
    #   不要再写：
    #       self.config["epochs"] = max(original_epochs, 100)
    #   否则脚本里 EPOCHS=90，最终阶段实际会偷偷跑 100。
    # ============================================================
    def run_pretrain_progressive_from_cv(self):
        """
        [MODE] 预训练渐进式策略（基于十折交叉验证）：
        1. 先跑完整的十折交叉验证，评估模型稳定性（诊断用，不阻止后续）
        2. 固定执行该阶段100%样本最终训练
        """
        print(f"\n{'█'*80}")
        print("🌟 预训练渐进式策略（基于十折交叉验证）")
        print("   Step 1: 运行十折交叉验证，评估模型稳定性")
        print("   Step 2: 分析十折结果（诊断用，不阻止后续）")
        print("   Step 3: 用当前累计池100%样本从头训练正式模型（复用数据集）")
        print(f"{'█'*80}")

        # ============ Step 1: 运行十折交叉验证，同时获取数据集 ============
        print(f"\n{'='*60}")
        print("📊 Step 1: 运行十折交叉验证")
        print(f"{'='*60}")

        # 🔥 修改1：接收返回的数据集
        cv_results, full_dataset = self.run_pretrain_cv_workflow()

        if cv_results is None:
            print("❌ 十折交叉验证失败")
            return None

        # 提取十折结果
        fold_r = [m.get('best_val_r', m.get('best_val_correlation', m.get('best_val_corr', np.nan)))
                  for m in cv_results.get('fold_metrics', [])]
        fold_r = [x for x in fold_r if np.isfinite(x)]

        if fold_r:
            mean_r = float(np.mean(fold_r))
            std_r = float(np.std(fold_r))
        else:
            mean_r = float('nan')
            std_r = float('nan')

        fold_loss = [m['best_val_loss'] for m in cv_results.get('fold_metrics', [])]
        mean_loss = np.mean(fold_loss) if fold_loss else float('nan')

        # ============ Step 2: 决策 ============
        print(f"\n{'='*60}")
        print("📊 Step 2: 十折结果诊断")
        print(f"{'='*60}")

        # 注意：
        # 这里不再用 NSE/R² 阈值阻止最终训练。
        # 十折交叉验证只用于评估稳定性；
        # 后续仍然固定执行当前阶段100%样本最终训练。
        if fold_r:
            print(f"   十折验证 r: {mean_r:.4f} ± {std_r:.4f}")
            print(f"   r 范围: [{min(fold_r):.4f}, {max(fold_r):.4f}]")
        else:
            print("   ⚠ 未找到 best_val_r，继续执行当前阶段全样本最终训练")

        print("   十折结果仅作为诊断，不阻止当前阶段100%样本最终训练。")
        auto_continue = True

        if not auto_continue:
            print(f"\n❌ 停止训练")
            return cv_results

        # ============ Step 3: 最终模型训练（参数化比例）============
        final_train_ratio = float(self.config.get('final_train_ratio', 1.0))
        final_epochs_mode = str(self.config.get('final_epochs_mode', 'fixed'))
        final_epochs = int(self.config.get('final_epochs', 100))
        final_scheduler = str(self.config.get('final_scheduler', 'cosine'))
        is_full_refit = (final_train_ratio >= 1.0)

        # 确定最终训练轮数
        if final_epochs_mode == 'cv_median':
            best_epochs = []
            for fm in cv_results.get('fold_metrics', []):
                be = fm.get('best_epoch')
                if be is not None:
                    best_epochs.append(int(be))
            if best_epochs:
                final_epochs = int(np.median(best_epochs))
                print(f"   cv_median: 十折最佳轮次中位数 = {final_epochs} (原始值: {best_epochs})")
            else:
                print(f"   ⚠ cv_median 模式未找到 best_epoch，回退到 fixed={final_epochs}")

        ratio_pct = final_train_ratio * 100
        print(f"\n{'='*60}")
        print(f"📊 Step 3: 最终模型训练（{ratio_pct:.0f}% 样本）")
        print(f"   训练轮数: {final_epochs} ({final_epochs_mode})")
        print(f"   调度器:   {final_scheduler}")
        print(f"   全量refit: {'是' if is_full_refit else '否'}")
        print(f"{'='*60}")

        # 🔥 直接使用十折传进来的 full_dataset，不重新创建！
        print(f"\n📊 使用十折已有的数据集（复用内存缓存）")
        print(f"   数据集样本数: {len(full_dataset):,}")
        print(f"   卷积特征维度: {full_dataset.C_conv}")
        print(f"   点特征维度: {full_dataset.C_point}")

        # 检查内存缓存是否存在
        if hasattr(full_dataset, '_cached_conv') and full_dataset._cached_conv is not None:
            print(f"   ✅ 内存缓存命中: {len(full_dataset._cached_conv):,} 个样本已预计算")
        else:
            print(f"   ℹ 未启用全样本内存预计算，训练时按需读取样本")

        # 同步维度信息到 config
        self.config["C_conv"] = full_dataset.C_conv
        self.config["C_point"] = full_dataset.C_point
        self._bind_swe_scale_from_dataset(full_dataset, context="pretrain_progressive_final")

        total_samples = len(full_dataset)
        indices = np.arange(total_samples)

        num_workers = int(self.config.get("num_workers", 4))
        batch_size = self.config['batch_size']

        if is_full_refit:
            # 100% 全样本训练，无验证集
            train_idx = indices
            val_idx = np.array([], dtype=np.int64)

            print(f"   训练样本: {len(train_idx):,} ({len(train_idx)/total_samples*100:.1f}%) — 全量")
            print(f"   验证样本: 0 — 全量 refit 模式，不设验证集")

            # drop_last: 当最后一个batch只有1个样本时drop，否则保留
            _drop_last = (total_samples % batch_size == 1)

            self.train_loader = DataLoader(
                full_dataset,  # 直接用 full_dataset，不需要 Subset
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=_drop_last,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )
            self.val_loader = None  # 全量 refit 无验证集
        else:
            # 保留部分验证集（如 95% 训练 + 5% 验证）
            test_size = 1.0 - final_train_ratio
            train_idx, val_idx = train_test_split(
                indices, test_size=test_size, random_state=self.config.get('seed', 42)
            )

            print(f"   训练样本: {len(train_idx):,} ({len(train_idx)/total_samples*100:.1f}%)")
            print(f"   验证样本: {len(val_idx):,} ({len(val_idx)/total_samples*100:.1f}%)")

            self.train_loader = DataLoader(
                Subset(full_dataset, train_idx),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )

            self.val_loader = DataLoader(
                Subset(full_dataset, val_idx),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )

        # [CUMULATIVE-SCRATCH] 最终100% refit也必须重新随机初始化，
        # 不从任一CV折、Stage 0或上一累计规模继续。
        is_from_scratch = bool(self.config.get("from_scratch", False))
        pretrain_init_model = (
            None if is_from_scratch else self.config.get("pretrained_model")
        )
        print(f"\n🏗️ 构建全量训练模型...")
        if pretrain_init_model:
            print(f"   初始化权重: {pretrain_init_model}")
        else:
            print("   初始化权重: 随机初始化（累计池 scratch）")

        # 将 final_scheduler 传入 config，供 build_model 使用
        self.config["_final_scheduler"] = final_scheduler
        self.config["_final_epochs"] = final_epochs
        self.config["_is_full_refit"] = is_full_refit

        success = self.build_model(
            load_pretrained=pretrain_init_model,
            freeze_backbone=False,
            is_cv_fold=False
        )

        if not success:
            print(f"❌ 模型构建失败")
            return cv_results

        # 全量训练
        print(f"\n🚀 开始最终模型训练...")

        original_epochs = self.config.get('epochs', 100)
        self.config['epochs'] = final_epochs

        train_result = self.train(
            fine_tune_mode=False,
            is_cv_sub_run=False,
            is_full_refit=is_full_refit,
        )

        self.config['epochs'] = original_epochs

        # 保存最终模型
        final_model_path = self.save_dir / "final_model.pth"
        if is_full_refit:
            print(f"\n💾 最终预训练模型: {final_model_path} (第 {final_epochs} 轮)")
        else:
            print(f"\n💾 最终预训练模型已由 train() 保存为: {final_model_path}")

        # 保存结果汇总
        final_result = {
            'cv_results': {
                'mean_r': float(mean_r),
                'std_r': float(std_r),
                'fold_r': [float(x) for x in fold_r],
                'fold_loss': [float(x) for x in fold_loss],
            },
            'full_training': {
                'n_train_samples': int(len(train_idx)),
                'n_val_samples': int(len(val_idx)),
                'total_samples': int(total_samples),
                'train_ratio': float(final_train_ratio),
                'epochs_mode': final_epochs_mode,
                'epochs': int(final_epochs),
                'scheduler': final_scheduler,
                'is_full_refit': is_full_refit,
                'final_train_loss': (
                    float(self.train_history[-1]) if self.train_history else None
                ),
                'last_val_loss': (
                    None if is_full_refit
                    else float(self.val_history[-1]) if self.val_history else None
                ),
                'last_val_r': (
                    None if is_full_refit
                    else float(self.val_history_metrics[-1].get('correlation'))
                    if hasattr(self, 'val_history_metrics') and self.val_history_metrics
                    else None
                ),
            },
            'model_path': str(final_model_path),
        }

        final_path = self.save_dir / "pretrain_progressive_results.json"
        with open(final_path, 'w') as f:
            json.dump(final_result, f, indent=2)

        # 绘制最终模型验证集散点图（仅当有验证集时）
        if self.val_loader is not None:
            print(f"\n📊 绘制最终模型验证集散点图...")
            self.model.eval()
            preds, targets, _ = self._make_predictions(self.val_loader)

            if preds is not None and len(preds) > 0:
                s_min, s_max = self._require_swe_scale(context="pretrain_progressive_final_plot")
                preds_denorm = preds * (s_max - s_min) + s_min
                targets_denorm = targets * (s_max - s_min) + s_min

                self.plot_density_scatter_hardcode(
                    preds_denorm, targets_denorm,
                    is_fine_tune=False,
                    fold_index="final_full_training"
                )
        else:
            print(f"\n📊 全量 refit 模式，无验证集，跳过散点图")

        print(f"\n{'█'*80}")
        print("✅ 预训练渐进式训练完成！")
        print(f"   十折 r: {mean_r:.4f} ± {std_r:.4f}")
        print(f"   最终模型: {final_model_path}")
        print(f"   结果保存: {final_path}")
        print(f"{'█'*80}")

        return final_result


    def _plot_cv_training_curves(self, fold_histories):
        """绘制十折的训练/验证损失曲线"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 左图：损失曲线
            ax1 = axes[0]
            for i, history in enumerate(fold_histories):
                epochs = range(1, len(history['train_losses']) + 1)
                ax1.plot(epochs, history['train_losses'], 'b-', alpha=0.3, linewidth=0.5)
                ax1.plot(epochs, history['val_losses'], 'r-', alpha=0.3, linewidth=0.5)

            # 绘制平均曲线
            max_epochs = max(len(h['train_losses']) for h in fold_histories)
            avg_train = []
            avg_val = []
            for e in range(max_epochs):
                train_vals = [h['train_losses'][e] for h in fold_histories if e < len(h['train_losses'])]
                val_vals = [h['val_losses'][e] for h in fold_histories if e < len(h['val_losses'])]
                avg_train.append(np.mean(train_vals))
                avg_val.append(np.mean(val_vals))

            ax1.plot(range(1, len(avg_train) + 1), avg_train, 'b-', linewidth=2, label='平均训练损失')
            ax1.plot(range(1, len(avg_val) + 1), avg_val, 'r-', linewidth=2, label='平均验证损失')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('十折交叉验证 - 损失曲线')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 右图：r 曲线
            ax2 = axes[1]
            for i, history in enumerate(fold_histories):
                epochs = range(1, len(history['val_r']) + 1)
                ax2.plot(epochs, history['val_r'], 'g-', alpha=0.3, linewidth=0.5)

            # 平均 r 曲线
            avg_r = []
            for e in range(max_epochs):
                r_vals = [h['val_r'][e] for h in fold_histories if e < len(h['val_r'])]
                avg_r.append(np.mean(r_vals))

            ax2.plot(range(1, len(avg_r) + 1), avg_r, 'g-', linewidth=2, label='平均验证 r')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('r')
            ax2.set_title('十折交叉验证 - r 曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.save_dir / "pretrain_cv_training_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 十折训练曲线已保存: {save_path}")

        except Exception as e:
            print(f"⚠ 绘制十折曲线失败: {e}")


    def _plot_cv_boxplot(self, best_losses, best_rs):
        """绘制十折指标的箱线图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 左图：损失箱线图
            ax1 = axes[0]
            bp1 = ax1.boxplot(best_losses, patch_artist=True)
            bp1['boxes'][0].set_facecolor('lightcoral')
            ax1.set_ylabel('Validation Loss')
            ax1.set_title('十折验证损失分布')
            ax1.grid(True, alpha=0.3)

            # 添加均值标注
            ax1.scatter(1, np.mean(best_losses), color='blue', s=50, zorder=5, 
                       label=f'Mean: {np.mean(best_losses):.4f}')
            ax1.legend()

            # 右图：r 箱线图
            ax2 = axes[1]
            bp2 = ax2.boxplot(best_rs, patch_artist=True)
            bp2['boxes'][0].set_facecolor('lightgreen')
            ax2.set_ylabel('r')
            ax2.set_title('十折验证 r 分布')
            ax2.grid(True, alpha=0.3)
            ax2.scatter(1, np.mean(best_rs), color='blue', s=50, zorder=5,
                       label=f'Mean: {np.mean(best_rs):.4f}')
            ax2.legend()

            plt.tight_layout()
            save_path = self.save_dir / "pretrain_cv_boxplot.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 十折箱线图已保存: {save_path}")

        except Exception as e:
            print(f"⚠ 绘制箱线图失败: {e}")
    
    def run_cv_workflow(self, freeze_strategy=None):
        """
        🚀 自动化十折交叉验证 - 支持从头训练模式

        Args:
            freeze_strategy: 冻结策略，如果为None则从config读取
        """

        print(f"\n{'█'*80}\n🌟 启动交叉验证: [随机划分 Train/Val] + [独立站点 Test]\n{'█'*80}")

        # --- 1. 确定参与 CV 划分的样本池 ---
        # CV_FULL_6936_POOL_FIX_V1
        #
        # split!='test' 的完整训练池已经由 load_data() 保存到
        # self.cv_pool_indices_override。
        #
        # 不再使用旧 DataLoader 的 train_indices + val_indices，
        # 因为旧 DataLoader 还含有一个临时 test 子集，会漏掉525条样本。
        base_ds = (
            self.train_loader.dataset.dataset
            if hasattr(self.train_loader.dataset, 'dataset')
            else self.train_loader.dataset
        )

        cv_pool_override = getattr(
            self,
            'cv_pool_indices_override',
            None
        )
        pretrain_aux_override = getattr(
            self,
            'pretrain_aux_indices_override',
            None
        )

        if cv_pool_override is not None:
            # 去重但保留原有顺序
            real_station_indices = np.asarray(
                list(dict.fromkeys(int(i) for i in cv_pool_override)),
                dtype=np.int64
            )

            pretrain_aux_indices = np.asarray(
                list(dict.fromkeys(
                    int(i) for i in (pretrain_aux_override or [])
                )),
                dtype=np.int64
            )

            print(
                f"✅ CV使用完整训练池 override: "
                f"{len(real_station_indices)} 条"
            )

        else:
            # 兼容没有设置override的旧数据流程
            train_idx = (
                list(self.train_loader.dataset.indices)
                if hasattr(self.train_loader.dataset, 'indices')
                else []
            )
            val_idx = (
                list(self.val_loader.dataset.indices)
                if hasattr(self.val_loader.dataset, 'indices')
                else []
            )

            cv_pool_indices = np.asarray(
                train_idx + val_idx,
                dtype=np.int64
            )

            meta_source = (
                base_ds.station_dataset.meta_index
                if hasattr(base_ds, 'station_dataset')
                else base_ds.meta_index
            )
            num_station_samples = len(meta_source)

            real_station_indices = cv_pool_indices[
                cv_pool_indices < num_station_samples
            ]
            pretrain_aux_indices = cv_pool_indices[
                cv_pool_indices >= num_station_samples
            ]

            print(
                "⚠ 未找到cv_pool_indices_override，"
                "退回旧的train+val拼接方式"
            )

        if len(real_station_indices) == 0:
            raise RuntimeError("CV训练池为空")

        # 当前纯站点模式下应严格等于6936条；
        # 也利用splits_info自动检查，避免以后再次静默漏样本。
        expected_cv_samples = None
        if hasattr(self, 'splits_info') and isinstance(self.splits_info, dict):
            expected_cv_samples = self.splits_info.get(
                'train_pool_samples'
            )

        if expected_cv_samples is not None:
            expected_cv_samples = int(expected_cv_samples)

            if len(real_station_indices) != expected_cv_samples:
                raise RuntimeError(
                    "CV池样本数量不一致: "
                    f"实际={len(real_station_indices)}, "
                    f"预期={expected_cv_samples}"
                )

            print(
                f"✅ CV完整性检查通过: "
                f"{len(real_station_indices)}/{expected_cv_samples}"
            )

        print(f"📊 CV池统计:")
        print(f"   ├─ 待随机划分的站点样本: {len(real_station_indices)}")
        print(f"   ├─ 固定作为训练辅助的预训练样本: {len(pretrain_aux_indices)}")
        print(f"   └─ 🔒 始终独立的测试集站点样本: {len(self.test_loader.dataset)}")

        # ============ 预训练模型路径配置（支持从头训练） ============
        is_from_scratch = self.config.get('from_scratch', False)
        pretrained_path = self.config.get('pretrained_model')

        if is_from_scratch:
            print("\n⚠️ 模式确认：[从头训练 Baseline] - 将跳过预训练权重，进行全参数随机初始化")
            pretrained_to_load = None
        elif pretrained_path and os.path.exists(pretrained_path):
            print(f"\n✅ 模式确认：[微调训练] - 将加载预训练模型: {pretrained_path}")
            pretrained_to_load = pretrained_path
        else:
            # 尝试自动查找默认路径
            default_path = "/root/autodl-tmp/experiments/swe_full_temporal_20260526_090443/pretrain_cv_fold1_best.pth"
            if os.path.exists(default_path):
                print(f"\n✅ 模式确认：[微调训练] - 使用默认预训练模型: {default_path}")
                pretrained_to_load = default_path
            else:
                print("\n⚠️ 模式确认：[从头训练] - 未找到预训练模型")
                pretrained_to_load = None

        freeze_backbone = self.config.get('freeze_backbone', True)

        # 🔥 关键修改：使用传入的 freeze_strategy 参数
        if freeze_strategy is None:
            freeze_strategy = self.config.get('freeze_strategy', 'fusion_ft')

        use_residual_injection = self.config.get('residual_injection', False)

        print(f"\n📋 CV 模型配置:")
        print(f"  预训练模型: {pretrained_to_load if pretrained_to_load else '无（从头训练）'}")
        print(f"  冻结主干: {freeze_backbone}")
        print(f"  冻结策略: {freeze_strategy}")
        print(f"  残差注入: {use_residual_injection}")

        # --- 3. 初始化容器 ---
        kf = KFold(n_splits=10, shuffle=True, random_state=self.config.get('seed', 42))
        all_fold_scatter_data = []
        all_fold_metrics = []

        # --- 4. 十折大循环 ---
        for fold, (t_idx_in_cv, v_idx_in_cv) in enumerate(kf.split(real_station_indices)):
            fold_idx = fold + 1
            self.current_fold = fold_idx
            self.config["freeze_strategy"] = freeze_strategy
            print(f"\n\n{'='*25} 🟢 FOLD {fold_idx} / 10 {'='*25}")

            # 4.1 随机分配样本索引
            f_train_station = real_station_indices[t_idx_in_cv].tolist()
            f_val_station = real_station_indices[v_idx_in_cv].tolist()

            f_train_total = f_train_station + pretrain_aux_indices.tolist()
            f_val_total = f_val_station

            self.train_loader = DataLoader(
                Subset(base_ds, f_train_total), 
                batch_size=self.config['batch_size'], 
                shuffle=True,
                num_workers=self.config.get('num_workers', 10),
                pin_memory=True,
                drop_last=True
            )
            self.val_loader = DataLoader(
                Subset(base_ds, f_val_total), 
                batch_size=self.config['batch_size'], 
                shuffle=False,
                num_workers=self.config.get('num_workers', 10),
                pin_memory=True
            )

            # 检查数据集独立性
            print(f"\n{'='*50}")
            print(f"🔍 [FOLD {fold_idx}] 验证数据集站点独立性")
            print(f"{'='*50}")
            self._check_fold_independence(fold_idx)

            # ============ 构建模型 ============
            print(f"\n🏗️ [FOLD {fold_idx}] 构建模型...")
            print(f"   使用冻结策略: {freeze_strategy}")

            success = self.build_model(
                load_pretrained=pretrained_to_load,
                freeze_backbone=freeze_backbone,
                freeze_strategy=freeze_strategy,
                use_residual=use_residual_injection,
                is_cv_fold=True
            )

            if not success:
                print(f"❌ [FOLD {fold_idx}] 模型构建失败，跳过此折")
                continue

            # ============ 验证冻结状态 ============
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())

            print(f"\n  📊 [FOLD {fold_idx}] build_model 后的冻结统计:")
            print(f"    总参数: {total_params:,}")
            print(f"    可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

            # 🔥 如果没有可训练参数，直接报错
            current_trainable_params = [p for p in self.model.parameters() if p.requires_grad]

            if not current_trainable_params:
                print(f"\n  ❌❌❌ 致命错误 [FOLD {fold_idx}] ❌❌❌")
                print(f"     没有可训练参数！")
                print(f"     请检查 freeze_strategy='{freeze_strategy}' 的关键词是否匹配")

                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i < 20:
                        print(f"       {name}: {param.requires_grad}")
                    else:
                        break

                raise RuntimeError(f"Fold {fold_idx}: 没有可训练参数！")

            # ============ 权重质量检测 ============
            if pretrained_to_load is not None:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if "spatial_encoder" in name and "weight" in name:
                            weight_val = param.mean().item()
                            weight_std = param.std().item()
                            print(f"\n  🧪 [权重质量检测] 层: {name}")
                            print(f"     均值: {weight_val:.6f}, 标准差: {weight_std:.6f}")

                            if abs(weight_val) < 1e-3 and abs(weight_std - 0.02) < 0.01:
                                print(f"     ❌ 警告：权重接近随机初始化！预训练权重可能未加载")
                            else:
                                print(f"     ✅ 权重分布正常，预训练权重已加载")
                            break

            # ============ 重新绑定优化器 ============
            lr = self.config.get("fine_tune_lr", 5e-5)
            self.optimizer = optim.AdamW(
                current_trainable_params, 
                lr=lr,
                weight_decay=1e-4
            )

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
            )

            print(f"    🚀 优化器已重新绑定 {len(current_trainable_params)} 组参数, lr={lr:.2e}")

            # ============ 执行训练 ============
            print(f"\n🚀 [FOLD {fold_idx}] 开始训练...")
            train_result = self.train(fine_tune_mode=True, is_cv_sub_run=True)

            # CV_LOAD_FOLD_BEST_BEFORE_TEST_V1
            # 每折训练结束后，必须加载本折选模得到的最佳检查点，
            # 不能直接使用最后一个epoch留在内存中的模型测试。
            from pathlib import Path as _CVPath
            import shutil as _cv_shutil

            best_checkpoint_path = (
                _CVPath(self.save_dir) / "best_fine_tuned_model.pth"
            )

            if not best_checkpoint_path.exists():
                raise FileNotFoundError(
                    f"[FOLD {fold_idx}] 找不到本折最佳模型: "
                    f"{best_checkpoint_path}"
                )

            try:
                best_checkpoint = torch.load(
                    best_checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                # 兼容不支持 weights_only 参数的旧版 PyTorch
                best_checkpoint = torch.load(
                    best_checkpoint_path,
                    map_location=self.device,
                )

            if isinstance(best_checkpoint, dict):
                if "model_state_dict" in best_checkpoint:
                    best_state_dict = best_checkpoint["model_state_dict"]
                elif "state_dict" in best_checkpoint:
                    best_state_dict = best_checkpoint["state_dict"]
                else:
                    # 兼容直接保存 state_dict 的检查点
                    best_state_dict = best_checkpoint
            else:
                raise TypeError(
                    f"[FOLD {fold_idx}] 检查点格式异常: "
                    f"{type(best_checkpoint)}"
                )

            self.model.load_state_dict(
                best_state_dict,
                strict=True,
            )
            self.model.to(self.device)
            self.model.eval()

            # 立即保存为折独立文件，下一折不能覆盖
            fold_best_path = (
                _CVPath(self.save_dir)
                / f"cv_fold_{fold_idx:02d}_best_model.pth"
            )
            _cv_shutil.copy2(
                best_checkpoint_path,
                fold_best_path,
            )

            best_epoch_raw = (
                best_checkpoint.get(
                    "epoch",
                    best_checkpoint.get("best_epoch", "unknown"),
                )
                if isinstance(best_checkpoint, dict)
                else "unknown"
            )

            print(
                f"✅ [FOLD {fold_idx}] 已加载本折最佳检查点用于固定测试"
            )
            print(f"   最佳检查点: {best_checkpoint_path}")
            print(f"   折独立备份: {fold_best_path}")
            print(f"   checkpoint epoch(raw): {best_epoch_raw}")

            if train_result:
                print(f"  ✅ [FOLD {fold_idx}] 训练完成")
            else:
                print(f"  ⚠️ [FOLD {fold_idx}] 训练可能未完成")

            # CV_DEFER_FIXED_TEST_TO_STAGE_RUNNER_V1
            # 本函数只负责当前fold的训练、验证选模和最佳模型保存。
            # 固定内部1000条及外部987条不在CV训练循环中测试，
            # 统一交给阶段脚本的 internal_test_10fold /
            # external_test_10fold 逐折评价和汇总。
            print(
                f"\n📦 [FOLD {fold_idx}] 本折训练和验证选模完成；"
                "固定测试延后统一执行"
            )
            print(
                "   当前阶段仅保留 cv_fold_XX_best_model.pth，"
                "不生成重复测试指标和散点图"
            )

            # ============ 显存回收 ============
            print(f"\n🧹 [FOLD {fold_idx}] 清理显存...")
            del self.model
            del self.optimizer
            del self.scheduler
            torch.cuda.empty_cache()
            gc.collect()

            # SERIAL_FOLD_EVAL_OWNED_BY_RUNNER_V1
            # 不在训练进程内启动评估子进程，避免v4约16.8 GB缓存同时驻留两份。
            # run_progressive_finetune_stage.sh会在十折训练进程完全退出后，
            # 再逐折串行运行内部1000条和外部987条评估。

            print(f"\n{'='*50}")
            print(f"✅ FOLD {fold_idx} / 10 完成")
            print(f"{'='*50}")

        # --- 5. 汇总报告 ---
        print(f"\n\n{'█'*80}")
        print(f"🏆 十折交叉验证完成！")
        print(f"{'█'*80}")

        if all_fold_scatter_data:
            self.plot_cv_panel_figure(all_fold_scatter_data)

        if all_fold_metrics:
            self.plot_cv_metrics_boxplot(all_fold_metrics)
            self._final_cv_report(all_fold_metrics)

        return {'all_metrics': all_fold_metrics, 'all_scatter_data': all_fold_scatter_data}
        
        
    def run_station_full_cv(self):
        """
        按站点的全样本十折交叉验证
        - 每折按站点划分（10%的站点做测试集）
        - 10折覆盖全部站点
        - 训练集内再随机划分训练/验证（8:2）
        - 收集所有测试集预测，最终计算全样本精度
        """


        print(f"\n{'█'*80}")
        print("🌟 按站点全样本十折交叉验证")
        print("   特点：每折10%站点做测试集，10折覆盖全部站点")
        print("   每个样本都会被预测一次，最终计算全样本精度")
        print(f"{'█'*80}")

        # ============ 1. 获取基础数据集 ============
        # 获取底层的 station_dataset
        if hasattr(self.train_loader, 'dataset'):
            current_ds = self.train_loader.dataset
            # 剥壳找到 station_dataset
            while hasattr(current_ds, 'dataset') and not hasattr(current_ds, 'station_dataset'):
                current_ds = current_ds.dataset

            if hasattr(current_ds, 'station_dataset'):
                station_ds = current_ds.station_dataset
            else:
                station_ds = current_ds
        else:
            station_ds = self.train_loader.dataset

        print(f"\n📊 数据集信息:")
        print(f"  站点数据集样本数: {len(station_ds)}")
        print(f"  预训练数据集: {'有' if hasattr(current_ds, 'pretrain_dataset') else '无'}")

        # ============ 2. 按站点分组 ============
        print(f"\n📊 正在按站点分组...")

        station_to_indices = defaultdict(list)
        for idx in range(len(station_ds)):
            meta = station_ds.meta_index[idx]
            station_id = meta['station_id']
            # 处理多站点情况（取第一个）
            if ',' in str(station_id):
                station_id = str(station_id).split(',')[0]
            station_to_indices[station_id].append(idx)

        unique_stations = list(station_to_indices.keys())
        n_stations = len(unique_stations)
        total_samples = len(station_ds)

        print(f"  唯一站点数: {n_stations}")
        print(f"  站点样本总数: {total_samples}")

        samples_per_station = [len(station_to_indices[s]) for s in unique_stations]
        print(f"  每站点样本数: min={min(samples_per_station)}, max={max(samples_per_station)}, "
              f"mean={np.mean(samples_per_station):.1f}")

        # ============ 3. 预训练模型路径 ============
        is_from_scratch = self.config.get('from_scratch', False)
        pretrained_path = self.config.get('pretrained_model')

        if is_from_scratch:
            print(f"\n⚠️ 从头训练模式 - 不加载预训练权重")
            pretrained_to_load = None
        elif pretrained_path and os.path.exists(pretrained_path):
            print(f"\n✅ 使用预训练模型: {pretrained_path}")
            pretrained_to_load = pretrained_path
        else:
            # 尝试自动查找
            default_paths = [
                "/root/autodl-tmp/experiments/swe_full_temporal_20260526_090443/pretrain_cv_fold1_best.pth",
                "/root/ablation/final_model.pth",
            ]
            pretrained_to_load = None
            for path in default_paths:
                if os.path.exists(path):
                    print(f"\n✅ 自动找到预训练模型: {path}")
                    pretrained_to_load = path
                    break

            if pretrained_to_load is None:
                print(f"\n⚠️ 未找到预训练模型，从头训练")

        freeze_backbone = self.config.get('freeze_backbone', True)
        freeze_strategy = self.config.get('freeze_strategy', 'partial')
        use_residual_injection = self.config.get('residual_injection', False)

        print(f"\n📋 CV 模型配置:")
        print(f"  预训练模型: {pretrained_to_load if pretrained_to_load else '无（从头训练）'}")
        print(f"  冻结主干: {freeze_backbone}")
        print(f"  冻结策略: {freeze_strategy}")

        # ============ 4. 按站点进行10折划分 ============
        kf = KFold(n_splits=10, shuffle=True, random_state=self.config.get('seed', 42))

        # 存储所有折的结果
        all_fold_metrics = []
        all_test_predictions = []   # 收集所有测试集预测
        self._test_meta_list = []     # 测试样本元数据
        all_test_targets = []       # 收集所有测试集真实值
        all_test_station_ids = []   # 收集站点ID用于分析

        fold_splits = []

        for fold, (train_station_idx, test_station_idx) in enumerate(kf.split(unique_stations)):
            fold_num = fold + 1

            train_stations = [unique_stations[i] for i in train_station_idx]
            test_stations = [unique_stations[i] for i in test_station_idx]

            # 收集索引
            train_indices = []
            test_indices = []

            for sid in train_stations:
                train_indices.extend(station_to_indices[sid])
            for sid in test_stations:
                test_indices.extend(station_to_indices[sid])

            # 从训练集中再划分验证集（8:2）
            train_indices, val_indices = train_test_split(
                train_indices, 
                test_size=0.2, 
                random_state=self.config.get('seed', 42) + fold_num
            )

            fold_splits.append({
                'fold': fold_num,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'train_stations': train_stations,
                'val_stations': [],  # 验证集站点不单独记录
                'test_stations': test_stations,
                'n_train_samples': len(train_indices),
                'n_val_samples': len(val_indices),
                'n_test_samples': len(test_indices),
                'n_train_stations': len(train_stations),
                'n_test_stations': len(test_stations),
            })

        # 打印划分统计
        print(f"\n{'='*70}")
        print(f"{'Fold':<6} {'测试站点':<10} {'测试样本':<10} {'训练站点':<10} {'训练样本':<10} {'验证样本':<10}")
        print(f"{'-'*60}")
        for split in fold_splits:
            print(f"{split['fold']:<6} {split['n_test_stations']:<10} {split['n_test_samples']:<10} "
                  f"{split['n_train_stations']:<10} {split['n_train_samples']:<10} {split['n_val_samples']:<10}")

        # ============ 5. 十折大循环 ============
        for split in fold_splits:
            fold_idx = split['fold']

            print(f"\n\n{'='*60}")
            print(f"🟢 FOLD {fold_idx} / 10")
            print(f"  测试站点: {split['n_test_stations']} 个, 样本: {split['n_test_samples']}")
            print(f"  训练站点: {split['n_train_stations']} 个, 样本: {split['n_train_samples']}")
            print(f"  验证样本: {split['n_val_samples']} 个（从训练集随机划分）")
            print(f"{'='*60}")

            # 5.1 创建本折的 DataLoader
            # 训练集
            train_subset = Subset(station_ds, split['train_indices'])
            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True,
                drop_last=True
            )

            # 验证集
            val_subset = Subset(station_ds, split['val_indices'])
            self.val_loader = DataLoader(
                val_subset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True
            )

            # 测试集
            test_subset = Subset(station_ds, split['test_indices'])
            self.test_loader = DataLoader(
                test_subset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 8),
                pin_memory=True
            )

            # 5.2 构建模型
            print(f"\n🏗️ [FOLD {fold_idx}] 构建模型...")

            success = self.build_model(
                load_pretrained=pretrained_to_load,
                freeze_backbone=freeze_backbone,
                freeze_strategy=freeze_strategy,
                use_residual=use_residual_injection,
                is_cv_fold=True
            )

            if not success:
                print(f"❌ [FOLD {fold_idx}] 模型构建失败，跳过")
                continue

            # 5.3 重新绑定优化器
            current_trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not current_trainable_params:
                print(f"❌ [FOLD {fold_idx}] 没有可训练参数，跳过")
                continue

            lr = self.config.get("fine_tune_lr", 5e-5)
            self.optimizer = optim.AdamW(
                current_trainable_params,
                lr=lr,
                weight_decay=1e-4
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
            )

            # 5.4 执行训练
            print(f"\n🚀 [FOLD {fold_idx}] 开始训练...")

            # 临时调整训练参数
            original_epochs = self.config.get('epochs', 100)
            original_patience = self.config.get('patience', 25)
            self.config['epochs'] = min(original_epochs, 50)
            self.config['patience'] = 10

            train_result = self.train(fine_tune_mode=True, is_cv_sub_run=True)

            self.config['epochs'] = original_epochs
            self.config['patience'] = original_patience

            # 5.5 评估测试集
            print(f"\n📊 [FOLD {fold_idx}] 评估测试集...")

            self.model.eval()

            # ============ Prior Ablation（仅显式配置先验列时运行） ============
            prior_col = self.config.get("physical_prior_col", None)
            c_point = int(self.config.get("C_point", 0) or 0)
            if (
                self.config.get("enable_prior_ablation", False)
                and isinstance(prior_col, int)
                and 0 <= prior_col < c_point
            ):
                print(f"\n🔬 [FOLD {fold_idx}] 开始 Prior Ablation 诊断...")
                self._run_prior_ablation_diagnosis(
                    dataloader=self.test_loader,
                    split_name="test",
                    fold_idx=fold_idx,
                )
            else:
                print(f"ℹ [FOLD {fold_idx}] Clean-{c_point}D 未配置 prior 列，跳过消融")
            # =====================================================

            preds, targets, is_zero = self._make_predictions(self.test_loader)

            if preds is not None and len(preds) > 0:
                # 反归一化
                s_min = getattr(self, 'swe_min', 0.0)
                s_max = getattr(self, 'swe_max', 200.0)
                preds_denorm = preds * (s_max - s_min) + s_min
                targets_denorm = targets * (s_max - s_min) + s_min

                # 收集到全样本列表
                all_test_predictions.extend(preds_denorm)
                all_test_targets.extend(targets_denorm)

                # 记录站点ID和元数据（用于坏样本分析）
                for i, idx in enumerate(split['test_indices']):
                    meta = station_ds.meta_index[idx]
                    station_id = meta.get('station_id', 'unknown')
                    if ',' in str(station_id):
                        station_id = str(station_id).split(',')[0]
                    all_test_station_ids.append(station_id)

                    # 收集元数据用于坏样本分析
                    if not hasattr(self, '_test_meta_list'):
                        self._test_meta_list = []
                    self._test_meta_list.append({
                        'station_id': str(station_id),
                        'date': str(meta.get('date', '')),
                        'obs': float(targets_denorm[i]) if i < len(targets_denorm) else np.nan,
                        'pred': float(preds_denorm[i]) if i < len(preds_denorm) else np.nan,
                        'fold_id': fold_idx,
                        'row': int(meta.get('row', -1)),
                        'col': int(meta.get('col', -1)),
                    })

                # 计算本折指标
                rmse = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))
                mae = np.mean(np.abs(preds_denorm - targets_denorm))
                bias = np.mean(preds_denorm - targets_denorm)

                ss_res = np.sum((preds_denorm - targets_denorm) ** 2)
                ss_tot = np.sum((targets_denorm - np.mean(targets_denorm)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                r, _ = stats.pearsonr(preds_denorm, targets_denorm)

                fold_metrics = {
                    'fold': fold_idx,
                    'r2': r2,
                    'r': r,
                    'rmse': rmse,
                    'mae': mae,
                    'bias': bias,
                    'n_samples': len(preds_denorm),
                    'n_test_stations': split['n_test_stations']
                }
                all_fold_metrics.append(fold_metrics)

                print(f"\n  📈 [FOLD {fold_idx}] 测试集评估:")
                print(f"    NSE: {r2:.4f}, RMSE: {rmse:.2f} mm, MAE: {mae:.2f} mm")

                # 绘制本折散点图
                self.plot_density_scatter_hardcode(
                    preds_denorm, targets_denorm,
                    is_fine_tune=True,
                    fold_index=fold_idx
                )
            else:
                print(f"  ⚠️ [FOLD {fold_idx}] 测试集预测失败")

            # 5.6 清理显存
            print(f"\n🧹 [FOLD {fold_idx}] 清理显存...")
            del self.model
            del self.optimizer
            del self.scheduler
            torch.cuda.empty_cache()
            gc.collect()

        # ============ 6. 计算全样本聚合精度 ============
        print(f"\n\n{'█'*80}")
        print("🏆 按站点全样本十折交叉验证完成！")
        print(f"{'█'*80}")

        if len(all_test_predictions) == 0:
            print("❌ 没有收集到任何测试结果")
            return None

        # 转换为 numpy 数组
        all_test_predictions = np.array(all_test_predictions)
        all_test_targets = np.array(all_test_targets)

        # 计算全样本指标
        final_rmse = np.sqrt(np.mean((all_test_predictions - all_test_targets) ** 2))
        final_mae = np.mean(np.abs(all_test_predictions - all_test_targets))
        final_bias = np.mean(all_test_predictions - all_test_targets)

        ss_res = np.sum((all_test_predictions - all_test_targets) ** 2)
        ss_tot = np.sum((all_test_targets - np.mean(all_test_targets)) ** 2)
        final_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        final_r, _ = stats.pearsonr(all_test_predictions, all_test_targets)

        print(f"\n{'='*70}")
        print(f"🎯 【全样本聚合精度】（{len(all_test_predictions)} 个样本，{len(set(all_test_station_ids))} 个站点）")
        print(f"{'='*70}")
        print(f"  r:    {final_r:.4f}")
        print(f"  RMSE: {final_rmse:.2f} mm")
        print(f"  MAE:  {final_mae:.2f} mm")
        print(f"  Bias: {final_bias:.2f} mm")
        print(f"{'='*70}")

        # 十折指标统计
        if all_fold_metrics:
            r_values = [m['r'] for m in all_fold_metrics]
            rmse_values = [m['rmse'] for m in all_fold_metrics]

            print(f"\n📊 十折指标统计:")
            print(f"  r:     {np.mean(r_values):.4f} ± {np.std(r_values):.4f}")
            print(f"  RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f} mm")

        # 绘制全样本散点图
        print(f"\n📊 绘制全样本聚合散点图...")
        self.plot_density_scatter_hardcode(
            all_test_predictions, all_test_targets,
            is_fine_tune=True,
            fold_index="full_sample_aggregated"
        )

        # 🔥 坏样本分析：obs>=80 且 pred<40
        self._analyze_bad_test_samples(all_test_predictions, all_test_targets)

        # 绘制十折指标箱线图
        if all_fold_metrics:
            self.plot_cv_metrics_boxplot(all_fold_metrics, save_name="station_full_cv_boxplot.png")

        # 保存结果
        results = {
            'cv_mode': 'station_full_cv',
            'n_folds': 10,
            'total_samples': len(all_test_predictions),
            'total_stations': len(set(all_test_station_ids)),
            'aggregated_metrics': {
                'r2': float(final_r2),
                'r': float(final_r),
                'rmse': float(final_rmse),
                'mae': float(final_mae),
                'bias': float(final_bias),
            },
            'fold_metrics': all_fold_metrics,
            'fold_stats': {
                'r2_mean': float(np.mean([m['r2'] for m in all_fold_metrics])),
                'r2_std': float(np.std([m['r2'] for m in all_fold_metrics])),
                'rmse_mean': float(np.mean([m['rmse'] for m in all_fold_metrics])),
                'rmse_std': float(np.std([m['rmse'] for m in all_fold_metrics])),
            }
        }

        results_path = self.save_dir / "station_full_cv_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 结果已保存: {results_path}")

        return results
        
        
    def _check_fold_independence(self, fold_idx):
        """检查当前折的训练集、验证集、测试集站点独立性"""

        def extract_stations(loader, name):
            stations = set()
            if loader is None:
                print(f"  {name}: loader 为 None")
                return stations

            try:
                dataset = loader.dataset
                if hasattr(dataset, 'dataset'):
                    base_ds = dataset.dataset
                    indices = dataset.indices
                else:
                    base_ds = dataset
                    indices = range(len(dataset))

                if hasattr(base_ds, 'station_dataset'):
                    station_ds = base_ds.station_dataset
                elif hasattr(base_ds, 'meta_index'):
                    station_ds = base_ds
                else:
                    print(f"  {name}: 无法获取站点数据集")
                    return stations

                for idx in indices:
                    if idx < len(station_ds.meta_index):
                        station_id = station_ds.meta_index[idx]['station_id']
                        if ',' in str(station_id):
                            for sid in str(station_id).split(','):
                                stations.add(sid.strip())
                        else:
                            stations.add(str(station_id))

                print(f"  {name}: {len(stations)} 个站点, {len(indices)} 个样本")

            except Exception as e:
                print(f"  {name}: 提取失败 - {e}")

            return stations

        print(f"\n📊 [Fold {fold_idx}] 站点统计:")

        train_stations = extract_stations(self.train_loader, "训练集")
        val_stations = extract_stations(self.val_loader, "验证集")
        test_stations = extract_stations(self.test_loader, "测试集")

        print(f"\n  总唯一站点数: {len(train_stations | val_stations | test_stations)}")

        # 检查重叠
        train_val_overlap = train_stations & val_stations
        train_test_overlap = train_stations & test_stations
        val_test_overlap = val_stations & test_stations

        print(f"\n🔍 站点重叠检查:")

        if train_val_overlap:
            print(f"  ⚠️ 训练集 ∩ 验证集: {len(train_val_overlap)} 个重叠站点")
            print(f"     重叠站点: {list(train_val_overlap)[:10]}")
        else:
            print(f"  ✅ 训练集 ∩ 验证集: 无重叠")

        if train_test_overlap:
            print(f"  ⚠️ 训练集 ∩ 测试集: {len(train_test_overlap)} 个重叠站点")
            print(f"     重叠站点: {list(train_test_overlap)[:10]}")
        else:
            print(f"  ✅ 训练集 ∩ 测试集: 无重叠")

        if val_test_overlap:
            print(f"  ⚠️ 验证集 ∩ 测试集: {len(val_test_overlap)} 个重叠站点")
            print(f"     重叠站点: {list(val_test_overlap)[:10]}")
        else:
            print(f"  ✅ 验证集 ∩ 测试集: 无重叠")

        is_independent = len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0

        if is_independent:
            print(f"\n✅ [Fold {fold_idx}] 数据集站点完全独立！")
        else:
            print(f"\n⚠️ [Fold {fold_idx}] 存在站点重叠，评估结果可能过于乐观！")

        return {
            'is_independent': is_independent,
            'train_val_overlap': len(train_val_overlap),
            'train_test_overlap': len(train_test_overlap),
            'val_test_overlap': len(val_test_overlap)
        }        
        
        
    def _final_cv_report(self, metrics_list):
        """输出最终的均值和标准差"""
        df = pd.DataFrame(metrics_list)
        print("\n" + "="*40)
        print("📁 十折交叉验证最终汇总报告")
        print("-"*40)
        for col in ['r2', 'rmse', 'mae', 'bias']:
            if col in df.columns:
                print(f"{col.upper():<5}: {df[col].mean():.4f} ± {df[col].std():.4f}")
        print("="*40 + "\n")
        
    def _save_fold_scatter_plot(self, fold_num):
        # 这里的预测逻辑复用你现有的 _make_predictions
        preds, targets, _ = self._make_predictions(self.test_loader)
        if preds is not None:
            # 反归一化
            s_min, s_max = getattr(self, 'swe_min', 0), getattr(self, 'swe_max', 200)
            p_denorm = preds * (s_max - s_min) + s_min
            t_denorm = targets * (s_max - s_min) + s_min
            # 传入折数作为后缀
            self.plot_density_scatter_hardcode(p_denorm, t_denorm, is_fine_tune=True, fold_suffix=str(fold_num))
    
    def save_checkpoint(self, filename, epoch, metrics):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "lr_history": self.lr_history,
            "fine_tune_history": self.fine_tune_history,
            "val_history_metrics": list(
                getattr(self, "val_history_metrics", [])
            ),
            "random_state": (
                self._capture_overfit_resume_random_state()
                if bool(
                    self.config.get("train_only_overfit", False)
                )
                else None
            ),
            "config": self.config,
            "metrics": metrics,
            "swe_min": float(self.swe_min) if hasattr(self, "swe_min") else None,
            "swe_max": float(self.swe_max) if hasattr(self, "swe_max") else None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path, pickle_protocol=4)
        print(f"✓ 检查点保存到: {save_path}")

    

    def plot_training_curves_with_r2(self, fine_tune_mode=False):
        """绘制训练曲线，包含R²评估 - 简化版本"""
        try:
            # 先调用原来的绘图方法
            self.plot_training_curves(fine_tune_mode)
            
            print(f"✓ 训练曲线已绘制 (包含R²评估)")
            
        except Exception as e:
            print(f"绘制包含R²的训练曲线失败: {e}")
            # 回退到原方法
            try:
                self.plot_training_curves(fine_tune_mode)
            except Exception as e2:
                print(f"连原绘图方法也失败: {e2}")

    def plot_gradient_monitoring(self, fine_tune_mode=False):
        """绘制梯度监控曲线"""
        try:
            if not self.gradient_history:
                print("没有梯度历史数据可绘制")
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. 梯度范数变化
            ax1 = axes[0, 0]
            epochs = [g['epoch'] for g in self.gradient_history]
            grad_means = [g['mean'] for g in self.gradient_history]
            grad_maxs = [g['max'] for g in self.gradient_history]
            grad_mins = [g['min'] for g in self.gradient_history]
            
            ax1.plot(epochs, grad_means, 'b-', label='平均梯度范数', linewidth=2)
            ax1.fill_between(epochs, grad_mins, grad_maxs, alpha=0.2, color='blue', label='梯度范围')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('梯度范数')
            ax1.set_title('梯度范数变化', fontsize=14, fontweight='bold')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 权重变化
            ax2 = axes[0, 1]
            if self.weight_change_history:
                weight_epochs = [w['epoch'] for w in self.weight_change_history]
                total_changes = [w['total_change'] for w in self.weight_change_history]
                
                ax2.plot(weight_epochs, total_changes, 'r-', label='总权重变化', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('权重变化')
                ax2.set_title('权重变化情况', fontsize=14, fontweight='bold')
                ax2.set_yscale('log')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. 参数数量统计
            ax3 = axes[1, 0]
            if self.gradient_history:
                param_counts = [g['num_params'] for g in self.gradient_history]
                
                ax3.plot(epochs, param_counts, 'g-', label='有梯度参数数量', linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('参数数量')
                ax3.set_title('有梯度参数数量变化', fontsize=14, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # 4. 梯度与损失关系
            ax4 = axes[1, 1]
            if len(self.train_history) == len(self.gradient_history):
                losses = self.train_history[-len(self.gradient_history):]
                grad_means = [g['mean'] for g in self.gradient_history]
                
                scatter = ax4.scatter(losses, grad_means, c=epochs, cmap='viridis', s=50)
                ax4.set_xlabel('训练损失')
                ax4.set_ylabel('平均梯度范数')
                ax4.set_title('梯度与损失关系', fontsize=14, fontweight='bold')
                ax4.set_xscale('log')
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3)
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Epoch')
            
            plt.suptitle('微调梯度监控', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存图像
            plot_path = self.save_dir / "gradient_monitoring.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"梯度监控曲线已保存到: {plot_path}")
            
            # 保存梯度统计到文件
            grad_stats_path = self.save_dir / "gradient_statistics.json"
            with open(grad_stats_path, 'w') as f:
                json.dump({
                    'gradient_history': self.gradient_history,
                    'weight_change_history': self.weight_change_history,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
                
            print(f"梯度统计数据已保存到: {grad_stats_path}")
            
        except Exception as e:
            print(f"绘制梯度监控曲线失败: {e}")
    
    def save_training_history(self, fine_tune_mode=False):
        """保存训练历史"""
        history = {
            "train_loss": self.train_history,
            "val_loss": self.val_history,
            "lr_history": self.lr_history,
            "fine_tune_history": self.fine_tune_history,
            "config": self.config,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        history_name = (
            "fine_tune_history.json" if fine_tune_mode else "training_history.json"
        )
        history_path = self.save_dir / history_name
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

        print(f"训练历史已保存到: {history_path}")

    def plot_training_curves(self, fine_tune_mode=False):
        """绘制训练曲线。全量refit时只画训练损失和学习率。"""
        try:
            is_full_refit = bool(self.config.get("_is_full_refit", False))
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. 损失曲线
            ax1 = axes[0, 0]

            if fine_tune_mode and self.fine_tune_history:
                # 微调曲线
                epochs = range(1, len(self.fine_tune_history) + 1)
                ax1.plot(
                    epochs, self.fine_tune_history, "g-", label="微调损失", linewidth=2
                )
                # 验证损失（微调阶段）
                val_epochs = range(
                    1, len(self.val_history[-len(self.fine_tune_history) :]) + 1
                )
                val_losses = self.val_history[-len(self.fine_tune_history) :]
                ax1.plot(val_epochs, val_losses, "r-", label="验证损失", linewidth=2)
            else:
                # 常规训练曲线
                epochs = range(1, len(self.train_history) + 1)
                ax1.plot(
                    epochs, self.train_history, "b-", label="训练损失", linewidth=2
                )
                if (not is_full_refit) and len(self.val_history) == len(self.train_history):
                    ax1.plot(epochs, self.val_history, "r-", label="验证损失", linewidth=2)

            ax1.set_xlabel("Epoch", fontsize=12)
            ax1.set_ylabel("Loss (MSE)", fontsize=12)
            if is_full_refit and not fine_tune_mode:
                title = "全量refit训练损失"
            else:
                title = "微调训练和验证损失" if fine_tune_mode else "训练和验证损失"
            ax1.set_title(title, fontsize=14, fontweight="bold")
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)

            # 标记最佳epoch
            if self.val_history:
                if fine_tune_mode and self.fine_tune_history:
                    val_losses = self.val_history[-len(self.fine_tune_history) :]
                else:
                    val_losses = self.val_history
                best_idx = np.argmin(val_losses)
                ax1.scatter(
                    best_idx + 1,
                    val_losses[best_idx],
                    color="red",
                    s=100,
                    zorder=5,
                    label=f"最佳 (Epoch {best_idx + 1})",
                )
                ax1.legend(fontsize=11)

            # 2. 学习率曲线
            ax2 = axes[0, 1]
            epochs = range(1, len(self.lr_history) + 1)
            ax2.plot(epochs, self.lr_history, "g-", linewidth=2)
            ax2.set_xlabel("Epoch", fontsize=12)
            ax2.set_ylabel("Learning Rate", fontsize=12)
            ax2.set_title("学习率变化", fontsize=14, fontweight="bold")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

            # 3. 损失对比（对数坐标）
            ax3 = axes[1, 0]
            if fine_tune_mode and self.fine_tune_history:
                epochs = range(1, len(self.fine_tune_history) + 1)
                ax3.plot(
                    epochs, self.fine_tune_history, "g-", label="微调", linewidth=2
                )
                val_epochs = range(
                    1, len(self.val_history[-len(self.fine_tune_history) :]) + 1
                )
                val_losses = self.val_history[-len(self.fine_tune_history) :]
                ax3.plot(val_epochs, val_losses, "r-", label="验证", linewidth=2)
            else:
                epochs = range(1, len(self.train_history) + 1)
                ax3.plot(epochs, self.train_history, "b-", label="训练", linewidth=2)
                if (not is_full_refit) and len(self.val_history) == len(self.train_history):
                    ax3.plot(epochs, self.val_history, "r-", label="验证", linewidth=2)

            ax3.set_xlabel("Epoch", fontsize=12)
            ax3.set_ylabel("Loss (log)", fontsize=12)
            title = "损失对比（对数坐标）" + (" - 微调" if fine_tune_mode else "")
            ax3.set_title(title, fontsize=14, fontweight="bold")
            ax3.set_yscale("log")
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)

            # 4. 损失比率
            ax4 = axes[1, 1]
            if (
                fine_tune_mode
                and self.fine_tune_history
                and len(self.val_history) >= len(self.fine_tune_history)
            ):
                val_losses = self.val_history[-len(self.fine_tune_history) :]
                overfit_ratio = [
                    v / t if t > 0 else 0
                    for t, v in zip(self.fine_tune_history, val_losses)
                ]
                ax4.plot(
                    range(1, len(overfit_ratio) + 1),
                    overfit_ratio,
                    "purple",
                    linewidth=2,
                )
            elif len(self.train_history) > 0 and len(self.val_history) > 0:
                overfit_ratio = [
                    v / t if t > 0 else 0
                    for t, v in zip(self.train_history, self.val_history)
                ]
                ax4.plot(
                    range(1, len(overfit_ratio) + 1),
                    overfit_ratio,
                    "purple",
                    linewidth=2,
                )

            if is_full_refit and not fine_tune_mode:
                ax4.text(
                    0.5, 0.5, "全量refit\n无验证集",
                    ha="center", va="center", transform=ax4.transAxes, fontsize=14
                )
                ax4.set_title("全量refit说明", fontsize=14, fontweight="bold")
                ax4.set_xticks([])
                ax4.set_yticks([])
            else:
                ax4.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
                ax4.set_xlabel("Epoch", fontsize=12)
                ax4.set_ylabel("验证/训练损失比率", fontsize=12)
                ax4.set_title("过拟合监测", fontsize=14, fontweight="bold")
            ax4.grid(True, alpha=0.3)

            title_suffix = " - 微调" if fine_tune_mode else ""
            plt.suptitle(
                f'SWE模型训练曲线 - {self.config["model_type"]}{title_suffix}',
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()

            # 保存图像
            plot_name = (
                "fine_tune_curves.png" if fine_tune_mode else "training_curves.png"
            )
            plot_path = self.save_dir / plot_name
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"训练曲线已保存到: {plot_path}")

        except Exception as e:
            print(f"绘制训练曲线失败: {e}")

    def evaluate_fine_tune(self, model_path=None, use_tta=False, tta_num=8):
        """
        评估微调模型 - 添加 FusedSWE 对比

        Args:
            model_path: 模型路径
            use_tta: 是否使用测试时增强
            tta_num: TTA 增强次数
        """
        print("\n" + "=" * 60)
        print("评估微调模型")
        if use_tta:
            print(f"使用 TTA (增强次数: {tta_num})")
        print("=" * 60)

        # 如果没有测试集，使用验证集
        test_loader = (
            self.test_loader
            if hasattr(self, "test_loader") and self.test_loader
            else self.val_loader
        )

        if test_loader is None:
            print("✗ 没有可用的测试数据加载器")
            return None

        # ============ 添加：检查高值样本是否在测试集中 ============
        print("\n" + "="*60)
        print("【高值样本检查】")
        print("="*60)

        high_value_indices = [8, 96, 196, 1580, 1606, 3411]

        if hasattr(test_loader.dataset, 'indices'):
            test_indices = set(test_loader.dataset.indices)
            print(f"测试集索引数量: {len(test_indices)}")

            in_test = []
            not_in_test = []

            for idx in high_value_indices:
                if idx in test_indices:
                    in_test.append(idx)
                else:
                    not_in_test.append(idx)

            print(f"\n高值样本索引检查:")
            print(f"  在测试集中的: {in_test}")
            print(f"  不在测试集中的: {not_in_test}")

            if in_test:
                print(f"\n⚠️ 有 {len(in_test)} 个高值样本在测试集中！")
                print(f"   如果微调图里看不到，说明数据在加载或画图时被过滤了")

                dataset = test_loader.dataset.dataset if hasattr(test_loader.dataset, 'dataset') else test_loader.dataset

                for idx in in_test:
                    if idx < len(dataset.meta_index):
                        meta = dataset.meta_index[idx]
                        date_val = meta.get('feature_date') or meta.get('label_date') or meta.get('date')
                        if date_val is not None:
                            if hasattr(date_val, 'strftime'):
                                date_str = date_val.strftime('%Y-%m-%d')
                            else:
                                date_str = str(date_val)
                        else:
                            date_str = 'unknown'

                        print(f"\n  样本索引 {idx}:")
                        print(f"    SWE值: {meta['swe']} mm")
                        print(f"    日期: {date_str}")
                        print(f"    位置: 行={meta['row']}, 列={meta['col']}")
                    else:
                        print(f"\n  样本索引 {idx}: 超出 meta_index 范围")
            else:
                print(f"\n✅ 所有高值样本都不在测试集中")
                print(f"   这就是微调图里看不到高值的原因")
        else:
            print("test_loader.dataset 没有 indices 属性")
        # ====================================================

        # 加载模型（如果提供）
        if model_path:
            self._load_model_for_evaluation(model_path)

        # 🔥 评估模式：强制冻结所有参数
        print("\n🔒 [EVALUATE MODE] 强制冻结所有参数（只推理，不训练）")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print("   ✅ 所有参数已冻结，模型只用于推理")

        # ============ Train / Val / Test各预测一次 ============
        print(
            "\n1. Train / Val / Test各执行一次完整预测..."
        )

        eval_loaders = {
            "train": self._build_complete_evaluation_loader(
                getattr(self, "train_loader", None),
                split_name="Train",
            ),
            "val": self._build_complete_evaluation_loader(
                getattr(self, "val_loader", None),
                split_name="Val",
            ),
            "test": self._build_complete_evaluation_loader(
                test_loader,
                split_name="Test",
            ),
        }

        split_prediction_results = {}

        for split_name in ["train", "val", "test"]:
            loader = eval_loaders.get(split_name)

            if loader is None:
                continue

            print(
                f"\n  [{split_name.upper()}] "
                "执行唯一一次模型前向..."
            )

            old_aug = self._set_loader_augmentation(
                loader,
                is_train=False,
            )

            try:
                if split_name == "test" and use_tta:
                    split_result = (
                        self._make_predictions_with_tta(
                            loader,
                            num_augmentations=tta_num,
                        )
                    )
                else:
                    split_result = self._make_predictions(
                        loader
                    )
            finally:
                self._restore_loader_augmentation(
                    loader,
                    old_aug,
                )

            if split_result is None:
                raise RuntimeError(
                    f"{split_name}预测失败"
                )

            actual_n = len(split_result[0])
            expected_n = len(loader.dataset)

            if actual_n != expected_n:
                raise RuntimeError(
                    f"{split_name}预测不完整: "
                    f"actual={actual_n}, "
                    f"expected={expected_n}"
                )

            split_prediction_results[split_name] = (
                split_result
            )

            print(
                f"    ✅ {split_name}: "
                f"{actual_n}/{expected_n}"
            )

        if "test" not in split_prediction_results:
            print("✗ 没有获得测试集预测")
            return None

        test_loader = eval_loaders["test"]
        result = split_prediction_results["test"]

        predictions, targets, is_zero = result

        print(
            f"  获取到 {len(predictions)} 个完整测试样本"
        )

        # ============ 获取 FusedSWE 网格值 ============
        print("\n2. 获取 FusedSWE 网格值...")
        all_fused_swe = []
        all_sample_indices = []

        try:
            if hasattr(test_loader.dataset, 'dataset'):
                dataset = test_loader.dataset.dataset
                subset_indices = list(test_loader.dataset.indices)
            else:
                dataset = test_loader.dataset
                subset_indices = list(range(len(dataset)))

            all_sample_indices = subset_indices[:len(predictions)]

            for idx in all_sample_indices:
                if idx < len(dataset.meta_index):
                    meta = dataset.meta_index[idx]
                    date = meta.get('feature_date') or meta.get('label_date') or meta.get('date')
                    if date is None:
                        if isinstance(meta, (list, tuple)):
                            date = meta[0]
                            r, c = meta[1], meta[2]
                        else:
                            all_fused_swe.append(np.nan)
                            continue
                    else:
                        r, c = meta.get('row'), meta.get('col')
                        if r is None or c is None:
                            all_fused_swe.append(np.nan)
                            continue

                    if hasattr(dataset, 'label_data') and date in dataset.label_data:
                        label_arr, label_nodata = dataset.label_data[date]
                        if 0 <= r < label_arr.shape[0] and 0 <= c < label_arr.shape[1]:
                            val = label_arr[r, c]
                            if label_nodata is None or val != label_nodata:
                                all_fused_swe.append(float(val))
                            else:
                                all_fused_swe.append(np.nan)
                        else:
                            all_fused_swe.append(np.nan)
                    else:
                        all_fused_swe.append(np.nan)
                else:
                    all_fused_swe.append(np.nan)

            all_fused_swe = np.array(all_fused_swe)
            valid_fused = np.sum(~np.isnan(all_fused_swe))
            print(f"  获取到 {valid_fused}/{len(all_fused_swe)} 个 FusedSWE 值")

        except Exception as e:
            print(f"  ⚠ 获取 FusedSWE 失败: {e}")
            all_fused_swe = None

        # 获取反归一化参数
        swe_min = getattr(self, 'swe_min', 0.0)
        swe_max = getattr(self, 'swe_max', 200.0)
        print(f"  反归一化参数: min={swe_min:.2f}, max={swe_max:.2f}")

        # ============ 调试代码 ============
        print("\n🔍 【归一化边界/极值检查】")
        print(f"  swe_min, swe_max: [{swe_min:.4f}, {swe_max:.4f}]")
        print(f"  target_norm 范围: [{targets.min():.6f}, {targets.max():.6f}]")
        print(f"  pred_norm 范围:   [{predictions.min():.6f}, {predictions.max():.6f}]")

        if all_fused_swe is not None:
            all_fused_swe_arr = np.asarray(
                all_fused_swe,
                dtype=np.float64,
            )

            finite_product = np.isfinite(all_fused_swe_arr)

            if np.any(finite_product):
                product_raw_mm = self._to_mm_auto(
                    all_fused_swe_arr,
                    swe_min,
                    swe_max,
                )

                swe_range = max(
                    float(swe_max - swe_min),
                    1e-12,
                )

                product_norm = (
                    product_raw_mm - swe_min
                ) / swe_range

                valid_norm = (
                    finite_product
                    & np.isfinite(product_norm)
                )

                print(
                    "  product_raw_mm 范围: "
                    f"[{np.nanmin(product_raw_mm):.6f}, "
                    f"{np.nanmax(product_raw_mm):.6f}] mm"
                )
                print(
                    "  product_norm 范围:   "
                    f"[{np.nanmin(product_norm):.6f}, "
                    f"{np.nanmax(product_norm):.6f}]"
                )
                print(
                    "  product_norm >= 0.999: "
                    f"{int(np.sum(valid_norm & (product_norm >= 0.999)))} 个"
                )
                print(
                    "  product_norm超出[0,1]: "
                    f"<0={int(np.sum(valid_norm & (product_norm < 0)))}, "
                    f">1={int(np.sum(valid_norm & (product_norm > 1)))}"
                )
            else:
                print(
                    "  product_raw_mm/product_norm: "
                    "无有效值"
                )

        print(f"  target_norm >= 0.999: {np.sum(targets >= 0.999)} 个")
        print(f"  pred_norm >= 0.999:   {np.sum(predictions >= 0.999)} 个")
        print(f"  超出范围: target<0={np.sum(targets<0)}, target>1={np.sum(targets>1)}")
        # ===========================================

        # ============ 🔥 关键诊断：产品为0但站点有雪 ============
        if all_fused_swe is not None:
            pred_mm = predictions * (swe_max - swe_min) + swe_min
            target_mm = targets * (swe_max - swe_min) + swe_min
            fused_arr = np.array(all_fused_swe)
            fused_valid = ~np.isnan(fused_arr)

            if fused_valid.sum() > 0:
                # 对齐长度
                n = min(len(pred_mm), len(fused_arr))
                pred_mm = pred_mm[:n]
                target_mm = target_mm[:n]
                fused_mm = self._to_mm_auto(fused_arr[:n], swe_min, swe_max)
                fused_valid = fused_valid[:n]

                print("\n【关键诊断：产品为0但站点有雪】")
                for thresh in [20, 50, 80]:
                    mask = fused_valid & (fused_mm <= 1.0) & (target_mm >= thresh)
                    count = int(mask.sum())
                    if count > 0:
                        print(f"  FusedSWE<=1mm & station>={thresh}mm: {count} 样本")
                        print(f"    target mean: {target_mm[mask].mean():.2f} mm")
                        print(f"    pred mean:   {pred_mm[mask].mean():.2f} mm")
                        print(f"    RMSE: {np.sqrt(np.mean((pred_mm[mask] - target_mm[mask])**2)):.2f} mm")
                    else:
                        print(f"  FusedSWE<=1mm & station>={thresh}mm: 0 样本")
            else:
                print("\n【关键诊断：产品为0但站点有雪】")
                print(f"  ⚠ 无有效 FusedSWE 数据，跳过")
        else:
            print("\n【关键诊断：产品为0但站点有雪】")
            print(f"  ⚠ FusedSWE 数据为空，跳过")
        # ===========================================

        # 计算评估指标
        print("\n3. 计算评估指标...")
        try:
            eval_results = self._compute_metrics(predictions, targets)

            if isinstance(eval_results, tuple) and len(eval_results) == 3:
                eval_metrics, predictions_denorm, targets_denorm = eval_results
            elif isinstance(eval_results, tuple) and len(eval_results) == 2:
                eval_metrics, _ = eval_results
                predictions_denorm = predictions * (swe_max - swe_min) + swe_min
                targets_denorm = targets * (swe_max - swe_min) + swe_min
            elif isinstance(eval_results, dict):
                eval_metrics = eval_results
                predictions_denorm = predictions * (swe_max - swe_min) + swe_min
                targets_denorm = targets * (swe_max - swe_min) + swe_min
            else:
                print(f"✗ 指标计算返回格式错误: {type(eval_results)}")
                return None

        except Exception as e:
            print(f"✗ 指标计算失败: {e}")
            traceback.print_exc()
            return None

        if eval_metrics is None:
            print("✗ 指标计算失败")
            return None

        # 添加 TTA 信息到 metrics
        if use_tta:
            eval_metrics['tta'] = {
                'enabled': True,
                'num_augmentations': tta_num
            }

        # ============ 反归一化 FusedSWE ============
        if all_fused_swe is not None:
            if all_fused_swe.max() <= 1.0 and all_fused_swe.min() >= 0:
                fused_denorm = all_fused_swe * (swe_max - swe_min) + swe_min
            else:
                fused_denorm = all_fused_swe

            valid_mask = ~np.isnan(fused_denorm)
            if np.sum(valid_mask) > 0:
                fused_valid = fused_denorm[valid_mask]
                targets_valid = targets_denorm[valid_mask]

                fused_rmse = np.sqrt(np.mean((fused_valid - targets_valid) ** 2))
                fused_mae = np.mean(np.abs(fused_valid - targets_valid))
                ss_res_fused = np.sum((fused_valid - targets_valid) ** 2)
                ss_tot = np.sum((targets_valid - np.mean(targets_valid)) ** 2)
                fused_r2 = 1 - (ss_res_fused / ss_tot) if ss_tot > 0 else 0

                fused_r, _ = stats.pearsonr(fused_valid, targets_valid)

                eval_metrics['fused_swe'] = {
                    'r2': float(fused_r2),
                    'rmse': float(fused_rmse),
                    'mae': float(fused_mae),
                    'r': float(fused_r),
                    'n_samples': int(np.sum(valid_mask))
                }

                print(f"\n  FusedSWE 指标:")
                print(f"    R²: {fused_r2:.4f}")
                print(f"    RMSE: {fused_rmse:.2f} mm")
                print(f"    MAE: {fused_mae:.2f} mm")

        # ============ 🔥 偏差诊断（内联实现） ============
        print("\n4. 运行模型偏差诊断...")
        diagnosis_results = self._run_diagnose_swe(targets_denorm, predictions_denorm)
        if diagnosis_results:
            eval_metrics['diagnosis'] = diagnosis_results
            
        print(
            "\n4b. 基于单次预测计算 "
            "Train / Val / Test完整诊断..."
        )

        split_diagnostics = {}

        for split_name in ["train", "val", "test"]:
            loader = eval_loaders.get(split_name)
            cached_result = split_prediction_results.get(
                split_name
            )

            if loader is None or cached_result is None:
                continue

            split_diagnostics[split_name] = (
                self._evaluate_loader_diagnostics(
                    loader,
                    split_name=split_name,
                    high_threshold=80.0,
                    prediction_result=cached_result,
                )
            )

        split_diagnostics = {
            key: value
            for key, value in split_diagnostics.items()
            if value is not None
        }

        if split_diagnostics:
            eval_metrics["split_diagnostics"] = (
                split_diagnostics
            )
            self._print_split_diagnostics_table(
                split_diagnostics
            )

        diagnosis_payload = {
            "main_test_metrics": eval_metrics,
            "single_test_diagnosis": diagnosis_results,
            "split_diagnostics": split_diagnostics,
            "swe_min": float(getattr(self, "swe_min", 0.0)),
            "swe_max": float(getattr(self, "swe_max", 200.0)),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._save_diagnosis_json(diagnosis_payload, filename="diagnosis_results.json")


        print(
            "\n4c. 跳过 Frozen + Linear Calibration："
            "评估仅保留Train/Val/Test各一次预测"
        )

        # 保存评估结果：失败必须立即暴露，禁止“程序完成但文件不存在”
        print("\n5. 保存评估结果...")
        eval_path = self._save_fine_tune_evaluation_results(
            eval_metrics,
            predictions_denorm,
            targets_denorm,
        )
        print(f"  ✓ 已验证评估结果文件存在: {eval_path}")

        # 生成对比散点图
        print("\n6. 生成对比散点图...")
        try:
            self._generate_comparison_scatter_plots(
                targets_denorm, 
                predictions_denorm, 
                fused_denorm if all_fused_swe is not None else None,
                eval_metrics
            )

            if 'predictions_denorm' in locals() and predictions_denorm is not None:
                self._generate_fine_tune_plots(predictions_denorm, targets_denorm, eval_metrics)
            else:
                self._generate_fine_tune_plots(predictions, targets, eval_metrics, use_raw=True)
        except Exception as e:
            print(f"  ✗ 图表生成失败: {e}")
            traceback.print_exc()

        print(
            "\n7. 跳过测试集全标注散点图："
            "避免再次预测Test"
        )

        # ============ 修改：传入正确的预测数据 ============
        print("\n8. 生成测试集完整特征分析...")
        df = self.analyze_test_set_features(
            test_loader=test_loader,
            pretrained_model_path=None,
            predictions_denorm=predictions_denorm,
            targets_denorm=targets_denorm,
            fused_denorm=fused_denorm,
        )

        print(
            "\n9. 跳过all_finetune_samples_scatter："
            "避免再次预测Train/Val/Test"
        )

        # 打印最终摘要
        print("\n" + "=" * 60)
        print("微调评估完成!")
        if use_tta:
            print(f"使用 TTA (增强次数: {tta_num})")
        print("=" * 60)

        # 额外诊断信息
        print(f"\n📊 详细诊断信息:")
        print(f"  归一化预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  归一化真实范围: [{targets.min():.4f}, {targets.max():.4f}]")

        if 'predictions_denorm' in locals() and predictions_denorm is not None:
            print(f"  反归一化预测范围: [{predictions_denorm.min():.2f}, {predictions_denorm.max():.2f}] mm")
            print(f"  反归一化真实范围: [{targets_denorm.min():.2f}, {targets_denorm.max():.2f}] mm")
            print(f"  预测均值: {predictions_denorm.mean():.2f} mm")
            print(f"  真实均值: {targets_denorm.mean():.2f} mm")

        # 约束效果统计
        if is_zero is not None:
            zero_count = np.sum(is_zero == 0)
            pos_count = len(is_zero) - zero_count
            print(f"\n  约束效果统计:")
            print(f"    target=0样本数: {zero_count} ({zero_count/len(is_zero)*100:.1f}%)")
            print(f"    target>0样本数: {pos_count} ({pos_count/len(is_zero)*100:.1f}%)")

            if zero_count > 0:
                zero_predictions = predictions[is_zero == 0]
                zero_abs_mean = np.mean(np.abs(zero_predictions))
                print(f"    target=0样本预测绝对均值: {zero_abs_mean:.6f}")

        print(f"  r:    {eval_metrics.get('r', 'N/A'):.4f}")
        print(f"  RMSE: {eval_metrics.get('rmse', 'N/A'):.2f} mm")
        print(f"  MAE:  {eval_metrics.get('mae', 'N/A'):.2f} mm")

        if 'fused_swe' in eval_metrics:
            print(f"\n【FusedSWE 对比】")
            print(f"  FusedSWE R²: {eval_metrics['fused_swe']['r2']:.4f}")
            print(f"  FusedSWE RMSE: {eval_metrics['fused_swe']['rmse']:.2f} mm")
            print(f"  您的模型 R²: {eval_metrics.get('r2', 'N/A'):.4f}")
            print(f"  您的模型 RMSE: {eval_metrics.get('rmse', 'N/A'):.2f} mm")

            if eval_metrics.get('r2', 0) > eval_metrics['fused_swe']['r2']:
                print(f"\n  ✅ 您的模型比 FusedSWE 好！")
            else:
                print(f"\n  ⚠️ 您的模型不如 FusedSWE，需要改进")

        # 🔥 打印偏差诊断摘要
        if diagnosis_results:
            print("\n" + "=" * 60)
            print("📈 偏差诊断摘要")
            print("=" * 60)
            orig = diagnosis_results['original']
            cal = diagnosis_results['calibrated']
            print(f"  原始 NSE: {orig['nse']:.4f}  |  后校准 NSE: {cal['nse']:.4f}")
            print(f"  原始 R:   {orig['r']:.4f}  |  后校准 R:   {cal['r']:.4f}")
            print(f"  α (std ratio): {orig['alpha']:.4f}  (理想值=1)")
            print(f"  β (bias/std):  {orig['beta']:.4f}  (理想值=0)")
            print(f"  回归线: pred = {orig['intercept']:.4f} + {orig['slope']:.4f} * obs (理想: intercept=0, slope=1)")
            if cal['nse'] > orig['nse'] + 0.05:
                print(f"  ✅ 后校准有效 (NSE提升 {cal['nse'] - orig['nse']:.4f})")
            else:
                print(f"  ⚠️ 后校准无效，偏差可能不是线性的")

        # ============ Prior Ablation 诊断 ============
        # 当前 Clean-18D 默认没有先验列；只有显式配置后才运行。
        prior_col = self.config.get("physical_prior_col", None)
        c_point = int(self.config.get("C_point", 0) or 0)
        can_run_prior_ablation = (
            bool(self.config.get("enable_prior_ablation", False))
            and isinstance(prior_col, int)
            and 0 <= prior_col < c_point
        )

        if can_run_prior_ablation:
            self._run_prior_ablation_diagnosis(
                dataloader=self.test_loader,
                split_name="test",
                fold_idx=getattr(self, "current_fold", None),
            )
        else:
            print(
                "ℹ 跳过 prior ablation："
                f"当前 C_point={c_point}, physical_prior_col={prior_col}, "
                f"enable_prior_ablation={self.config.get('enable_prior_ablation', False)}"
            )
        # ===========================================

        return {
            "metrics": eval_metrics,
            "predictions_norm": predictions,
            "targets_norm": targets,
            "predictions_denorm": predictions_denorm if 'predictions_denorm' in locals() else None,
            "targets_denorm": targets_denorm if 'targets_denorm' in locals() else None,
            "fused_swe": fused_denorm if all_fused_swe is not None else None,
            "is_zero": is_zero,
            "use_tta": use_tta,
            "tta_num": tta_num if use_tta else None,
            "diagnosis": diagnosis_results
        }
    
    def _run_prior_ablation_diagnosis(self, dataloader=None, split_name="test", fold_idx=None):
        """
        对显式配置在 point_feats 中的 physical prior 做消融。

        Clean-18D 默认没有 prior 列，因此默认直接跳过。只有同时满足：
        1) enable_prior_ablation=True；
        2) physical_prior_col 为合法整数；
        3) physical_prior_col < C_point；
        才会执行消融并输出 CSV。
        """

        prior_col = self.config.get("physical_prior_col", None)
        c_point = int(self.config.get("C_point", 0) or 0)
        enabled = bool(self.config.get("enable_prior_ablation", False))

        if not enabled:
            print("ℹ prior ablation 已关闭（enable_prior_ablation=False）")
            return None

        if not isinstance(prior_col, int) or not (0 <= prior_col < c_point):
            print(
                "ℹ prior ablation 跳过：未配置合法先验列 "
                f"(C_point={c_point}, physical_prior_col={prior_col})"
            )
            return None

        if dataloader is None:
            dataloader = self.test_loader

        if dataloader is None:
            print("⚠ prior ablation 跳过：dataloader 为空")
            return None

        if self.model is None:
            print("⚠ prior ablation 跳过：模型为空")
            return None

        strategy = self.config.get("freeze_strategy", "unknown")
        if fold_idx is None:
            fold_idx = getattr(self, "current_fold", None)

        print("\n" + "=" * 70)
        print(f"【Prior Ablation】strategy={strategy}, split={split_name}, fold={fold_idx}")
        print(f"  dataloader 样本数: {len(dataloader.dataset)}")
        print("=" * 70)

        self.model.eval()

        rows = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):

                # ============ 解析 batch ============
                if len(batch_data) == 6:
                    conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe, sample_indices = batch_data
                elif len(batch_data) == 5:
                    conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe = batch_data
                    sample_indices = None
                elif len(batch_data) == 4:
                    conv_feats, point_feats, targets, is_zero_mask = batch_data
                    raw_fused_swe = torch.full_like(targets, float("nan"))
                    sample_indices = None
                elif len(batch_data) == 3:
                    conv_feats, point_feats, targets = batch_data
                    is_zero_mask = torch.ones_like(targets)
                    raw_fused_swe = torch.full_like(targets, float("nan"))
                    sample_indices = None
                else:
                    print(f"  ⚠ batch {batch_idx}: 未知 batch 长度 {len(batch_data)}，跳过")
                    continue

                conv_feats = conv_feats.to(self.device, non_blocking=True)
                point_feats = point_feats.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # 运行时再次核对真实 batch 维度，防止缓存/模型契约漂移
                if point_feats.shape[1] <= prior_col:
                    if batch_idx == 0:
                        print(
                            f"⚠ point_feats 只有 {point_feats.shape[1]} 维，"
                            f"无法访问 prior 列 {prior_col}，跳过 ablation"
                        )
                    continue

                if batch_idx == 0:
                    print(
                        f"  batch 0: point_feats.shape={point_feats.shape}, "
                        f"batch_size={len(targets)}, "
                        f"prior_col={prior_col}, "
                        f"prior range=[{point_feats[:, prior_col].min():.4f}, "
                        f"{point_feats[:, prior_col].max():.4f}]"
                    )

                # ============ 原始输入 ============
                pred_original = self.model(conv_feats, point_feats)
                if isinstance(pred_original, tuple):
                    pred_original = pred_original[0]
                pred_original = pred_original.reshape(-1)

                # ============ 第21维置0 ============
                point_no_prior = point_feats.clone()
                point_no_prior[:, prior_col] = 0.0

                pred_no_prior = self.model(conv_feats, point_no_prior)
                if isinstance(pred_no_prior, tuple):
                    pred_no_prior = pred_no_prior[0]
                pred_no_prior = pred_no_prior.reshape(-1)

                # ============ 第21维置0.5 ============
                point_prior_05 = point_feats.clone()
                point_prior_05[:, prior_col] = 0.5

                pred_prior_05 = self.model(conv_feats, point_prior_05)
                if isinstance(pred_prior_05, tuple):
                    pred_prior_05 = pred_prior_05[0]
                pred_prior_05 = pred_prior_05.reshape(-1)

                # ============ 转 numpy ============
                pred_np = pred_original.detach().cpu().numpy()
                pred_no_prior_np = pred_no_prior.detach().cpu().numpy()
                pred_prior_05_np = pred_prior_05.detach().cpu().numpy()

                target_np = targets.reshape(-1).detach().cpu().numpy()
                fused_np = raw_fused_swe.reshape(-1).detach().cpu().numpy()
                zero_np = is_zero_mask.reshape(-1).detach().cpu().numpy()
                point_np = point_feats.detach().cpu().numpy()
                prior21_np = point_np[:, prior_col]

                if sample_indices is not None:
                    sample_indices_np = sample_indices.reshape(-1).detach().cpu().numpy()
                else:
                    sample_indices_np = np.full(len(pred_np), -1)

                # ============ 记录样本 ============
                for i in range(len(pred_np)):
                    rows.append({
                        "strategy": strategy,
                        "fold": fold_idx if fold_idx is not None else -1,
                        "split": split_name,
                        "batch_idx": batch_idx,
                        "local_i": i,
                        "sample_index": int(sample_indices_np[i]),

                        "pred_norm": float(pred_np[i]),
                        "pred_norm_no_prior": float(pred_no_prior_np[i]),
                        "pred_norm_prior_05": float(pred_prior_05_np[i]),
                        "target_norm": float(target_np[i]),

                        "fused_value": float(fused_np[i]) if np.isfinite(fused_np[i]) else np.nan,
                        "prior21": float(prior21_np[i]),
                        "is_zero_mask": float(zero_np[i]),
                    })

        if len(rows) == 0:
            print("⚠ prior ablation 没有有效样本")
            return None

        diag_df = pd.DataFrame(rows)

        swe_min = getattr(self, "swe_min", 0.0)
        swe_max = getattr(self, "swe_max", 170.0)

        # ============ 反归一化（模型输出一定是归一化值，强制转换） ============
        def _denorm(x):
            x = np.asarray(x, dtype=np.float32)
            return x * (swe_max - swe_min) + swe_min

        diag_df["pred_mm"] = _denorm(diag_df["pred_norm"].values)
        diag_df["pred_mm_no_prior"] = _denorm(diag_df["pred_norm_no_prior"].values)
        diag_df["pred_mm_prior_05"] = _denorm(diag_df["pred_norm_prior_05"].values)
        diag_df["target_mm"] = _denorm(diag_df["target_norm"].values)
        diag_df["fused_mm"] = _denorm(diag_df["fused_value"].values)

        # ============ 低值带诊断 ============
        diag_df["bad_low20_high50"] = (
            (diag_df["pred_mm"] <= 20.0) &
            (diag_df["target_mm"] >= 50.0)
        )
        diag_df["bad_low20_high50_no_prior"] = (
            (diag_df["pred_mm_no_prior"] <= 20.0) &
            (diag_df["target_mm"] >= 50.0)
        )
        diag_df["bad_low20_high50_prior_05"] = (
            (diag_df["pred_mm_prior_05"] <= 20.0) &
            (diag_df["target_mm"] >= 50.0)
        )

        diag_df["bad_low30_high80"] = (
            (diag_df["pred_mm"] <= 30.0) &
            (diag_df["target_mm"] >= 80.0)
        )
        diag_df["bad_low30_high80_no_prior"] = (
            (diag_df["pred_mm_no_prior"] <= 30.0) &
            (diag_df["target_mm"] >= 80.0)
        )
        diag_df["bad_low30_high80_prior_05"] = (
            (diag_df["pred_mm_prior_05"] <= 30.0) &
            (diag_df["target_mm"] >= 80.0)
        )

        diag_df["bad_low40_high100"] = (
            (diag_df["pred_mm"] <= 40.0) &
            (diag_df["target_mm"] >= 100.0)
        )
        diag_df["bad_low40_high100_no_prior"] = (
            (diag_df["pred_mm_no_prior"] <= 40.0) &
            (diag_df["target_mm"] >= 100.0)
        )
        diag_df["bad_low40_high100_prior_05"] = (
            (diag_df["pred_mm_prior_05"] <= 40.0) &
            (diag_df["target_mm"] >= 100.0)
        )

        diag_df["fused_zero_snow20"] = (
            (diag_df["fused_mm"] <= 1.0) &
            (diag_df["target_mm"] >= 20.0)
        )
        diag_df["fused_zero_snow80"] = (
            (diag_df["fused_mm"] <= 1.0) &
            (diag_df["target_mm"] >= 80.0)
        )

        # ============ 保存 CSV ============
        fold_tag = f"fold{fold_idx}" if fold_idx is not None else "nofold"
        csv_path = self.save_dir / f"prior_ablation_diag_{strategy}_{split_name}_{fold_tag}.csv"
        diag_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"\n💾 prior ablation CSV 已保存: {csv_path}")

        # ============ 打印核心汇总 ============
        print("\n【第21维先验 ablation：低值带诊断】")
        print(f"  原始 pred<=20 & target>=50: {diag_df['bad_low20_high50'].sum()}")
        print(f"  第21维置0 pred<=20 & target>=50: {diag_df['bad_low20_high50_no_prior'].sum()}")
        print(f"  第21维置0.5 pred<=20 & target>=50: {diag_df['bad_low20_high50_prior_05'].sum()}")

        print(f"  原始 pred<=30 & target>=80: {diag_df['bad_low30_high80'].sum()}")
        print(f"  第21维置0 pred<=30 & target>=80: {diag_df['bad_low30_high80_no_prior'].sum()}")
        print(f"  第21维置0.5 pred<=30 & target>=80: {diag_df['bad_low30_high80_prior_05'].sum()}")

        print(f"  原始 pred<=40 & target>=100: {diag_df['bad_low40_high100'].sum()}")
        print(f"  第21维置0 pred<=40 & target>=100: {diag_df['bad_low40_high100_no_prior'].sum()}")
        print(f"  第21维置0.5 pred<=40 & target>=100: {diag_df['bad_low40_high100_prior_05'].sum()}")

        # ============ 高值样本均值对比 ============
        print("\n【高 SWE 样本 prior ablation 均值】")

        for threshold in [50.0, 80.0, 100.0]:
            sub = diag_df[diag_df["target_mm"] >= threshold]
            if len(sub) == 0:
                continue

            print(f"\n  target>={threshold:.0f}mm: {len(sub)} 样本")
            print(f"    target mean:        {sub['target_mm'].mean():.2f}")
            print(f"    fused mean:         {sub['fused_mm'].mean():.2f}")
            print(f"    prior21 mean:       {sub['prior21'].mean():.4f}")
            print(f"    pred original mean: {sub['pred_mm'].mean():.2f}")
            print(f"    pred prior=0 mean:  {sub['pred_mm_no_prior'].mean():.2f}")
            print(f"    pred prior=0.5 mean:{sub['pred_mm_prior_05'].mean():.2f}")

        # ============ FusedSWE=0 但站点厚雪 ============
        sub = diag_df[
            (diag_df["fused_mm"] <= 1.0) &
            (diag_df["target_mm"] >= 80.0)
        ]

        print("\n【FusedSWE<=1mm & target>=80mm prior ablation】")
        print(f"  样本数: {len(sub)}")

        if len(sub) > 0:
            print(f"    target mean:        {sub['target_mm'].mean():.2f}")
            print(f"    pred original mean: {sub['pred_mm'].mean():.2f}")
            print(f"    pred prior=0 mean:  {sub['pred_mm_no_prior'].mean():.2f}")
            print(f"    pred prior=0.5 mean:{sub['pred_mm_prior_05'].mean():.2f}")

        return diag_df

    def _make_predictions_with_tta(self, dataloader=None, num_augmentations=8):
        """
        使用测试时增强进行预测

        Args:
            dataloader: 数据加载器
            num_augmentations: TTA 次数

        Returns:
            predictions, targets, is_zero
        """
        print(f"  使用 TTA 进行预测 (增强次数: {num_augmentations})...")

        if dataloader is None:
            dataloader = self.val_loader
            if dataloader is None:
                print("  ❌ 错误: 未提供 dataloader 且默认加载器为空")
                return None, None, None

        all_predictions = []
        all_targets = []
        all_is_zero = []

        self.model.eval()

        # 检查模型类型
        is_residual_model = hasattr(self.model, 'correction_mlp')
        is_gate_model = hasattr(self.model, 'pretrained_model') and hasattr(self.model, 'fine_tune_net')

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # 解析批次数据
                if len(batch_data) >= 5:
                    conv_feats = batch_data[0]
                    point_feats = batch_data[1]
                    targets = batch_data[2]
                    is_zero_mask = batch_data[3] if len(batch_data) > 3 else None
                    raw_fused_swe = batch_data[4] if len(batch_data) > 4 else None
                else:
                    conv_feats, point_feats, targets = batch_data[:3]
                    is_zero_mask = batch_data[3] if len(batch_data) > 3 else None
                    raw_fused_swe = None

                batch_size = len(targets)

                # 对每个样本进行 TTA
                batch_predictions = []

                for i in range(batch_size):
                    # 提取单个样本
                    conv_single = conv_feats[i:i+1].to(self.device)
                    point_single = point_feats[i:i+1].to(self.device)

                    # 获取原始点特征用于增强
                    point_original = point_single.clone()

                    single_preds = []

                    for aug_idx in range(num_augmentations):
                        # 复制点特征
                        point_aug = point_original.clone()

                        # 对经纬度添加噪声 (索引 10, 11)
                        lon_idx, lat_idx = 15, 16  # 经纬度在21维中的正确索引
                        if point_aug.shape[1] > max(lon_idx, lat_idx):
                            # 🔥 修复：使用 .item() 确保是标量
                            point_aug[0, lon_idx] += torch.randn(1).to(self.device).item() * 0.01
                            point_aug[0, lat_idx] += torch.randn(1).to(self.device).item() * 0.01

                        # 对微波信号添加噪声 (索引 6,7,8,9)
                        microwave_indices = [6, 7, 8, 9]
                        for m_idx in microwave_indices:
                            if m_idx < point_aug.shape[1]:
                                point_aug[0, m_idx] *= (1 + torch.randn(1).to(self.device).item() * 0.005)

                        # 裁剪到有效范围
                        point_aug = torch.clamp(point_aug, 0.0, 1.0)

                        # 预测
                        if raw_fused_swe is not None:
                            fused_single = raw_fused_swe[i:i+1].to(self.device)
                            if is_residual_model:
                                output, _ = self.model(conv_single, point_aug, fused_single)
                            else:
                                output = self.model(conv_single, point_aug)
                        else:
                            output = self.model(conv_single, point_aug)

                        single_preds.append(output.item())

                    # 取中位数作为该样本的最终预测（对异常值更鲁棒）
                    # SWE_NONNEGATIVE_PHYSICAL_CONSTRAINT_V1
                    final_pred = max(
                        0.0,
                        float(np.median(single_preds)),
                    )
                    batch_predictions.append(final_pred)

                # 收集结果
                batch_predictions = torch.tensor(batch_predictions, device=self.device)
                all_predictions.extend(batch_predictions.cpu().numpy())
                all_targets.extend(targets.numpy())

                if is_zero_mask is not None:
                    all_is_zero.extend(is_zero_mask.numpy())
                else:
                    all_is_zero.extend((targets.numpy() > 0).astype(np.float32))

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_is_zero = np.array(all_is_zero)

        print(f"  TTA预测完成，样本数: {len(all_predictions)}")

        return all_predictions, all_targets, all_is_zero
    
    
    def _generate_comparison_scatter_plots(self, targets_denorm, predictions_denorm, fused_denorm, eval_metrics):
        """
        生成对比散点图：FusedSWE vs 微调模型
        确保两个图用完全相同的样本（有 FusedSWE 的样本）
        """
        # 设置英文标签，避免乱码
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
        plt.rcParams['axes.unicode_minus'] = False

        # ============ 确保两个图用同一批样本 ============
        if fused_denorm is not None:
            fused_valid_mask = ~np.isnan(fused_denorm)
            n_fused = np.sum(fused_valid_mask)
            print(f"  有 FusedSWE 值的样本数: {n_fused}/{len(targets_denorm)}")

            # 统一用有 FusedSWE 的样本
            common_targets = targets_denorm[fused_valid_mask]
            common_predictions = predictions_denorm[fused_valid_mask]
            common_fused = fused_denorm[fused_valid_mask]
        else:
            print("  ⚠ 没有 FusedSWE 数据，无法生成对比图")
            return

        # 🔥 改为 1x2 布局
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 计算整体范围
        all_values = np.concatenate([common_targets, common_predictions, common_fused])
        min_val = all_values.min()
        max_val = all_values.max()
        margin = (max_val - min_val) * 0.05
        plot_min = max(0, min_val - margin)
        plot_max = max_val + margin

        # ============ 图1: ERA5-Land SWE vs 站点实测值 ============
        ax1 = axes[0]

        fused_rmse = np.sqrt(np.mean((common_fused - common_targets) ** 2))
        fused_mae = np.mean(np.abs(common_fused - common_targets))
        fused_bias = np.mean(common_fused - common_targets)
        ss_res_fused = np.sum((common_fused - common_targets) ** 2)
        ss_tot = np.sum((common_targets - np.mean(common_targets)) ** 2)
        fused_r2 = 1 - (ss_res_fused / ss_tot) if ss_tot > 0 else 0
        fused_r, _ = stats.pearsonr(common_fused, common_targets)

        ax1.scatter(common_targets, common_fused, alpha=0.5, s=20, c='red', edgecolors='none')
        ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=2, alpha=0.8, label='1:1 Line')

        ax1.set_xlabel('Station SWE (mm)', fontsize=12)
        ax1.set_ylabel('ERA5-Land SWE (mm)', fontsize=12)
        ax1.set_title('ERA5-Land SWE vs Station', fontsize=12)

        text1 = (
            f'N = {len(common_targets)}\n'
            f'R = {fused_r:.4f}\n'
            f'RMSE = {fused_rmse:.2f} mm\n'
            f'MAE = {fused_mae:.2f} mm\n'
            f'Bias = {fused_bias:.2f} mm'
        )
        ax1.text(
            0.97, 0.03, text1,
            transform=ax1.transAxes,
            fontsize=14,
            fontweight='bold',
            linespacing=1.20,
            va='bottom',
            ha='right',
            zorder=10
        )

        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(plot_min, plot_max)
        ax1.set_ylim(plot_min, plot_max)
        ax1.legend(loc='upper right', fontsize=9, frameon=True)

        # ============ 图2: 模型预测 vs 站点实测值 ============
        # 根据结果目录自动区分冻结评估和真正微调
        is_frozen_eval = "frozen" in str(self.save_dir).lower()
        model_name = "Pretrained Model" if is_frozen_eval else "Fine-tuned Model"
        prediction_name = (
            "Pretrained Prediction"
            if is_frozen_eval
            else "Fine-tuned Prediction"
        )

        ax2 = axes[1]

        final_rmse = np.sqrt(np.mean((common_predictions - common_targets) ** 2))
        final_mae = np.mean(np.abs(common_predictions - common_targets))
        final_bias = np.mean(common_predictions - common_targets)
        ss_res_final = np.sum((common_predictions - common_targets) ** 2)
        final_r2 = 1 - (ss_res_final / ss_tot) if ss_tot > 0 else 0
        final_r, _ = stats.pearsonr(common_predictions, common_targets)

        ax2.scatter(common_targets, common_predictions, alpha=0.5, s=20, c='green', edgecolors='none')
        ax2.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=2, alpha=0.8, label='1:1 Line')

        ax2.set_xlabel('Station SWE (mm)', fontsize=12)
        ax2.set_ylabel(f'{prediction_name} (mm)', fontsize=12)
        ax2.set_title(f'{model_name} vs Station', fontsize=12)

        text2 = (
            f'N = {len(common_targets)}\n'
            f'R = {final_r:.4f}\n'
            f'RMSE = {final_rmse:.2f} mm\n'
            f'MAE = {final_mae:.2f} mm\n'
            f'Bias = {final_bias:.2f} mm'
        )
        ax2.text(
            0.97, 0.03, text2,
            transform=ax2.transAxes,
            fontsize=14,
            fontweight='bold',
            linespacing=1.20,
            va='bottom',
            ha='right',
            zorder=10
        )
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(plot_min, plot_max)
        ax2.set_ylim(plot_min, plot_max)
        ax2.legend(loc='upper right', fontsize=9, frameon=True)

        plt.suptitle(f'Comparison: ERA5-Land SWE vs {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存图片
        save_path = self.save_dir / 'comparison_scatter.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n  ✓ 对比散点图已保存: {save_path}")
        print(f"    使用样本数: {len(common_targets)}（两图一致）")

        # 打印数据范围
        print(f"\n  【统一样本数据范围】")
        print(f"    实测值范围: [{common_targets.min():.2f}, {common_targets.max():.2f}] mm")
        print(f"    ERA5-Land SWE范围: [{common_fused.min():.2f}, {common_fused.max():.2f}] mm")
        print(f"    模型预测范围: [{common_predictions.min():.2f}, {common_predictions.max():.2f}] mm")  
        
    def _to_mm_auto(self, arr, swe_min=None, swe_max=None):
        """自动判断数组是归一化值(0~1)还是原始mm值，返回mm"""
        arr = np.asarray(arr, dtype=np.float32)
        if swe_min is None:
            swe_min = getattr(self, "swe_min", 0.0)
        if swe_max is None:
            swe_max = getattr(self, "swe_max", 170.0)
        valid = np.isfinite(arr)
        if valid.sum() == 0:
            return arr
        vmin = np.nanmin(arr[valid])
        vmax = np.nanmax(arr[valid])
        # 归一化值：0~1
        if vmin >= -0.05 and vmax <= 1.05:
            return arr * (swe_max - swe_min) + swe_min
        # 否则认为已经是 mm
        return arr

    def _make_predictions(self, dataloader=None, ablate_prior_dim: bool = False):
        """进行预测，兼容普通、残差与门控模型。

        ablate_prior_dim=True 仅在 physical_prior_col 被显式配置为合法
        point_feats 列时可用；Clean-18D 默认不支持该消融。
        """
        print("  开始预测...")

        prior_col = self.config.get("physical_prior_col", None)
        c_point = int(self.config.get("C_point", 0) or 0)
        has_point_prior = isinstance(prior_col, int) and 0 <= prior_col < c_point

        if ablate_prior_dim and not has_point_prior:
            raise ValueError(
                "请求 point prior ablation，但当前没有合法先验列："
                f"C_point={c_point}, physical_prior_col={prior_col}"
            )

        if dataloader is None:
            dataloader = self.val_loader
            if dataloader is None:
                print("  ❌ 错误: 未提供 dataloader 且默认加载器为空")
                if ablate_prior_dim:
                    return None, None, None, None, None
                return None, None, None

        all_predictions = []
        all_targets = []
        all_is_zero = []
        all_fused_swe = []
        all_no_prior = [] if ablate_prior_dim else None
        all_prior_05 = [] if ablate_prior_dim else None
        diag_rows = []

        self.model.eval()
        # 检查模型属性以决定前向传播分支
        is_residual_model = hasattr(self.model, 'correction_mlp')
        is_gate_model = hasattr(self.model, 'pretrained_model') and hasattr(self.model, 'fine_tune_net')

        # 获取归一化参数，用于 raw_fused_swe 的默认值
        swe_min = getattr(self, 'swe_min', 0.0)
        swe_max = getattr(self, 'swe_max', 200.0)

        # 用于控制警告只打印一次
        warned_raw_fused = False

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # 1. 灵活解包：支持 3/4/5/6 个返回值
                if len(batch_data) == 6:
                    conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe, sample_indices = batch_data
                elif len(batch_data) == 5:
                    conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe = batch_data
                    sample_indices = None
                elif len(batch_data) == 4:
                    conv_feats, point_feats, targets, is_zero_mask = batch_data
                    raw_fused_swe = torch.full_like(targets, np.nan)
                    sample_indices = None
                else:
                    conv_feats, point_feats, targets = batch_data
                    is_zero_mask = torch.ones_like(targets)
                    raw_fused_swe = torch.full_like(targets, np.nan)
                    sample_indices = None

                # 2. 处理NaN（保留原版的插值逻辑）
                # 处理卷积特征NaN（使用插值，保持原版精度）
                if torch.isnan(conv_feats).any():
                    conv_np = conv_feats.cpu().numpy()
                    conv_interp = np.zeros_like(conv_np)

                    for b in range(conv_np.shape[0]):
                        for c in range(conv_np.shape[1]):
                            patch = conv_np[b, c]
                            if np.isnan(patch).any():
                                conv_interp[b, c] = self._interpolate_nan_patch(patch)
                            else:
                                conv_interp[b, c] = patch

                    conv_feats = torch.from_numpy(conv_interp)

                # 处理点特征NaN（简单置零即可）
                if torch.isnan(point_feats).any():
                    point_feats = torch.nan_to_num(point_feats, nan=0.0)

                # 处理目标值NaN
                if torch.isnan(targets).any():
                    targets = torch.nan_to_num(targets, nan=0.0)
                    is_zero_mask = torch.where(targets > 0, 
                                             torch.ones_like(targets), 
                                             torch.zeros_like(targets))

                # 处理 raw_fused_swe 的NaN
                if torch.isnan(raw_fused_swe).any():
                    raw_fused_swe = torch.nan_to_num(raw_fused_swe, nan=0.0)

                # 最终检查（确保没有遗漏的NaN）
                conv_feats = torch.nan_to_num(conv_feats, nan=0.0)
                point_feats = torch.nan_to_num(point_feats, nan=0.0)
                targets = torch.nan_to_num(targets, nan=0.0)
                is_zero_mask = torch.nan_to_num(is_zero_mask, nan=1.0)

                # 移动到设备
                conv_feats = conv_feats.to(self.device)
                point_feats = point_feats.to(self.device)
                targets = targets.to(self.device)
                is_zero_mask = is_zero_mask.to(self.device)
                raw_fused_swe = raw_fused_swe.to(self.device)

                # 3. 点特征维度强制对齐 - 🔥 从配置获取期望维度
                expected_dim = self.config.get("C_point", 18)
                if point_feats.shape[1] != expected_dim:
                    if point_feats.shape[1] < expected_dim:
                        padding = torch.zeros(point_feats.shape[0], expected_dim - point_feats.shape[1], 
                                             device=self.device)
                        point_feats = torch.cat([point_feats, padding], dim=1)
                    else:
                        point_feats = point_feats[:, :expected_dim]

                # 4. 前向传播分支
                if is_residual_model:
                    # 确保 raw_fused_swe 形状正确
                    if raw_fused_swe.dim() == 1:
                        raw_fused_swe = raw_fused_swe.unsqueeze(1)
                    elif raw_fused_swe.dim() > 2:
                        raw_fused_swe = raw_fused_swe.view(raw_fused_swe.size(0), -1)
                        raw_fused_swe = raw_fused_swe[:, :1]

                    outputs_raw = self.model(conv_feats, point_feats, raw_fused_swe)
                    if isinstance(outputs_raw, tuple):
                        outputs_raw = outputs_raw[0]
                elif is_gate_model:
                    outputs_raw, _, _, _ = self.model(conv_feats, point_feats)
                else:
                    outputs_raw = self.model(conv_feats, point_feats)

                outputs_raw = outputs_raw.reshape(-1)

                # SWE_NONNEGATIVE_PHYSICAL_CONSTRAINT_V1
                outputs_physical = torch.clamp(
                    outputs_raw,
                    min=0.0,
                )

                # ===== 记录 raw 输出（不修改，用于诊断） =====
                pred_raw_np = outputs_raw.detach().cpu().numpy()
                target_np = targets.reshape(-1).detach().cpu().numpy()
                fused_np = raw_fused_swe.reshape(-1).detach().cpu().numpy()
                is_zero_np = is_zero_mask.reshape(-1).detach().cpu().numpy()

                point_np = point_feats.detach().cpu().numpy()
                prior21_np = (
                    point_np[:, prior_col]
                    if has_point_prior and point_np.shape[1] > prior_col
                    else np.full(len(pred_raw_np), np.nan)
                )

                all_fused_swe.extend(fused_np.tolist())

                # ===== 批次级 raw 输出诊断 =====
                if batch_idx < 3:
                    print(f"\n【raw预测诊断 batch {batch_idx}】")
                    print(f"  pred_norm range: [{np.nanmin(pred_raw_np):.6f}, {np.nanmax(pred_raw_np):.6f}]")
                    print(f"  raw pred_norm <= 0: {(pred_raw_np <= 0).sum()} / {len(pred_raw_np)}")
                    physical_np = outputs_physical.detach().cpu().numpy()
                    print(f"  physical pred_norm < 0: {(physical_np < 0).sum()} / {len(physical_np)}")
                    print(f"  abs(raw pred_norm)<1e-4: {(np.abs(pred_raw_np) < 1e-4).sum()} / {len(pred_raw_np)}")
                    print(f"  target_norm range: [{np.nanmin(target_np):.6f}, {np.nanmax(target_np):.6f}]")
                    print(f"  is_zero_mask==0: {(is_zero_np == 0).sum()} / {len(is_zero_np)}")
                    if has_point_prior:
                        print(
                            f"  prior_col={prior_col} range: "
                            f"[{np.nanmin(prior21_np):.6f}, {np.nanmax(prior21_np):.6f}]"
                        )
                    else:
                        print("  point prior: 未配置（Clean-18D）")
                    print(f"  fused range: [{np.nanmin(fused_np):.6f}, {np.nanmax(fused_np):.6f}]")

                # ===== 保存每个样本的诊断信息 =====
                bs = len(pred_raw_np)
                for i in range(bs):
                    row = {
                        "batch_idx": batch_idx,
                        "local_i": i,
                        "pred_norm_raw": float(pred_raw_np[i]),
                        "target_norm": float(target_np[i]),
                        "fused_value": float(fused_np[i]) if np.isfinite(fused_np[i]) else np.nan,
                        "prior21": float(prior21_np[i]) if np.isfinite(prior21_np[i]) else np.nan,
                        "is_zero_mask": float(is_zero_np[i]),
                    }
                    if sample_indices is not None:
                        row["sample_index"] = int(sample_indices.reshape(-1)[i].item())
                    diag_rows.append(row)

                # ===== 显式 point prior 屏蔽测试 =====
                if ablate_prior_dim and has_point_prior and point_feats.shape[1] > prior_col:
                    # 置0
                    point_no_prior = point_feats.clone()
                    point_no_prior[:, prior_col] = 0.0
                    if is_residual_model:
                        outputs_no_prior, _ = self.model(conv_feats, point_no_prior, raw_fused_swe)
                    elif is_gate_model:
                        outputs_no_prior, _, _, _ = self.model(conv_feats, point_no_prior)
                    else:
                        outputs_no_prior = self.model(conv_feats, point_no_prior)

                    # 置0.5
                    point_mean_prior = point_feats.clone()
                    point_mean_prior[:, prior_col] = 0.5
                    if is_residual_model:
                        outputs_mean_prior, _ = self.model(conv_feats, point_mean_prior, raw_fused_swe)
                    elif is_gate_model:
                        outputs_mean_prior, _, _, _ = self.model(conv_feats, point_mean_prior)
                    else:
                        outputs_mean_prior = self.model(conv_feats, point_mean_prior)

                    outputs_no_prior = torch.clamp(
                        outputs_no_prior.flatten(),
                        min=0.0,
                    )
                    outputs_mean_prior = torch.clamp(
                        outputs_mean_prior.flatten(),
                        min=0.0,
                    )

                    all_no_prior.extend(outputs_no_prior.detach().clone().cpu().numpy())
                    all_prior_05.extend(outputs_mean_prior.detach().clone().cpu().numpy())
                # ===============================================

                # 5. 标签无关的SWE非负物理约束
                outputs = outputs_physical
                targets_f = targets.flatten()
                is_zero_mask_f = is_zero_mask.flatten()

                # 6. 收集最终预测结果
                all_predictions.extend(outputs.detach().clone().cpu().numpy())
                all_targets.extend(targets_f.detach().clone().cpu().numpy())
                all_is_zero.extend(is_zero_mask_f.detach().clone().cpu().numpy())

        # 7. 汇总与 NaN 清洗
        if len(all_predictions) == 0:
            print("  ⚠ 警告: 没有收集到任何预测结果")
            if ablate_prior_dim:
                return None, None, None, None, None
            return None, None, None

        all_predictions = np.array(all_predictions, dtype=np.float32)
        all_targets = np.array(all_targets, dtype=np.float32)
        all_is_zero = np.array(all_is_zero, dtype=np.float32)

        # 移除NaN值
        valid_mask = ~np.isnan(all_predictions) & ~np.isnan(all_targets)
        valid_count = np.sum(valid_mask)

        print(f"  预测完成，总样本: {len(all_predictions)}, 有效样本: {valid_count}")

        if valid_count == 0:
            print("  ⚠ 警告: 没有有效样本（全部为NaN）")
            if ablate_prior_dim:
                return None, None, None, None, None
            return None, None, None

        # ============ 🔥 保存 0 平线诊断 CSV ============
        if diag_rows:
            try:
                diag_df = pd.DataFrame(diag_rows)

                diag_df["pred_mm_raw"] = diag_df["pred_norm_raw"].values * (swe_max - swe_min) + swe_min
                diag_df["target_mm"] = diag_df["target_norm"].values * (swe_max - swe_min) + swe_min
                diag_df["fused_mm"] = diag_df["fused_value"].values * (swe_max - swe_min) + swe_min

                diag_df["low_pred_10"] = diag_df["pred_mm_raw"] <= 10.0
                diag_df["low_pred_15"] = diag_df["pred_mm_raw"] <= 15.0
                diag_df["low_pred_20"] = diag_df["pred_mm_raw"] <= 20.0
                diag_df["low_pred_30"] = diag_df["pred_mm_raw"] <= 30.0
                diag_df["low_pred_40"] = diag_df["pred_mm_raw"] <= 40.0

                diag_df["bad_low20_high50"] = (
                    (diag_df["pred_mm_raw"] <= 20.0) &
                    (diag_df["target_mm"] >= 50.0)
                )
                diag_df["bad_low30_high80"] = (
                    (diag_df["pred_mm_raw"] <= 30.0) &
                    (diag_df["target_mm"] >= 80.0)
                )
                diag_df["bad_low40_high100"] = (
                    (diag_df["pred_mm_raw"] <= 40.0) &
                    (diag_df["target_mm"] >= 100.0)
                )

                diag_df["fused_zero_snow20"] = (
                    (diag_df["fused_mm"] <= 1.0) &
                    (diag_df["target_mm"] >= 20.0)
                )
                diag_df["fused_zero_snow80"] = (
                    (diag_df["fused_mm"] <= 1.0) &
                    (diag_df["target_mm"] >= 80.0)
                )

                save_path = self.save_dir / "low_value_diagnosis.csv"
                diag_df.to_csv(save_path, index=False, encoding="utf-8-sig")

                print("\n【低值带诊断汇总】")
                print(f"  pred<=10mm: {diag_df['low_pred_10'].sum()} / {len(diag_df)}")
                print(f"  pred<=15mm: {diag_df['low_pred_15'].sum()} / {len(diag_df)}")
                print(f"  pred<=20mm: {diag_df['low_pred_20'].sum()} / {len(diag_df)}")
                print(f"  pred<=30mm: {diag_df['low_pred_30'].sum()} / {len(diag_df)}")
                print(f"  pred<=20mm & target>=50mm:  {diag_df['bad_low20_high50'].sum()}")
                print(f"  pred<=30mm & target>=80mm:  {diag_df['bad_low30_high80'].sum()}")
                print(f"  pred<=40mm & target>=100mm: {diag_df['bad_low40_high100'].sum()}")
                print(f"  fused<=1mm & target>=20mm: {diag_df['fused_zero_snow20'].sum()}")
                print(f"  fused<=1mm & target>=80mm: {diag_df['fused_zero_snow80'].sum()}")

                bad = diag_df[diag_df["bad_low20_high50"]]
                if len(bad) > 0:
                    print("\n【bad_low20_high50 样本统计】")
                    print(f"  target mean: {bad['target_mm'].mean():.2f} mm")
                    print(f"  pred mean:   {bad['pred_mm_raw'].mean():.2f} mm")
                    print(f"  fused mean:  {bad['fused_mm'].mean():.2f} mm")
                    if has_point_prior:
                        print(f"  prior(col={prior_col}) mean: {bad['prior21'].mean():.4f}")
                    print(f"  fused<=1mm比例: {(bad['fused_mm'] <= 1.0).mean()*100:.1f}%")

                print(f"  诊断CSV已保存: {save_path}")
            except Exception as e:
                print(f"  ⚠ 保存 low_value_diagnosis.csv 失败: {e}")
        # ===============================================

        if ablate_prior_dim:
            all_no_prior = np.array(all_no_prior, dtype=np.float32)
            all_prior_05 = np.array(all_prior_05, dtype=np.float32)
            return (all_predictions[valid_mask], all_targets[valid_mask], all_is_zero[valid_mask],
                    all_no_prior[valid_mask], all_prior_05[valid_mask])

        return all_predictions[valid_mask], all_targets[valid_mask], all_is_zero[valid_mask]

    def _run_high_swe_transfer_audit(
        self,
        station_ds,
        train_indices,
        inner_indices,
        fold_seed,
    ):
        """
        对既有checkpoint执行完整训练池—Inner高SWE迁移诊断。

        该函数只做model.eval()前向推理：
        - 不创建Outer DataLoader；
        - 不执行optimizer/scheduler step；
        - 不参与checkpoint选择或early stopping；
        - 不修改任何checkpoint。
        """
        checkpoint_specs = list(
            self.config.get("transfer_audit_checkpoints", []) or []
        )
        if not checkpoint_specs:
            raise ValueError(
                "high_swe_transfer_audit_only要求至少提供一个"
                "--transfer_audit_checkpoint NAME=PATH"
            )

        parsed_checkpoints = []
        seen_names = set()
        for spec in checkpoint_specs:
            text = str(spec).strip()
            if "=" not in text:
                raise ValueError(
                    "transfer audit checkpoint格式必须为NAME=PATH: "
                    f"{text!r}"
                )
            raw_name, raw_path = text.split("=", 1)
            safe_name = "".join(
                char if (char.isalnum() or char in "-_") else "_"
                for char in raw_name.strip()
            ).strip("_")
            if not safe_name:
                raise ValueError(f"checkpoint名称无效: {raw_name!r}")
            if safe_name in seen_names:
                raise ValueError(f"checkpoint名称重复: {safe_name}")
            checkpoint_path = Path(raw_path).expanduser().resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"transfer audit checkpoint不存在: {checkpoint_path}"
                )
            seen_names.add(safe_name)
            parsed_checkpoints.append((safe_name, checkpoint_path))

        output_dir = self.save_dir / "high_swe_transfer_audit"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_indices = [int(index) for index in train_indices]
        inner_indices = [int(index) for index in inner_indices]
        if set(train_indices) & set(inner_indices):
            raise RuntimeError("高SWE迁移诊断的Train与Inner索引重叠")

        loader_kwargs = {
            "batch_size": self.config["batch_size"],
            "shuffle": False,
            "num_workers": self.config.get("num_workers", 8),
            "pin_memory": True,
        }
        split_specs = [
            (
                "train",
                train_indices,
                DataLoader(
                    Subset(station_ds, train_indices),
                    **loader_kwargs,
                    **_dataloader_seed_kwargs(int(fold_seed) + 101),
                ),
            ),
            (
                "inner",
                inner_indices,
                DataLoader(
                    Subset(station_ds, inner_indices),
                    **loader_kwargs,
                    **_dataloader_seed_kwargs(int(fold_seed) + 102),
                ),
            ),
        ]

        swe_min = float(getattr(self, "swe_min", 0.0))
        swe_max = float(getattr(self, "swe_max", 400.0))
        swe_range = max(swe_max - swe_min, 1e-8)
        bin_edges = [-np.inf, 20.0, 50.0, 80.0, np.inf]
        bin_labels = ["lt20", "20to50", "50to80", "ge80"]

        def finite_or_none(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        def compute_metrics(frame):
            target = frame["target_mm"].to_numpy(dtype=np.float64)
            prediction = frame["prediction_mm"].to_numpy(dtype=np.float64)
            valid = np.isfinite(target) & np.isfinite(prediction)
            target = target[valid]
            prediction = prediction[valid]
            n_samples = int(target.size)
            n_stations = int(frame.loc[valid, "station_id"].nunique())
            if n_samples == 0:
                return {
                    "n_samples": 0,
                    "n_stations": 0,
                    "target_mean_mm": None,
                    "prediction_mean_mm": None,
                    "rmse_mm": None,
                    "mae_mm": None,
                    "bias_mm": None,
                    "r": None,
                    "nse": None,
                    "slope": None,
                    "std_ratio": None,
                    "ccc": None,
                }

            error = prediction - target
            target_mean = float(np.mean(target))
            prediction_mean = float(np.mean(prediction))
            target_std = float(np.std(target))
            prediction_std = float(np.std(prediction))
            rmse = float(np.sqrt(np.mean(error ** 2)))
            mae = float(np.mean(np.abs(error)))
            bias = float(np.mean(error))

            if (
                n_samples > 1
                and target_std > 1e-12
                and prediction_std > 1e-12
            ):
                correlation = float(
                    np.corrcoef(target, prediction)[0, 1]
                )
            else:
                correlation = float("nan")

            ss_res = float(np.sum(error ** 2))
            ss_tot = float(np.sum((target - target_mean) ** 2))
            nse = (
                1.0 - ss_res / ss_tot
                if ss_tot > 1e-12
                else float("nan")
            )
            slope = self._safe_regression_slope(target, prediction)
            std_ratio = (
                prediction_std / target_std
                if target_std > 1e-12
                else float("nan")
            )
            covariance = float(
                np.mean(
                    (target - target_mean)
                    * (prediction - prediction_mean)
                )
            )
            ccc_denom = (
                target_std ** 2
                + prediction_std ** 2
                + (target_mean - prediction_mean) ** 2
            )
            ccc = (
                2.0 * covariance / ccc_denom
                if ccc_denom > 1e-12
                else float("nan")
            )
            return {
                "n_samples": n_samples,
                "n_stations": n_stations,
                "target_mean_mm": target_mean,
                "prediction_mean_mm": prediction_mean,
                "rmse_mm": rmse,
                "mae_mm": mae,
                "bias_mm": bias,
                "r": finite_or_none(correlation),
                "nse": finite_or_none(nse),
                "slope": finite_or_none(slope),
                "std_ratio": finite_or_none(std_ratio),
                "ccc": finite_or_none(ccc),
            }

        prediction_frames = []
        original_save_dir = self.save_dir
        try:
            for checkpoint_name, checkpoint_path in parsed_checkpoints:
                try:
                    payload = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                        weights_only=False,
                    )
                except TypeError:
                    # 兼容不支持weights_only参数的旧版PyTorch。
                    payload = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                    )
                if isinstance(payload, dict):
                    state_dict = None
                    for state_key in (
                        "model_state_dict",
                        "state_dict",
                        "model",
                    ):
                        if state_key in payload:
                            state_dict = payload[state_key]
                            break
                else:
                    state_dict = payload
                if state_dict is None:
                    raise RuntimeError(
                        f"checkpoint中没有模型权重: {checkpoint_path}"
                    )

                self.model.load_state_dict(state_dict, strict=True)
                self.model.to(self.device)
                self.model.eval()

                for split_name, dataset_indices, loader in split_specs:
                    split_output_dir = (
                        output_dir / checkpoint_name / split_name
                    )
                    split_output_dir.mkdir(parents=True, exist_ok=True)
                    self.save_dir = split_output_dir

                    predictions_norm, targets_norm, _ = (
                        self._make_predictions(loader)
                    )
                    if predictions_norm is None:
                        raise RuntimeError(
                            f"{checkpoint_name}/{split_name}没有预测结果"
                        )
                    predictions_norm = np.asarray(
                        predictions_norm,
                        dtype=np.float64,
                    ).reshape(-1)
                    targets_norm = np.asarray(
                        targets_norm,
                        dtype=np.float64,
                    ).reshape(-1)
                    if (
                        len(predictions_norm) != len(dataset_indices)
                        or len(targets_norm) != len(dataset_indices)
                    ):
                        raise RuntimeError(
                            f"{checkpoint_name}/{split_name}预测数量不一致: "
                            f"pred={len(predictions_norm)}, "
                            f"target={len(targets_norm)}, "
                            f"indices={len(dataset_indices)}"
                        )

                    rows = []
                    for dataset_index, target_norm, prediction_norm in zip(
                        dataset_indices,
                        targets_norm.tolist(),
                        predictions_norm.tolist(),
                    ):
                        meta = station_ds.meta_index[int(dataset_index)]
                        station_id = str(
                            meta.get("station_id", "unknown")
                        ).split(",")[0]
                        label_date_value = meta.get(
                            "label_date",
                            meta.get("feature_date", meta.get("date")),
                        )
                        feature_date_value = meta.get(
                            "feature_date",
                            meta.get("label_date", meta.get("date")),
                        )
                        label_date = (
                            pd.to_datetime(label_date_value).strftime(
                                "%Y-%m-%d"
                            )
                            if label_date_value is not None
                            else ""
                        )
                        feature_date = (
                            pd.to_datetime(feature_date_value).strftime(
                                "%Y-%m-%d"
                            )
                            if feature_date_value is not None
                            else ""
                        )
                        target_mm = float(target_norm * swe_range + swe_min)
                        prediction_mm = float(
                            prediction_norm * swe_range + swe_min
                        )
                        rows.append({
                            "checkpoint": checkpoint_name,
                            "checkpoint_path": str(checkpoint_path),
                            "split": split_name,
                            "dataset_index": int(dataset_index),
                            "station_id": station_id,
                            "label_date": label_date,
                            "feature_date": feature_date,
                            "row": meta.get("row"),
                            "col": meta.get("col"),
                            "target_mm": target_mm,
                            "prediction_mm": prediction_mm,
                            "error_mm": prediction_mm - target_mm,
                            "absolute_error_mm": abs(
                                prediction_mm - target_mm
                            ),
                        })

                    split_frame = pd.DataFrame(rows)
                    split_frame["swe_bin"] = pd.cut(
                        split_frame["target_mm"],
                        bins=bin_edges,
                        labels=bin_labels,
                        right=False,
                        include_lowest=True,
                    ).astype(str)
                    prediction_frames.append(split_frame)
        finally:
            self.save_dir = original_save_dir

        predictions_frame = pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        predictions_path = output_dir / "predictions_all.csv"
        predictions_frame.to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )

        bin_rows = []
        station_rows = []
        for (checkpoint_name, split_name), split_frame in (
            predictions_frame.groupby(
                ["checkpoint", "split"],
                sort=False,
            )
        ):
            overall_metrics = compute_metrics(split_frame)
            bin_rows.append({
                "checkpoint": checkpoint_name,
                "split": split_name,
                "swe_bin": "all",
                **overall_metrics,
            })
            for bin_name in bin_labels:
                bin_frame = split_frame[
                    split_frame["swe_bin"] == bin_name
                ]
                bin_rows.append({
                    "checkpoint": checkpoint_name,
                    "split": split_name,
                    "swe_bin": bin_name,
                    **compute_metrics(bin_frame),
                })

            for station_id, station_frame in split_frame.groupby(
                "station_id",
                sort=True,
            ):
                metrics = compute_metrics(station_frame)
                station_rows.append({
                    "checkpoint": checkpoint_name,
                    "split": split_name,
                    "station_id": station_id,
                    "max_target_mm": float(
                        station_frame["target_mm"].max()
                    ),
                    "n_ge80": int(
                        (station_frame["target_mm"] >= 80.0).sum()
                    ),
                    **metrics,
                })

        bin_summary = pd.DataFrame(bin_rows)
        station_summary = pd.DataFrame(station_rows)
        bin_summary_path = output_dir / "swe_bin_summary.csv"
        station_summary_path = output_dir / "station_summary.csv"
        bin_summary.to_csv(
            bin_summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        station_summary.to_csv(
            station_summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        high_frame = bin_summary[
            bin_summary["swe_bin"] == "ge80"
        ].copy()
        high_compare_path = (
            output_dir / "high_swe_train_vs_inner.csv"
        )
        high_frame.to_csv(
            high_compare_path,
            index=False,
            encoding="utf-8-sig",
        )

        def json_safe_records(frame):
            records = []
            for record in frame.to_dict(orient="records"):
                safe_record = {}
                for key, value in record.items():
                    if isinstance(value, np.generic):
                        value = value.item()
                    if value is None or pd.isna(value):
                        safe_record[key] = None
                    else:
                        safe_record[key] = value
                records.append(safe_record)
            return records

        summary_payload = {
            "mode": "high_swe_transfer_audit_only",
            "training_performed": False,
            "optimizer_step_performed": False,
            "checkpoint_selection_performed": False,
            "outer_fold_dataloader_created": False,
            "outer_fold_evaluated": False,
            "fixed_test_evaluated": False,
            "external_test_evaluated": False,
            "n_train_samples": len(train_indices),
            "n_inner_samples": len(inner_indices),
            "checkpoints": [
                {
                    "name": name,
                    "path": str(path),
                }
                for name, path in parsed_checkpoints
            ],
            "files": {
                "predictions_all": str(predictions_path),
                "swe_bin_summary": str(bin_summary_path),
                "station_summary": str(station_summary_path),
                "high_swe_train_vs_inner": str(high_compare_path),
            },
            "ge80_summary": json_safe_records(high_frame),
        }
        summary_path = output_dir / "audit_summary.json"
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(
                summary_payload,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        print("\n" + "=" * 78)
        print("✅ 完整训练池—Inner高SWE迁移诊断完成")
        print("   training_performed=False")
        print("   optimizer_step_performed=False")
        print("   outer_fold_dataloader_created=False")
        print("   outer_fold_evaluated=False")
        print(f"   逐样本: {predictions_path}")
        print(f"   分SWE区间: {bin_summary_path}")
        print(f"   分站点: {station_summary_path}")
        print(f"   ≥80mm Train-vs-Inner: {high_compare_path}")
        print("=" * 78)

        display_columns = [
            "checkpoint",
            "split",
            "n_samples",
            "n_stations",
            "target_mean_mm",
            "prediction_mean_mm",
            "rmse_mm",
            "mae_mm",
            "bias_mm",
            "r",
            "slope",
            "std_ratio",
        ]
        print(high_frame[display_columns].to_string(index=False))
        return summary_payload
    
    def _init_new_pointencoder_weights(self, model):
        print("  [跳过] 干净版不再初始化新增点特征维度")
        return
    
    def _compute_advanced_metrics(self, y_true, y_pred, high_threshold=80.0):
        """
        计算反归一化后的完整诊断指标。
        y_true, y_pred 必须已经是 mm 单位。
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) < 2:
            return None

        err = y_pred - y_true

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        bias = np.mean(err)

        y_mean = np.mean(y_true)
        y_std = np.std(y_true, ddof=0)
        pred_std = np.std(y_pred, ddof=0)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if y_std > 0 and pred_std > 0:
            r = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            r = np.nan

        r_squared = r ** 2 if np.isfinite(r) else np.nan

        alpha = pred_std / y_std if y_std > 0 else np.nan
        beta = bias / y_std if y_std > 0 else np.nan

        # 回归线：pred = intercept + slope * obs
        if y_std > 0:
            slope = np.cov(y_true, y_pred, ddof=0)[0, 1] / (np.var(y_true, ddof=0) + 1e-12)
            intercept = np.mean(y_pred) - slope * np.mean(y_true)
        else:
            slope = np.nan
            intercept = np.nan

        # 后校准：obs = cal_intercept + cal_slope * pred
        if pred_std > 0:
            cal_slope = np.cov(y_pred, y_true, ddof=0)[0, 1] / (np.var(y_pred, ddof=0) + 1e-12)
            cal_intercept = np.mean(y_true) - cal_slope * np.mean(y_pred)
            y_cal = cal_intercept + cal_slope * y_pred
            cal_nse = r2_score(y_true, y_cal)
            cal_r = np.corrcoef(y_true, y_cal)[0, 1] if np.std(y_cal) > 0 else np.nan
        else:
            cal_slope = np.nan
            cal_intercept = np.nan
            cal_nse = np.nan
            cal_r = np.nan

        # 高 SWE 子集
        high_mask = y_true >= high_threshold
        if np.sum(high_mask) > 0:
            high_err = y_pred[high_mask] - y_true[high_mask]
            high_info = {
                "threshold": float(high_threshold),
                "n": int(np.sum(high_mask)),
                "ratio": float(np.sum(high_mask) / len(y_true)),
                "mae": float(np.mean(np.abs(high_err))),
                "rmse": float(np.sqrt(np.mean(high_err ** 2))),
                "bias": float(np.mean(high_err)),
                "target_mean": float(np.mean(y_true[high_mask])),
                "pred_mean": float(np.mean(y_pred[high_mask])),
                "target_range": [float(np.min(y_true[high_mask])), float(np.max(y_true[high_mask]))],
                "pred_range": [float(np.min(y_pred[high_mask])), float(np.max(y_pred[high_mask]))],
            }
        else:
            high_info = {
                "threshold": float(high_threshold),
                "n": 0,
                "ratio": 0.0,
                "mae": None,
                "rmse": None,
                "bias": None,
                "target_mean": None,
                "pred_mean": None,
                "target_range": None,
                "pred_range": None,
            }

        return {
            "n": int(len(y_true)),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "bias": float(bias),

            # 为了兼容旧代码，r2 仍保留，但这里明确等价于 NSE/error-based R2
            "nse": float(nse),
            "r2": float(nse),
            "r": float(r) if np.isfinite(r) else None,
            "r_squared": float(r_squared) if np.isfinite(r_squared) else None,

            "alpha": float(alpha) if np.isfinite(alpha) else None,
            "beta": float(beta) if np.isfinite(beta) else None,
            "slope": float(slope) if np.isfinite(slope) else None,
            "intercept": float(intercept) if np.isfinite(intercept) else None,

            "target_min": float(np.min(y_true)),
            "target_max": float(np.max(y_true)),
            "target_mean": float(np.mean(y_true)),
            "target_std": float(y_std),

            "pred_min": float(np.min(y_pred)),
            "pred_max": float(np.max(y_pred)),
            "pred_mean": float(np.mean(y_pred)),
            "pred_std": float(pred_std),

            "calibrated": {
                "nse": float(cal_nse) if np.isfinite(cal_nse) else None,
                "r": float(cal_r) if np.isfinite(cal_r) else None,
                "slope": float(cal_slope) if np.isfinite(cal_slope) else None,
                "intercept": float(cal_intercept) if np.isfinite(cal_intercept) else None,
            },

            "high_swe": high_info,
        }


    def _compute_metrics(self, predictions, targets):
        """
        计算评估指标。
        输入 predictions / targets 是归一化值，函数内部反归一化到 mm。
        """
        print("  计算评估指标...")

        if predictions is None or targets is None:
            print("  ✗ 预测数据或目标数据为空")
            return None

        predictions = np.asarray(predictions).ravel()
        targets = np.asarray(targets).ravel()

        if len(predictions) == 0 or len(targets) == 0:
            print("  ✗ 预测数据或目标数据长度为零")
            return None

        swe_min = getattr(self, "swe_min", 0.0)
        swe_max = getattr(self, "swe_max", 200.0)

        print(f"  反归一化参数: min={swe_min:.2f}, max={swe_max:.2f}")

        predictions_denorm = predictions * (swe_max - swe_min) + swe_min
        targets_denorm = targets * (swe_max - swe_min) + swe_min

        print("  归一化范围检查:")
        print(f"    归一化预测值: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"    归一化真实值: [{targets.min():.4f}, {targets.max():.4f}]")

        print("  反归一化范围检查:")
        print(f"    反归一化预测值: [{predictions_denorm.min():.2f}, {predictions_denorm.max():.2f}] mm")
        print(f"    反归一化真实值: [{targets_denorm.min():.2f}, {targets_denorm.max():.2f}] mm")

        eval_results = self._compute_advanced_metrics(
            y_true=targets_denorm,
            y_pred=predictions_denorm,
            high_threshold=80.0
        )

        if eval_results is None:
            print("  ✗ 高级指标计算失败")
            return None

        eval_results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        eval_results["swe_min"] = float(swe_min)
        eval_results["swe_max"] = float(swe_max)

        # 兼容旧字段
        eval_results["num_samples"] = eval_results["n"]

        eval_results["predictions_denorm_stats"] = {
            "min": eval_results["pred_min"],
            "max": eval_results["pred_max"],
            "mean": eval_results["pred_mean"],
            "std": eval_results["pred_std"],
        }

        eval_results["targets_denorm_stats"] = {
            "min": eval_results["target_min"],
            "max": eval_results["target_max"],
            "mean": eval_results["target_mean"],
            "std": eval_results["target_std"],
        }

        eval_results["predictions_norm_stats"] = {
            "min": float(predictions.min()),
            "max": float(predictions.max()),
            "mean": float(predictions.mean()),
            "std": float(predictions.std()),
        }

        eval_results["targets_norm_stats"] = {
            "min": float(targets.min()),
            "max": float(targets.max()),
            "mean": float(targets.mean()),
            "std": float(targets.std()),
        }

        print("\n  评估结果（反归一化后）:")
        print(f"    NSE/R²: {eval_results['nse']:.6f}")
        print(f"    R:      {eval_results['r']:.6f}")
        print(f"    RMSE:   {eval_results['rmse']:.6f}")
        print(f"    MAE:    {eval_results['mae']:.6f}")
        print(f"    Bias:   {eval_results['bias']:.6f}")
        print(f"    alpha:  {eval_results['alpha']:.6f}")
        print(f"    beta:   {eval_results['beta']:.6f}")
        print(f"    slope:  {eval_results['slope']:.6f}")
        print(f"    intercept: {eval_results['intercept']:.6f}")
        print(f"    pred range:   [{eval_results['pred_min']:.2f}, {eval_results['pred_max']:.2f}] mm")
        print(f"    target range: [{eval_results['target_min']:.2f}, {eval_results['target_max']:.2f}] mm")

        hs = eval_results["high_swe"]
        print(f"\n  高 SWE 子集 obs >= {hs['threshold']:.0f} mm:")
        print(f"    N: {hs['n']}")
        if hs["n"] > 0:
            print(f"    MAE:  {hs['mae']:.2f} mm")
            print(f"    RMSE: {hs['rmse']:.2f} mm")
            print(f"    Bias: {hs['bias']:.2f} mm")
            print(f"    obs mean:  {hs['target_mean']:.2f} mm")
            print(f"    pred mean: {hs['pred_mean']:.2f} mm")

        return eval_results, predictions_denorm, targets_denorm
    
    
    def _get_loader_predictions_mm(self, loader, split_name="unknown"):
        """
        用当前模型对一个 loader 做预测，并反归一化到 mm。
        适用于 Frozen / Fine-tuned / 任意当前 self.model。
        """

        if loader is None:
            print(f"[{split_name}] loader 为空")
            return None, None

        print(f"\n[{split_name.upper()}] 获取模型预测...")

        self.model.eval()

        result = self._make_predictions(loader)

        if result is None:
            print(f"[{split_name}] _make_predictions 返回 None")
            return None, None

        if len(result) == 3:
            preds_norm, targets_norm, _ = result
        else:
            preds_norm, targets_norm = result[:2]

        if preds_norm is None or targets_norm is None:
            print(f"[{split_name}] 预测或目标为空")
            return None, None

        preds_norm = np.asarray(preds_norm).reshape(-1)
        targets_norm = np.asarray(targets_norm).reshape(-1)

        swe_min = getattr(self, "swe_min", 0.0)
        swe_max = getattr(self, "swe_max", 200.0)

        preds_mm = preds_norm * (swe_max - swe_min) + swe_min
        targets_mm = targets_norm * (swe_max - swe_min) + swe_min

        valid = np.isfinite(preds_mm) & np.isfinite(targets_mm)
        preds_mm = preds_mm[valid]
        targets_mm = targets_mm[valid]

        print(f"[{split_name.upper()}] 有效样本: {len(preds_mm)}")
        print(f"  pred range: [{preds_mm.min():.2f}, {preds_mm.max():.2f}] mm")
        print(f"  obs  range: [{targets_mm.min():.2f}, {targets_mm.max():.2f}] mm")

        return preds_mm, targets_mm
    
    
    
    def _fit_linear_calibration(self, y_pred_train, y_true_train):
        """
        拟合线性校正:
            y_true = a * y_pred + b

        只允许用 train 集拟合，不能用 test。
        """

        y_pred_train = np.asarray(y_pred_train).reshape(-1)
        y_true_train = np.asarray(y_true_train).reshape(-1)

        valid = np.isfinite(y_pred_train) & np.isfinite(y_true_train)
        y_pred_train = y_pred_train[valid]
        y_true_train = y_true_train[valid]

        if len(y_pred_train) < 2:
            raise ValueError("训练样本不足，无法拟合线性校正")

        if np.std(y_pred_train) < 1e-8:
            raise ValueError("训练集预测值方差太小，无法拟合线性校正")

        # 拟合 y_true = a * y_pred + b
        a, b = np.polyfit(y_pred_train, y_true_train, deg=1)

        print("\n" + "=" * 70)
        print("Frozen + Linear Calibration 参数")
        print("=" * 70)
        print(f"  y_cal = a * y_frozen + b")
        print(f"  a = {a:.6f}")
        print(f"  b = {b:.6f}")
        print("=" * 70)

        return float(a), float(b)
    
    def evaluate_frozen_linear_calibration(self):
        """
        Frozen + Linear Calibration 基线。

        流程:
        1. 用当前 frozen 模型分别预测 train/val/test
        2. 只用 train 拟合 y_cal = a * y0 + b
        3. 在 train/val/test 上评估校正后结果
        4. 保存 json 和测试集散点图

        注意:
        - 不能用 test 拟合 a,b
        - 这个函数默认 self.model 已经是 frozen 预训练模型
        """

        print("\n" + "█" * 80)
        print("运行 Frozen + Linear Calibration")
        print("█" * 80)

        if self.model is None:
            print("✗ 当前没有模型，无法运行 Frozen + Linear Calibration")
            return None

        if self.train_loader is None:
            print("✗ train_loader 为空，无法拟合校正层")
            return None

        # 1. 获取 train/val/test frozen 预测
        train_pred, train_obs = self._get_loader_predictions_mm(
            self.train_loader,
            split_name="train"
        )

        val_pred, val_obs = None, None
        if hasattr(self, "val_loader") and self.val_loader is not None:
            val_pred, val_obs = self._get_loader_predictions_mm(
                self.val_loader,
                split_name="val"
            )

        test_pred, test_obs = None, None
        if hasattr(self, "test_loader") and self.test_loader is not None:
            test_pred, test_obs = self._get_loader_predictions_mm(
                self.test_loader,
                split_name="test"
            )

        if train_pred is None or train_obs is None:
            print("✗ 无法获取训练集预测，终止")
            return None

        # 2. 只用训练集拟合 a,b
        a, b = self._fit_linear_calibration(train_pred, train_obs)

        def apply_calibration(y_pred):
            y_cal = a * y_pred + b

            # SWE 不能为负，先做最简单物理约束
            y_cal = np.maximum(y_cal, 0.0)

            return y_cal

        # 3. 分别评估原 frozen 和 calibrated
        results = {
            "method": "Frozen + Linear Calibration",
            "formula": "y_cal = max(0, a * y_frozen + b)",
            "a": float(a),
            "b": float(b),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "splits": {}
        }

        split_data = {
            "train": (train_pred, train_obs),
            "val": (val_pred, val_obs),
            "test": (test_pred, test_obs),
        }

        for split_name, data in split_data.items():
            y0, y_true = data

            if y0 is None or y_true is None:
                continue

            y_cal = apply_calibration(y0)

            frozen_metrics = self._compute_advanced_metrics(
                y_true=y_true,
                y_pred=y0,
                high_threshold=80.0
            )

            calibrated_metrics = self._compute_advanced_metrics(
                y_true=y_true,
                y_pred=y_cal,
                high_threshold=80.0
            )

            results["splits"][split_name] = {
                "frozen": frozen_metrics,
                "linear_calibrated": calibrated_metrics
            }

            print("\n" + "-" * 80)
            print(f"[{split_name.upper()}] Frozen vs Linear Calibration")
            print("-" * 80)

            print(
                f"Frozen: "
                f"R={frozen_metrics['r']:.4f}, "
                f"NSE={frozen_metrics['nse']:.4f}, "
                f"RMSE={frozen_metrics['rmse']:.2f}, "
                f"MAE={frozen_metrics['mae']:.2f}, "
                f"Bias={frozen_metrics['bias']:.2f}, "
                f"alpha={frozen_metrics['alpha']:.3f}, "
                f"slope={frozen_metrics['slope']:.3f}"
            )

            print(
                f"Calibrated: "
                f"R={calibrated_metrics['r']:.4f}, "
                f"NSE={calibrated_metrics['nse']:.4f}, "
                f"RMSE={calibrated_metrics['rmse']:.2f}, "
                f"MAE={calibrated_metrics['mae']:.2f}, "
                f"Bias={calibrated_metrics['bias']:.2f}, "
                f"alpha={calibrated_metrics['alpha']:.3f}, "
                f"slope={calibrated_metrics['slope']:.3f}"
            )

            hs_frozen = frozen_metrics["high_swe"]
            hs_cal = calibrated_metrics["high_swe"]

            if hs_frozen["n"] > 0:
                print(
                    f"High SWE obs>=80:"
                    f"\n  Frozen     Bias={hs_frozen['bias']:.2f}, "
                    f"MAE={hs_frozen['mae']:.2f}, "
                    f"pred_mean={hs_frozen['pred_mean']:.2f}"
                    f"\n  Calibrated Bias={hs_cal['bias']:.2f}, "
                    f"MAE={hs_cal['mae']:.2f}, "
                    f"pred_mean={hs_cal['pred_mean']:.2f}"
                )

            # 4. 给 test 画一张校正后散点图
            if split_name == "test":
                self.plot_density_scatter_hardcode(
                    y_cal,
                    y_true,
                    is_fine_tune=True,
                    use_raw=True,
                    fold_index=None
                )

        # 5. 保存结果
        save_path = self.save_dir / "frozen_linear_calibration_results.json"

        def sanitize(obj):

            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            if isinstance(obj, tuple):
                return [sanitize(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return sanitize(obj.tolist())
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                v = float(obj)
                if np.isnan(v) or np.isinf(v):
                    return None
                return v
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            return obj

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sanitize(results), f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print("Frozen + Linear Calibration 完成")
        print(f"结果已保存: {save_path}")
        print("=" * 80)

        return results 
   


    def _compute_nse_oriented_loss(self, outputs_flat, targets_flat, point_feats_for_prior, 
                                    epoch, batch_idx, is_mixed_mode=False, source_flag=None, 
                                    use_curriculum=False, indices=None, threshold=None,
                                    is_residual_model=False, delta_y=None,
                                    is_gate_model=False, alpha=None, y_pre=None, y_fine=None):
        """
        计算 NSE-oriented loss（可复用版本）

        Returns:
            loss: 标量张量
            prior_mm: 用于诊断
        """
        swe_min = getattr(self, "swe_min", 0.0)
        swe_max = getattr(self, "swe_max", 170.0)
        swe_range = swe_max - swe_min

        target_mm = targets_flat * swe_range + swe_min

        # 第21维 prior
        prior_col = self.config.get("physical_prior_col", None)
        prior_norm = None
        prior_mm = None

        if isinstance(prior_col, int) and point_feats_for_prior is not None and point_feats_for_prior.dim() == 2 and point_feats_for_prior.shape[1] > prior_col:
            prior_norm = point_feats_for_prior[:, prior_col].detach().reshape(-1)
            prior_mm = prior_norm * swe_range + swe_min

        # 1. MSE 主导，Huber 辅助稳定
        mse_each = F.mse_loss(outputs_flat, targets_flat, reduction="none")
        huber_each = F.smooth_l1_loss(
            outputs_flat,
            targets_flat,
            beta=0.01,
            reduction="none"
        )

        loss_each = 0.8 * mse_each + 0.2 * huber_each

        # 基础权重
        weights = torch.ones_like(loss_each)

        # 2. 高 SWE 样本加权
        weights = weights + 1.0 * (target_mm >= 20.0).float()
        weights = weights + 2.0 * (target_mm >= 50.0).float()
        weights = weights + 4.0 * (target_mm >= 80.0).float()

        # FusedSWE/prior 接近0，但站点有雪：重点惩罚
        if prior_mm is not None and prior_mm.numel() == targets_flat.numel():
            bad_prior20 = (prior_mm <= 1.0) & (target_mm >= 20.0)
            bad_prior50 = (prior_mm <= 1.0) & (target_mm >= 50.0)
            bad_prior80 = (prior_mm <= 1.0) & (target_mm >= 80.0)

            weights = weights + 2.0 * bad_prior20.float()
            weights = weights + 4.0 * bad_prior50.float()
            weights = weights + 10.0 * bad_prior80.float()

            if epoch == 0 and batch_idx == 0:
                print("  🔥 Bad-prior snow loss weighting:")
                print(f"    prior<=1 & target>=20mm: {int(bad_prior20.sum().item())}")
                print(f"    prior<=1 & target>=50mm: {int(bad_prior50.sum().item())}")
                print(f"    prior<=1 & target>=80mm: {int(bad_prior80.sum().item())}")

        # 课程学习权重
        if use_curriculum and indices is not None:
            try:
                if torch.is_tensor(indices):
                    indices_np = indices.detach().cpu().numpy()
                else:
                    indices_np = np.array(indices)

                batch_difficulties = self.sample_difficulties[indices_np]
                curriculum_weights = (
                    torch.from_numpy(batch_difficulties)
                    .to(self.device)
                    .reshape(-1)
                )
                curriculum_weights = (curriculum_weights <= threshold).float()

                if curriculum_weights.sum() < 1:
                    curriculum_weights = torch.ones_like(curriculum_weights) * 0.1

                if curriculum_weights.numel() == weights.numel():
                    weights = weights * curriculum_weights

            except Exception as e:
                if batch_idx == 0:
                    print(f"  ⚠ 课程学习权重计算失败: {e}")

        # mixed mode source-aware 权重
        if is_mixed_mode and source_flag is not None:
            source_flag = source_flag.reshape(-1).long()

            if source_flag.numel() == loss_each.numel():
                pretrain_loss_weight = float(self.config.get("pretrain_loss_weight", 0.0))

                station_mask = source_flag == 0
                pretrain_mask = source_flag == 1

                weights = torch.where(
                    pretrain_mask,
                    torch.full_like(weights, pretrain_loss_weight),
                    weights
                )

                if epoch == 0 and batch_idx == 0:
                    print(
                        f"    [Mixed Loss] station={station_mask.sum().item()}, "
                        f"pretrain={pretrain_mask.sum().item()}, "
                        f"pretrain_loss_weight={pretrain_loss_weight}"
                    )

        # 权重归一化
        weights = weights / weights.mean().clamp_min(1e-6)
        base_loss = (loss_each * weights).mean()

        # 3. Bias 约束
        err = outputs_flat - targets_flat
        global_bias_loss = err.mean().pow(2)

        high_mask = target_mm >= 80.0
        if high_mask.sum() >= 2:
            high_bias_loss = err[high_mask].mean().pow(2)
        else:
            high_bias_loss = torch.tensor(0.0, device=outputs_flat.device)

        if prior_mm is not None and prior_mm.numel() == targets_flat.numel():
            bad_prior_high = (prior_mm <= 1.0) & (target_mm >= 50.0)
            if bad_prior_high.sum() >= 2:
                bad_prior_bias_loss = err[bad_prior_high].mean().pow(2)
            else:
                bad_prior_bias_loss = torch.tensor(0.0, device=outputs_flat.device)
        else:
            bad_prior_bias_loss = torch.tensor(0.0, device=outputs_flat.device)

        # 4. 方差约束
        target_std = targets_flat.std()
        pred_std = outputs_flat.std()

        if target_std > 1e-6:
            var_loss = torch.relu(0.8 * target_std - pred_std).pow(2)
        else:
            var_loss = torch.tensor(0.0, device=outputs_flat.device)

        loss = (
            base_loss
            + 0.5 * global_bias_loss
            + 1.5 * high_bias_loss
            + 1.0 * bad_prior_bias_loss
            + 0.1 * var_loss
        )

        # 残差模式弹性约束
        if is_residual_model and delta_y is not None:
            elastic_loss = torch.mean(torch.abs(delta_y))
            lambda_elastic = self.config.get("lambda_elastic", 0.1)
            loss = loss + lambda_elastic * elastic_loss

        # 门控模式约束
        if is_gate_model and alpha is not None and y_pre is not None and y_fine is not None:
            with torch.no_grad():
                error_pre = torch.abs(y_pre.reshape(-1) - targets_flat)
                error_fine = torch.abs(y_fine.reshape(-1) - targets_flat)
                target_alpha = (error_pre < error_fine).float()

            gate_loss = F.binary_cross_entropy(
                alpha.reshape(-1),
                target_alpha
            )
            loss = loss + 0.1 * gate_loss

        return loss, prior_mm


    def analyze_test_set_features(self, test_loader=None, model_path=None, pretrained_model_path=None,
                                          predictions_denorm=None, targets_denorm=None, fused_denorm=None):
        """
        完整分析测试集63个样本的所有特征
        生成详细的CSV报告和可视化
        """
        print("\n" + "="*80)
        print("📊 测试集63个样本完整特征分析（含预训练对比 + FusedSWE原始值）")
        print("="*80)

        try:
            # 获取测试集数据加载器
            if test_loader is None:
                if hasattr(self, 'test_loader') and self.test_loader is not None:
                    test_loader = self.test_loader
                    print(f"  使用测试集，样本数: {len(test_loader.dataset)}")
                else:
                    print("❌ 没有测试集数据加载器")
                    return

            # ============ 加载微调模型（当前模型） ============
            if model_path:
                print(f"\n🔍 加载微调模型: {model_path}")
                self._load_model_for_evaluation(model_path)

            # ============ 加载预训练模型（用于对比） ============
            pretrained_model = None
            if pretrained_model_path and os.path.exists(pretrained_model_path):
                print(f"\n🔍 加载预训练模型: {pretrained_model_path}")
                from models_swe import create_model

                if hasattr(test_loader.dataset, 'dataset'):
                    dataset_obj = test_loader.dataset.dataset
                else:
                    dataset_obj = test_loader.dataset

                print(f"  数据集维度: C_conv={dataset_obj.C_conv}, C_point={dataset_obj.C_point}")

                pretrained_model = create_model(
                    model_type=self.config["model_type"],
                    C_spatial=dataset_obj.C_conv,
                    C_point=dataset_obj.C_point,
                    d_model=self.config["d_model"],
                    use_wide_branch=False,
                )

                print(f"  加载预训练权重...")
                checkpoint = torch.load(pretrained_model_path, map_location='cpu')

                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[name] = v

                try:
                    pretrained_model.load_state_dict(new_state_dict, strict=False)
                    print(f"    ✓ 权重加载成功")
                except Exception as e:
                    print(f"    ✗ 权重加载失败: {e}")

                pretrained_model = pretrained_model.to(self.device)
                pretrained_model.eval()
                print(f"  ✓ 预训练模型已移动到设备")
            else:
                print(f"  ⚠ 预训练模型未加载: path={pretrained_model_path}")

            # ============ 获取数据集对象（兼容混合模式） ============
            if hasattr(test_loader.dataset, 'dataset'):
                base_dataset = test_loader.dataset.dataset
                indices = test_loader.dataset.indices
                print(f"\n  数据集类型: Subset, 原始数据集样本数: {len(base_dataset)}")
            else:
                base_dataset = test_loader.dataset
                indices = range(len(base_dataset))
                print(f"\n  数据集类型: 直接数据集, 样本数: {len(base_dataset)}")

            # ============ 🆕 兼容混合模式：提取真正的元数据源 ============
            if hasattr(base_dataset, 'station_dataset'):
                # 混合模式：使用内部的 station_dataset
                dataset = base_dataset.station_dataset
                print(f"  检测到混合模式，使用 station_dataset 作为元数据源")
            else:
                dataset = base_dataset
            # ============================================================

            print(f"\n1. 收集测试集 {len(indices)} 个样本的完整信息...")

            # 存储所有样本的完整信息
            samples_data = []

            self.model.eval()

            swe_min = getattr(self, 'swe_min', 0.0)
            swe_max = getattr(self, 'swe_max', 200.0)
            print(f"  反归一化参数: min={swe_min:.2f}, max={swe_max:.2f}")

            total_samples = len(indices)
            processed = 0
            pretrain_success = 0
            pretrain_fail = 0
            fused_success = 0
            fused_fail = 0

            # 复用evaluate_fine_tune中唯一一次Test预测
            print("\n  【复用已缓存的测试预测】")

            if (
                predictions_denorm is None
                or targets_denorm is None
            ):
                raise RuntimeError(
                    "缺少缓存的测试预测，"
                    "禁止在特征分析中重新执行模型前向"
                )

            finetune_pred_raw = np.asarray(
                predictions_denorm,
                dtype=np.float64,
            ).reshape(-1)

            finetune_targets_raw = np.asarray(
                targets_denorm,
                dtype=np.float64,
            ).reshape(-1)

            if (
                len(finetune_pred_raw) != total_samples
                or len(finetune_targets_raw) != total_samples
            ):
                raise RuntimeError(
                    "缓存预测数量与Test Loader不一致: "
                    f"pred={len(finetune_pred_raw)}, "
                    f"target={len(finetune_targets_raw)}, "
                    f"loader={total_samples}"
                )

            swe_range = max(
                float(swe_max - swe_min),
                1e-12,
            )

            finetune_pred_norm = (
                finetune_pred_raw - swe_min
            ) / swe_range

            finetune_targets_norm = (
                finetune_targets_raw - swe_min
            ) / swe_range

            print(
                f"  ✓ 复用测试预测: "
                f"{len(finetune_pred_raw)}个样本"
            )

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    # 🔥 兼容不同长度的 batch_data
                    if len(batch_data) >= 6:
                        conv_feats, point_feats, targets, is_zero_mask, raw_fused, batch_indices = batch_data[:6]
                    elif len(batch_data) >= 5:
                        conv_feats, point_feats, targets, is_zero_mask, raw_fused = batch_data[:5]
                    elif len(batch_data) >= 4:
                        conv_feats, point_feats, targets, is_zero_mask = batch_data[:4]
                        raw_fused = None
                    else:
                        conv_feats, point_feats, targets = batch_data[:3]
                        is_zero_mask = (targets > 0).float()
                        raw_fused = None

                    batch_size = len(targets)
                    start_idx = batch_idx * test_loader.batch_size

                    for i in range(batch_size):
                        if start_idx + i < len(indices):
                            meta_idx = indices[start_idx + i]
                            if meta_idx < len(dataset.meta_index):
                                meta = dataset.meta_index[meta_idx]

                                # 🔥 兼容不同的日期键名
                                date_val = meta.get('feature_date') or meta.get('label_date') or meta.get('date')
                                if date_val is None:
                                    continue
                                date = date_val
                                r = meta.get('row', 0)
                                c = meta.get('col', 0)

                                # ============ 获取FusedSWE原始值 ============
                                fused_swe_raw = None
                                if hasattr(dataset, 'label_data'):
                                    matched = False

                                    if date in dataset.label_data:
                                        label_arr, label_nodata = dataset.label_data[date]
                                        if 0 <= r < label_arr.shape[0] and 0 <= c < label_arr.shape[1]:
                                            val = label_arr[r, c]
                                            if label_nodata is None or val != label_nodata:
                                                fused_swe_raw = float(val)
                                                fused_success += 1
                                                matched = True

                                    if not matched:
                                        date_str = date.strftime('%Y-%m-%d')
                                        for key, (label_arr, label_nodata) in dataset.label_data.items():
                                            if isinstance(key, str) and key == date_str:
                                                if 0 <= r < label_arr.shape[0] and 0 <= c < label_arr.shape[1]:
                                                    val = label_arr[r, c]
                                                    if label_nodata is None or val != label_nodata:
                                                        fused_swe_raw = float(val)
                                                        fused_success += 1
                                                        matched = True
                                                break

                                    if not matched:
                                        fused_fail += 1

                                # ============ 构建样本信息 ============
                                sample_info = {
                                    '样本索引': meta_idx,
                                    '站点ID': meta.get('station_id', 'unknown'),
                                    '日期': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                                    'DOY': date.timetuple().tm_yday if hasattr(date, 'timetuple') else 0,
                                    '原始经度': meta.get('original_longitude', 0),
                                    '原始纬度': meta.get('original_latitude', 0),
                                    '行列号_row': r,
                                    '行列号_col': c,
                                    '站点SWE_raw': meta.get('swe', 0),
                                    'FusedSWE_raw': fused_swe_raw,
                                    '站点SWE_norm': targets[i].item(),
                                    '微调模型预测_norm': finetune_pred_norm[start_idx + i] if start_idx + i < len(finetune_pred_norm) else None,
                                    '微调模型预测_raw': finetune_pred_raw[start_idx + i] if start_idx + i < len(finetune_pred_raw) else None,
                                    '预训练模型预测_norm': None,
                                    '预训练模型预测_raw': None,
                                }

                                # ============ 卷积特征原始值 ============
                                date_idx = dataset.date_to_index.get(date) if hasattr(dataset, 'date_to_index') else None

                                if date_idx is not None and hasattr(dataset, 'conv_dyn_data'):
                                    if 'chelsa_sfxwind' in dataset.conv_dyn_data:
                                        sample_info['chelsa_sfxwind'] = dataset.conv_dyn_data['chelsa_sfxwind'][date_idx, r, c] if date_idx < dataset.conv_dyn_data['chelsa_sfxwind'].shape[0] else None
                                    else:
                                        sample_info['chelsa_sfxwind'] = None

                                    if 'lst' in dataset.conv_dyn_data:
                                        sample_info['lst'] = dataset.conv_dyn_data['lst'][date_idx, r, c] if date_idx < dataset.conv_dyn_data['lst'].shape[0] else None
                                    else:
                                        sample_info['lst'] = None

                                    if 'rh' in dataset.conv_dyn_data:
                                        sample_info['rh'] = dataset.conv_dyn_data['rh'][date_idx, r, c] if date_idx < dataset.conv_dyn_data['rh'].shape[0] else None
                                    else:
                                        sample_info['rh'] = None
                                else:
                                    sample_info['chelsa_sfxwind'] = None
                                    sample_info['lst'] = None
                                    sample_info['rh'] = None

                                if hasattr(dataset, 'clamday_data') and dataset.clamday_data is not None:
                                    sample_info['clamday'] = dataset.clamday_data[r, c]
                                else:
                                    sample_info['clamday'] = None

                                if hasattr(dataset, 'dem_data') and len(dataset.dem_data) > 0:
                                    sample_info['dem_mean'] = dataset.dem_data[0][r, c]
                                else:
                                    sample_info['dem_mean'] = None

                                if hasattr(dataset, 'dem_data') and len(dataset.dem_data) > 1:
                                    sample_info['dem_std'] = dataset.dem_data[1][r, c]
                                else:
                                    sample_info['dem_std'] = None

                                # LS特征
                                if hasattr(dataset, 'ls_data'):
                                    if isinstance(dataset.ls_data, dict):
                                        # 根据年份获取LS数据
                                        year = date.year if hasattr(date, 'year') else 2016
                                        ls_arr = dataset.ls_data.get(year, dataset.ls_data.get(list(dataset.ls_data.keys())[0]))
                                    else:
                                        ls_arr = dataset.ls_data
                                    for band_idx in range(min(6, ls_arr.shape[0])):
                                        sample_info[f'LS_band{band_idx+1}'] = ls_arr[band_idx, r, c]
                                else:
                                    for band_idx in range(6):
                                        sample_info[f'LS_band{band_idx+1}'] = None

                                # 哨兵1原始值
                                if hasattr(dataset, 's1_data') and dataset.s1_data:
                                    if hasattr(dataset, 'all_s1_dates') and dataset.all_s1_dates:
                                        closest_date = min(dataset.all_s1_dates, key=lambda d: abs((d - date).days))
                                        day_gap = abs((closest_date - date).days)
                                        sample_info['S1_最近日期'] = closest_date.strftime('%Y-%m-%d') if hasattr(closest_date, 'strftime') else str(closest_date)
                                        sample_info['S1_时间差_天'] = day_gap

                                        if closest_date in dataset.s1_data:
                                            if 'VV' in dataset.s1_data[closest_date]:
                                                sample_info['S1_VV_raw'] = dataset.s1_data[closest_date]['VV'][r, c]
                                            else:
                                                sample_info['S1_VV_raw'] = None
                                            if 'VH' in dataset.s1_data[closest_date]:
                                                sample_info['S1_VH_raw'] = dataset.s1_data[closest_date]['VH'][r, c]
                                            else:
                                                sample_info['S1_VH_raw'] = None
                                        else:
                                            sample_info['S1_VV_raw'] = None
                                            sample_info['S1_VH_raw'] = None
                                    else:
                                        sample_info['S1_VV_raw'] = None
                                        sample_info['S1_VH_raw'] = None
                                        sample_info['S1_最近日期'] = None
                                        sample_info['S1_时间差_天'] = None
                                else:
                                    sample_info['S1_VV_raw'] = None
                                    sample_info['S1_VH_raw'] = None
                                    sample_info['S1_最近日期'] = None
                                    sample_info['S1_时间差_天'] = None

                                # SMAP原始值
                                if hasattr(dataset, 'smap_data') and dataset.smap_data:
                                    if hasattr(dataset, 'all_smap_dates') and dataset.all_smap_dates:
                                        closest_date = min(dataset.all_smap_dates, key=lambda d: abs((d - date).days))
                                        day_gap = abs((closest_date - date).days)
                                        sample_info['SMAP_最近日期'] = closest_date.strftime('%Y-%m-%d') if hasattr(closest_date, 'strftime') else str(closest_date)
                                        sample_info['SMAP_时间差_天'] = day_gap

                                        if closest_date in dataset.smap_data:
                                            if 'TBV' in dataset.smap_data[closest_date]:
                                                sample_info['SMAP_TBV_raw'] = dataset.smap_data[closest_date]['TBV'][r, c]
                                            else:
                                                sample_info['SMAP_TBV_raw'] = None
                                            if 'TBH' in dataset.smap_data[closest_date]:
                                                sample_info['SMAP_TBH_raw'] = dataset.smap_data[closest_date]['TBH'][r, c]
                                            else:
                                                sample_info['SMAP_TBH_raw'] = None
                                        else:
                                            sample_info['SMAP_TBV_raw'] = None
                                            sample_info['SMAP_TBH_raw'] = None
                                    else:
                                        sample_info['SMAP_TBV_raw'] = None
                                        sample_info['SMAP_TBH_raw'] = None
                                        sample_info['SMAP_最近日期'] = None
                                        sample_info['SMAP_时间差_天'] = None
                                else:
                                    sample_info['SMAP_TBV_raw'] = None
                                    sample_info['SMAP_TBH_raw'] = None
                                    sample_info['SMAP_最近日期'] = None
                                    sample_info['SMAP_时间差_天'] = None

                                # 微波特征插值后
                                if hasattr(dataset, '_get_microwave_value_with_interpolation'):
                                    s1_vv, s1_vh, smap_tbv, smap_tbh = dataset._get_microwave_value_with_interpolation(date, r, c)
                                    sample_info['S1_VV_interp'] = s1_vv
                                    sample_info['S1_VH_interp'] = s1_vh
                                    sample_info['SMAP_TBV_interp'] = smap_tbv
                                    sample_info['SMAP_TBH_interp'] = smap_tbh
                                else:
                                    sample_info['S1_VV_interp'] = None
                                    sample_info['S1_VH_interp'] = None
                                    sample_info['SMAP_TBV_interp'] = None
                                    sample_info['SMAP_TBH_interp'] = None

                                # 经纬度
                                if hasattr(dataset, '_pixel_to_lonlat'):
                                    lon, lat = dataset._pixel_to_lonlat(r, c)
                                    sample_info['经度'] = lon
                                    sample_info['纬度'] = lat
                                else:
                                    sample_info['经度'] = None
                                    sample_info['纬度'] = None

                                samples_data.append(sample_info)

                                # ============ 预训练模型预测 ============
                                if pretrained_model is not None:
                                    try:
                                        conv_t = torch.from_numpy(conv_feats[i].numpy()).unsqueeze(0).to(self.device)
                                        point_t = torch.from_numpy(point_feats[i].numpy()).unsqueeze(0).to(self.device)
                                        pretrained_output = pretrained_model(conv_t, point_t)
                                        pretrained_pred_norm = pretrained_output.cpu().item()
                                        samples_data[-1]['预训练模型预测_norm'] = pretrained_pred_norm
                                        samples_data[-1]['预训练模型预测_raw'] = pretrained_pred_norm * (swe_max - swe_min) + swe_min
                                        pretrain_success += 1
                                    except Exception as e:
                                        pretrain_fail += 1

                                processed += 1
                                if processed % 100 == 0:
                                    print(f"    已处理 {processed}/{total_samples} 个样本")

            print(f"\n  完成! 共处理 {processed} 个样本")
            print(f"  FusedSWE 获取成功: {fused_success}, 失败: {fused_fail}")
            if pretrained_model is not None:
                print(f"  预训练预测成功: {pretrain_success}, 失败: {pretrain_fail}")

            # 转换为DataFrame
            df = pd.DataFrame(samples_data)

            print(f"\n2. 生成特征报告...")
            print(f"  DataFrame形状: {df.shape}")
            print(f"  列名: {list(df.columns)}")

            if 'FusedSWE_raw' in df.columns:
                valid_fused = df['FusedSWE_raw'].notna().sum()
                print(f"  FusedSWE_raw 有效值: {valid_fused}/{len(df)} ({valid_fused/len(df)*100:.1f}%)")

            if '预训练模型预测_raw' in df.columns:
                valid_pretrain = df['预训练模型预测_raw'].notna().sum()
                print(f"  预训练模型预测_raw 有效值: {valid_pretrain}/{len(df)} ({valid_pretrain/len(df)*100:.1f}%)")

            # 保存CSV
            csv_path = self.save_dir / "test_set_features_complete_with_pretrained.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 完整特征CSV已保存: {csv_path}")

            # ============ 生成对比统计 ============
            print("\n3. 生成模型对比统计...")

            required_cols = ['站点SWE_raw', '微调模型预测_raw', '预训练模型预测_raw']
            if all(col in df.columns for col in required_cols):
                valid_mask = (df['站点SWE_raw'].notna() & 
                             df['微调模型预测_raw'].notna() & 
                             df['预训练模型预测_raw'].notna())

                if valid_mask.sum() > 0:
                    station_swe = df.loc[valid_mask, '站点SWE_raw'].values
                    finetune_pred = df.loc[valid_mask, '微调模型预测_raw'].values
                    pretrain_pred = df.loc[valid_mask, '预训练模型预测_raw'].values
                    fused_swe = df.loc[valid_mask, 'FusedSWE_raw'].values if 'FusedSWE_raw' in df.columns else None

                    print(f"\n【模型对比 - 基于站点实测值】")
                    print(f"{'指标':<15} {'微调模型':<15} {'预训练模型':<15} {'改进'}")
                    print("-" * 55)

                    ft_rmse = np.sqrt(np.mean((finetune_pred - station_swe) ** 2))
                    pt_rmse = np.sqrt(np.mean((pretrain_pred - station_swe) ** 2))
                    improvement = (pt_rmse - ft_rmse) / pt_rmse * 100 if pt_rmse > 0 else 0
                    print(f"{'RMSE (mm)':<15} {ft_rmse:<15.2f} {pt_rmse:<15.2f} {improvement:+.1f}%")

                    ft_mae = np.mean(np.abs(finetune_pred - station_swe))
                    pt_mae = np.mean(np.abs(pretrain_pred - station_swe))
                    improvement = (pt_mae - ft_mae) / pt_mae * 100 if pt_mae > 0 else 0
                    print(f"{'MAE (mm)':<15} {ft_mae:<15.2f} {pt_mae:<15.2f} {improvement:+.1f}%")

                    ss_res_ft = np.sum((finetune_pred - station_swe) ** 2)
                    ss_res_pt = np.sum((pretrain_pred - station_swe) ** 2)
                    ss_tot = np.sum((station_swe - np.mean(station_swe)) ** 2)

                    ft_r2 = 1 - (ss_res_ft / ss_tot) if ss_tot > 0 else 0
                    pt_r2 = 1 - (ss_res_pt / ss_tot) if ss_tot > 0 else 0
                    improvement = (ft_r2 - pt_r2) / abs(pt_r2) * 100 if pt_r2 != 0 else 0
                    print(f"{'R²':<15} {ft_r2:<15.4f} {pt_r2:<15.4f} {improvement:+.1f}%")

                    ft_bias = np.mean(finetune_pred - station_swe)
                    pt_bias = np.mean(pretrain_pred - station_swe)
                    print(f"{'Bias (mm)':<15} {ft_bias:<15.2f} {pt_bias:<15.2f}")

                    if fused_swe is not None:
                        valid_fused_mask = ~np.isnan(fused_swe)
                        if valid_fused_mask.sum() > 0:
                            fused_swe_valid = fused_swe[valid_fused_mask]
                            station_swe_valid = station_swe[valid_fused_mask]

                            fused_rmse = np.sqrt(np.mean((fused_swe_valid - station_swe_valid) ** 2))
                            fused_mae = np.mean(np.abs(fused_swe_valid - station_swe_valid))
                            ss_res_fused = np.sum((fused_swe_valid - station_swe_valid) ** 2)
                            ss_tot_fused = np.sum((station_swe_valid - np.mean(station_swe_valid)) ** 2)
                            fused_r2 = 1 - (ss_res_fused / ss_tot_fused) if ss_tot_fused > 0 else 0

                            print(f"\n【FusedSWE vs 站点实测】")
                            print(f"  RMSE: {fused_rmse:.2f} mm")
                            print(f"  MAE: {fused_mae:.2f} mm")
                            print(f"  R²: {fused_r2:.4f}")

            # ============ 生成对比散点图 ============
            print("\n4. 生成对比散点图...")

            # 直接从 df 读取数据
            if '站点SWE_raw' in df.columns and '微调模型预测_raw' in df.columns and 'FusedSWE_raw' in df.columns:
                # 强制转换为数值类型
                station_swe = pd.to_numeric(df['站点SWE_raw'], errors='coerce').values
                finetune_pred = pd.to_numeric(df['微调模型预测_raw'], errors='coerce').values
                fused_swe = pd.to_numeric(df['FusedSWE_raw'], errors='coerce').values

                # ============ 新增：提取站点 ID ============
                if '站点ID' in df.columns:
                    station_ids = df['站点ID'].astype(str).values
                    print(f"  ✓ 成功提取 {len(station_ids)} 个站点ID")
                else:
                    station_ids = np.array(['N/A'] * len(df))
                    print(f"  ⚠ DataFrame中没有'站点ID'列")
                # ========================================

                # 调试信息
                print(f"\n  【数据检查】")
                print(f"    站点SWE前5: {station_swe[:5]}")
                print(f"    微调预测前5: {finetune_pred[:5]}")
                print(f"    FusedSWE前5: {fused_swe[:5]}")
                if station_ids is not None:
                    print(f"    站点ID前5: {station_ids[:5]}")

                # 统计NaN数量
                station_nan = np.isnan(station_swe).sum()
                finetune_nan = np.isnan(finetune_pred).sum()
                fused_nan = np.isnan(fused_swe).sum()

                print(f"\n  【NaN统计】")
                print(f"    站点SWE NaN: {station_nan}/{len(df)}")
                print(f"    微调预测 NaN: {finetune_nan}/{len(df)}")
                print(f"    FusedSWE NaN: {fused_nan}/{len(df)}")

                # 找到有效样本
                valid_mask = ~np.isnan(station_swe) & ~np.isnan(finetune_pred) & ~np.isnan(fused_swe)
                n_valid = np.sum(valid_mask)
                print(f"\n    有效样本数: {n_valid}/{len(df)}")

                if n_valid > 0:
                    station_swe_valid = station_swe[valid_mask]
                    finetune_pred_valid = finetune_pred[valid_mask]
                    fused_swe_valid = fused_swe[valid_mask]

                    # ============ 新增：同步过滤站点 ID ============
                    if station_ids is not None:
                        station_ids_valid = station_ids[valid_mask]
                    else:
                        station_ids_valid = None
                    # ============================================

                    # 计算指标
                    ft_rmse = np.sqrt(np.mean((finetune_pred_valid - station_swe_valid) ** 2))
                    fused_rmse = np.sqrt(np.mean((fused_swe_valid - station_swe_valid) ** 2))

                    ss_res_ft = np.sum((finetune_pred_valid - station_swe_valid) ** 2)
                    ss_res_fused = np.sum((fused_swe_valid - station_swe_valid) ** 2)
                    ss_tot = np.sum((station_swe_valid - np.mean(station_swe_valid)) ** 2)

                    ft_r2 = 1 - (ss_res_ft / ss_tot) if ss_tot > 0 else 0
                    fused_r2 = 1 - (ss_res_fused / ss_tot) if ss_tot > 0 else 0

                    # 绘制散点图
                    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                    min_val = min(station_swe_valid.min(), finetune_pred_valid.min(), fused_swe_valid.min())
                    max_val = max(station_swe_valid.max(), finetune_pred_valid.max(), fused_swe_valid.max())
                    margin = (max_val - min_val) * 0.05
                    plot_min = max(0, min_val - margin)
                    plot_max = max_val + margin

                    # 图1: 微调模型
                    ax1 = axes[0]
                    ax1.scatter(station_swe_valid, finetune_pred_valid, alpha=0.6, s=30, c='blue', edgecolors='white')
                    ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, alpha=0.8, label='1:1 Line')
                    ax1.set_xlabel('站点实测 SWE (mm)', fontsize=12)
                    ax1.set_ylabel('微调模型预测 (mm)', fontsize=12)
                    ax1.set_title(f'微调模型\nR²={ft_r2:.3f}, RMSE={ft_rmse:.2f}mm\nN={n_valid}', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    ax1.set_xlim(plot_min, plot_max)
                    ax1.set_ylim(plot_min, plot_max)
                    ax1.legend()

                    # 图2: FusedSWE（带站点ID标注）
                    ax2 = axes[1]
                    ax2.scatter(station_swe_valid, fused_swe_valid, alpha=0.6, s=30, c='orange', edgecolors='white')
                    ax2.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, alpha=0.8, label='1:1 Line')

                    # ============ 新增：标注贴在横轴上的异常点 ============
                    if station_ids_valid is not None:
                        anomaly_count = 0
                        annotated_positions = {}

                        for i in range(len(fused_swe_valid)):
                            if fused_swe_valid[i] <= 1e-3 and station_swe_valid[i] > 1e-3:
                                x_pos = station_swe_valid[i]
                                y_pos = fused_swe_valid[i]
                                key = f"{x_pos:.1f}"
                                if key not in annotated_positions:
                                    annotated_positions[key] = 0
                                y_offset = 8 + annotated_positions[key] * 12
                                ax2.annotate(
                                    station_ids_valid[i], 
                                    (x_pos, y_pos),
                                    xytext=(0, y_offset),
                                    textcoords='offset points',
                                    ha='center', va='bottom',
                                    fontsize=7, fontweight='bold', color='red',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red', linewidth=0.5)
                                )
                                annotated_positions[key] += 1
                                anomaly_count += 1

                        if anomaly_count > 0:
                            print(f"\n    标注了 {anomaly_count} 个 FusedSWE=0 但实测值>0 的异常点")
                    # ====================================================

                    ax2.set_xlabel('站点实测 SWE (mm)', fontsize=12)
                    ax2.set_ylabel('FusedSWE (mm)', fontsize=12)
                    ax2.set_title(f'FusedSWE\nR²={fused_r2:.3f}, RMSE={fused_rmse:.2f}mm\nN={n_valid}', fontsize=12)
                    ax2.grid(True, alpha=0.3)
                    ax2.set_xlim(plot_min, plot_max)
                    ax2.set_ylim(plot_min, plot_max)
                    ax2.legend()

                    plt.suptitle('对比：微调模型 vs FusedSWE', fontsize=14, fontweight='bold')
                    plt.tight_layout()

                    plot_path = self.save_dir / "model_comparison_scatter.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"\n  ✓ 对比散点图已保存: {plot_path}")

                    print(f"\n  【统一样本数据范围】")
                    print(f"    实测值范围: [{station_swe_valid.min():.2f}, {station_swe_valid.max():.2f}] mm")
                    print(f"    FusedSWE范围: [{fused_swe_valid.min():.2f}, {fused_swe_valid.max():.2f}] mm")
                    print(f"    微调预测范围: [{finetune_pred_valid.min():.2f}, {finetune_pred_valid.max():.2f}] mm")

                else:
                    print(f"\n  ⚠ 没有有效的样本数据")
                    print("\n  【问题诊断】")
                    print(f"    站点SWE有效: {np.sum(~np.isnan(station_swe))}")
                    print(f"    微调预测有效: {np.sum(~np.isnan(finetune_pred))}")
                    print(f"    FusedSWE有效: {np.sum(~np.isnan(fused_swe))}")

                    print(f"\n  前5个样本的原始值:")
                    for i in range(min(5, len(df))):
                        if '站点ID' in df.columns:
                            print(f"    {i}: 站点={df.iloc[i]['站点SWE_raw']}, "
                                  f"微调={df.iloc[i]['微调模型预测_raw']}, "
                                  f"Fused={df.iloc[i]['FusedSWE_raw']}, "
                                  f"ID={df.iloc[i]['站点ID']}")
                        else:
                            print(f"    {i}: 站点={df.iloc[i]['站点SWE_raw']}, "
                                  f"微调={df.iloc[i]['微调模型预测_raw']}, "
                                  f"Fused={df.iloc[i]['FusedSWE_raw']}")
            else:
                print(f"  ⚠ 缺少必要列，无法生成对比散点图")

            print(f"\n✅ 测试集特征分析完成!")
            print(f"  完整报告已保存至: {self.save_dir}")

            return df

        except Exception as e:
            print(f"❌ 分析失败: {e}")
            traceback.print_exc()
            return None
    
    def _get_point_feature_names(self):
        """获取当前 Clean-18D 点特征名称。"""
        return [
            'LS1', 'LS2', 'LS3', 'LS4', 'LS5', 'LS6',
            'S1_VV', 'S1_VH', 'S1_VV_cov', 'S1_VH_cov', 'S1_angle',
            'SMAP_TBV', 'SMAP_TBH', 'SMAP_mask_V', 'SMAP_mask_H',
            'lon_norm', 'lat_norm', 'doy_norm',
        ]

    def _save_fine_tune_evaluation_results(
        self,
        eval_results,
        predictions_denorm=None,
        targets_denorm=None,
    ):
        """保存微调评估结果；任何写入失败都必须向上抛出。"""
        print("  保存微调评估结果...")

        if eval_results is None:
            raise RuntimeError("评估结果为空，禁止继续保存")

        self.save_dir.mkdir(parents=True, exist_ok=True)

        eval_path = self.save_dir / "fine_tune_evaluation_results.json"
        txt_path = self.save_dir / "fine_tune_summary.txt"

        try:
            with eval_path.open("w", encoding="utf-8") as f:
                json.dump(
                    eval_results,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )

            if not eval_path.is_file() or eval_path.stat().st_size == 0:
                raise RuntimeError(f"评估结果文件写入后为空或不存在: {eval_path}")

            print(f"  ✓ 评估结果保存到: {eval_path}")

            def _fmt(value, fmt, default="N/A"):
                if value is None:
                    return default
                try:
                    return format(float(value), fmt)
                except (TypeError, ValueError):
                    return str(value)

            with txt_path.open("w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("SWE微调模型评估结果摘要（反归一化后）\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"评估时间: {eval_results.get('timestamp', 'N/A')}\n")
                f.write(f"样本数量: {eval_results.get('num_samples', 'N/A')}\n\n")
                f.write("主要指标（反归一化后，单位mm）:\n")
                f.write("-" * 40 + "\n")
                f.write(f"MSE:  {_fmt(eval_results.get('mse'), '.2f')}\n")
                f.write(f"RMSE: {_fmt(eval_results.get('rmse'), '.2f')}\n")
                f.write(f"MAE:  {_fmt(eval_results.get('mae'), '.2f')}\n")
                f.write(f"Bias: {_fmt(eval_results.get('bias'), '.2f')}\n")
                f.write(f"r:    {_fmt(eval_results.get('r'), '.4f')}\n\n")

                for key in ("alpha", "beta", "slope", "intercept"):
                    if eval_results.get(key) is not None:
                        f.write(f"{key}: {_fmt(eval_results.get(key), '.4f')}\n")

                hs = eval_results.get("high_swe")
                if isinstance(hs, dict):
                    f.write("\n高 SWE 子集:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"阈值: obs >= {hs.get('threshold', 80)} mm\n")
                    f.write(f"样本数: {hs.get('n', 0)}\n")
                    if int(hs.get("n", 0) or 0) > 0:
                        f.write(f"MAE:  {_fmt(hs.get('mae'), '.2f')} mm\n")
                        f.write(f"RMSE: {_fmt(hs.get('rmse'), '.2f')} mm\n")
                        f.write(f"Bias: {_fmt(hs.get('bias'), '.2f')} mm\n")
                        f.write(f"obs mean:  {_fmt(hs.get('target_mean'), '.2f')} mm\n")
                        f.write(f"pred mean: {_fmt(hs.get('pred_mean'), '.2f')} mm\n")

                if predictions_denorm is not None and targets_denorm is not None:
                    f.write("\n数据范围:\n")
                    f.write("-" * 40 + "\n")
                    f.write(
                        f"预测值范围: [{predictions_denorm.min():.2f}, "
                        f"{predictions_denorm.max():.2f}] mm\n"
                    )
                    f.write(
                        f"真实值范围: [{targets_denorm.min():.2f}, "
                        f"{targets_denorm.max():.2f}] mm\n"
                    )
                    f.write(f"预测值均值: {predictions_denorm.mean():.2f} mm\n")
                    f.write(f"真实值均值: {targets_denorm.mean():.2f} mm\n")

            if not txt_path.is_file() or txt_path.stat().st_size == 0:
                raise RuntimeError(f"评估摘要文件写入后为空或不存在: {txt_path}")

            print(f"  ✓ 评估摘要保存到: {txt_path}")
            print("\n" + "=" * 60)
            print("微调评估结果摘要（反归一化后）:")
            print(f"  r:    {_fmt(eval_results.get('r'), '.4f')}")
            print(f"  RMSE: {_fmt(eval_results.get('rmse'), '.2f')} mm")
            print(f"  MAE:  {_fmt(eval_results.get('mae'), '.2f')} mm")
            print(f"  Bias: {_fmt(eval_results.get('bias'), '.2f')} mm")
            print("=" * 60)

            return eval_path

        except Exception as exc:
            print(f"  ✗ 保存评估结果失败: {exc}")
            raise


    def _run_diagnose_swe(self, y_true, y_pred):
        """
        快速诊断模型预测的系统性偏差。
        y_true / y_pred 必须是反归一化后的 mm。
        """
        metrics = self._compute_advanced_metrics(
            y_true=y_true,
            y_pred=y_pred,
            high_threshold=80.0
        )

        if metrics is None:
            print("  样本数不足，跳过诊断")
            return None

        results = {
            "original": {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "nse": metrics["nse"],
                "r": metrics["r"],
                "r_squared": metrics["r_squared"],
                "bias": metrics["bias"],
                "alpha": metrics["alpha"],
                "beta": metrics["beta"],
                "slope": metrics["slope"],
                "intercept": metrics["intercept"],
                "pred_min": metrics["pred_min"],
                "pred_max": metrics["pred_max"],
                "target_min": metrics["target_min"],
                "target_max": metrics["target_max"],
                "high_swe": metrics["high_swe"],
            },
            "calibrated": metrics["calibrated"]
        }

        orig = results["original"]
        cal = results["calibrated"]

        print("\n" + "-" * 60)
        print("📈 偏差诊断:")
        print(f"  NSE = {orig['nse']:.4f}  |  r = {orig['r']:.4f}  |  r² = {orig['r_squared']:.4f}")
        print(f"  α (std ratio) = {orig['alpha']:.4f}  (理想=1)")
        print(f"  β (bias/std)  = {orig['beta']:.4f}  (理想=0)")
        print(f"  回归线: pred = {orig['intercept']:.2f} + {orig['slope']:.2f} * obs")
        print(f"  预测范围: [{orig['pred_min']:.2f}, {orig['pred_max']:.2f}] mm")
        print(f"  真实范围: [{orig['target_min']:.2f}, {orig['target_max']:.2f}] mm")

        print(f"  后校准 NSE = {cal['nse']:.4f}")
        print(f"  后校准关系: obs = {cal['intercept']:.2f} + {cal['slope']:.2f} * pred")

        hs = orig["high_swe"]
        print(f"\n  高 SWE 子集 obs >= {hs['threshold']:.0f} mm:")
        print(f"    N = {hs['n']}")
        if hs["n"] > 0:
            print(f"    MAE  = {hs['mae']:.2f} mm")
            print(f"    RMSE = {hs['rmse']:.2f} mm")
            print(f"    Bias = {hs['bias']:.2f} mm")
            print(f"    obs mean  = {hs['target_mean']:.2f} mm")
            print(f"    pred mean = {hs['pred_mean']:.2f} mm")

        if orig["nse"] < orig["r_squared"] - 0.05:
            print("  ⚠️ 存在系统性偏差：NSE 明显低于 r²")
            if orig["alpha"] < 0.8:
                print("     → 预测幅度被压缩，高值上不去")
            elif orig["alpha"] > 1.2:
                print("     → 预测幅度被放大")
        else:
            print("  ✅ NSE 与 r² 差异不大，系统性幅度偏差不明显")

        print("-" * 60)

        return results
           
        
    # EVAL_SINGLE_FORWARD_PER_SPLIT_V2
    def _build_complete_evaluation_loader(
            self,
            loader,
            split_name="eval",
    ):
        """构建shuffle=False、drop_last=False的评估Loader。"""
        if loader is None:
            return None

        from torch.utils.data import DataLoader

        batch_size = getattr(loader, "batch_size", None)

        if batch_size is None:
            batch_size = int(
                self.config.get("batch_size", 32)
            )

        num_workers = int(
            getattr(loader, "num_workers", 0) or 0
        )

        kwargs = {
            "dataset": loader.dataset,
            "batch_size": batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": num_workers,
            "collate_fn": getattr(
                loader,
                "collate_fn",
                None,
            ),
            "pin_memory": bool(
                getattr(loader, "pin_memory", False)
            ),
            "worker_init_fn": getattr(
                loader,
                "worker_init_fn",
                None,
            ),
        }

        if num_workers > 0:
            prefetch_factor = getattr(
                loader,
                "prefetch_factor",
                None,
            )

            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = (
                    prefetch_factor
                )

            kwargs["persistent_workers"] = bool(
                getattr(
                    loader,
                    "persistent_workers",
                    False,
                )
            )

        eval_loader = DataLoader(**kwargs)

        print(
            f"  ✅ {split_name}评估Loader: "
            f"N={len(eval_loader.dataset)}, "
            f"batch_size={batch_size}, "
            "shuffle=False, drop_last=False"
        )

        return eval_loader


    def _set_loader_augmentation(self, loader, is_train=False):
        """
        评估时临时关闭 Dataset 增强。
        """
        if loader is None:
            return None

        ds = loader.dataset
        base_ds = ds.dataset if hasattr(ds, "dataset") else ds

        old_value = None
        if hasattr(base_ds, "current_augment"):
            old_value = base_ds.current_augment
            base_ds.current_augment = is_train

        if hasattr(base_ds, "set_augmentation_mode"):
            try:
                base_ds.set_augmentation_mode(is_train)
            except Exception:
                pass

        return old_value


    def _restore_loader_augmentation(self, loader, old_value):
        """
        恢复 Dataset 增强状态。
        """
        if loader is None or old_value is None:
            return

        ds = loader.dataset
        base_ds = ds.dataset if hasattr(ds, "dataset") else ds

        if hasattr(base_ds, "current_augment"):
            base_ds.current_augment = old_value


    def _evaluate_loader_diagnostics(
            self,
            loader,
            split_name="val",
            high_threshold=80.0,
            prediction_result=None,
    ):
        """
        对任意 loader 计算反归一化后的完整诊断。
        用于 train / val / test 对比。
        """

        if loader is None:
            print(f"  [{split_name}] loader 为空，跳过")
            return None

        print(f"\n  [{split_name.upper()}] 诊断评估...")

        if prediction_result is not None:
            print(
                f"    ♻️ 复用{split_name}单次预测，"
                "不再执行模型前向"
            )
            result = prediction_result
        else:
            old_aug = self._set_loader_augmentation(
                loader,
                is_train=False,
            )

            try:
                result = self._make_predictions(loader)
            finally:
                self._restore_loader_augmentation(
                    loader,
                    old_aug,
                )

        if result is None:
            print(f"  [{split_name}] 预测失败")
            return None

        pred_norm, target_norm, is_zero = result

        if pred_norm is None or target_norm is None:
            print(f"  [{split_name}] 预测结果为空")
            return None

        swe_min = getattr(self, "swe_min", 0.0)
        swe_max = getattr(self, "swe_max", 200.0)

        pred_mm = pred_norm * (swe_max - swe_min) + swe_min
        target_mm = target_norm * (swe_max - swe_min) + swe_min

        metrics = self._compute_advanced_metrics(
            y_true=target_mm,
            y_pred=pred_mm,
            high_threshold=high_threshold
        )

        if metrics is None:
            return None

        metrics["split"] = split_name
        metrics["norm_pred_range"] = [float(np.min(pred_norm)), float(np.max(pred_norm))]
        metrics["norm_target_range"] = [float(np.min(target_norm)), float(np.max(target_norm))]

        print(
            f"    N={metrics['n']} | "
            f"R={metrics['r']:.4f} | "
            f"NSE={metrics['nse']:.4f} | "
            f"RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f} | "
            f"Bias={metrics['bias']:.2f} | "
            f"alpha={metrics['alpha']:.3f} | "
            f"slope={metrics['slope']:.3f} | "
            f"pred=[{metrics['pred_min']:.1f},{metrics['pred_max']:.1f}]"
        )

        hs = metrics["high_swe"]
        if hs["n"] > 0:
            print(
                f"    obs>={high_threshold:.0f}: "
                f"N={hs['n']} | "
                f"MAE={hs['mae']:.2f} | "
                f"Bias={hs['bias']:.2f} | "
                f"obs_mean={hs['target_mean']:.2f} | "
                f"pred_mean={hs['pred_mean']:.2f}"
            )
        else:
            print(f"    obs>={high_threshold:.0f}: N=0")

        return metrics


    def _print_split_diagnostics_table(self, split_diagnostics):
        """
        打印 train/val/test 横向对比表。
        """
        print("\n" + "=" * 110)
        print("📊 Train / Val / Test 完整诊断")
        print("=" * 110)
        print(
            f"{'Split':8s} {'N':>6s} {'R':>8s} {'NSE':>8s} {'RMSE':>9s} "
            f"{'MAE':>9s} {'Bias':>9s} {'alpha':>8s} {'slope':>8s} "
            f"{'PredRange':>20s} {'HighBias':>10s}"
        )
        print("-" * 110)

        for split in ["train", "val", "test"]:
            m = split_diagnostics.get(split)
            if not m:
                continue

            hs = m.get("high_swe", {})
            high_bias = hs.get("bias", None)
            high_bias_str = f"{high_bias:.2f}" if high_bias is not None else "N/A"

            pred_range = f"[{m['pred_min']:.1f},{m['pred_max']:.1f}]"

            print(
                f"{split:8s} "
                f"{m['n']:6d} "
                f"{m['r']:8.4f} "
                f"{m['nse']:8.4f} "
                f"{m['rmse']:9.2f} "
                f"{m['mae']:9.2f} "
                f"{m['bias']:9.2f} "
                f"{m['alpha']:8.3f} "
                f"{m['slope']:8.3f} "
                f"{pred_range:>20s} "
                f"{high_bias_str:>10s}"
            )

        print("=" * 110)


    def _save_diagnosis_json(self, diagnosis_payload, filename="diagnosis_results.json"):
        """
        保存完整诊断 JSON。
        """

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            if isinstance(obj, tuple):
                return [sanitize(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                v = float(obj)
                if np.isnan(v) or np.isinf(v):
                    return None
                return v
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            return obj

        path = self.save_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitize(diagnosis_payload), f, indent=2, ensure_ascii=False, default=str)

        print(f"  ✅ 诊断结果已保存: {path}")
        return path
        
    
    def evaluate_pretrained_at_pixels(self, test_loader=None):
        """
        评估预训练模型在站点位置所在像元的表现
        对比：预训练预测值 vs 该像元的网格SWE值
        """
        print("\n" + "="*70)
        print("📊 评估预训练模型在站点像元的表现")
        print("="*70)
        
        if test_loader is None:
            test_loader = self.test_loader
        
        # 需要获取每个样本对应的像元网格SWE值
        # 假设 dataset 中保存了 grid_swe 或可以从标签数据获取
        
        all_preds = []
        all_grid_swe = []  # 像元网格SWE值，不是站点SWE！
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                if len(batch_data) == 4:
                    conv_feats, point_feats, station_swe, _ = batch_data
                else:
                    conv_feats, point_feats, station_swe = batch_data
                
                # 这里需要获取每个样本对应的像元网格SWE
                # 假设可以从dataset中通过某种方式获取
                # 比如 dataset.get_grid_swe(batch_idx)
                
                conv_feats = conv_feats.to(self.device)
                point_feats = point_feats.to(self.device)
                
                # 预训练模型预测（对像元的预测）
                pred = self.model(conv_feats, point_feats)
                
                all_preds.extend(pred.cpu().numpy())
                # all_grid_swe.extend(grid_swe.numpy())  # 需要实际获取
        
        # 反归一化
        swe_min = getattr(self, 'swe_min', 0.0)
        swe_max = getattr(self, 'swe_max', 200.0)
        pred_denorm = np.array(all_preds) * (swe_max - swe_min) + swe_min
        grid_denorm = np.array(all_grid_swe) * (swe_max - swe_min) + swe_min
        
        # 计算指标
        rmse = np.sqrt(np.mean((pred_denorm - grid_denorm) ** 2))
        mae = np.mean(np.abs(pred_denorm - grid_denorm))
        
        print(f"\n【预训练模型在站点像元的表现】")
        print(f"  对比对象: 像元网格SWE值")
        print(f"  RMSE: {rmse:.2f} mm")
        print(f"  MAE: {mae:.2f} mm")
        
        return {
            'pred': pred_denorm,
            'grid_swe': grid_denorm,
            'rmse': rmse,
            'mae': mae
        }
        
    def plot_lora_training_curves(self):
        """绘制LoRA训练监控曲线"""
        try:
            if not hasattr(self, 'lora_grad_history') or not self.lora_grad_history:
                print("没有LoRA训练历史数据可绘制")
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. LoRA梯度变化
            ax1 = axes[0, 0]
            epochs = [h['epoch'] for h in self.lora_grad_history]
            avg_grads = [h['avg_grad_norm'] for h in self.lora_grad_history]
            max_grads = [h['max_grad_norm'] for h in self.lora_grad_history]
            min_grads = [h['min_grad_norm'] for h in self.lora_grad_history]
            
            ax1.plot(epochs, avg_grads, 'b-', label='平均梯度', linewidth=2)
            ax1.fill_between(epochs, min_grads, max_grads, alpha=0.2, color='blue', label='梯度范围')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('梯度范数')
            ax1.set_title('LoRA梯度变化', fontsize=14, fontweight='bold')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. LoRA权重变化
            ax2 = axes[0, 1]
            if hasattr(self, 'lora_weight_change_history') and self.lora_weight_change_history:
                weight_epochs = [h['epoch'] for h in self.lora_weight_change_history]
                weight_changes = [h['total_change'] for h in self.lora_weight_change_history]
                
                ax2.plot(weight_epochs, weight_changes, 'r-', label='权重总变化', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('权重变化')
                ax2.set_title('LoRA权重变化', fontsize=14, fontweight='bold')
                ax2.set_yscale('log')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. LoRA参数数量
            ax3 = axes[1, 0]
            num_samples = [h['num_samples'] for h in self.lora_grad_history]
            
            ax3.plot(epochs, num_samples, 'g-', label='梯度样本数', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('样本数')
            ax3.set_title('LoRA梯度样本数', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. LoRA梯度与损失关系
            ax4 = axes[1, 1]
            if len(self.train_history) >= len(self.lora_grad_history):
                start_idx = len(self.train_history) - len(self.lora_grad_history)
                losses = self.train_history[start_idx:]
                
                scatter = ax4.scatter(losses, avg_grads, c=epochs, cmap='viridis', s=50)
                ax4.set_xlabel('训练损失')
                ax4.set_ylabel('LoRA平均梯度')
                ax4.set_title('LoRA梯度与损失关系', fontsize=14, fontweight='bold')
                ax4.set_xscale('log')
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3)
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Epoch')
            
            plt.suptitle('LoRA微调监控', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存图像
            plot_path = self.save_dir / "lora_training_monitoring.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"LoRA训练监控曲线已保存到: {plot_path}")
            
        except Exception as e:
            print(f"绘制LoRA训练曲线失败: {e}")

    # SCATTER_METRIC_BOX_STYLE_V1
    # 统一放大散点图指标框，并降低框背景透明度。
    def _plot_constraint_scatter(self, predictions, targets, is_zero, eval_results):
        """绘制包含约束效果的散点图"""
        try:
            # 创建图形
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # ============ 图1：所有样本的散点图 ============
            ax1 = axes[0]
            
            # 分离target=0和target>0的样本
            zero_mask = (is_zero == 0)
            pos_mask = (is_zero == 1)
            
            # target>0的样本（蓝色）
            ax1.scatter(targets[pos_mask], predictions[pos_mask], 
                       alpha=0.5, s=20, c='blue', label=f'产品值>0 (n={np.sum(pos_mask)})',
                       edgecolors='none')
            
            # target=0的样本（红色）
            ax1.scatter(targets[zero_mask], predictions[zero_mask],
                       alpha=0.5, s=20, c='red', label=f'产品值=0 (n={np.sum(zero_mask)})',
                       edgecolors='none', marker='x')
            
            # 1:1线
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            margin = (max_val - min_val) * 0.05
            
            ax1.plot([min_val, max_val], [min_val, max_val],
                    'k--', linewidth=2, alpha=0.8, label='1:1线')
            
            # 设置图形属性
            ax1.set_xlabel('产品值 (归一化)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('预测值 (归一化)', fontsize=14, fontweight='bold')
            ax1.set_title('预训练模型预测结果 - 约束效果可视化', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=12, loc='upper left')
            ax1.set_xlim(min_val - margin, max_val + margin)
            ax1.set_ylim(min_val - margin, max_val + margin)
            
            # 添加指标框
            r_value = eval_results.get('r', 0)
            rmse = eval_results.get('rmse', 0)
            mae = eval_results.get('mae', 0)
            
            metrics_text = (f'R = {r_value:.4f}\n'
                           f'RMSE = {rmse:.4f}\n'
                           f'MAE = {mae:.4f}\n'
                           f'N = {len(targets):,}')
            
            bbox_props = dict(
                boxstyle="round,pad=0.65",
                # 背景透明，文字本身仍完全不透明
                facecolor=(1.0, 1.0, 1.0, 0.45),
                edgecolor=(0.0, 0.0, 0.0, 0.75),
                linewidth=1.5,
            )
            
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                    fontsize=16, fontweight='bold', verticalalignment='top',
                    horizontalalignment='left', bbox=bbox_props)
            
            # ============ 图2：约束效果详细分析 ============
            ax2 = axes[1]
            
            if np.sum(zero_mask) > 0:
                # 计算target=0样本的预测值分布
                zero_predictions = predictions[zero_mask]
                
                # 创建直方图
                ax2.hist(zero_predictions, bins=50, color='red', alpha=0.7, 
                        edgecolor='black', linewidth=0.5)
                ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
                
                # 计算统计量
                zero_mean = np.mean(zero_predictions)
                zero_std = np.std(zero_predictions)
                zero_abs_mean = np.mean(np.abs(zero_predictions))
                
                # 添加统计信息
                stats_text = (f'产品值=0样本统计:\n'
                             f'数量: {len(zero_predictions)}\n'
                             f'预测均值: {zero_mean:.6f}\n'
                             f'预测标准差: {zero_std:.6f}\n'
                             f'绝对均值: {zero_abs_mean:.6f}\n'
                             f'约束效果评估:')
                
                if zero_abs_mean < 0.001:
                    constraint_eval = "✓ 优秀"
                    color = 'green'
                elif zero_abs_mean < 0.01:
                    constraint_eval = "✓ 良好"
                    color = 'orange'
                elif zero_abs_mean < 0.05:
                    constraint_eval = "⚠ 一般"
                    color = 'yellow'
                else:
                    constraint_eval = "✗ 较差"
                    color = 'red'
                    
                stats_text += f'\n{constraint_eval}'
                
                ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                        fontsize=16, fontweight='bold', verticalalignment='top',
                        horizontalalignment='right', bbox=bbox_props,
                        color=color if '✗' in constraint_eval else 'black')
                
                ax2.set_xlabel('预测值 (归一化)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('频数', fontsize=14, fontweight='bold')
                ax2.set_title('产品值=0样本的预测值分布', fontsize=16, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, '没有target=0的样本', 
                        transform=ax2.transAxes, fontsize=14, fontweight='bold',
                        horizontalalignment='center', verticalalignment='center')
                ax2.set_title('产品值=0样本的预测值分布', fontsize=16, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            plot_path = self.save_dir / "constraint_effect_scatter.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 约束效果散点图已保存: {plot_path}")
            
            # 打印约束效果总结
            print(f"\n约束效果总结:")
            print(f"  target>0样本数: {np.sum(pos_mask)}")
            print(f"  target=0样本数: {np.sum(zero_mask)}")
            
            if np.sum(zero_mask) > 0:
                print(f"  target=0样本预测统计:")
                print(f"    预测均值: {zero_mean:.6f}")
                print(f"    预测标准差: {zero_std:.6f}")
                print(f"    绝对均值: {zero_abs_mean:.6f}")
                print(f"    约束效果: {constraint_eval}")
                
                # 检查有多少target=0的样本预测值不为0
                non_zero_pred = np.sum(np.abs(zero_predictions) > 0.001)
                if non_zero_pred > 0:
                    print(f"    ⚠ 警告: {non_zero_pred}个target=0样本预测值不为0")
                    
            return plot_path
            
        except Exception as e:
            print(f"绘制约束效果散点图失败: {e}")
            traceback.print_exc()
            return None

    def _generate_fine_tune_plots(self, predictions, targets, eval_results):
        """生成微调可视化图表 - 使用反归一化数据"""
        print("  生成微调可视化图表...")

        try:
            # 1. 生成密度散点图 - 确保传入的是反归一化数据
            print("  1. 生成密度散点图...")

            # 🔥 修复：定义 use_raw 变量
            use_raw = False

            # 检查数据是否是反归一化的
            if predictions.max() <= 1.0 and targets.max() <= 1.0:
                print("  ⚠ 警告：数据可能还是归一化的！尝试反归一化...")
                swe_min = getattr(self, 'swe_min', 0.0)
                swe_max = getattr(self, 'swe_max', 200.0)
                predictions = predictions * (swe_max - swe_min) + swe_min
                targets = targets * (swe_max - swe_min) + swe_min
                use_raw = True  # 🔥 标记已反归一化

            self.plot_density_scatter_hardcode(predictions, targets, is_fine_tune=True, use_raw=use_raw)

            # 2. 如果有微调历史，生成微调曲线
            if hasattr(self, "fine_tune_history") and len(self.fine_tune_history) > 0:
                print("  2. 生成微调训练曲线...")
                self.plot_training_curves(fine_tune_mode=True)

            print("  ✓ 图表生成完成")

        except Exception as e:
            print(f"  ✗ 图表生成失败: {e}")
            traceback.print_exc()
        
    def plot_all_finetune_samples(self, model_path=None):
        """
        绘制所有微调样本（训练集+验证集+测试集）的散点图
        用不同颜色区分三个数据集
        """
        print("\n" + "="*70)
        print("📊 绘制所有微调样本散点图")
        print("="*70)

        try:
            # 设置中文字体
            self.setup_chinese_fonts()

            # 检查是否有三个数据集
            if not hasattr(self, 'train_loader') or not hasattr(self, 'val_loader') or not hasattr(self, 'test_loader'):
                print("❌ 缺少数据集加载器")
                return

            # 加载模型（如果指定）
            if model_path:
                self._load_model_for_evaluation(model_path)

            # 收集所有数据集的预测结果
            all_data = {}

            for dataset_name, loader in [('train', self.train_loader), 
                                          ('val', self.val_loader), 
                                          ('test', self.test_loader)]:
                print(f"\n1. 处理 {dataset_name}集...")

                predictions, targets, is_zero = self._make_predictions(loader)

                if predictions is not None and len(predictions) > 0:
                    # 反归一化
                    swe_min = getattr(self, 'swe_min', 0.0)
                    swe_max = getattr(self, 'swe_max', 200.0)
                    pred_denorm = predictions * (swe_max - swe_min) + swe_min
                    target_denorm = targets * (swe_max - swe_min) + swe_min

                    all_data[dataset_name] = {
                        'pred': pred_denorm,
                        'target': target_denorm,
                        'is_zero': is_zero,
                        'n_samples': len(pred_denorm)
                    }

                    print(f"  ✓ 收集到 {len(pred_denorm)} 个样本")
                else:
                    print(f"  ⚠ 没有有效样本")

            if not all_data:
                print("❌ 没有收集到任何数据")
                return

            # 创建图形
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # ============ 左图：所有样本散点图 ============
            ax1 = axes[0]

            # 颜色和标记设置
            colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
            markers = {'train': 'o', 'val': 's', 'test': '^'}
            labels = {'train': '训练集', 'val': '验证集', 'test': '测试集'}

            # 绘制每个数据集的点
            for dataset_name, data in all_data.items():
                ax1.scatter(data['target'], data['pred'], 
                           alpha=0.6, s=30, c=colors[dataset_name], 
                           marker=markers[dataset_name], 
                           label=f"{labels[dataset_name]} (n={data['n_samples']})",
                           edgecolors='white', linewidth=0.5)

            # 1:1线
            all_targets = np.concatenate([d['target'] for d in all_data.values()])
            all_preds = np.concatenate([d['pred'] for d in all_data.values()])
            min_val = min(all_targets.min(), all_preds.min())
            max_val = max(all_targets.max(), all_preds.max())
            margin = (max_val - min_val) * 0.05

            ax1.plot([min_val, max_val], [min_val, max_val], 
                    'k--', linewidth=2, alpha=0.8, label='1:1线')

            # 设置属性
            ax1.set_xlabel('真实值 (mm)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('预测值 (mm)', fontsize=14, fontweight='bold')
            ax1.set_title('所有微调样本预测结果', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=12, loc='lower right')
            ax1.set_xlim(min_val - margin, max_val + margin)
            ax1.set_ylim(min_val - margin, max_val + margin)

            # 添加整体统计信息
            all_corr = np.corrcoef(all_preds, all_targets)[0, 1]
            all_rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
            all_mae = np.mean(np.abs(all_preds - all_targets))
            all_bias = np.mean(all_preds - all_targets)
            all_r2 = 1 - np.sum((all_preds - all_targets) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)

            stats_text = (f'所有样本 (n={len(all_targets)})\n'
                         f'R = {all_corr:.4f}\n'
                         f'R² = {all_r2:.4f}\n'
                         f'RMSE = {all_rmse:.2f} mm\n'
                         f'MAE = {all_mae:.2f} mm\n'
                         f'Bias = {all_bias:.2f} mm')

            bbox_props = dict(
                boxstyle="round,pad=0.65",
                # 背景透明，文字本身仍完全不透明
                facecolor=(1.0, 1.0, 1.0, 0.45),
                edgecolor=(0.0, 0.0, 0.0, 0.75),
                linewidth=1.5,
            )

            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                    fontsize=16, fontweight='bold', verticalalignment='top',
                    horizontalalignment='left', bbox=bbox_props)

            # ============ 右图：各数据集性能对比 ============
            ax2 = axes[1]

            # 准备数据
            dataset_names = []
            r_values = []
            rmse_values = []
            mae_values = []
            colors_bar = []

            for dataset_name, data in all_data.items():
                pred = data['pred']
                target = data['target']

                corr = np.corrcoef(pred, target)[0, 1]
                rmse = np.sqrt(np.mean((pred - target) ** 2))
                mae = np.mean(np.abs(pred - target))

                dataset_names.append(labels[dataset_name])
                r_values.append(corr)
                rmse_values.append(rmse)
                mae_values.append(mae)
                colors_bar.append(colors[dataset_name])

            # 设置条形图位置
            x = np.arange(len(dataset_names))
            width = 0.25

            # 绘制条形图：只展示 r / RMSE/10 / MAE/10
            bars1 = ax2.bar(x - width, r_values, width, label='r', color='goldenrod', alpha=0.8)
            bars2 = ax2.bar(x, [r/10 for r in rmse_values], width, label='RMSE/10', color='lightblue', alpha=0.8)
            bars3 = ax2.bar(x + width, [m/10 for m in mae_values], width, label='MAE/10', color='lightgreen', alpha=0.8)

            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            for bars, orig_values in [(bars2, rmse_values), (bars3, mae_values)]:
                for bar, orig in zip(bars, orig_values):
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01,
                        f'{orig:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

            ax2.set_xlabel('数据集', fontsize=14, fontweight='bold')
            ax2.set_ylabel('指标值', fontsize=14, fontweight='bold')
            ax2.set_title('各数据集性能对比', fontsize=16, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(dataset_names, fontsize=12)
            ax2.legend(loc='upper right', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1.1)

            plt.tight_layout()

            # 保存图像
            plot_path = self.save_dir / "all_finetune_samples_scatter.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\n✅ 散点图已保存: {plot_path}")

            # 打印详细统计
            print("\n📊 各数据集详细统计:")
            print("-" * 60)
            print(f"{'数据集':8s} {'样本数':8s} {'r':8s} {'RMSE':8s} {'MAE':8s} {'Bias':8s}")
            print("-" * 60)

            for dataset_name, data in all_data.items():
                pred = data['pred']
                target = data['target']

                corr = np.corrcoef(pred, target)[0, 1]
                rmse = np.sqrt(np.mean((pred - target) ** 2))
                mae = np.mean(np.abs(pred - target))
                bias = np.mean(pred - target)

                print(f"{labels[dataset_name]:8s} {data['n_samples']:8d} {corr:8.4f} {rmse:8.2f} {mae:8.2f} {bias:8.2f}")

            print("-" * 60)
            print(f"{'总体':8s} {len(all_targets):8d} {all_corr:8.4f} {all_rmse:8.2f} {all_mae:8.2f} {all_bias:8.2f}")

            return {
                'plot_path': str(plot_path),
                'train': all_data.get('train', {}),
                'val': all_data.get('val', {}),
                'test': all_data.get('test', {})
            }

        except Exception as e:
            print(f"❌ 绘制散点图失败: {e}")
            traceback.print_exc()
            return None

    def plot_density_scatter_hardcode(self, predictions, targets, is_fine_tune=False, use_raw=False, fold_index=None):
        """密度散点图 - 完整功能版（集成十折交叉验证命名）"""
        try:
            from matplotlib.colors import LogNorm
            from scipy.stats import gaussian_kde

            # ============ 设置中文字体 ============
            self.setup_chinese_fonts()

            # 准备数据
            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()

            # ============ 如果 use_raw=True，直接使用原始值 ============
            if use_raw:
                print("  【使用原始站点SWE值，不进行额外处理】")

            # 移除 NaN / inf
            mask = np.isfinite(predictions) & np.isfinite(targets)
            predictions = predictions[mask]
            targets = targets[mask]

            if len(predictions) == 0:
                print("警告：没有有效数据用于绘图")
                return None

            # ============ 计算指标 ============
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(np.mean((predictions - targets) ** 2))
            bias = np.mean(predictions - targets)

            # NSE（内部保留，不直接画在图上）
            if len(targets) > 1:
                ss_res = np.sum((predictions - targets) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                nse = np.nan

            # Pearson R
            if len(targets) > 1 and np.std(predictions) > 0 and np.std(targets) > 0:
                try:
                    r_value, _ = stats.pearsonr(predictions, targets)
                except Exception:
                    corr_matrix = np.corrcoef(predictions, targets)
                    r_value = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else np.nan
            else:
                r_value = np.nan

            # ============ 创建图形 ============
            fig, ax = plt.subplots(figsize=(10, 8))

            # ============ 计算密度并绘图 ============
            z_sorted = None
            if len(predictions) > 1:
                try:
                    xy = np.vstack([targets, predictions])
                    z = gaussian_kde(xy)(xy)

                    # 按密度排序，保证高密度点画在上面
                    idx = z.argsort()
                    x_sorted, y_sorted, z_sorted = targets[idx], predictions[idx], z[idx]

                    scatter = ax.scatter(
                        x_sorted,
                        y_sorted,
                        c=z_sorted,
                        cmap="viridis",
                        s=30,
                        alpha=0.7,
                        edgecolors="none",
                        norm=LogNorm(),
                    )
                except Exception:
                    # KDE 失败时退化为普通散点
                    scatter = ax.scatter(
                        targets,
                        predictions,
                        s=30,
                        alpha=0.7,
                        edgecolors="none",
                        color="blue",
                    )
                    z_sorted = None
            else:
                scatter = ax.scatter(
                    targets,
                    predictions,
                    s=30,
                    alpha=0.7,
                    edgecolors="none",
                    color="blue",
                )

            # ============================================================
            # [DIAG] 绘图硬检查：确认 Matplotlib 实际接收了多少个点。
            # 密集区大量重叠时，肉眼看到的点数会远少于真实 N。
            # ============================================================
            rendered_n = int(scatter.get_offsets().shape[0])
            low_n = int(np.sum(targets <= 5.0))
            mid_n = int(np.sum((targets > 5.0) & (targets <= 30.0)))
            high_n = int(np.sum(targets > 30.0))
            print("\n[PLOT CHECK]")
            print(f"  有效 target 数:          {len(targets):,}")
            print(f"  有效 prediction 数:      {len(predictions):,}")
            print(f"  Matplotlib实际散点数:    {rendered_n:,}")
            print(f"  图中 SWE ≤ 5 mm:        {low_n:,} ({low_n/len(targets)*100:.2f}%)")
            print(f"  图中 5 < SWE ≤ 30 mm:   {mid_n:,} ({mid_n/len(targets)*100:.2f}%)")
            print(f"  图中 SWE > 30 mm:       {high_n:,} ({high_n/len(targets)*100:.2f}%)")
            print(f"  图中 SWE最大值:          {targets.max():.2f} mm")
            if hasattr(self, "swe_min") and hasattr(self, "swe_max"):
                print(f"  反归一化范围:            [{self.swe_min:.6f}, {self.swe_max:.6f}] mm")

            # ============ 1:1 线 ============
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())

            # 防止所有值一样导致 margin=0
            if max_val == min_val:
                margin = max(1.0, abs(max_val) * 0.05 + 1.0)
            else:
                margin = (max_val - min_val) * 0.05

            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=3,
                alpha=0.8,
                label="1:1线",
            )

            # ============ 回归线 ============
            if len(targets) > 1:
                try:
                    coeffs = np.polyfit(targets, predictions, 1)
                    reg_line = np.poly1d(coeffs)
                    x_range = np.linspace(min_val, max_val, 100)
                    ax.plot(
                        x_range,
                        reg_line(x_range),
                        color="orange",
                        linewidth=3,
                        alpha=0.8,
                        label="回归线",
                    )
                except Exception:
                    print("回归线计算失败，跳过")

            # ============ 设置图形属性 ============
            # 预训练标签来自ERA5-Land产品，不得表述为真实值或观测值。
            if is_fine_tune:
                x_axis_label = "观测值 (mm)"
                title_prefix = "SWE微调预测结果"
            else:
                x_axis_label = "产品值 (mm)"
                title_prefix = "SWE预训练预测结果"

            ax.set_xlabel(
                x_axis_label,
                fontsize=16,
                fontweight="bold"
            )
            ax.set_ylabel(
                "预测值 (mm)",
                fontsize=16,
                fontweight="bold"
            )

            # 动态标题（加上折数）
            fold_label = f" (第 {fold_index} 折)" if fold_index is not None else ""
            title = f"{title_prefix}{fold_label}"

            ax.set_title(title, fontsize=18, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12, loc="lower right")
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)

            # ============ 添加颜色条 ============
            if z_sorted is not None:
                cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
                cbar.set_label("点密度", fontsize=14, fontweight="bold")

            # ============ 左上角指标框 ============
            # 按你的要求：不显示 R² / NSE
            metrics_text = (
                f"N = {len(targets)}\n"
                f"R = {r_value:.4f}\n"
                f"RMSE = {rmse:.2f} mm\n"
                f"MAE = {mae:.2f} mm\n"
                f"Bias = {bias:.2f} mm"
            )

            bbox_props = dict(
                boxstyle="round,pad=0.65",
                # 背景透明，文字本身仍完全不透明
                facecolor=(1.0, 1.0, 1.0, 0.45),
                edgecolor=(0.0, 0.0, 0.0, 0.75),
                linewidth=1.5,
            )

            ax.text(
                0.05,
                0.95,
                metrics_text,
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="left",
                bbox=bbox_props,
            )

            # ============ 调整布局并保存 ============
            plt.tight_layout()

            # 动态文件名（加上 fold_index）
            prefix = "fine_tune" if is_fine_tune else "density"
            suffix = f"_fold_{fold_index}" if fold_index is not None else ""
            plot_name = f"{prefix}_scatter_chinese{suffix}.png"

            plot_path = self.save_dir / plot_name
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✓ 密度散点图已保存: {plot_path}")

            # 返回值里保留 nse；同时保留 r2 兼容旧代码
            return {
                "mae": float(mae),
                "rmse": float(rmse),
                "r": float(r_value) if np.isfinite(r_value) else None,
                "nse": float(nse) if np.isfinite(nse) else None,
                "r2": float(nse) if np.isfinite(nse) else None,   # 兼容旧代码
                "bias": float(bias),
                "n_samples": int(len(targets)),
                "plot_path": str(plot_path),
            }

        except Exception as e:
            print(f"绘图失败: {e}")
            traceback.print_exc()
            return None
    

    
    def plot_test_set_labeled_scatter(self, test_loader=None, model_path=None):
        """

        """
        print("\n" + "="*70)
        print("📊 生成测试集全标注散点图")
        print("="*70)

        try:
            # 设置中文字体
            self.setup_chinese_fonts()

            # 获取测试集数据加载器
            if test_loader is None:
                if hasattr(self, 'test_loader') and self.test_loader is not None:
                    test_loader = self.test_loader
                    print(f"  使用测试集，样本数: {len(test_loader.dataset)}")
                else:
                    print("❌ 没有测试集数据加载器")
                    return

            # 加载模型（如果指定）
            if model_path:
                self._load_model_for_evaluation(model_path)

            # 收集测试集的所有预测结果和元数据
            print("\n1. 对测试集进行预测并收集元数据...")

            all_predictions = []
            all_targets = []
            all_station_ids = []
            all_dates = []
            all_locations = []  # 存储经纬度用于调试

            self.model.eval()

            # 获取数据集对象以访问meta_index
            if hasattr(test_loader.dataset, 'dataset'):
                # 如果是 Subset
                dataset = test_loader.dataset.dataset
                indices = test_loader.dataset.indices
                print(f"  数据集类型: Subset, 原始数据集样本数: {len(dataset.meta_index)}")
            else:
                # 如果不是 Subset
                dataset = test_loader.dataset
                indices = range(len(dataset))
                print(f"  数据集类型: 直接数据集, 样本数: {len(dataset)}")

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    # 🔥 兼容不同长度的 batch_data
                    if len(batch_data) >= 6:
                        conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe, indices_batch = batch_data[:6]
                    elif len(batch_data) >= 5:
                        conv_feats, point_feats, targets, is_zero_mask, raw_fused_swe = batch_data[:5]
                    elif len(batch_data) >= 4:
                        conv_feats, point_feats, targets, is_zero_mask = batch_data[:4]
                        raw_fused_swe = None
                    else:
                        conv_feats, point_feats, targets = batch_data[:3]
                        is_zero_mask = (targets > 0).float()
                        raw_fused_swe = None

                    batch_size = len(targets)

                    # 获取当前批次的元数据
                    start_idx = batch_idx * test_loader.batch_size
                    for i in range(batch_size):
                        if start_idx + i < len(indices):
                            meta_idx = indices[start_idx + i]
                            if meta_idx < len(dataset.meta_index):
                                meta = dataset.meta_index[meta_idx]
                                # 🔥 兼容不同的日期键名
                                date_val = meta.get('feature_date') or meta.get('label_date') or meta.get('date')
                                if date_val is not None:
                                    if hasattr(date_val, 'strftime'):
                                        date_str = date_val.strftime('%m-%d')
                                    else:
                                        date_str = str(date_val)
                                else:
                                    date_str = '未知'
                                all_station_ids.append(str(meta.get('station_id', 'unknown')))
                                all_dates.append(date_str)
                                all_locations.append((meta.get('original_longitude', 0), meta.get('original_latitude', 0)))
                            else:
                                all_station_ids.append(f"IDX{meta_idx}")
                                all_dates.append("未知")
                                all_locations.append((0, 0))

                    # 移动到设备
                    conv_feats = conv_feats.to(self.device)
                    point_feats = point_feats.to(self.device)
                    targets = targets.to(self.device)

                    # 前向传播
                    outputs = self.model(conv_feats, point_feats)

                    # 强制target为0的样本预测值为0
                    if torch.any(is_zero_mask == 0):
                        zero_indices = (is_zero_mask == 0).nonzero(as_tuple=True)[0]
                        if len(zero_indices) > 0:
                            outputs[zero_indices] = 0.0

                    # 收集结果
                    all_predictions.extend(outputs.cpu().numpy().flatten())
                    all_targets.extend(targets.cpu().numpy().flatten())

            # 转换为numpy数组
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            all_station_ids = np.array(all_station_ids)
            all_dates = np.array(all_dates)

            print(f"\n  收集到 {len(all_predictions)} 个测试样本")

            if len(all_predictions) == 0:
                print("❌ 没有预测结果")
                return

            # 移除NaN值
            mask = ~np.isnan(all_predictions) & ~np.isnan(all_targets)
            all_predictions = all_predictions[mask]
            all_targets = all_targets[mask]
            all_station_ids = all_station_ids[mask]
            all_dates = all_dates[mask]

            print(f"  有效样本数: {len(all_predictions)}")

            # 反归一化
            swe_min = getattr(self, 'swe_min', 0.0)
            swe_max = getattr(self, 'swe_max', 200.0)
            predictions_display = all_predictions * (swe_max - swe_min) + swe_min
            targets_display = all_targets * (swe_max - swe_min) + swe_min

            print(f"\n2. 生成全标注散点图...")

            # 创建图形 - 加大尺寸以适应标注
            fig, ax = plt.subplots(figsize=(20, 16))

            # 绘制所有点（大一点，便于查看）
            scatter = ax.scatter(targets_display, predictions_display, 
                                alpha=0.8, s=80, c='blue', edgecolors='black', linewidth=1, zorder=2)

            # 1:1线
            min_val = min(targets_display.min(), predictions_display.min())
            max_val = max(targets_display.max(), predictions_display.max())
            margin = (max_val - min_val) * 0.1
            plot_min = max(0, min_val - margin)
            plot_max = max_val + margin

            ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                   'r--', linewidth=2, alpha=0.8, label='1:1线', zorder=1)

            # 为每个点添加标注
            print("  添加站点ID和日期标注...")
            for i in range(len(targets_display)):
                # 组合标注文本
                label = f"{all_station_ids[i]}\n{all_dates[i]}"

                # 计算偏移量，避免标注重叠
                offset_x = 0
                offset_y = 0

                # 根据位置智能调整标注方向
                if i % 3 == 0:
                    offset_x, offset_y = 5, 5
                elif i % 3 == 1:
                    offset_x, offset_y = -5, -5
                else:
                    offset_x, offset_y = 5, -5

                # 添加标注
                ax.annotate(label, 
                           (targets_display[i], predictions_display[i]),
                           xytext=(offset_x, offset_y), 
                           textcoords='offset points',
                           fontsize=8,  # 字体大小
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='yellow', 
                                    alpha=0.7,
                                    edgecolor='black'),
                           ha='center', va='center',
                           zorder=3)

            # 设置图形属性
            ax.set_xlabel(f'真实值 (mm)', fontsize=16, fontweight='bold')
            ax.set_ylabel(f'预测值 (mm)', fontsize=16, fontweight='bold')
            ax.set_title('测试集预测结果 - 全标注（站点ID + 日期）', fontsize=18, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=14, loc='lower right')
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)

            # 添加统计信息框
            if len(targets_display) > 1:
                r_value, _ = stats.pearsonr(predictions_display, targets_display)
                rmse = np.sqrt(np.mean((predictions_display - targets_display) ** 2))
                mae = np.mean(np.abs(predictions_display - targets_display))
                bias = np.mean(predictions_display - targets_display)

                # 计算R²
                ss_res = np.sum((predictions_display - targets_display) ** 2)
                ss_tot = np.sum((targets_display - np.mean(targets_display)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                stats_text = (f'R = {r_value:.4f}\n'
                             f'R² = {r2:.4f}\n'
                             f'RMSE = {rmse:.2f} mm\n'
                             f'MAE = {mae:.2f} mm\n'
                             f'Bias = {bias:.2f} mm\n'
                             f'N = {len(targets_display)}')

                bbox_props = dict(
                    boxstyle="round,pad=0.65",
                    # 背景透明，文字本身仍完全不透明
                    facecolor=(1.0, 1.0, 1.0, 0.45),
                    edgecolor=(0.0, 0.0, 0.0, 0.75),
                    linewidth=1.5,
                )

                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=16, fontweight='bold', verticalalignment='top',
                       horizontalalignment='left', bbox=bbox_props)

            plt.tight_layout()

            # 保存图像
            plot_path = self.save_dir / "test_set_labeled_scatter.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\n✅ 测试集全标注散点图已保存: {plot_path}")

            # 同时生成一个只有站点ID的版本（更简洁）
            fig2, ax2 = plt.subplots(figsize=(18, 14))

            ax2.scatter(targets_display, predictions_display, 
                       alpha=0.8, s=60, c='blue', edgecolors='black', linewidth=1)
            ax2.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, alpha=0.8)

            # 只标注站点ID
            for i in range(len(targets_display)):
                ax2.annotate(all_station_ids[i], 
                            (targets_display[i], predictions_display[i]),
                            xytext=(3, 3), textcoords='offset points',
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))

            ax2.set_xlabel('真实值 (mm)', fontsize=16, fontweight='bold')
            ax2.set_ylabel('预测值 (mm)', fontsize=16, fontweight='bold')
            ax2.set_title('测试集预测结果 - 站点ID标注', fontsize=18, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(plot_min, plot_max)
            ax2.set_ylim(plot_min, plot_max)

            plt.tight_layout()

            # 保存简洁版本
            plot_path2 = self.save_dir / "test_set_labeled_scatter_simple.png"
            plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ 简洁版标注散点图已保存: {plot_path2}")

            # 打印异常点列表（预测值明显偏离的点）
            print("\n📋 异常点列表（预测误差较大）:")
            errors = np.abs(predictions_display - targets_display)
            threshold = np.percentile(errors, 90)  # 前10%误差最大的点
            outlier_indices = np.where(errors > threshold)[0]

            for idx in outlier_indices[:10]:  # 最多显示10个
                print(f"  站点 {all_station_ids[idx]}, 日期 {all_dates[idx]}: "
                      f"真实={targets_display[idx]:.1f}mm, 预测={predictions_display[idx]:.1f}mm, "
                      f"误差={errors[idx]:.1f}mm")

            return {
                'predictions': predictions_display,
                'targets': targets_display,
                'station_ids': all_station_ids,
                'dates': all_dates,
                'plot_path': str(plot_path)
            }

        except Exception as e:
            print(f"❌ 生成标注散点图失败: {e}")
            traceback.print_exc()
            return None

    def run_rf_baseline(self):
        """
        运行随机森林基线模型作为对比
        自动检测混合模式，只使用站点数据（不使用预训练样本）
        """
        print("\n" + "="*70)
        print("🌲 运行随机森林基线模型（与当前微调对比）")
        print("="*70)

        try:
            from sklearn.ensemble import RandomForestRegressor
            from scipy.stats import pearsonr
            import joblib

            # ============ 1. 准备训练数据 ============
            print("\n1. 准备训练数据...")

            X_train = []
            y_train = []
            X_test = []
            y_test = []

            # 检测是否为混合模式
            is_mixed_mode = False
            station_dataset = None

            if hasattr(self.train_loader, 'dataset'):
                dataset = self.train_loader.dataset
                if hasattr(dataset, 'station_indices'):
                    is_mixed_mode = True
                    station_dataset = dataset.station_dataset
                    print("  检测到混合模式，只提取站点样本...")
                elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'station_indices'):
                    is_mixed_mode = True
                    station_dataset = dataset.dataset.station_dataset
                    print("  检测到混合模式（嵌套），只提取站点样本...")

            if is_mixed_mode and station_dataset is not None:
                # 混合模式：从 train_loader 的 indices 中筛选站点样本
                train_indices = self.train_loader.dataset.indices
                station_count = 0

                for idx in train_indices:
                    if idx < len(station_dataset):
                        conv, point, target, is_zero, _ = station_dataset[idx]
                        # 混合模式下 point 是 (16,)
                        X_train.append(point.numpy())
                        y_train.append(target.numpy())
                        station_count += 1

                print(f"    提取到 {station_count} 个站点训练样本")

                # 测试集：test_loader 本身就是站点数据
                print("  收集测试集特征...")
                for batch in self.test_loader:
                    if len(batch) == 4:
                        _, point, target, _ = batch
                    else:
                        _, point, target, *_ = batch
                    # 测试集 loader 中 point 是 (batch_size, 16)
                    X_test.append(point.numpy())
                    y_test.append(target.numpy())

            else:
                # 普通模式：直接使用 train_loader 和 test_loader
                print("  普通模式，直接使用 train_loader...")

                print("  收集训练集特征...")
                for batch in self.train_loader:
                    if len(batch) == 4:
                        _, point, target, _ = batch
                    else:
                        _, point, target, *_ = batch
                    # 普通模式下 point 是 (batch_size, 16)
                    X_train.append(point.numpy())
                    y_train.append(target.numpy())

                print("  收集测试集特征...")
                for batch in self.test_loader:
                    if len(batch) == 4:
                        _, point, target, _ = batch
                    else:
                        _, point, target, *_ = batch
                    X_test.append(point.numpy())
                    y_test.append(target.numpy())

            # 检查是否有数据
            if len(X_train) == 0:
                print("  ❌ 没有收集到训练样本！")
                return None

            # ============ 彻底修复：鲁棒的合并逻辑 ============
            print(f"\n  合并前样本块数量: {len(X_train)}")

            # 1. 确保所有特征数组都是 2D 矩阵 (N, 16)
            processed_X = []
            for arr in X_train:
                if arr.ndim == 1:
                    # 如果是单个样本 (16,) -> 变成 (1, 16)
                    processed_X.append(arr.reshape(1, -1))
                else:
                    # 如果已经是 batch (batch_size, 16) -> 保持不变
                    processed_X.append(arr)

            X_train = np.vstack(processed_X)
            y_train = np.concatenate([np.atleast_1d(y) for y in y_train]).flatten()

            # 对测试集做同样处理
            if len(X_test) > 0:
                processed_X_test = [a.reshape(1, -1) if a.ndim == 1 else a for a in X_test]
                X_test = np.vstack(processed_X_test)
                y_test = np.concatenate([np.atleast_1d(y) for y in y_test]).flatten()
            else:
                X_test = np.array([])
                y_test = np.array([])

            print(f"  🏁 最终合并形状: X_train={X_train.shape}, y_train={y_train.shape}")
            if len(X_test) > 0:
                print(f"  🏁 最终合并形状: X_test={X_test.shape}, y_test={y_test.shape}")

            # 检查维度一致性
            if X_train.shape[0] != len(y_train):
                print(f"  ❌ 维度不一致: X_train={X_train.shape[0]}, y_train={len(y_train)}")
                return None

            print(f"\n  训练集: {X_train.shape[0]} 个样本, {X_train.shape[1]} 个特征")
            if len(X_test) > 0:
                print(f"  测试集: {X_test.shape[0]} 个样本")

            if len(y_train) > 0:
                print(f"  训练集目标值范围: [{y_train.min():.4f}, {y_train.max():.4f}]")
            if len(y_test) > 0:
                print(f"  测试集目标值范围: [{y_test.min():.4f}, {y_test.max():.4f}]")

            if X_train.shape[0] < 10:
                print(f"  ⚠ 训练样本太少 ({X_train.shape[0]} < 10)，跳过随机森林")
                return None

            # ============ 2. 训练随机森林 ============
            print("\n2. 训练随机森林模型...")

            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            rf_model.fit(X_train, y_train.ravel())
            print("  ✓ 训练完成")

            # ============ 3. 预测 ============
            print("\n3. 预测...")
            y_pred_train = rf_model.predict(X_train)
            if len(X_test) > 0:
                y_pred_test = rf_model.predict(X_test)
            else:
                y_pred_test = np.array([])

            # ============ 4. 计算指标 ============
            print("\n4. 计算评估指标...")

            swe_min = getattr(self, 'swe_min', 0.0)
            swe_max = getattr(self, 'swe_max', 200.0)

            y_train_denorm = y_train * (swe_max - swe_min) + swe_min
            y_pred_train_denorm = y_pred_train * (swe_max - swe_min) + swe_min

            if len(y_test) > 0:
                y_test_denorm = y_test * (swe_max - swe_min) + swe_min
                y_pred_test_denorm = y_pred_test * (swe_max - swe_min) + swe_min

            train_rmse = np.sqrt(mean_squared_error(y_train_denorm, y_pred_train_denorm))
            train_mae = mean_absolute_error(y_train_denorm, y_pred_train_denorm)
            train_r2 = r2_score(y_train_denorm, y_pred_train_denorm)
            train_r, _ = pearsonr(y_train_denorm.flatten(), y_pred_train_denorm.flatten())

            if len(y_test) > 0:
                test_rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_test_denorm))
                test_mae = mean_absolute_error(y_test_denorm, y_pred_test_denorm)
                test_r2 = r2_score(y_test_denorm, y_pred_test_denorm)
                test_r, _ = pearsonr(y_test_denorm.flatten(), y_pred_test_denorm.flatten())
            else:
                test_rmse = test_mae = test_r2 = test_r = 0.0

            # ============ 5. 打印结果 ============
            print("\n" + "="*60)
            print("📊 随机森林 vs 深度学习模型对比")
            print("="*60)

            print(f"\n【训练集表现】")
            print(f"  RMSE: {train_rmse:.2f} mm")
            print(f"  MAE:  {train_mae:.2f} mm")
            print(f"  R²:   {train_r2:.4f}")
            print(f"  R:    {train_r:.4f}")

            if len(y_test) > 0:
                print(f"\n【测试集表现】")
                print(f"  RMSE: {test_rmse:.2f} mm")
                print(f"  MAE:  {test_mae:.2f} mm")
                print(f"  R²:   {test_r2:.4f}")
                print(f"  R:    {test_r:.4f}")

            # ============ 6. 保存结果 ============
            print("\n6. 保存结果...")

            rf_path = self.save_dir / "random_forest_model.pkl"
            joblib.dump(rf_model, rf_path)
            print(f"  ✓ 随机森林模型保存到: {rf_path}")

            results = {
                'random_forest': {
                    'train': {
                        'rmse': float(train_rmse),
                        'mae': float(train_mae),
                        'r2': float(train_r2),
                        'r': float(train_r)
                    },
                    'test': {
                        'rmse': float(test_rmse) if len(y_test) > 0 else None,
                        'mae': float(test_mae) if len(y_test) > 0 else None,
                        'r2': float(test_r2) if len(y_test) > 0 else None,
                        'r': float(test_r) if len(y_test) > 0 else None
                    }
                },
                'n_train': len(X_train),
                'n_test': len(X_test) if len(X_test) > 0 else 0,
            }

            results_path = self.save_dir / "rf_comparison_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ 结果保存到: {results_path}")

            print("\n" + "="*60)
            print("✅ 随机森林对比实验完成!")
            print("="*60)

            return results

        except Exception as e:
            print(f"\n❌ 随机森林实验失败: {e}")
            traceback.print_exc()
            return None
        
    def run_cv_workflow_by_station(self):
        """
        按站点划分的十折交叉验证，支持 mixed mode。

        核心逻辑：
        1. 正式协议将旧split='test'内部1000条并回开发池，
           由全部7936条内部样本进行Nested 8/1/1。
        2. 每轮使用独立inner fold选checkpoint，outer fold只报告OOF；
           外部987条始终不参与训练、选模或内部OOF。
        3. mixed_mode=True 时：
           每折训练集 = 当前 fold 的训练站点样本 + 预训练伪标签样本
           每折验证集 = 当前 fold 的验证站点样本，不加入预训练样本
        4. mixed mode 训练样本统一返回 6 个值：
           conv, point, target, mask, raw_grid, source_flag
           source_flag=0 表示站点实测样本
           source_flag=1 表示预训练伪标签样本
        """

        import shutil

        train_only_overfit = bool(
            self.config.get("train_only_overfit", False)
        )
        inner_pilot_only = bool(
            self.config.get("inner_pilot_only", False)
        )
        nested_max_folds = int(
            self.config.get("nested_max_folds", 10)
        )
        if train_only_overfit and inner_pilot_only:
            raise ValueError(
                "train_only_overfit与inner_pilot_only不能同时启用"
            )
        if inner_pilot_only and nested_max_folds != 1:
            raise ValueError(
                "inner_pilot_only当前只允许nested_max_folds=1"
            )

        print(f"\n{'█' * 80}")
        if train_only_overfit:
            print("🧪 完整训练集过拟合诊断")
            print("   - 只运行Fold 1的8-fold训练池")
            print("   - 不使用inner选模，不生成outer OOF")
            print("   - 完整训练池每轮仅作同集评估")
        elif inner_pilot_only:
            print("🧪 Nested 8/1/1单Inner pilot")
            print("   - 只运行第一个确定性Nested划分")
            print("   - 8 folds训练，1 fold负责Inner选模")
            print("   - outer fold只保留，不加载、不预测、不输出OOF")
        else:
            print("🌟 按站点划分的十折交叉验证")
            print("   - 每折训练、inner选模、outer OOF三部分站点互斥")
        print("   - mixed_mode=True 时：训练集加入预训练伪标签样本")
        print("   - inner/outer均只使用站点实测样本")
        if self.config.get("include_fixed_internal_in_cv", False):
            print("   - 旧固定内部1000条已并回Nested CV")
            print("   - 外部987条仍在训练流程之外")
        else:
            print("   - 固定独立测试集 self.test_loader 不参与十折划分")
        print(f"{'█' * 80}")

        # ============================================================
        # 0. 保存固定独立测试集
        # ============================================================
        fixed_test_loader = self.test_loader

        if fixed_test_loader is not None:
            try:
                print(f"\n📌 固定独立测试集已保留: {len(fixed_test_loader.dataset)} 个样本")
            except Exception:
                print("\n📌 固定独立测试集已保留")
        else:
            print("\n⚠ 当前 self.test_loader 为空，后面不会做固定测试集评估")

        # ============================================================
        # 1. SourceFlagDataset：给样本追加来源标记
        # ============================================================
        class SourceFlagDataset(Dataset):
            """
            给任意 Dataset / Subset 的样本统一转成 6 个返回值：

            return conv, point, target, mask, raw_grid, source_flag

            source_flag:
                0 = station observation
                1 = pretrain pseudo-label
            """

            def __init__(self, dataset, source_flag):
                self.dataset = dataset
                self.source_flag = int(source_flag)

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                item = self.dataset[idx]

                if not isinstance(item, (tuple, list)):
                    raise RuntimeError(f"样本格式异常: {type(item)}")

                if len(item) >= 5:
                    conv = item[0]
                    point = item[1]
                    target = item[2]
                    mask = item[3]
                    raw_grid = item[4]

                elif len(item) == 4:
                    conv, point, target, mask = item
                    raw_grid = torch.as_tensor(0.0, dtype=torch.float32)

                elif len(item) == 3:
                    conv, point, target = item
                    mask = torch.where(
                        target > 0,
                        torch.ones_like(target),
                        torch.zeros_like(target)
                    )
                    raw_grid = torch.as_tensor(0.0, dtype=torch.float32)

                else:
                    raise RuntimeError(f"样本元素数量异常: len(item)={len(item)}")

                source = torch.as_tensor(self.source_flag, dtype=torch.long)

                return conv, point, target, mask, raw_grid, source

        # ============================================================
        # 2. 获取 station_ds / pretrain_ds
        # ============================================================
        is_mixed = self.config.get("mixed_mode", False)
        if train_only_overfit and is_mixed:
            raise ValueError(
                "完整训练集过拟合诊断只允许纯站点监督，"
                "不得加入预训练伪标签回放"
            )

        print(f"\n📊 检查数据集类型:")
        print(f"  mixed_mode 配置: {is_mixed}")
        print(f"  train_loader 类型: {type(self.train_loader)}")
        print(f"  train_loader.dataset 类型: {type(self.train_loader.dataset)}")

        current_ds = self.train_loader.dataset

        depth = 0
        while hasattr(current_ds, "dataset") and not hasattr(current_ds, "station_dataset"):
            print(f"  剥壳第 {depth + 1} 层: {type(current_ds).__name__} -> .dataset")
            current_ds = current_ds.dataset
            depth += 1

        print(f"  剥壳完成，共 {depth} 层，最终类型: {type(current_ds).__name__}")

        if is_mixed:
            if hasattr(self, "station_dataset") and hasattr(self, "pretrain_dataset"):
                station_ds = self.station_dataset
                pretrain_ds = self.pretrain_dataset
                pretrain_indices = getattr(self, "pretrain_indices", [])

                print("\n📊 使用 load_data() 保存的 mixed dataset 引用:")
                print(f"  station_dataset: {len(station_ds)} 个样本")
                print(f"  pretrain_dataset: {len(pretrain_ds)} 个样本")
                print(f"  selected pretrain: {len(pretrain_indices)} 个样本")

            elif hasattr(current_ds, "station_dataset"):
                station_ds = current_ds.station_dataset
                pretrain_ds = current_ds.pretrain_dataset
                pretrain_indices = (
                    current_ds.selected_pretrain
                    if hasattr(current_ds, "selected_pretrain")
                    else []
                )

                print("\n📊 从 train_loader 剥壳获得 mixed dataset:")
                print(f"  station_dataset: {len(station_ds)} 个样本")
                print(f"  pretrain_dataset: {len(pretrain_ds)} 个样本")
                print(f"  selected pretrain: {len(pretrain_indices)} 个样本")

            else:
                print("❌ mixed_mode=True 但无法找到 station_dataset / pretrain_dataset")
                print(f"  当前对象类型: {type(current_ds)}")
                return None

        else:
            if hasattr(self, "station_dataset"):
                station_ds = self.station_dataset
            else:
                station_ds = current_ds

            pretrain_ds = None
            pretrain_indices = []

            print(f"\n📊 纯站点模式:")
            print(f"  station_dataset: {len(station_ds)} 个样本")

        if not hasattr(station_ds, "meta_index"):
            print("❌ station_ds 没有 meta_index，无法按站点划分")
            return None

        # ============================================================
        # 3. 确定 station_cv 样本池
        # ============================================================
        if hasattr(self, "cv_pool_indices_override") and self.cv_pool_indices_override is not None:
            cv_pool_indices = list(self.cv_pool_indices_override)
            print("\n✅ 使用 load_data() 提供的完整 station_cv pool")
            print(f"  CV pool 样本数: {len(cv_pool_indices)}")
        else:
            cv_pool_indices = list(range(len(station_ds)))
            print("\n⚠ 未检测到 cv_pool_indices_override，默认使用 station_ds 全部样本")
            print(f"  CV pool 样本数: {len(cv_pool_indices)}")

        # 保险：过滤越界
        cv_pool_indices = [
            int(i) for i in cv_pool_indices
            if 0 <= int(i) < len(station_ds)
        ]

        if len(cv_pool_indices) == 0:
            print("❌ CV pool 为空，无法进行十折")
            return None

        # ============================================================
        # 4. 按站点分组
        # ============================================================
        print("\n📊 正在按站点分组...")

        station_to_samples = defaultdict(list)

        for idx in cv_pool_indices:
            meta = station_ds.meta_index[idx]
            station_id = meta.get("station_id", "unknown")

            if "," in str(station_id):
                station_id = str(station_id).split(",")[0]

            station_to_samples[station_id].append(idx)

        unique_stations = list(station_to_samples.keys())
        n_stations = len(unique_stations)
        total_station_samples = sum(len(v) for v in station_to_samples.values())

        if n_stations < 2:
            print("❌ 站点数不足，无法做交叉验证")
            return None

        n_splits = min(10, n_stations)

        print("\n📊 站点统计:")
        print(f"  唯一站点数: {n_stations}")
        print(f"  CV池站点样本总数: {total_station_samples}")
        print(f"  n_splits: {n_splits}")

        samples_per_station = [len(station_to_samples[s]) for s in unique_stations]
        print(
            f"  每站点样本数: "
            f"min={min(samples_per_station)}, "
            f"max={max(samples_per_station)}, "
            f"mean={np.mean(samples_per_station):.1f}"
        )

        # ============================================================
        # 5. 检查预训练模型路径
        # ============================================================
        pretrained_path = self.config.get("pretrained_model")

        if pretrained_path is None or not os.path.exists(pretrained_path):
            possible_paths = [
                Path("/root/autodl-tmp/experiments/swe_full_temporal_20260526_090443/pretrain_cv_fold1_best.pth"),
                Path("/root/autodl-tmp/DSTM/src/main/experiments/swe_full_random_fine_tune_encoders_20260320_220623/final_model.pth"),
                self.save_dir / "best_model.pth",
                self.save_dir / "final_model.pth",
            ]

            for path in possible_paths:
                if path.exists():
                    pretrained_path = str(path)
                    break

        if not pretrained_path or not os.path.exists(pretrained_path):
            print("\n❌ 错误: 预训练模型不存在")
            print(f"  pretrained_model = {self.config.get('pretrained_model')}")
            return None

        print(f"\n✅ 预训练模型: {pretrained_path}")

        # ============================================================
        # 6. 确定性平衡站点十折
        # ============================================================
        # BALANCED_STATION_CV10_V1
        # 只平衡三个最必要的量：总样本数、高SWE样本数、站点数。
        # 不使用随机数，不加入区域/年份等复杂约束。
        print("\n📊 加载确定性平衡站点十折...")

        from balanced_station_cv10 import (
            create_or_load_manifest,
            mapping_for_dataset_indices,
        )

        include_fixed_internal_in_cv = bool(
            self.config.get("include_fixed_internal_in_cv", False)
        )
        default_fold_manifest = (
            "/root/autodl-tmp/shared_cache/progressive_finetune/"
            "balanced_station_nested_cv10_all7936_manifest.csv"
            if include_fixed_internal_in_cv
            else
            "/root/autodl-tmp/shared_cache/progressive_finetune/"
            "balanced_station_cv10_manifest.csv"
        )
        fold_manifest_path = Path(
            self.config.get(
                "station_cv_fold_manifest_path",
                default_fold_manifest,
            )
        ).expanduser().resolve()

        station_csv_for_manifest = Path(
            self.config.get("station_data_path")
        ).expanduser().resolve()

        fold_manifest, fold_balance_summary, fold_metadata = (
            create_or_load_manifest(
                station_csv=station_csv_for_manifest,
                manifest_path=fold_manifest_path,
                n_splits=n_splits,
                high_threshold_mm=80.0,
                force_rebuild=False,
                include_fixed_test=include_fixed_internal_in_cv,
            )
        )

        fold_by_index, balanced_records = mapping_for_dataset_indices(
            dataset=station_ds,
            cv_indices=cv_pool_indices,
            manifest=fold_manifest,
        )

        fold_splits = []
        for record in balanced_records:
            fold_splits.append({
                "fold": int(record["fold"]),
                "train_station_indices": record["train_indices"],
                "val_station_indices": record["test_indices"],
                "train_stations": record["train_stations"],
                "val_stations": record["test_stations"],
                "n_train_samples": record["n_train_samples"],
                "n_val_samples": record["n_test_samples"],
                "n_train_stations": record["n_train_stations"],
                "n_val_stations": record["n_test_stations"],
            })

        # NESTED_STATION_CV_SELECTION_ISOLATION_V1
        # 原实现直接用外层留出fold选epoch，再在同一fold上报告OOF，
        # 会产生checkpoint选择偏差。现在采用确定性的rolling inner fold：
        #   outer fold       = 只报告OOF，从不参与选模
        #   next fold        = inner selection validation
        #   remaining folds  = 参数训练
        if bool(self.config.get("nested_station_cv", True)):
            offset = int(
                self.config.get("nested_inner_fold_offset", 1)
            )
            if offset <= 0 or offset >= len(fold_splits):
                raise ValueError(
                    "nested_inner_fold_offset必须位于"
                    f"[1, {len(fold_splits) - 1}]，实际={offset}"
                )

            nested_splits = []
            cv_pool_set = set(int(i) for i in cv_pool_indices)

            for position, outer_record in enumerate(fold_splits):
                inner_record = fold_splits[
                    (position + offset) % len(fold_splits)
                ]
                outer_indices = [
                    int(i)
                    for i in outer_record["val_station_indices"]
                ]
                selection_indices = [
                    int(i)
                    for i in inner_record["val_station_indices"]
                ]
                excluded = set(outer_indices) | set(selection_indices)
                train_indices = sorted(cv_pool_set - excluded)

                train_set = set(train_indices)
                selection_set = set(selection_indices)
                outer_set = set(outer_indices)
                if train_set & selection_set:
                    raise RuntimeError("nested CV训练集与选模集重叠")
                if train_set & outer_set:
                    raise RuntimeError("nested CV训练集与外层OOF集重叠")
                if selection_set & outer_set:
                    raise RuntimeError("nested CV选模集与外层OOF集重叠")
                if train_set | selection_set | outer_set != cv_pool_set:
                    raise RuntimeError("nested CV三部分没有完整覆盖CV池")

                nested_splits.append({
                    **outer_record,
                    "train_station_indices": train_indices,
                    "selection_val_station_indices": selection_indices,
                    "outer_test_station_indices": outer_indices,
                    "train_stations": sorted({
                        str(
                            station_ds.meta_index[i]
                            .get("station_id", "unknown")
                        ).split(",")[0]
                        for i in train_indices
                    }),
                    "selection_val_stations": list(
                        inner_record["val_stations"]
                    ),
                    "outer_test_stations": list(
                        outer_record["val_stations"]
                    ),
                    "n_train_samples": len(train_indices),
                    "n_selection_val_samples": len(selection_indices),
                    "n_outer_test_samples": len(outer_indices),
                    "n_train_stations": len({
                        str(
                            station_ds.meta_index[i]
                            .get("station_id", "unknown")
                        ).split(",")[0]
                        for i in train_indices
                    }),
                    "n_selection_val_stations": len(
                        inner_record["val_stations"]
                    ),
                    "n_outer_test_stations": len(
                        outer_record["val_stations"]
                    ),
                    "selection_fold": int(inner_record["fold"]),
                })

            fold_splits = nested_splits

        print(f"  fold manifest: {fold_manifest_path}")
        print(f"  method: {fold_metadata['method']}")
        print("  randomized: False")
        print("  平衡摘要:")
        print(fold_balance_summary.to_string(index=False))

        print(
            f"\n  {'Outer':<6} {'Inner':<6} {'训练站点':<10} "
            f"{'选模站点':<10} {'OOF站点':<10} "
            f"{'训练样本':<10} {'选模样本':<10} {'OOF样本':<10}"
        )
        print(f"  {'-' * 90}")
        for split in fold_splits:
            print(
                f"  {split['fold']:<6} "
                f"{split.get('selection_fold', '-'):<6} "
                f"{split['n_train_stations']:<10} "
                f"{split.get('n_selection_val_stations', split['n_val_stations']):<10} "
                f"{split.get('n_outer_test_stations', split['n_val_stations']):<10} "
                f"{split['n_train_samples']:<10} "
                f"{split.get('n_selection_val_samples', split['n_val_samples']):<10} "
                f"{split.get('n_outer_test_samples', split['n_val_samples']):<10}"
            )

        # ============================================================
        # 7. 保存 fold split 信息
        # ============================================================
        try:
            split_save_path = self.save_dir / "station_cv_fold_splits.json"
            serializable_splits = []

            for s in fold_splits:
                serializable_splits.append({
                    "fold": s["fold"],
                    "selection_fold": s.get("selection_fold"),
                    "n_train_samples": s["n_train_samples"],
                    "n_selection_val_samples": s.get(
                        "n_selection_val_samples",
                        s["n_val_samples"],
                    ),
                    "n_outer_test_samples": s.get(
                        "n_outer_test_samples",
                        s["n_val_samples"],
                    ),
                    "n_train_stations": s["n_train_stations"],
                    "n_selection_val_stations": s.get(
                        "n_selection_val_stations",
                        s["n_val_stations"],
                    ),
                    "n_outer_test_stations": s.get(
                        "n_outer_test_stations",
                        s["n_val_stations"],
                    ),
                    "train_stations": [str(x) for x in s["train_stations"]],
                    "selection_val_stations": [
                        str(x)
                        for x in s.get(
                            "selection_val_stations",
                            s["val_stations"],
                        )
                    ],
                    "outer_test_stations": [
                        str(x)
                        for x in s.get(
                            "outer_test_stations",
                            s["val_stations"],
                        )
                    ],
                })

            with open(split_save_path, "w", encoding="utf-8") as f:
                json.dump(serializable_splits, f, indent=2, ensure_ascii=False)

            print(f"\n💾 fold 划分已保存: {split_save_path}")

        except Exception as e:
            print(f"⚠ 保存 fold 划分失败: {e}")

        # ============================================================
        # 8. 十折训练
        # ============================================================
        all_val_predictions = []
        all_val_targets = []
        all_fold_metrics = []
        all_oof_rows = []
        fold_model_paths = {}

        original_test_loader = fixed_test_loader

        fold_iteration_splits = (
            [fold_splits[0]]
            if train_only_overfit or inner_pilot_only
            else fold_splits
        )

        for split in fold_iteration_splits:
            fold_idx = split["fold"]
            fold_seed = (
                int(self.config.get("seed", 43))
                + int(fold_idx) * 1000
            )
            self.config["_active_dataloader_seed"] = fold_seed
            _configure_reproducibility(
                fold_seed,
                verbose=False,
            )

            print(f"\n\n{'=' * 70}")
            print(f"🟢 FOLD {fold_idx} / {n_splits}")
            print(f"  确定性fold seed: {fold_seed}")
            print(f"  训练站点: {split['n_train_stations']} 个, 样本: {split['n_train_samples']}")
            if train_only_overfit:
                print("  诊断模式: inner与outer均不参与本次运行")
            elif inner_pilot_only:
                print(
                    "  Pilot模式: Inner参与选模；Outer仅保留，"
                    "本次完全不评估"
                )
                print(
                    "  内层选模站点: "
                    f"{split.get('n_selection_val_stations', split['n_val_stations'])} 个, "
                    "样本: "
                    f"{split.get('n_selection_val_samples', split['n_val_samples'])}"
                )
                print(
                    "  保留Outer站点: "
                    f"{split.get('n_outer_test_stations', split['n_val_stations'])} 个, "
                    "样本: "
                    f"{split.get('n_outer_test_samples', split['n_val_samples'])}"
                )
            else:
                print(
                    "  内层选模站点: "
                    f"{split.get('n_selection_val_stations', split['n_val_stations'])} 个, "
                    "样本: "
                    f"{split.get('n_selection_val_samples', split['n_val_samples'])}"
                )
                print(
                    "  外层OOF站点: "
                    f"{split.get('n_outer_test_stations', split['n_val_stations'])} 个, "
                    "样本: "
                    f"{split.get('n_outer_test_samples', split['n_val_samples'])}"
                )

            if is_mixed and pretrain_indices:
                print(f"  预训练样本: {len(pretrain_indices)} 个，固定加入训练")
                print(f"  pretrain_loss_weight = {self.config.get('pretrain_loss_weight', 0.05)}")

            print(f"{'=' * 70}")

            # ============ 8.1 创建训练集 ============
            station_train_subset_raw = Subset(
                station_ds,
                split["train_station_indices"]
            )
            station_train_subset = SourceFlagDataset(
                station_train_subset_raw,
                source_flag=0
            )

            if is_mixed and pretrain_indices and len(pretrain_indices) > 0:
                pretrain_subset_raw = Subset(
                    pretrain_ds,
                    pretrain_indices
                )
                pretrain_subset = SourceFlagDataset(
                    pretrain_subset_raw,
                    source_flag=1
                )

                train_dataset = ConcatDataset([
                    station_train_subset,
                    pretrain_subset
                ])

                print(
                    f"  📦 训练集: 站点样本 {len(station_train_subset)} "
                    f"+ 预训练样本 {len(pretrain_subset)} "
                    f"= {len(train_dataset)}"
                )

            else:
                train_dataset = station_train_subset
                print(f"  📦 训练集: 站点样本 {len(station_train_subset)}")

            # mixed mode 下建议不使用 dataset 内部增强，防止 source 混乱
            if hasattr(station_ds, "set_augmentation_mode"):
                if is_mixed:
                    station_ds.set_augmentation_mode(False)
                    print("  mixed_mode: 关闭 station dataset augmentation")
                else:
                    station_ds.set_augmentation_mode(True)
                    print("  station mode: 启用 station dataset augmentation")

            if not is_mixed:
                train_targets_mm = [
                    float(station_ds.meta_index[int(i)]["swe"])
                    for i in split["train_station_indices"]
                ]
                self.train_loader = self._make_swe_stratified_train_loader(
                    train_dataset,
                    train_targets_mm,
                    context=f"Fold {fold_idx}",
                )
            else:
                # mixed模式包含伪标签样本，暂时保持原随机shuffle，
                # 避免把站点SWE配额错误应用到ConcatDataset的伪标签部分。
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config["batch_size"],
                    shuffle=True,
                    num_workers=self.config.get("num_workers", 8),
                    pin_memory=True,
                    drop_last=True,
                    **_dataloader_seed_kwargs(fold_seed),
                )

            # ============ 8.2 创建验证集 ============
            if train_only_overfit:
                selection_val_indices = list(
                    split["train_station_indices"]
                )
                outer_test_indices = []
            else:
                selection_val_indices = split.get(
                    "selection_val_station_indices",
                    split["val_station_indices"],
                )
                outer_test_indices = split.get(
                    "outer_test_station_indices",
                    split["val_station_indices"],
                )

            val_subset = Subset(
                station_ds,
                selection_val_indices,
            )

            if hasattr(station_ds, "set_augmentation_mode"):
                station_ds.set_augmentation_mode(False)

            self.val_loader = DataLoader(
                val_subset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config.get("num_workers", 8),
                pin_memory=True,
                **_dataloader_seed_kwargs(fold_seed + 1),
            )
            if train_only_overfit:
                outer_oof_loader = None
                self.train_audit_loader = None
                print(
                    f"  🧪 完整训练池同集评估: {len(val_subset)} 个样本"
                )
                print(
                    "     augmentation=False, shuffle=False；"
                    "不参与选模、调度或早停"
                )
            else:
                if inner_pilot_only:
                    outer_oof_subset = None
                    outer_oof_loader = None
                else:
                    outer_oof_subset = Subset(
                        station_ds,
                        outer_test_indices,
                    )
                    outer_oof_loader = DataLoader(
                        outer_oof_subset,
                        batch_size=self.config["batch_size"],
                        shuffle=False,
                        num_workers=self.config.get("num_workers", 8),
                        pin_memory=True,
                        **_dataloader_seed_kwargs(fold_seed + 2),
                    )

                # TRAIN_VS_INNER_AUDIT_V1
                # 从本折8-fold训练池中固定抽取与inner fold等量的站点样本。
                # 只做model.eval()前向诊断，不参与反向传播、scheduler、选模或早停。
                audit_size = min(
                    len(split["train_station_indices"]),
                    len(selection_val_indices),
                )
                audit_rng = np.random.default_rng(fold_seed + 3)
                audit_indices = sorted(
                    int(i)
                    for i in audit_rng.choice(
                        np.asarray(
                            split["train_station_indices"],
                            dtype=np.int64,
                        ),
                        size=audit_size,
                        replace=False,
                    ).tolist()
                )
                train_audit_subset = Subset(
                    station_ds,
                    audit_indices,
                )
                self.train_audit_loader = DataLoader(
                    train_audit_subset,
                    batch_size=self.config["batch_size"],
                    shuffle=False,
                    num_workers=self.config.get("num_workers", 8),
                    pin_memory=True,
                    **_dataloader_seed_kwargs(fold_seed + 3),
                )

                if inner_pilot_only:
                    print(
                        f"  📦 内层选模集: {len(val_subset)} 个站点样本"
                    )
                    print(
                        "  🔒 Outer fold已保留且完全未加载: "
                        f"{len(outer_test_indices)} 个站点样本"
                    )
                else:
                    print(
                        f"  📦 内层选模集: {len(val_subset)} 个站点样本；"
                        f"外层OOF集: {len(outer_oof_subset)} 个站点样本"
                    )
                print(
                    f"  🔬 固定训练审计集: {len(train_audit_subset)} 个样本；"
                    "仅输出Train-vs-Inner诊断，不参与选模"
                )

            # ============ 8.3 检查一个训练 batch ============
            try:
                batch_data = next(iter(self.train_loader))
                print(f"  🔎 训练 batch 返回元素数: {len(batch_data)}")

                if len(batch_data) >= 6:
                    source_flag_batch = batch_data[5]
                    n_station = int((source_flag_batch == 0).sum().item())
                    n_pretrain = int((source_flag_batch == 1).sum().item())
                    print(f"     source_flag: station={n_station}, pretrain={n_pretrain}")
                else:
                    print("     ⚠ 训练 batch 没有 source_flag，请检查 SourceFlagDataset")

            except Exception as e:
                print(f"  ⚠ 训练 batch 检查失败: {e}")

            # ============ 8.4 构建模型 ============
            print(f"\n🏗️ [FOLD {fold_idx}] 构建模型...")

            freeze_backbone = self.config.get("freeze_backbone", True)
            freeze_strategy = self.config.get("freeze_strategy", "fusion_ft")
            use_residual_injection = self.config.get("residual_injection", False)

            success = self.build_model(
                load_pretrained=pretrained_path,
                freeze_backbone=freeze_backbone,
                freeze_strategy=freeze_strategy,
                use_residual=use_residual_injection,
                is_cv_fold=True,
            )

            if not success:
                print(f"❌ [FOLD {fold_idx}] 模型构建失败，跳过")
                continue

            # HIGH_SWE_TRANSFER_AUDIT_V1
            # 只复用本折既定8-fold Train与1-fold Inner；Outer没有DataLoader。
            # build_model可能为普通微调预建optimizer；这里显式丢弃，
            # 保证后续不可能执行optimizer/scheduler step，然后直接返回。
            if bool(
                self.config.get(
                    "high_swe_transfer_audit_only",
                    False,
                )
            ):
                if not inner_pilot_only:
                    raise RuntimeError(
                        "high_swe_transfer_audit_only仅允许"
                        "inner_pilot_only模式"
                    )
                self.optimizer = None
                self.scheduler = None
                return self._run_high_swe_transfer_audit(
                    station_ds=station_ds,
                    train_indices=split[
                        "train_station_indices"
                    ],
                    inner_indices=selection_val_indices,
                    fold_seed=fold_seed,
                )

            # ============ 8.5 优化器 ============
            # FINETUNE_STABLE_GROUPED_OPTIMIZER_V1
            current_trainable_params = [
                p for p in self.model.parameters()
                if p.requires_grad
            ]

            if not current_trainable_params:
                print("\n❌ 错误: 没有可训练参数")
                continue

            # build_model 已按 head / transformer / encoder 创建分组优化器。
            # 旧代码在这里重新创建单一LR AdamW，导致Transformer也可能使用5e-4，
            # 在分层采样+高值损失后很容易一步破坏预训练表征。
            if self.optimizer is None:
                fallback_lr = min(
                    float(self.config.get("fine_tune_lr", 5e-5)),
                    1e-4,
                )
                self.optimizer = optim.AdamW(
                    current_trainable_params,
                    lr=fallback_lr,
                    weight_decay=1e-4,
                )
                print(
                    f"  ⚠ build_model未返回优化器，使用保守统一LR={fallback_lr:.2e}"
                )

            strategy_name = str(
                self.config.get("freeze_strategy", "fusion_ft")
            ).lower()

            progressive_fusion_configured = False
            self._fusion_ft_progressive_optimizer = False
            if strategy_name == "fusion_ft":
                progressive_fusion_configured = (
                    self._configure_fusion_ft_progressive_optimizer()
                )

            if not progressive_fusion_configured:
                safe_caps = {
                    # 即使显式关闭渐进式解冻，Fusion FT也使用降低后的保守上限。
                    "fusion_ft":       {"head": 5e-5, "transformer": 2e-6, "encoder": 2e-6},
                    "point_ft":        {"head": 2e-4, "transformer": 2e-5, "encoder": 2e-5},
                    "spatial_ft":      {"head": 2e-4, "transformer": 2e-5, "encoder": 2e-5},
                    "partial":         {"head": 2e-4, "transformer": 1e-5, "encoder": 1e-6},
                    "none":            {"head": 1e-4, "transformer": 1e-5, "encoder": 1e-5},
                }
                caps = safe_caps.get(
                    strategy_name,
                    {"head": 1e-4, "transformer": 1e-5, "encoder": 1e-5},
                )

                name_by_id = {
                    id(param): name
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                }

                print(f"  ✅ 使用分组学习率（strategy={strategy_name}）:")
                for group_idx, group in enumerate(self.optimizer.param_groups, start=1):
                    group_names = [
                        name_by_id.get(id(param), "")
                        for param in group.get("params", [])
                    ]
                    joined = " ".join(group_names).lower()
                    if any(k in joined for k in ["head", "regression", "output", "correction"]):
                        group_type = "head"
                    elif "transformer" in joined:
                        group_type = "transformer"
                    else:
                        group_type = "encoder"

                    old_lr = float(group.get("lr", caps[group_type]))
                    new_lr = min(old_lr, float(caps[group_type]))
                    group["lr"] = new_lr
                    print(
                        f"     group{group_idx}: {group_type:<11s} "
                        f"lr {old_lr:.2e} -> {new_lr:.2e}, "
                        f"params={len(group.get('params', []))}"
                    )

            self.scheduler_step_per_batch = False
            if train_only_overfit:
                self.scheduler = None
                print(
                    "  🧪 过拟合诊断使用固定峰值学习率："
                    "warmup结束后不再自动衰减"
                )
            else:
                plateau_patience = int(
                    self.config.get(
                        "fusion_ft_plateau_patience",
                        8,
                    )
                )
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=0.5,
                    patience=plateau_patience,
                    min_lr=1e-7,
                )
                print(
                    "  ✅ 正式Fusion FT调度器: "
                    "ReduceLROnPlateau("
                    f"patience={plateau_patience}, factor=0.5)"
                )

            # PROGRESSIVE_CV_CLEAR_SHARED_BEST_V1
            # train()在所有折中共用self.save_dir；如果不清理，
            # 本折训练失败时可能误复制上一折的best checkpoint。
            shared_fold_candidates = [
                self.save_dir / "best_fine_tuned_model.pth",
                self.save_dir / "best_fine_tuned_global_rmse.pth",
                self.save_dir / "best_fine_tuned_frozen_baseline.pth",
                self.save_dir / "final_fine_tuned_model.pth",
                self.save_dir / "last_trained_model.pth",
                self.save_dir / "best_model.pth",
                self.save_dir / "final_model.pth",
            ]
            for stale_path in shared_fold_candidates:
                if stale_path.exists():
                    stale_path.unlink()
                    print(
                        f"  🧹 [FOLD {fold_idx}] 清除上一折共享checkpoint: "
                        f"{stale_path.name}"
                    )

            # ============ 8.6 训练 ============
            print(f"\n🚀 [FOLD {fold_idx}] 开始训练...")

            original_epochs = self.config.get("epochs", 100)
            original_patience = self.config.get("patience", 25)

            fold_max_epochs = int(min(
                original_epochs,
                self.config.get("fine_tune_epochs", 50),
            ))
            self.config["epochs"] = fold_max_epochs
            if train_only_overfit:
                self.config["patience"] = fold_max_epochs + 1
                print(
                    "  🧪 已关闭early stopping、Frozen-relative门槛、"
                    "回滚和outer OOF"
                )
                print(
                    f"     固定运行 {fold_max_epochs} epochs，"
                    "每轮在完整训练池上评估"
                )
            elif strategy_name == "fusion_ft":
                self.config["patience"] = int(
                    self.config.get("fusion_ft_patience", 20)
                )
                print(
                    f"  ⏳ Fusion FT early stopping: "
                    f"min_epochs={int(self.config.get('fusion_ft_min_epochs_before_early_stop', 40))}, "
                    f"patience={self.config['patience']}, "
                    f"max_epochs={fold_max_epochs}"
                )
            else:
                self.config["patience"] = min(original_patience, 10)

            train_result = self.train(
                fine_tune_mode=True,
                is_cv_sub_run=True
            )

            fold_epochs_ran = int(
                (train_result or {}).get("total_epochs", fold_max_epochs)
            )
            fold_best_epoch_admissible = int(
                (train_result or {}).get("best_epoch", -1)
            ) + 1
            fold_best_epoch_global_rmse = None
            fold_selected_is_frozen_fallback = bool(
                (train_result or {}).get(
                    "selected_is_frozen_fallback",
                    True,
                )
            )
            fold_admissible_improvement_found = bool(
                (train_result or {}).get(
                    "admissible_improvement_found",
                    False,
                )
            )

            self.config["epochs"] = original_epochs
            self.config["patience"] = original_patience

            if train_only_overfit:
                final_train_metrics = self.validate(
                    is_fine_tune=True
                )
                metric_keys = [
                    "loss",
                    "rmse_mm",
                    "mae_mm",
                    "correlation",
                    "ccc",
                    "r2",
                    "bias_mm",
                    "slope",
                    "std_ratio",
                    "pred_std_mm",
                    "obs_std_mm",
                    "rmse_ge50_mm",
                    "bias_ge80_mm",
                    "n_samples",
                ]

                def _json_scalar(value):
                    if isinstance(value, np.generic):
                        return value.item()
                    if isinstance(value, (int, float, str, bool)):
                        return value
                    return None

                final_payload = {
                    "mode": "train_only_overfit",
                    "formal_cv_result": False,
                    "resumed": bool(
                        self.config.get(
                            "_overfit_resume_source"
                        )
                    ),
                    "resume_checkpoint": self.config.get(
                        "_overfit_resume_source"
                    ),
                    "epochs_completed_before_resume": int(
                        self.config.get(
                            "_overfit_resume_completed_epochs",
                            0,
                        )
                    ),
                    "fold": int(fold_idx),
                    "n_train_samples": int(
                        len(split["train_station_indices"])
                    ),
                    "n_train_stations": int(
                        split["n_train_stations"]
                    ),
                    "epochs_requested": int(fold_max_epochs),
                    "epochs_ran": int(fold_epochs_ran),
                    "scheduler": "disabled_after_warmup",
                    "early_stopping": False,
                    "inner_selection": False,
                    "outer_oof": False,
                    "metrics": {
                        key: _json_scalar(final_train_metrics.get(key))
                        for key in metric_keys
                    },
                }
                overfit_summary_path = (
                    self.save_dir
                    / "train_only_overfit_summary.json"
                )
                with open(
                    overfit_summary_path,
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        final_payload,
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                epoch_rows = []
                for epoch_index, metrics in enumerate(
                    getattr(self, "val_history_metrics", []),
                    start=1,
                ):
                    row = {"epoch": int(epoch_index)}
                    for key in metric_keys:
                        row[key] = _json_scalar(metrics.get(key))
                    epoch_rows.append(row)
                epoch_metrics_path = (
                    self.save_dir
                    / "train_only_overfit_epoch_metrics.csv"
                )
                pd.DataFrame(epoch_rows).to_csv(
                    epoch_metrics_path,
                    index=False,
                )

                print("\n" + "=" * 70)
                print("🧪 完整训练集过拟合诊断完成")
                print(
                    f"  样本/站点: "
                    f"{final_payload['n_train_samples']}/"
                    f"{final_payload['n_train_stations']}"
                )
                print(
                    f"  Epochs: {fold_epochs_ran}/{fold_max_epochs}"
                )
                print(
                    f"  R={final_train_metrics.get('correlation', float('nan')):.4f}, "
                    f"CCC={final_train_metrics.get('ccc', float('nan')):.4f}, "
                    f"RMSE={final_train_metrics.get('rmse_mm', float('nan')):.2f} mm"
                )
                print(
                    f"  slope={final_train_metrics.get('slope', float('nan')):.4f}, "
                    f"std_ratio={final_train_metrics.get('std_ratio', float('nan')):.4f}"
                )
                print(f"  汇总: {overfit_summary_path}")
                print(f"  逐轮指标: {epoch_metrics_path}")
                print("  ⚠ 该结果仅用于拟合能力诊断，不是CV/OOF结果")
                print("=" * 70)
                return final_payload

            # ============ 8.7 保存当前 fold 的正式模型 ============
            # FROZEN_RELATIVE_CANONICAL_CHECKPOINT_V1
            # canonical只允许指向：
            #   - 通过Frozen-relative硬门槛的最佳模型；或
            #   - 没有合格epoch时的epoch-0 Frozen安全回退。
            # 无约束最低RMSE仅作诊断，不得覆盖canonical。
            fold_model_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_best_model.pth"
            )
            fold_admissible_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_best_admissible.pth"
            )
            fold_global_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_best_global_rmse.pth"
            )
            fold_frozen_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_frozen_baseline.pth"
            )
            fold_last_trained_path = (
                self.save_dir
                / f"cv_fold_{fold_idx:02d}_last_trained.pth"
            )

            global_source = (
                self.save_dir
                / "best_fine_tuned_global_rmse.pth"
            )
            canonical_source = (
                self.save_dir
                / "best_fine_tuned_model.pth"
            )
            frozen_source = (
                self.save_dir
                / "best_fine_tuned_frozen_baseline.pth"
            )
            last_trained_source = (
                self.save_dir
                / "last_trained_model.pth"
            )

            if not last_trained_source.exists():
                raise RuntimeError(
                    f"Fold {fold_idx}缺少真实最后训练态: "
                    f"{last_trained_source}"
                )

            if not canonical_source.exists():
                raise RuntimeError(
                    f"Fold {fold_idx}缺少Frozen-relative canonical模型: "
                    f"{canonical_source}"
                )

            if not frozen_source.exists():
                raise RuntimeError(
                    f"Fold {fold_idx}缺少epoch-0 Frozen基线: "
                    f"{frozen_source}"
                )

            shutil.copy2(
                canonical_source,
                fold_admissible_path,
            )
            shutil.copy2(
                canonical_source,
                fold_model_path,
            )
            shutil.copy2(
                frozen_source,
                fold_frozen_path,
            )
            shutil.copy2(
                last_trained_source,
                fold_last_trained_path,
            )
            if global_source.exists():
                shutil.copy2(
                    global_source,
                    fold_global_path,
                )
                global_checkpoint_meta = torch.load(
                    global_source,
                    map_location="cpu",
                    weights_only=False,
                )
                if isinstance(global_checkpoint_meta, dict):
                    fold_best_epoch_global_rmse = int(
                        global_checkpoint_meta.get("epoch", -1)
                    ) + 1
                del global_checkpoint_meta

            copied = fold_model_path.exists()
            print(
                f"  💾 [FOLD {fold_idx}] 正式canonical模型: "
                f"{fold_model_path}"
            )
            print(
                f"     admissible_improvement_found="
                f"{fold_admissible_improvement_found}, "
                f"selected_is_frozen_fallback="
                f"{fold_selected_is_frozen_fallback}"
            )
            print(
                f"  💾 [FOLD {fold_idx}] Frozen基线审计副本: "
                f"{fold_frozen_path}"
            )
            print(
                f"  💾 [FOLD {fold_idx}] 真实最后训练态: "
                f"{fold_last_trained_path}"
            )
            if global_source.exists():
                print(
                    f"  🔎 [FOLD {fold_idx}] 无约束最低RMSE诊断模型: "
                    f"{fold_global_path}"
                )

            if copied:
                fold_model_paths[fold_idx] = str(fold_model_path)

                # PROGRESSIVE_CV_RELOAD_BEST_V1
                # 指标必须由该折“最佳checkpoint”计算，而不是最后epoch模型。
                try:
                    best_checkpoint = torch.load(
                        fold_model_path,
                        map_location=self.device,
                        weights_only=False,
                    )

                    if isinstance(best_checkpoint, dict):
                        best_state_dict = None
                        for state_key in (
                            "model_state_dict",
                            "state_dict",
                            "model",
                        ):
                            if state_key in best_checkpoint:
                                best_state_dict = best_checkpoint[state_key]
                                break
                    else:
                        best_state_dict = best_checkpoint

                    if best_state_dict is None:
                        raise RuntimeError(
                            f"最佳checkpoint中没有模型权重: {fold_model_path}"
                        )

                    self.model.load_state_dict(
                        best_state_dict,
                        strict=True,
                    )
                    self.model.to(self.device)
                    if isinstance(best_checkpoint, dict):
                        fold_best_epoch_admissible = int(
                            best_checkpoint.get("epoch", -1)
                        ) + 1
                    print(
                        f"  ✅ [FOLD {fold_idx}] 已重新载入该折最佳模型，"
                        "后续外层OOF指标基于canonical checkpoint"
                    )
                    print(
                        f"     epochs_ran={fold_epochs_ran}, "
                        f"best_admissible_epoch={fold_best_epoch_admissible}, "
                        f"frozen_fallback={fold_selected_is_frozen_fallback}, "
                        f"max_epochs={fold_max_epochs}"
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Fold {fold_idx} 最佳模型重新载入失败: {exc}"
                    ) from exc

            if inner_pilot_only:
                def _pilot_json_safe(value):
                    if isinstance(value, np.generic):
                        return value.item()
                    if torch.is_tensor(value):
                        if value.numel() == 1:
                            return value.detach().cpu().item()
                        return value.detach().cpu().tolist()
                    if isinstance(value, dict):
                        return {
                            str(key): _pilot_json_safe(item)
                            for key, item in value.items()
                        }
                    if isinstance(value, (list, tuple)):
                        return [
                            _pilot_json_safe(item)
                            for item in value
                        ]
                    if isinstance(value, float) and not np.isfinite(value):
                        return None
                    if isinstance(
                        value,
                        (str, int, float, bool),
                    ) or value is None:
                        return value
                    return str(value)

                def _checkpoint_metrics_cpu(path):
                    payload = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    if not isinstance(payload, dict):
                        return {}, None
                    return (
                        payload.get("metrics", {}) or {},
                        int(payload.get("epoch", -1)) + 1,
                    )

                selected_metrics, selected_epoch = (
                    _checkpoint_metrics_cpu(fold_model_path)
                )
                frozen_metrics, frozen_epoch = (
                    _checkpoint_metrics_cpu(fold_frozen_path)
                )

                pilot_metric_keys = [
                    "loss",
                    "rmse_mm",
                    "mae_mm",
                    "correlation",
                    "ccc",
                    "r2",
                    "bias_mm",
                    "slope",
                    "std_ratio",
                    "pred_std_mm",
                    "target_std_mm",
                    "rmse_ge50_mm",
                    "bias_ge80_mm",
                    "selection_score",
                    "checkpoint_admissible",
                    "checkpoint_control_score",
                ]
                epoch_rows = []
                for epoch_number, metrics in enumerate(
                    getattr(self, "val_history_metrics", []),
                    start=1,
                ):
                    row = {"epoch": int(epoch_number)}
                    for key in pilot_metric_keys:
                        row[key] = _pilot_json_safe(
                            metrics.get(key)
                        )
                    row["checkpoint_violations"] = json.dumps(
                        _pilot_json_safe(
                            metrics.get(
                                "checkpoint_violations",
                                [],
                            )
                        ),
                        ensure_ascii=False,
                    )
                    epoch_rows.append(row)

                pilot_epoch_path = (
                    self.save_dir
                    / "inner_pilot_epoch_metrics.csv"
                )
                pd.DataFrame(epoch_rows).to_csv(
                    pilot_epoch_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                pilot_summary = {
                    "mode": "inner_pilot_only",
                    "formal_oof_result": False,
                    "outer_evaluated": False,
                    "outer_dataset_loaded": False,
                    "split_protocol": "nested_8_train_1_inner_1_outer_reserved",
                    "fold": int(fold_idx),
                    "selection_fold": int(
                        split.get("selection_fold")
                    ),
                    "n_train_samples": int(
                        len(split["train_station_indices"])
                    ),
                    "n_inner_samples": int(
                        len(selection_val_indices)
                    ),
                    "n_reserved_outer_samples": int(
                        len(outer_test_indices)
                    ),
                    "max_epochs": int(fold_max_epochs),
                    "epochs_ran": int(fold_epochs_ran),
                    "selected_epoch": int(
                        selected_epoch
                        if selected_epoch is not None
                        else fold_best_epoch_admissible
                    ),
                    "admissible_improvement_found": bool(
                        fold_admissible_improvement_found
                    ),
                    "selected_is_frozen_fallback": bool(
                        fold_selected_is_frozen_fallback
                    ),
                    "checkpoint_selection": (
                        "Frozen-relative admissibility gate, then "
                        "minimum composite selection_score"
                    ),
                    "optimization": {
                        "head_lr": float(
                            self.config.get(
                                "fusion_ft_head_lr",
                                5e-5,
                            )
                        ),
                        "transformer_peak_lr": float(
                            self.config.get(
                                "fusion_ft_transformer_lr",
                                3e-5,
                            )
                        ),
                        "warmup_epochs": int(
                            self.config.get(
                                "fusion_ft_warmup_epochs",
                                10,
                            )
                        ),
                        "unfreeze_interval": int(
                            self.config.get(
                                "fusion_ft_unfreeze_interval",
                                1,
                            )
                        ),
                        "max_unfrozen_transformer_layers": int(
                            self.config.get(
                                "fusion_ft_max_unfrozen_layers",
                                0,
                            )
                        ),
                        "plateau_patience": int(
                            self.config.get(
                                "fusion_ft_plateau_patience",
                                8,
                            )
                        ),
                        "early_stop_min_epochs": int(
                            self.config.get(
                                "fusion_ft_min_epochs_before_early_stop",
                                40,
                            )
                        ),
                        "early_stop_patience": int(
                            self.config.get(
                                "fusion_ft_patience",
                                20,
                            )
                        ),
                    },
                    "frozen_epoch": int(
                        frozen_epoch
                        if frozen_epoch is not None
                        else 0
                    ),
                    "frozen_inner_metrics": _pilot_json_safe(
                        frozen_metrics
                    ),
                    "selected_inner_metrics": _pilot_json_safe(
                        selected_metrics
                    ),
                    "selected_checkpoint": str(fold_model_path),
                    "frozen_checkpoint": str(fold_frozen_path),
                    "epoch_metrics_csv": str(pilot_epoch_path),
                }
                pilot_summary_path = (
                    self.save_dir
                    / "inner_pilot_summary.json"
                )
                with open(
                    pilot_summary_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        pilot_summary,
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )

                print("\n" + "=" * 70)
                print("🧪 单Inner pilot完成；Outer保持未评估")
                print(
                    f"  Train/Inner/Reserved Outer样本: "
                    f"{pilot_summary['n_train_samples']}/"
                    f"{pilot_summary['n_inner_samples']}/"
                    f"{pilot_summary['n_reserved_outer_samples']}"
                )
                print(
                    f"  epochs_ran={fold_epochs_ran}, "
                    f"selected_epoch={pilot_summary['selected_epoch']}, "
                    "admissible_improvement_found="
                    f"{fold_admissible_improvement_found}"
                )
                print(f"  汇总: {pilot_summary_path}")
                print(f"  Inner逐轮指标: {pilot_epoch_path}")
                print(
                    "  🔒 本次没有创建Outer DataLoader，"
                    "没有Outer预测或OOF结果"
                )
                print("=" * 70)
                return pilot_summary

            # ============ 8.8 外层OOF预测 ============
            # outer_oof_loader从未参与训练、scheduler、early stopping或选模。
            print(f"\n📊 [FOLD {fold_idx}] 收集外层OOF预测...")

            self.model.eval()

            preds, targets, is_zero = self._make_predictions(
                outer_oof_loader
            )

            if preds is not None and len(preds) > 0:
                s_min = getattr(self, "swe_min", 0.0)
                s_max = getattr(self, "swe_max", 200.0)

                preds_denorm = preds * (s_max - s_min) + s_min
                targets_denorm = targets * (s_max - s_min) + s_min

                preds_denorm = np.asarray(preds_denorm).reshape(-1)
                targets_denorm = np.asarray(targets_denorm).reshape(-1)
                val_dataset_indices = np.asarray(
                    outer_test_indices,
                    dtype=np.int64,
                )

                if len(preds_denorm) != len(val_dataset_indices):
                    raise RuntimeError(
                        f"Fold {fold_idx}外层OOF预测数量不一致: "
                        f"pred={len(preds_denorm)}, "
                        f"indices={len(val_dataset_indices)}"
                    )

                valid = np.isfinite(preds_denorm) & np.isfinite(targets_denorm)
                preds_denorm = preds_denorm[valid]
                targets_denorm = targets_denorm[valid]
                valid_dataset_indices = val_dataset_indices[valid]

                all_val_predictions.append(preds_denorm)
                all_val_targets.append(targets_denorm)

                for dataset_idx, target_mm, prediction_mm in zip(
                    valid_dataset_indices.tolist(),
                    targets_denorm.tolist(),
                    preds_denorm.tolist(),
                ):
                    meta = station_ds.meta_index[int(dataset_idx)]
                    station_id = str(meta.get("station_id", "unknown")).split(",")[0]
                    label_date = pd.to_datetime(
                        meta.get(
                            "label_date",
                            meta.get("feature_date", meta.get("date")),
                        )
                    ).strftime("%Y-%m-%d")
                    feature_date = pd.to_datetime(
                        meta.get(
                            "feature_date",
                            meta.get("label_date", meta.get("date")),
                        )
                    ).strftime("%Y-%m-%d")
                    all_oof_rows.append({
                        "fold": int(fold_idx),
                        "dataset_index": int(dataset_idx),
                        "station_id": station_id,
                        "label_date": label_date,
                        "feature_date": feature_date,
                        "day_gap": int(meta.get("day_gap", 0)),
                        "row": int(meta["row"]),
                        "col": int(meta["col"]),
                        "target_mm": float(target_mm),
                        "prediction_mm": float(prediction_mm),
                        "freeze_strategy": str(freeze_strategy),
                        "model_path": fold_model_paths.get(fold_idx),
                        "selection_fold": split.get("selection_fold"),
                        "selected_is_frozen_fallback": bool(
                            fold_selected_is_frozen_fallback
                        ),
                    })

                rmse = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))
                mae = np.mean(np.abs(preds_denorm - targets_denorm))
                bias = np.mean(preds_denorm - targets_denorm)

                ss_res = np.sum((preds_denorm - targets_denorm) ** 2)
                ss_tot = np.sum((targets_denorm - np.mean(targets_denorm)) ** 2)
                nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                if len(preds_denorm) > 1 and np.std(preds_denorm) > 0 and np.std(targets_denorm) > 0:
                    r, _ = stats.pearsonr(preds_denorm, targets_denorm)
                else:
                    r = np.nan

                # 高 SWE 子集
                high_mask = targets_denorm >= 80.0
                high_n = int(high_mask.sum())

                if high_n > 0:
                    high_mae = np.mean(np.abs(preds_denorm[high_mask] - targets_denorm[high_mask]))
                    high_rmse = np.sqrt(np.mean((preds_denorm[high_mask] - targets_denorm[high_mask]) ** 2))
                    high_bias = np.mean(preds_denorm[high_mask] - targets_denorm[high_mask])
                    high_pred_mean = np.mean(preds_denorm[high_mask])
                    high_obs_mean = np.mean(targets_denorm[high_mask])
                else:
                    high_mae = np.nan
                    high_rmse = np.nan
                    high_bias = np.nan
                    high_pred_mean = np.nan
                    high_obs_mean = np.nan

                fold_metrics = {
                    "fold": fold_idx,
                    "nse": float(nse) if np.isfinite(nse) else None,
                    "r2": float(nse) if np.isfinite(nse) else None,
                    "r": float(r) if np.isfinite(r) else None,
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "bias": float(bias),
                    "n_samples": int(len(preds_denorm)),
                    "max_epochs": int(fold_max_epochs),
                    "epochs_ran": int(fold_epochs_ran),
                    "best_epoch_global_rmse": (
                        int(fold_best_epoch_global_rmse)
                        if fold_best_epoch_global_rmse is not None
                        else None
                    ),
                    "best_epoch_admissible": int(
                        fold_best_epoch_admissible
                    ),
                    "admissible_improvement_found": bool(
                        fold_admissible_improvement_found
                    ),
                    "selected_is_frozen_fallback": bool(
                        fold_selected_is_frozen_fallback
                    ),
                    "selection_fold": split.get("selection_fold"),
                    "stopped_early": bool(fold_epochs_ran < fold_max_epochs),
                    "hit_epoch_cap": bool(fold_epochs_ran >= fold_max_epochs),
                    "best_epoch_near_cap": bool(
                        fold_best_epoch_admissible is not None
                        and fold_best_epoch_admissible
                        >= int(np.ceil(0.90 * fold_max_epochs))
                    ),
                    "high_swe": {
                        "threshold": 80.0,
                        "n": high_n,
                        "mae": float(high_mae) if np.isfinite(high_mae) else None,
                        "rmse": float(high_rmse) if np.isfinite(high_rmse) else None,
                        "bias": float(high_bias) if np.isfinite(high_bias) else None,
                        "obs_mean": float(high_obs_mean) if np.isfinite(high_obs_mean) else None,
                        "pred_mean": float(high_pred_mean) if np.isfinite(high_pred_mean) else None,
                    },
                    "model_path": fold_model_paths.get(fold_idx, None),
                }

                all_fold_metrics.append(fold_metrics)

                print(f"\n  📈 [FOLD {fold_idx}] 外层OOF评估:")
                print(f"    NSE:  {nse:.4f}")
                print(f"    R:    {r:.4f}")
                print(f"    RMSE: {rmse:.2f} mm")
                print(f"    MAE:  {mae:.2f} mm")
                print(f"    Bias: {bias:.2f} mm")

                if high_n > 0:
                    print(
                        f"    obs>=80: N={high_n}, "
                        f"MAE={high_mae:.2f}, "
                        f"Bias={high_bias:.2f}, "
                        f"obs_mean={high_obs_mean:.2f}, "
                        f"pred_mean={high_pred_mean:.2f}"
                    )

                self.plot_density_scatter_hardcode(
                    preds_denorm,
                    targets_denorm,
                    is_fine_tune=True,
                    fold_index=fold_idx,
                )

            else:
                print(f"  ⚠ [FOLD {fold_idx}] 外层OOF预测失败")

            # ============ 8.9 清理显存 ============
            print(f"\n🧹 [FOLD {fold_idx}] 清理显存...")

            try:
                del self.model
                del self.optimizer
                del self.scheduler
            except Exception:
                pass

            self.model = None
            self.optimizer = None
            self.scheduler = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        # ============================================================
        # 9. 聚合验证结果
        # ============================================================
        print(f"\n\n{'█' * 80}")
        print("🏆 十折交叉验证完成，计算聚合验证精度")
        print(f"{'█' * 80}")

        if len(all_val_predictions) == 0:
            print("❌ 没有收集到任何验证结果")
            return None

        agg_predictions = np.concatenate([p.reshape(-1) for p in all_val_predictions])
        agg_targets = np.concatenate([t.reshape(-1) for t in all_val_targets])

        valid = np.isfinite(agg_predictions) & np.isfinite(agg_targets)
        agg_predictions = agg_predictions[valid]
        agg_targets = agg_targets[valid]

        agg_rmse = np.sqrt(np.mean((agg_predictions - agg_targets) ** 2))
        agg_mae = np.mean(np.abs(agg_predictions - agg_targets))
        agg_bias = np.mean(agg_predictions - agg_targets)

        ss_res = np.sum((agg_predictions - agg_targets) ** 2)
        ss_tot = np.sum((agg_targets - np.mean(agg_targets)) ** 2)
        agg_nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        if len(agg_predictions) > 1 and np.std(agg_predictions) > 0 and np.std(agg_targets) > 0:
            agg_r, _ = stats.pearsonr(agg_predictions, agg_targets)
        else:
            agg_r = np.nan

        high_mask = agg_targets >= 80.0
        high_n = int(high_mask.sum())

        if high_n > 0:
            agg_high_mae = np.mean(np.abs(agg_predictions[high_mask] - agg_targets[high_mask]))
            agg_high_rmse = np.sqrt(np.mean((agg_predictions[high_mask] - agg_targets[high_mask]) ** 2))
            agg_high_bias = np.mean(agg_predictions[high_mask] - agg_targets[high_mask])
            agg_high_obs_mean = np.mean(agg_targets[high_mask])
            agg_high_pred_mean = np.mean(agg_predictions[high_mask])
        else:
            agg_high_mae = np.nan
            agg_high_rmse = np.nan
            agg_high_bias = np.nan
            agg_high_obs_mean = np.nan
            agg_high_pred_mean = np.nan

        print(f"\n{'=' * 60}")
        print(f"🎯 【聚合验证精度】基于 {len(agg_predictions)} 个验证样本")
        print(f"{'=' * 60}")
        print(f"  NSE:   {agg_nse:.4f}")
        print(f"  R:     {agg_r:.4f}")
        print(f"  RMSE:  {agg_rmse:.2f} mm")
        print(f"  MAE:   {agg_mae:.2f} mm")
        print(f"  Bias:  {agg_bias:.2f} mm")

        if high_n > 0:
            print(f"\n  高 SWE 子集 obs>=80 mm:")
            print(f"    N:         {high_n}")
            print(f"    MAE:       {agg_high_mae:.2f} mm")
            print(f"    RMSE:      {agg_high_rmse:.2f} mm")
            print(f"    Bias:      {agg_high_bias:.2f} mm")
            print(f"    obs mean:  {agg_high_obs_mean:.2f} mm")
            print(f"    pred mean: {agg_high_pred_mean:.2f} mm")

        print(f"{'=' * 60}")

        # 每折统计
        if all_fold_metrics:
            nse_values = [
                m["nse"] for m in all_fold_metrics
                if m.get("nse") is not None
            ]
            rmse_values = [
                m["rmse"] for m in all_fold_metrics
                if m.get("rmse") is not None
            ]
            mae_values = [
                m["mae"] for m in all_fold_metrics
                if m.get("mae") is not None
            ]

            print("\n📊 十折指标统计:")
            if nse_values:
                print(f"  NSE:  {np.mean(nse_values):.4f} ± {np.std(nse_values):.4f}")
            if rmse_values:
                print(f"  RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f} mm")
            if mae_values:
                print(f"  MAE:  {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f} mm")

        # 聚合散点图
        print("\n📊 绘制聚合验证散点图...")
        self.plot_density_scatter_hardcode(
            agg_predictions,
            agg_targets,
            is_fine_tune=True,
            fold_index="aggregated_10fold",
        )

        # 十折箱线图
        if all_fold_metrics:
            self.plot_cv_metrics_boxplot(
                all_fold_metrics,
                save_name="cv_station_level_boxplot.png"
            )

        # ============================================================
        # 10. 保存结果
        # ============================================================
        # BALANCED_STATION_CV10_OOF_V1
        oof_frame = pd.DataFrame(all_oof_rows)
        expected_oof_indices = set(int(i) for i in cv_pool_indices)

        if len(oof_frame) != len(expected_oof_indices):
            raise RuntimeError(
                "OOF结果没有完整覆盖内部CV池: "
                f"actual={len(oof_frame)}, expected={len(expected_oof_indices)}"
            )
        if oof_frame["dataset_index"].duplicated().any():
            raise RuntimeError("OOF结果存在重复dataset_index")
        if set(oof_frame["dataset_index"].astype(int)) != expected_oof_indices:
            raise RuntimeError("OOF结果的dataset_index集合与CV池不一致")
        if (oof_frame["feature_date"] != oof_frame["label_date"]).any():
            raise RuntimeError("OOF结果仍存在feature_date != label_date")
        if (oof_frame["day_gap"] != 0).any():
            raise RuntimeError("OOF结果仍存在day_gap != 0")

        oof_frame = oof_frame.sort_values(
            ["fold", "station_id", "label_date"]
        ).reset_index(drop=True)
        oof_path = self.save_dir / "station_cv10_oof_predictions.csv"
        oof_frame.to_csv(oof_path, index=False, encoding="utf-8-sig")

        fold_metrics_path = self.save_dir / "station_cv10_fold_metrics.csv"
        pd.DataFrame(all_fold_metrics).to_csv(
            fold_metrics_path, index=False, encoding="utf-8-sig"
        )

        print(f"\n💾 OOF预测已保存: {oof_path}")
        print(f"💾 十折指标已保存: {fold_metrics_path}")

        self.plot_cv10_fold_scatter_outputs(
            oof_frame,
            panel_name="cv_station_level_scatter_panel.png",
        )

        epoch_rows = []
        for item in all_fold_metrics:
            epoch_rows.append({
                "fold": item.get("fold"),
                "max_epochs": item.get("max_epochs"),
                "epochs_ran": item.get("epochs_ran"),
                "best_epoch_global_rmse": item.get("best_epoch_global_rmse"),
                "best_epoch_admissible": item.get("best_epoch_admissible"),
                "selected_is_frozen_fallback": item.get(
                    "selected_is_frozen_fallback"
                ),
                "stopped_early": item.get("stopped_early"),
                "hit_epoch_cap": item.get("hit_epoch_cap"),
                "best_epoch_near_cap": item.get("best_epoch_near_cap"),
            })
        epoch_frame = pd.DataFrame(epoch_rows)
        epoch_path = self.save_dir / "station_cv10_epoch_diagnostics.csv"
        epoch_frame.to_csv(epoch_path, index=False, encoding="utf-8-sig")
        self.plot_cv_epoch_diagnostics(epoch_frame)
        print(f"💾 Epoch诊断已保存: {epoch_path}")

        agg_results = {
            "cv_mode": "balanced_station_level_cv",
            "split_method": "deterministic_balanced_greedy_v1",
            "randomized": False,
            "checkpoint_selection_protocol": (
                "nested rolling inner-fold validation with "
                "Frozen-relative hard non-degradation gate, then "
                "minimum composite selection_score"
            ),
            "outer_oof_used_for_checkpoint_selection": False,
            "fold_manifest": str(fold_manifest_path),
            "mixed_mode": bool(is_mixed),
            "n_folds": int(n_splits),
            "pretrain_loss_weight": self.config.get("pretrain_loss_weight", None),
            "station_ratio": self.config.get("station_ratio", None),
            "optimization_policy": {
                "fusion_ft_progressive_unfreeze": bool(
                    self.config.get(
                        "fusion_ft_progressive_unfreeze",
                        True,
                    )
                ),
                "fusion_ft_head_lr": float(
                    self.config.get("fusion_ft_head_lr", 5e-5)
                ),
                "fusion_ft_transformer_lr": float(
                    self.config.get(
                        "fusion_ft_transformer_lr",
                        3e-5,
                    )
                ),
                "fusion_ft_warmup_epochs": int(
                    self.config.get("fusion_ft_warmup_epochs", 10)
                ),
                "fusion_ft_unfreeze_interval": int(
                    self.config.get(
                        "fusion_ft_unfreeze_interval",
                        1,
                    )
                ),
                "fusion_ft_max_unfrozen_layers": int(
                    self.config.get(
                        "fusion_ft_max_unfrozen_layers",
                        0,
                    )
                ),
                "fusion_ft_unfreeze_order": (
                    "head_only_then_transformer_top_to_bottom"
                ),
                "fusion_ft_sampling": (
                    "natural_random_without_forced_tail_quota"
                    if not bool(
                        self.config.get(
                            "fusion_ft_use_stratified_batches",
                            False,
                        )
                    )
                    else "stratified"
                ),
                "fusion_ft_tail_weights": {
                    "ge20": float(
                        self.config.get(
                            "fusion_ft_swe_weight_ge20",
                            1.2,
                        )
                    ),
                    "ge50": float(
                        self.config.get(
                            "fusion_ft_swe_weight_ge50",
                            1.5,
                        )
                    ),
                    "ge80": float(
                        self.config.get(
                            "fusion_ft_swe_weight_ge80",
                            2.0,
                        )
                    ),
                },
                "fusion_ft_high_bias_weight": float(
                    self.config.get(
                        "fusion_ft_high_bias_weight",
                        0.0,
                    )
                ),
                "fusion_ft_ccc_weight": float(
                    self.config.get(
                        "fusion_ft_ccc_weight",
                        0.005,
                    )
                ),
                "fusion_ft_variance_weight": float(
                    self.config.get(
                        "fusion_ft_variance_weight",
                        0.0,
                    )
                ),
                "fusion_ft_patience": int(
                    self.config.get("fusion_ft_patience", 20)
                ),
                "fusion_ft_plateau_patience": int(
                    self.config.get(
                        "fusion_ft_plateau_patience",
                        8,
                    )
                ),
                "fusion_ft_min_epochs_before_early_stop": int(
                    self.config.get(
                        "fusion_ft_min_epochs_before_early_stop",
                        40,
                    )
                ),
                "admissible_checkpoint_ranking": (
                    "minimum_composite_selection_score"
                ),
            },
            "aggregated_metrics": {
                "nse": float(agg_nse) if np.isfinite(agg_nse) else None,
                "r2": float(agg_nse) if np.isfinite(agg_nse) else None,
                "r": float(agg_r) if np.isfinite(agg_r) else None,
                "rmse": float(agg_rmse),
                "mae": float(agg_mae),
                "bias": float(agg_bias),
                "n_samples": int(len(agg_predictions)),
                "high_swe": {
                    "threshold": 80.0,
                    "n": high_n,
                    "mae": float(agg_high_mae) if np.isfinite(agg_high_mae) else None,
                    "rmse": float(agg_high_rmse) if np.isfinite(agg_high_rmse) else None,
                    "bias": float(agg_high_bias) if np.isfinite(agg_high_bias) else None,
                    "obs_mean": float(agg_high_obs_mean) if np.isfinite(agg_high_obs_mean) else None,
                    "pred_mean": float(agg_high_pred_mean) if np.isfinite(agg_high_pred_mean) else None,
                },
            },
            "fold_metrics": all_fold_metrics,
            "fold_model_paths": fold_model_paths,
            "metric_definition": {
                "primary_overall": "pooled OOF metrics after concatenating ten held-out folds",
                "fold_distribution": "ten fold-specific values summarized by mean/std and boxplot",
                "note": "R and NSE are nonlinear; pooled values are not simple means of fold values",
            },
            "epoch_diagnostics": {
                "max_epochs": int(epoch_frame["max_epochs"].max()) if not epoch_frame.empty else None,
                "median_best_epoch_global_rmse": (
                    float(pd.to_numeric(epoch_frame["best_epoch_global_rmse"], errors="coerce").median())
                    if not epoch_frame.empty else None
                ),
                "median_best_epoch_admissible": (
                    float(
                        pd.to_numeric(
                            epoch_frame["best_epoch_admissible"],
                            errors="coerce",
                        ).median()
                    )
                    if not epoch_frame.empty else None
                ),
                "frozen_fallback_fold_count": (
                    int(
                        epoch_frame["selected_is_frozen_fallback"]
                        .fillna(False)
                        .sum()
                    )
                    if not epoch_frame.empty else 0
                ),
                "near_cap_fold_count": (
                    int(epoch_frame["best_epoch_near_cap"].fillna(False).sum())
                    if not epoch_frame.empty else 0
                ),
                "hit_cap_fold_count": (
                    int(epoch_frame["hit_epoch_cap"].fillna(False).sum())
                    if not epoch_frame.empty else 0
                ),
                "decision_rule": (
                    "The configured epoch count is a maximum cap, not proven optimal. "
                    "Increase it when at least three folds select a best epoch in the final 10% of the cap."
                ),
            },
        }

        if all_fold_metrics:
            nse_values = [
                m["nse"] for m in all_fold_metrics
                if m.get("nse") is not None
            ]
            rmse_values = [
                m["rmse"] for m in all_fold_metrics
                if m.get("rmse") is not None
            ]
            mae_values = [
                m["mae"] for m in all_fold_metrics
                if m.get("mae") is not None
            ]

            agg_results["fold_stats"] = {
                "nse_mean": float(np.mean(nse_values)) if nse_values else None,
                "nse_std": float(np.std(nse_values)) if nse_values else None,
                "rmse_mean": float(np.mean(rmse_values)) if rmse_values else None,
                "rmse_std": float(np.std(rmse_values)) if rmse_values else None,
                "mae_mean": float(np.mean(mae_values)) if mae_values else None,
                "mae_std": float(np.std(mae_values)) if mae_values else None,
            }

        if not epoch_frame.empty:
            near_cap_count = int(epoch_frame["best_epoch_near_cap"].fillna(False).sum())
            if near_cap_count >= 3:
                print(
                    f"\n⚠ Epoch上限可能不足：{near_cap_count}/10折的最低RMSE最佳轮次"
                    "落在当前上限最后10%内。建议该策略把FINE_TUNE_EPOCHS提高到100后复核。"
                )
            else:
                print(
                    f"\n✅ Epoch上限诊断：仅{near_cap_count}/10折的最佳轮次"
                    "落在当前上限最后10%内，暂未显示明显截断。"
                )

        agg_path = self.save_dir / "cv_station_level_aggregated_results.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 聚合结果已保存: {agg_path}")

        # ============================================================
        # 11. 正式内部评估策略
        # ============================================================
        # 每个fold使用独立inner fold选择checkpoint，再对从未参与选模的
        # outer fold评估。十折合并后，内部CV池每条样本恰好形成一次OOF预测。
        # 旧固定1000条已并回内部开发池，不再闲置为未使用holdout。
        agg_results["evaluation_policy"] = {
            "internal": "nested_inner_selection_outer_heldout_oof",
            "internal_cv_samples_evaluated_once": int(len(oof_frame)),
            "outer_oof_used_for_checkpoint_selection": False,
            "checkpoint_gate": "frozen_relative_non_degradation",
            "admissible_checkpoint_ranking": (
                "minimum_composite_selection_score"
            ),
            "fixed_internal_1000_merged_into_cv": bool(
                self.config.get(
                    "include_fixed_internal_in_cv",
                    False,
                )
            ),
            "external": "one_predeclared_cv10_ensemble_result_after_cv",
            "fold_selection_by_test_performance": False,
        }

        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print("📋 正式内部评估：inner fold选模，outer fold只做OOF")
        print("   10折OOF恰好覆盖内部CV池一次")
        if self.config.get("include_fixed_internal_in_cv", False):
            print("   内部7936条全部参与；旧固定1000条已并回")
        else:
            print("   旧固定1000条保持在CV池之外")
        print(f"{'=' * 60}")

        return agg_results
    
    def _draw_cv_fold_scatter_axis(
            self, ax, targets, predictions, fold_metrics, title, upper):
        targets = np.asarray(targets, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        valid = np.isfinite(targets) & np.isfinite(predictions)
        targets = targets[valid]
        predictions = predictions[valid]
        ax.scatter(targets, predictions, alpha=0.38, s=10)
        ax.plot([0, upper], [0, upper], '--', linewidth=1.2, label='1:1')
        if len(targets) >= 2 and np.std(targets) > 0:
            slope, intercept = np.polyfit(targets, predictions, 1)
            xx = np.array([0.0, upper])
            ax.plot(xx, intercept + slope * xx, color='red', linewidth=1.4, label='Fit')
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(alpha=0.22)
        ax.text(
            0.03, 0.97,
            (
                f"N={fold_metrics['n']}\n"
                f"R={fold_metrics['r']:.3f}\n"
                f"RMSE={fold_metrics['rmse']:.2f}\n"
                f"MAE={fold_metrics['mae']:.2f}\n"
                f"Bias={fold_metrics['bias']:.2f}"
            ),
            transform=ax.transAxes, ha='left', va='top', fontsize=8.2,
        )

    def plot_cv10_fold_scatter_outputs(
            self, oof_frame, panel_name="cv_station_level_scatter_panel.png"):
        required = {"fold", "target_mm", "prediction_mm"}
        if not required.issubset(set(oof_frame.columns)):
            print(f"⚠ 无法绘制十折散点组图，缺少列: {required - set(oof_frame.columns)}")
            return
        fold_dir = self.save_dir / "cv10_fold_scatter"
        fold_dir.mkdir(parents=True, exist_ok=True)
        target_all = pd.to_numeric(oof_frame["target_mm"], errors="coerce").to_numpy()
        pred_all = pd.to_numeric(oof_frame["prediction_mm"], errors="coerce").to_numpy()
        finite = np.isfinite(target_all) & np.isfinite(pred_all)
        upper = max(200.0, float(np.nanmax(np.r_[target_all[finite], pred_all[finite]])))
        upper = float(np.ceil(upper / 20.0) * 20.0)
        panel, axes = plt.subplots(2, 5, figsize=(19, 7.6), sharex=True, sharey=True)
        axes = axes.ravel()
        for fold in range(1, 11):
            frame = oof_frame[oof_frame["fold"] == fold]
            targets = pd.to_numeric(frame["target_mm"], errors="coerce").to_numpy()
            predictions = pd.to_numeric(frame["prediction_mm"], errors="coerce").to_numpy()
            valid = np.isfinite(targets) & np.isfinite(predictions)
            targets = targets[valid]
            predictions = predictions[valid]
            error = predictions - targets
            r = float(np.corrcoef(targets, predictions)[0, 1]) if len(targets) > 1 and np.std(targets) > 0 and np.std(predictions) > 0 else float('nan')
            metrics = {
                "n": int(len(targets)),
                "r": r,
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
            self._draw_cv_fold_scatter_axis(axes[fold - 1], targets, predictions, metrics, f"Fold {fold}", upper)
            fig, ax = plt.subplots(figsize=(6.4, 5.6))
            self._draw_cv_fold_scatter_axis(ax, targets, predictions, metrics, f"Held-out station Fold {fold}", upper)
            ax.set_xlabel("Station SWE (mm)")
            ax.set_ylabel("Model SWE (mm)")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            fig.savefig(fold_dir / f"fold_{fold:02d}_scatter.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        for row in range(2):
            axes[row * 5].set_ylabel("Model SWE (mm)")
        for col in range(5):
            axes[5 + col].set_xlabel("Station SWE (mm)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            panel.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        panel.suptitle("Balanced station-wise 10-fold held-out scatter", fontsize=15, fontweight="bold")
        panel.tight_layout(rect=(0, 0.04, 1, 0.95))
        panel.savefig(self.save_dir / panel_name, dpi=300, bbox_inches="tight")
        plt.close(panel)
        print(f"📊 十折散点组图已保存: {self.save_dir / panel_name}")

    def plot_cv_epoch_diagnostics(self, epoch_frame):
        if epoch_frame is None or epoch_frame.empty:
            return
        frame = epoch_frame.copy().sort_values("fold")
        x = np.arange(1, len(frame) + 1)
        best_global = pd.to_numeric(frame["best_epoch_global_rmse"], errors="coerce").to_numpy(dtype=float)
        best_admissible = pd.to_numeric(
            frame["best_epoch_admissible"],
            errors="coerce",
        ).to_numpy(dtype=float)
        epochs_ran = pd.to_numeric(frame["epochs_ran"], errors="coerce").to_numpy(dtype=float)
        max_epochs = float(pd.to_numeric(frame["max_epochs"], errors="coerce").max())
        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        ax.plot(
            x,
            best_global,
            marker="o",
            linewidth=1.5,
            label="Lowest global RMSE (diagnostic only)",
        )
        ax.plot(
            x,
            best_admissible,
            marker="s",
            linewidth=1.8,
            label="Canonical Frozen-relative epoch",
        )
        ax.plot(x, epochs_ran, marker="^", linestyle="--", linewidth=1.3, label="Epochs actually run")
        ax.axhline(max_epochs, linestyle=":", linewidth=1.5, label=f"Max epoch cap={int(max_epochs)}")
        ax.axhline(0.9 * max_epochs, linestyle="--", linewidth=1.0, label="90% of cap")
        ax.set_xticks(x)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Epoch")
        ax.set_title("Cross-validation epoch diagnostics")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        path = self.save_dir / "cv10_epoch_diagnostics.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"📈 Epoch诊断图已保存: {path}")

    def plot_cv_metrics_boxplot(self, all_fold_metrics, save_name="cv_metrics_boxplot_en.png"):
        """
        [Option B: English Version] 
        绘制十折指标箱线图 - 规避乱码并符合 SCI 论文标准
        all_fold_metrics: [{'r': 0.8, 'rmse': 5.2, 'mae': 3.1}, ...]
        """
        import seaborn as sns

        # 1. 准备数据
        df = pd.DataFrame(all_fold_metrics)
        # 展示层面只展示 r，不展示 NSE/R²
        plot_cols = ['r', 'rmse', 'mae']
        
        # 2. 设置绘图风格 (无需调用 setup_chinese_fonts)
        sns.set_theme(style="whitegrid")
        # 创建 1x3 的子图矩阵
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 定义学术化的英文标题和颜色
        titles = ['Pearson correlation (r)', 'RMSE (mm)', 'MAE (mm)']
        colors = ['#4E79A7', '#F28E2B', '#E15759']

        for i, col in enumerate(plot_cols):
            ax = axes[i]
            
            # 绘制箱线图：展示分布中位数、四分位数
            sns.boxplot(y=df[col], ax=ax, color=colors[i], width=0.4, 
                        linewidth=2, fliersize=5, whis=1.5)
            
            # 叠加 Stripplot：展示 10 个具体折数的原始散点，增加数据透明度
            sns.stripplot(y=df[col], ax=ax, color='black', alpha=0.5, size=6, jitter=0.05)
            
            # 设置子图标题
            ax.set_title(titles[i], fontsize=16, fontweight='bold')
            ax.set_ylabel('')
            
            # 计算本指标的统计值
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # 设置 X 轴标签为 Mean 和 Std (学术规范格式)
            ax.set_xlabel(f"Mean: {mean_val:.3f}\nStd Dev: {std_val:.3f}", fontsize=13)

        # 设置整张大图的总标题
        plt.suptitle("Stability Analysis of 10-Fold Cross-Validation Metrics", 
                     fontsize=20, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # 及时关闭窗口释放显存/内存
        
        print(f"✅ English Boxplot generated successfully: {save_path}")
    
    def plot_cv_panel_figure(self, all_fold_data, save_name="cv_10fold_panel_matrix.png"):
        """将十折结果绘制成 2x5 的论文级矩阵图"""

        self.setup_chinese_fonts()
        # 创建 2行 5列 的画布
        fig, axes = plt.subplots(2, 5, figsize=(25, 12), sharex=True, sharey=True)
        axes = axes.flatten()

        # 确定坐标轴量程统一
        all_vals = []
        for fold in all_fold_data:
            all_vals.extend(fold['preds'])
            all_vals.extend(fold['targets'])
        max_val = max(all_vals) * 1.05 if all_vals else 200

        for i, fold in enumerate(all_fold_data):
            ax = axes[i]
            preds = np.array(fold['preds'])
            targets = np.array(fold['targets'])

            # 绘制散点
            ax.scatter(targets, preds, alpha=0.5, s=20, c='#2c7bb6', edgecolors='none')

            # 1:1线
            ax.plot([0, max_val], [0, max_val], 'r--', lw=1.5, alpha=0.8)

            # 1. 新增：计算 MAE
            mae = np.mean(np.abs(preds - targets))

            # 2. 原有的 RMSE 和 R2 计算保持不变
            rmse = np.sqrt(np.mean((preds - targets)**2))
            mae = np.mean(np.abs(preds - targets))
            bias = np.mean(preds - targets)

            if len(targets) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
                r = np.corrcoef(preds, targets)[0, 1]
            else:
                r = np.nan

            metrics_box = (
                f"N: {len(targets)}\n"
                f"R: {r:.3f}\n"
                f"RMSE: {rmse:.2f}\n"
                f"MAE: {mae:.2f}\n"
                f"Bias: {bias:.2f}"
            )

            ax.text(
                0.05,
                0.95,
                metrics_box,
                transform=ax.transAxes,
                fontsize=15,
                fontweight='bold',
                linespacing=1.15,
                verticalalignment='top',
                bbox=dict(
                    boxstyle="round,pad=0.55",
                    facecolor=(1.0, 1.0, 1.0, 0.42),
                    edgecolor=(0.0, 0.0, 0.0, 0.70),
                    linewidth=1.2,
                ),
            )

            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.grid(True, linestyle=':', alpha=0.5)

        # 设置公共轴标签
        fig.text(0.5, 0.02, '真实观测值 (SWE mm)', ha='center', fontsize=22, fontweight='bold')
        fig.text(0.01, 0.5, '模型预测值 (SWE mm)', va='center', rotation='vertical', fontsize=22, fontweight='bold')

        plt.suptitle("十折交叉验证预测结果对比矩阵", fontsize=28, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 论文级矩阵汇总图已保存: {save_path}")
        
        
    def run_spatial_block_cv_experiment(self):
        """
        空间块十折交叉验证实验 - 二维网格划分
        规范划分：Train(80%) + Val(20%) + Test(10%)
        每个策略使用独立超参数（支持分层学习率）
        """
        print(f"\n{'█'*80}")
        print("🌟 空间块十折交叉验证实验（二维网格划分）")
        print("   规范划分：训练集(80%) + 验证集(20%) + 测试集(10%)")
        print("   每个策略使用独立超参数（支持分层学习率）")
        print(f"{'█'*80}")

        # ============ 1. 获取站点数据集 ============
        print(f"\n📊 获取站点数据集...")

        current_ds = self.train_loader.dataset
        while hasattr(current_ds, 'dataset') and not hasattr(current_ds, 'station_dataset'):
            current_ds = current_ds.dataset

        if hasattr(current_ds, 'station_dataset'):
            station_ds = current_ds.station_dataset
        else:
            station_ds = current_ds

        print(f"  站点数据集样本数: {len(station_ds)}")
        print(f"  图像尺寸: {station_ds.H}行 × {station_ds.W}列")

        # ============ 2. 二维网格划分 ============
        print(f"\n📊 按二维网格划分空间块...")

        N_BLOCKS = self.config.get('spatial_cv_blocks', 10)

        # ============ 修正：确保总块数等于 N_BLOCKS ============
        if 'spatial_cv_rows' in self.config and self.config['spatial_cv_rows'] is not None:
            N_ROWS = self.config['spatial_cv_rows']
            N_COLS = self.config['spatial_cv_cols']
            total_blocks = N_ROWS * N_COLS
            print(f"   使用配置网格: {N_ROWS}行 × {N_COLS}列 = {total_blocks}个空间块")
            if total_blocks != N_BLOCKS:
                print(f"   ⚠️ 警告: 配置块数({total_blocks}) != 折数({N_BLOCKS})，将使用配置的块数")
                N_BLOCKS = total_blocks
        else:
            import math
            N_ROWS = int(math.sqrt(N_BLOCKS))
            while N_BLOCKS % N_ROWS != 0:
                N_ROWS -= 1
            N_COLS = N_BLOCKS // N_ROWS
            total_blocks = N_BLOCKS
            print(f"   自动生成网格: {N_ROWS}行 × {N_COLS}列 = {total_blocks}个空间块")

        row_block_size = station_ds.H // N_ROWS
        col_block_size = station_ds.W // N_COLS

        print(f"   行块大小: {row_block_size} 像素")
        print(f"   列块大小: {col_block_size} 像素")

        sample_blocks = []
        block_info = {}

        for block_id in range(total_blocks):
            row_idx = block_id // N_COLS
            col_idx = block_id % N_COLS
            row_start = row_idx * row_block_size
            row_end = (row_idx + 1) * row_block_size if row_idx < N_ROWS - 1 else station_ds.H
            col_start = col_idx * col_block_size
            col_end = (col_idx + 1) * col_block_size if col_idx < N_COLS - 1 else station_ds.W
            block_info[block_id] = {
                'row_idx': row_idx,
                'col_idx': col_idx,
                'row_range': (row_start, row_end),
                'col_range': (col_start, col_end)
            }

        for idx in range(len(station_ds)):
            meta = station_ds.meta_index[idx]
            r, c = meta['row'], meta['col']
            row_block = min(r // row_block_size, N_ROWS - 1)
            col_block = min(c // col_block_size, N_COLS - 1)
            block_id = row_block * N_COLS + col_block
            sample_blocks.append(block_id)

        sample_blocks = np.array(sample_blocks)

        block_counts = defaultdict(int)
        for bid in sample_blocks:
            if bid < N_BLOCKS:
                block_counts[bid] += 1

        print(f"\n  空间块统计:")
        for bid in range(N_BLOCKS):
            info = block_info[bid]
            print(f"    Block {bid:2d} (行{info['row_idx']},列{info['col_idx']}): {block_counts[bid]} 个样本")
        print(f"    总样本: {len(station_ds)}")

        empty_blocks = [bid for bid in range(N_BLOCKS) if block_counts[bid] == 0]
        if empty_blocks:
            print(f"\n  ⚠️ 警告: 以下块没有样本: {empty_blocks}")
            print(f"   建议减少折数或调整网格大小")

        # ============ 3. 预训练模型路径 ============
        pretrained_path = self.config.get('pretrained_model')
        if pretrained_path is None or not os.path.exists(pretrained_path):
            possible_paths = [
                "/root/autodl-tmp/experiments/swe_full_temporal_20260609_085603/best_model.pth",
                self.save_dir / "best_model.pth",
                self.save_dir / "final_model.pth",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pretrained_path = str(path)
                    break

        if not pretrained_path or not os.path.exists(pretrained_path):
            print(f"\n  ❌ 错误: 预训练模型不存在")
            return None
        else:
            print(f"\n  ✅ 预训练模型: {pretrained_path}")

        # ============ 4. 定义策略（带独立超参数，支持分层学习率）============
        strategies = [
            {
                'name': 'fusion_ft',
                'freeze_backbone': True,
                'freeze_strategy': 'fusion_ft',
                'lr_config': {'lr_transformer': 5e-4, 'lr_head': 5e-4},  # 训练融合层+Head
                'weight_decay': 1e-4,
                'epochs': 30,
                'patience': 10,
                'description': '训练Fusion Layer（Transformer及其内含回归Head），冻结两个Encoder'
            },
            {
                'name': 'point_ft',
                'freeze_backbone': True,
                'freeze_strategy': 'point_ft',
                'lr_config': {'lr_encoder': 1e-4, 'lr_transformer': 1e-4, 'lr_head': 1e-4},
                'weight_decay': 1e-4,
                'epochs': 40,
                'patience': 10,
                'description': '训练Point Encoder + Fusion Layer'
            },
            {
                'name': 'spatial_ft',
                'freeze_backbone': True,
                'freeze_strategy': 'spatial_ft',
                'lr_config': {'lr_encoder': 1e-4, 'lr_transformer': 1e-4, 'lr_head': 1e-4},
                'weight_decay': 1e-4,
                'epochs': 40,
                'patience': 10,
                'description': '训练Spatial Encoder + Fusion Layer'
            },
            {
                'name': 'partial',
                'freeze_backbone': True,
                'freeze_strategy': 'partial',
                'lr_config': {'lr_encoder': 1e-6, 'lr_transformer': 4e-5, 'lr_head': 6e-4},  # 你原来的配置
                'weight_decay': 1e-4,
                'epochs': 45,
                'patience': 12,
                'description': '训练两个Encoder顶层 + Fusion Layer'
            },
            {
                'name': 'full_ft',
                'freeze_backbone': False,
                'freeze_strategy': 'none',
                'lr_config': {'lr_encoder': 3e-5, 'lr_transformer': 3e-5, 'lr_head': 5e-4},
                'weight_decay': 1e-5,
                'epochs': 50,
                'patience': 15,
                'description': '全部参数可训练，分层学习率'
            },
        ]

        print(f"\n📋 待测试策略 ({len(strategies)}种):")
        for s in strategies:
            lr_str = f"enc={s['lr_config'].get('lr_encoder', 'N/A'):.0e}, trans={s['lr_config'].get('lr_transformer', 'N/A'):.0e}, head={s['lr_config'].get('lr_head', 'N/A'):.0e}"
            print(f"    - {s['name']}: {s['description']} ({lr_str}, epochs={s['epochs']})")

        # ============ 5. 存储所有结果 ============
        all_fold_results = []
        strategy_results = {s['name']: [] for s in strategies}

        # ============ 6. 十折循环 ============
        for test_block in range(N_BLOCKS):
            print(f"\n\n{'█'*70}")
            print(f"🟢 FOLD {test_block + 1} / {N_BLOCKS}")
            info = block_info[test_block]
            print(f"   测试集: Block {test_block} (行{info['row_idx']},列{info['col_idx']})")
            print(f"         行范围: {info['row_range'][0]}-{info['row_range'][1]}")
            print(f"         列范围: {info['col_range'][0]}-{info['col_range'][1]}")
            print(f"         样本数: {block_counts[test_block]}")
            print(f"{'█'*70}")

            if block_counts[test_block] < 50:
                print(f"  ⚠️ 警告: 测试集样本数({block_counts[test_block]})少于50，评估结果可能不稳定")

            # 6.1 划分训练/测试索引
            all_train_indices = [idx for idx, bid in enumerate(sample_blocks) if bid != test_block]
            test_indices = [idx for idx, bid in enumerate(sample_blocks) if bid == test_block]

            print(f"   候选训练样本: {len(all_train_indices)}")
            print(f"   测试样本: {len(test_indices)}")

            # 从训练集中再划分验证集
            val_ratio = self.config.get('spatial_cv_val_ratio', 0.2)
            train_indices, val_indices = train_test_split(
                all_train_indices,
                test_size=val_ratio,
                random_state=self.config.get('seed', 42) + test_block
            )

            print(f"\n   最终划分:")
            print(f"     训练样本: {len(train_indices)} ({len(train_indices)/len(all_train_indices)*100:.1f}%)")
            print(f"     验证样本: {len(val_indices)} ({len(val_indices)/len(all_train_indices)*100:.1f}%)")
            print(f"     测试样本: {len(test_indices)}")

            fold_strategy_results = []

            # 6.2 对每种策略进行训练和评估
            for strategy in strategies:
                strategy_name = strategy['name']
                print(f"\n  {'='*50}")
                print(f"  🔬 测试策略: {strategy_name}")
                print(f"     {strategy['description']}")
                lr_config = strategy['lr_config']
                lr_str = f"enc={lr_config.get('lr_encoder', 'N/A'):.0e}, trans={lr_config.get('lr_transformer', 'N/A'):.0e}, head={lr_config.get('lr_head', 'N/A'):.0e}"
                print(f"     分层学习率: {lr_str}")
                print(f"     epochs={strategy['epochs']}, patience={strategy['patience']}")
                print(f"  {'='*50}")

                torch.cuda.empty_cache()
                gc.collect()

                # 创建DataLoader
                train_subset = Subset(station_ds, train_indices)
                val_subset = Subset(station_ds, val_indices)
                test_subset = Subset(station_ds, test_indices)

                if hasattr(station_ds, 'set_augmentation_mode'):
                    station_ds.set_augmentation_mode(True)

                self.train_loader = DataLoader(
                    train_subset,
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    num_workers=self.config.get('num_workers', 8),
                    pin_memory=True,
                    drop_last=True
                )

                if hasattr(station_ds, 'set_augmentation_mode'):
                    station_ds.set_augmentation_mode(False)

                self.val_loader = DataLoader(
                    val_subset,
                    batch_size=self.config['batch_size'],
                    shuffle=False,
                    num_workers=self.config.get('num_workers', 8),
                    pin_memory=True
                )

                self.test_loader = DataLoader(
                    test_subset,
                    batch_size=self.config['batch_size'],
                    shuffle=False,
                    num_workers=self.config.get('num_workers', 8),
                    pin_memory=True
                )

                # ============ 关键：将分层学习率配置合并到 self.config ============
                for k, v in lr_config.items():
                    self.config[k] = v

                # 构建模型
                print(f"\n  🏗️ 构建模型...")
                success = self.build_model(
                    load_pretrained=pretrained_path,
                    freeze_backbone=strategy['freeze_backbone'],
                    freeze_strategy=strategy['freeze_strategy'],
                    use_residual=False,
                    is_cv_fold=True
                )

                if not success:
                    print(f"  ❌ 模型构建失败，跳过策略 {strategy_name}")
                    fold_strategy_results.append({
                        'strategy': strategy_name,
                        'fold': test_block + 1,
                        'success': False,
                        'metrics': None
                    })
                    continue

                # 检查可训练参数
                current_trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                if not current_trainable_params:
                    print(f"  ❌ 没有可训练参数！")
                    fold_strategy_results.append({
                        'strategy': strategy_name,
                        'fold': test_block + 1,
                        'success': False,
                        'metrics': None
                    })
                    continue

                # 保存原始配置
                original_epochs = self.config.get('epochs', 100)
                original_patience = self.config.get('patience', 25)

                # 使用策略的配置
                self.config['epochs'] = strategy['epochs']
                self.config['patience'] = strategy['patience']

                print(f"\n  🚀 开始训练 {strategy_name}...")
                print(f"     分层学习率: enc={lr_config.get('lr_encoder', 'N/A'):.0e}, trans={lr_config.get('lr_transformer', 'N/A'):.0e}, head={lr_config.get('lr_head', 'N/A'):.0e}")
                print(f"     轮次: {strategy['epochs']}, 早停耐心: {strategy['patience']}")
                print(f"     验证集用于 Early Stopping ({len(val_indices)} 样本)")
                print(f"     测试集完全隔离 ({len(test_indices)} 样本)")

                # 训练
                original_val_loader = self.val_loader
                train_result = self.train(fine_tune_mode=True, is_cv_sub_run=True)

                # 恢复原始配置
                self.config['epochs'] = original_epochs
                self.config['patience'] = original_patience

                # 在测试集上评估
                print(f"\n  📊 评估 {strategy_name} 在测试集上的表现...")

                self.val_loader = self.test_loader
                self.model.eval()
                preds, targets, is_zero = self._make_predictions(self.val_loader)
                self.val_loader = original_val_loader

                if preds is not None and len(preds) > 0:
                    s_min = getattr(self, 'swe_min', 0.0)
                    s_max = getattr(self, 'swe_max', 200.0)
                    preds_denorm = preds * (s_max - s_min) + s_min
                    targets_denorm = targets * (s_max - s_min) + s_min

                    rmse = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))
                    mae = np.mean(np.abs(preds_denorm - targets_denorm))

                    ss_res = np.sum((preds_denorm - targets_denorm) ** 2)
                    ss_tot = np.sum((targets_denorm - np.mean(targets_denorm)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    r, _ = stats.pearsonr(preds_denorm.flatten(), targets_denorm.flatten())

                    metrics = {
                        'r2': float(r2),
                        'r': float(r),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'n_samples': len(preds_denorm)
                    }

                    print(f"\n  📈 {strategy_name} 测试集结果:")
                    print(f"    R²: {r2:.4f}, RMSE: {rmse:.2f} mm, MAE: {mae:.2f} mm")

                    self.plot_density_scatter_hardcode(
                        preds_denorm, targets_denorm,
                        is_fine_tune=True,
                        fold_index=f"fold{test_block+1}_{strategy_name}"
                    )

                    fold_strategy_results.append({
                        'strategy': strategy_name,
                        'fold': test_block + 1,
                        'success': True,
                        'metrics': metrics,
                        'n_train': len(train_indices),
                        'n_val': len(val_indices),
                        'n_test': len(test_indices),
                        'lr_config': lr_config,
                        'epochs': strategy['epochs']
                    })

                    strategy_results[strategy_name].append(metrics)

                else:
                    print(f"  ⚠️ 预测失败")
                    fold_strategy_results.append({
                        'strategy': strategy_name,
                        'fold': test_block + 1,
                        'success': False,
                        'metrics': None
                    })

                # 清理显存
                del self.model
                del self.optimizer
                del self.scheduler
                torch.cuda.empty_cache()
                gc.collect()

            all_fold_results.append({
                'fold': test_block + 1,
                'test_block': test_block,
                'block_info': block_info[test_block],
                'n_train_samples': len(train_indices),
                'n_val_samples': len(val_indices),
                'n_test_samples': block_counts[test_block],
                'strategies': fold_strategy_results
            })

        # ============ 7. 汇总分析 ============
        print(f"\n\n{'█'*80}")
        print("🏆 空间块十折交叉验证实验完成！")
        print(f"{'█'*80}")

        # 打印策略配置表
        print(f"\n📋 微调策略配置说明:")
        print(f"{'Method':<24} {'Spatial Encoder':<18} {'Point Encoder':<18} {'Fusion Layer':<16} {'LR Head':<12} {'LR Trans':<12} {'LR Enc':<12}")
        print("-" * 118)
        config_table = [
            ('Fusion-Layer FT', '✗', '✗', '✓', '5e-4', '5e-4', '-'),
            ('Point-Branch FT', '✗', '✓', '✓', '1e-4', '1e-4', '1e-4'),
            ('Spatial-Branch FT', '✓', '✗', '✓', '1e-4', '1e-4', '1e-4'),
            ('Top-Layer FT', 'Top Layer Only', 'Top Layer Only', '✓', '6e-4', '4e-5', '1e-6'),
            ('Full FT', '✓', '✓', '✓', '5e-4', '3e-5', '3e-5'),
        ]
        for name, se, pe, fl, lr_h, lr_t, lr_e in config_table:
            print(f"{name:<24} {se:<18} {pe:<18} {fl:<16} {lr_h:<12} {lr_t:<12} {lr_e:<12}")

        # 打印各策略汇总统计
        print(f"\n📊 各策略在{N_BLOCKS}折中的表现汇总:")
        print(f"\n{'策略':<20} {'r':<18} {'RMSE(mm)':<16} {'MAE(mm)':<16}")
        print("-" * 70)

        summary_results = {}
        for strategy_name, metrics_list in strategy_results.items():
            if metrics_list:
                r_values = [m['r'] for m in metrics_list]
                rmse_values = [m['rmse'] for m in metrics_list]
                mae_values = [m['mae'] for m in metrics_list]

                mean_r = np.mean(r_values)
                std_r = np.std(r_values)
                mean_rmse = np.mean(rmse_values)
                std_rmse = np.std(rmse_values)
                mean_mae = np.mean(mae_values)
                std_mae = np.std(mae_values)

                summary_results[strategy_name] = {
                    'r_mean': float(mean_r),
                    'r_std': float(std_r),
                    'rmse_mean': float(mean_rmse),
                    'rmse_std': float(std_rmse),
                    'mae_mean': float(mean_mae),
                    'mae_std': float(std_mae),
                    'n_folds': len(metrics_list)
                }

                print(f"{strategy_name:<20} {mean_r:.4f}±{std_r:.4f}   {mean_rmse:.2f}±{std_rmse:.2f}   {mean_mae:.2f}±{std_mae:.2f}")
            else:
                print(f"{strategy_name:<20} {'无有效结果':<18} {'无有效结果':<16} {'无有效结果':<16}")

        # 找出最佳策略
        if summary_results:
            best_by_r = max(summary_results.items(), key=lambda x: x[1]['r_mean'])
            best_by_rmse = min(summary_results.items(), key=lambda x: x[1]['rmse_mean'])
            best_by_mae = min(summary_results.items(), key=lambda x: x[1]['mae_mean'])

            print(f"\n🏆 最佳策略 (按 r): {best_by_r[0]} (r={best_by_r[1]['r_mean']:.4f}±{best_by_r[1]['r_std']:.4f})")
            print(f"🏆 最佳策略 (按RMSE): {best_by_rmse[0]} (RMSE={best_by_rmse[1]['rmse_mean']:.2f}±{best_by_rmse[1]['rmse_std']:.2f} mm)")
            print(f"🏆 最佳策略 (按MAE): {best_by_mae[0]} (MAE={best_by_mae[1]['mae_mean']:.2f}±{best_by_mae[1]['mae_std']:.2f} mm)")

        # 绘制 Violin+Box 对比图
        self._plot_strategy_comparison_violin_box(strategy_results)

        # 保存结果
        final_results = {
            'experiment': 'spatial_block_2d_grid_cv',
            'grid_config': {
                'n_rows': N_ROWS,
                'n_cols': N_COLS,
                'n_blocks': N_BLOCKS,
                'row_block_size': row_block_size,
                'col_block_size': col_block_size,
                'val_ratio': val_ratio
            },
            'strategies_config': [
                {
                    'name': s['name'],
                    'lr_config': s['lr_config'],
                    'weight_decay': s['weight_decay'],
                    'epochs': s['epochs'],
                    'patience': s['patience'],
                    'description': s['description']
                }
                for s in strategies
            ],
            'block_info': {str(k): v for k, v in block_info.items() if k < N_BLOCKS},
            'block_samples': dict(block_counts),
            'strategy_results': strategy_results,
            'summary': summary_results,
            'best_by_r': best_by_r[0] if summary_results else None,
            'best_by_rmse': best_by_rmse[0] if summary_results else None,
            'best_by_mae': best_by_mae[0] if summary_results else None,
            'fold_details': all_fold_results
        }

        save_path = self.save_dir / "spatial_block_cv_results.json"
        with open(save_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        print(f"\n💾 结果已保存: {save_path}")

        # 生成CSV报告
        df_rows = []
        for fold_result in all_fold_results:
            for strat_result in fold_result['strategies']:
                if strat_result['success'] and strat_result['metrics']:
                    lr_config = strat_result.get('lr_config', {})
                    df_rows.append({
                        'fold': strat_result['fold'],
                        'strategy': strat_result['strategy'],
                        'r2': strat_result['metrics']['r2'],
                        'rmse': strat_result['metrics']['rmse'],
                        'mae': strat_result['metrics']['mae'],
                        'n_train': strat_result['n_train'],
                        'n_val': strat_result['n_val'],
                        'n_test': strat_result['n_test'],
                        'lr_head': lr_config.get('lr_head', 'N/A'),
                        'lr_transformer': lr_config.get('lr_transformer', 'N/A'),
                        'lr_encoder': lr_config.get('lr_encoder', 'N/A'),
                        'epochs': strat_result.get('epochs', 'N/A')
                    })

        if df_rows:
            df = pd.DataFrame(df_rows)
            csv_path = self.save_dir / "spatial_block_cv_results.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"💾 CSV报告已保存: {csv_path}")

        return final_results

    def _plot_strategy_comparison_violin_box(self, strategy_results):
        """绘制 Violin + Box 组合图（论文级别）- 只包含 r, RMSE, MAE"""
        import seaborn as sns

        from matplotlib.patches import Patch

        # 准备数据
        all_data = []

        for strategy_name, metrics_list in strategy_results.items():
            if not metrics_list:
                continue
            for m in metrics_list:
                all_data.append({
                    'Strategy': strategy_name,
                    'Metric': 'r',
                    'Value': m.get('r', m.get('correlation', np.nan))
                })
                all_data.append({
                    'Strategy': strategy_name,
                    'Metric': 'RMSE (mm)',
                    'Value': m['rmse']
                })
                all_data.append({
                    'Strategy': strategy_name,
                    'Metric': 'MAE (mm)',
                    'Value': m['mae']
                })

        if not all_data:
            print("⚠️ 无有效数据可绘制")
            return

        df_plot = pd.DataFrame(all_data)

        # 论文展示顺序：输出Head归入Fusion Layer，不再单列。
        strategy_order = ['fusion_ft', 'point_ft', 'spatial_ft', 'partial', 'full_ft']
        strategy_labels = {
            'fusion_ft': 'Fusion-Layer FT',
            'point_ft': 'Point-Branch FT',
            'spatial_ft': 'Spatial-Branch FT',
            'partial': 'Top-Layer FT',
            'full_ft': 'Full FT'
        }

        # 只保留存在的策略
        existing_strategies = [s for s in strategy_order if s in df_plot['Strategy'].values]
        df_plot['Strategy'] = pd.Categorical(df_plot['Strategy'], categories=existing_strategies, ordered=True)

        def draw_violin_box(x, y, **kwargs):
            """小提琴图 + 5%-95%箱线图"""
            color = kwargs.pop('color', 'skyblue')
            sns.violinplot(x=x, y=y, inner=None, color=color, cut=0, linewidth=0, alpha=0.3, **kwargs)
            sns.boxplot(x=x, y=y, whis=[5, 95], width=0.35, showfliers=False,
                        color=color,
                        medianprops={'color': 'red', 'linewidth': 2.5, 'label': 'Median'},
                        boxprops={'edgecolor': 'black', 'linewidth': 1.2, 'alpha': 0.85},
                        whiskerprops={'color': 'black', 'linewidth': 1},
                        capprops={'color': 'black', 'linewidth': 1},
                        **kwargs)

        # 创建FacetGrid
        metrics_order = ['r', 'RMSE (mm)', 'MAE (mm)']
        g = sns.FacetGrid(df_plot, col="Metric", hue="Strategy",
                          sharey=False, height=5, aspect=1.2,
                          palette="viridis")
        g.map(draw_violin_box, "Strategy", "Value", order=existing_strategies)

        # 标注中位数
        axes = g.axes.flat
        for i, ax in enumerate(axes):
            if i >= len(metrics_order):
                continue

            metric = metrics_order[i]
            subset = df_plot[df_plot['Metric'] == metric]

            if subset.empty:
                continue

            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min

            for j, strat in enumerate(existing_strategies):
                vals = subset[subset['Strategy'] == strat]['Value'].values
                if len(vals) == 0:
                    continue

                median_val = np.median(vals)

                if metric == 'r':
                    offset = y_range * 0.05
                    va_pos = 'bottom'
                    y_pos = median_val + offset
                    if y_pos > y_max - offset:
                        y_pos = median_val - offset
                        va_pos = 'top'
                else:
                    offset = y_range * 0.05
                    va_pos = 'top'
                    y_pos = median_val - offset
                    if y_pos < y_min + offset:
                        y_pos = median_val + offset
                        va_pos = 'bottom'

                ax.text(j, y_pos, f'{median_val:.3f}',
                        ha='center', va=va_pos, fontsize=9, color='darkred',
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

            # 参考线
            if metric == 'r':
                ax.axhline(1, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
                ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
                ax.set_ylim(-0.1, 1.1)
            elif metric in ['RMSE (mm)', 'MAE (mm)']:
                ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

        # 图例
        handles, labels = axes[0].get_legend_handles_labels()
        unique = []
        for h, l in zip(handles, labels):
            if l not in [u[1] for u in unique]:
                unique.append((h, l))

        unique.append((Patch(facecolor='none', edgecolor='red', linewidth=2.5, label='Median'), 'Median'))

        axes[0].legend(*zip(*unique), loc='upper left', fontsize=10, frameon=True, fancybox=True, shadow=True)

        plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
        g.fig.suptitle("Spatial Block 10-Fold CV: Fine-tuning Strategy Comparison\n(Violin + Box: 5%-95% quantiles, Red = Median)",
                       fontsize=16, fontweight='bold', y=0.98)

        save_path = self.save_dir / "spatial_block_cv_violin_box.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Violin+Box 对比图已保存: {save_path}")

        # 保存中位数汇总表
        median_stats = df_plot.groupby(["Strategy", "Metric"])["Value"].median().reset_index()
        median_pivot = median_stats.pivot(index="Strategy", columns="Metric", values="Value").reset_index()
        median_csv = self.save_dir / "spatial_block_cv_median_summary.csv"
        median_pivot.to_csv(median_csv, index=False, encoding='utf-8-sig')
        print(f"💾 中位数汇总表已保存: {median_csv}")

        return median_pivot

    def run_cv_fine_tuning(self):
        """
        十折交叉验证的编排器 - 独立于 train 函数之外
        """
        print(f"\n{'█'*70}\n🌟 启动自动化十折交叉验证 (固定测试集模式)\n{'█'*70}")

        # 1. 提取所有非测试集的样本索引 (CV 池)
        # 假设你在初始化时已经分出了固定测试集
        train_val_dataset = self.train_loader.dataset # 此时是 Subset
        base_ds = train_val_dataset.dataset
        cv_indices = np.array(train_val_dataset.indices)

        # 2. 按站点 ID 分组，防止空间数据泄露
        station_to_indices = {}
        # 兼容混合模式
        meta_source = base_ds.station_dataset.meta_index if hasattr(base_ds, 'station_dataset') else base_ds.meta_index
        
        for idx in cv_indices:
            sid = meta_source[idx]['station_id']
            if sid not in station_to_indices: station_to_indices[sid] = []
            station_to_indices[sid].append(idx)
        
        unique_stations = np.array(list(station_to_indices.keys()))
        
        # 3. 初始化 KFold
        kf = KFold(n_splits=10, shuffle=True, random_state=self.args.seed)
        all_fold_metrics = []

        # 4. 十折大循环
        for fold, (train_sid_idx, val_sid_idx) in enumerate(kf.split(unique_stations)):
            print(f"\n\n[Fold {fold+1}/10] ---------------------------------------")
            
            # 4.1 准备本折索引
            f_train_idx = []
            f_val_idx = []
            for i in train_sid_idx: f_train_idx.extend(station_to_indices[unique_stations[i]])
            for i in val_sid_idx: f_val_idx.extend(station_to_indices[unique_stations[i]])

            # 4.2 重新构建本折的 Loader
            self.train_loader = DataLoader(Subset(base_ds, f_train_idx), batch_size=self.args.batch_size, shuffle=True)
            self.val_loader = DataLoader(Subset(base_ds, f_val_idx), batch_size=self.args.batch_size, shuffle=False)
            # self.test_loader 保持不变

            # 4.3 关键：每一折彻底重置模型状态
            print(f"  -> 重置模型并加载预训练权重: {self.args.pretrained_model}")
            self.build_model(load_pretrained=self.args.pretrained_model, 
                             freeze_strategy=self.args.freeze_strategy)

            # 4.4 调用原本的 train 函数 (它现在只需要负责跑完 100 Epoch)
            # 注意：我们将训练逻辑封装为单次执行
            self.train(fine_tune_mode=True, is_cv_sub_run=True) 

            # 4.5 本折结束，立即进行测试集预测并画散点图
            print(f"  -> 绘制第 {fold+1} 折测试集预测图...")
            fold_results = self.evaluate_fine_tune(
                loader=self.test_loader, 
                plot_suffix=f"fold_{fold+1}" # 传入后缀，防止图片被覆盖
            )
            all_fold_metrics.append(fold_results['metrics'])

            # 4.6 显存回收
            del self.model
            torch.cuda.empty_cache()
            gc.collect()

        # 5. 最后输出汇总报告
        self._print_cv_final_report(all_fold_metrics)
        
        
    def diagnose_point_encoder_weights(self, save_path=None):
        """诊断 PointEncoder 第一层的权重分布"""
        print("\n" + "="*70)
        print("🔍 PointEncoder 权重诊断")
        print("="*70)

        # 查找 point_encoder 的第一层线性层
        target_layer = None
        for name, module in self.model.named_modules():
            if "point_encoder" in name and isinstance(module, nn.Linear):
                if target_layer is None:  # 取第一层
                    target_layer = module
                    print(f"找到目标层: {name}")
                    print(f"  权重形状: {module.weight.shape}")
                    break

        if target_layer is None:
            print("❌ 未找到 point_encoder 层")
            return

        # 获取权重
        weight = target_layer.weight.data.cpu().numpy()
        print(f"\n权重矩阵形状: {weight.shape} (输出维度, 输入维度)")

        # 分析各维度的权重统计
        input_dim = weight.shape[1]
        print(f"\n【各输入维度的权重统计】")
        print(f"{'维度':<6} {'特征':<20} {'权重均值':<12} {'权重标准差':<12} {'权重绝对值均值':<12}")
        print("-" * 70)

        feature_names = [
            'LS1', 'LS2', 'LS3', 'LS4', 'LS5', 'LS6',
            'S1_VV', 'S1_VH', 'SMAP_TBV', 'SMAP_TBH',
            'lon_norm', 'lat_norm', 'doy_norm'
        ]

        for i in range(min(input_dim, len(feature_names))):
            col_weights = weight[:, i]
            mean_val = np.mean(col_weights)
            std_val = np.std(col_weights)
            abs_mean = np.mean(np.abs(col_weights))

            # 标记新增维度
            marker = " 🔥新增" if i >= 13 else ""
            print(f"{i:<6} {feature_names[i]:<20} {mean_val:<12.6f} {std_val:<12.6f} {abs_mean:<12.6f}{marker}")

        # 特别关注新增的3个维度
        print("\n" + "="*70)
        print("🎯 重点关注：新增维度（降水累积 + 原产品值）")
        print("="*70)

        for i in [13, 14, 15]:
            if i < input_dim:
                col_weights = weight[:, i]
                mean_val = np.mean(col_weights)
                std_val = np.std(col_weights)
                abs_mean = np.mean(np.abs(col_weights))
                zero_ratio = np.sum(np.abs(col_weights) < 1e-6) / len(col_weights)

                status = "✅ 已学习" if abs_mean > 0.01 else "⚠️ 接近零" if abs_mean > 0.001 else "❌ 几乎为零"
                print(f"\n特征 {i} ({feature_names[i]}):")
                print(f"  权重均值: {mean_val:.8f}")
                print(f"  权重标准差: {std_val:.8f}")
                print(f"  权重绝对值均值: {abs_mean:.8f}")
                print(f"  接近零的比例: {zero_ratio*100:.1f}%")
                print(f"  学习状态: {status}")

        # 可选：保存权重到文件
        if save_path:
            np.savez(save_path, weight=weight, feature_names=feature_names[:input_dim])
            print(f"\n💾 权重已保存到: {save_path}")

        return weight
        
        
# 以下是原有代码，保持原有功能...
# 由于代码长度限制，这里省略了原有的 evaluate, run_ablation_experiment, _run_shap_with_hexbin 等方法
# 您可以将原有的这些方法复制到这里


def main():
    """主函数"""
    print("=" * 70)
    print("SWE反演模型训练系统（支持微调）")
    print("=" * 70)

    # ============ CUDA / cuDNN 加速设置 ============
    # 正式训练不要开启 CUDA_LAUNCH_BLOCKING / TORCH_USE_CUDA_DSA
    # 它们只用于定位 CUDA 报错，会显著拖慢训练。

    if os.environ.get("DEBUG_CUDA", "0") == "1":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print("⚠ DEBUG_CUDA=1：开启 CUDA 同步调试模式，训练会变慢")
    else:
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        os.environ.pop("TORCH_USE_CUDA_DSA", None)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print("✅ 正式训练模式：启用确定性 cuDNN")

    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(description="训练SWE反演模型，支持微调")

    # 模式选择
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=[
            "train",
            "evaluate",
            "test",
            "all",
            "ablation",
            "fine_tune",
            "fine_tune_all",
            "pretrain_cv",
            "pretrain_spatial_cv",
            "pretrain_progressive",
            "build_incremental_manifest",
        ],
        help="运行模式"
    )

    # 模型类型
    parser.add_argument(
        "--model_type",
        type=str,
        default="full",
        choices=["full", "spatial_only", "point_only"],
        help="模型类型",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--d_model", type=int, default=256, help="模型维度")

    parser.add_argument(
        "--split_method",
        type=str,
        default="temporal",
        choices=["random", "temporal", "spatial"],
        help="数据划分方法",
    )

    parser.add_argument("--train_year", type=int, default=2015, help="训练年份")
    parser.add_argument("--val_year", type=int, default=2016, help="验证年份")
    
    parser.add_argument(
        "--spatial_grid_lon_step",
        type=float,
        default=1.0,
        help="空间网格经度步长（度），建议1.0°（约111km）"
    )
    parser.add_argument(
        "--spatial_grid_lat_step", 
        type=float,
        default=1.0,
        help="空间网格纬度步长（度）"
    )
    parser.add_argument(
        "--min_samples_per_grid",
        type=int,
        default=10,
        help="每个网格最少样本数"
    )
    parser.add_argument(
        "--cv_mode",
        type=str,
        default="standard",
        choices=["standard", "station_cv", "station_full_cv"],
        help="交叉验证模式"
    )

    # ============ 新增：空间块CV参数 ============
    parser.add_argument('--run_spatial_block_cv', action='store_true',
                        help='运行空间块十折交叉验证实验（独立于cv_mode）')
    parser.add_argument('--spatial_cv_rows', type=int, default=None,
                        help='空间块CV的行数（None则自动计算）')
    parser.add_argument('--spatial_cv_cols', type=int, default=None,
                        help='空间块CV的列数（None则自动计算）')
    parser.add_argument('--spatial_cv_blocks', type=int, default=10,
                        help='空间块CV的块数（默认10折）')
    parser.add_argument('--spatial_cv_val_ratio', type=float, default=0.2,
                        help='空间块CV中验证集比例（从训练集中划分）')
    # ============================================

    parser.add_argument("--fine_tune", action="store_true", help="进行微调训练")
    parser.add_argument("--fine_tune_epochs", type=int, default=50, help="微调轮次")
    parser.add_argument("--fine_tune_lr", type=float, default=5e-4, help="微调学习率")
    parser.add_argument("--freeze_backbone", action="store_true", help="微调时冻结主干网络")
    
    parser.add_argument(
        "--station_data_path", 
        type=str, 
        default="/root/autodl-tmp/combined_station.csv",
        help="站点数据路径（用于微调），可以是文件或目录"
    )
    parser.add_argument(
        "--pretrained_model", 
        type=str, 
        default=None,  # 保持None，让程序自动查找
        help="预训练模型路径（不指定则自动查找）"
    )
    parser.add_argument(
        '--freeze_strategy', 
        type=str, 
        default='fusion_ft',
        choices=['fusion_ft', 'point_ft', 'spatial_ft', 'partial', 'none'],
        help='冻结策略'
    )

    parser.add_argument("--save_dir", type=str, default="./experiments", help="保存目录")
    parser.add_argument("--exp_name", type=str, default=None, help="实验名称")
    parser.add_argument(
        "--resume_pretrain_cv",
        action="store_true",
        help=(
            "在同一实验目录中继续预训练十折CV；跳过已完整结束的折。"
            "必须保持相同数据缓存、样本顺序、seed和CV参数。"
        ),
    )
    parser.add_argument(
        "--redraw_completed_cv_plots",
        action="store_true",
        help=(
            "续跑时对已完成折加载 best checkpoint 重新推理并覆盖散点图；"
            "只重绘、不训练，用于修复旧图错误的 SWE 反归一化尺度。"
        ),
    )

    # 预训练采样来源（显式参数化）
    parser.add_argument(
        '--sampling_mode', type=str, default='auto',
        choices=['auto', 'random', 'station', 'hybrid', 'incremental'],
        help=('auto=兼容旧参数；random=仅全国随机；station=仅站点位置；'
              'hybrid=随机+站点；incremental=读取固定152000清单中的当前新增包'),
    )
    parser.add_argument('--use_station_guide', action='store_true',
                        help='旧参数兼容；sampling_mode=auto 时启用 hybrid')
    parser.add_argument('--station_guide_file', type=str, default=None,
                        help='站点引导使用的单个 xlsx/csv 文件；推荐传完整路径')
    parser.add_argument('--station_neighborhood', type=int, default=3,
                        help='站点中心格点邻域半径；0=仅站点所在格点')
    parser.add_argument('--station_samples_per_day', type=int, default=2000,
                        help='每天最多添加的站点样本数；<=0=使用全部有效站点格点')
    parser.add_argument('--station_include_zero_target', action='store_true',
                        help='站点引导时保留 ERA5-Land SWE=0 样本；默认过滤')
    parser.add_argument(
        '--station_sampling_unit', type=str, default='positions_all_dates',
        choices=['positions_all_dates', 'records'],
        help=(
            'positions_all_dates=旧逻辑，站点位置遍历全部标签日期；'
            'records=只使用站点文件中实际存在的站点-日期记录'
        ),
    )
    parser.add_argument(
        '--station_record_dedup', type=str, default='grid_date',
        choices=['grid_date', 'none'],
        help='records模式去重规则；grid_date按(date,row,col)去重，推荐',
    )
    parser.add_argument(
        '--station_record_manifest_path', type=str, default=None,
        help='正式Stage 0直接读取预先冻结的有效(date,row,col)清单',
    )
    parser.add_argument(
        '--station_date_column', type=str, default=None,
        help='records模式日期列名；默认自动识别 date/datetime/日期 等',
    )
    parser.add_argument('--station_csv_dir', type=str, default="/root/ablation",
                        help='兼容旧目录模式；传 station_guide_file 时忽略')
    parser.add_argument(
        '--external_station_glob', type=str, default=None,
        help='第二类外部测试站点CSV规则，例如 /root/ablation/external_test/*.csv',
    )
    parser.add_argument(
        '--external_station_exclusion_radius', type=int, default=0,
        help='外部测试站点在ERA5网格上的排除半径；5x5 Patch推荐2格',
    )
    parser.add_argument(
        '--external_station_strict', action='store_true',
        help='外部测试CSV未匹配、缺经纬度或无有效格点时直接停止',
    )
    parser.add_argument(
        '--external_station_report_path', type=str, default=None,
        help='外部测试站点排除格点报告CSV路径',
    )

    # ============ 固定152000增量样本池 ============
    parser.add_argument('--incremental_manifest_path', type=str, default=None,
                        help='固定152000增量样本清单CSV路径')
    parser.add_argument('--incremental_stage', type=int, default=1,
                        help='当前训练的增量阶段编号；阶段上限由incremental_stage_sizes决定')
    parser.add_argument('--build_incremental_manifest', action='store_true',
                        help='首次Stage 1运行时构建/覆盖固定152000清单')
    parser.add_argument('--incremental_pool_size', type=int, default=152000,
                        help='固定随机样本池总量')
    parser.add_argument('--incremental_stage_sizes', type=int, nargs='+',
                        default=[12000, 20000, 40000, 80000],
                        help='各个互不重叠新增包的样本数，例如12k 20k 40k 80k 160k')
    parser.add_argument('--incremental_seed', type=int, default=43,
                        help='固定池、阶段分配和fold分配种子')
    parser.add_argument('--incremental_candidate_oversample_factor', type=float, default=3.0,
                        help='严格特征验证前的候选过采倍数；样本不足时增大')
    parser.add_argument('--incremental_ratio_config', type=str, default=None,
                        help='可选JSON：SWE细分箱目标比例；不传则使用33/33/34默认分布')
    parser.add_argument('--incremental_glacier_mask_path', type=str, default=None,
                        help='可选冰川/永久冰雪掩膜，>0为排除区域')
    parser.add_argument('--incremental_fold_block_pixels', type=int, default=0,
                        help='0=按stage和SWE箱固定随机十折；>0=按指定像元块固定空间折')
    parser.add_argument('--allow_station_pixels_in_incremental_pool', action='store_true',
                        help='允许固定随机池包含站点格点；默认排除，保证与Stage0位置来源分开')

    # 季节性判据只作用于152000随机池，绝不作用于Stage0站点位置样本。
    parser.add_argument('--seasonal_min_peak_swe_mm', type=float, default=1.0,
                        help='积雪年最大SWE严格下限，默认要求>1 mm')
    parser.add_argument('--seasonal_max_swe_mm', type=float, default=400.0,
                        help='随机池质量控制上限，要求年最大及单日SWE均小于该值')
    parser.add_argument('--seasonal_snow_free_threshold_mm', type=float, default=1.0,
                        help='最后一次年峰值后近无雪状态阈值，默认SWE≤1 mm')
    parser.add_argument('--seasonal_min_warm_snow_free_ratio', type=float, default=0.0,
                        help='旧参数兼容；当前不作为硬筛选条件')
    parser.add_argument('--seasonal_min_consecutive_snow_free_days', type=int, default=5,
                        help='最后一次年峰值后最少连续近无雪日数，默认5天')
    parser.add_argument('--seasonal_min_snow_year_coverage_ratio', type=float, default=0.90,
                        help='格点在完整积雪年内的最低有效日期覆盖率')

    # Stage0写出；Stage1-4只读取，避免跨阶段归一化漂移。
    parser.add_argument('--normalization_config_path', type=str, default=None,
                        help='跨阶段共享归一化JSON路径')
    parser.add_argument('--normalization_mode', type=str, default='auto',
                        choices=['auto', 'create', 'load', 'skip'],
                        help='正式训练用load；清单/统一统计准备可用skip')
    parser.add_argument('--fixed_label_min_mm', type=float, default=0.0,
                        help='所有阶段固定的SWE归一化下限')
    parser.add_argument('--fixed_label_max_mm', type=float, default=400.0,
                        help='所有阶段固定的SWE归一化上限')
    
    parser.add_argument('--use_adaptive_supplement', action='store_true',
                        help='启用自适应修正')
    parser.add_argument('--adaptive_alpha', type=float, default=0.5,
                        help='自适应修正平衡强度')
    parser.add_argument('--adaptive_threshold', type=float, default=1.5,
                        help='短缺阈值')
    
    # 消融实验参数
    parser.add_argument("--run_ablation", action="store_true", help="运行消融实验")
    parser.add_argument("--ablation_samples", type=int, default=None, help="消融实验使用的样本数")
    parser.add_argument("--ablation_max_samples", type=int, default=None, help="消融实验最大样本数")
    parser.add_argument("--model_path", type=str, default=None, help="模型文件路径")
    parser.add_argument("--ablation_method", type=str, default="retrain",
                        choices=["retrain", "zeroing"], help="消融实验方法")
    parser.add_argument("--retrain_epochs", type=int, default=50, help="重新训练的轮次")
    
    # LoRA参数
    parser.add_argument("--use_lora", action="store_true", help="使用LoRA进行微调")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA缩放参数")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout率")
    parser.add_argument("--lora_target_modules", type=str, nargs='+',
                        default=['linear', 'conv'], help="LoRA目标模块")
    
    # 混合模式参数
    parser.add_argument('--mixed_mode', action='store_true', help='使用混合模式')
    parser.add_argument('--station_ratio', type=float, default=0.7, help='站点数据的比例')
    
    # 高级训练参数
    parser.add_argument('--use_residual', action='store_true', help='使用残差学习模式')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--val_every', type=int, default=1,
                        help='每隔多少个 epoch 验证一次（默认每 epoch 验证）')
    parser.add_argument(
        '--train_only_overfit',
        action='store_true',
        help=(
            '仅用Fold 1的8-fold训练池进行完整训练集过拟合诊断；'
            '跳过inner选模、outer OOF、early stopping和Plateau调度'
        ),
    )
    parser.add_argument(
        '--inner_pilot_only',
        action='store_true',
        help=(
            '仅运行一个Nested 8/1/1的train+inner选模pilot；'
            'outer fold只保留，不创建DataLoader、不预测、不输出OOF'
        ),
    )
    parser.add_argument(
        '--nested_max_folds',
        type=int,
        default=10,
        help=(
            'Nested运行折数；正式模式必须为10，'
            'inner_pilot_only模式当前必须为1'
        ),
    )
    parser.add_argument(
        '--include_fixed_internal_in_cv',
        action='store_true',
        help=(
            '将原split=test的旧内部1000条并回站点级Nested CV；'
            '原始CSV不改写，外部987条仍不参与选模'
        ),
    )
    parser.add_argument(
        '--overfit_resume_checkpoint',
        type=str,
        default=None,
        help=(
            '训练集过拟合诊断断点续训checkpoint；恢复模型、AdamW、'
            '历史轮次与可用随机状态，fine_tune_epochs表示续训后的总轮数'
        ),
    )
    parser.add_argument(
        '--overfit_resume_transformer_lr',
        type=float,
        default=None,
        help=(
            '仅在训练集过拟合断点恢复后覆盖Transformer参数组学习率；'
            'Head和其他训练设置保持checkpoint配置'
        ),
    )
    parser.add_argument(
        '--high_swe_transfer_audit_only',
        action='store_true',
        help=(
            '只在Nested单Inner划分上评估既有checkpoint的完整训练池与'
            'Inner高SWE迁移差距；不训练、不创建Outer DataLoader'
        ),
    )
    parser.add_argument(
        '--transfer_audit_checkpoint',
        action='append',
        default=None,
        metavar='NAME=PATH',
        help=(
            '高SWE迁移诊断checkpoint，可重复提供，例如'
            '--transfer_audit_checkpoint frozen=/path/model.pth'
        ),
    )

    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='plateau',
        choices=['plateau', 'warmup_cosine'],
        help='预训练CV学习率调度器'
    )
    parser.add_argument(
        '--warmup_start_lr',
        type=float,
        default=1e-5,
        help='warmup起始学习率'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        help='cosine最终最小学习率'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.05,
        help='warmup占总optimizer steps比例'
    )
    parser.add_argument(
        '--pretrain_cv_max_folds',
        type=int,
        default=10,
        help='预训练CV最多运行多少折；诊断时可设为1'
    )
    parser.add_argument('--disable_pretrain_cv_early_stopping', action='store_true',
                        help='预训练十折每折固定跑满epochs，不提前停止')
    parser.add_argument('--profile_timing', action='store_true',
                        help='开启 batch 级时间诊断（cuda.synchronize + step_time 打印，正式训练建议关闭）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 分层学习率
    parser.add_argument('--lr_head', type=float, default=5e-4, help='Head层学习率')
    parser.add_argument('--lr_transformer', type=float, default=2e-5, help='Transformer层学习率')
    parser.add_argument('--lr_encoder', type=float, default=5e-5, help='编码器层学习率')
    parser.add_argument(
        '--disable_fusion_ft_progressive_unfreeze',
        action='store_true',
        help='关闭Fusion FT逐层解冻（默认开启）',
    )
    parser.add_argument(
        '--fusion_ft_head_lr',
        type=float,
        default=5e-5,
        help='Fusion FT回归Head峰值学习率',
    )
    parser.add_argument(
        '--fusion_ft_transformer_lr',
        type=float,
        default=3e-5,
        help='Fusion FT Transformer峰值学习率',
    )
    parser.add_argument(
        '--fusion_ft_warmup_epochs',
        type=int,
        default=10,
        help='Fusion FT按optimizer step线性warmup轮数',
    )
    parser.add_argument(
        '--fusion_ft_unfreeze_interval',
        type=int,
        default=1,
        help='Fusion FT从顶层向底层逐层解冻的epoch间隔',
    )
    parser.add_argument(
        '--fusion_ft_max_unfrozen_layers',
        type=int,
        default=0,
        help=(
            'Fusion FT最多解冻的顶部Transformer层数；'
            '0保持旧行为并最终解冻全部层'
        ),
    )
    parser.add_argument(
        '--fusion_ft_plateau_patience',
        type=int,
        default=8,
        help='正式Fusion FT ReduceLROnPlateau耐心轮数',
    )
    parser.add_argument(
        '--fusion_ft_patience',
        type=int,
        default=20,
        help='正式Fusion FT early stopping耐心轮数',
    )
    parser.add_argument(
        '--fusion_ft_min_epochs_before_early_stop',
        type=int,
        default=40,
        help='正式Fusion FT允许early stopping前的最少训练轮数',
    )
    parser.add_argument(
        '--fusion_ft_ccc_weight',
        type=float,
        default=0.005,
        help='Fusion FT的CCC结构损失权重',
    )
    parser.add_argument(
        '--fusion_ft_variance_weight',
        type=float,
        default=0.0,
        help='Fusion FT旧batch方差惩罚权重（CCC版本默认关闭）',
    )
    
    # 预训练样本筛选
    parser.add_argument('--spatial_balance', action='store_true', help='空间平衡采样')
    parser.add_argument('--temporal_balance', action='store_true', help='时间平衡采样')
    parser.add_argument('--max_pretrain', type=int, default=8000, help='最大预训练样本数')
    
    # MixUp参数
    parser.add_argument('--use_mixup', action='store_true', help='使用MixUp数据增强')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='MixUp的Beta分布参数')
    parser.add_argument('--mixup_prob', type=float, default=0.5, help='应用MixUp的概率')
    parser.add_argument('--mixup_warmup', type=int, default=5, help='前几轮不用MixUp')
    
    # 课程学习参数
    parser.add_argument('--use_curriculum', action='store_true', help='使用课程学习')
    parser.add_argument('--curriculum_start', type=float, default=0.3, help='起始样本比例')
    parser.add_argument('--curriculum_update_freq', type=int, default=5, help='难度更新频率')
    
    # 残差注入和门控模型
    parser.add_argument('--residual_injection', action='store_true', help='启用特征级残差注入')
    parser.add_argument('--lambda_elastic', type=float, default=0.1, help='弹性约束系数')
    parser.add_argument('--use_gate', action='store_true', help='使用门控融合模型')
    
    # 数据增强参数
    parser.add_argument('--coord_jitter_std', type=float, default=0.02, help='坐标抖动标准差')
    parser.add_argument('--microwave_noise_std', type=float, default=0.01, help='微波信号噪声标准差')
    parser.add_argument('--coord_mask_prob', type=float, default=0.2, help='坐标掩码概率')
    parser.add_argument('--use_tta', action='store_true', help='使用测试时增强')
    parser.add_argument('--tta_num', type=int, default=8, help='TTA增强次数')
    
    # 其他参数
    parser.add_argument("--num_workers", type=int, default=10, help="数据加载线程数")
    parser.add_argument("--from_scratch", action="store_true", help="从头训练模式")
    parser.add_argument("--pretrain_years", type=int, nargs='+', default=[2015, 2016, 2017],
                        help="预训练年份")
    parser.add_argument("--pretrain_samples_per_day", "--samples_per_day",
                        dest="pretrain_samples_per_day",
                        type=int,
                        default=20000,
                        help="预训练随机采样每日样本数；例如 20000 的 70%% 是 14000")
    parser.add_argument("--patch_size", type=int, default=5,
                        help="预训练 patch 大小")
    parser.add_argument("--min_valid_pixels", type=int, default=100,
                        help="预训练 patch 内最少有效像元数")
    parser.add_argument("--clamday_threshold", type=float, default=0.5,
                        help="Calm day 阈值")
    parser.add_argument("--shared_cache_dir", type=str, default="/root/autodl-tmp/shared_cache",
                        help="预训练共享缓存目录")
    parser.add_argument("--disable_dataset_cache", action="store_true",
                        help="清单/统计准备阶段禁用Dataset pickle缓存")
    parser.add_argument("--force_reload", action="store_true",
                        help="强制忽略现有 SWEDataset 缓存并重建")

    # ============ 最终全样本 refit 参数 ============
    parser.add_argument('--final_train_ratio', type=float, default=1.0,
                        help='最终全样本训练比例（1.0=100%%, 0.95=95%%）')
    parser.add_argument('--final_epochs_mode', type=str, default='fixed',
                        choices=['fixed', 'cv_median'],
                        help='最终训练轮数模式：fixed=固定轮数, cv_median=十折最佳轮次中位数')
    parser.add_argument('--final_epochs', type=int, default=100,
                        help='final_epochs_mode=fixed 时的训练轮数')
    parser.add_argument('--final_scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'warmup_cosine'],
                        help='最终全样本训练的学习率调度器')

    parser.add_argument('--split_cache_file', type=str, default=None,
                    help='划分缓存文件路径（多个策略共享时使用）')
    parser.add_argument('--force_recompute_split', action='store_true',
                        help='强制重新计算划分（忽略缓存）')
    
    
    parser.add_argument('--shared_cache_mode', action='store_true',
                    help='启用共享缓存模式（十折CV共用缓存）')
    
    parser.add_argument('--use_counterfactual_prior_loss', action='store_true',
                        help='启用反事实 prior 损失（强制模型在没有 prior 时也能预测）')
    parser.add_argument('--counterfactual_prior_loss_weight', type=float, default=0.5,
                        help='反事实 prior 损失的权重')   
    
    
    parser.add_argument(
        '--pretrain_loss_weight',
        type=float,
        default=0,
        help='mixed mode 中预训练伪标签样本的loss权重'
    )
    parser.add_argument(
        '--use_high_swe_weight',
        action='store_true',
        default=True,
        help='是否对站点高SWE样本进行温和加权'
    )

    # 预训练样本筛选阈值
    parser.add_argument(
        '--pretrain_snow_min_mm',
        type=float,
        default=20.0,
        help='判定为雪样本的SWE阈值(mm)'
    )
    parser.add_argument(
        '--quality_threshold',
        type=float,
        default=0.83,
        help='非雪样本的质量阈值'
    )
    parser.add_argument(
        '--snow_quality_threshold',
        type=float,
        default=0.60,
        help='雪样本的质量阈值（放宽）'
    )
    parser.add_argument(
        '--pretrain_snow_priority_ratio',
        type=float,
        default=1.0,
        help='预训练雪样本优先比例（1.0=尽量全选>=20mm样本）'
    )
    
    parser.add_argument(
        "--use_product_correction",
        action="store_true",
        help="开启 full_sample_predictions.csv / zero_misclassifications.csv 对第21维产品值的修正"
    )
    
    args = parser.parse_args()
    if (
        args.overfit_resume_checkpoint
        and not args.train_only_overfit
    ):
        parser.error(
            "--overfit_resume_checkpoint必须与"
            "--train_only_overfit一起使用"
        )
    if (
        args.overfit_resume_transformer_lr is not None
        and not args.overfit_resume_checkpoint
    ):
        parser.error(
            "--overfit_resume_transformer_lr必须与"
            "--overfit_resume_checkpoint一起使用"
        )
    if (
        args.overfit_resume_transformer_lr is not None
        and args.overfit_resume_transformer_lr <= 0
    ):
        parser.error(
            "--overfit_resume_transformer_lr必须大于0"
        )
    if args.inner_pilot_only and args.train_only_overfit:
        parser.error(
            "--inner_pilot_only与--train_only_overfit互斥"
        )
    if args.high_swe_transfer_audit_only:
        if not args.inner_pilot_only:
            parser.error(
                "--high_swe_transfer_audit_only要求"
                "--inner_pilot_only"
            )
        if args.train_only_overfit:
            parser.error(
                "--high_swe_transfer_audit_only不能与"
                "--train_only_overfit同时使用"
            )
        if not args.transfer_audit_checkpoint:
            parser.error(
                "--high_swe_transfer_audit_only至少需要一个"
                "--transfer_audit_checkpoint NAME=PATH"
            )
    if args.inner_pilot_only and args.nested_max_folds != 1:
        parser.error(
            "--inner_pilot_only当前要求--nested_max_folds=1"
        )
    if not args.inner_pilot_only and args.nested_max_folds != 10:
        parser.error(
            "正式Nested OOF要求--nested_max_folds=10"
        )
    if args.fusion_ft_transformer_lr <= 0:
        parser.error("--fusion_ft_transformer_lr必须大于0")
    if args.fusion_ft_warmup_epochs < 0:
        parser.error("--fusion_ft_warmup_epochs不能小于0")
    if args.fusion_ft_max_unfrozen_layers < 0:
        parser.error("--fusion_ft_max_unfrozen_layers不能小于0")
    if args.fusion_ft_plateau_patience < 0:
        parser.error("--fusion_ft_plateau_patience不能小于0")
    if args.fusion_ft_patience <= 0:
        parser.error("--fusion_ft_patience必须大于0")
    if args.fusion_ft_min_epochs_before_early_stop < 0:
        parser.error(
            "--fusion_ft_min_epochs_before_early_stop不能小于0"
        )
    print("=" * 80)
    print(f"[DEBUG] 命令行接收到的参数:")
    print(f"  freeze_strategy = {args.freeze_strategy}")
    print(f"  mode = {args.mode}")
    print(f"  cv_mode = {args.cv_mode}")
    print("=" * 80)
    
    # 设置随机种子
    if args.seed == 42:
        experiment_seed = int(time.time())
        print(f"🔑 使用动态种子: {experiment_seed}")
    else:
        experiment_seed = args.seed
        print(f"🔑 使用指定种子: {experiment_seed}")

    # DETERMINISTIC_TRAINING_V1
    # 必须在Dataset、DataLoader和模型构建之前统一设置。
    _configure_reproducibility(
        experiment_seed,
        verbose=True,
    )
    
    # 创建基础配置
    config = {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "d_model": args.d_model,
        "save_dir": args.save_dir,
        "experiment_name": args.exp_name,
        "resume_pretrain_cv": args.resume_pretrain_cv,
        "redraw_completed_cv_plots": args.redraw_completed_cv_plots,
        "split_method": args.split_method,
        "train_year": args.train_year,
        "val_year": args.val_year,
        "spatial_grid_lon_step": args.spatial_grid_lon_step,
        "spatial_grid_lat_step": args.spatial_grid_lat_step,
        "pretrain_years": args.pretrain_years,
        "pretrain_samples_per_day": args.pretrain_samples_per_day,
        "patch_size": args.patch_size,
        "min_valid_pixels": args.min_valid_pixels,
        "clamday_threshold": args.clamday_threshold,
        "shared_cache_dir": args.shared_cache_dir,
        "disable_dataset_cache": args.disable_dataset_cache,
        "force_reload": args.force_reload,
        "use_adaptive_supplement": args.use_adaptive_supplement,
        "adaptive_alpha": args.adaptive_alpha,
        "adaptive_threshold": args.adaptive_threshold,
        "fine_tune": args.fine_tune,
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_lr": args.fine_tune_lr,
        "freeze_backbone": args.freeze_backbone,
        "freeze_strategy": args.freeze_strategy,
        "station_data_path": args.station_data_path,
        "cv_mode": args.cv_mode,
        "sampling_mode": args.sampling_mode,
        "use_station_guide": args.use_station_guide,
        "station_guide_file": args.station_guide_file,
        "station_neighborhood": args.station_neighborhood,
        "station_samples_per_day": args.station_samples_per_day,
        "station_filter_zero_target": not args.station_include_zero_target,
        "station_sampling_unit": args.station_sampling_unit,
        "station_record_dedup": args.station_record_dedup,
        "station_date_column": args.station_date_column,
        "station_record_manifest_path": args.station_record_manifest_path,
        "station_csv_dir": args.station_csv_dir,
        "external_station_glob": args.external_station_glob,
        "external_station_exclusion_radius": args.external_station_exclusion_radius,
        "external_station_strict": args.external_station_strict,
        "external_station_report_path": args.external_station_report_path,
        "pretrained_model": args.pretrained_model,
        # 固定增量池
        "incremental_manifest_path": args.incremental_manifest_path,
        "incremental_stage": args.incremental_stage,
        "build_incremental_manifest": args.build_incremental_manifest,
        "incremental_pool_size": args.incremental_pool_size,
        "incremental_stage_sizes": args.incremental_stage_sizes,
        "incremental_seed": args.incremental_seed,
        "incremental_candidate_oversample_factor": args.incremental_candidate_oversample_factor,
        "incremental_exclude_station_pixels": not args.allow_station_pixels_in_incremental_pool,
        "incremental_ratio_config": args.incremental_ratio_config,
        "incremental_glacier_mask_path": args.incremental_glacier_mask_path,
        "incremental_fold_block_pixels": args.incremental_fold_block_pixels,
        "seasonal_min_peak_swe_mm": args.seasonal_min_peak_swe_mm,
        "seasonal_max_swe_mm": args.seasonal_max_swe_mm,
        "seasonal_snow_free_threshold_mm": args.seasonal_snow_free_threshold_mm,
        "seasonal_min_warm_snow_free_ratio": args.seasonal_min_warm_snow_free_ratio,
        "seasonal_min_consecutive_snow_free_days": args.seasonal_min_consecutive_snow_free_days,
        "seasonal_min_snow_year_coverage_ratio": args.seasonal_min_snow_year_coverage_ratio,
        "normalization_config_path": args.normalization_config_path,
        "normalization_mode": args.normalization_mode,
        "fixed_label_min_mm": args.fixed_label_min_mm,
        "fixed_label_max_mm": args.fixed_label_max_mm,
        "use_lora": args.use_lora,
        "lora_config": {
            "target_modules": args.lora_target_modules,
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        "mixed_mode": args.mixed_mode,
        "station_ratio": args.station_ratio,

        # ============ Mixed mode loss 控制 ============
        "pretrain_loss_weight": args.pretrain_loss_weight,
        "use_high_swe_weight": args.use_high_swe_weight,

        "quality_threshold": args.quality_threshold,
        "snow_quality_threshold": args.snow_quality_threshold,
        "pretrain_snow_min_mm": args.pretrain_snow_min_mm,
        "pretrain_snow_priority_ratio": args.pretrain_snow_priority_ratio,
        "spatial_balance": args.spatial_balance,
        "temporal_balance": args.temporal_balance,
        "max_pretrain": args.max_pretrain,
        "use_mixup": args.use_mixup,
        "mixup_alpha": args.mixup_alpha,
        "mixup_prob": args.mixup_prob,
        "mixup_warmup": args.mixup_warmup,
        "use_curriculum": args.use_curriculum,
        "curriculum_start": args.curriculum_start,
        "curriculum_update_freq": args.curriculum_update_freq,
        "use_gate": args.use_gate,
        "residual_injection": args.residual_injection,
        "lambda_elastic": args.lambda_elastic,
        "coord_jitter_std": args.coord_jitter_std,
        "microwave_noise_std": args.microwave_noise_std,
        "coord_mask_prob": args.coord_mask_prob,
        "from_scratch": args.from_scratch,
        "val_ratio": 0.2,
        "num_workers": args.num_workers,
        "weight_decay": 1e-5,
        "patience": 25,
        "seed": experiment_seed,
        "use_amp": args.use_amp,
        "val_every": args.val_every,
        "train_only_overfit": args.train_only_overfit,
        "inner_pilot_only": args.inner_pilot_only,
        "nested_max_folds": args.nested_max_folds,
        "include_fixed_internal_in_cv": (
            args.include_fixed_internal_in_cv
        ),
        "overfit_resume_checkpoint": args.overfit_resume_checkpoint,
        "overfit_resume_transformer_lr": (
            args.overfit_resume_transformer_lr
        ),
        "high_swe_transfer_audit_only": (
            args.high_swe_transfer_audit_only
        ),
        "transfer_audit_checkpoints": (
            args.transfer_audit_checkpoint or []
        ),
        "lr_scheduler": args.lr_scheduler,
        "warmup_start_lr": args.warmup_start_lr,
        "min_lr": args.min_lr,
        "warmup_ratio": args.warmup_ratio,
        "lr_head": args.lr_head,
        "lr_transformer": args.lr_transformer,
        "lr_encoder": args.lr_encoder,
        "fusion_ft_progressive_unfreeze": (
            not args.disable_fusion_ft_progressive_unfreeze
        ),
        "fusion_ft_head_lr": args.fusion_ft_head_lr,
        "fusion_ft_transformer_lr": args.fusion_ft_transformer_lr,
        "fusion_ft_warmup_epochs": args.fusion_ft_warmup_epochs,
        "fusion_ft_unfreeze_interval": args.fusion_ft_unfreeze_interval,
        "fusion_ft_max_unfrozen_layers": (
            args.fusion_ft_max_unfrozen_layers
        ),
        "fusion_ft_plateau_patience": (
            args.fusion_ft_plateau_patience
        ),
        "fusion_ft_patience": args.fusion_ft_patience,
        "fusion_ft_min_epochs_before_early_stop": (
            args.fusion_ft_min_epochs_before_early_stop
        ),
        "fusion_ft_ccc_weight": args.fusion_ft_ccc_weight,
        "fusion_ft_variance_weight": args.fusion_ft_variance_weight,
        "pretrain_cv_max_folds": args.pretrain_cv_max_folds,
        "disable_pretrain_cv_early_stopping": args.disable_pretrain_cv_early_stopping,
        "profile_timing": args.profile_timing,
        "clip_grad": 1.0,
        "save_freq": 10,
        # ============ 空间块CV配置 ============
        "spatial_cv_rows": args.spatial_cv_rows,
        "spatial_cv_cols": args.spatial_cv_cols,
        "spatial_cv_blocks": args.spatial_cv_blocks,
        "spatial_cv_val_ratio": args.spatial_cv_val_ratio,

        "split_cache_file": args.split_cache_file,
        "force_recompute_split": args.force_recompute_split,
        "shared_cache_mode": args.shared_cache_mode,
        
        
        "use_product_correction": args.use_product_correction,
        "use_counterfactual_prior_loss": args.use_counterfactual_prior_loss,
        "counterfactual_prior_loss_weight": args.counterfactual_prior_loss_weight,
        
        "use_prior_dropout": False,
        "prior_dropout_p": 0.0,

        # ============ 最终全样本 refit 配置 ============
        "final_train_ratio": args.final_train_ratio,
        "final_epochs_mode": args.final_epochs_mode,
        "final_epochs": args.final_epochs,
        "final_scheduler": args.final_scheduler,
    }
    # ============ 设置实验名称 ============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.run_spatial_block_cv:
        # 空间块CV的特殊命名
        config["experiment_name"] = f"spatial_block_cv_{timestamp}"
    elif args.exp_name:
        config["experiment_name"] = args.exp_name
    elif args.mode in ["fine_tune", "fine_tune_all"]:
        suffix = "_fine_tune"
        strategy_suffix = f"_{args.freeze_strategy}"
        mixed_suffix = "_mixed" if args.mixed_mode else ""
        config["experiment_name"] = (
            f"swe_{args.model_type}_{args.split_method}{suffix}{strategy_suffix}{mixed_suffix}_{timestamp}"
        )
    else:
        config["experiment_name"] = (
            f"swe_{args.model_type}_{args.split_method}_{timestamp}"
        )

    print(f"\n完整配置:")
    for key, value in config.items():
        if key not in ["experiment_name", "station_data_path"]:
            print(f"  {key:25s}: {value}")
    
    print(f"  实验名称:          {config['experiment_name']}")
    print(f"  站点数据路径:      {config['station_data_path']}")
    print(f"  混合模式:          {config['mixed_mode']}")
    print(f"  站点比例:          {config['station_ratio']}")
    
    # 检查站点数据路径
    station_path = Path(config['station_data_path'])
    if args.mode in ["fine_tune", "fine_tune_all"] and not station_path.exists():
        print(f"\n⚠ 警告: 站点数据路径不存在: {station_path}")
        # 尝试查找可用文件
        possible_files = [
            "/root/autodl-tmp/combined_station.csv",
            "/root/ablation/station_swe_data.xlsx",
            "/root/ablation/external_test/long_comb.csv",
        ]
        found = [f for f in possible_files if Path(f).exists()]
        if found:
            config['station_data_path'] = found[0]
            print(f"  将使用: {found[0]}")
        else:
            print(f"  ✗ 没有找到任何数据文件，微调将无法进行")

    # 创建训练器
    trainer = SWETrainer(config)
    
    # 保存随机种子
    seed_file = trainer.save_dir / "experiment_seed.txt"
    with open(seed_file, 'w') as f:
        f.write(f"seed = {experiment_seed}\n")
        f.write(f"timestamp = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"💾 种子已保存: {seed_file}")

    # ============ 优先执行：空间块CV ============
    if args.run_spatial_block_cv:
        print("\n" + "=" * 70)
        print("开始空间块十折交叉验证实验")
        print("=" * 70)
        
        # 空间块CV需要加载数据
        if trainer.load_data(fine_tune_mode=True, mixed_mode=args.mixed_mode, station_ratio=args.station_ratio):
            trainer.run_spatial_block_cv_experiment()
        return

    # ============ 根据模式执行 ============
    if args.mode == "train":
        print("\n" + "=" * 70)
        print("开始训练模式")
        print("=" * 70)
        
        if trainer.load_data():
            if trainer.build_model():
                trainer.train()
                
    elif args.mode == "evaluate":
        print("\n" + "=" * 70)
        print("开始评估模式")
        print("=" * 70)
        
        if trainer.load_data(fine_tune_mode=True):
            if trainer.build_model(load_pretrained=args.pretrained_model):
                trainer.config["freeze_strategy"] = "frozen"
                trainer.evaluate_fine_tune(model_path=args.model_path, use_tta=False, tta_num=8)

    elif args.mode == "test":
        print("\n" + "=" * 70)
        print("开始测试模式")
        print("=" * 70)
        
        print("1. 测试模型结构...")
        try:
            test_model()
        except Exception as e:
            print(f"模型测试失败: {e}")
        
        print("\n2. 测试数据加载...")
        if trainer.load_data():
            print("\n3. 测试模型构建...")
            if trainer.build_model():
                print("\n✓ 所有测试通过!")

    elif args.mode == "all":
        print("\n" + "=" * 70)
        print("开始完整训练流程")
        print("=" * 70)
        
        if trainer.load_data():
            if trainer.build_model():
                trainer.train()
                trainer.evaluate()

    elif args.mode == "ablation":
        print("\n" + "=" * 70)
        print("开始消融实验模式")
        print("=" * 70)
        
        model_path_to_use = args.model_path
        if model_path_to_use is None:
            possible_paths = [
                trainer.save_dir / "best_model.pth",
                trainer.save_dir / "final_model.pth",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path_to_use = path
                    print(f"找到模型文件: {model_path_to_use}")
                    break
        
        if trainer.load_data():
            if trainer.build_model():
                trainer.run_ablation_experiment(
                    model_path=model_path_to_use,
                    output_dir=trainer.save_dir / f"ablation_{timestamp}",
                    ablation_method=args.ablation_method,
                    retrain_epochs=args.retrain_epochs,
                )

    elif args.mode == "build_incremental_manifest":
        print("\n" + "=" * 70)
        print("准备固定152000增量样本清单（只构建，不训练）")
        print("=" * 70)
        if args.sampling_mode != "incremental":
            raise ValueError("build_incremental_manifest 模式必须配合 --sampling_mode incremental")
        trainer.config["build_incremental_manifest"] = True
        trainer.run_pretrain_cv_workflow(manifest_only=True)

    elif args.mode == "pretrain_cv":
        print("\n" + "=" * 70)
        print("开始预训练随机十折交叉验证")
        print("=" * 70)
        trainer.run_pretrain_cv_workflow()

    elif args.mode == "pretrain_spatial_cv":
        print("\n" + "=" * 70)
        print("开始预训练空间网格十折交叉验证")
        print("=" * 70)
        trainer.run_pretrain_spatial_cv()
        
    elif args.mode == "pretrain_progressive":
        print("\n" + "=" * 70)
        print("开始预训练渐进式训练")
        print("=" * 70)
        trainer.run_pretrain_progressive_from_cv()
                    
    elif args.mode == "fine_tune":
        if args.cv_mode == "station_cv":
            print("\n" + "=" * 70)
            print("开始按站点十折交叉验证模式（固定测试集）")
            print("=" * 70)

            if trainer.load_data(
                fine_tune_mode=True,
                mixed_mode=args.mixed_mode,
                station_ratio=args.station_ratio
            ):
                # BALANCED_STATION_CV10_V1
                # 正式微调必须按站点划分，并在每个fold训练结束后立即
                # 对该fold留出站点进行评估。
                trainer.config["freeze_strategy"] = args.freeze_strategy
                trainer.run_cv_workflow_by_station()

        elif args.cv_mode == "station_full_cv":
            print("\n" + "=" * 70)
            print("开始按站点全样本十折交叉验证模式")
            print("=" * 70)

            if trainer.load_data(
                fine_tune_mode=True,
                mixed_mode=args.mixed_mode,
                station_ratio=args.station_ratio
            ):
                trainer.run_station_full_cv()

        else:
            print("\n" + "=" * 70)
            print("开始标准微调模式")
            print("=" * 70)

            if trainer.load_data(
                fine_tune_mode=True,
                mixed_mode=args.mixed_mode,
                station_ratio=args.station_ratio
            ):
                if trainer.build_model(
                    load_pretrained=args.pretrained_model,
                    freeze_backbone=args.freeze_backbone,
                    freeze_strategy=args.freeze_strategy
                ):
                    trainer.train(fine_tune_mode=True)

    elif args.mode == "fine_tune_all":
        print("\n" + "=" * 70)
        print("开始完整微调流程")
        print("=" * 70)
        
        print("\n1. 训练基础模型...")
        if trainer.load_data():
            if trainer.build_model():
                trainer.train()
        
        print("\n2. 微调模型...")
        if trainer.load_data(
            fine_tune_mode=True,
            mixed_mode=args.mixed_mode,
            station_ratio=args.station_ratio
        ):
            if trainer.build_model(
                load_pretrained=args.pretrained_model,
                freeze_backbone=args.freeze_backbone,
                freeze_strategy=args.freeze_strategy
            ):
                trainer.train(fine_tune_mode=True)

    print("\n" + "=" * 70)
    print("程序执行完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
