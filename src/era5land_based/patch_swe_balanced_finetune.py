#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import py_compile
import shutil

ROOT = Path('/root/autodl-tmp')
target = ROOT / 'main_tune.py'

if not target.exists():
    raise FileNotFoundError(target)

required_markers = [
    'SWE_STRATIFIED_BATCH_SAMPLER_V1',
    'SIMPLIFIED_NSE_LOGCOSH_V1',
    'FINETUNE_HIGH_VALUE_METRICS_V1',
]

original = target.read_text(encoding='utf-8')
if all(marker in original for marker in required_markers):
    py_compile.compile(str(target), doraise=True)
    print('✅ main_tune.py 已包含本次损失/分层采样/复合选模修复')
    raise SystemExit(0)

text = original

# 1) imports
if 'import math\n' not in text:
    text = text.replace('import random\n', 'import random\nimport math\n', 1)

# 2) sampler class
sampler_marker = 'SWE_STRATIFIED_BATCH_SAMPLER_V1'
if sampler_marker not in text:
    anchor = '''\n    \nclass MixUp:\n'''
    sampler_code = r'''

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
'''
    if anchor not in text:
        raise RuntimeError('找不到MixUp插入锚点')
    text = text.replace(anchor, sampler_code + anchor, 1)

# 3) config defaults
config_marker = 'SIMPLIFIED_NSE_LOGCOSH_V1'
if config_marker not in text:
    old = '''            "pretrain_loss_weight": 0.0,  # ← 改为 0.0\n            "use_high_swe_weight": True,\n'''
    new = '''            "pretrain_loss_weight": 0.0,  # ← 改为 0.0\n            "use_high_swe_weight": True,\n\n            # SIMPLIFIED_NSE_LOGCOSH_V1\n            # 逐样本基础损失：MSE + Log-Cosh；再加高值Bias和轻量方差约束。\n            "finetune_mse_weight": 0.7,\n            "finetune_logcosh_weight": 0.3,\n            "finetune_high_bias_weight": 0.3,\n            "finetune_variance_weight": 0.02,\n            "finetune_target_std_ratio": 0.70,\n            "finetune_high_bias_threshold_mm": 80.0,\n            "finetune_high_bias_min_count": 2,\n\n            # 每个训练batch的SWE分层采样。\n            "use_swe_stratified_batches": True,\n            "swe_batch_bin_edges_mm": [20.0, 50.0, 80.0],\n            "swe_batch_bin_fractions": [0.55, 0.25, 0.125, 0.075],\n\n            # checkpoint复合评分（越小越好）。\n            "checkpoint_w_global_rmse": 1.0,\n            "checkpoint_w_rmse_ge50": 0.50,\n            "checkpoint_w_abs_bias_ge80": 0.30,\n            "checkpoint_w_slope": 0.10,\n            "checkpoint_w_std_ratio": 0.10,\n            "checkpoint_w_correlation": 0.05,\n'''
    if old not in text:
        raise RuntimeError('找不到config插入锚点')
    text = text.replace(old, new, 1)

# 4) helper methods in SWETrainer
method_marker = 'def _compute_simplified_nse_logcosh_loss('
if method_marker not in text:
    anchor = '''    def setup_chinese_fonts(self):\n'''
    methods = r'''    def _compute_simplified_nse_logcosh_loss(
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
        if bool(self.config.get("use_high_swe_weight", True)):
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
            target_std = targets_flat[metric_mask].std(unbiased=False)
            pred_std = outputs_flat[metric_mask].std(unbiased=False)
            target_std_ratio = float(
                self.config.get("finetune_target_std_ratio", 0.70)
            )
            variance_loss = torch.relu(
                target_std_ratio * target_std - pred_std
            ).pow(2)
        else:
            target_std = torch.zeros((), device=error.device, dtype=error.dtype)
            pred_std = torch.zeros((), device=error.device, dtype=error.dtype)
            variance_loss = torch.zeros((), device=error.device, dtype=error.dtype)

        high_bias_weight = float(
            self.config.get("finetune_high_bias_weight", 0.3)
        )
        variance_weight = float(
            self.config.get("finetune_variance_weight", 0.02)
        )

        loss = (
            base_loss
            + high_bias_weight * high_bias_loss
            + variance_weight * variance_loss
        )

        diagnostics = {
            "base_loss": base_loss.detach(),
            "high_bias_loss": high_bias_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "target_std": target_std.detach(),
            "pred_std": pred_std.detach(),
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

        common_kwargs = {
            "num_workers": self.config.get("num_workers", 8),
            "pin_memory": True,
        }

        if not use_stratified:
            return DataLoader(
                dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                drop_last=True,
                **common_kwargs,
            )

        sampler = SWEStratifiedBatchSampler(
            targets_mm=targets_mm,
            batch_size=self.config["batch_size"],
            seed=self.config.get("seed", 43),
            drop_last=True,
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
            "selection_score": float(selection_score),
        }

'''
    if anchor not in text:
        raise RuntimeError('找不到SWETrainer方法插入锚点')
    text = text.replace(anchor, methods + anchor, 1)

# 5) replace train loss block
old_train_loss = '''                    if is_fine_tune:\n                        outputs_flat = outputs_flat.reshape(-1)\n                        targets_flat = targets_flat.reshape(-1)\n\n                        swe_min = getattr(self, "swe_min", 0.0)\n                        swe_max = getattr(self, "swe_max", 170.0)\n                        target_mm = targets_flat * (swe_max - swe_min) + swe_min\n\n                        loss_each = F.smooth_l1_loss(\n                            outputs_flat,\n                            targets_flat,\n                            beta=0.01,\n                            reduction="none"\n                        )\n\n                        weights = torch.ones_like(targets_flat)\n\n                        # 只保留轻量高雪样本权重\n                        weights = weights + 1.0 * (target_mm >= 20.0).float()\n                        weights = weights + 2.0 * (target_mm >= 50.0).float()\n                        weights = weights + 3.0 * (target_mm >= 80.0).float()\n\n                        loss = (loss_each * weights).sum() / (weights.sum() + 1e-8)\n\n                        # 轻量方差约束\n                        if epoch < 15:\n                            target_var = targets_flat.var()\n                            pred_var = outputs_flat.var()\n                            if target_var > 1e-6 and pred_var / target_var < 0.5:\n                                variance_loss = torch.relu(0.5 * target_var - pred_var)\n                                loss = loss + 0.02 * variance_loss\n\n                    else:\n'''
new_train_loss = '''                    if is_fine_tune:\n                        # SIMPLIFIED_NSE_LOGCOSH_V1\n                        loss, loss_diag = self._compute_simplified_nse_logcosh_loss(\n                            outputs_flat,\n                            targets_flat,\n                            source_flag=source_flag,\n                        )\n\n                        if epoch == 0 and batch_idx == 0:\n                            print("    简化NSE-oriented微调损失已启用:")\n                            print(\n                                f"      base={float(loss_diag['base_loss']):.6f}, "\n                                f"high_bias={float(loss_diag['high_bias_loss']):.6f}, "\n                                f"variance={float(loss_diag['variance_loss']):.6f}, "\n                                f"high_count={loss_diag['high_count']}"\n                            )\n\n                    else:\n'''
if old_train_loss not in text:
    raise RuntimeError('找不到当前微调损失代码块')
text = text.replace(old_train_loss, new_train_loss, 1)

# 6) validation loss
old_val_loss = '''                if is_fine_tune:\n                    loss = smooth_l1_criterion(outputs.reshape(-1), targets.reshape(-1))\n                else:\n'''
new_val_loss = '''                if is_fine_tune:\n                    loss, _ = self._compute_simplified_nse_logcosh_loss(\n                        outputs.reshape(-1),\n                        targets.reshape(-1),\n                        source_flag=None,\n                    )\n                else:\n'''
if old_val_loss not in text:
    raise RuntimeError('找不到验证SmoothL1代码块')
text = text.replace(old_val_loss, new_val_loss, 1)

# 7) add selection metrics after r2 calculation, before detailed print
metric_anchor = '''        # ============ 微调专用：详细分析 ============\n        if is_fine_tune:\n'''
metric_insert = '''        # FINETUNE_HIGH_VALUE_METRICS_V1\n        finetune_extra_metrics = {}\n        if is_fine_tune:\n            finetune_extra_metrics = self._compute_finetune_selection_metrics(\n                all_predictions,\n                all_targets,\n            )\n\n        # ============ 微调专用：详细分析 ============\n        if is_fine_tune:\n'''
if metric_anchor not in text:
    raise RuntimeError('找不到验证指标插入锚点')
text = text.replace(metric_anchor, metric_insert, 1)

# 8) print extra metrics
print_anchor = '''            print(f"    NSE:   {r2:.4f}")\n\n            zero_count = np.sum(all_is_zero == 0)\n'''
print_new = '''            print(f"    NSE:   {r2:.4f}")\n            print(f"    RMSE(mm): {finetune_extra_metrics['rmse_mm']:.2f}")\n            print(\n                f"    obs>=50 RMSE(mm): "\n                f"{finetune_extra_metrics['rmse_ge50_mm']:.2f} "\n                f"(N={finetune_extra_metrics['n_ge50']})"\n            )\n            print(\n                f"    obs>=80 Bias(mm): "\n                f"{finetune_extra_metrics['bias_ge80_mm']:.2f} "\n                f"(N={finetune_extra_metrics['n_ge80']})"\n            )\n            print(f"    回归斜率: {finetune_extra_metrics['slope']:.4f}")\n            print(\n                f"    std(pred)/std(obs): "\n                f"{finetune_extra_metrics['std_ratio']:.4f}"\n            )\n            print(\n                f"    checkpoint selection score: "\n                f"{finetune_extra_metrics['selection_score']:.6f}"\n            )\n\n            zero_count = np.sum(all_is_zero == 0)\n'''
if print_anchor not in text:
    raise RuntimeError('找不到验证详细打印锚点')
text = text.replace(print_anchor, print_new, 1)

# 9) metrics dict add extra
old_metrics = '''        metrics = {\n            "loss": avg_loss,\n            "rmse": rmse,\n            "mae": mae,\n            "correlation": correlation,\n            "r2": r2,\n            "n_samples": len(all_predictions),\n        }\n\n        return metrics\n'''
new_metrics = '''        metrics = {\n            "loss": avg_loss,\n            "rmse": rmse,\n            "mae": mae,\n            "correlation": correlation,\n            "r2": r2,\n            "n_samples": len(all_predictions),\n        }\n        if is_fine_tune:\n            metrics.update(finetune_extra_metrics)\n\n        return metrics\n'''
if old_metrics not in text:
    raise RuntimeError('找不到metrics返回块')
text = text.replace(old_metrics, new_metrics, 1)

# 10) stratified loader in station CV
old_cv_loader = '''            self.train_loader = DataLoader(\n                train_dataset,\n                batch_size=self.config["batch_size"],\n                shuffle=True,\n                num_workers=self.config.get("num_workers", 8),\n                pin_memory=True,\n                drop_last=True,\n            )\n'''
new_cv_loader = '''            if not is_mixed:\n                train_targets_mm = [\n                    float(station_ds.meta_index[int(i)]["swe"])\n                    for i in split["train_station_indices"]\n                ]\n                self.train_loader = self._make_swe_stratified_train_loader(\n                    train_dataset,\n                    train_targets_mm,\n                    context=f"Fold {fold_idx}",\n                )\n            else:\n                # mixed模式包含伪标签样本，暂时保持原随机shuffle，\n                # 避免把站点SWE配额错误应用到ConcatDataset的伪标签部分。\n                self.train_loader = DataLoader(\n                    train_dataset,\n                    batch_size=self.config["batch_size"],\n                    shuffle=True,\n                    num_workers=self.config.get("num_workers", 8),\n                    pin_memory=True,\n                    drop_last=True,\n                )\n'''
if old_cv_loader not in text:
    raise RuntimeError('找不到station CV训练DataLoader')
text = text.replace(old_cv_loader, new_cv_loader, 1)

# 11) custom split loader
old_custom_loader = '''                    self.train_loader = DataLoader(\n                        dataset_train,\n                        batch_size=self.config["batch_size"],\n                        shuffle=True,\n                        num_workers=self.config.get("num_workers", 10),\n                        pin_memory=True,\n                        drop_last=True,\n                    )\n'''
new_custom_loader = '''                    train_targets_mm = [\n                        float(meta["swe"])\n                        for meta in dataset_train.meta_index\n                    ]\n                    self.train_loader = self._make_swe_stratified_train_loader(\n                        dataset_train,\n                        train_targets_mm,\n                        context="custom split",\n                    )\n'''
if old_custom_loader in text:
    text = text.replace(old_custom_loader, new_custom_loader, 1)

# 12) best selection init
old_best_init = '''        best_val_loss = float("inf")\n        best_val_r2 = -float("inf")\n        patience_counter = 0\n'''
new_best_init = '''        best_val_loss = float("inf")\n        best_val_r2 = -float("inf")\n        best_selection_score = float("inf")\n        patience_counter = 0\n'''
if old_best_init not in text:
    raise RuntimeError('找不到best初始化')
text = text.replace(old_best_init, new_best_init, 1)

# 13) scheduler use selection score for finetune
old_scheduler = '''                elif need_validate:\n                    self.scheduler.step(\n                        val_metrics["loss"]\n                    )\n'''
new_scheduler = '''                elif need_validate:\n                    scheduler_metric = (\n                        val_metrics.get("selection_score", val_metrics["loss"])\n                        if fine_tune_mode\n                        else val_metrics["loss"]\n                    )\n                    self.scheduler.step(scheduler_metric)\n'''
if old_scheduler not in text:
    raise RuntimeError('找不到scheduler验证更新块')
text = text.replace(old_scheduler, new_scheduler, 1)

# 14) verbose epoch print extra
old_epoch_print = '''                    print(f"  验证相关系数: {val_metrics['correlation']:.4f}")\n                print(f"  学习率:    {current_lr:.2e}")\n'''
new_epoch_print = '''                    print(f"  验证相关系数: {val_metrics['correlation']:.4f}")\n                    if fine_tune_mode:\n                        print(f"  obs>=50 RMSE: {val_metrics.get('rmse_ge50_mm', float('nan')):.2f} mm")\n                        print(f"  obs>=80 Bias: {val_metrics.get('bias_ge80_mm', float('nan')):.2f} mm")\n                        print(f"  回归斜率:    {val_metrics.get('slope', float('nan')):.4f}")\n                        print(f"  Std Ratio:   {val_metrics.get('std_ratio', float('nan')):.4f}")\n                        print(f"  选模评分:     {val_metrics.get('selection_score', float('nan')):.6f}")\n                print(f"  学习率:    {current_lr:.2e}")\n'''
if old_epoch_print not in text:
    raise RuntimeError('找不到epoch打印块')
text = text.replace(old_epoch_print, new_epoch_print, 1)

# 15) checkpoint selection logic
old_selection = '''            elif need_validate:\n                is_best_by_loss = val_metrics["loss"] < best_val_loss\n                is_best_by_r2 = 'r2' in val_metrics and val_metrics['r2'] > best_val_r2\n\n                if is_best_by_loss or is_best_by_r2:\n                    if is_best_by_loss:\n                        best_val_loss = val_metrics["loss"]\n                        if verbose:\n                            print(f"  🎉 新的最佳验证损失: {best_val_loss:.6f}")\n                    if is_best_by_r2:\n                        best_val_r2 = val_metrics['r2']\n                        best_val_r = val_metrics.get('correlation', 0)\n                        if verbose:\n                            print(f"  🎉 新的最佳 r: {best_val_r:.4f}")\n                    best_epoch = epoch\n                    patience_counter = 0\n                    model_name = "best_fine_tuned_model.pth" if fine_tune_mode else "best_model.pth"\n                    self.save_checkpoint(model_name, epoch, val_metrics)\n                    if verbose:\n                        print(f"\\n💾 保存最佳{mode}模型: {model_name} (Epoch {epoch+1}, val_loss={val_metrics['loss']:.6f})")\n                else:\n                    patience_counter += 1\n                    if verbose:\n                        print(f"\\n⏳ 连续 {patience_counter} 轮未改善最佳指标")\n'''
new_selection = '''            elif need_validate:\n                if fine_tune_mode:\n                    current_selection_score = float(\n                        val_metrics.get("selection_score", float("inf"))\n                    )\n                    is_better = current_selection_score < best_selection_score\n\n                    if is_better:\n                        best_selection_score = current_selection_score\n                        best_val_loss = float(val_metrics["loss"])\n                        best_val_r2 = float(val_metrics.get("r2", -float("inf")))\n                        best_val_r = float(val_metrics.get("correlation", 0.0))\n                        best_epoch = epoch\n                        patience_counter = 0\n                        model_name = "best_fine_tuned_model.pth"\n                        self.save_checkpoint(model_name, epoch, val_metrics)\n                        print(\n                            f"\\n💾 保存最佳微调模型: {model_name} "\n                            f"(Epoch {epoch + 1}, "\n                            f"selection_score={best_selection_score:.6f}, "\n                            f"RMSE50={val_metrics.get('rmse_ge50_mm', float('nan')):.2f} mm, "\n                            f"Bias80={val_metrics.get('bias_ge80_mm', float('nan')):.2f} mm, "\n                            f"slope={val_metrics.get('slope', float('nan')):.3f}, "\n                            f"std_ratio={val_metrics.get('std_ratio', float('nan')):.3f})"\n                        )\n                    else:\n                        patience_counter += 1\n                        if verbose:\n                            print(\n                                f"\\n⏳ 连续 {patience_counter} 轮未改善复合选模评分 "\n                                f"(current={current_selection_score:.6f}, "\n                                f"best={best_selection_score:.6f})"\n                            )\n                else:\n                    is_best_by_loss = val_metrics["loss"] < best_val_loss\n                    is_best_by_r2 = 'r2' in val_metrics and val_metrics['r2'] > best_val_r2\n\n                    if is_best_by_loss or is_best_by_r2:\n                        if is_best_by_loss:\n                            best_val_loss = val_metrics["loss"]\n                            if verbose:\n                                print(f"  🎉 新的最佳验证损失: {best_val_loss:.6f}")\n                        if is_best_by_r2:\n                            best_val_r2 = val_metrics['r2']\n                            best_val_r = val_metrics.get('correlation', 0)\n                            if verbose:\n                                print(f"  🎉 新的最佳 r: {best_val_r:.4f}")\n                        best_epoch = epoch\n                        patience_counter = 0\n                        model_name = "best_model.pth"\n                        self.save_checkpoint(model_name, epoch, val_metrics)\n                        if verbose:\n                            print(f"\\n💾 保存最佳{mode}模型: {model_name} (Epoch {epoch+1}, val_loss={val_metrics['loss']:.6f})")\n                    else:\n                        patience_counter += 1\n                        if verbose:\n                            print(f"\\n⏳ 连续 {patience_counter} 轮未改善最佳指标")\n'''
if old_selection not in text:
    raise RuntimeError('找不到checkpoint选模代码块')
text = text.replace(old_selection, new_selection, 1)

# 16) summary/return include score
old_cv_summary = '''        else:\n            # 交叉验证时简化输出\n            print(f"  ✅ 训练完成: best_val_loss={best_val_loss:.6f}, best_r2={best_val_r2:.4f}")\n'''
new_cv_summary = '''        else:\n            # 交叉验证时简化输出\n            if fine_tune_mode:\n                print(\n                    f"  ✅ 训练完成: best_selection_score={best_selection_score:.6f}, "\n                    f"best_val_loss={best_val_loss:.6f}, best_nse={best_val_r2:.4f}"\n                )\n            else:\n                print(f"  ✅ 训练完成: best_val_loss={best_val_loss:.6f}, best_r2={best_val_r2:.4f}")\n'''
if old_cv_summary not in text:
    raise RuntimeError('找不到CV summary块')
text = text.replace(old_cv_summary, new_cv_summary, 1)

old_return = '''        return {\n            "best_val_loss": best_val_loss,\n            "best_val_r2": best_val_r2 if hasattr(self, 'val_history_metrics') else 0,\n            "best_epoch": best_epoch,\n            "total_epochs": epoch + 1\n        }\n'''
new_return = '''        return {\n            "best_val_loss": best_val_loss,\n            "best_val_r2": best_val_r2 if hasattr(self, 'val_history_metrics') else 0,\n            "best_selection_score": best_selection_score if fine_tune_mode else None,\n            "best_epoch": best_epoch,\n            "total_epochs": epoch + 1\n        }\n'''
if old_return not in text:
    raise RuntimeError('找不到train return块')
text = text.replace(old_return, new_return, 1)

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir = ROOT / 'code_backups' / f'before_swe_balanced_finetune_fix_{stamp}'
backup_dir.mkdir(parents=True, exist_ok=False)
backup_path = backup_dir / target.name
shutil.copy2(target, backup_path)

target.write_text(text, encoding='utf-8')
py_compile.compile(str(target), doraise=True)

print('✅ main_tune.py 已完成微调目标修复')
print(f'   backup={backup_path}')
print(f'   target={target}')
print('   已启用:')
print('     1) 0.7*MSE + 0.3*Log-Cosh + high-bias + variance')
print('     2) SWE分层batch采样（默认约18/8/4/2）')
print('     3) 每epoch高值指标与复合checkpoint评分')
