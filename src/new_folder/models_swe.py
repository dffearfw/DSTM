"""
SWE反演模型 - 空间特征+点特征融合
支持7x7 patch的空间编码器，集成ConvNeXt风格和Attention Residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ============= 辅助模块 =============
class GlobalAttentionPool(nn.Module):
    """全局注意力池化 - 比简单平均池化更好"""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        attn = self.attention(x)
        x_weighted = x * attn
        x_pooled = self.gap(x_weighted)
        return x_pooled.flatten(1)


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvNeXtBlockWithAttnRes(nn.Module):
    """
    ConvNeXt块 + Attention Residuals (Kimi 2026)
    让每一层能动态选择性地聚合前序层的输出
    """
    def __init__(self, dim, layer_scale_init_value=1e-6, use_attn_res=True, kernel_size=7):
        super().__init__()
        self.use_attn_res = use_attn_res
        
        # 深度可分离卷积（大核，感受野足够覆盖7x7或4x4）
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        
        # 两层MLP（类似Transformer的FFN）
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 可学习的层缩放参数
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        
        # Attention Residuals 组件
        if use_attn_res:
            self.query_proj = nn.Linear(dim, dim)
            self.key_proj = nn.Linear(dim, dim)
    
    def forward(self, x, history_outputs: Optional[List[torch.Tensor]] = None):
        residual = x
        B, C, H, W = x.shape
        
        # 1. 深度卷积
        x = self.dwconv(x)
        
        # 2. 通道维度转换到最后一维
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        
        # 3. MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 4. 层缩放
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # 5. Attention Residuals
        if self.use_attn_res and history_outputs is not None and len(history_outputs) > 0:
            # 堆叠历史输出
            stacked_history = torch.stack(history_outputs, dim=0)  # (L, B, C, H, W)
            L = stacked_history.shape[0]
            
            # 重塑历史输出: (B, L, H*W, C)
            # (L, B, C, H, W) -> (B, L, C, H, W) -> (B, L, C, H*W) -> (B, L, H*W, C)
            history_flat = stacked_history.permute(1, 0, 2, 3, 4)  # (B, L, C, H, W)
            history_flat = history_flat.flatten(3)  # (B, L, C, H*W)
            history_flat = history_flat.permute(0, 1, 3, 2)  # (B, L, H*W, C)
            
            # Query: 当前残差分支 (B, H*W, C)
            query = residual.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
            query = self.query_proj(query)  # (B, H*W, C)
            
            # Key: 历史输出投影 (B, L, H*W, C)
            keys = self.key_proj(history_flat)  # (B, L, H*W, C)
            
            # 在空间维度上聚合
            history_pooled = history_flat.mean(dim=2)  # (B, L, C)
            query_pooled = query.mean(dim=1, keepdim=True)  # (B, 1, C)
            
            # 计算注意力权重 (B, 1, L)
            attn_weights = torch.matmul(query_pooled, history_pooled.transpose(1, 2)) / (C ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)  # (B, 1, L)
            
            # 加权聚合历史输出 (B, C)
            aggregated_history = torch.matmul(attn_weights, history_pooled).squeeze(1)  # (B, C)
            
            # 扩展回空间维度 (B, C, H, W)
            aggregated_history = aggregated_history.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            x = x + aggregated_history
        
        # 标准残差
        x = residual + x
        
        # 更新历史输出列表
        new_history = (history_outputs or []) + [x.detach()]
        
        return x, new_history
# ============= 空间编码器（适配7x7 patch） =============
class HighPerformanceSpatialEncoder_7x7(nn.Module):
    """
    适配7x7 patch的高性能空间编码器（ConvNeXt风格 + Attention Residuals）
    输入: (B, C_spatial, 7, 7)
    输出: (B, d_model)
    """
    def __init__(self, C_spatial, d_model, use_attn_res=True, kernel_size=7, num_blocks=6):
        """
        Args:
            C_spatial: 输入通道数
            d_model: 输出维度
            use_attn_res: 是否使用Attention Residuals
            kernel_size: ConvNeXt块中的深度卷积核大小（7或5）
            num_blocks: ConvNeXt块数量
        """
        super().__init__()
        self.use_attn_res = use_attn_res
        
        # Stem: 7x7 -> 4x4 (单次下采样，保留空间结构)
        self.stem = nn.Sequential(
            nn.Conv2d(C_spatial, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 7 -> 4
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        # 构建ConvNeXt块序列（所有块都在4x4特征图上操作）
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                ConvNeXtBlockWithAttnRes(
                    dim=128,
                    use_attn_res=use_attn_res,
                    kernel_size=kernel_size
                )
            )
        
        # SE模块增强通道注意力
        self.se_module = SEModule(128, reduction=8)
        
        # 全局注意力池化（替代进一步下采样）
        self.global_attention = GlobalAttentionPool(128)
        
        # 最终投影到d_model
        self.final_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem: (B, C, 7, 7) -> (B, 128, 4, 4)
        x = self.stem(x)
        
        # 历史输出列表
        history = []
        
        # 逐块前向传播
        for block in self.blocks:
            x, history = block(x, history if self.use_attn_res else None)
        
        # SE增强
        x = self.se_module(x)
        
        # 全局注意力池化
        x = self.global_attention(x)  # (B, 128)
        
        # 投影到d_model
        x = self.final_proj(x)  # (B, d_model)
        
        return x


# ============= 兼容原版的空间编码器（适用于较大patch） =============
class SpatialEncoder(nn.Module):
    """
    原版空间编码器（用于兼容性，适用于较大patch）
    输入: (B, C_spatial, P, P)  # P >= 16
    输出: (B, d_model)
    """
    def __init__(self, C_spatial, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_spatial, 64, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, d_model) if d_model != 128 else nn.Identity()
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


# ============= 点特征编码器 =============
class PointEncoder(nn.Module):
    """
    点特征编码器
    输入: point_feats (B, C_point)
    输出: point_vec (B, d_point)
    """
    def __init__(self, C_point, d_point=256, dropout=0.1):
        super().__init__()
        
        print(f"  点编码器: 输入维度={C_point}, 输出维度={d_point}")
        
        self.mlp = nn.Sequential(
            nn.Linear(C_point, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, d_point),
        )
        self.out_dim = d_point
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.mlp(x)  # (B, d_point)


# ============= Transformer融合模块 =============
class FusionTransformer(nn.Module):
    """
    Transformer融合模块 + Wide & Deep 架构
    输入: spatial_vec (B, d_model), point_vec (B, d_model), raw_product_val (B, 1)
    输出: y_pred (B,)
    """
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            max_len=10,
            use_wide_branch=True  # 新增：是否使用 Wide 分支
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_wide_branch = use_wide_branch
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        # Transformer编码器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 位置编码
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
        
        # ============ Wide & Deep: 回归头 ============
        if use_wide_branch:
            # Deep 分支输入: d_model (Transformer输出)
            # Wide 分支输入: 1 (原产品值)
            # 拼接后: d_model + 1
            self.head = nn.Sequential(
                nn.Linear(d_model + 1, 128),  # ← 关键修改：d_model + 1
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
            print(f"  Wide & Deep: 输入维度={d_model + 1} (Transformer {d_model} + 产品值 1)")
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
        
        self._init_weights()
        
        print(f"  Transformer: d_model={d_model}, heads={nhead}, layers={num_layers}")
        print(f"  Wide & Deep: use_wide_branch={use_wide_branch}")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, spatial_vec, point_vec, raw_product_val=None):
        """
        Args:
            spatial_vec: (B, d_model) or None
            point_vec: (B, d_model) or None
            raw_product_val: (B, 1) 原产品值（归一化后），用于 Wide 分支
        """
        # 确定批次大小
        B = None
        if spatial_vec is not None:
            B = spatial_vec.shape[0]
        elif point_vec is not None:
            B = point_vec.shape[0]
        else:
            raise ValueError("至少需要一个特征输入")
        
        # 检查维度并投影
        if spatial_vec is not None and spatial_vec.shape[1] != self.d_model:
            if not hasattr(self, 'spatial_proj'):
                self.spatial_proj = nn.Linear(spatial_vec.shape[1], self.d_model).to(spatial_vec.device)
                nn.init.xavier_uniform_(self.spatial_proj.weight)
                if self.spatial_proj.bias is not None:
                    nn.init.zeros_(self.spatial_proj.bias)
            spatial_vec = self.spatial_proj(spatial_vec)
        
        if point_vec is not None and point_vec.shape[1] != self.d_model:
            if not hasattr(self, 'point_proj'):
                self.point_proj = nn.Linear(point_vec.shape[1], self.d_model).to(point_vec.device)
                nn.init.xavier_uniform_(self.point_proj.weight)
                if self.point_proj.bias is not None:
                    nn.init.zeros_(self.point_proj.bias)
            point_vec = self.point_proj(point_vec)
        
        # 构建token序列
        tokens = []
        tokens.append(self.cls_token.expand(B, -1, -1))
        
        if spatial_vec is not None:
            spatial_token = spatial_vec.unsqueeze(1)
            tokens.append(spatial_token)
        
        if point_vec is not None:
            point_token = point_vec.unsqueeze(1)
            tokens.append(point_token)
        
        x = torch.cat(tokens, dim=1)
        
        # 添加位置编码
        L = x.shape[1]
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)
        x = self.norm(x)
        x = self.encoder(x)
        cls_output = x[:, 0, :]  # (B, d_model)
        
        # ============ Wide & Deep: 拼接原产品值 ============
        if self.use_wide_branch and raw_product_val is not None:
            # 确保 raw_product_val 形状正确
            if raw_product_val.dim() == 1:
                raw_product_val = raw_product_val.unsqueeze(1)
            # 拼接 Deep 特征和 Wide 特征
            final_input = torch.cat([cls_output, raw_product_val], dim=-1)  # (B, d_model + 1)
            y_pred = self.head(final_input).squeeze(-1)
        else:
            y_pred = self.head(cls_output).squeeze(-1)
        
        return y_pred


# ============= 完整SWE反演模型 =============
class SWENet(nn.Module):
    """
    完整SWE反演模型 + Wide & Deep 架构
    输入:
        spatial_patch: (B, C_spatial, P, P)  # 卷积特征，P=7
        point_feats: (B, C_point)           # 点特征（21维，第21维是原产品值）
    输出:
        y_pred: (B,) 标准化后的SWE值
    """
    def __init__(
            self,
            C_spatial=7,
            C_point=21,  
            d_model=256,
            use_spatial=True,
            use_point=True,
            use_attn_res=True,
            spatial_encoder_type='high_performance',
            use_wide_branch=True
    ):
        super().__init__()
        
        self.use_spatial = use_spatial
        self.use_point = use_point
        self.d_model = d_model
        self.use_wide_branch = use_wide_branch
        self.C_point = C_point
        
        print(f"\n[SWENet模型初始化]")
        print(f"  输入维度: C_spatial={C_spatial}, C_point={C_point}")
        print(f"  特征开关: 空间={use_spatial}, 点={use_point}")
        print(f"  模型维度: d_model={d_model}")
        print(f"  空间编码器类型: {spatial_encoder_type}")
        print(f"  Attention Residuals: {use_attn_res}")
        print(f"  Wide & Deep: {use_wide_branch}")
        
        # 1) 空间编码器
        if use_spatial:
            if spatial_encoder_type == 'high_performance':
                self.spatial_encoder = HighPerformanceSpatialEncoder_7x7(
                    C_spatial=C_spatial,
                    d_model=d_model,
                    use_attn_res=use_attn_res,
                    kernel_size=7,
                    num_blocks=6
                )
            else:
                self.spatial_encoder = SpatialEncoder(
                    C_spatial=C_spatial,
                    d_model=d_model
                )
        else:
            self.spatial_encoder = None
        

        if use_point:

            point_encoder_input_dim = C_point - 1 if use_wide_branch else C_point
            self.point_encoder = PointEncoder(
                C_point=point_encoder_input_dim,
                d_point=d_model,
                dropout=0.1
            )
        else:
            self.point_encoder = None
        
        # 3) 计算最大序列长度
        max_len = 1  # [CLS] token
        if use_spatial:
            max_len += 1
        if use_point:
            max_len += 1
        
        print(f"  Transformer最大序列长度: {max_len}")
        
        # 4) Transformer融合 + Wide & Deep
        self.fusion_transformer = FusionTransformer(
            d_model=d_model,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            max_len=max_len,
            use_wide_branch=use_wide_branch
        )
        
        # 注意：不在这里统一初始化，因为各个子模块已经各自初始化了
        # 只初始化可能遗漏的层
        self._init_missing_weights()
    
    def _init_missing_weights(self):
        """初始化可能遗漏的权重（主要是线性层和BN层）"""
        for m in self.modules():
            if isinstance(m, nn.Linear) and not hasattr(m, '_initialized'):
                # 只初始化没有被子模块初始化的线性层
                if not hasattr(m, 'weight') or m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    m._initialized = True
            elif isinstance(m, nn.BatchNorm2d) and not hasattr(m, '_initialized'):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m._initialized = True
            elif isinstance(m, nn.LayerNorm) and not hasattr(m, '_initialized'):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                m._initialized = True
    
    def forward(self, spatial_patch, point_feats):
        # 初始化特征向量
        spatial_vec = None
        point_vec = None
        raw_product_val = None
        
        # A. 空间编码
        if self.use_spatial and spatial_patch is not None:
            spatial_vec = self.spatial_encoder(spatial_patch)  # (B, d_model)
        elif self.use_spatial:
            raise ValueError("模型配置使用空间特征，但未提供spatial_patch")
        
        # B. 点编码（分离：前15维走 Deep 分支，第16维走 Wide 分支）
        if self.use_point and point_feats is not None:
            if self.use_wide_branch and point_feats.shape[1] >= self.C_point:
                # 前15维用于 Deep 分支（经过点编码器）
                point_features_deep = point_feats[:, :self.C_point - 1]
                point_vec = self.point_encoder(point_features_deep)  # (B, d_model)
                
                # 第16维（原产品值）用于 Wide 分支，直接传给 FusionTransformer
                raw_product_val = point_feats[:, self.C_point - 1:self.C_point]  # (B, 1)
            else:
                # 不使用 Wide 分支时，所有维度都经过点编码器
                point_vec = self.point_encoder(point_feats)
        elif self.use_point:
            raise ValueError("模型配置使用点特征，但未提供point_feats")
        
        # C. Transformer融合 + Wide & Deep
        y_pred = self.fusion_transformer(spatial_vec, point_vec, raw_product_val)
        
        return y_pred

    
    
#===========================门控=============================

class SpatiallyGatedSWENet(nn.Module):
    def __init__(self, pretrained_model, C_point=15, d_model=256):
        super().__init__()
        
        # 1. 专家 A：预训练模型（完全冻结）
        self.pretrained_model = pretrained_model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
            
        # 2. 专家 B：微调网络（MLP，用于纠偏和捕捉站点实测分布）
        # 输入 15 维点特征，输出 1 维 SWE
        self.fine_tune_net = nn.Sequential(
            nn.Linear(C_point, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # 3. 裁判：门控网络（环境感知路由）
        # 输入：点特征 (15) + 空间特征向量 (d_model, 通常是256)
        self.gate = nn.Sequential(
            nn.Linear(C_point + d_model, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 alpha，代表对预训练模型的信任度
        )
        
    def forward(self, conv_feats, point_feats):
        # A. 提取预训练专家的意见和空间背景
        with torch.no_grad():
            # 这里的输出假设为 (B,)
            y_pre = self.pretrained_model(conv_feats, point_feats)
            # 提取空间编码器生成的 256 维环境向量
            # 需确保你的 SWENet 暴露了空间编码器接口
            spatial_context = self.pretrained_model.spatial_encoder(conv_feats) # (B, 256)
            
        # B. 提取微调专家的意见
        y_fine = self.fine_tune_net(point_feats).squeeze(1) # (B,)
        
        # C. 门控决策：结合当前点特征和空间环境，决定信任谁
        gate_input = torch.cat([point_feats, spatial_context], dim=1) # (B, 15 + 256)
        alpha = self.gate(gate_input).squeeze(1) # (B,)
        
        # D. 动态加权融合
        # alpha 越大，越听预训练的；alpha 越小，越听微调的
        y_final = alpha * y_pre + (1.0 - alpha) * y_fine
        
        # 训练模式下返回所有组件用于计算结果导向的 Loss
        if self.training:
            return y_final, y_pre, y_fine, alpha
        return y_final    

# ============= 残差SWE模型 =============
class ResidualInjectionSWENet(nn.Module):
    def __init__(self, pretrained_model, C_point=15, d_model=256):
        super().__init__()
        # 1. 冻结预训练模型主干（保留其提取空间环境的能力）
        self.backbone = pretrained_model
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # 2. 纠偏专家 (MLP)：专门学习产品与实测的差值
        # 输入：点特征 + 空间上下文 + 产品值本身
        self.correction_mlp = nn.Sequential(
            nn.Linear(C_point + d_model + 1, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1) # 输出 delta_y
        )

    def forward(self, conv_feats, point_feats, raw_fused_swe):
        # A. 提取空间环境特征（利用预训练好的编码器）
        with torch.no_grad():
            spatial_context = self.backbone.spatial_encoder(conv_feats)
        
        # B. 拼接纠偏特征：环境 + 点特征 + 此时的产品值
        # 让 MLP 知道产品现在报了多少，从而决定要补多少
        correction_input = torch.cat([
            point_feats, 
            spatial_context, 
            raw_fused_swe.unsqueeze(1)
        ], dim=1)
        
        # C. 计算残差 (delta)
        delta_y = self.correction_mlp(correction_input).squeeze(1)
        
        # D. 注入：最终预测 = 原始产品 + 纠偏补丁
        y_final = raw_fused_swe + delta_y
        
        # 为了 Loss 约束，返回最终值和残差值
        return y_final, delta_y

# ============= 模型工厂函数 =============
def create_model(model_type, use_wide_branch=True, **kwargs):
    """
    根据类型创建模型
    
    参数:
        model_type: 模型类型
            "full" - 完整模型（空间+点）
            "spatial_only" - 仅空间
            "point_only" - 仅点特征
        use_wide_branch: 是否使用 Wide & Deep 架构
        **kwargs: 传递给SWENet的参数
    """
    type_config = {
        "full": {"use_spatial": True, "use_point": True},
        "spatial_only": {"use_spatial": True, "use_point": False},
        "point_only": {"use_spatial": False, "use_point": True},
    }
    
    if model_type not in type_config:
        raise ValueError(f"未知模型类型: {model_type}。可选: {list(type_config.keys())}")
    
    config = type_config[model_type]
    config['use_wide_branch'] = use_wide_branch
    all_config = {**config, **kwargs}
    
    return SWENet(**all_config)

def convert_to_gated_model(pretrained_model_path, device, C_point=15, d_model=256):
    """从本地路径加载预训练模型并包装成门控模型"""
    from models_swe import create_model # 确保引用路径正确
    
    # 1. 创建并加载基础模型
    base_model = create_model("full", C_spatial=7, C_point=C_point, d_model=d_model)
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    # 自动处理 state_dict 键名
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    base_model.load_state_dict(state_dict, strict=False)
    
    # 2. 包装
    gated_model = SpatiallyGatedSWENet(base_model, C_point=C_point, d_model=d_model)
    return gated_model.to(device)


def create_high_performance_model(**kwargs):
    """创建高性能模型（使用Attention Residuals）"""
    return create_model("full", spatial_encoder_type='high_performance', use_attn_res=True, **kwargs)


def create_basic_model(**kwargs):
    """创建基础模型"""
    return create_model("full", spatial_encoder_type='basic', use_attn_res=False, **kwargs)


# ============= 快捷方式函数 =============
def SWENet_Full(**kwargs):
    """完整模型（基础版）"""
    return create_model("full", spatial_encoder_type='basic', use_attn_res=False, **kwargs)


def SWENet_SpatialOnly(**kwargs):
    """仅空间模型"""
    return create_model("spatial_only", spatial_encoder_type='basic', **kwargs)


def SWENet_PointOnly(**kwargs):
    """仅点特征模型"""
    return create_model("point_only", **kwargs)


def SWENet_HighPerformance(**kwargs):
    """高性能模型（ConvNeXt + Attention Residuals）"""
    return create_high_performance_model(**kwargs)



# ============= 测试函数 =============
def test_model():
    """测试模型"""
    print("=" * 60)
    print("测试SWE模型...")
    print("=" * 60)
    
    # 测试参数
    batch_size = 4
    C_spatial = 7
    C_point = 13
    d_model = 256
    patch_size = 7  # 7x7 patch
    
    # 创建测试数据
    spatial_test = torch.randn(batch_size, C_spatial, patch_size, patch_size)
    point_test = torch.randn(batch_size, C_point)
    
    # ========== 测试1: 基础完整模型 ==========
    print("\n1. 测试基础完整模型:")
    model_basic = SWENet_Full(
        C_spatial=C_spatial,
        C_point=C_point,
        d_model=d_model
    )
    
    with torch.no_grad():
        output_basic = model_basic(spatial_test, point_test)
    
    print(f"  输入形状: spatial={spatial_test.shape}, point={point_test.shape}")
    print(f"  输出形状: {output_basic.shape}")
    print(f"  输出示例: {output_basic[:3]}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model_basic.parameters())
    trainable_params = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # ========== 测试2: 高性能模型 ==========
    print("\n2. 测试高性能模型 (ConvNeXt + Attention Residuals):")
    model_high = SWENet_HighPerformance(
        C_spatial=C_spatial,
        C_point=C_point,
        d_model=d_model,
        use_attn_res=True
    )
    
    with torch.no_grad():
        output_high = model_high(spatial_test, point_test)
    
    print(f"  输出形状: {output_high.shape}")
    
    # 参数统计
    total_params_high = sum(p.numel() for p in model_high.parameters())
    trainable_params_high = sum(p.numel() for p in model_high.parameters() if p.requires_grad)
    print(f"  总参数: {total_params_high:,}")
    print(f"  可训练参数: {trainable_params_high:,}")
    
    # ========== 测试3: 残差模型 ==========
    print("\n3. 测试残差模型:")
    model_residual = ResidualSWENet(
        pretrained_model=model_basic,
        C_point=C_point,
        d_model=d_model
    )
    
    with torch.no_grad():
        output_residual = model_residual(spatial_test, point_test)
    
    print(f"  输出形状: {output_residual.shape}")
    
    # ========== 测试4: 简化模型 ==========
    print("\n4. 测试仅空间模型:")
    model_spatial = SWENet_SpatialOnly(
        C_spatial=C_spatial,
        d_model=d_model
    )
    
    with torch.no_grad():
        output_spatial = model_spatial(spatial_test, None)
    
    print(f"  输出形状: {output_spatial.shape}")
    
    print("\n5. 测试仅点特征模型:")
    model_point = SWENet_PointOnly(
        C_point=C_point,
        d_model=d_model
    )
    
    with torch.no_grad():
        output_point = model_point(None, point_test)
    
    print(f"  输出形状: {output_point.shape}")
    
    print("\n" + "=" * 60)
    print("✓ 所有模型测试完成!")
    print("=" * 60)
    
    # 返回模型供进一步使用
    return {
        'basic': model_basic,
        'high_performance': model_high,
        'residual': model_residual
    }


if __name__ == "__main__":
    models = test_model()