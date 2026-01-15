# models_swe.py
import torch
import torch.nn as nn


# ============= 空间编码器（卷积特征） =============
class SpatialEncoder(nn.Module):
    """
    输入: spatial_patch (B, C_spatial, P, P)
    输出: spatial_vec (B, d_spatial)

    卷积特征：chelsa_sfxwind, lst, rh, clamday, dem
    """

    def __init__(self, C_spatial, d_spatial=256):
        super().__init__()

        print(f"  空间编码器: 输入通道={C_spatial}, 输出维度={d_spatial}")

        # 简化卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(C_spatial, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, d_spatial)
        self.out_dim = d_spatial

    def forward(self, x):
        # x: (B, C_spatial, P, P)
        h = self.conv(x)  # (B, 128, P, P)
        h = self.pool(h)  # (B, 128, 1, 1)
        h = h.flatten(1)  # (B, 128)
        h = self.proj(h)  # (B, d_spatial)
        return h


# ============= 点特征编码器 =============
class PointEncoder(nn.Module):
    """
    输入: point_feats (B, C_point)
    输出: point_vec (B, d_point)

    点特征：ls, 经纬度, doy等
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

    def forward(self, x):
        return self.mlp(x)


# ============= Transformer 融合模块 =============
class FusionTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            max_len=10  # 减少最大长度（只有几个tokens）
    ):
        super().__init__()

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

        # 回归头
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        print(f"  Transformer: d_model={d_model}, heads={nhead}, layers={num_layers}")

    def forward(self, spatial_vec, point_vec):
        # 确定批次大小
        B = None
        if spatial_vec is not None:
            B = spatial_vec.shape[0]
        elif point_vec is not None:
            B = point_vec.shape[0]
        else:
            raise ValueError("至少需要一个特征输入")

        # 收集所有tokens
        tokens = []

        # 1. [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens.append(cls_token)

        # 2. 空间token
        if spatial_vec is not None:
            spatial_token = spatial_vec.unsqueeze(1)  # (B, 1, d_model)
            tokens.append(spatial_token)

        # 3. 点特征token
        if point_vec is not None:
            point_token = point_vec.unsqueeze(1)  # (B, 1, d_model)
            tokens.append(point_token)

        # 拼接所有tokens
        x = torch.cat(tokens, dim=1)  # (B, L, d_model)
        L = x.shape[1]

        # 添加位置编码
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)

        # LayerNorm
        x = self.norm(x)

        # Transformer编码
        x = self.encoder(x)  # (B, L, d_model)

        # 使用[CLS] token进行预测
        cls_output = x[:, 0, :]  # (B, d_model)

        # 回归预测
        y_pred = self.head(cls_output).squeeze(-1)  # (B,)

        return y_pred


# ============= 完整SWE反演模型 =============
class SWENet(nn.Module):
    """
    修改后的SWE反演模型（无时序）
    输入:
        spatial_patch: (B, C_spatial, P, P)  # 卷积特征
        point_feats: (B, C_point)           # 点特征
    输出:
        y_pred: (B,) 标准化后的SWE值
    """

    def __init__(
            self,
            C_spatial=6,    # chelsa_sfxwind, lst, rh, clamday, dem_mean, dem_std
            C_point=10,     # ls + 经纬度 + doy + 其他点特征，默认10
            d_model=256,
            use_spatial=True,
            use_point=True
    ):
        super().__init__()

        self.use_spatial = use_spatial
        self.use_point = use_point

        print(f"\n[SWENet模型初始化]")
        print(f"  输入维度: C_spatial={C_spatial}, C_point={C_point}")
        print(f"  特征开关: 空间={use_spatial}, 点={use_point}")

        # 1) 空间编码器
        if use_spatial:
            self.spatial_encoder = SpatialEncoder(C_spatial=C_spatial, d_spatial=d_model)
        else:
            self.spatial_encoder = None

        # 2) 点编码器
        if use_point:
            if C_point is None:
                # 如果C_point为None，给出警告并使用默认值
                print("警告: C_point为None，使用默认值10")
                C_point = 10
            self.point_encoder = PointEncoder(C_point=C_point, d_point=d_model)
        else:
            self.point_encoder = None

        # 3) 计算最大序列长度
        max_len = 1  # [CLS] token
        if use_spatial:
            max_len += 1
        if use_point:
            max_len += 1

        print(f"  Transformer最大序列长度: {max_len}")

        # 4) Transformer融合
        self.fusion_transformer = FusionTransformer(
            d_model=d_model,
            max_len=max_len
        )

        # 5) 模型维度信息
        self.d_model = d_model
        self.max_len = max_len

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spatial_patch, point_feats):
        # 初始化特征向量
        spatial_vec = None
        point_vec = None

        # A. 空间编码
        if self.use_spatial and spatial_patch is not None:
            # spatial_patch: (B, C_spatial, P, P)
            spatial_vec = self.spatial_encoder(spatial_patch)  # (B, d_model)
        elif self.use_spatial:
            raise ValueError("模型配置使用空间特征，但未提供spatial_patch")

        # B. 点编码
        if self.use_point and point_feats is not None:
            # point_feats: (B, C_point)
            point_vec = self.point_encoder(point_feats)  # (B, d_model)
        elif self.use_point:
            raise ValueError("模型配置使用点特征，但未提供point_feats")

        # C. Transformer融合
        y_pred = self.fusion_transformer(spatial_vec, point_vec)

        return y_pred


# ============= 模型工厂函数 =============
def create_model(model_type, **kwargs):
    """
    根据类型创建模型

    参数:
        model_type: 模型类型
            "full" - 完整模型（空间+点）
            "spatial_only" - 仅空间
            "point_only" - 仅点特征
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
    all_config = {**config, **kwargs}

    return SWENet(**all_config)


# ============= 快捷方式函数 =============
def SWENet_Full(**kwargs):
    """完整模型"""
    return create_model("full", **kwargs)


def SWENet_SpatialOnly(**kwargs):
    """仅空间模型"""
    return create_model("spatial_only", **kwargs)


def SWENet_PointOnly(**kwargs):
    """仅点特征模型"""
    return create_model("point_only", **kwargs)


# ============= 测试函数 =============
def test_model():
    """测试模型"""
    print("=" * 50)
    print("测试模型...")
    print("=" * 50)

    # 测试完整模型
    print("\n1. 测试完整模型:")
    model = SWENet_Full(C_spatial=5, C_point=10)  # 假设有10个点特征

    # 创建测试数据
    batch_size = 2
    spatial_test = torch.randn(batch_size, 5, 5, 5)  # C_spatial=5, P=5
    point_test = torch.randn(batch_size, 10)  # C_point=10

    # 前向传播
    with torch.no_grad():
        output = model(spatial_test, point_test)

    print(f"  输入形状: spatial={spatial_test.shape}, point={point_test.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出示例: {output[:2]}")

    # 测试参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 测试简化模型
    print("\n2. 测试仅空间模型:")
    model_simple = SWENet_SpatialOnly(C_spatial=5)

    with torch.no_grad():
        output_simple = model_simple(spatial_test, None)

    print(f"  输出形状: {output_simple.shape}")

    print("\n✓ 模型测试完成!")


if __name__ == "__main__":
    test_model()