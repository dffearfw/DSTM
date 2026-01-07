# models_swe.py
import torch
import torch.nn as nn


# ============= 针对单动态变量优化的时序编码器 =============
class SpatioTemporalEncoder(nn.Module):
    """
    输入: dyn_seq (B, C_dyn, T, P, P)
    输出: temporal_tokens (B, T, d_temporal)

    针对 C_dyn=1 (chelsa_sfxwind) 优化
    """

    def __init__(
            self,
            C_dyn=1,  # 修改：默认1个动态变量
            d_3d=64,
            lstm_hidden=128,
            lstm_layers=1,
            dropout=0.1
    ):
        super().__init__()

        # 针对单通道优化的3D卷积
        self.conv3d = nn.Sequential(
            nn.Conv3d(C_dyn, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, d_3d, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(d_3d),
            nn.ReLU(inplace=True),
        )

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=d_3d,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.lstm_dropout = nn.Dropout(dropout)
        self.out_dim = lstm_hidden * 2  # 双向

        # 打印信息
        print(f"  时序编码器: C_dyn={C_dyn}, 3D输出={d_3d}, LSTM隐藏={lstm_hidden}")

    def forward(self, dyn_seq):
        # dyn_seq: (B, C_dyn, T, P, P)
        B, C_dyn, T, P, _ = dyn_seq.shape

        x = self.conv3d(dyn_seq)  # (B, d_3d, T, P, P)
        x = x.mean(dim=[3, 4])  # 空间全局平均池化 -> (B, d_3d, T)
        x = x.transpose(1, 2)  # (B, T, d_3d)

        # LSTM处理
        out, _ = self.lstm(x)  # (B, T, 2*lstm_hidden)
        out = self.lstm_dropout(out)

        return out  # temporal tokens


# ============= 针对占位符静态变量的空间编码器 =============
class SpatialEncoder(nn.Module):
    """
    输入: spatial_patch (B, C_spatial, P, P)
    输出: spatial_vec (B, d_spatial)

    针对C_spatial=24优化（23个静态占位符 + 1个动态）
    """

    def __init__(self, C_spatial=24, d_spatial=256):
        super().__init__()

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

        print(f"  空间编码器: 输入通道={C_spatial}, 输出维度={d_spatial}")

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

    针对C_point=30优化（23静态 + 1动态 + 6时间）
    """

    def __init__(self, C_point=30, d_point=256, dropout=0.1):
        super().__init__()

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

        print(f"  点编码器: 输入维度={C_point}, 输出维度={d_point}")

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
            max_len=400  # 足够大以容纳所有tokens
    ):
        super().__init__()

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # 使用GELU激活函数
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

        # 打印配置
        print(f"  Transformer: d_model={d_model}, heads={nhead}, layers={num_layers}")

    def forward(self, temporal_seq, spatial_vec, point_vec):
        # 确定批次大小
        B = None
        if temporal_seq is not None:
            B = temporal_seq.shape[0]
        elif spatial_vec is not None:
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

        # 4. 时间序列tokens
        if temporal_seq is not None:
            # temporal_seq: (B, T, d_model)
            tokens.append(temporal_seq)

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
    完整的SWE反演模型
    输入:
        dyn_seq: (B, C_dyn, T, P, P)
        spatial_patch: (B, C_spatial, P, P)
        point_feats: (B, C_point)
    输出:
        y_pred: (B,) 标准化后的SWE值
    """

    def __init__(
            self,
            C_dyn=1,
            C_spatial=24,
            C_point=30,
            d_model=256,
            T_max=365,
            use_temporal=True,
            use_spatial=True,
            use_point=True
    ):
        super().__init__()

        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.use_point = use_point

        print(f"\n[SWENet模型初始化]")
        print(f"  输入维度: C_dyn={C_dyn}, C_spatial={C_spatial}, C_point={C_point}")
        print(f"  特征开关: 时序={use_temporal}, 空间={use_spatial}, 点={use_point}")

        # 1) 时序编码器
        if use_temporal:
            self.temporal_encoder = SpatioTemporalEncoder(C_dyn=C_dyn)
            d_temporal = self.temporal_encoder.out_dim
            # 投影到d_model维度
            self.temporal_proj = nn.Linear(d_temporal, d_model)
        else:
            self.temporal_encoder = None
            self.temporal_proj = None

        # 2) 空间编码器
        if use_spatial:
            self.spatial_encoder = SpatialEncoder(C_spatial=C_spatial, d_spatial=d_model)
        else:
            self.spatial_encoder = None

        # 3) 点编码器
        if use_point:
            self.point_encoder = PointEncoder(C_point=C_point, d_point=d_model)
        else:
            self.point_encoder = None

        # 4) 计算最大序列长度
        max_len = 1  # [CLS] token
        if use_spatial:
            max_len += 1
        if use_point:
            max_len += 1
        if use_temporal:
            max_len += T_max

        print(f"  Transformer最大序列长度: {max_len}")

        # 5) Transformer融合
        self.fusion_transformer = FusionTransformer(
            d_model=d_model,
            max_len=max_len
        )

        # 6) 模型维度信息（方便外部访问）
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
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dyn_seq, spatial_patch, point_feats):
        # 初始化特征向量
        temporal_seq = None
        spatial_vec = None
        point_vec = None

        # A. 时序编码
        if self.use_temporal and dyn_seq is not None:
            # dyn_seq: (B, C_dyn, T, P, P)
            temporal_seq = self.temporal_encoder(dyn_seq)  # (B, T, d_temporal)
            temporal_seq = self.temporal_proj(temporal_seq)  # (B, T, d_model)
        elif self.use_temporal:
            raise ValueError("模型配置使用时序特征，但未提供dyn_seq")

        # B. 空间编码
        if self.use_spatial and spatial_patch is not None:
            # spatial_patch: (B, C_spatial, P, P)
            spatial_vec = self.spatial_encoder(spatial_patch)  # (B, d_model)
        elif self.use_spatial:
            raise ValueError("模型配置使用空间特征，但未提供spatial_patch")

        # C. 点编码
        if self.use_point and point_feats is not None:
            # point_feats: (B, C_point)
            point_vec = self.point_encoder(point_feats)  # (B, d_model)
        elif self.use_point:
            raise ValueError("模型配置使用点特征，但未提供point_feats")

        # D. Transformer融合
        y_pred = self.fusion_transformer(temporal_seq, spatial_vec, point_vec)

        return y_pred


# ============= 模型工厂函数 =============
def create_model(model_type, **kwargs):
    """
    根据类型创建模型

    参数:
        model_type: 模型类型
            "full" - 完整模型
            "temporal_only" - 仅时序
            "spatial_only" - 仅空间
            "point_only" - 仅点特征
            "no_temporal" - 无时序
            "no_spatial" - 无空间
            "no_point" - 无点特征
        **kwargs: 传递给SWENet的参数
    """
    type_config = {
        "full": {"use_temporal": True, "use_spatial": True, "use_point": True},
        "temporal_only": {"use_temporal": True, "use_spatial": False, "use_point": False},
        "spatial_only": {"use_temporal": False, "use_spatial": True, "use_point": False},
        "point_only": {"use_temporal": False, "use_spatial": False, "use_point": True},
        "no_temporal": {"use_temporal": False, "use_spatial": True, "use_point": True},
        "no_spatial": {"use_temporal": True, "use_spatial": False, "use_point": True},
        "no_point": {"use_temporal": True, "use_spatial": True, "use_point": False},
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


def SWENet_TemporalOnly(**kwargs):
    """仅时序模型"""
    return create_model("temporal_only", **kwargs)


def SWENet_SpatialOnly(**kwargs):
    """仅空间模型"""
    return create_model("spatial_only", **kwargs)


def SWENet_PointOnly(**kwargs):
    """仅点特征模型"""
    return create_model("point_only", **kwargs)


def SWENet_NoTemporal(**kwargs):
    """无时序模型"""
    return create_model("no_temporal", **kwargs)


def SWENet_NoSpatial(**kwargs):
    """无空间模型"""
    return create_model("no_spatial", **kwargs)


def SWENet_NoPoint(**kwargs):
    """无点特征模型"""
    return create_model("no_point", **kwargs)


# ============= 测试函数 =============
def test_model():
    """测试模型"""
    print("=" * 50)
    print("测试模型...")
    print("=" * 50)

    # 测试完整模型
    print("\n1. 测试完整模型:")
    model = SWENet_Full(C_dyn=1, C_spatial=24, C_point=30, T_max=365)

    # 创建测试数据
    batch_size = 2
    dyn_test = torch.randn(batch_size, 1, 365, 5, 5)
    spatial_test = torch.randn(batch_size, 24, 5, 5)
    point_test = torch.randn(batch_size, 30)

    # 前向传播
    with torch.no_grad():
        output = model(dyn_test, spatial_test, point_test)

    print(f"  输入形状: dyn={dyn_test.shape}, spatial={spatial_test.shape}, point={point_test.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出示例: {output[:2]}")

    # 测试参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 测试简化模型
    print("\n2. 测试仅时序模型:")
    model_simple = SWENet_TemporalOnly(C_dyn=1, T_max=365)

    with torch.no_grad():
        output_simple = model_simple(dyn_test, None, None)

    print(f"  输出形状: {output_simple.shape}")

    print("\n✓ 模型测试完成!")


if __name__ == "__main__":
    test_model()