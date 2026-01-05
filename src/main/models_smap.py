# models_smap.py
import torch
import torch.nn as nn


# --------- 3DConv + LSTM 时序编码器 ---------
class SpatioTemporalEncoder(nn.Module):
    """
    输入: dyn_seq (B, C_dyn, T, P, P)
    输出: temporal_tokens (B, T, d_temporal)
    """
    def __init__(
        self,
        C_dyn=5,
        d_3d=64,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.1
    ):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(C_dyn, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, d_3d, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(d_3d),
            nn.ReLU(inplace=True),
        )

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=d_3d,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = lstm_hidden * 2  # 双向

    def forward(self, dyn_seq):
        # dyn_seq: (B, C_dyn, T, P, P)
        x = self.conv3d(dyn_seq)          # (B, d_3d, T, P, P)
        x = x.mean(dim=[3, 4])            # 空间 GAP -> (B, d_3d, T)
        x = x.transpose(1, 2)             # (B, T, d_3d)

        out, _ = self.lstm(x)             # (B, T, 2*lstm_hidden)
        out = self.dropout(out)
        return out                        # temporal tokens


# --------- Spatial Conv 模块 ---------
class SpatialEncoder(nn.Module):
    """
    输入: spatial_patch (B, C_spatial, P, P)
    输出: spatial_vec (B, d_spatial)
    """
    def __init__(self, C_spatial=26, d_spatial=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_spatial, 64, kernel_size=3, padding=1),
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
        h = self.conv(x)                  # (B,128,P,P)
        h = self.pool(h).flatten(1)       # (B,128)
        h = self.proj(h)                  # (B,d_spatial)
        return h


# --------- Point 特征 MLP ---------
class PointEncoder(nn.Module):
    """
    输入: point_feats (B, C_point)
    输出: point_vec  (B, d_point)
    """
    def __init__(self, C_point, d_point=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(C_point, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, d_point),
        )
        self.out_dim = d_point

    def forward(self, x):
        return self.mlp(x)


# --------- Transformer 融合 + 回归头 ---------
class FusionTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=400   # T + 3 的上限，361+3=364 < 400
    ):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 位置编码
        self.pos_emb = nn.Embedding(max_len, d_model)

        # 最终回归头
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, temporal_seq, spatial_vec, point_vec):
        ## 修改## 这里改动了！添加了对 None 的处理
        B = None

        # 确定批次大小
        if temporal_seq is not None:
            B = temporal_seq.shape[0]
        elif spatial_vec is not None:
            B = spatial_vec.shape[0]
        elif point_vec is not None:
            B = point_vec.shape[0]
        else:
            raise ValueError("至少需要一个特征输入")

        # 组装 tokens
        tokens = []

        # [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens.append(cls)

        # 空间 token
        ## 修改## 添加 if 判断
        if spatial_vec is not None:
            spatial_tok = spatial_vec.unsqueeze(1)
            tokens.append(spatial_tok)

        # 点 token
        ## 修改## 添加 if 判断
        if point_vec is not None:
            point_tok = point_vec.unsqueeze(1)
            tokens.append(point_tok)

        # 时间序列 tokens
        ## 修改## 添加 if 判断
        if temporal_seq is not None:
            tokens.append(temporal_seq)

        # 拼接所有 tokens
        h = torch.cat(tokens, dim=1)  # (B, L, D)
        L = h.size(1)

        # 位置编码
        pos_ids = torch.arange(L, device=h.device).unsqueeze(0)  # (1,L)
        pos_enc = self.pos_emb(pos_ids)  # (1,L,D)
        h = h + pos_enc

        h = self.encoder(h)  # (B,L,D)
        cls_out = h[:, 0]  # (B,D)

        y_pred = self.head(cls_out).squeeze(-1)  # (B,)
        return y_pred

# --------- 整体模型 ---------
class SoilMoistureNet(nn.Module):
    """
    综合上面所有模块的完整网络
    """
    def __init__(
        self,
        C_dyn=5,
        C_spatial=26,
        C_point=34,      # 按你实际的 point_feats 维度修改
        d_model=256,
        T_max=361,
        use_temporal=True,
        use_spatial=True,
        use_point=True
    ):
        super().__init__()

        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.use_point = use_point

        print(f"\n[模型配置] 特征开关:")
        print(f"  使用时序特征: {use_temporal}")
        print(f"  使用空间特征: {use_spatial}")
        print(f"  使用点特征: {use_point}")

        # 1) 时序编码
        if use_temporal:
            self.st_encoder = SpatioTemporalEncoder(C_dyn=C_dyn)
            d_temporal = self.st_encoder.out_dim
            self.temporal_proj = nn.Linear(d_temporal, d_model)
        else:
            self.st_encoder = None
            self.temporal_proj = None

        # 2) 空间编码
        if use_spatial:
            self.spatial_encoder = SpatialEncoder(C_spatial=C_spatial, d_spatial=d_model)
        else:
            self.spatial_encoder = None

        # 3) 点编码
        if use_point:
            self.point_encoder = PointEncoder(C_point=C_point, d_point=d_model)
        else:
            self.point_encoder = None

        # 5) 计算Transformer最大序列长度
        max_len = 1  # [CLS]
        if use_spatial:
            max_len += 1
        if use_point:
            max_len += 1
        if use_temporal:
            max_len += T_max

        # 5) Transformer 融合 + 回归
        self.fusion = FusionTransformer(
            d_model=d_model,
            max_len=max_len  # ## 修改## 使用动态计算的max_len
        )

    def forward(self, dyn_seq, spatial_patch, point_feats):
        """
        dyn_seq:       (B, C_dyn, T, P, P)
        spatial_patch: (B, C_spatial, P, P)
        point_feats:   (B, C_point)
        """
        temporal_seq = None
        spatial_vec = None
        point_vec = None
        
        # A. 时序编码
        if self.use_temporal and dyn_seq is not None:
            temporal_seq = self.st_encoder(dyn_seq)         # (B,T,d_temporal)
            temporal_seq = self.temporal_proj(temporal_seq) # (B,T,d_model)
        elif self.use_temporal:
            raise ValueError("模型配置使用时序特征，但未提供 dyn_seq")

        # B. 空间 & 点编码
        if self.use_spatial and spatial_patch is not None:
            spatial_vec = self.spatial_encoder(spatial_patch)   # (B,d_model)
        elif self.use_spatial:
            raise ValueError("模型配置使用空间特征，但未提供 spatial_patch")
        
        if self.use_point and point_feats is not None:
            point_vec = self.point_encoder(point_feats)       # (B,d_model)
        elif self.use_point:
            raise ValueError("模型配置使用点特征，但未提供 point_feats")

        # C. Transformer 融合 & 回归
        y_pred = self.fusion(temporal_seq, spatial_vec, point_vec)  # (B,)

        return y_pred


def create_model_by_type(model_type, **kwargs):
    """
    根据模型类型创建对应的 SoilMoistureNet 实例

    参数:
        model_type: 
            "full" - 完整模型
            "no_temporal" - 无时序
            "no_spatial" - 无空间
            "no_point" - 无点特征
            "temporal_only" - 仅时序
            "spatial_only" - 仅空间
            "point_only" - 仅点特征
    """
    type_config = {
        "full": {"use_temporal": True, "use_spatial": True, "use_point": True},
        "no_temporal": {"use_temporal": False, "use_spatial": True, "use_point": True},
        "no_spatial": {"use_temporal": True, "use_spatial": False, "use_point": True},
        "no_point": {"use_temporal": True, "use_spatial": True, "use_point": False},
        "temporal_only": {"use_temporal": True, "use_spatial": False, "use_point": False},
        "spatial_only": {"use_temporal": False, "use_spatial": True, "use_point": False},
        "point_only": {"use_temporal": False, "use_spatial": False, "use_point": True},
    }

    if model_type not in type_config:
        raise ValueError(f"未知的模型类型: {model_type}。可选: {list(type_config.keys())}")

    config = type_config[model_type]
    return SoilMoistureNet(**config, **kwargs)


## 新增## 快捷方式函数
def SoilMoistureNet_Full(**kwargs):
    """完整模型"""
    return create_model_by_type("full", **kwargs)


def SoilMoistureNet_NoTemporal(**kwargs):
    """无时序模型"""
    return create_model_by_type("no_temporal", **kwargs)


def SoilMoistureNet_NoSpatial(**kwargs):
    """无空间模型"""
    return create_model_by_type("no_spatial", **kwargs)


def SoilMoistureNet_OnlyPoint(**kwargs):
    """仅点特征模型"""
    return create_model_by_type("point_only", **kwargs)


def SoilMoistureNet_OnlySpatial(**kwargs):
    """仅空间特征模型"""
    return create_model_by_type("spatial_only", **kwargs)


def SoilMoistureNet_OnlyTemporal(**kwargs):
    """仅时序特征模型"""
    return create_model_by_type("temporal_only", **kwargs)