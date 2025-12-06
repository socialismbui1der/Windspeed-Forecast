import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer

# 保持辅助函数不变...
def get_torch_trans(heads=8, layers=2, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=2*channels, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64):
  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 128, # 注意：如果 N 很大，这里可能需要调整
        n_local_attn_heads = 4, 
        local_attn_window_size = 12,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SpatialGNN(nn.Module):
    """
    使用 GATv2Conv 实现的多层稀疏图卷积层。
    处理 (B, C, N, K, L) -> (B, C, N, K, L) 的转换。
    """
    # 🌟 关键修改: 增加 num_layers 参数，默认为 2
    def __init__(self, channels, edge_index, edge_weight, nheads=4, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.nheads = nheads
        
        self.convs = nn.ModuleList()
        in_dim = channels
        out_dim_per_head = channels // nheads
        
        # 🌟 堆叠多层 GATv2Conv
        for i in range(num_layers):
            # GATv2Conv 的输入和输出维度保持一致 (channels)，方便堆叠
            self.convs.append(
                GATv2Conv(
                    in_dim,
                    out_dim_per_head, # 内部隐藏层维度
                    heads=nheads,
                    concat=True,      # 拼接后输出维度仍为 channels
                    edge_dim=1,
                    dropout=0.1,
                    add_self_loops=False
                )
            )
            # 除最后一层外，添加 ReLU 激活函数
            if i < num_layers - 1:
                self.convs.append(nn.ReLU(inplace=True)) 
        
        # 注册图结构作为 Buffer
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight.unsqueeze(-1)) 

    def forward(self, x):
        B, C, N, K, L = x.shape
        E = self.edge_index.size(1)
        
        # 1. 维度重排与展平 (N_total, C)
        num_graphs = B * K * L
        x_flat = x.permute(0, 3, 4, 2, 1).reshape(num_graphs * N, C) 
        
        # 2. 构造 PyG 兼容输入：重复图结构 (只需要计算一次重复的 edge_index 和 edge_weight)
        offsets = (torch.arange(num_graphs, device=x.device) * N).view(1, -1) 
        repeated_edge_index = self.edge_index.unsqueeze(-1).repeat(1, 1, num_graphs).reshape(2, E * num_graphs)
        repeated_offsets = offsets.unsqueeze(0).repeat(2, E, 1).reshape(2, E * num_graphs)
        batch_edge_index = repeated_edge_index + repeated_offsets
        batch_edge_weight = self.edge_weight.repeat(num_graphs, 1)

        # 3. 🌟 循环 GATv2Conv 计算
        h = x_flat
        # self.convs 包含 GAT层和 ReLU层
        for layer in self.convs:
            if isinstance(layer, GATv2Conv):
                # GAT 层需要图结构输入
                h = layer(h, batch_edge_index, edge_attr=batch_edge_weight)
            else:
                # 激活层
                h = layer(h)
                
        out_node_features = h

        # 4. 恢复形状 (B*K*L * N, C) -> (B, C, N, K, L)
        out = out_node_features.reshape(B, K, L, N, C).permute(0, 4, 3, 1, 2)
        
        return out

class DiffusionEmbedding(nn.Module):
    # ... (保持原代码不变) ...
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    """
    精简版 ResidualBlock：
    - 只做 Time Attention + Feature MLP
    - 不再在 block 内做 GNN（空间聚合），避免你提到的 over-smoothing / 干扰噪声 / 计算爆炸问题
    - 空间相关性只通过外部 GCN → side_info 注入（你在 main_modelB 里已经做了）
    """
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        # 条件信息和中间投影
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection  = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear

        # 时间维注意力（沿 L）
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=2, channels=channels)

        # 特征维 mixing（沿 K）
        self.feature_mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels)
        )

        # 只保留 Time / Feature 的 LN
        self.norm_time = nn.LayerNorm(channels)
        self.norm_feat = nn.LayerNorm(channels)

    def forward_time(self, y, base_shape):
        """
        Attention over L
        Input y: (B, C, N, K, L)
        Reshape to: (B*N*K, C, L) -> Attention -> Restore
        """
        B, channel, N, K, L = base_shape
        if L == 1:
            return y

        # (B, C, N, K, L) -> (B, N, K, C, L) -> (B*N*K, C, L)
        y = y.permute(0, 2, 3, 1, 4).reshape(B * N * K, channel, L)

        if self.is_linear:
            # LinearAttentionTransformer: (B, L, C)
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # nn.TransformerEncoder: (L, B, C)
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        # Restore: (B*N*K, C, L) -> (B, N, K, C, L) -> (B, C, N, K, L)
        y = y.reshape(B, N, K, channel, L).permute(0, 3, 1, 2, 4)
        return y

    def forward_feature(self, y, base_shape):
        """
        MLP over K (对每个 (B, N, L) 上的特征维做 channels-mixing)
        Input y: (B, C, N, K, L)
        """
        B, channel, N, K, L = base_shape
        if K == 1:
            return y

        # (B, C, N, K, L) -> (B, N, L, K, C) -> (B*N*L*K, C)
        y = y.permute(0, 2, 4, 3, 1).reshape(B * N * L * K, channel)

        y = self.feature_mlp(y)

        # Restore -> (B, C, N, K, L)
        y = y.reshape(B, N, L, K, channel).permute(0, 4, 1, 3, 2)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        """
        x:         (B, C, N, K, L)
        cond_info: (B, side_dim, N, K, L)
        diffusion_emb: (B, emb_dim)
        """
        B, channel, N, K, L = x.shape
        base_shape = x.shape

        # 1. 展平 (N,K,L) 做 1x1 conv + 加 diffusion embedding
        x_flat = x.reshape(B, channel, N * K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B, C, 1)
        y = x_flat + diffusion_emb  # (B, C, N*K*L)

        # 恢复 5D
        y = y.reshape(base_shape)  # (B, C, N, K, L)

        # 2. 时间注意力
        y = self.forward_time(y, base_shape)
        # LN over C
        y = y.permute(0, 2, 3, 4, 1)     # (B,N,K,L,C)
        y = self.norm_time(y)
        y = y.permute(0, 4, 1, 2, 3)     # (B,C,N,K,L)

        # 3. 特征维 MLP
        y = self.forward_feature(y, base_shape)
        y = y.permute(0, 2, 3, 4, 1)     # (B,N,K,L,C)
        y = self.norm_feat(y)
        y = y.permute(0, 4, 1, 2, 3)     # (B,C,N,K,L)

        # 4. 再次展平做 gating + cond 注入
        y = y.reshape(B, channel, N * K * L)  # (B,C,N*K*L)

        # 中间投影
        y_mid = self.mid_projection(y)        # (B,2C,N*K*L)

        # cond_info: (B, side_dim, N, K, L) -> (B, side_dim, N*K*L)
        _, side_dim, _, _, _ = cond_info.shape
        cond_info_flat = cond_info.reshape(B, side_dim, N * K * L)
        cond_info_flat = self.cond_projection(cond_info_flat)  # (B,2C,N*K*L)

        y = y_mid + cond_info_flat           # (B,2C,N*K*L)

        # Gating
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        # Output projection -> residual & skip
        y = self.output_projection(y)        # (B,2C,N*K*L)
        residual, skip = torch.chunk(y, 2, dim=1)

        # reshape 回 5D
        x_5d       = x.reshape(base_shape)
        residual_5d = residual.reshape(base_shape)
        skip_5d     = skip.reshape(base_shape)

        return (x_5d + residual_5d) / math.sqrt(2.0), skip_5d

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        """
        为了不改动 CSDI_base 的调用签名，这里仍然接收 edge_index / edge_weight，
        但在本类内部已经不再使用 GNN（对应“方案 C”）：
        - 空间信息全部由 main_modelB 里的 GCN/GAT 先算成 gcn_feat，再拼到 side_info 里。
        """
        super().__init__()

        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        # 输入: (B, inputdim, N, K, L)
        self.input_projection  = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        # Residual blocks: 不再传 edge_index/edge_weight，下层也不再做 GNN
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        """
        x:           (B, inputdim, N, K, L)
        cond_info:   (B, side_dim, N, K, L)
        diffusion_step: (B,)
        返回:
            predicted noise: (B, N, K, L)
        """
        B, inputdim, N, K, L = x.shape

        # 1. Input projection
        x = x.reshape(B, inputdim, N * K * L)       # (B,inputdim,N*K*L)
        x = self.input_projection(x)               # (B,channels,N*K*L)
        x = F.relu(x)
        x = x.reshape(B, self.channels, N, K, L)   # (B,C,N,K,L)

        # 2. diffusion embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)  # (B,emb_dim)

        # 3. residual blocks（只做时序+特征 mixing，空间信息来自 cond_info 里的 GCN 通道）
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        # 4. 聚合 skip，再输出噪声
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, N * K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)            # (B,1,N*K*L)

        x = x.reshape(B, N, K, L)                # (B,N,K,L)
        return x
