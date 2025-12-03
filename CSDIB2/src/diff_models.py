import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from torch_geometric.nn import GATv2Conv

# 保持辅助函数不变...
def get_torch_trans(heads=8, layers=2, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels, activation="gelu"
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
    def __init__(self, channels, edge_index, edge_weight, nheads, num_layers):
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
                    dropout=0.2,
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
    修改后的 ResidualBlock，支持 (B, C, N, K, L) 输入
    增加了 Spatial Attention 层
    """
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,edge_index, edge_weight, is_linear=False):
        super().__init__()

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        
        # side_dim 依然是外部传入的，但在 forward 中我们会处理维度匹配
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            # self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            # self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        self.feature_mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels)
        )

        self.spatial_layer = SpatialGNN(
            channels=channels, 
            edge_index=edge_index, 
            edge_weight=edge_weight, 
            nheads=nheads,
            num_layers = 2
        )

        self.norm_time = nn.LayerNorm(channels)
        self.norm_space = nn.LayerNorm(channels) # 新增 Norm
        self.norm_feat = nn.LayerNorm(channels)

    def forward_time(self, y, base_shape):
        """
        Attention over L
        Input y: (B, C, N, K, L)
        Reshape to: (B*N*K, C, L) -> Attention -> Restore
        """
        B, channel, N, K, L = base_shape
        if L == 1: return y

        # (B, C, N, K, L) -> (B, N, K, C, L) -> (B*N*K, C, L)
        y = y.permute(0, 2, 3, 1, 4).reshape(B * N * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
            
        # Restore: (B*N*K, C, L) -> (B, N, K, C, L) -> (B, C, N, K, L)
        y = y.reshape(B, N, K, channel, L).permute(0, 3, 1, 2, 4)
        return y

    def forward_space(self, y, base_shape):
        """
        使用 SpatialGNN (GATv2Conv) 处理空间维度
        Input y: (B, C, N, K, L)
        """
        # 直接调用 SpatialGNN 模块，其内部负责维度转换
        y = self.spatial_layer(y)
        return y

    def forward_feature(self, y, base_shape):
        """
        MLP over K (Channels mixing per node)
        Input y: (B, C, N, K, L)
        """
        B, channel, N, K, L = base_shape
        if K == 1: return y

        # (B, C, N, K, L) -> (B, N, L, K, C) -> (Flat, C)
        # 这里把通道放到最后做 Linear
        y = y.permute(0, 2, 4, 3, 1).reshape(B * N * L * K, channel)
        
        y = self.feature_mlp(y)

        # Restore
        y = y.reshape(B, N, L, K, channel).permute(0, 4, 1, 3, 2)
        return y

    # def forward_feature(self, y, base_shape):
    #     B, channel, N, K, L = base_shape
    #     if K == 1: 
    #         return y
    #     y = y.permute(0, 2, 4, 1, 3).reshape(B * N * L, channel, K)
    #     if self.is_linear:
    #         y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
    #     else:
    #         y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

    #     y = y.reshape(B, N, K, channel, L).permute(0, 3, 1, 2, 4)
    #     return y


    def forward(self, x, cond_info, diffusion_emb):
        """
        x: (B, C, N, K, L)
        cond_info: (B, side_dim, N, K, L)
        diffusion_emb: (B, emb_dim)
        """
        B, channel, N, K, L = x.shape
        base_shape = x.shape

        # 1. 展平所有维度除了 Channel，以便进行 Conv1d 投影和加法
        # (B, C, N*K*L)
        x_flat = x.reshape(B, channel, N * K * L)

        # 2. Diffusion Embedding
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1) # (B, C, 1)
        y = x_flat + diffusion_emb
        
        # 恢复 5D 形状进行 Attention
        y = y.reshape(base_shape)

        # 3. 三维处理：Time -> Space -> Feature
        # Time Attention
        y = self.forward_time(y, base_shape)
        # LayerNorm (Reshape for norm over C)
        y = y.permute(0, 2, 3, 4, 1) # B,N,K,L,C
        y = self.norm_time(y)
        y = y.permute(0, 4, 1, 2, 3) # B,C,N,K,L

        # Spatial GNN (使用 GATv2Conv)
        y = self.forward_space(y, base_shape)
        y = y.permute(0, 2, 3, 4, 1)
        y = self.norm_space(y)
        y = y.permute(0, 4, 1, 2, 3)

        # Feature Mix
        y = self.forward_feature(y, base_shape)
        y = y.permute(0, 2, 3, 4, 1)
        y = self.norm_feat(y)
        y = y.permute(0, 4, 1, 2, 3)

        # 4. 再次展平进行门控和跳连
        y = y.reshape(B, channel, N * K * L)
        
        # 处理 Projection
        y = self.mid_projection(y) # (B, 2C, N*K*L)

        # 处理 cond_info
        # cond_info 输入是 (B, side_dim, N, K, L)
        _, side_dim, _, _, _ = cond_info.shape
        cond_info_flat = cond_info.reshape(B, side_dim, N * K * L)
        cond_info_flat = self.cond_projection(cond_info_flat) # (B, 2C, N*K*L)
        
        y = y + cond_info_flat

        # Gating
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        # Output projection
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)

        # Reshape back to 5D
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip


class diff_CSDI(nn.Module):
    def __init__(self, config, edge_index,edge_weight,inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    edge_index = edge_index,
                    edge_weight = edge_weight,
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        """
        x: (B, inputdim, N, K, L)  <-- 注意这里增加了 N 维度
        cond_info: (B, side_dim, N, K, L)
        diffusion_step: (B,)
        """
        # 1. 解析输入形状
        B, inputdim, N, K, L = x.shape

        # 2. Input Projection
        # 展平后通过 1x1 卷积投射到高维 channel
        x = x.reshape(B, inputdim, N * K * L)
        x = self.input_projection(x) # (B, channels, N*K*L)
        x = F.relu(x)
        x = x.reshape(B, self.channels, N, K, L) # 恢复 5D

        # 3. Diffusion Embedding
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        # 4. Residual Layers
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        # 5. Output Projection
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        
        # 展平输出
        x = x.reshape(B, self.channels, N * K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x) # (B, 1, N*K*L)
        
        # 恢复形状 (B, N, K, L) (注意：这里不需要 channel 维了，因为输出是噪声)
        x = x.reshape(B, N, K, L)
        
        return x