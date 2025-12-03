import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=2*channels, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 128,
        n_local_attn_heads = 4, 
        local_attn_window_size = 12,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """
    扩散步 (diffusion step, t) 的位置编码模块。
    类似 Transformer 的 Positional Encoding，但这里针对时间步 t，
    通过正余弦函数构造一个固定表，再经两层 MLP 投射到高维。
    """
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim  # 默认输出维度 = 输入维度

        # register_buffer 表示这个张量不是可训练参数，但会随模型保存/加载。
        # persistent=False 意味着在保存状态时不会持久化（可省空间）
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),  # 生成正余弦表
            persistent=False,
        )

        # 两层线性层，用于让嵌入在训练中能被进一步非线性投射
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        """
        输入: diffusion_step (B,) 或 (1,) —— 整数步编号
        输出: 对应步的嵌入向量，形状 (B, projection_dim)
        """
        # 从固定表中取出对应步的正余弦嵌入
        x = self.embedding[diffusion_step]  # shape: (B, embedding_dim)

        # 通过两层 MLP + SiLU 激活（即 Swish 函数）增强表达能力
        x = self.projection1(x)
        x = F.silu(x)   # SiLU(x) = x * sigmoid(x)
        x = self.projection2(x)
        x = F.silu(x)

        return x

    def _build_embedding(self, num_steps, dim=64):
        """
        构建固定的正余弦时间步嵌入表：
        类似于 Transformer 的 positional encoding，但基于 10^频率尺度。
        """
        # 步编号: [0, 1, 2, ..., num_steps-1]  → 形状 (num_steps, 1)
        steps = torch.arange(num_steps).unsqueeze(1)

        # 生成 dim 个频率（指数变化的频率）
        # 10.0 ** (i / (dim-1) * 4.0)  → 频率从 1 到 10^4 之间指数增长
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        # frequencies 形状: (1, dim)

        # 每个 step × 每个频率  → (num_steps, dim)
        table = steps * frequencies

        # 拼接 sin 和 cos，得到 (num_steps, dim*2)
        # 即每个时间步的高维嵌入向量
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)

        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        # 时间嵌入层
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
                    side_dim=config["side_dim"]+1,
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
        :param x:含噪数据。即扩散过程中的 x_t（或 x）。 (B, inputdim, K, L)
        :param cond_info:条件信息。包括已知观测值和 Side Info（时间嵌入、特征嵌入等） (B, side_dim, K, L)
        :param diffusion_step:时间步 t。表示当前的噪声级别。 (B)
        :return:
        """
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L) # [B, 输入通道, K*L]
        # 把原始输入（低维）投射成模型的高维特征空间，以便后续残差块处理。
        x = self.input_projection(x) # -> [B, channels, K*L]
        x = F.relu(x)

        x = x.reshape(B, self.channels, K, L)# -> [B, channels, K, L]

        # 扩散步嵌入
        diffusion_emb = self.diffusion_embedding(diffusion_step) # [B, projection_dim]

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L) # (B,channel,K*L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x  # 预测的噪声


class ResidualBlock(nn.Module):
    """
    CSDI 模型的核心残差块。
    作用：
        - 将时间方向（L）与特征方向（K）分别进行注意力或线性变换；
        - 融合扩散步 embedding (t信息) 与条件信息 cond_info；
        - 输出残差 (residual) 与跳连特征 (skip)，供后续层叠与最终输出使用。
    """
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()

        # ---- (1) 扩散步嵌入线性投射 ----
        # 把 DiffusionEmbedding(t) 输出的向量映射到与主通道相同维度，以便相加
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)

        # ---- (2) 条件信息 cond_info 线性投射 ----
        # cond_info 通常是观测值或掩码（side_dim通道），映射为 2*channels，
        # 因为后面要拆成 gate 和 filter。
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)

        # ---- (3) 主分支中间变换 ----
        # 对当前层特征做线性（1x1卷积）变换，同样映射成 2*channels，准备门控
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # ---- (4) 输出投射 ----
        # 将门控后的输出映射到 2*channels，后面要拆成 residual 和 skip
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # ---- (5) 选择时间/特征方向的建模层类型 ----
        # 如果 is_linear=True，用线性 attention（Fast Transformer）
        # 否则用标准多头注意力（Torch Transformer）
        self.is_linear = is_linear
        if is_linear:
            self.time_layer = LinearAttentionTransformer(
                dim=channels,
                depth=1,
                heads=nheads,
                max_seq_len=128,
                n_local_attn_heads=2,   # 或者一部分，比如 nheads//2
                local_attn_window_size=12,   # 72 % 18 = 0
            )
            # self.feature_layer = LinearAttentionTransformer(
            #     dim=channels,
            #     depth=1,
            #     heads=nheads,
            #     max_seq_len=256,
            #     n_local_attn_heads=0,   
            #     local_attn_window_size=0,   
            # )
            #self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            #self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            
        # ---- (6) 特征维：不用 attention，改为简单 MLP（逐位置通道 MLP）----
        # 这里的 MLP 是对每个 (k, l) 位置上的 channel 向量做非线性变换，
        # 不在 K / L 方向上做注意力，只做通道混合。
        self.feature_mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels)
        )
        self.norm_time = nn.LayerNorm(channels)
        self.norm_feat = nn.LayerNorm(channels)

    # ============================================================
    # ↓↓↓ 以下两个函数分别对“时间维”和“特征维”做注意力 / 线性变换
    # ============================================================

    def forward_time(self, y, base_shape):
        """
        在时间维 L 上进行建模。
        让模型捕捉同一变量在不同时间点之间的依赖关系。
        """
        B, channel, K, L = base_shape

        # 若 L=1（只有一个时间步），则跳过此操作
        if L == 1:
            return y

        # 重塑形状：把时间维抽出来，方便在 L 上做 attention
        # (B, channel, K, L) → (B*K, channel, L)
        # torch.Size([64, 64, 288])->torch.Size([256, 64, 72])
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        # 线性 attention / torch attention 输入顺序不同，需要调维
        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)

        # 恢复形状回 (B, channel, K*L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        """
        在特征维 K 上进行建模。
        让模型捕捉不同变量（特征）之间的空间相关性。
        """
        B, channel, K, L = base_shape

        if K == 1:
            # 只有一个特征，没啥可混的，直接返回
            return y

        # (B, channel, K*L) → (B, channel, K, L)
        y = y.reshape(B, channel, K, L)
        # 把通道维移到最后，对每个 (k,l) 的 channel 向量做 MLP
        # (B, channel, K, L) → (B, K, L, channel)
        y = y.permute(0, 2, 3, 1).reshape(B * K * L, channel)  # (B*K*L, channel)

        y = self.feature_mlp(y)  # (B*K*L, channel)

        # 还原回 (B, channel, K*L)
        y = y.reshape(B, K, L, channel).permute(0, 3, 1, 2).reshape(B, channel, K * L)
        return y

    # 原版使用注意力的特征维
    # def forward_feature(self, y, base_shape):
    #     B, channel, K, L = base_shape
    #     if K == 1:
    #         return y
    #     y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
    #     if self.is_linear:
    #         y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
    #     else:
    #         y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
    #     y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
    #     return y

    # ============================================================
    # ↓↓↓ 前向传播：融合扩散步、条件信息，并生成残差+skip输出
    # ============================================================

    def forward(self, x, cond_info, diffusion_emb):
        """
        输入:
            x: 当前层输入特征 (B, channel, K, L)
            cond_info: 条件输入 (B, side_dim, K, L)
            diffusion_emb: 扩散步嵌入 (B, diffusion_embedding_dim) 
        输出:
            (x_next, skip): 残差输出与跳连特征
        """
        B, channel, K, L = x.shape
        base_shape = x.shape

        # ---- (1) 展平空间维 (K,L) ----
        # torch.Size([64, 64, 288])
        x = x.reshape(B, channel, K * L)

        # ---- (2) 将扩散步嵌入映射并加到特征上 ----
        # diffusion_emb -> (B, channel, 1)
        # torch.Size([64, 128])->torch.Size([64, 64, 1])
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)

        # y：torch.Size([64, 64, 288])
        y = x + diffusion_emb   # 让每一层知道当前扩散步的信息

        # ---- (3) 在时间维、特征维上分别进行注意力/线性变换 ----
        y = self.forward_time(y, base_shape)
        y = self.norm_time(y.transpose(1,2)).transpose(1,2)
        y = self.forward_feature(y, base_shape)   # 输出形状: (B, channel, K*L)
        y = self.norm_feat(y.transpose(1,2)).transpose(1,2)

        # ---- (4) 对主分支特征做 1x1 卷积，映射到 2*channels ----
        y = self.mid_projection(y)  # (B, 2*channel, K*L)

        # ---- (5) 对条件信息做同样的线性投射，并相加 ----
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B, 2*channel, K*L)
        # (B, 2*channel, K*L)
        y = y + cond_info  # 条件调制

        # ---- (6) 门控机制（GLU-like）----
        # 用门控 GLU（Gated Linear Unit）控制信息流，增强非线性表达
        # gate和filter形状都为: (B, channels, K*L)
        gate, filter = torch.chunk(y, 2, dim=1)  # 前一半为 gate，后一半为 filter
        y = torch.sigmoid(gate) * torch.tanh(filter)
        # Sigmoid 控制门开关，tanh 生成非线性输出

        # ---- (7) 输出头，生成 residual & skip ----
        # # 把处理好的 y 再映射成两个东西：残差输出 residual 和 skip 输出 skip
        y = self.output_projection(y)  # (B, 2*channel, K*L)
        residual, skip = torch.chunk(y, 2, dim=1)

        # ---- (8) 残差连接：与输入 x 相加并缩放 ----
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        # (x + residual) / sqrt(2):层内的残差连接
        # skip 是跳连输出,不参与下一层，而是直接贡献给模型最终输出。
        return (x + residual) / math.sqrt(2.0), skip
