import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device,edge_index,edge_weight):
        super().__init__()
        """
        target_dim:特征维数 K
        """

        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_graph = config["graph"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        else:
            s = 0.008
            steps = self.num_steps + 1
            x = np.linspace(0, self.num_steps, steps)
            alphas_cumprod = np.cos(((x / self.num_steps) + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = np.clip(betas, 1e-5, 0.999)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        # self.feature_weights = torch.ones(self.target_dim)  # 先默认 1

        # self.feature_weights[0] = 1  
        # self.feature_weights[1] = 1
        # self.feature_weights[2] = 1
        # self.feature_weights[3] = 1

        self.num_stations = config_graph["num_stations"]
        self.gcn_hidden = config_graph["hidden"]
        self.gcn_out_dim = config_graph["gcn_out"]
        self.gcn_layers = config_graph["gcn_layers"]
        self.feature_dim = 4
        self.gcn_in_dim = self.feature_dim*2

        self.edge_weight = edge_weight.to(device)
        self.edge_index = edge_index.to(device)
        self.diffmodel = diff_CSDI(config_diff, self.edge_index,self.edge_weight,input_dim)




    # 正余弦位置编码
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe # [B, L, d_model]

    # 对每个样本 i，先在其 observed_mask（原始可见处）里随机抽一个样本内的缺失比例（0~1）
    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            low, high = 0.1, 0.4
            # sample_ratio = np.random.rand()  # missing ratio
            sample_ratio = low + (high - low) * np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        # cond_mask（1=作为条件保留；0=作为目标预测）。
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask # （[B,K,L]）

    # 历史模式挖洞。当 target_strategy="mix"：50% 用 get_randmask 的随机洞；否则用上一条样本（i-1）的掩码模式去挖当前样本，模拟“历史缺失模式迁移”。
    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def make_mixed_impute_mask(self,
        observed_mask: torch.Tensor,
        block_prob: float = 0.4,
        max_block_len: int = 4,
        missing_ratio_range=(0.2, 0.4),
        special_feat_idx: int = 0,          # 要“重点挖掉”的特征索引，这里默认第一个特征 k=0
        special_extra_ratio: float = 0.1,   # 在原有基础上，对该特征额外再挖掉的比例（相对剩余可用点）
    ):
        """
        在插补任务中生成 cond_mask（1=条件保留，0=目标插补）。

        - 对每个样本 i 独立选择一种模式：
            * with prob = block_prob: 时间方向连续块缺失（block masking）
            * with prob = 1 - block_prob: 随机点缺失（random masking）
        - 连续块长度在 [1, max_block_len] 内均匀随机
        - 随机点缺失的比例在 missing_ratio_range 区间内随机采样

        参数:
            observed_mask: (B, K, L) 原始观测 mask，1=有值，0=原始缺失
            block_prob: 使用 block 掩码的概率
            max_block_len: 连续块最大长度（不会超过 L）
            missing_ratio_range: (r_min, r_max)，随机点缺失占当前样本观测点的比例区间

        返回:
            cond_mask: (B, K, L) float，1=保留，0=挖掉作为插补目标
        """
        assert observed_mask.dim() == 3, "observed_mask 必须是 [B, K, L]"
        B, K, L = observed_mask.shape
        device = observed_mask.device

        cond_mask = observed_mask.clone().float()  # 从原始 mask 出发挖洞

        r_min, r_max = missing_ratio_range
        r_min = float(r_min)
        r_max = float(r_max)

        max_block_len = min(int(max_block_len), L)
        if max_block_len <= 0:
            raise ValueError("max_block_len 必须 >= 1")

        for i in range(B):
            obs_i = observed_mask[i]            # (K, L)
            num_obs = int(obs_i.sum().item()) # 获取有效点的总数
            if num_obs == 0:
                # 整条序列本来就全缺，直接跳过
                continue

            use_block = (torch.rand(1, device=device) < block_prob).item()

            if use_block:
                # ---------- 连续块缺失 ----------
                # 块长度 [1, max_block_len]
                block_len = int(torch.randint(1, max_block_len + 1, (1,), device=device).item())
                # 块起点 [0, L - block_len]
                start = int(torch.randint(0, L - block_len + 1, (1,), device=device).item())
                end = start + block_len  # slice 右开区间

                # 只在原本有观测的地方挖掉：obs_i==1 的点 → 设为 0
                sub_obs = obs_i[:, start:end]           # (K, block_len)
                sub_cond = cond_mask[i, :, start:end]   # (K, block_len)
                sub_cond[sub_obs > 0] = 0.0
                cond_mask[i, :, start:end] = sub_cond

            else:
                # ---------- 随机点缺失 ----------
                # 在所有 observed=1 的点上随机选一部分变成 0
                obs_flat = obs_i.reshape(-1)  # (K*L,)
                idx_valid = torch.nonzero(obs_flat > 0, as_tuple=False).squeeze(1)
                num_valid = idx_valid.numel()
                if num_valid == 0:
                    continue

                # 当前样本的缺失比例
                mr = float(torch.empty(1).uniform_(r_min, r_max).item())
                num_masked = int(round(num_valid * mr))
                if num_masked <= 0:
                    continue

                # 随机选 num_masked 个 index
                perm = torch.randperm(num_valid, device=device)
                chosen = idx_valid[perm[:num_masked]]

                cond_flat = cond_mask[i].reshape(-1)
                cond_flat[chosen] = 0.0
                cond_mask[i] = cond_flat.view(K, L)

            # ---------- 额外：对 special_feat_idx 这一维再造一部分缺失 ----------
            if special_extra_ratio > 0:
                # 只在：原始观测=1 且 当前 cond_mask 仍为 1 的位置上，再挖
                obs_feat = obs_i[special_feat_idx]          # (L,)
                cond_feat = cond_mask[i, special_feat_idx]  # (L,)

                # 可被额外挖掉的候选点：原来有观测 & 还没被挖掉
                candidates = torch.nonzero(
                    (obs_feat > 0) & (cond_feat > 0), as_tuple=False
                ).squeeze(1)  # (N_candidate,)

                n_cand = candidates.numel()
                if n_cand > 0:
                    n_extra = int(round(n_cand * special_extra_ratio))
                    if n_extra > 0:
                        perm2 = torch.randperm(n_cand, device=device)
                        chosen2 = candidates[perm2[:n_extra]]
                        cond_mask[i, special_feat_idx, chosen2] = 0.0

        return cond_mask
    
    def lerp(self,start, end, alpha: float):
        """线性插值: alpha=0 → start, alpha=1 → end"""
        return start + (end - start) * alpha


    def make_curriculum_impute_mask(self,
        observed_mask: torch.Tensor,
        global_step: int,
        total_steps: int,
        # 下面这些可以按需改默认值
        mr_start: float = 0.15,   # 训练早期的缺失比例（简单）
        mr_end: float = 0.4,      # 训练后期的缺失比例（困难）
        block_len_start: int = 1, # 早期块长度（点状 / 短块）
        block_len_end: int = 4,  # 后期块最大长度（长块）
        block_prob_start: float = 0.15,  # 早期 block 掩码占比
        block_prob_end: float = 0.4,    # 后期 block 掩码占比
        special_feat_idx: int = 3,
        special_extra_start: float = 0.05,  # 训练早期：额外挖掉比例（相对剩余点），比如 10%
        special_extra_end: float = 0.25,    # 训练后期：额外挖掉比例，比如 40%
    ):
        """
        Curriculum 式插补掩码：
        - 训练早期：缺失比例低，块短，以随机点为主；
        - 训练后期：缺失比例高，块长，以连续块为主。

        参数:
            observed_mask: (B, K, L)
            global_step: 当前训练步数（或 epoch）
            total_steps: 总训练步数（或 max_epoch）
            mr_start, mr_end: 缺失比例从 mr_start 线性渐变到 mr_end
            block_len_start, block_len_end: 连续块 max_block_len 的线性 schedule
            block_prob_start, block_prob_end: 使用 block 掩码的概率线性从 start→end

        返回:
            cond_mask: (B, K, L) float
        """
        if total_steps <= 0:
            alpha = 1.0
        else:
            alpha = float(global_step) / float(max(total_steps, 1))
            alpha = max(0.0, min(1.0, alpha))  # clamp 到 [0,1]

        # 线性插值出当前阶段的超参数
        current_mr = self.lerp(mr_start, mr_end, alpha)
        current_block_len = int(round(self.lerp(block_len_start, block_len_end, alpha)))
        current_block_prob = float(self.lerp(block_prob_start, block_prob_end, alpha))

        # 选中特征额外缺失的比例 schedule
        current_special_extra = float(self.lerp(special_extra_start, special_extra_end, alpha))
        current_special_extra = max(0.0, min(0.99, current_special_extra))

        # 缺失比例可以给一个小范围波动，如果你想更稳定可以就用 (current_mr, current_mr)
        missing_ratio_range = (max(0.0, current_mr - 0.02),
                            min(0.99, current_mr + 0.02))

        cond_mask = self.make_mixed_impute_mask(
            observed_mask=observed_mask,
            block_prob=current_block_prob,
            max_block_len=current_block_len,
            missing_ratio_range=missing_ratio_range,
            special_feat_idx=special_feat_idx,
            special_extra_ratio=current_special_extra,
        )
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        # 仅有observed_mask和test_pattern_mask都为1的部分才为1
        return observed_mask * test_pattern_mask

    # 把时间嵌入 + 特征嵌入(+ 条件掩码)拼成供 UNet/扩散网络用的“侧信息”。
    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        # time_embed: [B, L, emb_time_dim] → 扩到 [B, L, K, emb_time_dim]
        # 把每个时间点（observed_tp）映射成一个向量
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        # feature_embed: 每个通道一个向量 [K, emb_feature_dim] → 扩到 [B, L, K, emb_feature_dim]
        # self.embed_layer 是一个可训练的 embedding 层；
        # 它为每个特征（第 k 个变量）分配一个嵌入向量；
        # 扩展成 [B, L, K, emb_feature_dim]，与时间嵌入对齐。
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb_feature_dim)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        # 拼接后转置成 [B, emb_total_dim, K, L]；
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        # 若有条件，再把 cond_mask 以 [B,1,K,L] 拼进去变成 [B, emb_total_dim(+1), K, L]。
        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
            self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    # 单步扩散损失。
    # 单步扩散损失。
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        # 注意：现在的输入 observed_data 应该是 (B, N, K, L) 而不是 (B*N, K, L)
        # 如果你之前已经在 forward 里展平了，你需要改 forward
        
        # 假设输入形状修正为 (B, N, K, L)
        B, N, K, L = observed_data.shape 
        
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t] # (B,1,1,1) 注意广播维度
        noise = torch.randn_like(observed_data)
        
        # 扩充 alpha 维度以匹配 (B, N, K, L)
        # self.alpha_torch 是 (num_steps, 1, 1). t取出来是 (B, 1, 1)
        # 需要 unsqueeze 使得它是 (B, 1, 1, 1) 来匹配 N, K, L
        current_alpha = current_alpha.unsqueeze(1) 

        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        # total_input shape: (B, 2, N, K, L)

        predicted = self.diffmodel(total_input, side_info, t) 
        # predicted shape: (B, N, K, L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss


    # 根据是否“有条件”组装喂入扩散网络的输入张量。
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        # 无条件（也就是没用cond_mask）：直接把所有数据都加噪
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)

        # 有条件：拼 cond_obs = cond_mask * observed_data 和 noisy_target = (1-cond_mask) * noisy_data 得到 [B,2,K,L]。
        else:
            # 已知点不加噪
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            # 目标点改成加噪后的数据
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    # 反向扩散生成插补样本（每条可以采 n_samples 个）。
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        """
        参数:
            observed_data: (B, N, K, L)
            cond_mask:     (B, N, K, L)
            side_info:     (B, C, N, K, L)
            n_samples:     int
        返回:
            imputed_samples: (B, n_samples, N, K, L)
        """
        # 1. 解析 4D 形状
        B, N, K, L = observed_data.shape
        device = self.device

        # 2. 初始化输出张量 (B, n_samples, N, K, L)
        imputed_samples = torch.zeros(B, n_samples, N, K, L, device=device)

        for i in range(n_samples):
            
            # --- 处理无条件情况 (仅在 is_unconditional=True 时执行) ---
            if self.is_unconditional:
                # noisy_obs, noise, noisy_cond_history 都是 (B, N, K, L)
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)
            
            # 3. 初始化当前样本 current_sample (B, N, K, L)
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                # 保持 t 的 batch 形状为 (B,)
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)

                if self.is_unconditional:
                    # 构造 x_t_uncond：缺失位置是当前扩散样本，观测位置是扩散历史中的观测值
                    x_t_uncond = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    
                    # 构造 diff_input: [x_t_uncond, cond_mask] -> (B, 2, N, K, L)
                    diff_input = torch.cat([x_t_uncond.unsqueeze(1), cond_mask.unsqueeze(1)], dim=1)

                else:
                    # 有条件情况：调用 set_input_to_diffmodel
                    # current_sample (x_t) 和 cond_mask 都是 (B, N, K, L)
                    # 假设 set_input_to_diffmodel 内部已修复为返回 (B, 2, N, K, L)
                    diff_input = self.set_input_to_diffmodel(
                        noisy_data=current_sample,  # x_t
                        observed_data=observed_data, # x_cond (实际只用于构造条件输入，本身不参与cat)
                        cond_mask=cond_mask
                    ) # 预期 (B, 2, N, K, L)

                # 4. 调用扩散模型：输入 5D，输出 4D (噪声 ε_θ)
                # diff_input: (B, 2, N, K, L), side_info: (B, C, N, K, L)
                predicted = self.diffmodel(diff_input, side_info, t_batch)  # 预期 (B, N, K, L)

                # DDPM 反向更新
                coeff1 = 1.0 / (self.alpha_hat[t] ** 0.5)
                coeff2 = (1.0 - self.alpha_hat[t]) / (1.0 - self.alpha[t]) ** 0.5

                # current_sample 和 predicted 都是 (B, N, K, L)
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                # if t > 0:
                #     # 噪声和 sigma 都是 (B, N, K, L) 兼容
                #     noise = torch.randn_like(current_sample)
                #     sigma = (
                #         (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                #     ) ** 0.5
                #     current_sample += sigma * noise
                
                # CSDI 的关键：在每一步反向传播后，强制将观测到的部分修正回观测值
                # 这一步是 CSDI 采样时的核心条件步骤。
                # 确保在采样过程中，所有被当作条件的点（即 cond_mask 为 1 的位置）的数据值始终保持为原始的观测值
                current_sample = current_sample * (1.0 - cond_mask) + observed_data * cond_mask

            # 5. 存储样本 (B, n_samples, N, K, L)
            imputed_samples[:, i] = current_sample.detach()

        # 最终返回 (B, n_samples, N, K, L)
        return imputed_samples

    def build_graph_feature(self, observed_data, observed_mask,cond_mask_flat):
        """
        observed_data: (B, K_flat, L)  K_flat = N * F_raw
        observed_mask: (B, K_flat, L)
        返回: gcn_feat_4d: (B, N, F_gcn, L)
        """
        gcn_feat_4d = self.apply_gcn_with_mask_flat(observed_data, observed_mask,cond_mask_flat)  # (B,N,F_out,L)
        return gcn_feat_4d

 
    def apply_gcn_with_mask_flat(self, data_flat, obs_mask_flat, cond_mask_flat):
        """
        data_flat:      (B, K_flat, L)，含 NaN
        obs_mask_flat:  (B, K_flat, L)，1=原始观测有效，0=原始缺失
        cond_mask_flat: (B, K_flat, L)，1=允许作为条件使用；0=作为预测目标（不可泄露）
                        若为 None，则退化为只用 obs_mask_flat（原来的行为）
        return:         (B, N, F_out, L)
        """
        B, K_flat, L = data_flat.shape
        N     = self.num_stations
        F_raw = self.feature_dim          # 原始特征数
        assert K_flat == N * F_raw, f"K_flat={K_flat}, but N*F_raw={N*F_raw}"

        # ---------- 1) 计算“图里可用”的有效掩码 ----------
        # 既要原始有观测，又要当前这次允许作为条件
        effective_mask_flat = obs_mask_flat * cond_mask_flat  # (B, K_flat, L)

        # ---------- 2) 还原成 4D ----------
        x_val_4d    = data_flat.view(B, N, F_raw, L)               # (B,N,F_raw,L)
        eff_mask_4d = effective_mask_flat.view(B, N, F_raw, L).float()  # (B,N,F_raw,L)

        # ---------- 3) 用 mask 决定谁能用，谁要盖掉 ----------
        # 目标：
        #   - eff_mask_4d=1 的位置：保留归一化后的真实数值（包括真实 0）
        #   - eff_mask_4d=0 的位置：无论原来是多少（0 或其它），一律改成 masked_value

        masked_value = self.masked_value.item()
        x_filled = torch.where(
            eff_mask_4d > 0.5,
            x_val_4d,
            torch.full_like(x_val_4d, masked_value),
        )  # (B,N,F_raw,L)

        # ---------- 4) 掩码通道 ----------
        # 给 GCN 一个显式标记：这个特征在这个时刻是否是“可用历史”
        mask_channel = eff_mask_4d  # (B,N,F_raw,L)

        # ---------- 5) 拼接特征：值通道 + 掩码通道 ----------
        x_concat_4d = torch.cat([x_filled, mask_channel], dim=2)   # (B,N,2*F_raw,L)

        F_in = x_concat_4d.shape[2]
        assert hasattr(self, "gcn_in_dim"), "请在 __init__ 里定义 self.gcn_in_dim"
        assert F_in == self.gcn_in_dim, f"GCN 输入维度不匹配: F_in={F_in}, gcn_in_dim={self.gcn_in_dim}"

        # ---------- 6) 展开时间维，送入 GCN ----------
        # 当前形状: (B,N,F_in,L)
        x = x_concat_4d.permute(0, 3, 1, 2)   # (B,L,N,F_in)
        x = x.reshape(B * L, N, F_in)         # (B*L,N,F_in)

        x = self.spatial_gcn(x)               # (B*L,N,F_out)

        # ---------- 7) 还原成 (B,N,F_out,L) ----------
        x = x.reshape(B, L, N, self.gcn_out_dim)  # (B,L,N,F_out)
        x = x.permute(0, 2, 3, 1)                 # (B,N,F_out,L)

        return x

    def forward(self, batch, is_train=1):
        (
            observed_data,   # (B, K_flat, L)，含 NaN
            observed_mask,   # (B, K_flat, L)，1=原始有效
            observed_tp,     # (B, L)
            gt_mask,         # (B, K_flat, L)
            for_pattern_mask,
            cut_length,
            absolute_time
        ) = self.process_data(batch)

        B, K_flat, L = observed_data.shape
        N     = self.num_stations
        F_raw = self.feature_dim
        assert K_flat == N * F_raw, f"K_flat={K_flat}, N={N}, F_raw={F_raw}"

        # ----- 1) 先在 (B,K_flat,L) 上生成 cond_mask（挖洞策略不变） -----
        if is_train == 0:
            cond_mask_flat = gt_mask
        else:
            cond_mask_flat = self.get_randmask(observed_mask)  # (B,K_flat,L)

        # ===== 2) 生成图特征（在原始空间上算 GCN）=====
        gcn_feat_4d = self.build_graph_feature(observed_data, observed_mask,cond_mask_flat)  # (B,N,F_gcn,L)

        # ===== 3) 展平台站维，保持“原始空间”为 (B_eff,F_raw,L) =====
        obs_data_4d  = observed_data.view(B, N, F_raw, L)
        obs_mask_4d  = observed_mask.view(B, N, F_raw, L)
        cond_mask_4d = cond_mask_flat.view(B, N, F_raw, L)

        B_eff = B * N

        obs_data_eff  = obs_data_4d.reshape(B_eff, F_raw, L)      # x_t / 目标空间
        obs_mask_eff  = obs_mask_4d.reshape(B_eff, F_raw, L)
        cond_mask_eff = cond_mask_4d.reshape(B_eff, F_raw, L)

        # 图特征也展平： (B,N,F_gcn,L) -> (B_eff,F_gcn,L)
        Bg, Ng, F_gcn, Lg = gcn_feat_4d.shape
        assert Bg == B and Ng == N and Lg == L
        gcn_feat_eff = gcn_feat_4d.reshape(B_eff, F_gcn, L)

        gcn_feat_eff = self.gcn_proj(gcn_feat_eff)            # (B_eff, F_raw, L)
        gcn_feat_eff = self.gcn_norm(gcn_feat_eff.transpose(1,2)).transpose(1,2)

        # ===== 4) 时间戳展开 =====
        observed_tp_eff = observed_tp.repeat_interleave(N, dim=0)   # (B_eff,L)

        # ===== 5) 构造 side_info：在原来基础上拼上 GCN 通道 =====
        side_info = self.get_side_info(observed_tp_eff, cond_mask_eff)  # (B_eff, C_base, F_raw, L)

        # gcn_feat_eff: (B_eff, F_gcn, L) -> (B_eff, 1, F_gcn, L) 或直接对齐到 K 维
        # 这里为了简单，假设 F_gcn == F_raw，这样就可以当成“每个特征一个 gcn 值”
        # assert F_gcn == F_raw, "建议先把 gcn_out_dim 设成 feature_dim，方便对齐"

        # gcn_side = gcns_feat_eff.unsqueeze(1)  # (B_eff,1,F_raw,L)
        gcn_side = self.gcn_scale * gcn_feat_eff.unsqueeze(1)  # (B_eff,1,K,L)
        side_info = torch.cat([side_info, gcn_side], dim=1)  # (B_eff, C_base+1, F_raw, L)

        # ===== 6) 计算 loss：注意用 obs_data_eff（原始空间），NOT gcn_feat =====
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        loss = loss_func(
            obs_data_eff,     # (B_eff, F_raw, L)，标准化后的“真实空间”
            cond_mask_eff,    # (B_eff, F_raw, L)
            obs_mask_eff,     # (B_eff, F_raw, L)
            side_info,        # (B_eff, *, F_raw, L)，包含 GCN 辅助信息
            is_train,
        )
        return loss

    def evaluate(self, batch, n_samples):
        (
            observed_data,   # (B, K_flat, L)，这里 K_flat = N * F_raw
            observed_mask,   # (B, K_flat, L)
            observed_tp,     # (B, L)
            gt_mask,         # (B, K_flat, L)
            _,
            cut_length,
            absolute_time,
        ) = self.process_data(batch)

        with torch.no_grad():
            # ---------- 1) 构造 cond_mask_flat & target_mask_flat ----------
            cond_mask_flat = gt_mask  # 评估时：gt_mask 就是条件保留位置

            # target_mask_flat：真正要评估的位置 = observed_mask - cond_mask_flat
            target_mask_flat = (observed_mask - cond_mask_flat).clamp(min=0.0)  # (B, K_flat, L)

            # 避免 double evaluation：前 cut_length[i] 个时间步不算进指标
            B, K_flat, L = observed_data.shape
            for i in range(B):
                cl = cut_length[i].item()
                if cl > 0:
                    target_mask_flat[i, :, :cl] = 0.0

            # ---------- 2) 展开多台站 ----------
            N     = self.num_stations
            F_raw = self.feature_dim
            assert K_flat == N * F_raw, f"K_flat={K_flat}, but N*F_raw={N*F_raw}"

            # (B, K_flat, L) -> (B, N, F_raw, L)
            obs_data_4d    = observed_data.view(B, N, F_raw, L)
            obs_mask_4d    = observed_mask.view(B, N, F_raw, L)
            cond_mask_4d   = cond_mask_flat.view(B, N, F_raw, L)
            target_mask_4d = target_mask_flat.view(B, N, F_raw, L)

            B_eff = B * N

            # 这是 CSDI 做预测所处的“原始特征空间”：每个 station 独立一条样本
            observed_data_eff = obs_data_4d.reshape(B_eff, F_raw, L)    # (B_eff, K=F_raw, L)
            observed_mask_eff = obs_mask_4d.reshape(B_eff, F_raw, L)    # (B_eff, K, L)
            cond_mask_eff     = cond_mask_4d.reshape(B_eff, F_raw, L)   # (B_eff, K, L)
            target_mask_eff   = target_mask_4d.reshape(B_eff, F_raw, L) # (B_eff, K, L)

            # ---------- 3) 时间戳展开 ----------
            observed_tp_eff = observed_tp.repeat_interleave(N, dim=0)   # (B_eff, L)

            # ---------- 4) 基础 side_info（不含图） ----------
            # 这里的通道数是 emb_total_dim（time_emb + feature_emb + optional cond_mask）
            base_side_info = self.get_side_info(observed_tp_eff, cond_mask_eff)  # (B_eff, C_base, K, L)

            # ---------- 5) 图特征 -> side_info 的额外通道 ----------
            # 利用已经验证过的 build_graph_feature（内部一般调用 apply_gcn_with_mask_flat）
            gcn_feat_4d = self.build_graph_feature(observed_data, observed_mask,cond_mask_flat)  # (B, N, F_gcn, L)

            B0, N0, F_gcn, L0 = gcn_feat_4d.shape
            assert B0 == B and N0 == N and L0 == L, \
                f"gcn_feat_4d shape mismatch: {(B0,N0,F_gcn,L0)} vs (B={B},N={N},L={L})"

            # 展开 station 维：-> (B_eff, K, L)，此时 K = F_gcn = F_raw
            gcn_feat_eff = gcn_feat_4d.reshape(B_eff, F_gcn, L)   # (B_eff, K, L)
            gcn_feat_eff = self.gcn_proj(gcn_feat_eff)

            # 作为 side_info 的一个新通道： (B_eff, 1, K, L)
            gcn_side = gcn_feat_eff.unsqueeze(1)

            # 拼接到原始 side_info 通道维上
            side_info = torch.cat([base_side_info, gcn_side], dim=1)  # (B_eff, C_base+1, K, L)

            # 如果在 __init__ 里定义了 self.side_dim = self.emb_total_dim + 1，可以 sanity check：
            if hasattr(self, "side_dim"):
                assert side_info.shape[1] == self.side_dim, \
                    f"side_info C={side_info.shape[1]}, side_dim={self.side_dim}"

            # ---------- 6) 调用 impute，在原始特征空间插补 ----------
            samples = self.impute(
                observed_data_eff,  # (B_eff, K, L) 标准化后的原始数据
                cond_mask_eff,      # (B_eff, K, L)
                side_info,
                n_samples
            )   # (B_eff, nsample, K, L)

        # 把 target_mask_eff 一起返回，外层直接用它来计算 MAE/RMSE / 写 CSV
        return (
            samples,             # (B_eff, nsample, K, L)
            observed_data_eff,   # (B_eff, K, L)   —— 原始空间真实值（标准化后）
            cond_mask_eff,       # (B_eff, K, L)
            observed_mask_eff,   # (B_eff, K, L)
            observed_tp_eff,     # (B_eff, L)
            absolute_time,       # (B, L) 原始窗口时间信息
            target_mask_eff      # (B_eff, K, L)，已应用 cut_length
        )


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim,edge_index,edge_weight):
        super(CSDI_PM25, self).__init__(target_dim, config, device,edge_index,edge_weight)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        absolute_time = batch["absolute_time"]
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            absolute_time
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        absolute_time = batch['absolute_time']
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            absolute_time
        )


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim,edge_index,edge_weight):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device,edge_index,edge_weight)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()  # (B,L,K)
        observed_mask = batch["observed_mask"].to(self.device).float()  # (B,L,K)
        observed_tp = batch["timepoints"].to(self.device).float()  # (B,L)
        gt_mask = batch["gt_mask"].to(self.device).float()  # (B,L,K)
        absolute_time = batch["absolute_time"]

        observed_data = observed_data.permute(0, 2, 1)  # → (B,K,L)
        observed_mask = observed_mask.permute(0, 2, 1)  # → (B,K,L)
        gt_mask = gt_mask.permute(0, 2, 1)  # → (B,K,L)

        # cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        cut_length = batch["cut_length"].to(self.device).long()
        
        for_pattern_mask = observed_mask

        feature_id = torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0], -1).to(self.device)
        # feature_id : (B, K) = 通道索引，每个样本里存的是通道编号，用于后面随机抽部分特征再做 embedding。

        return (
            observed_data,  # (B,K_total,L)
            observed_mask,  # (B,K_total,L)
            observed_tp,  # (B,L)
            gt_mask,  # (B,K_total,L)
            for_pattern_mask, # for_pattern_mask 简单设置为 observed_mask
            cut_length,
            feature_id,  # ★ 额外返回“特征编号”
            absolute_time
        )

    def sample_features(self, observed_data, observed_mask, feature_id, gt_mask):
        # 选择需要保留的特征嵌入数量
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []

        for k in range(len(observed_data)):
            # 每个样本随机打乱通道顺序，然后取前 size 个。通过 feature_id 保留“原通道是谁”的信息，后面嵌入用。
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k, ind[:size]])
            extracted_mask.append(observed_mask[k, ind[:size]])
            extracted_feature_id.append(feature_id[k, ind[:size]])
            extracted_gt_mask.append(gt_mask[k, ind[:size]])

        extracted_data = torch.stack(extracted_data, 0)  # (B,size,L)
        extracted_mask = torch.stack(extracted_mask, 0)  # (B,size,L)
        extracted_feature_id = torch.stack(extracted_feature_id, 0)  # (B,size)
        extracted_gt_mask = torch.stack(extracted_gt_mask, 0)  # (B,size,L)
        return extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask

    def get_forecast_cond_mask_tail(self,observed_mask, horizon):
        """
        observed_mask: (B, K, L)  1=有真值，0=原始缺失
        horizon: 预测步数 H（从序列尾部往前数 H 个时间步作为预测目标）

        返回:
            cond_mask: (B, K, L)
        """
        B, K, L = observed_mask.shape
        assert horizon > 0 and horizon <= L

        cond_mask = observed_mask.clone()

        # 最后 H 步不允许作为条件，只能预测
        cond_mask[..., -horizon:] = 0

        return cond_mask

    # （支持特征子采样）
    def get_side_info(self, observed_tp, cond_mask):
        """
        构造 5D 的 side_info，保持 N 和 K 维度独立。
        
        参数:
            observed_tp: (B, L)
            cond_mask:   (B, N, K, L)
            
        返回:
            side_info: (B, side_dim, N, K, L)
                       其中 side_dim = emb_time_dim + emb_feature_dim + 1 (若有condition)
        """
        B, N, K, L = cond_mask.shape
        device = cond_mask.device

        # ===== 1) 时间嵌入 (Time Embedding) =====
        # 原始: (B, L, emb_time_dim)
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        
        # 目标: (B, emb_time_dim, N, K, L)
        # 逻辑: 时间 t 对于所有的站点 N 和特征 K 都是一样的，所以在 N, K 维度复制
        time_embed = time_embed.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L, emb_time_dim)
        time_embed = time_embed.expand(-1, N, K, -1, -1)   # (B, N, K, L, emb_time_dim)
        time_embed = time_embed.permute(0, 4, 1, 2, 3)     # (B, emb_time_dim, N, K, L)

        # ===== 2) 特征嵌入 (Feature Embedding) =====
        # 原始: (K, emb_feature_dim)
        # 这里假设特征没有被 shuffle，直接用 arange(K)
        # 如果你有 self.target_dim_base 或 feature_id，请在这里替换
        feature_ids = torch.arange(self.target_dim_base, device=device)
        feature_embed = self.embed_layer(feature_ids)      # (K, emb_feature_dim)
        
        # 目标: (B, emb_feature_dim, N, K, L)
        # 逻辑: 特征 k 对于所有的 batch B, 站点 N, 时间 L 都是一样的
        feature_embed = feature_embed.reshape(1, 1, K, 1, -1)        # (1, 1, K, 1, emb_feature_dim)
        feature_embed = feature_embed.expand(B, N, -1, L, -1)        # (B, N, K, L, emb_feature_dim)
        feature_embed = feature_embed.permute(0, 4, 1, 2, 3)         # (B, emb_feature_dim, N, K, L)

        # ===== 3) 拼接 Side Info =====
        if not self.is_unconditional:
            # cond_mask: (B, N, K, L) -> (B, 1, N, K, L)
            # 保持 mask 的细粒度，不要做 any() 聚合
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([time_embed, feature_embed, side_mask], dim=1)
        else:
            side_info = torch.cat([time_embed, feature_embed], dim=1)

        # 最终形状检查: (B, total_channels, N, K, L)
        return side_info

    def forward(self, batch, global_step=0, total_steps=0, is_train=1):
        (
            observed_data,     # (B, K_flat, L)
            observed_mask,     # (B, K_flat, L)
            observed_tp,       # (B, L)
            gt_mask,           # (B, K_flat, L)
            _,
            cut_length,
            _,
            absolute_time
        ) = self.process_data(batch)

        self.num_station = 108
        self.feature_dim = 4
        B, K_flat, L = observed_data.shape
        N = self.num_station
        K = self.feature_dim   # 4

        if is_train == 0:
            cond_mask_flat = gt_mask
        else:
            cond_mask_flat = self.get_forecast_cond_mask_tail(
                observed_mask, horizon=1
            )  # (B, K_flat, L)
            diff = (cond_mask_flat - gt_mask).abs().sum()
            if diff > 0:
                print("Warning: train/eval cond_mask 分布差异较大，建议统一规则")

        # ===== 3) 展平台站维，保持“原始空间”为 (B_eff,F_raw,L) =====
        # 1. reshape → (B, N, K, L)
        obs_data  = observed_data.view(B, N, K, L)
        obs_mask  = observed_mask.view(B, N, K, L)
        cond_mask = cond_mask_flat.view(B, N, K, L)

        # 2. 生成 cond_info（站点维度 N）
        side_info = self.get_side_info(
            observed_tp,   # (B, L)
            cond_mask      # (B, N, K, L)
        )               

        # ===== 6) 计算 loss：注意用 obs_data_eff（原始空间），NOT gcn_feat =====
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        loss = loss_func(
            obs_data,     
            cond_mask,   
            obs_mask,     
            side_info,        
            is_train,
        )
        return loss

    @torch.no_grad()
    def evaluate(self, batch, n_samples):
        (
            observed_data,   # (B, K_flat, L)  = (B, N*K, L)
            observed_mask,
            observed_tp,
            gt_mask,         # (B, K_flat, L)
            for_pattern_mask,
            cut_length,
            _,
            absolute_time
        ) = self.process_data(batch)

        B, K_flat, L = observed_data.shape
        N = self.num_stations
        K = self.feature_dim
        assert K_flat == N * K

        # ======== reshape 到 (B,N,K,L) ========
        obs_data = observed_data.view(B, N, K, L)
        obs_mask = observed_mask.view(B, N, K, L)
        cond_mask = gt_mask.view(B, N, K, L)

        # target_mask (B,N,K,L)
        target_mask = (obs_mask - cond_mask).clamp(min=0.)

        # 应用 cut_length
        for i in range(B):
            cl = cut_length[i].item()
            if cl > 0:
                target_mask[i, :, :, :cl] = 0

        # (B, side_dim, N, K, L)   
        side_info = self.get_side_info(observed_tp, cond_mask)

        # ======== 调用新结构 impute()，输出 (B,ns,N,K,L) ========
        samples_bnkl = self.impute(
            obs_data,        # (B,N,K,L)
            cond_mask,       # (B,N,K,L)
            side_info,       # (B, side_dim, N, K, L) <-- 5D 输入
            n_samples
        )  # 结果：(B, n_samples, N, K, L)

        # ======== flatten 回原始格式 B_eff = B×N ========
        B_eff = B * N

        # samples_bnkl: (B,ns,N,K,L)
        samples_bnkl = samples_bnkl.permute(0,2,1,3,4)
        samples = samples_bnkl.reshape(B_eff, n_samples, K, L)

        # 其它需要 flatten 的张量
        observed_data_eff = obs_data.reshape(B_eff, K, L)
        cond_mask_eff     = cond_mask.reshape(B_eff, K, L)
        observed_mask_eff = obs_mask.reshape(B_eff, K, L)
        target_mask_eff   = target_mask.reshape(B_eff, K, L)

        # 时间戳：简单 repeat_interleave
        observed_tp_eff = observed_tp.repeat_interleave(N, dim=0)  # (B_eff,L)

        # absolute_time 不需要 flatten（你的评估器预期就是 B,L）
        # 若你想为每个站点复制，也可以 repeat_interleave，但我保持原版行为

        return (
            samples,             # (B_eff, nsample, K, L)
            observed_data_eff,   # (B_eff, K, L)
            cond_mask_eff,       # (B_eff, K, L)
            observed_mask_eff,   # (B_eff, K, L)
            observed_tp_eff,     # (B_eff, L)
            absolute_time,       # (B, L)
            target_mask_eff      # (B_eff, K, L)
        )
