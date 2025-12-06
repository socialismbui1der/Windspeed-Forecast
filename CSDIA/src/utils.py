import os

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd

from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import wandb
import time

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=2,
    foldername="",
    patience = 8
):

    wandb.init(
        project="csdi_wind",
        name=f"exp_{int(time.time())}",
        config=config,
        settings=wandb.Settings(start_method="fork")
    )
    

    config = config["train"]
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-7)
    if foldername != "":
        output_path = foldername + "/model.pth"

    # -------------------------------
    # ✅ 修改为余弦退火学习率调度器
    # T_max 通常设为总 epoch 数，eta_min 是最小学习率
    # -------------------------------
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],   # 一个完整的余弦周期长度（一般设为总训练轮数）
        eta_min=1e-5              # 学习率最小值，可根据需要调节
    )

    scaler = GradScaler()

    # 下面是重复“退火–恢复”循环的版本
    # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-5)

    best_valid_loss = 1e10
    patience_count = 0
    
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad(set_to_none=True)

                with autocast(dtype=torch.float16):
                    # loss = model(train_batch,epoch_no,config["epochs"])
                    loss = model(train_batch)
                

                # loss.backward()
                scaler.scale(loss).backward()
                avg_loss += loss.item()
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                if batch_no % 10 == 0:
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "step": epoch_no * len(train_loader) + batch_no
                    })

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                        "lr": optimizer.param_groups[0]["lr"],  # ✅ 可实时显示当前学习率
                    },
                    refresh=False,
                )

            # ✅ 每个 epoch 调整一次学习率
            lr_scheduler.step()

        wandb.log({
            "train/loss_epoch": avg_loss / batch_no,
            "epoch": epoch_no
        })

        if (epoch_no + 1 )>5 and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        with autocast(dtype=torch.float16):
                            loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
                        
            # 用“平均验证 loss”做 early stopping 判断更合理一点
            mean_valid_loss = avg_loss_valid / batch_no

            wandb.log({
                "valid/loss": mean_valid_loss,
                "epoch": epoch_no
            })
            if best_valid_loss > mean_valid_loss:
                patience_count = 0
                best_valid_loss = mean_valid_loss
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                    f"\n 模型权重已保存到{output_path}"
                )
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
            else:
                patience_count += 1
                if patience_count > patience:
                    print("early stop at ", epoch_no)
                    break
                
    wandb.finish()


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def _to_numpy_1d_or_scalar(x):
    """把 x 变成 numpy 标量或 (K,) 向量。允许 x 是 float / np / torch.Tensor。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 0:
        return x.astype(float)
    x = x.squeeze()
    return x.astype(float)


def _broadcast_scaler(x, K: int):
    """把标量/向量广播到 (B,L,K) 的第三维。"""
    x = np.asarray(x)
    if x.ndim == 0:
        return x.astype(float) # 标量，后面 numpy 自动广播
    if x.ndim == 1:
        n = x.shape[0]

        # 完全对齐 K：已经是每通道一个尺度
        if n == K:
            return x.astype(float).reshape(1, 1, K)

        # 典型情况：n = FEAT_DIM = 4，K = 432 = 4 * 108
        if K % n == 0:
            reps = K // n
            x_rep = np.tile(x, reps)          # [x0,x1,x2,x3, x0,x1,x2,x3, ...]
            return x_rep.astype(float).reshape(1, 1, K)

        # 其他情况一律认为是错误配置
        raise ValueError(
            f"scaler/mean_scaler 长度 {n} 与特征数 K={K} 不匹配，且不能整除 K"
        )

    raise ValueError(f"不支持的 scaler 形状: {x.shape}")


def _parse_absolute_time(absolute_time, B_batch: int, L: int):
    """
    把 model.evaluate 返回的 absolute_time 统一转成 (B_batch, L) 的二维数组（字符串）。
    适配三种常见情况：

    1) DataLoader 默认 collate 的结构：
       - absolute_time 是一个长度 L 的 list
       - 每个元素是一个 tuple，长度 = B_batch
       - 逻辑结构是 (L, B) → 转置成 (B, L)

    2) batch_size == 1，Dataset 直接返回 list[str] / list[标量]：
       - absolute_time 是长度 L 的 list
       - 直接当成一条时间轴 (1, L)

    3) 已经是 per-sample 的 list：
       - absolute_time 长度 = B_batch
       - 每个元素是 list / tuple / ndarray，长度 = L
    """
    # ---------- list 情况 ----------
    if isinstance(absolute_time, list):
        outer_len = len(absolute_time)
        first = absolute_time[0]

        # case 1: DataLoader collate 后的典型结构 -> [tuple(B), ...]，长度 L
        if outer_len == L and isinstance(first, (tuple, list)):
            # 结构是 (L, B_batch) ，转成 (B_batch, L)
            arr = np.array(absolute_time, dtype=object)  # (L, B)
            if arr.shape[1] != B_batch:
                raise ValueError(
                    f"absolute_time 形状 {arr.shape} 与 batch 大小 {B_batch} 不匹配"
                )
            arr = arr.T  # -> (B_batch, L)
            return arr.astype(str)

        # case 2: batch_size == 1，Dataset 返回 list[str] / list[标量]
        if outer_len == L and B_batch == 1 and not isinstance(first, (list, tuple)):
            row = [str(x) for x in absolute_time]
            return np.array(row, dtype=object).reshape(1, L)  # (1, L)

        # case 3: per-sample list：len == B_batch，每个是长度 L 的序列
        if outer_len == B_batch and isinstance(first, (list, tuple, np.ndarray)):
            rows = []
            for seq in absolute_time:
                seq = list(seq)
                if len(seq) != L:
                    raise ValueError(
                        f"某个样本的 absolute_time 长度为 {len(seq)}，但 L={L}"
                    )
                row = []
                for item in seq:
                    if isinstance(item, tuple):
                        val = item[0] if len(item) > 0 else ""
                    else:
                        val = item
                    row.append(str(val))
                rows.append(row)
            return np.array(rows, dtype=object)  # (B_batch, L)

        raise ValueError(
            f"absolute_time 是 list，但结构不符合预期：len={outer_len}，"
            f"第0个元素类型={type(first)}。"
        )

    # ---------- Tensor / ndarray 情况 ----------
    if isinstance(absolute_time, torch.Tensor):
        arr = absolute_time.detach().cpu().numpy()
    elif isinstance(absolute_time, np.ndarray):
        arr = absolute_time
    else:
        raise ValueError(f"无法识别 absolute_time 类型: {type(absolute_time)}")

    if arr.ndim == 1:
        if arr.shape[0] != L:
            raise ValueError(f"absolute_time 一维长度 {arr.shape[0]} 与 L={L} 不匹配")
        row = arr.astype(str)
        return np.tile(row[None, :], (B_batch, 1))  # (B_batch, L)
    elif arr.ndim == 2:
        if arr.shape != (B_batch, L):
            raise ValueError(
                f"absolute_time 形状 {arr.shape} 与 (B_batch={B_batch}, L={L}) 不匹配"
            )
        return arr.astype(str)

    raise ValueError(f"不支持的 absolute_time 维度: {arr.shape}")


def evaluate(model,
             test_loader,
             nsample=3,
             scaler=None,
             mean_scaler=None,
             foldername="",
             full_datetime_index=None,
             station_id=1,
             ):
    """
    model.evaluate 返回：
        samples, c_target, cond_mask, observed_mask, observed_time, absolute_time, target_mask
    其中：
        samples:      (B_eff, nsample, K, L)
        c_target:     (B_eff, K, L)     # observed_data_eff（GCN+展平）
        cond_mask:    (B_eff, K, L)
        observed_mask:(B_eff, K, L)
        observed_time:(B_eff, L)
        target_mask:  (B_eff, K, L)     # 已减 cond_mask 且 cut_length 前段=0

    scaler, mean_scaler:
        一维张量，形状 (K,)，例如：
        scaler      = tensor([12.6, 24.1, 32.7, 1.63], device='cuda:0')
        mean_scaler = tensor([13.7, 57.0, 910.3, 2.36], device='cuda:0')

    station_id:
        1..N（台站编号），只导出这一个台站的数据到 CSV。
    """

    if full_datetime_index is None:
        raise ValueError("需要传入 full_datetime_index 才能按原始时间轴重建结果表。")

    device = next(model.parameters()).device

    if scaler is None or mean_scaler is None:
        raise ValueError("scaler 和 mean_scaler 不能为 None，必须传入。")

    # 确保是在 model 同一设备上的 1D 向量
    scaler_tensor = torch.as_tensor(scaler, dtype=torch.float32, device=device)
    mean_scaler_tensor = torch.as_tensor(mean_scaler, dtype=torch.float32, device=device)

    # 用于 CSV 的 numpy 版本
    scaler_np_global = scaler_tensor.detach().cpu().numpy()   # (K,)
    mean_scaler_np_global = mean_scaler_tensor.detach().cpu().numpy()  # (K,)

    # 指标累加器
    mse_total = 0.0
    mae_total = 0.0
    ss_res_total = 0.0
    ss_tot_total = 0.0
    evalpoints_total = 0.0

    # 单台站 CSV 的 df 列表
    df_list_station = []

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:

            for batch_no, test_batch in enumerate(it, start=1):
                # ================== 1. 调用模型内部 evaluate =================
                # with autocast(dtype=torch.float16):
                (
                    samples,        # (B_eff, nsample, K, L)
                    c_target,       # (B_eff, K, L)
                    cond_mask,      # (B_eff, K, L)
                    observed_mask,  # (B_eff, K, L)
                    observed_time,  # (B_eff, L)
                    absolute_time,  # 原始时间信息
                    target_mask     # (B_eff, K, L)，已处理 cut_length、cond_mask
                ) = model.evaluate(test_batch, nsample)
                # ===== 2. 出了 autocast 以后，全转回 float32 再算指标 =====
                # samples      = samples.float()
                # c_target     = c_target.float()
                # cond_mask    = cond_mask.float()
                # observed_mask= observed_mask.float()
                # target_mask  = target_mask.float()

                B_eff, K, L = c_target.shape

                # 需要知道 N 和 B（原始 batch 大小）：
                N = model.num_stations           # 你在 CSDI_base.__init__ 里应该有 num_stations
                assert B_eff % N == 0, f"B_eff={B_eff} 不能整除 N={N}"
                B = B_eff // N

                # ================== 2. 统一到 (B_eff, L, K) 方便计算 ==================
                samples = samples.permute(0, 1, 3, 2)           # (B_eff, nsample, L, K)
                c_target = c_target.permute(0, 2, 1)            # (B_eff, L, K)
                target_mask = target_mask.permute(0, 2, 1)      # (B_eff, L, K)

                # ================== 3. 沿采样维度取中位数：插补点估计 ==================
                samples_median = samples.median(dim=1)
                pred_batch = samples_median.values              # (B_eff, L, K)

                # ================== 4. 误差（在 target_mask==1 的点上） ==================
                # 扩展 scaler_tensor 到 (1,1,K) 方便广播
                st = scaler_tensor
                while st.dim() < 3:
                    st = st.unsqueeze(0)    # 最终形状 (1,1,K)

                diff = (pred_batch - c_target) * target_mask            # (B_eff,L,K)
                diff_rescaled = diff * st                               # 反标准化后的残差

                mse_current = diff_rescaled ** 2                        # (B_eff,L,K)
                mae_current = diff_rescaled.abs()                       # (B_eff,L,K)
                ss_res_current = mse_current

                # ---- SS_tot（简化版 R^2，用当前 batch 中 target_mask==1 的点）----
                valid_y = c_target * target_mask                         # (B_eff,L,K)
                sum_points = valid_y.sum()
                count_points = target_mask.sum()

                if count_points > 0:
                    y_mean = sum_points / count_points
                    ss_tot_current = (((c_target - y_mean) * target_mask) * st) ** 2
                    ss_tot_total += ss_tot_current.sum().item()

                # ---- 累加全局统计 ----
                ss_res_total += ss_res_current.sum().item()
                mse_total += ss_res_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += target_mask.sum().item()

                # ================== 5. 更新进度条 ==================
                current_rmse = np.sqrt(mse_total / evalpoints_total) if evalpoints_total > 0 else 0.0
                current_mae = mae_total / evalpoints_total if evalpoints_total > 0 else 0.0
                current_r2 = 1.0 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0.0

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": current_rmse,
                        "mae_total": current_mae,
                        "r2_total": current_r2,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

                # ================== 6. 单台站 CSV 数据收集 ==================
                # 只保存 station_id 这一台站的数据
                station_idx = station_id - 1                     # 0-based
                if not (0 <= station_idx < N):
                    raise ValueError(f"station_id={station_id} 超出范围 1..{N}")

                # B_eff 展开方式：b_eff = b * N + n
                # 所以当前批次中该台站的索引：
                idx = torch.arange(0, B_eff, N, device=pred_batch.device) + station_idx
                # idx 形状 (B,)

                # 取出该台站的预测 / 真实 / mask
                pred_single = pred_batch[idx]        # (B, L, K)
                true_single = c_target[idx]          # (B, L, K)
                mask_single = target_mask[idx]       # (B, L, K)

                # 反到 CPU+numpy，用于 DataFrame
                pred_single_np = pred_single.detach().cpu().numpy()
                true_single_np = true_single.detach().cpu().numpy()
                mask_single_np = mask_single.detach().cpu().numpy().astype(bool)

                # 时间戳解析：absolute_time -> (B,L)
                abs_time_arr = _parse_absolute_time(absolute_time, B, L)  # (B,L)

                # 反标准化
                # scaler_np_global / mean_scaler_np_global: (K,)
                S = scaler_np_global.reshape(1, 1, K)         # (1,1,K)
                M = mean_scaler_np_global.reshape(1, 1, K)    # (1,1,K)

                true_denorm = true_single_np * S + M          # (B,L,K)
                pred_denorm = pred_single_np * S + M          # (B,L,K)

                # 构造 DataFrame（逐窗口）
                # 这里假设 K=4，对应 TEM/RHU/PRS/WINS；否则自动用 feat0..featK-1
                default_feature_names = ["TEM", "RHU", "PRS", "WINS"]
                if len(default_feature_names) == K:
                    feature_names = default_feature_names
                else:
                    feature_names = [f"feat{k}" for k in range(K)]

                for b in range(B):
                    datetimes = abs_time_arr[b]  # 长度 L
                    row_dict = {"Datetime": datetimes}

                    for k_i in range(K):
                        base = feature_names[k_i]

                        true_col = true_denorm[b, :, k_i]  # (L,)
                        pred_col = pred_denorm[b, :, k_i]  # (L,)
                        m = mask_single_np[b, :, k_i]      # (L,)

                        true_out = np.full(L, np.nan, dtype=float)
                        pred_out = np.full(L, np.nan, dtype=float)
                        err_out = np.full(L, np.nan, dtype=float)

                        true_out[m] = true_col[m]
                        pred_out[m] = pred_col[m]
                        err_out[m] = pred_out[m] - true_out[m]

                        row_dict[f"{base}_True"]    = true_out
                        row_dict[f"{base}_Imputed"] = pred_out
                        row_dict[f"{base}_Error"]   = err_out

                    df_list_station.append(pd.DataFrame(row_dict))

    # ================== 7. 拼接单台站 DataFrame，按完整时间轴对齐并保存 CSV ==================
    if not df_list_station:
        print("⚠️ 没有生成任何数据行，检查 test_loader / model.evaluate。")
        return

    df_all = pd.concat(df_list_station, axis=0, ignore_index=True)
    df_all["Datetime"] = pd.to_datetime(df_all["Datetime"])
    value_cols = [c for c in df_all.columns if c != "Datetime"]
    df_all = df_all.dropna(subset=value_cols, how="all")

    # 用 full_datetime_index 作为完整时间线对齐
    full_dt = pd.to_datetime(full_datetime_index)
    df_timeline = pd.DataFrame({"Datetime": full_dt})

    df_final = pd.merge(df_timeline, df_all, on="Datetime", how="left")
    df_final = df_final.sort_values("Datetime")
    df_final["Datetime"] = df_final["Datetime"].dt.strftime("%Y/%m/%d %H:%M")

    # 保存 CSV（单台站）
    os.makedirs(foldername, exist_ok=True)
    csv_path = os.path.join(foldername, f"csdi_imputed_station_{station_id}.csv")
    df_final.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ================== 8. 最终指标 ==================
    final_rmse = np.sqrt(mse_total / evalpoints_total)
    final_mae = mae_total / evalpoints_total
    final_r2 = 1.0 - (ss_res_total / ss_tot_total) if ss_tot_total > 0 else 0.0

    print("\n--- Final Evaluation Metrics ---")
    print(f"Station: {station_id}/{N}")
    print(f"RMSE:    {final_rmse:.4f}")
    print(f"MAE:     {final_mae:.4f}")
    print(f"R^2:     {final_r2:.4f}")
    print("--------------------------------")