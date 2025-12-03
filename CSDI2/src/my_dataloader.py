import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import os

# --- 配置常量 ---
ATTRIBUTES = ['TEM', 'RHU', 'PRS', 'WINS']
FEAT_DIM = len(ATTRIBUTES)
# 假设你的数据文件名和路径
Dir_PATH = "/workspace/six_features/only-shanxi/all_six"
EARTH_RADIUS_KM = 6371.0

def haversine_distance(lat1, lon1, lat2, lon2):
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def build_station_graph_from_csv(
    csv_path,
    k=5,
    self_loop=True,
    use_gaussian_weight=True,
):
    """
    根据台站经纬度构造图:
    - k 近邻无向图
    返回:
        edge_index  : (2, E) torch.long
        edge_weight : (E,)   torch.float
    另外顺手返回一个稠密 adj_np，备用（如果你别处要用）。
    """
    df = pd.read_csv(csv_path)

    lats = df["lat"].values.astype(np.float64)
    lons = df["lon"].values.astype(np.float64)

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    lat1 = lat_rad[:, None]
    lat2 = lat_rad[None, :]
    lon1 = lon_rad[:, None]
    lon2 = lon_rad[None, :]

    dist_mat = haversine_distance(lat1, lon1, lat2, lon2)  # (N,N)
    N = dist_mat.shape[0]

    adj = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        d = dist_mat[i].copy()
        d[i] = np.inf
        nn_idx = np.argsort(d)[:k]
        adj[i, nn_idx] = 1.0

    # 对称化
    adj = np.maximum(adj, adj.T)

    if self_loop:
        np.fill_diagonal(adj, 1.0)

    if use_gaussian_weight:
        d_nonzero = dist_mat[adj > 0]
        sigma = np.median(d_nonzero) if d_nonzero.size > 0 else 1.0
        sigma = max(sigma, 1e-6)
        weight = np.exp(-(dist_mat ** 2) / (2 * sigma ** 2)).astype(np.float32)
        weight[adj == 0] = 0.0
    else:
        weight = adj.copy()

    src, dst = np.nonzero(adj)
    edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))  # (2,E)
    edge_weight = torch.from_numpy(weight[src, dst].astype(np.float32))    # (E,)

    return adj.astype(np.float32), edge_index, edge_weight

# --- 辅助函数：获取全部数据和索引划分 ---
def get_all_data_and_indices(dir_path, eval_length, train_ratio=0.7, valid_ratio=0.1):
    """
    现在 dir_path 下存的是 4 个文件，每个文件形状 (T, num_stations)，
    分别对应 ATTRIBUTES 里的 4 个特征（例如 TEM / RHU / PRS / WINS）。
    
    本函数会把它们重新组织成：
        full_data_with_nan: (T, num_stations * FEAT_DIM)
    且列顺序为： [站1的4维, 站2的4维, ..., 站N的4维]，
    这样与原来“每个文件是一个站点、列是4个特征”的效果一致。
    """

    # 1. 先把每个特征文件读进来： attr_name -> DataFrame(T, num_stations)
    feature_dfs = {}  # { 'TEM': df_tem, 'RHU': df_rhu, ... }

    for fname in os.listdir(dir_path):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(dir_path, fname)
        for attr in ATTRIBUTES:
            # 简单匹配：文件名里包含特征名，比如 TEM_xxx.csv、RHU.csv 等
            if attr in fname:
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                feature_dfs[attr] = df
                break

    # 简单防呆：确认 4 个特征都找到了
    if len(feature_dfs) != len(ATTRIBUTES):
        raise ValueError(f"在目录 {dir_path} 下没有找到所有特征文件，期望 {ATTRIBUTES}，实际 {list(feature_dfs.keys())}")

    # 2. 统一时间索引和台站列顺序
    ref_attr = ATTRIBUTES[0]
    ref_df = feature_dfs[ref_attr]
    full_datetime_index = ref_df.index.values  # 时间轴
    station_ids = list(ref_df.columns)         # 台站顺序
    T = len(full_datetime_index)
    num_stations = len(station_ids)

    # 3. 按“站点优先”的方式重新拼接： [站1的4维, 站2的4维, ...]
    full_data_list = []
    full_mask_list = []

    for sid in station_ids:
        station_feat_list = []
        station_mask_list = []
        for attr in ATTRIBUTES:
            df_attr = feature_dfs[attr]

            # 保证时间和列顺序一致，如果不一致你就得先在外面清洗数据了
            series = df_attr[sid]                     # (T,)
            values = series.values.reshape(T, 1)      # (T,1)
            mask = (~series.isna()).astype(np.float32).values.reshape(T, 1)  # 1=观测,0=NaN

            station_feat_list.append(values)  # 该站的一个特征
            station_mask_list.append(mask)

        # 该站的 4 维特征： (T, FEAT_DIM)
        station_data = np.concatenate(station_feat_list, axis=1)
        station_c_mask = np.concatenate(station_mask_list, axis=1)

        full_data_list.append(station_data)
        full_mask_list.append(station_c_mask)

    # 沿特征维拼接所有站点 → (T, num_stations * FEAT_DIM)
    full_data_with_nan = np.concatenate(full_data_list, axis=1)
    full_c_mask = np.concatenate(full_mask_list, axis=1)

    K_total = full_data_with_nan.shape[1]
    print(f"[INFO] Loaded {num_stations} stations × {FEAT_DIM} features -> feature dim = {K_total}")

    # 4. 滑动窗口索引（这部分保持不变）
    N = T - eval_length + 1
    all_indices = np.arange(N)
    n_train = int(N * train_ratio)
    n_valid = int(N * valid_ratio)

    train_indices = all_indices[:n_train]
    valid_indices = all_indices[n_train:n_train + n_valid]
    test_indices = all_indices[n_train + n_valid:]

    train_start_idx = train_indices[0] if len(train_indices) > 0 else -1
    valid_start_idx = valid_indices[0] if len(valid_indices) > 0 else -1
    test_start_idx = test_indices[0] if len(test_indices) > 0 else -1

    return (
        full_data_with_nan,
        full_c_mask,
        full_datetime_index,
        train_indices,
        valid_indices,
        test_indices,
        train_start_idx,
        valid_start_idx,
        test_start_idx,
    )


# --- 辅助函数：仅在训练集上计算 Mean/Std ---
def calculate_train_mean_std(full_data_with_nan, full_c_mask, train_indices, eval_length):
    """
    在多台站场景下计算训练集上的均值和方差。

    - full_data_with_nan: (T, num_stations * FEAT_DIM)
    - full_c_mask       : 同形状，1=观测, 0=原始缺失
    - 返回:
        mean: (FEAT_DIM,)，每个原始气象特征一个均值（跨所有站点 + 时间）
        std : (FEAT_DIM,)
    结果会保存在 STATS_CACHE_PATH，下次优先从磁盘加载。
    """
    STATS_CACHE_PATH = "/workspace/CSDI2/Cache/train_mean_std_multi_station.pkl"
    # ---------- 先尝试从缓存读取 ----------
    if os.path.exists(STATS_CACHE_PATH):
        try:
            with open(STATS_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            mean = cache["mean"]
            std = cache["std"]
            print(f"[INFO] Loaded cached train mean/std from {STATS_CACHE_PATH}")
            return mean, std
        except Exception as e:
            print(f"[WARN] Failed to load mean/std cache: {e}, recomputing...")

    print("Calculating mean and std from training set (multi-station)...")

    # 收集所有训练窗口中的观测值
    train_values = []
    train_masks = []

    for start_index in train_indices:
        seq = full_data_with_nan[start_index: start_index + eval_length]   # (L, K_total)
        mask = full_c_mask[start_index: start_index + eval_length]        # (L, K_total)

        train_values.append(seq)
        train_masks.append(mask)

    # 拼成 (M, K_total)
    tmp_values = np.concatenate(train_values, axis=0)
    tmp_masks = np.concatenate(train_masks, axis=0)

    M, K_total = tmp_values.shape
    if K_total % FEAT_DIM != 0:
        raise ValueError(
            f"K_total={K_total} 无法被 FEAT_DIM={FEAT_DIM} 整除，"
            f"请检查 full_data_with_nan 的列布局是否是 [站1的{FEAT_DIM}维, 站2的{FEAT_DIM}维, ...]"
        )

    num_stations = K_total // FEAT_DIM

    # 重新整理成 (M, num_stations, FEAT_DIM)
    tmp_values = tmp_values.reshape(M, num_stations, FEAT_DIM)
    tmp_masks = tmp_masks.reshape(M, num_stations, FEAT_DIM)

    mean = np.zeros(FEAT_DIM, dtype=np.float32)
    std = np.zeros(FEAT_DIM, dtype=np.float32)

    # 对每个“原始特征”维度（TEM/RHU/PRS/WINS）做统计，跨所有站点 + 时间
    for k in range(FEAT_DIM):
        # 取出该特征在所有站点的观测值
        c_data = tmp_values[:, :, k][tmp_masks[:, :, k] == 1]

        if c_data.size == 0:
            mean[k] = 0.0
            std[k] = 1.0
        else:
            m = c_data.mean()
            s = c_data.std()
            mean[k] = m
            std[k] = s if s > 1e-6 else 1.0

    # ---------- 持久化到磁盘 ----------
    try:
        with open(STATS_CACHE_PATH, "wb") as f:
            pickle.dump({"mean": mean, "std": std}, f)
        print(f"[INFO] Saved train mean/std to {STATS_CACHE_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save mean/std cache: {e}")

    return mean, std

# 这是插补任务另一个创造验证/测试集的掩码的函数，随机缺失和连续块损失共存
def create_full_gt_mask2(
    full_c_mask,
    row_indices,
    missing_ratio,
    block_prob=0.3,      # 连续块缺失点数占总缺失点数的大约比例（0~1）
    max_block_len=4,    # 连续块的最大长度（沿时间轴 / 行方向）
    seed=None,
):
    """
    在 `row_indices` 指定的行范围内、针对原始观测点(=1)，生成混合缺失 Mask：
    - 一部分缺失来自沿时间轴的「连续块」
    - 剩余缺失来自「随机散点」
    
    参数:
        full_c_mask (np.ndarray): 原始观测 Mask (1=观测, 0=原始缺失)。形状 (H, W)。
        row_indices (list/np.ndarray): 需要在其中生成新缺失的行下标（如 [6000, 6001, ...]）。
        missing_ratio (float): 目标缺失比例（相对于 row_indices 范围内、观测点=1 的数量）。
        block_prob (float): 大约有多少比例的缺失点通过「连续块」方式生成，范围 [0,1]。
                            例如 0.7 表示约 70% 缺失点属于连续块，其余 30% 是随机点。
        max_block_len (int): 连续块的最大长度 (>=1)。实际块长会在 [1, max_block_len] 内随机采样。
        seed (int, optional): 随机种子。

    返回:
        np.ndarray: 新的 Mask，形状与 full_c_mask 相同。
    """
    # ---------- 基本检查 ----------
    if not (0.0 <= float(missing_ratio) <= 1.0):
        raise ValueError(f"missing_ratio 必须在 [0,1]，给到 {missing_ratio}")

    if not (0.0 <= float(block_prob) <= 1.0):
        raise ValueError(f"block_prob 必须在 [0,1]，给到 {block_prob}")

    if not (isinstance(max_block_len, int) and max_block_len >= 1):
        raise ValueError(f"max_block_len 必须是 >=1 的整数，给到 {max_block_len}")

    masks = full_c_mask.copy()
    H, W = masks.shape

    # 规范 row_indices
    if isinstance(row_indices, list):
        row_indices = np.array(row_indices, dtype=int)
    else:
        row_indices = np.asarray(row_indices, dtype=int)

    if row_indices.size == 0:
        raise ValueError("row_indices 为空。")

    if row_indices.min() < 0 or row_indices.max() >= H:
        raise ValueError(f"row_indices 有越界：合法范围 [0, {H-1}]")

    if seed is not None:
        np.random.seed(seed)

    # 只在 row_indices 这一子区间内操作
    sub_mask = masks[row_indices, :]          # (R, W)
    sub_obs = (sub_mask == 1)                 # True = 原始观测点，可挖

    num_total_obs = sub_obs.sum()
    if num_total_obs == 0:
        raise RuntimeError("指定行范围内没有可用的原始观测点(=1)，无法生成缺失。")

    # 总缺失点数
    num_to_miss = int(round(num_total_obs * missing_ratio))
    if num_to_miss <= 0:
        raise RuntimeError(
            f"目标缺失数为 {num_to_miss}（可能是 missing_ratio 太小或可用观测点过少）。"
        )

    # 其中一部分用「连续块」产生
    num_block_points = int(round(num_to_miss * block_prob))
    # 剩余部分用「随机点」产生
    num_random_points = num_to_miss - num_block_points

    # 在子区间里维护一个「还可挖」的布尔矩阵
    placeable = sub_obs.copy()   # True 表示当前仍可挖掉

    # 用于记录所有选中的 (sub_row_offset, col) 坐标
    chosen_rc = []

    # ---------- 辅助函数：找到 True 的连续区间 ----------
    def find_true_runs(x):
        """
        在 1D bool 数组 x 中找到所有连续 True 的区间 [s, e]（闭区间）
        返回: list of (s, e)
        """
        x = np.asarray(x, dtype=bool)
        if x.size == 0:
            return []

        # 找到 True→False 的边界 和 False→True 的边界
        # run_starts = False->True 的位置
        # run_ends   = True->False 的位置
        diff = np.diff(x.astype(int))

        run_starts = np.where(diff == 1)[0] + 1
        run_ends = np.where(diff == -1)[0]

        # 如果第一个元素就是 True，则它是一个连续段的开头
        if x[0]:
            run_starts = np.concatenate(([0], run_starts))

        # 如果最后一个元素是 True，则它是一个连续段的结尾
        if x[-1]:
            run_ends = np.concatenate((run_ends, [x.size - 1]))

        # 组装成区间
        runs = list(zip(run_starts, run_ends))
        return runs

    # ---------- 第一步：放置连续块 ----------
    def place_blocks(placeable, target_points, max_block_len):
        """
        在 2D 的 placeable(True/False) 中放置若干连续块，直到
        - 放够 target_points，或者
        - 已无足够空间
        返回：
            chosen_list: [(sub_row, col), ...]
            placeable   : 更新后的 placeable
        """
        R, C = placeable.shape
        chosen_list = []
        remain = target_points

        while remain > 0 and placeable.any():
            # 找出当前还有 True 的列
            cols = np.where(placeable.any(axis=0))[0]
            if cols.size == 0:
                break

            # 随机挑一列
            j = np.random.choice(cols)
            col_vec = placeable[:, j]

            # 该列中连续 True 段
            runs = find_true_runs(col_vec)
            # 过滤掉长度为 0 的段
            runs = [(s, e) for (s, e) in runs if (e - s + 1) > 0]
            if not runs:
                # 这一列没法放，直接把该列标记为不可用再继续
                placeable[:, j] = False
                continue

            # 当前列中能放的最长 run 长度
            max_len_col = max(e - s + 1 for (s, e) in runs)

            # 如果这一列整体最长 run 也为 0，则放弃这一列
            if max_len_col <= 0:
                placeable[:, j] = False
                continue

            # 这次尝试的块长 L：在 1 ~ min(max_block_len, max_len_col, remain) 内随机取
            L_upper = min(max_block_len, max_len_col, remain)
            if L_upper <= 0:
                break
            L = np.random.randint(1, L_upper + 1)  # [1, L_upper]

            # 再在 runs 中挑出能容纳 L 的 run
            candidate_runs = [(s, e) for (s, e) in runs if (e - s + 1) >= L]
            if not candidate_runs:
                # 如果因为 L 过大而找不到合适 run，那就直接缩到该列最大 run 长度
                L = min(max_len_col, remain)
                candidate_runs = [(s, e) for (s, e) in runs if (e - s + 1) >= L]
                if not candidate_runs:
                    placeable[:, j] = False
                    continue

            # 随机选一个 run
            s, e = candidate_runs[np.random.randint(len(candidate_runs))]
            # 在 run 内随机选择起点
            start = np.random.randint(s, e - L + 2)  # [s, e-L+1]
            end = start + L - 1

            # 标记这一段为已使用，并记录坐标
            placeable[start:end + 1, j] = False
            for r in range(start, end + 1):
                chosen_list.append((r, j))
            remain -= L

        return chosen_list, placeable

    # 连续块缺失部分
    if num_block_points > 0:
        block_chosen, placeable = place_blocks(
            placeable=placeable,
            target_points=num_block_points,
            max_block_len=max_block_len,
        )
        chosen_rc.extend(block_chosen)

    # ---------- 第二步：随机点补齐剩余 ----------
    # 剩余需要缺失的点数（如果块没放满，会增加随机部分）
    used_block_points = len(chosen_rc)
    remain_random = num_to_miss - used_block_points
    if remain_random > 0:
        # 当前仍可挖的候选点
        candidates = np.column_stack(np.where(placeable))  # (N, 2) -> (sub_row, col)
        if candidates.size == 0:
            # 放不满就只能接受「实际缺失点数 < 目标缺失点数」
            # 这里不报错，只是警告
            print(
                f"[WARN] 目标缺失点数={num_to_miss}，"
                f"实际最多只能放置 {used_block_points} 个（连续块+随机点）。"
            )
        else:
            take = min(remain_random, candidates.shape[0])
            idx = np.random.choice(np.arange(candidates.shape[0]), take, replace=False)
            extra = candidates[idx]
            for r, j in extra:
                chosen_rc.append((int(r), int(j)))

    if not chosen_rc:
        raise RuntimeError("无法放置任何缺失块/点（可能可用观测段过短/过少）。")

    # ---------- 映射回原始全局行列，并置 0 ----------
    chosen_rc = np.array(chosen_rc, dtype=int)
    sub_rows = chosen_rc[:, 0]  # 相对于 row_indices 的偏移
    cols = chosen_rc[:, 1]

    global_rows = row_indices[sub_rows]
    global_cols = cols

    masks[global_rows, global_cols] = 0
    return masks

# 这是插补任务构造gt_mask的函数
def create_full_gt_mask(full_c_mask,row_indices,missing_ratio,
    seed=None,
    mode="random",                # "random"（原逻辑）或 "block"（连续块）
    block_len=None,               # 固定块长（正整数）；与 block_len_range 二选一
    block_len_range=None,         # (Lmin, Lmax) ；含端点，均匀随机
    per_col=True                  # 缺失配额是否按各列可用观测点比例分摊
):
    """
    在 `row_indices` 指定的行范围内、针对原始观测点(=1)，生成验证/测试缺失 Mask。
    - mode="random"：随机点状缺失（与你原逻辑一致）
    - mode="block" ：生成沿时间轴（行方向）的连续缺失块

    参数:
        full_c_mask (np.ndarray): 原始观测 Mask (1=观测, 0=原始缺失)。形状 (H, W)。
        row_indices (list/np.ndarray): 需要在其中生成新缺失的行下标（如 [6000, 6001, ...]）。
        missing_ratio (float): 目标缺失比例（相对于 row_indices 范围内、观测点=1 的数量）。
        seed (int, optional): 随机种子。
        mode (str): "random" 或 "block"。
        block_len (int): 连续块固定长度；与 block_len_range 二选一。
        block_len_range (tuple): (Lmin, Lmax)；含端点，均匀抽样。
        per_col (bool): 是否按各列的“可用观测点数”按比例分摊缺失配额。

    返回:
        np.ndarray: 新的 Mask，形状与 full_c_mask 相同。
    """
    # ---------- 基本检查 ----------
    if not (0.0 <= float(missing_ratio) <= 1.0):
        raise ValueError(f"missing_ratio 必须在 [0,1]，给到 {missing_ratio}")

    masks = full_c_mask.copy()
    H, W = masks.shape

    if isinstance(row_indices, list):
        row_indices = np.array(row_indices, dtype=int)
    else:
        row_indices = np.asarray(row_indices, dtype=int)

    if row_indices.size == 0:
        raise ValueError("row_indices 为空。")

    if row_indices.min() < 0 or row_indices.max() >= H:
        raise ValueError(f"row_indices 有越界：合法范围 [0, {H-1}]")

    if seed is not None:
        np.random.seed(seed)

    # 扁平化辅助
    masks_flat = masks.reshape(-1)

    # 目标候选：row_indices × 全列
    R, C = np.meshgrid(row_indices, np.arange(W), indexing='ij')
    target_flat_indices_candidates = (R * W + C).ravel()

    # 只允许对原始观测点=1动刀
    target_obs_indices = target_flat_indices_candidates[masks_flat[target_flat_indices_candidates] == 1]
    num_total_obs = target_obs_indices.size

    if num_total_obs == 0:
        raise RuntimeError("指定行范围内没有可用的原始观测点(=1)，无法生成缺失。")

    num_to_miss = int(round(num_total_obs * missing_ratio))
    if num_to_miss <= 0:
        raise RuntimeError(
            f"目标缺失数为 {num_to_miss}（可能是 missing_ratio 太小或可用观测点过少）。"
        )

    # ---------- 模式一：随机点状（与你的旧逻辑一致） ----------
    if mode == "random":
        miss_indices = np.random.choice(target_obs_indices, num_to_miss, replace=False)
        masks_flat[miss_indices] = 0
        return masks_flat.reshape(masks.shape)

    # ---------- 模式二：连续块 ----------
    if mode != "block":
        raise ValueError(f"mode 只能是 'random' 或 'block'，给到 {mode}")

    # 块长配置检查
    if (block_len is None) == (block_len_range is None):
        raise ValueError("block_len 与 block_len_range 需二选一。")
    if block_len is not None:
        if not (isinstance(block_len, int) and block_len >= 1):
            raise ValueError("block_len 必须是 >=1 的整数")
        def sample_block_len():
            return block_len
    else:
        Lmin, Lmax = block_len_range
        if not (isinstance(Lmin, int) and isinstance(Lmax, int) and 1 <= Lmin <= Lmax):
            raise ValueError("block_len_range 需为 (Lmin, Lmax) 且 1 <= Lmin <= Lmax，整数")
        def sample_block_len():
            return np.random.randint(Lmin, Lmax + 1)

    # 把 row_indices 对应的子区段拿出来，便于列内操作
    sub_mask = masks[row_indices, :]        # 形状 (R, W)
    sub_obs  = (sub_mask == 1)              # True 表示可放置缺失

    # 各列可用观测点统计
    # col_obs_counts：对sub_obs按行求和
    col_obs_counts = sub_obs.sum(axis=0)    # (W,)
    # total_obs_in_sub：对col_obs_counts求和，相当于对sub_obs整体求和
    total_obs_in_sub = col_obs_counts.sum()
    assert total_obs_in_sub == num_total_obs

    # 为避免重复/重叠，在 sub 里维护一个“还能放”的工作副本
    placeable = sub_obs.copy()  # True=当前仍可放置缺失

    # 列配额：按比例分摊 or 统一池
    if per_col:# 让缺失点在每个列上按原来观测点的多少比例分摊。
        # 按列占比计算配额，并用“最大余数法”收尾，保证总和等于 num_to_miss
        # eg：
        #   col_obs_counts = [100, 200, 700]   # 每列观测点数
        #   num_to_miss = 100   # 想总共缺失100个点
        #   raw_alloc = (col_obs_counts / total_obs_in_sub) * num_to_miss = [10.0, 20.0, 70.0]
        #   col_quota = floor(raw_alloc) = [10, 20, 70]
        raw_alloc = (col_obs_counts / (total_obs_in_sub + 1e-12)) * num_to_miss
        col_quota = np.floor(raw_alloc).astype(int)
        remainder = num_to_miss - col_quota.sum()
        if remainder > 0:# 如果总和比目标少（比如因为小数部分被砍掉）
            # 把小数部分最大的列优先补齐
            frac = raw_alloc - col_quota # 计算每列被砍掉的小数部分：
            order = np.argsort(-frac)  # 降序
            # 假如 raw_alloc = [33.4, 33.3, 33.3]，总和=100但取整后 [33,33,33]=99，还缺1个。
            # 就把小数最大的一列（第0列）再+1 → [34,33,33]。
            for idx in order[:remainder]: # 然后按从大到小排序，用“最大余数法”补齐：
                col_quota[idx] += 1
    else:
        # 不按列分摊：后面统一在整张 sub 上放
        col_quota = np.zeros(W, dtype=int)
        col_quota[0] = num_to_miss  # 全部配额先放第一列的名义上，随后“跨列”处理

    # 统计最终要置零的 (row_offset, col) 坐标集合
    chosen_rc = []

    def pick_blocks_in_one_column(col, quota, placeable_col):
        """
        作用：在单列 (长度 = len(row_indices)) 的 placeable_col(True/False) 中放置 quota 个缺失点，以连续块形式优先；若块放不满，回退为点状随机补齐。
        返回：选中的行下标 list（相对于 row_indices 的偏移）
        """
        if quota <= 0:
            return []

        # 工作副本，避免修改外层
        plc = placeable_col.copy()
        picked = 0
        chosen_rows = []

        # 辅助：找 plc==True 的连续段 [s,e]（闭区间）

        def find_true_runs(x):# 找到所有连续的 True 区间（可用于放缺失的区域）
            # x: 1D bool
            if x.size == 0: return []
            dx = np.diff(x.astype(np.int8))
            # run starts: where x goes 0->1
            starts = np.where((np.concatenate(([x[0]], dx == 1)) & x))[0]
            # run ends: where x goes 1->0
            ends = np.where((np.concatenate(((dx == -1), [x[-1]])) & x))[0]
            # 组装
            runs = list(zip(starts, ends))
            return runs

        # 先尽量用块放
        while picked < quota:
            runs = find_true_runs(plc)
            if not runs:
                break  # 已无可放区域

            # 在有足够长度容纳“至少 1”的 run 中尝试
            # 我们每次抽一个块长，然后在能容纳该块的 runs 里随机挑一个 run，再在 run 里随机挑起点
            L = sample_block_len()
            # 能放下 L 的 run 列表
            candidate_runs = [(s, e) for (s, e) in runs if (e - s + 1) >= 1]
            if not candidate_runs:
                break

            # 若 L 太长，改短以不超过还需放置的 quota
            remain_need = quota - picked
            if L > remain_need:
                L = remain_need

            # 在 candidate_runs 中筛出能容纳 L 的
            candidate_runs = [(s, e) for (s, e) in candidate_runs if (e - s + 1) >= L]
            if not candidate_runs:
                # 这个 L 放不下，尝试把 L 调到当前能放的最大 run 长度
                max_len = 0
                best_runs = []
                for (s, e) in runs:
                    length = e - s + 1
                    if length > max_len:
                        max_len = length
                        best_runs = [(s, e)]
                    elif length == max_len and length > 0:
                        best_runs.append((s, e))
                if max_len == 0:
                    break
                # 新 L 是 min(max_len, remain_need)
                L = min(max_len, remain_need)
                candidate_runs = best_runs

            # 随机挑一个 run
            ridx = np.random.randint(0, len(candidate_runs))
            s, e = candidate_runs[ridx]
            # 在该 run 内挑起点
            start = np.random.randint(s, e - L + 2)  # [s, e-L+1]
            end = start + L - 1

            # 标记选中区域为“已使用”，并记录
            plc[start:end + 1] = False
            chosen_rows.extend(range(start, end + 1))
            picked += L

        # 如果块方式没有放满，回退为随机点填补剩余 quota
        remain = quota - picked
        if remain > 0:
            candidates = np.where(plc)[0]
            if candidates.size > 0:
                take = min(remain, candidates.size)
                extra = np.random.choice(candidates, take, replace=False)
                chosen_rows.extend(extra.tolist())
                picked += take

        return chosen_rows

        # end pick_blocks_in_one_column

    if per_col:
        # 按列放置（互不干扰）
        for j in range(W):
            quota_j = int(col_quota[j])
            if quota_j <= 0:
                continue
            rows_j = pick_blocks_in_one_column(j, quota_j, placeable[:, j])
            # 记录选择
            for r in rows_j:
                chosen_rc.append((r, j))
                placeable[r, j] = False  # 占用，避免后续重复
    else:
        # 统一池：逐列循环放，直到用完配额（更均衡些）
        remain = num_to_miss
        # 先估个“每轮目标 chunk 数”并在各列尝试一次块放，随后随机补齐
        # 这里简化为轮转式分配，每列尽量放一块，再轮转
        col_order = list(range(W))
        while remain > 0 and placeable.any():
            progressed = False
            for j in col_order:
                if remain <= 0:
                    break
                # 每次尝试至少放一个块（长度由采样或区间决定，但不超过 remain）
                quota_try = min(remain, sample_block_len())
                rows_j = pick_blocks_in_one_column(j, quota_try, placeable[:, j])
                if rows_j:
                    progressed = True
                    remain -= len(rows_j)
                    for r in rows_j:
                        chosen_rc.append((r, j))
                        placeable[r, j] = False
            if not progressed:
                break
        # 若依旧有剩余，整张 sub 随机补齐
        if remain > 0:
            candidates = np.column_stack(np.where(placeable))  # (N,2) -> (r, j)
            if candidates.size > 0:
                take = min(remain, candidates.shape[0])
                idx = np.random.choice(np.arange(candidates.shape[0]), take, replace=False)
                extra = candidates[idx]
                for r, j in extra:
                    chosen_rc.append((r, j))
                    placeable[r, j] = False

    # 将 (row_offset, col) 转成原始全局行、列，并置 0
    if not chosen_rc:
        raise RuntimeError("无法放置任何缺失块/点（可能可用观测段过短/过少）。")

    chosen_rc = np.array(chosen_rc, dtype=int)
    global_rows = row_indices[chosen_rc[:, 0]]
    global_cols = chosen_rc[:, 1]
    masks[global_rows, global_cols] = 0

    return masks


# --- 主 Dataset 类 (保持精简，使用预计算的 Mean/Std) ---
class Weather_Dataset(Dataset):
    def __init__(self, eval_length, full_data_with_nan, full_c_mask, train_mean, train_std, use_index,full_gt_mask,full_datetime_index,start_index):
        # full_data_with_nan：包含 NaN 的原始数据
        # full_c_mask：原始 observed_mask (1=观测到, 0=原始缺失)

        self.eval_length = eval_length
        self.use_index = use_index
        # self.cut_length = [0] * len(use_index)
        self.full_datetime_index = full_datetime_index

        default_cut = eval_length - 1
        cut_length_list = [default_cut] * len(use_index)

        if len(use_index) > 0:
            # 找到 use_index 中对应全局 start_index 的位置
            # np.where 找到第一个匹配的索引
            first_idx_in_set = np.where(np.array(use_index) == start_index)[0]

            if len(first_idx_in_set) > 0:
                # 将该集合中第一个窗口的 cut_length 设置为 0 (无需屏蔽)
                cut_length_list[first_idx_in_set[0]] = 0

        self.cut_length = cut_length_list  # 👈 更新 self.cut_length

        # 1. 应用归一化参数 (在所有数据上应用，但参数只来自训练集)

        # 将 NaN 替换为 0 (以便归一化公式 X - mean)
        c_data = np.nan_to_num(full_data_with_nan)


        K_total = c_data.shape[1]

        FEAT_DIM = 4
        if K_total % FEAT_DIM != 0:
            raise ValueError(
                f"K_total={K_total} 不能被 FEAT_DIM={FEAT_DIM} 整除，"
                f"当前列布局不是 [站1的{FEAT_DIM}维, 站2的{FEAT_DIM}维, ...]，请先检查 get_all_data_and_indices。"
            )

        num_stations = K_total // FEAT_DIM

        # train_mean: (FEAT_DIM,)  → 重复到每个站点上 → (K_total,)
        mean_tile = np.tile(train_mean, num_stations)   # (K_total,)
        std_tile = np.tile(train_std, num_stations)     # (K_total,)

        mean_2d = mean_tile.reshape(1, K_total)         # (1, K_total)
        std_2d = std_tile.reshape(1, K_total)           # (1, K_total)

        self.full_observed_data = ((c_data - mean_2d) / std_2d) * full_c_mask

        # 2. 存储 Mask
        self.full_observed_data = self.full_observed_data.astype(np.float32)
        self.full_observed_mask = full_c_mask.astype(np.float32)

        self.full_gt_mask = full_gt_mask.astype(np.float32)
        self.full_hist_mask = np.copy(self.full_observed_mask)  # dummy

    def __getitem__(self, org_index):
        # ... (与之前版本一致，根据 self.use_index 提取切片)
        
        index = self.use_index[org_index]  # 滑动窗口的起始位置
        current_datetime = self.full_datetime_index[index: index + self.eval_length]

        s = {
            # 归一化后的数据：模型输入的核心数据。 它包含所有特征的归一化值，其中原始缺失的位置已经被填充为 0。 (L, K)，即（L，4）
            "observed_data": self.full_observed_data[index: index + self.eval_length],
            # 原始观测 Mask：标记原始数据的质量。 1 表示该点在原始数据中是观测到的；0 表示该点在原始数据中是缺失的。(L, K)
            "observed_mask": self.full_observed_mask[index: index + self.eval_length],
            # 评估/测试目标 Mask：决定训练/评估的目标。 1 表示该点在训练/测试时已知（作为模型输入）；0 表示该点是插值目标（原始缺失+人造缺失）。 (L, K)
            "gt_mask": self.full_gt_mask[index: index + self.eval_length],
            # 历史模式 Mask：PM2.5 数据集遗留的兼容字段。 在你的简化版中，它只是 observed_mask 的一个副本，作为**虚拟（dummy）**输入，因为你的模型结构可能需要这个字段。
            "hist_mask": self.full_hist_mask[index: index + self.eval_length],
            # 时间点索引：时间编码输入。 序列中每个时间点的相对索引，通常是从 0 到 L-1。用于生成位置/时间嵌入（Time Embedding）。 (L)
            "timepoints": np.arange(self.eval_length),
            # 切割长度：测试集评估的边界。 在某些数据集（如 PM2.5）中，为了避免滑动窗口重复评估，会标记序列开头或结尾不参与评估的长度。在你的简化代码中，它总是 0。 标量
            "cut_length": self.cut_length[org_index],
            "absolute_time": current_datetime.astype(str).tolist(),
        }
        return s

    def __len__(self):
        return len(self.use_index)

    
class Forecast_Weather_Dataset(Dataset):
    def __init__(self, eval_length, full_data_with_nan, full_c_mask, train_mean, train_std, use_index,full_datetime_index,start_index,horizon):
        # full_data_with_nan：包含 NaN 的原始数据
        # full_c_mask：原始 observed_mask (1=观测到, 0=原始缺失)

        self.eval_length = eval_length
        self.use_index = use_index
        # self.cut_length = [0] * len(use_index)
        self.full_datetime_index = full_datetime_index
        self.horizon = horizon

        default_cut = eval_length - 1
        cut_length_list = [default_cut] * len(use_index)

        if len(use_index) > 0:
            # 找到 use_index 中对应全局 start_index 的位置
            # np.where 找到第一个匹配的索引
            first_idx_in_set = np.where(np.array(use_index) == start_index)[0]

            if len(first_idx_in_set) > 0:
                # 将该集合中第一个窗口的 cut_length 设置为 0 (无需屏蔽)
                cut_length_list[first_idx_in_set[0]] = 0

        self.cut_length = cut_length_list  # 👈 更新 self.cut_length

        # 1. 应用归一化参数 (在所有数据上应用，但参数只来自训练集)

        # 将 NaN 替换为 0 (以便归一化公式 X - mean)
        c_data = np.nan_to_num(full_data_with_nan)

        K_total = c_data.shape[1]

        FEAT_DIM = 4
        if K_total % FEAT_DIM != 0:
            raise ValueError(
                f"K_total={K_total} 不能被 FEAT_DIM={FEAT_DIM} 整除，"
                f"当前列布局不是 [站1的{FEAT_DIM}维, 站2的{FEAT_DIM}维, ...]，请先检查 get_all_data_and_indices。"
            )

        num_stations = K_total // FEAT_DIM

        # train_mean: (FEAT_DIM,)  → 重复到每个站点上 → (K_total,)
        mean_tile = np.tile(train_mean, num_stations)   # (K_total,)
        std_tile = np.tile(train_std, num_stations)     # (K_total,)

        mean_2d = mean_tile.reshape(1, K_total)         # (1, K_total)
        std_2d = std_tile.reshape(1, K_total)           # (1, K_total)

        self.full_observed_data = ((c_data - mean_2d) / std_2d) * full_c_mask

        # 2. 存储 Mask
        self.full_observed_data = self.full_observed_data.astype(np.float32)
        self.full_observed_mask = full_c_mask.astype(np.float32)

        # self.full_gt_mask = full_gt_mask.astype(np.float32)
        self.full_hist_mask = np.copy(self.full_observed_mask)  # dummy

    def __getitem__(self, org_index):
        # 滑动窗口的起始位置（在全局时间轴上的 index）
        index = self.use_index[org_index]
        L = self.eval_length

        # 当前窗口对应的绝对时间（DatetimeIndex -> str）
        current_datetime = self.full_datetime_index[index: index + L]

        # 原始观测 mask 窗口 (L, K_total)
        obs_mask_win = self.full_observed_mask[index: index + L]

        # 预测任务中：历史部分作为输入，后面 horizon 部分作为预测目标
        hist_len = L - self.horizon
        gt_mask_win = obs_mask_win.copy()
        gt_mask_win[hist_len:] = 0.0    # 这里统一对所有站点、所有特征做“future”为0

        s = {
            # 归一化后的数据：(L, K_total)，这里的 K_total = num_stations * FEAT_DIM
            # 每一列对应一个「站点-特征」组合，而不是以前的「单站 4 维」
            "observed_data": self.full_observed_data[index: index + L],

            # 原始观测 Mask：1=原始观测到，0=原始缺失。(L, K_total)
            "observed_mask": obs_mask_win,

            # 评估/测试目标 Mask：1=作为条件输入，0=作为插补/预测目标。(L, K_total)
            "gt_mask": gt_mask_win,

            # 历史模式 Mask：这里仍然只是 observed_mask 的一个副本，做 dummy 字段。(L, K_total)
            "hist_mask": self.full_hist_mask[index: index + L],

            # 时间点索引：0..L-1，用于做 time embedding。(L,)
            "timepoints": np.arange(L),

            # 切割长度：保持你原来的逻辑（一般是 L-1 或 0，看你前面怎么设的）
            "cut_length": self.cut_length[org_index],

            # 绝对时间字符串列表（方便你后处理或画图）
            "absolute_time": current_datetime.astype(str).tolist(),
        }
        return s

    def __len__(self):
        return len(self.use_index)


# --- Dataloader 获取函数 (更新调用流程) ---
def get_dataloader(batch_size, device, eval_length=36):
    # 1. 获取所有数据和索引划分
    full_data_with_nan, full_c_mask, full_datetime_index,train_indices, valid_indices, test_indices,train_start_idx,valid_start_idx,test_start_idx = get_all_data_and_indices(Dir_PATH, eval_length)

    train_gt_mask = full_c_mask
    valid_gt_mask = create_full_gt_mask2(full_c_mask,valid_indices, missing_ratio=0.3, seed=66)
    test_gt_mask = create_full_gt_mask(full_c_mask,test_indices, missing_ratio=0.2, seed=520)

    # 2. 仅在训练集索引上计算 Mean 和 Std (最佳实践)
    train_mean, train_std = calculate_train_mean_std(
        full_data_with_nan, full_c_mask, train_indices, eval_length
    )

    # 3. 初始化数据集 (所有数据集共享 train_mean/train_std)
    train_dataset = Weather_Dataset(
        eval_length, full_data_with_nan, full_c_mask, train_mean, train_std, train_indices, train_gt_mask,full_datetime_index,train_start_idx
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,pin_memory=True
    )

    valid_dataset = Weather_Dataset(
        eval_length, full_data_with_nan, full_c_mask, train_mean, train_std, valid_indices, valid_gt_mask,full_datetime_index,valid_start_idx
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=8, shuffle=False,pin_memory=True
    )

    test_dataset = Weather_Dataset(
        eval_length, full_data_with_nan, full_c_mask, train_mean, train_std, test_indices, test_gt_mask,full_datetime_index,test_start_idx
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8, shuffle=False,pin_memory=True
    )

    # 传递 Scalers
    scaler = torch.from_numpy(train_std).to(device).float()
    mean_scaler = torch.from_numpy(train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler,full_datetime_index


def get_forecast_dataloader(batch_size, device, eval_length=36):
    # 1. 获取所有数据和索引划分
    full_data_with_nan, full_c_mask, full_datetime_index,train_indices, valid_indices, test_indices,train_start_idx,valid_start_idx,test_start_idx = get_all_data_and_indices(Dir_PATH, eval_length)

    # train_gt_mask = full_c_mask
    # valid_gt_mask = create_full_gt_mask2(full_c_mask,valid_indices, missing_ratio=0.3, seed=66)
    # test_gt_mask = create_full_gt_mask(full_c_mask,test_indices, missing_ratio=0.2, seed=520)

    # 2. 仅在训练集索引上计算 Mean 和 Std (最佳实践)
    train_mean, train_std = calculate_train_mean_std(
        full_data_with_nan, full_c_mask, train_indices, eval_length
    )

    # 3. 初始化数据集 (所有数据集共享 train_mean/train_std)
    train_dataset = Forecast_Weather_Dataset(
        eval_length, full_data_with_nan, full_c_mask, train_mean, train_std, train_indices,full_datetime_index,train_start_idx,horizon=1
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,pin_memory=True
    )

    valid_dataset = Forecast_Weather_Dataset(
        eval_length, full_data_with_nan, full_c_mask, train_mean, train_std, valid_indices,full_datetime_index,valid_start_idx,horizon=1
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=8, shuffle=False,pin_memory=True
    )

    test_dataset = Forecast_Weather_Dataset(
        eval_length, full_data_with_nan, full_c_mask, train_mean, train_std, test_indices,full_datetime_index,test_start_idx,horizon=1
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8, shuffle=False,pin_memory=True
    )

    # 传递 Scalers
    scaler = torch.from_numpy(train_std).to(device).float()
    mean_scaler = torch.from_numpy(train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler,full_datetime_index