import os
import pandas as pd
import gc


def PRE(folder_path):
    all_data = []
    for fname in os.listdir(folder_path):
        file_path = os.path.join(folder_path, fname)
        print(f"正在处理 {file_path}")

        try:
            try:
                df = pd.read_csv(file_path, encoding="utf-8",low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="gb18030",low_memory=False)

            # 保留关键列
            cols = ["Datetime", "Station_Id_d", "City", "PRE"]
            df = df[[c for c in cols if c in df.columns]]

            # 缺测标记
            df["missing_flag"] = (df["PRE"] >= 10000).astype(int)

            # 时间标准化（保证是唯一字符串）
            # 统一分隔符和空格
            df["Datetime"] = (
                df["Datetime"]
                .astype(str)
                .str.replace("-", "/")              # 把连字符统一为斜杠
                .str.replace(r"\s+", " ", regex=True)  # 压缩多余空格
                .str.strip()
            )

            # 自动识别格式（新版 pandas 已默认启用）
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

            # 统一输出格式，保留秒
            df["Datetime"] = df["Datetime"].dt.strftime("%Y/%m/%d %H:%M:%S")

            all_data.append(df[["Station_Id_d", "City", "Datetime", "missing_flag"]])

            del df
            gc.collect()
        except Exception as e:
            print(f"❌ 出错 {fname}: {e}")

    # 合并所有天的数据
    full_df = pd.concat(all_data, ignore_index=True)

    # 构造宽表
    pivot_df = full_df.pivot_table(
        index=["Station_Id_d", "City"],
        columns="Datetime",
        values="missing_flag",
        fill_value=1 # 缺省值也用1
    )
    
    output_path = r"/workspace/six_features/pre.csv"
    # 保存
    pivot_df.to_csv(output_path, encoding="utf-8-sig")
    print(f"✅ 已保存到 {output_path}")

def choose_shanxi(file_path):
    file_path1 = r"/workspace/National_Station_InF.xlsx"

    df1 = pd.read_excel(file_path1)  # 包含 province 的表

    try:
        df2 = pd.read_csv(file_path, encoding="utf-8")  # 要筛选的表
    except UnicodeDecodeError:
        df2 = pd.read_csv(file_path, encoding="gb18030")

    # 提取 file1 中省份为山西省的 station_id 列表
    shanxi_ids = df1.loc[df1['Province'] == '山西省', 'id']

    # 在 file2 中筛选出 station_id 属于上述集合的行
    filtered = df2[df2['Station_Id_d'].isin(shanxi_ids)]

    output_path = "/workspace/six_features/only-shanxi/pre.csv"

    # 保存结果
    filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存到 {output_path}")

def choose_shanxi_from_dir(folder_path):
    file_path1 = r"/workspace/National_Station_InF.xlsx"

    df1 = pd.read_excel(file_path1)  # 包含 province 的表

    for fname in os.listdir(folder_path):
        file_path = os.path.join(folder_path, fname)
        print(f"正在处理 {file_path}")

        try:
            df2 = pd.read_csv(file_path, encoding="utf-8")  # 要筛选的表
        except UnicodeDecodeError:
            df2 = pd.read_csv(file_path, encoding="gb18030")

        # 提取 file1 中省份为山西省的 station_id 列表
        shanxi_ids = df1.loc[df1['Province'] == '山西省', 'id']

        # 在 file2 中筛选出 station_id 属于上述集合的行
        filtered = df2[df2['Station_Id_d'].isin(shanxi_ids)]

        output_dir = "/workspace/six_features/only-shanxi/all_data"
        output_path = os.path.join(output_dir, fname)

        # 保存结果
        filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 已保存到 {output_path}")

def summarize_nan(file_path):
    # 读取文件
    try:
        df = pd.read_csv(file_path, encoding="utf-8")  # 包含 province 的表
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gb18030")

    # 提取时间列（第3列之后的所有列）
    time_cols = df.columns[2:]

    # 将列名（例如 "2024/10/1 0:00"）转成 pandas 的时间对象
    dates = pd.to_datetime(time_cols, format="%Y/%m/%d %H:%M:%S")

    # 把列名映射为日期（只保留年月日）
    date_only = [d.date() for d in dates]

    # 建立新的 DataFrame，只保留数值部分
    values = df[time_cols].copy()
    values.columns = date_only  # 列名替换成日期

    # 按日期分组求每天的 1 数量
    daily_counts = values.groupby(values.columns, axis=1).apply(lambda x: (x == 1).sum(axis=1))

    # 拼回原始的站点信息
    result = pd.concat([df[["Station_Id_d", "City"]], daily_counts], axis=1)

    output_path = "/workspace/six_features/only-shanxi/缺损率/pre.csv"
    # 保存结果
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存到 {output_path}")


def shanxi_WIN_S_Avg_2mi_per_day(folder_path):
    all_data = []

    for fname in os.listdir(folder_path):
        file_path = os.path.join(folder_path, fname)
        print(f"正在处理 {file_path}")

        try:
            try:
                df = pd.read_csv(file_path, encoding="utf-8",low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="gb18030",low_memory=False)

            # 保留关键列
            cols = ["Datetime", "Station_Id_d", "City", "WIN_S_Avg_2mi"]
            df = df[[c for c in cols if c in df.columns]]

            # 将 PRS 转为数值（防止有字符串、空值）
            df["WIN_S_Avg_2mi"] = pd.to_numeric(df["WIN_S_Avg_2mi"], errors="coerce")
            df.loc[df["WIN_S_Avg_2mi"] >= 10000, "WIN_S_Avg_2mi"] = float("nan")

            df["Datetime"] = df["Datetime"].astype(str).str.strip()

            all_data.append(df[["Station_Id_d", "City", "Datetime", "WIN_S_Avg_2mi"]])

            del df
            gc.collect()
        except Exception as e:
            print(f"❌ 出错 {fname}: {e}")

    # 合并所有天的数据
    full_df = pd.concat(all_data, ignore_index=True)

    # --- 构造宽表（每个时间列是一个 WIN_S_Avg_2mi 值） ---
    pivot_df = full_df.pivot_table(
        index=["Station_Id_d", "City"],
        columns="Datetime",
        values="WIN_S_Avg_2mi",
        aggfunc="first",   # 如果同一站同一时间有多行，取第一条
        fill_value=None    # 缺失保留 NaN
    )
    output_path = "/workspace/six_features/only-shanxi/all_six/WIN_S_Avg_2mi.csv"
    # 保存
    pivot_df.to_csv(output_path, encoding="utf-8-sig")
    print(f"✅ 已保存到 {output_path}")

#------------------------------------------------------以下是全山西省一年数据处理
def PRE_per_day(file_path):
    # 读取 CSV（自动识别编码）
    try:
        df = pd.read_csv(file_path, encoding="utf-8",low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gb18030",low_memory=False)

    # 保留关键列
    df = df[["Datetime", "Station_Id_d", "WIN_S_Avg_2mi"]]

    # 3) 时间 & 数值规范
    # 自动识别格式（新版 pandas 已默认启用）
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # 统一输出格式，保留秒
    df["Datetime"] = df["Datetime"].dt.strftime("%Y/%m/%d %H:%M:%S")
    df["TWIN_S_Avg_2miEM"] = pd.to_numeric(df["WIN_S_Avg_2mi"], errors="coerce")

    # 把异常大值当缺测
    df.loc[df["WIN_S_Avg_2mi"] >= 10000, "WIN_S_Avg_2mi"] = pd.NA

    # 透视表操作
    pivot_df = df.pivot_table(
        index="Datetime",       # 行索引
        columns="Station_Id_d",   # 列名
        values="WIN_S_Avg_2mi",           # 填充值
        aggfunc="first"         # 如果重复时间点，取第一个
    )

    # 保存
    pivot_df.to_csv("/workspace/six_features/only-shanxi/all_six/WIN_S_Avg_2mi.csv",encoding="utf-8-sig")
    print(f"✅ 已保存")

def PRE_miss():
    infile = "/workspace/six_features/only-shanxi/all_six/WIN_S_Avg_2mi.csv"
    outfile = "/workspace/six_features/only-shanxi/缺损/缺损率/WIN_S_Avg_2mi.csv"

    try:
        df = pd.read_csv(infile, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(infile, encoding="gb18030", low_memory=False)

    # ---------- 2. 处理时间列 ----------
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.set_index("Datetime")

    # ---------- 3. 每天每列 NaN 计数 ----------
    daily_na = df.isna().groupby(df.index.normalize()).sum()

    # ---------- 4. 如果某天该列的 NA 数为 0，则改为 NaN ----------
    daily_na = daily_na.mask(daily_na == 0)

    # ---------- 5. 格式化日期并保存 ----------
    daily_na.index = daily_na.index.strftime("%Y/%m/%d")
    daily_na.to_csv(outfile, encoding="utf-8-sig")

    print("Done")

if __name__ == "__main__":
    #PRE(r"/workspace/meteorology_data_202410")
    # choose_shanxi(r"/workspace/six_features/pre.csv")
    #summarize_nan(r"/workspace/six_features/only-shanxi/pre.csv")
    #choose_shanxi_from_dir(r"/workspace/meteorology_data_202410")
    #shanxi_WIN_S_Avg_2mi_per_day(r"/workspace/six_features/only-shanxi/all_data")
    #PRE_per_day(r"/workspace/山西国家站观测数据.csv")
    PRE_miss()
