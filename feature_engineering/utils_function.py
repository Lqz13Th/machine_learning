import pandas as pd
import polars as pl

def pl_dropna(df: pl.DataFrame) -> pl.DataFrame:
    pdf = df.to_pandas()
    pdf_clean = pdf.dropna()
    return pl.from_pandas(pdf_clean)

def filter_pl_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    cols_to_keep = []
    for col in df.columns:
        series = df[col]
        # 检查是否有 null
        if series.null_count() > 0:
            continue
        # 检查是否所有值都相同
        if series.n_unique() <= 1:
            continue
        # 符合条件，保留该列
        cols_to_keep.append(col)
    # 返回只包含筛选列的 DataFrame
    return df.select(cols_to_keep)


import polars as pl
import numpy as np

def kl_div_udf(series_a: pl.Series, series_b: pl.Series, bins: int = 20) -> float:
    """
    计算两个 Polars Series 的 KL 散度。
    这个 UDF 接收两个 Series，它们代表了窗口中的两个子段。
    """
    a = series_a.to_numpy()
    b = series_b.to_numpy()

    # 确保没有 NaN 或 inf，否则 np.histogram 会报错
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan # 或者你希望的默认值

    # 合并数据以确定统一的 bin 边界，避免不同子段的 bin 边界不一致
    min_val = min(a.min(), b.min())
    max_val = max(a.max(), b.max())
    if min_val == max_val: # 避免分箱范围为0
        return np.nan # 或者你希望的默认值

    # 计算直方图
    pa, _ = np.histogram(a, bins=bins, range=(min_val, max_val), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(min_val, max_val), density=True)

    # 避免 log(0)
    pa[pa == 0] = 1e-10
    pb[pb == 0] = 1e-10

    # 计算 KL 散度
    return np.sum(pa * np.log(pa / pb))

def kl_div_rolling_expr(col_name: str, window1: int, window2: int, bins: int = 20) -> pl.Expr:
    """
    创建 Polars 表达式，用于滚动计算 KL 散度。
    使用 `rolling` 配合 `map_batches`。
    """
    total_window_size = window1 + window2

    # 创建一个滚动窗口表达式
    return pl.col(col_name).rolling(window_size=f"{total_window_size}i").map_batches(
        lambda s: [ # map_batches 期望返回一个 Series 或列表
            kl_div_udf(s[i:i+window1], s[i+window1:i+total_window_size], bins=bins)
            for i in range(len(s) - total_window_size + 1)
        ]
    ).alias(f"{col_name}_kl_{window1}_{window2}")


# 示例使用
df = pl.DataFrame({
    "value": np.random.rand(1000) * 100
})

# 假设计算 20 个数据点的第一个窗口和 30 个数据点的第二个窗口的 KL 散度
# 总窗口大小为 50
df_with_kl = df.with_columns(
    kl_div_rolling_expr("value", window1=20, window2=30)
)

print(df_with_kl.tail(10))

# 另一种更直接的 map_batches 写法，适用于整个窗口传递给 UDF
# 如果你的 KL 散度 UDF 需要一个完整的窗口，然后内部再切分
def kl_div_full_window_udf(window_series: pl.Series, w1: int, w2: int) -> float:
    """
    接收整个窗口的 Series，然后在 UDF 内部切分并计算 KL 散度。
    """
    if len(window_series) != w1 + w2:
        return np.nan # 窗口大小不符合预期

    arr = window_series.to_numpy()
    a = arr[:w1]
    b = arr[w1:]

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    min_val = min(a.min(), b.min())
    max_val = max(a.max(), b.max())
    if min_val == max_val:
        return np.nan

    pa, _ = np.histogram(a, bins=20, range=(min_val, max_val), density=True)
    pb, _ = np.histogram(b, bins=20, range=(min_val, max_val), density=True)

    pa[pa == 0] = 1e-10
    pb[pb == 0] = 1e-10

    return np.sum(pa * np.log(pa / pb))

# 使用这种 UDF 的表达式
df_with_kl_full_window = df.with_columns(
    pl.col("value").rolling(
        period=f"{20+30}i", # 总窗口大小
        # offset="-49i" # 如果你想让结果对应窗口的起点，可以设置 offset
    ).map_batches(
        lambda s: s.map_elements(lambda x: kl_div_full_window_udf(x, w1=20, w2=30), return_dtype=pl.Float64)
    ).alias("value_kl_full_window")
)

print("\n--- Full Window UDF Example ---")
print(df_with_kl_full_window.tail(10))

def lyapunov_expr(col: str, window: int) -> pl.Expr:
    def lyap(x: list[float]) -> float:
        x = np.array(x)
        if len(x) < 2: return np.nan
        return np.mean(np.log(np.abs(np.diff(x)) + 1e-8))
    return pl.col(col).map_elements(lyap, window_size=window).alias(f"{col}_lyap_{window}")
