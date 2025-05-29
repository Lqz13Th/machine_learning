import polars as pl
import numpy as np
from typing import List, Union

from numpy.linalg import LinAlgError
from sklearn.decomposition import PCA
from statsmodels.api import OLS, add_constant

# 衡量市场不确定性、混乱程度，非常适合用于 regime detection：
def entropy_series(s: pl.Series, window: int) -> pl.Series:
    values = s.to_numpy()
    out = [None] * len(values)
    for i in range(window - 1, len(values)):
        window_vals = values[i - window + 1:i + 1]
        hist, _ = np.histogram(window_vals, bins='auto', density=True)
        hist = hist[hist > 0]
        out[i] = -np.sum(hist * np.log(hist))
    return pl.Series(f"{s.name}_entropy_{window}", out)


# 衡量一个时间段分布与另一段的分布差异，用于捕捉 regime change：
def kl_div_series(s: pl.Series, window1: int, window2: int) -> pl.Series:
    values = s.to_numpy()
    total_window = window1 + window2
    out = [None] * len(values)
    for i in range(total_window - 1, len(values)):
        a = values[i - total_window + 1:i - window2 + 1]
        b = values[i - window2 + 1:i + 1]
        pa, _ = np.histogram(a, bins=20, density=True)
        pb, _ = np.histogram(b, bins=20, density=True)
        pa += 1e-10
        pb += 1e-10
        out[i] = np.sum(pa * np.log(pa / pb))
    return pl.Series(f"{s.name}_kl_{window1}_{window2}", out)


# 判断序列是收敛、发散还是混沌系统：
def lyapunov_series(s: pl.Series, window: int) -> pl.Series:
    values = s.to_numpy()
    out = [None] * len(values)
    for i in range(window - 1, len(values)):
        x = values[i - window + 1:i + 1]
        lyap = np.mean(np.log(np.abs(np.diff(x)) + 1e-8))
        out[i] = lyap
    return pl.Series(name=f"{s.name}_lyap_{window}", values=out)


# 模拟带有先验信念的滤波器，适用于不确定市场：
def bayesian_ema_series(s: pl.Series, window: int = 20, alpha_prior: float = 0.1) -> pl.Series:
    values = s.to_numpy()
    out = [None] * (window - 1)
    for i in range(window - 1, len(values)):
        x = values[i - window + 1:i + 1]
        est = x[0]
        for val in x[1:]:
            est = alpha_prior * val + (1 - alpha_prior) * est
        out.append(est)
    return pl.Series(name=f"{s.name}_bayes_ema", values=out)


# 可在训练前提取全市场主因子变化：
def pca_first_component_series(df: pl.DataFrame, cols: list[str], window: int) -> pl.Series:
    X = df.select(cols).to_numpy()
    out = [None] * (window - 1)
    for i in range(window - 1, len(X)):
        block = X[i - window + 1:i + 1]
        pca = PCA(n_components=1)
        val = pca.fit_transform(block)[-1, 0]
        out.append(val)
    return pl.Series(name=f"pca1_{window}", values=out)

# 识别资产是否协整，输出协整残差作为 feature：
# 用 OLS 回归计算两序列在窗口内的残差，输出当前窗口最后一点的残差。
# 这是衡量两个资产协整关系的经典做法，残差小表示协整。
# 用 try-except 保护避免回归失败，默认残差0.0。
def cointegration_residual_series(s1: pl.Series, s2: pl.Series, window: int) -> pl.Series:
    x1 = s1.to_numpy()
    x2 = s2.to_numpy()
    out = [None] * len(x1)

    for i in range(window - 1, len(x1)):
        y = x1[i - window + 1:i + 1]
        x = x2[i - window + 1:i + 1]
        try:
            model = OLS(y, add_constant(x)).fit()
            residual = y[-1] - model.predict([1, x[-1]])[0]

        except (LinAlgError, ValueError):
            residual = 0.0

        out[i] = residual

    return pl.Series(f"{s1.name}_{s2.name}_coint_resid_{window}", out)

def fft_power_topk_series(series: pl.Series, window: int = 64, k: int = 3) -> pl.Series:
    values = series.to_list()
    out = []

    for i in range(len(values)):
        if i < window - 1:
            out.append(None)
        else:
            window_data = values[i - window + 1 : i + 1]
            fft = np.fft.fft(window_data)
            powers = np.abs(fft[: window // 2])
            out.append(np.sum(np.sort(powers)[-k:]))

    return pl.Series(f"{series.name}_fft_power_top{k}", out)


def frac_diff_series(series: pl.Series, d: float, threshold: float = 1e-5, max_len: int = 20) -> pl.Series:
    weights = [1.0]
    for k in range(1, max_len):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    weights = np.array(weights[::-1])

    values = series.to_numpy()
    out = np.full_like(values, np.nan, dtype=np.float64)

    for i in range(len(weights) - 1, len(values)):
        window = values[i - len(weights) + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = np.dot(weights, window)

    return pl.Series(f"{series.name}_fracdiff_d{d:.2f}", out)


# def batch_apply_series()

def cross_corr_expr(col1: str, col2: str, window: int) -> pl.Expr:
    return (
        (pl.col(col1).rolling_mean(window) * pl.col(col2).rolling_mean(window))
        .alias(f"{col1}_{col2}_corr_{window}")
    )

def soft_trend_expr(col: str, window: int) -> pl.Expr:
    diff = pl.col(col) - pl.col(col).shift(1)
    smoothed = diff.rolling_mean(window)
    return smoothed.map_elements(
        lambda x: 2 / (1 + np.exp(-10 * x)) - 1,
        return_dtype=pl.Float64,
        returns_scalar=True
    ).alias(f"{col}_soft_trend_{window}")

def rolling_skew_expr(col: str, window: int) -> pl.Expr:
    mean = pl.col(col).rolling_mean(window)
    std = pl.col(col).rolling_std(window) + 1e-8
    m3 = ((pl.col(col) - mean) ** 3).rolling_mean(window)
    return (m3 / (std ** 3)).alias(f"{col}_skew")

def rolling_volatility_expr(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_std(window).alias(f"{col}_volatility_{window}")

def rolling_kurt_expr(col: str, window: int) -> pl.Expr:
    mean = pl.col(col).rolling_mean(window)
    std = pl.col(col).rolling_std(window) + 1e-8
    m4 = ((pl.col(col) - mean) ** 4).rolling_mean(window)
    return (m4 / (std ** 4)).alias(f"{col}_kurt")

def diff_expr(col: str, lag: int = 1) -> pl.Expr:
    return (pl.col(col) - pl.col(col).shift(lag)).alias(f"{col}_diff_{lag}")

def second_order_diff_expr(col: str, lag: int = 1) -> pl.Expr:
    # 二阶差分 = 一阶差分的差分
    first_diff = pl.col(col) - pl.col(col).shift(lag)
    second_diff = first_diff - first_diff.shift(lag)
    return second_diff.alias(f"{col}_second_order_diff_{lag}")

def momentum_expr(col: str, lag: int = 200) -> pl.Expr:
    # 动量 = x_t - x_{t-lag}
    return (pl.col(col) - pl.col(col).shift(lag)).alias(f"{col}_momentum_{lag}")

def momentum_ratio_expr(col: str, lag: int = 200) -> pl.Expr:
    # 动量比率 = x_t / x_{t-lag}
    return (pl.col(col) / (pl.col(col).shift(lag) + 1e-8)).alias(f"{col}_momentum_ratio_{lag}")

def lag_expr(col: str, lag: int = 200) -> pl.Expr:
    return pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")

def inverse_expr(col: str) -> pl.Expr:
    return (1 / (pl.col(col) + 1e-8)).alias(f"{col}_inverse")

def reciprocal_sqrt_expr(col: str) -> pl.Expr:
    return (1 / (pl.col(col).sqrt() + 1e-8)).alias(f"{col}_rsqrt")

def abs_expr(col: str) -> pl.Expr:
    return pl.col(col).abs().alias(f"{col}_abs")

def sigmoid_expr(col: str) -> pl.Expr:
    return (1 / (1 + (-pl.col(col)).exp())).alias(f"{col}_sigmoid")

def tanh_expr(col: str) -> pl.Expr:
    return pl.col(col).tanh().alias(f"{col}_tanh")

def boxcox_expr(col: str, lam: float = 0.0) -> pl.Expr:
    # Box-Cox: 当 lam=0时，等同于 log(x + 1)
    # 这里为了安全，先加1避免0或负数
    if lam == 0:
        return (pl.col(col) + 1).log().alias(f"{col}_boxcox_{lam}")
    else:
        return (((pl.col(col) + 1) ** lam - 1) / lam).alias(f"{col}_boxcox_{lam}")

def squared_expr(col: str) -> pl.Expr:
    return (pl.col(col) ** 2).alias(f"{col}_squared")

def sqrt_expr(col: str) -> pl.Expr:
    return pl.col(col).sqrt().alias(f"{col}_sqrt")

def log1p_expr(col: str) -> pl.Expr:
    return pl.col(col).log1p().alias(f"{col}_log1p")

def conditioned_cross_expr(a: str, b: str) -> pl.Expr:
    mean_a = pl.col(a).mean()
    return (
        pl.when(pl.col(a) > mean_a)
        .then(pl.col(a) * pl.col(b))
        .otherwise(0.0)
        .alias(f"{a}_X_{b}_cond_mean")
    )

def rolling_skew_shift_expr(col: str, window: int) -> pl.Expr:
    return (
        (pl.col(col).rolling_skew(window) - pl.col(col).rolling_skew(window).shift(1))
        .alias(f"{col}_skew_shift_{window}")
    )

def batch_apply_transforms(df: pl.DataFrame, window: int, lag: int, exclude_cols: List[str] = None) -> pl.DataFrame:
    if exclude_cols is None:
        exclude_cols = ['price', 'timestamp', 'timestamp_dt', 'symbol']

    if isinstance(df, pl.LazyFrame):
        cols = df.collect_schema().names()
    else:
        cols = df.columns

    base_cols = [
        col for col in cols
        if col not in exclude_cols and not (
            col.endswith('_rolling_mean') or
            col.endswith('_rolling_std') or
            col.endswith('_scaled')
        )
    ]

    exprs = []
    series = []
    # single features transformation
    for col in base_cols:
        exprs.extend([
            rolling_skew_expr(col, window),
            rolling_kurt_expr(col, window),
            diff_expr(col, lag),
            second_order_diff_expr(col, lag),
            momentum_expr(col, lag),
            momentum_ratio_expr(col, lag),
            lag_expr(col, lag),
            inverse_expr(col),
            reciprocal_sqrt_expr(col),
            abs_expr(col),
            sigmoid_expr(col),
            tanh_expr(col),
            boxcox_expr(col, lam=0),
            rolling_skew_shift_expr(col, window),
            squared_expr(col),
            sqrt_expr(col),
            log1p_expr(col),
        ])

    # double features transformation
    n = len(base_cols)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = base_cols[i], base_cols[j]
            exprs.extend([
                (pl.col(a) * pl.col(b)).alias(f"{a}_X_{b}"),
                (pl.col(a) / (pl.col(b) + 1e-8)).alias(f"{a}_DIV_{b}"),
                ((pl.col(a) / (pl.col(b) + 1e-8)).log()).alias(f"{a}_LOGR_{b}"),
                conditioned_cross_expr(a, b)
            ])

    return df.with_columns(exprs)

if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    t = np.arange(n)
    btc = np.cumsum(np.random.randn(n)) + 20000  # BTC price
    eth = btc * 0.05 + np.random.randn(n) * 10 + 1000  # ETH price, cointegrated with noise

    df = pl.DataFrame({
        "timestamp": t,
        "btc_close": btc,
        "eth_close": eth,
    })

    # ---- 应用所有指标 ----

    btc_s = df["btc_close"]
    eth_s = df["eth_close"]
    print(df)
    feat_df = df.with_columns([
        cointegration_residual_series(btc_s, eth_s, window=50),
        cross_corr_expr("btc_close", "eth_close", window=30),
        soft_trend_expr("btc_close", window=20),
    ])
    print(feat_df.tail(10))

    # 逐列手动添加 entropy、kl、lyapunov、bayesian_ema 结果
    feat_df = feat_df.with_columns([
        entropy_series(btc_s, window=30),
        kl_div_series(btc_s, window1=20, window2=10),
        lyapunov_series(btc_s, window=25),
        bayesian_ema_series(btc_s, window=20),
        pca_first_component_series(df, cols=["btc_close", "eth_close"], window=50),
    ])

    feat_df.hstack()


    print(feat_df.tail(10))

