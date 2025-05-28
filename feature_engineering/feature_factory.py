import polars as pl
import numpy as np
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
        except:
            residual = 0.0
        out[i] = residual

    return pl.Series(f"{s1.name}_{s2.name}_coint_resid_{window}", out)

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



print(feat_df.tail(10))