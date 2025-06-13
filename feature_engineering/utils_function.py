from typing import List

import pandas as pd
import polars as pl
import numpy as np
import math
import gc

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from tqdm import tqdm

from feature_engineering.feature_factory import *

def split_df_by_week(
        origin_input_df: pl.DataFrame,
        ts_col: str = "timestamp"
) -> List[pl.DataFrame]:
    origin_input_df = origin_input_df.with_columns([
        pl.col(ts_col).cast(pl.Datetime).alias(f"{ts_col}_dt")
    ])

    origin_input_df = origin_input_df.with_columns([
        pl.col(f"{ts_col}_dt").dt.truncate("1w").alias("week_start")
    ])

    unique_weeks = origin_input_df.select("week_start").unique().sort("week_start")

    weekly_dfs = [
        origin_input_df.filter(pl.col("week_start") == wk).drop("week_start")
        for wk in unique_weeks["week_start"]
    ]

    return weekly_dfs


def split_df_by_month(
        df: pl.DataFrame,
        ts_col: str = "timestamp"
) -> List[pl.DataFrame]:
    df = df.with_columns([
        pl.col(ts_col).cast(pl.Datetime).alias(f"{ts_col}_dt")
    ])

    df = df.with_columns([
        pl.col(f"{ts_col}_dt").dt.truncate("1mo").alias("month_start")
    ])

    unique_months = df.select("month_start").unique().sort("month_start")

    monthly_dfs = [
        df.filter(pl.col("month_start") == mo).drop("month_start")
        for mo in unique_months["month_start"]
    ]

    return monthly_dfs


def clean_df_drop_nulls(
        df_to_clean: pl.DataFrame,
        null_threshold: int = 5000,
        verbose: bool = True
) -> pl.DataFrame:
    pd_df = df_to_clean.to_pandas()

    null_counts = pd_df.isnull().sum()
    cols_to_drop = null_counts[null_counts > null_threshold].index

    pd_df_cleaned = pd_df.drop(columns=cols_to_drop)
    pd_df_clean = pd_df_cleaned.dropna()
    pl_df_clean = pl.from_pandas(pd_df_clean)

    if verbose:
        max_null_col = null_counts.idxmax()
        max_null_count = null_counts.max()
        print("各列空值数量：")
        print(null_counts[null_counts > 0])
        print(f"删除空值超过 {null_threshold} 的列：{list(cols_to_drop)}")
        print(f"删除列后，DataFrame形状：{pd_df_cleaned.shape}")
        print(f"空值最多的列是：{max_null_col}，共有 {max_null_count} 个空值")
        print(f"删除空值行后，DataFrame形状：{pd_df_clean.shape}")

    return pl_df_clean

def avg_steps_to_volatility(prices: np.ndarray, target_ratio: float) -> int:
    n = len(prices)
    steps_list = []
    for i in tqdm(range(n), desc=f"cal abs change {target_ratio*100:.2f}% avg steps"):
        start_price = prices[i]
        steps = -1
        for j in range(i + 1, n):
            change = abs(prices[j] / start_price - 1)
            if change >= target_ratio:
                steps = j - i
                break
        if steps != -1:
            steps_list.append(steps)
    if len(steps_list) == 0:
        return -1
    return int(np.mean(steps_list))

def future_return_expr(price_col: str, step: int) -> pl.Expr:
    return ((pl.col(price_col).shift(-step) - pl.col(price_col)) / pl.col(price_col)).alias(f"future_return_{step}")

def fast_spearman_ic(df: pl.DataFrame, target_col: str, exclude_prefixes: list[str]) -> dict:
    exclude_prefixes += [col for col in df.columns if col.startswith("future_return_")]

    feature_cols = [col for col in df.columns if col not in exclude_prefixes]

    ic_dict = {}

    rank_cols = feature_cols + [target_col]
    df_ranked = df.with_columns([
        pl.col(col).rank(method="average").alias(col + "_rank") for col in rank_cols
    ])

    target_rank = target_col + "_rank"

    for feat in tqdm(feature_cols, desc="Calculating IC"):
        feat_rank = feat + "_rank"
        corr = df_ranked.select(
            pl.corr(pl.col(feat_rank), pl.col(target_rank)).alias("corr")
        ).to_series()[0]
        ic_dict[feat] = corr

    return ic_dict


def calc_hourly_rankic(
        df_to_cal_rankic: pl.DataFrame,
        timestamp_col: str,
        target_col: str,
        exclude_cols: list[str] = None,
        factor_prefix_exclude: str = "future_return_"
) -> pl.DataFrame:
    if exclude_cols is None:
        exclude_cols = []

    factor_cols = [
        col for col in df_to_cal_rankic.columns
        if col not in exclude_cols and not col.startswith(factor_prefix_exclude)
    ]

    agg_exprs = []
    for factor in factor_cols:
        agg_exprs.append(
            pl.corr(pl.col(factor).rank(method="average"), pl.col(target_col).rank(method="average"),
                    method="spearman").alias(factor)
        )

    ic_df = (
        df_to_cal_rankic
        .with_columns([
            pl.col(timestamp_col).cast(pl.Int64).cast(pl.Datetime("us")),
        ])
        .with_columns([
            pl.col(timestamp_col).dt.truncate("1h").alias("hour_group"),
        ])
        .group_by("hour_group")
        .agg(agg_exprs)
        .sort("hour_group")
    )

    return ic_df

def summarize_ic_df_wide(ic_df: pl.DataFrame, exclude_prefixes: list[str] = None) -> pl.DataFrame:
    if exclude_prefixes is None:
        exclude_prefixes = []

    factor_cols = [
        col for col in ic_df.columns
        if col.endswith("_scaled")
           and all(not col.startswith(prefix) for prefix in exclude_prefixes)
           and not col.startswith("future_return_")
           and col != "price"
    ]

    # 使用 pl.DataFrame.select 批量计算 mean 和 std
    means = ic_df.select([pl.col(col).mean().alias(col) for col in factor_cols])
    stds = ic_df.select([pl.col(col).std().alias(col) for col in factor_cols])

    # 构造结果
    data = []
    for col in factor_cols:
        mean_ic = means[0, col]
        std_ic = stds[0, col]
        ir = mean_ic / std_ic if std_ic and std_ic != 0 else None
        data.append({"factor": col, "mean_ic": mean_ic, "std_ic": std_ic, "ir": ir})

    return pl.DataFrame(data)

def get_top_bottom_ic_ir(
    ic_summary: pl.DataFrame,
    top_n: int = 5
):
    pdf = ic_summary.to_pandas()
    pdf = pdf.dropna(subset=["mean_ic", "ir"])

    def format_rows(rows, metric_name):
        if not rows:
            return ["(No valid factors)"]

        max_len = max(len(row['factor']) for row in rows)
        return [
            f"{i+1:>2}. {row['factor']:<{max_len}} {metric_name}: {row[metric_name]:.6f}"
            for i, row in enumerate(rows)
        ]

    ic_top = pdf.sort_values("mean_ic", ascending=False).head(top_n).to_dict(orient="records")
    ic_bottom = pdf.sort_values("mean_ic", ascending=True).head(top_n).to_dict(orient="records")
    ir_top = pdf.sort_values("ir", ascending=False).head(top_n).to_dict(orient="records")

    # print("📈 Top IC Factors:")
    # print("\n".join(format_rows(ic_top, "mean_ic")), end="\n\n")
    #
    # print("📉 Bottom IC Factors:")
    # print("\n".join(format_rows(ic_bottom, "mean_ic")), end="\n\n")
    #
    # print("📈 Top IR Factors:")
    # print("\n".join(format_rows(ir_top, "ir")), end="\n\n")

    # 同时返回因子名列表（如需进一步操作）
    ic_top_names = [row["factor"] for row in ic_top]
    ic_bot_names = [row["factor"] for row in ic_bottom]

    ir_top_names = [row["factor"] for row in ir_top]
    return ic_top_names, ic_bot_names, ir_top_names


def calc_monotonicity(bin_returns) -> float:
    bins = list(range(len(bin_returns)))
    rho, _ = spearmanr(bins, bin_returns)
    return rho


def calc_binned_return_and_stability(
        df: pl.DataFrame,
        future_return_col: str,
        exclude_prefixes: list[str] = None,
        n_bins: int = 5,
):
    if exclude_prefixes is None:
        exclude_prefixes = []

    factors = [
        col for col in df.columns
        if col.endswith("_scaled")
           and all(not col.startswith(prefix) for prefix in exclude_prefixes)
           and not col.startswith("future_return_")
           and col != "price"
    ]

    pdf = df.select([*factors, future_return_col]).to_pandas().dropna()

    results = []

    for factor in factors:
        try:
            pdf['factor_bin'] = pd.qcut(pdf[factor], q=n_bins, duplicates='drop')
            grouped = pdf.groupby('factor_bin', observed=False)[future_return_col].agg(['mean', 'std']).reset_index()

            if len(grouped) < 40:
                continue

            bin_means = grouped['mean'].tolist()
            spread = bin_means[-1] - bin_means[0]
            mean_std = grouped['std'].mean()
            mean_mean = grouped['mean'].mean()
            stability = mean_std / abs(mean_mean) if mean_mean != 0 else np.nan
            monotonicity = calc_monotonicity(bin_means)

            results.append({
                'factor': factor,
                'spread': spread,
                'stability': stability,
                'monotonicity': monotonicity,
                'mean_return_top_bin': bin_means[-1],
                'mean_return_bottom_bin': bin_means[0],
                'bin_returns': bin_means,
            })

        except Exception as e:
            print(f"Skipped factor {factor} due to error: {e}")
            continue

    results_sorted_by_spread = sorted(results, key=lambda x: x['spread'], reverse=True)
    results_sorted_by_stability = sorted(results, key=lambda x: x['stability'] if not np.isnan(x['stability']) else 1e9)
    results_sorted_by_monotonicity = sorted(results, key=lambda x: abs(x['monotonicity']), reverse=True)
    return {
        'by_spread': results_sorted_by_spread,
        'by_stability': results_sorted_by_stability,
        'by_monotonicity': results_sorted_by_monotonicity,
        'raw': results
    }

def filter_by_spearman_corr(df: pl.DataFrame, factor_cols: list[str], threshold: float = 0.9) -> list[str]:
    matrix = np.column_stack([df[col].to_numpy() for col in factor_cols])

    corr_matrix, _ = spearmanr(matrix)  # 一次性计算所有因子的 Spearman 相关系数矩阵，shape = (n, n)
    n = len(factor_cols)
    keep = []
    removed = set()

    print("Spearman correlation matrix:")
    print(np.round(corr_matrix, 3))

    for i in range(n):
        if factor_cols[i] in removed:
            # print(f"Skip {factor_cols[i]} as it is removed")
            continue
        keep.append(factor_cols[i])
        # print(f"Keep {factor_cols[i]}")

        for j in range(i + 1, n):
            if factor_cols[j] in removed:
                continue

            corr_value = corr_matrix[i, j]

            if abs(corr_value) >= threshold:
                removed.add(factor_cols[j])
                # print(f"Remove {factor_cols[j]} because corr({factor_cols[i]}, {factor_cols[j]}) = {corr_value:.3f} >= {threshold}")

    return keep

def filter_by_mutual_info(df: pl.DataFrame, factor_cols: list[str], target_col: str, top_k: int = 50) -> list[str]:
    X = np.column_stack([df[col].to_numpy() for col in factor_cols])
    y = df[target_col].to_numpy()

    mi_scores = mutual_info_regression(X, y, discrete_features=False)
    sorted_idx = np.argsort(mi_scores)[::-1]
    selected = [factor_cols[i] for i in sorted_idx[:top_k]]

    return selected


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def single_asset_rolling_quantile_backtest(
        df,
        signal_col='signal',
        price_col='price',
        timestamp_col='timestamp',
        long_quantile=0.9,
        short_quantile=0.1,
        window=500,
        fee=0.001,
        signal_mode='normal',
        plot=True,
):
    df = df.select([signal_col, price_col, timestamp_col]).to_pandas()
    if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='us')
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    if signal_mode == 'reverse':
        df[signal_col] = df[signal_col] * -1

    # 收益
    df['raw_ret'] = df[price_col].shift(-1) / df[price_col] - 1
    df = df.dropna(subset=['raw_ret'])

    # 滚动分位数（使用过去的数据）
    rolling = df[signal_col].rolling(window=window, min_periods=window)
    df['long_thresh'] = rolling.quantile(long_quantile)
    df['short_thresh'] = rolling.quantile(short_quantile)
    df = df.dropna(subset=['long_thresh', 'short_thresh'])

    # 仓位：信号必须翻转才换仓，避免闪烁
    df['position'] = 0
    pos = 0
    positions = []

    for i, row in df.iterrows():
        signal = row[signal_col]
        # long_th = row['long_thresh']
        # short_th = row['short_thresh']

        long_th = short_quantile
        short_th = long_quantile

        if signal < long_th and pos != 1:
            pos = 1  # 做多
        elif signal > short_th and pos != -1:
            pos = -1  # 做空
        # 否则保持原来仓位
        positions.append(pos)

    df['position'] = positions

    # 换仓手续费
    df['position_shift'] = df['position'].shift(1).fillna(0)
    df['turnover'] = (df['position'] != df['position_shift']).astype(int)
    df['strategy_ret'] = df['position_shift'] * df['raw_ret'] - df['turnover'] * fee
    df['cumulative'] = (1 + df['strategy_ret']).cumprod()
    df['buy_hold'] = df[price_col] / df[price_col].iloc[0]

    # ➕ 绩效指标
    trade_id = (df['position'] != df['position_shift']).cumsum()
    df['trade_id'] = trade_id
    trades = df[df['turnover'] == 1]
    trade_returns = []

    for tid in trades['trade_id'].unique():
        sub = df[df['trade_id'] == tid]
        if len(sub) > 1:
            trade_return = (1 + sub['strategy_ret']).prod() - 1
            trade_returns.append(trade_return)

    trade_returns = np.array(trade_returns)
    trade_count = len(trade_returns)
    win_rate = (trade_returns > 0).sum() / trade_count

    total_return = df['cumulative'].iloc[-1] - 1
    sharpe = df['strategy_ret'].mean() / df['strategy_ret'].std()
    max_dd = ((df['cumulative'].cummax() - df['cumulative']) / df['cumulative'].cummax()).max()
    total_fee = df['turnover'].sum() * fee

    if plot:
        print("\n📊 策略绩效指标:")

        print(f"总交易次数: {trade_count}")
        print(f"总手续费: {total_fee:.4f}（费率: {fee:.4f}）")

        print(f"胜率: {win_rate:.2%}")
        print(f"总收益: {total_return:.2%}")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"最大回撤: {max_dd:.2%}")

        # 画净值曲线和持仓信号
        plt.figure(figsize=(14, 6))
        plt.plot(df[timestamp_col], df['cumulative'], label='Strategy')
        plt.plot(df[timestamp_col], df['buy_hold'], label='Buy & Hold', linestyle='--')
        plt.fill_between(df[timestamp_col], 0.98, 1.02, where=df['position'] == 1, color='green', alpha=0.1,
                         label='Long')
        plt.fill_between(df[timestamp_col], 0.98, 1.02, where=df['position'] == -1, color='red', alpha=0.1,
                         label='Short')
        plt.title('Rolling Quantile Backtest')
        plt.legend()
        plt.grid(True)
        plt.show()

    return df


def backtest_all_signals(
        df,
        signal_cols,
        price_col='price',
        timestamp_col='timestamp',
        long_quantile=0.8,
        short_quantile=0.2,
        window=500,
        fee=0.001,
        top_n=50
):
    results = []

    for signal_col in tqdm(signal_cols, desc="Backtesting Signals"):  # ⬅️ 加进度条
        df_result = single_asset_rolling_quantile_backtest(
            df=df,
            signal_col=signal_col,
            price_col=price_col,
            timestamp_col=timestamp_col,
            long_quantile=long_quantile,
            short_quantile=short_quantile,
            window=window,
            fee=fee,
            plot=False  # 改成可控的
        )

        # 取最后一行的累计收益等信息
        total_return = df_result['cumulative'].iloc[-1] - 1
        sharpe = df_result['strategy_ret'].mean() / df_result['strategy_ret'].std() if df_result[
                                                                                           'strategy_ret'].std() > 0 else 0
        max_dd = ((df_result['cumulative'].cummax() - df_result['cumulative']) / df_result['cumulative'].cummax()).max()
        total_fee = df_result['turnover'].sum() * fee

        results.append({
            'signal': signal_col,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_fee': total_fee,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by='total_return', ascending=False).reset_index(drop=True)

    # 打印前 top_n 的因子收益率
    print(f"\n📊 前 {top_n} 因子（按总收益排序）:")
    print(result_df.head(top_n).round(4))

    return result_df

def cal_z_score(x: pl.Expr, mean: pl.Expr, std: pl.Expr) -> pl.Expr:
    return (x - mean) / std

def scaled_sigmoid_expr(x: pl.Expr, start: pl.Expr, end: pl.Expr) -> pl.Expr:
    n = (start - end).abs()
    score = pl.lit(2) / (
            pl.lit(1) + (pl.lit(2.71828) ** (-pl.lit(4_000_000).log(10) * ((x - start - n) / n) + pl.lit(5e-3).log(10)))
    )
    return score / pl.lit(2)

def rolling_scaled_sigmoid_expr(
        x: str,
        mean_col: str,
        std_col: str,
        max_col: str,
        min_col: str,
) -> pl.Expr:
    return (
        pl.when(pl.col(std_col) == 0)
        .then(pl.lit(0.5))
        .otherwise(
            scaled_sigmoid_expr(
                cal_z_score(pl.col(x), pl.col(mean_col), pl.col(std_col)),
                pl.col(min_col),
                pl.col(max_col)
            )
        )
        .alias(f"{x}_scaled")  # 列命名
    )

def rolling_sum_expr(col_name: str, window: int) -> pl.Expr:
    return pl.col(col_name).rolling_sum(window).alias(f"{col_name}_sum_{window}")

def rolling_normalize_data(rollin_df: pl.DataFrame, window: int) -> pl.DataFrame:
    columns_to_normalize = [
        col for col in rollin_df.columns
        if col not in ['price', 'timestamp', 'timestamp_dt', 'symbol']
        and not col.startswith("future_return_")
        and not col.endswith('_scaled')  # scaled 是最终产物，保留
        and not (
            col.endswith('_rolling_mean') or
            col.endswith('_rolling_std') or
            col.endswith('_rolling_max') or
            col.endswith('_rolling_min')
        )
    ]

    rolling_cols = []
    for column in columns_to_normalize:
        rolling_cols.extend([
            pl.col(column).rolling_mean(window).alias(f"{column}_rolling_mean"),
            pl.col(column).rolling_std(window).alias(f"{column}_rolling_std"),
            pl.col(column).rolling_max(window).alias(f"{column}_rolling_max"),
            pl.col(column).rolling_min(window).alias(f"{column}_rolling_min"),
        ])

    intermediate_cols = [
        f"{column}_rolling_mean" for column in columns_to_normalize
    ] + [
        f"{column}_rolling_std" for column in columns_to_normalize
    ] + [
        f"{column}_rolling_max" for column in columns_to_normalize
    ] + [
        f"{column}_rolling_min" for column in columns_to_normalize
    ]

    normalized_df = (
        rollin_df
        .with_columns(rolling_cols)
        .with_columns([
            rolling_scaled_sigmoid_expr(
                column,
                f"{column}_rolling_mean",
                f"{column}_rolling_std",
                f"{column}_rolling_max",
                f"{column}_rolling_min",
            ) for column in columns_to_normalize
        ])
        .drop(intermediate_cols)
    )

    return normalized_df


def rolling_minmax_scaled_expr(
        col: str,
        min_col: str,
        max_col: str,
        scaled_col: str
) -> pl.Expr:
    return (
        ((pl.col(col) - pl.col(min_col)) / (pl.col(max_col) - pl.col(min_col) + 1e-9))
        .clip(0.0, 1.0)
        .alias(scaled_col)
    )

def rolling_minmax_normalize(rollin_df: pl.DataFrame, window: int) -> pl.DataFrame:
    columns_to_normalize = [
        col for col in rollin_df.columns
        if col not in ['price', 'timestamp', 'timestamp_dt', 'symbol']
           and not col.startswith("future_return_")
           and not col.endswith('_scaled')  # scaled 是最终产物，保留
           and not (
                col.endswith('_rolling_mean') or
                col.endswith('_rolling_std') or
                col.endswith('_rolling_max') or
                col.endswith('_rolling_min')
        )
    ]

    rolling_cols = []
    for column in columns_to_normalize:
        rolling_cols.extend([
            pl.col(column).rolling_max(window).alias(f"{column}_rolling_max"),
            pl.col(column).rolling_min(window).alias(f"{column}_rolling_min"),
        ])

    intermediate_cols = [
                            f"{column}_rolling_max" for column in columns_to_normalize
                        ] + [
                            f"{column}_rolling_min" for column in columns_to_normalize
                        ]

    return (
        rollin_df
        .with_columns(rolling_cols)
        .with_columns([
            rolling_minmax_scaled_expr(
                col=column,
                min_col=f"{column}_rolling_min",
                max_col=f"{column}_rolling_max",
                scaled_col=f"{column}_scaled"
            ) for column in columns_to_normalize
        ])
        .drop(intermediate_cols)
    )

def rolling_mean_tanh_scaled_expr(
        col: str,
        scaled_col: str,
        window: int
) -> pl.Expr:
    return (
        pl.col(col)
        .rolling_mean(window, min_samples=1)
        .tanh()
        .rolling_mean(window, min_samples=1)
        .alias(scaled_col)
    )

def rolling_mean_tanh_normalize(rollin_df: pl.DataFrame, window: int) -> pl.DataFrame:
    columns_to_normalize = [
        col for col in rollin_df.columns
        if col not in ['px', 'timestamp', 'timestamp_dt', 'symbol']
           and not col.startswith("future_return_")
           and not col.endswith('_scaled')
    ]

    return rollin_df.with_columns([
        rolling_mean_tanh_scaled_expr(
            col=column,
            scaled_col=f"{column}_scaled",
            window=window
        ) for column in columns_to_normalize
    ])

def features_automation(
    input_df_path: str,
):
    origin_df = pl.read_csv(input_df_path)
    monthly_dataframes = split_df_by_month(origin_df)
    for mo_df in monthly_dataframes:
        mo_df = batch_apply_transforms(mo_df, 200, 1)

        # check ram
        print(f"Polars DataFrame size: {mo_df.estimated_size() / (1024 ** 2):.4f} MB")

        # check avg steps
        prices_np = mo_df["price"].to_numpy()
        avg_steps_05pct = avg_steps_to_volatility(prices_np, 0.005)  # 波动1%
        avg_steps_1pct = avg_steps_to_volatility(prices_np, 0.01)  # 波动1%
        avg_steps_2pct = avg_steps_to_volatility(prices_np, 0.02)  # 波动1%
        print("波动 ±0.5% 的均值步数:", avg_steps_05pct)
        print("波动 ±1% 的均值步数:", avg_steps_1pct)
        print("波动 ±2% 的均值步数:", avg_steps_2pct)

        # cal future ret
        mo_df = mo_df.with_columns([
            future_return_expr("price", avg_steps_05pct),
            future_return_expr("price", avg_steps_1pct),
            future_return_expr("price", avg_steps_2pct),
        ])
        print(mo_df)

        mo_df = rolling_minmax_normalize(mo_df, 500)
        print(mo_df)
        # clean df via pandas drop cols and rows
        mo_df = clean_df_drop_nulls(mo_df)
        print(mo_df)
        # define exclude
        exclude_prefixes = ['price', 'timestamp', 'timestamp_dt', 'symbol']
        exclude_cols = exclude_prefixes + ['hour_group']

        # define target col
        target_col = f"future_return_{avg_steps_2pct}"

        # cal hourly ic
        ic_hourly = calc_hourly_rankic(
            mo_df,
            timestamp_col="timestamp",
            target_col=target_col,
            exclude_cols=exclude_cols
        )

        # print(ic_hourly)

        # get factor list filter by ic
        ic_summary = summarize_ic_df_wide(ic_hourly, exclude_cols)
        # print(ic_summary)

        ic_top_rank, ic_bot_rank, ir_top_rank = get_top_bottom_ic_ir(ic_summary, top_n=50)

        # get factor list filter by qcut
        res = calc_binned_return_and_stability(mo_df, future_return_col=target_col, n_bins=50)
        top_bin_ret = [r['factor'] for r in res['by_spread'][:50]]
        top_stability = [r['factor'] for r in res['by_stability'][:50]]
        top_monotonicity = [r['factor'] for r in res['by_monotonicity'][:50]]

        # print(ic_top_rank)
        # print(ic_bot_rank)
        # print(ir_top_rank)
        # print(top_bin_ret)
        # print(top_stability)
        # print(top_monotonicity)

        # filter factors
        factors = [
            col for col in mo_df.columns
            if col.endswith("_scaled")
               and all(not col.startswith(prefix) for prefix in exclude_prefixes)
               and not col.startswith("future_return_")
               and col != "price"
        ]
        keep_set = set(ic_top_rank) | set(ic_bot_rank) | set(ir_top_rank) | set(top_bin_ret) | set(top_stability) | set(top_monotonicity)
        filtered_factors = [f for f in factors if f in keep_set]

        result_df = backtest_all_signals(mo_df, filtered_factors)

        # 可选：打印保留了多少
        print(f"保留因子数：{len(filtered_factors)} / 原始因子数：{len(factors)}")

        # 可选：提取出新的 Polars 子 DataFrame
        filtered_df = mo_df.select(filtered_factors + [target_col])  # 加上目标列方便后续建模
        low_corr_factors = filter_by_spearman_corr(filtered_df, factor_cols=filtered_factors, threshold=0.99)

        # 第二步：再用 MI 选出前 k 个信息量最大的因子
        final_selected_factors = filter_by_mutual_info(filtered_df, low_corr_factors, target_col=target_col, top_k=30)
        print(final_selected_factors)

        top_k = 500
        for i, row in (result_df[['signal', 'total_return', 'sharpe', 'max_drawdown', 'total_fee']].head(top_k).round(4).iterrows()):
            print(f"{i + 1:>2}. Signal: {row['signal']:<40} | Return: {row['total_return']:<7} | Sharpe: {row['sharpe']:<5} | MaxDD: {row['max_drawdown']:<5} | Fee: {row['total_fee']}")

        del mo_df
        gc.collect()

def rolling_ic_ir_icto_index(
        df: pl.DataFrame,
        target_col: str,
        exclude_prefixes: list[str],
        window_size: int,
        step: int = 1,
) -> pl.DataFrame:
    feature_cols = [
        col for col in df.columns
        if col.endswith("_scaled")
           and all(not col.startswith(prefix) for prefix in exclude_prefixes)
           and not col.startswith("future_return_")
           and col != "price"
    ]

    n = df.height
    results = []
    prev_ranks = {}

    for start in tqdm(range(0, n - window_size + 1, step), desc="Rolling IC & ICTO"):
        end = start + window_size
        df_win = df.slice(start, window_size)

        # rank 转换
        df_ranked = df_win.with_columns([
            (pl.col(c).rank(method="average") / window_size).alias(c + "_rank") for c in feature_cols + [target_col]
        ])
        target_rank_col = target_col + "_rank"

        for feat in feature_cols:
            feat_rank_col = feat + "_rank"
            ic = df_ranked.select(
                pl.corr(pl.col(feat_rank_col), pl.col(target_rank_col)).alias("ic")
            ).to_series()[0]

            turnover = None
            if feat in prev_ranks:
                cur_ranks = df_ranked[feat_rank_col].to_numpy()
                prev = prev_ranks[feat]
                if len(prev) == len(cur_ranks):
                    turnover = np.mean(np.abs(cur_ranks - prev))

            # 更新 prev_ranks
            prev_ranks[feat] = df_ranked[feat_rank_col].to_numpy()

            results.append({
                "window_start": int(start),
                "window_end": int(end - 1),
                "factor": str(feat),
                "ic": float(ic) if not np.isnan(ic) else None,
                "turnover": float(turnover) if turnover is not None else None
            })

    df_result = pl.DataFrame(
        results,
        schema={
            "window_start": pl.Int64,
            "window_end": pl.Int64,
            "factor": pl.Utf8,
            "ic": pl.Float64,
            "turnover": pl.Float64,
        }
    )

    return (
        df_result
        .group_by("factor")
        .agg([
            pl.mean("ic").alias("mean_ic"),
            pl.std("ic").alias("std_ic"),
            pl.mean("turnover").alias("mean_turnover")
        ])
        .with_columns([
            (pl.col("mean_ic") / pl.col("std_ic")).alias("ir"),
            (pl.col("mean_ic") / (pl.col("mean_turnover") + 1e-8)).abs().alias("icto")
        ])
        .sort("icto", descending=True)
    )


def calculate_half_life(signal_series: pl.Series):
    df = pl.DataFrame({
        "x": signal_series
    }).with_columns([
        pl.col("x").shift(1).alias("x_lag")
    ]).drop_nulls()

    # 取出为 numpy 方便回归
    x = df["x"].to_numpy()
    x_lag = df["x_lag"].to_numpy()

    # 加上常数项做 OLS 回归
    X = np.vstack([np.ones_like(x_lag), x_lag]).T
    beta = np.linalg.lstsq(X, x, rcond=None)[0][1]

    # 防止 log(负数) 报错
    if beta <= 0 or beta >= 1:
        return float("nan")

    half_life = -math.log(2) / math.log(beta)
    return half_life

features_automation("C:/quant/data/binance_resampled_data/BTCUSDT_factors_threshold0.0005_rolling200.csv")
