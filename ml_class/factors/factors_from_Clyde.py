import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numba as nb
import time
import datetime
import scipy.stats as st
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import bootstrap
from functools import partial
from scipy.special import expit
import sys, traceback


# import swifter

def safe_log(x):
    if np.any(x == 0):
        raise ValueError(f"Log input contains zero.\nTraceback: {traceback.format_stack()}")
    return np.log(x)


# def safe_log(x):
#     if np.any(x == 0):
#         return np.log1p(x)
#     else:
#         # raise ValueError(f"Log input contains zero.\nTraceback: {traceback.format_stack()}")
#         return np.log(x)

# %%  计算这一行基于bid和ask的wap
def calc_wap1(depth, trade, pricetick=3):
    wap1 = (depth['bid_price1'] * depth['ask_size1'] + depth['ask_price1'] * depth['bid_size1']) / (
            depth['bid_size1'] + depth['ask_size1'])
    return round(wap1, pricetick)


def calc_swap1(df):
    return df['wap1'] - df['wap3']


def calc_swap12(df):
    return df['wap12'] - df['wap34']


def calc_tswap1(df):
    return -df['swap1'].diff()


def calc_tswap12(df):
    return -df['swap12'].diff()


def calc_wss12(df):
    ask = (df['ask_price1'] * df['ask_size1'] + df['ask_price2'] * df['ask_size2']) / (
            df['ask_size1'] + df['ask_size2'])
    bid = (df['bid_price1'] * df['bid_size1'] + df['bid_price2'] * df['bid_size2']) / (
            df['bid_size1'] + df['bid_size2'])
    mid = (df['ask_price1'] + df['bid_price1']) / 2
    return (ask - bid) / mid


# Calculate order book slope
def calc_slope(depth, trade):
    v0 = (depth['bid_size1'] + depth['ask_size1']) / 2
    p0 = (depth['bid_price1'] + depth['ask_price1']) / 2
    slope_bid = ((depth['bid_size1'] / v0) - 1) / abs((depth['bid_price1'] / p0) - 1) + (
            (depth['bid_size2'] / depth['bid_size1']) - 1) / abs((depth['bid_price2'] / depth['bid_price1']) - 1)
    slope_ask = ((depth['ask_size1'] / v0) - 1) / abs((depth['ask_price1'] / p0) - 1) + (
            (depth['ask_size2'] / depth['ask_size1']) - 1) / abs((depth['ask_price2'] / depth['ask_price1']) - 1)
    return (slope_bid + slope_ask) / 2, abs(slope_bid - slope_ask)


# Calculate order book dispersion
def calc_dispersion(depth, trade):
    bspread = depth['bid_price1'] - depth['bid_price2']
    aspread = depth['ask_price2'] - depth['ask_price1']
    bmid = (depth['bid_price1'] + depth['ask_price1']) / 2 - depth['bid_price1']
    bmid2 = (depth['bid_price1'] + depth['ask_price1']) / 2 - depth['bid_price2']
    amid = depth['ask_price1'] - (depth['bid_price1'] + depth['ask_price1']) / 2
    amid2 = depth['ask_price2'] - (depth['bid_price1'] + depth['ask_price1']) / 2
    bdisp = (depth['bid_size1'] * bmid + depth['bid_size2'] * bspread) / (depth['bid_size1'] + depth['bid_size2'])
    bdisp2 = (depth['bid_size1'] * bmid + depth['bid_size2'] * bmid2) / (depth['bid_size1'] + depth['bid_size2'])
    adisp = (depth['ask_size1'] * amid + depth['ask_size2'] * aspread) / (depth['ask_size1'] + depth['ask_size2'])
    adisp2 = (depth['ask_size1'] * amid + depth['ask_size2'] * amid2) / (depth['ask_size1'] + depth['ask_size2'])
    return (bdisp + adisp) / 2, (bdisp2 + adisp2) / 2


# Calculate order book depth
def calc_depth(df):
    depth = df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1'] + df['bid_price2'] * df[
        'bid_size2'] + df['ask_price2'] * df['ask_size2']
    return depth


#  order flow imbalance
def calc_ofi(depth, trade, rolling):
    a = depth['traded_bid_volume1'] * np.where(
        depth['traded_bid_price1'] - depth['traded_bid_price1'].shift(rolling) >= 0, 1, 0)
    b = (depth['traded_bid_volume1'] - depth['traded_bid_volume1'].shift(rolling)) * np.where(
        depth['traded_bid_price1'] - depth['traded_bid_price1'].shift(rolling) <= 0, 1, 0)
    c = depth['traded_ask_volume1'] * np.where(
        depth['traded_ask_price1'] - depth['traded_ask_price1'].shift(rolling) <= 0, 1, 0)
    d = (depth['traded_ask_volume1'] - depth['traded_ask_volume1'].shift(rolling)) * np.where(
        depth['traded_ask_price1'] - depth['traded_ask_price1'].shift(rolling) >= 0, 1, 0)
    calc_ofi = (a - b - c + d)
    sign_mask = np.sign(calc_ofi)
    value_mask = abs(calc_ofi)

    return sign_mask * np.sqrt(value_mask)


# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
@nb.njit
def log_return(series):
    return safe_log(series).diff()


# Calculate the realized volatility
@nb.njit
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))


@nb.njit
def realized_quarticity(series):
    return (np.count_nonzero(~np.isnan(series)) / 3) * np.sum(series ** 4)


@nb.njit
def reciprocal_transformation(series):
    return np.sqrt(1 / series) * 100000


@nb.njit
def square_root_translation(series):
    return series ** (1 / 2)


@nb.njit
def realized_absvar(series):
    # Calculate the realized absolute variation
    if np.count_nonzero(~np.isnan(series)) == 0:
        realized_absvar = 0
    else:
        realized_absvar = np.sqrt(np.pi / (2 * np.count_nonzero(~np.isnan(series)))) * np.sum(np.abs(series))
    return realized_absvar


@nb.njit
def realized_skew(series):
    if np.count_nonzero(~np.isnan(series)) == 0 or (realized_volatility(series) ** 3) == 0:
        realized_skew = 0
    else:
        realized_skew = np.sqrt(np.count_nonzero(~np.isnan(series))) * np.sum(series ** 3) / (
                realized_volatility(series) ** 3)
    return realized_skew


@nb.njit
# Calculate the realized kurtosis
def realized_kurtosis(series):
    if np.count_nonzero(~np.isnan(series)) == 0 or (realized_volatility(series) ** 4) == 0:
        realized_kurtosis = 0
    else:
        realized_kurtosis = np.count_nonzero(~np.isnan(series)) * np.sum(series ** 4) / (
                realized_volatility(series) ** 4)
    return realized_kurtosis


@nb.jit
def get_age(prices):
    last_value = prices[-1]
    age = 0
    for i in range(2, len(prices)):
        if prices[-i] != last_value:
            return age
        age += 1
    return age


def bid_age(depth, rolling=100):
    bp1 = depth['traded_bid_price1']
    bp1_changes = bp1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
    return bp1_changes


def ask_age(depth, rolling=100):
    ap1 = depth['traded_ask_price1']
    ap1_changes = ap1.rolling(rolling).apply(get_age, engine='numba', raw=True).fillna(0)
    return ap1_changes


def inf_ratio(depth=None, trade=None, rolling=100):
    quasi = (trade.price).diff().abs().rolling(rolling).sum()
    dif = (trade.price).diff(rolling).abs()
    return quasi / (dif + quasi)


def ask_price_range(depth=None, trade=None, rolling=100):
    ask_price_range = (depth.traded_ask_price1.rolling(rolling).max() - depth.traded_ask_price1.rolling(
        rolling).min()) / (depth.traded_ask_price1.rolling(rolling).max() + depth.traded_ask_price1.rolling(
        rolling).min())
    return ask_price_range


def bid_price_range(depth=None, trade=None, rolling=100):
    return safe_log(depth.traded_bid_price1.rolling(rolling).max() / depth.traded_bid_price1.rolling(rolling).min() - 1)


def avg_price_range(depth, trade=None, rolling=100, pricetick=7):
    # Data preprocessing
    avg_price = round(trade['amount'] / trade['volume'], pricetick)
    # Calculate rolling max and min
    avg_max = pd.Series(avg_price, index=trade.index).rolling(rolling).max()
    avg_min = pd.Series(avg_price, index=trade.index).rolling(rolling).min()

    # Calculate the range
    ask_range = (avg_max - avg_min) / (avg_max + avg_min)

    # Apply exponential weighted moving average for smoothing
    smoothed_ask_range = ask_range.ewm(span=rolling, adjust=False).mean()

    # Calculate rolling volatility
    volatility = smoothed_ask_range.rolling(rolling).std()

    # Normalize the factor
    normalized_factor = (smoothed_ask_range - smoothed_ask_range.rolling(rolling).mean()) / volatility

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(normalized_factor, [1, 99])
    winsorized_factor = np.clip(normalized_factor, lower, upper)

    return winsorized_factor


def arrive_rate(depth, trade, rolling=300):
    res = np.sqrt(trade['closetime'].diff(rolling) / rolling)
    return res


def bp_rank(depth, trade, rolling=100):
    return ((depth.traded_bid_price1.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)


def bp_rank_amount_tail(depth, trade, rolling=100):
    rank = ((depth.traded_bid_price1.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)
    high = pd.Series(rank, index=depth.index).rolling(rolling).quantile(0.9)
    low = pd.Series(rank, index=depth.index).rolling(rolling).quantile(0.1)
    high_amount = np.where(rank >= high, 1, 0)
    low_amount = np.where(rank <= low, 1, 0)

    high_amount_tail = pd.Series(high_amount, index=depth.index).rolling(rolling).sum() / rolling
    low_amount_tail = pd.Series(low_amount, index=depth.index).rolling(rolling).sum() / rolling

    return np.sqrt(high_amount_tail), np.sqrt(low_amount_tail)


def ap_rank(depth, trade, rolling=100):
    return ((depth.traded_ask_price1.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)


def ap_rank_amount_tail(depth, trade, rolling=100):
    rank = ((depth.traded_ask_price1.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)
    high = pd.Series(rank, index=depth.index).rolling(rolling).quantile(0.9)
    low = pd.Series(rank, index=depth.index).rolling(rolling).quantile(0.1)
    high_amount = np.where(rank >= high, 1, 0)
    low_amount = np.where(rank <= low, 1, 0)

    high_amount_tail = pd.Series(high_amount, index=depth.index).rolling(rolling).sum() / rolling
    low_amount_tail = pd.Series(low_amount, index=depth.index).rolling(rolling).sum() / rolling

    return np.sqrt(high_amount_tail), np.sqrt(low_amount_tail)


def avg_rank(depth, trade, rolling=100, pricetick=7):
    a = trade['amount'].copy().ffill()
    v = trade['volume'].copy().ffill()

    avg = round(a / v, pricetick)
    return ((pd.Series(avg, index=trade.index).rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)


def amount_rank(depth, trade, rolling=120):
    signl_amount = trade['amount'].copy().ffill()
    return ((signl_amount.rolling(rolling).rank()) / rolling * 2 - 1).fillna(0)


def price_impact(depth, trade, level=10):
    ask = depth['total_ask_amount']
    bid = depth['total_bid_amount']
    ask_v = depth['total_ask_volume']
    bid_v = depth['total_bid_volume']
    ask /= ask_v
    bid /= bid_v
    return pd.Series(
        -(depth['traded_bid_price1'] - ask) / depth['traded_ask_price1'] - (depth['traded_bid_price1'] - bid) / depth[
            'traded_ask_price1'],
        name="price_impact")


# todo
def depth_price_skew(depth, trade):
    prices = ["bid_price10", "bid_price9", "bid_price8", "bid_price7", "bid_price6", "bid_price5", "bid_price4",
              "bid_price3", "bid_price2", "bid_price1",
              "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5", "ask_price6", "ask_price7",
              "ask_price8", "ask_price9", "ask_price10"]
    return depth[prices].skew(axis=1)


# todo
def depth_price_kurt(depth, trade):
    prices = ["bid_price10", "bid_price9", "bid_price8", "bid_price7", "bid_price6", "bid_price5", "bid_price4",
              "bid_price3", "bid_price2", "bid_price1", "ask_price1", "ask_price2",
              "ask_price3", "ask_price4", "ask_price5", "ask_price6", "ask_price7", "ask_price8", "ask_price9",
              "ask_price10"]
    return depth[prices].kurt(axis=1)


def rolling_return(depth, trade, rolling=100):
    mp = ((depth['traded_ask_price1'].ffill() + depth['traded_bid_price1'].ffill()) / 2)
    rolling_return = (mp.diff(rolling) / mp)
    # sign_mask = np.sign(rolling_return)
    # value_mask = abs(rolling_return)
    return rolling_return


def buy_increasing(depth, trade, rolling=100):
    v = trade['buy_qty']
    return np.log1p(
        (((v.fillna(0)).rolling(2 * rolling).sum() + 1) / ((v.fillna(0)).rolling(rolling).sum() + 1)).fillna(1))


def buy_increasing(depth, trade, rolling=100):
    # Data preprocessing
    buy_qty = trade['buy_qty']

    # Calculate rolling sums
    long_sum = pd.Series(buy_qty, index=trade.index).rolling(2 * rolling).sum()
    short_sum = pd.Series(buy_qty, index=trade.index).rolling(rolling).sum()

    # Calculate the ratio
    ratio = (long_sum + 1) / (short_sum + 1)

    # Apply log1p transformation
    log_ratio = np.log1p(ratio)

    # Apply exponential weighted moving average for smoothing
    smoothed_log_ratio = log_ratio.ewm(span=rolling, adjust=False).mean()

    # Calculate rolling volatility
    volatility = smoothed_log_ratio.rolling(rolling).std()

    # Normalize the factor
    normalized_factor = (smoothed_log_ratio - smoothed_log_ratio.rolling(rolling).mean()) / volatility

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(normalized_factor, [1, 99])
    winsorized_factor = np.clip(normalized_factor, lower, upper)

    return winsorized_factor


def sell_increasing(depth, trade, rolling=100):
    # Data preprocessing
    sell_qty = abs(trade['sell_qty'])

    # Calculate rolling sums
    long_sum = pd.Series(sell_qty, index=trade.index).rolling(2 * rolling).sum()
    short_sum = pd.Series(sell_qty, index=trade.index).rolling(rolling).sum()

    # Calculate the ratio
    ratio = (long_sum + 1) / (short_sum + 1)

    # Apply log1p transformation
    log_ratio = np.log1p(ratio)

    # Apply exponential weighted moving average for smoothing
    smoothed_log_ratio = log_ratio.ewm(span=rolling / 2, adjust=False).mean()

    # Calculate rolling volatility
    volatility = smoothed_log_ratio.rolling(rolling).std()

    # Normalize the factor
    normalized_factor = (smoothed_log_ratio - smoothed_log_ratio.rolling(rolling).mean()) / volatility

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(normalized_factor, [1, 99])
    winsorized_factor = np.clip(normalized_factor, lower, upper)

    return winsorized_factor


@nb.jit
def first_location_of_maximum(x):
    max_value = max(x)  # 一个for 循环
    for loc in range(len(x)):
        if x[loc] == max_value:
            return loc + 1


@nb.jit
def last_location_of_maximum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN


@nb.jit
def first_location_of_minimum(x):
    x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN


@nb.jit
def last_location_of_minimum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def avg_price_first_idxmax(depth, trade, rolling=20, pricetick=7):
    avg_price = round(trade['amount'] / trade['volume'], pricetick)
    return avg_price.rolling(rolling).apply(first_location_of_maximum, engine='numba',
                                            raw=True).fillna(0)


def avg_price_last_idxmin(depth, trade, rolling=20, pricetick=7):
    avg_price = round(trade['amount'] / trade['volume'], pricetick)
    return avg_price.rolling(rolling).apply(last_location_of_minimum, engine='numba',
                                            raw=True).fillna(0)


def ask_price_first_idxmax(depth, trade, rolling=20):
    return depth['traded_ask_price1'].rolling(rolling).apply(first_location_of_maximum, engine='numba',
                                                             raw=True).fillna(0)


def bid_price_first_idxmax(depth, trade, rolling=20):
    return depth['traded_bid_price1'].rolling(rolling).apply(first_location_of_maximum, engine='numba',
                                                             raw=True).fillna(0)


def ask_price_last_idxmax(depth, trade, rolling=20):
    return depth['traded_ask_price1'].rolling(rolling).apply(last_location_of_maximum, engine='numba', raw=True).fillna(
        0)


def bid_price_last_idxmax(depth, trade, rolling=20):
    return depth['traded_bid_price1'].rolling(rolling).apply(last_location_of_maximum, engine='numba', raw=True).fillna(
        0)


def ask_price_first_idxmin(depth, trade, rolling=20):
    return depth['traded_ask_price1'].rolling(rolling).apply(first_location_of_minimum, engine='numba',
                                                             raw=True).fillna(0)


def bid_price_first_idxmin(depth, trade, rolling=20):
    return depth['traded_bid_price1'].rolling(rolling).apply(first_location_of_minimum, engine='numba',
                                                             raw=True).fillna(0)


def ask_price_last_idxmin(depth, trade, rolling=20):
    return depth['traded_ask_price1'].rolling(rolling).apply(last_location_of_minimum, engine='numba', raw=True).fillna(
        0)


def bid_price_last_idxmin(depth, trade, rolling=20):
    return depth['traded_bid_price1'].rolling(rolling).apply(last_location_of_minimum, engine='numba', raw=True).fillna(
        0)


@nb.jit
def mean_second_derivative_centra(x):
    sum_value = 0
    for i in range(len(x) - 5):
        sum_value += (x[i + 5] - 2 * x[i + 3] + x[i]) / 2
    return sum_value / (2 * (len(x) - 5))


def center_deri_two(depth, trade, rolling=20, pricetick=7):
    avg_price = round(trade['amount'] / trade['volume'], pricetick)
    center_deri_two = pd.Series(avg_price, index=trade.index).rolling(rolling).apply(mean_second_derivative_centra,
                                                                                     engine='numba', raw=True)
    # sign_mask = np.sign(center_deri_two)
    # value_mask = abs(center_deri_two)
    return center_deri_two


def quasi(depth, trade, rolling=100):
    # quasi = trade.price.diff(1).abs().rolling(rolling).sum()
    return safe_log(trade.price.diff(1).abs().rolling(rolling).sum())


def last_range(depth, trade, rolling=100):
    # Data preprocessing
    price = trade['price']

    # Calculate log returns
    log_returns = safe_log(price / price.shift(1))

    # Calculate absolute log returns
    abs_log_returns = pd.Series(np.abs(log_returns), index=trade.index)

    # Apply exponential weighted moving average for smoothing
    smoothed_abs_log_returns = abs_log_returns.ewm(span=rolling, adjust=False).mean()

    # Calculate the rolling sum and take the square root
    last_range = np.sqrt(smoothed_abs_log_returns.rolling(rolling).sum())

    # Calculate rolling volatility of the factor
    volatility = last_range.rolling(rolling).std()

    # Normalize the factor
    normalized_factor = (last_range - last_range.rolling(rolling).mean()) / volatility

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(normalized_factor, [1, 99])
    winsorized_factor = np.clip(normalized_factor, lower, upper)

    return winsorized_factor


# def arrive_rate(depth, trade, rolling=100):
#     return (trade.ts.shift(rolling) - trade.ts).fillna(0)

def avg_trade_volume(depth, trade, rolling=100):
    return safe_log((((trade['volume'].ffill()))[::-1].abs().rolling(rolling).sum().shift(-rolling + 1))[::-1])


def avg_spread(depth, trade, rolling=200):
    avg_spread = (abs(depth.traded_ask_price1 - depth.traded_bid_price1).rolling(rolling).mean()) / (
        abs(depth.traded_ask_price1 - depth.traded_bid_price1).rolling(rolling).std())
    return np.sqrt(abs(avg_spread)) * np.sign(avg_spread)


# todo
def avg_turnover(depth, trade, rolling=500):
    return depth[
        ['ask_size1', 'ask_size2', 'ask_size3', 'ask_size4', "ask_size5", "ask_size6", "ask_size7", "ask_size8",
         "ask_size9", "ask_size10",
         'bid_size1', 'bid_size2', 'bid_size3', 'bid_size4', "bid_size5", "bid_size6", "bid_size7", "bid_size8",
         "bid_size9", "bid_size10"]].sum(axis=1)


def abs_volume_kurt(depth, trade, rolling=500):
    size = trade['buy_qty'] + trade['sell_qty']
    abs_volume_kurt = size.rolling(rolling).kurt()
    sign_mask = np.sign(abs_volume_kurt)
    value_mask = abs(abs_volume_kurt)
    return sign_mask * np.sqrt(value_mask)


def abs_volume_skew(depth, trade, rolling=500):
    size = trade['buy_qty'] + trade['sell_qty']
    abs_volume_skew = size.rolling(rolling).skew()
    sign_mask = np.sign(abs_volume_skew)
    value_mask = abs(abs_volume_skew)
    return sign_mask * np.sqrt(value_mask)


# todo
def price_kurt(depth, trade, rolling=500):
    return (trade.price.fillna(0)).rolling(rolling).kurt().fillna(0)


# todo
def price_skew(depth, trade, rolling=500):
    return (trade.price.fillna(0)).rolling(rolling).skew().abs().fillna(0)


def bv_divide_tn(depth, trade, rolling=10):
    bvs = depth['total_bid_volume']
    v = abs(trade['sell_qty'])
    return np.sqrt((v).rolling(rolling).sum() / bvs)


def av_divide_tn(depth, trade, rolling=10):
    avs = depth['total_ask_volume']
    v = trade['buy_qty']
    return np.sqrt(((v).rolling(rolling).sum() / avs))


def optimized_volume_imbalance_factors(depth, trade, rolling=10):
    # Data preprocessing
    bvs = depth['total_bid_volume']
    avs = depth['total_ask_volume']
    sell_qty = abs(trade['sell_qty'])
    buy_qty = trade['buy_qty']

    # Calculate ratios
    sell_ratio = (pd.Series(sell_qty, index=trade.index).rolling(rolling).sum() / bvs)
    buy_ratio = (pd.Series(buy_qty, index=trade.index).rolling(rolling).sum() / avs)

    # Apply log transformation to reduce skewness
    log_sell_ratio = np.log1p(sell_ratio)
    log_buy_ratio = np.log1p(buy_ratio)

    # Calculate z-scores
    def calculate_zscore(series, rolling):
        return (series - series.rolling(rolling).mean()) / series.rolling(rolling).std()

    zscore_sell = calculate_zscore(log_sell_ratio, rolling)
    zscore_buy = calculate_zscore(log_buy_ratio, rolling)

    # Apply Winsorization to handle outliers
    def winsorize(series, limits=(0.01, 0.99)):
        lower, upper = np.percentile(series, [limits[0] * 100, limits[1] * 100])
        return np.clip(series, lower, upper)

    winsorized_sell = winsorize(zscore_sell)
    winsorized_buy = winsorize(zscore_buy)

    # Apply exponential weighted moving average for smoothing
    ema_sell = winsorized_sell.ewm(span=rolling, adjust=False).mean()
    ema_buy = winsorized_buy.ewm(span=rolling, adjust=False).mean()

    # Calculate the final factors
    bv_divide_tn = ema_sell
    av_divide_tn = ema_buy

    # Calculate a composite factor
    composite_factor = bv_divide_tn - av_divide_tn

    return bv_divide_tn, av_divide_tn, composite_factor


def weighted_price_to_mid(depth, trade):
    mp = (depth['traded_ask_price1'] + depth['traded_bid_price1']) / 2
    return (depth['total_ask_amount'] + depth['total_bid_amount']) / (
            depth['total_ask_volume'] + depth['total_bid_volume']) - mp


@nb.njit
def _bid_withdraws_volume(l, n, levels=10):
    withdraws = 0
    for price_index in range(2, 2 + 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(2, 2 + 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min((n[price_index + 1]) - l[price_last_index + 1], 0)

    return withdraws


@nb.njit
def _ask_withdraws_volume(l, n, levels=10):
    withdraws = 0
    for price_index in range(0, 4 * levels, 4):
        now_p = n[price_index]
        for price_last_index in range(0, 4 * levels, 4):
            if l[price_last_index] == now_p:
                withdraws -= min(n[price_index + 1] - l[price_last_index + 1], 0)

    return withdraws


# todo
def ask_withdraws(depth, trade):
    ob_values = depth.iloc[:, 1:].values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _ask_withdraws_volume(ob_values[i - 1], ob_values[i])
    return flows


# todo
def bid_withdraws(depth, trade):
    ob_values = depth.iloc[:, 1:].values
    flows = np.zeros(len(ob_values))
    for i in range(1, len(ob_values)):
        flows[i] = _bid_withdraws_volume(ob_values[i - 1], ob_values[i])
    return flows


# def z_t(trade, depth):
#     """初探市场微观结构：指令单薄与指令单流——资金交易策略之四 成交价的对数减去中间价的对数"""
#     # data_dic = self.data_dic  # 调用的是属性
#     tick_fac_data = safe_log(trade['price']) - safe_log((depth['traded_bid_price1'] + depth['traded_ask_price1']) / 2)
#     sign_mask = np.sign(tick_fac_data)
#     value_mask = abs(tick_fac_data)
#     return sign_mask * np.sqrt(value_mask)
def z_t(trade, depth, rolling=10, pricetick=7):
    # Data preprocessing
    bid_price = depth['traded_bid_price1']
    ask_price = depth['traded_ask_price1']
    avg = round(trade['amount'] / trade['volume'], pricetick)
    # Calculate mid price
    mid_price = (bid_price + ask_price) / 2

    # Calculate log difference
    log_diff = safe_log(avg) - safe_log(mid_price)

    # Apply exponential weighted moving average for smoothing
    smoothed_log_diff = pd.Series(log_diff, index=trade.index).ewm(span=rolling, adjust=False).mean()

    # Calculate rolling volatility
    volatility = smoothed_log_diff.rolling(rolling).std()

    # Normalize the factor
    normalized_factor = smoothed_log_diff / volatility

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(normalized_factor, [1, 99])
    winsorized_factor = np.clip(normalized_factor, lower, upper)

    # Final transformation
    final_factor = np.sign(winsorized_factor) * np.sqrt(np.abs(winsorized_factor))

    return final_factor


def voi(depth, trade):
    """voi订单失衡 Volume Order Imbalance20200709-中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    bid_sub_price = depth['traded_bid_price1'] - depth['traded_bid_price1'].shift(1)
    ask_sub_price = depth['traded_ask_price1'] - depth['traded_ask_price1'].shift(1)

    bid_sub_volume = depth['traded_bid_volume1'] - depth['traded_bid_volume1'].shift(1)
    ask_sub_volume = depth['traded_ask_volume1'] - depth['traded_ask_volume1'].shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = depth['traded_bid_volume1'][bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = depth['traded_ask_volume1'][ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / (trade['volume'])
    # sign_mask = np.sign(tick_fac_data)
    # value_mask = abs(tick_fac_data)
    return np.sign(tick_fac_data) * np.sqrt(abs(tick_fac_data))


# todo
def cal_weight_volume(depth):
    """计算加权的盘口挂单量"""
    # data_dic = self.data_dic
    w = [1 - (i - 1) / 10 for i in range(1, 11)]
    w = np.array(w) / sum(w)
    wb = depth['bid_size1'] * w[0] + depth['bid_size2'] * w[1] + depth['bid_size3'] * w[2] + depth['bid_size4'] * w[3] + \
         depth['bid_size5'] * w[4] + depth['bid_size6'] * w[5] + depth['bid_size7'] * w[6] + depth['bid_size8'] * w[7] + \
         depth['bid_size9'] * w[8] + depth['bid_size10'] * w[9]
    wa = depth['ask_size1'] * w[0] + depth['ask_size2'] * w[1] + depth['ask_size3'] * w[2] + depth['ask_size4'] * w[3] + \
         depth['ask_size5'] * w[4] + depth['ask_size6'] * w[5] + depth['ask_size7'] * w[6] + depth['ask_size8'] * w[7] + \
         depth['ask_size9'] * w[8] + depth['ask_size10'] * w[9]
    return wb, wa


# todo
def voi2(depth, trade):
    """同voi，衰减加权，"""
    # data_dic = self.data_dic
    wb, wa = cal_weight_volume(depth)
    bid_sub_price = depth['bid_price1'] - depth['bid_price1'].shift(1)
    ask_sub_price = depth['ask_price1'] - depth['ask_price1'].shift(1)

    bid_sub_volume = wb - wb.shift(1)
    ask_sub_volume = wa - wa.shift(1)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = wb[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = wa[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / (trade['volume'])  # 自动行列对齐
    return tick_fac_data


def mpb(depth, trade):
    """市价偏离度 Mid-Price Basis 中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    tp = (trade['amount']) / (trade['volume'])  # 注意单位，
    # print(tp)
    tp[np.isinf(tp)] = np.nan
    tp.fillna(method='ffill', inplace=True)
    mid = (depth['traded_bid_price1'] + depth['traded_ask_price1']) / 2
    tick_fac_data = tp / ((mid + mid.shift(1)) / 2)
    return tick_fac_data.fillna(0)


def slope(depth, trade):
    """斜率 价差/深度"""
    # data_dic = self.data_dic
    tick_fac_data = (depth['traded_ask_price1'] - depth['traded_bid_price1']) / (
            depth['traded_ask_amount1'] + depth['traded_bid_amount1'])
    sign_mask = np.sign(tick_fac_data)
    value_mask = abs(tick_fac_data)
    return sign_mask * np.sqrt(value_mask)


# def positive_ratio(depth, trade,rolling=20 * 3):
#     """积极买入成交额占总成交额的比例"""
#     # data_dic = self.data_dic
#     buy_positive = pd.DataFrame(0, columns=['amount'], index=trade['amount'].index)
#     buy_positive['amount'] = trade['amount']
#     # buy_positive[trade['price'] >= depth['ask_price1'].shift(1)] = trade['amount'][trade['price'] >= depth['ask_price1'].shift(1)]
#     buy_positive['amount'] = np.where(trade['price']>depth['ask_price1'], buy_positive['amount'], 0)
#     tick_fac_data = buy_positive['amount'].rolling(rolling).sum() / trade['amount'].rolling(rolling).sum()
#     return tick_fac_data

# todo
def price_weighted_pressure(depth, kws):
    n1 = kws.setdefault("n1", 1)
    n2 = kws.setdefault("n2", 10)

    bench = kws.setdefault("bench_type", "MID")

    _ = np.arange(n1, n2 + 1)

    if bench == "MID":
        bench_prices = depth['ask_price1'] + depth['bid_price1']
    elif bench == "SPECIFIC":
        bench_prices = kws.get("bench_price")
    else:
        raise Exception("")

    def unit_calc(bench_price):
        """比结算价高的价单立马成交，权重=0"""

        bid_d = [bench_price / (bench_price - depth["bid_price%s" % s]) for s in _]
        # bid_d = [_.replace(np.inf,0) for _ in bid_d]
        bid_denominator = sum(bid_d)

        bid_weights = [(d / bid_denominator).replace(np.nan, 1) for d in bid_d]

        press_buy = sum([depth["bid_size%s" % (i + 1)] * w for i, w in enumerate(bid_weights)])

        ask_d = [bench_price / (depth['ask_price%s' % s] - bench_price) for s in _]
        # ask_d = [_.replace(np.inf,0) for _ in ask_d]
        ask_denominator = sum(ask_d)

        ask_weights = [d / ask_denominator for d in ask_d]

        press_sell = sum([depth['ask_size%s' % (i + 1)] * w for i, w in enumerate(ask_weights)])

        return (safe_log(press_buy) - safe_log(press_sell)).replace([-np.inf, np.inf], np.nan)

    return unit_calc(bench_prices)


# todo
def volume_order_imbalance(depth, kws):
    """
    Reference From <Order imbalance Based Strategy in High Frequency Trading>
    :param data:
    :param kws:
    :return:
    """
    drop_first = kws.setdefault("drop_first", True)

    current_bid_price = depth['bid_price1']

    bid_price_diff = current_bid_price - current_bid_price.shift()

    current_bid_vol = depth['bid_size1']

    nan_ = current_bid_vol[current_bid_vol == 0].index

    bvol_diff = current_bid_vol - current_bid_vol.shift()

    bid_increment = np.where(bid_price_diff > 0, current_bid_vol,
                             np.where(bid_price_diff < 0, 0, np.where(bid_price_diff == 0, bvol_diff, bid_price_diff)))

    current_ask_price = depth['ask_price1']

    ask_price_diff = current_ask_price - current_ask_price.shift()

    current_ask_vol = depth['ask_size1']

    avol_diff = current_ask_vol - current_ask_vol.shift()

    ask_increment = np.where(ask_price_diff < 0, current_ask_vol,
                             np.where(ask_price_diff > 0, 0, np.where(ask_price_diff == 0, avol_diff, ask_price_diff)))

    _ = pd.Series(bid_increment - ask_increment, index=depth.index)

    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan

    _.loc[nan_] = np.nan

    return _.fillna(0)


# todo
def get_mid_price_change(depth, drop_first=True):
    mid = (depth['ask_price1'] + depth['bid_price1']) / 2
    _ = mid.pct_change()
    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan
    return _


def mpc(depth, trade, rolling=500):
    mid = (depth['traded_ask_price1'] + depth['traded_bid_price1']) / 2
    mpc = (mid - mid.shift(rolling)) / mid.shift(rolling)
    sign_mask = np.sign(mpc)
    value_mask = abs(mpc)
    return sign_mask * np.sqrt(value_mask)


def mpb_500(depth, trade, rolling=500, pricetick=7):
    """市价偏离度 Mid-Price Basis 中信建投-因子深度研究系列：高频量价选股因子初探"""
    # data_dic = self.data_dic
    tp = round((trade['amount'].rolling(rolling).sum()) / (trade['volume'].rolling(rolling).sum()), pricetick)  # 注意单位，
    # print(tp)
    tp[np.isinf(tp)] = np.nan
    tp = tp.ffill()
    mid = (depth['traded_ask_price1'] + depth['traded_bid_price1']) / 2
    tick_fac_data = tp - ((mid + mid.shift(rolling)) / 2)
    sign_mask = np.sign(tick_fac_data)
    value_mask = abs(tick_fac_data)
    return sign_mask * np.sqrt(value_mask)


# todo best
def positive_buying(depth, trade, rolling=1000, pricetick=7):
    # v = trade['size'].ffill().copy()
    a = trade['amount'].ffill().copy()
    vo = trade['volume'].ffill().copy()
    avg = round(a / vo, pricetick)

    positive_buy = np.where(avg > depth['traded_ask_price1'], abs(vo), 0)
    caustious_buy = np.where(avg < depth['traded_bid_price1'], abs(vo), 0)
    bm = pd.Series(positive_buy, index=trade.index).rolling(rolling).sum() / pd.Series(caustious_buy,
                                                                                       index=trade.index).rolling(
        rolling).sum()

    # sign_mask = np.sign(bm)
    # value_mask = abs(bm)
    return np.sqrt(bm)


def positive_selling(depth, trade, rolling=60, pricetick=7):
    a = trade['amount'].ffill()
    vo = trade['volume'].ffill()
    avg = round(a / vo, pricetick)
    positive_sell = np.where(avg < depth['traded_bid_price1'], abs(vo), 0)
    caustious_sell = np.where(avg > depth['traded_ask_price1'], abs(vo), 0)
    sm = pd.Series(positive_sell, index=trade.index).rolling(rolling).sum() / pd.Series(caustious_sell,
                                                                                        index=trade.index).rolling(
        rolling).sum()
    # sign_mask = np.sign(sm)
    # value_mask = abs(sm)
    return np.sqrt(sm) * -1


def buying_amplification_ratio(depth, trade, rolling):
    a = trade['amount'].ffill()
    biding = depth['total_bid_amount']
    asking = depth['total_ask_amount']
    amplify_biding = np.where(biding > biding.shift(1), biding - biding.shift(1), 0)
    amplify_asking = np.where(asking > asking.shift(1), asking - asking.shift(1), 0)
    diff = amplify_biding - amplify_asking
    buying_ratio = (
                pd.Series(diff, index=trade.index).rolling(rolling).mean() / pd.Series(diff, index=trade.index).rolling(
            rolling).std())
    sign_mask = np.sign(buying_ratio)
    value_mask = abs(buying_ratio)
    return sign_mask * np.sqrt(value_mask)


def buying_amount_ratio(depth, trade, rolling, pricetick=7):
    a = trade['amount'].ffill()
    vo = trade['volume'].ffill()
    avg = round(a / vo, pricetick)
    positive_buy = np.where(avg > depth['traded_ask_price1'], abs(a),
                            0)
    positive_sell = np.where(avg < depth['traded_bid_price1'], abs(a),
                             0)
    diff = positive_buy - positive_sell
    buying_amount_ratio = ((pd.Series(diff, index=trade.index).rolling(rolling).sum()) /
                           (a.rolling(rolling).sum())) / rolling

    sign_mask = np.sign(buying_amount_ratio)
    value_mask = abs(buying_amount_ratio)
    return sign_mask * np.sqrt(value_mask)


def buying_willing(depth, trade, rolling, pricetick=7):
    a = trade['amount'].ffill().copy()
    vo = trade['volume'].ffill().copy()
    avg = round(a / vo, pricetick)
    biding = depth['total_bid_amount']
    asking = depth['total_ask_amount']
    amplify_biding = np.where(biding > biding.shift(1), biding - biding.shift(1), 0)
    amplify_asking = np.where(asking > asking.shift(1), asking - asking.shift(1), 0)
    positive_buy = np.where(avg > depth['traded_ask_price1'], abs(a), 0)
    positive_sell = np.where(avg < depth['traded_bid_price1'], abs(a), 0)
    diff = (amplify_biding - amplify_asking) + (positive_buy - positive_sell)
    buying_willing = pd.Series(
        (pd.Series(diff, index=trade.index).rolling(rolling).sum()) / (a.rolling(rolling).sum())) / rolling

    sign_mask = np.sign(buying_willing)
    value_mask = abs(buying_willing)
    return sign_mask * np.sqrt(value_mask)


def buying_willing_strength(depth, trade, rolling, pricetick):
    a = trade['amount'].ffill().copy()
    vo = trade['volume'].ffill().copy()
    avg = round(a / vo, pricetick)
    biding = depth['total_bid_volume']
    asking = depth['total_ask_volume']
    positive_buy = np.where(avg > depth['traded_ask_price1'], abs(a),
                            np.where(avg - depth['traded_bid_price1'] > depth['traded_ask_price1'] - avg, a, 0))
    positive_sell = np.where(avg < depth['traded_bid_price1'], abs(a),
                             np.where(avg - depth['traded_bid_price1'] < depth['traded_ask_price1'] - avg, a, 0))
    diff = ((positive_buy - positive_sell) / (biding - asking))
    buying_stength = (pd.Series(diff, index=trade.index).rolling(rolling).mean()) / (
        pd.Series(diff, index=trade.index).rolling(rolling).std())
    value_mask = abs(buying_stength)
    sign_mask = np.sign(buying_stength)
    return np.sqrt(value_mask) * sign_mask


def buying_amount_strength(depth, trade, rolling, pricetick):
    a = trade['amount'].ffill().copy()
    vo = trade['volume'].ffill().copy()
    avg = round(a / vo, pricetick)
    positive_buy = np.where(avg > depth['traded_ask_price1'], abs(a),
                            np.where(avg - depth['traded_bid_price1'] > depth['traded_ask_price1'] - avg, a, 0))
    positive_sell = np.where(avg < depth['traded_bid_price1'], abs(a),
                             np.where(avg - depth['traded_bid_price1'] < depth['traded_ask_price1'] - avg, a, 0))
    diff = (positive_buy - positive_sell)
    buying_amount_strength = (pd.Series(diff, index=trade.index).rolling(rolling).mean()) / (
        pd.Series(diff, index=trade.index).rolling(rolling).std())
    value_mask = abs(buying_amount_strength)
    sign_mask = np.sign(buying_amount_strength)
    return np.sqrt(value_mask) * sign_mask


def selling_ratio(depth, trade, rolling, pricetick):
    a = trade['amount'].ffill().copy()
    vo = trade['volume'].ffill().copy()
    avg = round(a / vo, pricetick)
    biding = depth['total_bid_volume']
    asking = depth['total_ask_volume']
    amplify_biding = np.where(biding > biding.shift(1), biding - biding.shift(1), 0)
    amplify_asking = np.where(asking > asking.shift(1), asking - asking.shift(1), 0)
    diff = amplify_asking - amplify_biding
    # amount = trade['amount'].copy().reset_index(drop=True)
    selling_ratio = (pd.Series(diff, index=trade.index).rolling(rolling).sum()) / (a.rolling(rolling).sum()) / rolling
    value_mask = abs(selling_ratio)
    sign_mask = np.sign(selling_ratio)
    return np.sqrt(value_mask) * sign_mask


@nb.njit
def intervel_time_mean(row):
    # differences = 0
    intervel_time_mean = np.nanmean(np.diff(row[row != 0]))
    # differences = np.nanmean(differences)
    return intervel_time_mean


@nb.njit
def intervel_time_std(row):
    # differences = 0
    # intervel_time_std
    intervel_time_std = np.nanstd(np.diff(row[row != 0]))
    # differences = np.nanstd(differences)
    # intervel_time_std = np.std(differences)
    return intervel_time_std


@nb.njit
def intervel_time_mean_std(row):
    intervel_time_mean = np.nanmean(np.diff(row[row != 0]))
    intervel_time_std = np.nanstd(np.diff(row[row != 0]))
    return intervel_time_mean / intervel_time_std


# todo
def buy_large_order(depth, trade, rolling=120 * 2):
    s = trade['buy_qty'] + trade['sell_qty']
    v = trade['volume']
    buy_order = np.where(s > 0, v, 0)
    mean = pd.Series(buy_order, index=trade.index).rolling(rolling).mean()
    std = pd.Series(buy_order, index=trade.index).rolling(rolling).std()
    large = np.where(buy_order > (mean + 3 * std), 1, 0)
    buy_large_order_ratio = np.sqrt((pd.Series(large, index=trade.index).rolling(rolling).sum()) / rolling)
    # buy_large_order_trend = (pd.Series(large, index=trade.index).rolling(rolling).sum()) / (
    #         pd.Series(large, index=trade.index).rolling(rolling * 3).sum() / (rolling * 3))

    buy_large_order_interval_time = np.where(buy_order > (mean + 3 * std),
                                             trade['closetime'],
                                             0)
    # buy_large_order_interval_time_mean = pd.Series(buy_large_order_interval_time, index=trade.index).rolling(
    #     rolling).apply(intervel_time_mean, engine='numba', raw=True)
    buy_large_order_interval_time_std = np.sqrt(pd.Series(buy_large_order_interval_time, index=trade.index).rolling(
        rolling).apply(intervel_time_std, engine='numba', raw=True))

    return buy_large_order_ratio.fillna(0), buy_large_order_interval_time_std.fillna(0)


# todo
def sell_large_order(depth, trade, rolling=120 * 2):
    s = trade['buy_qty'] + trade['sell_qty']
    v = trade['volume']
    sell_order = np.where(s < 0, v, 0)
    mean = pd.Series(sell_order, index=trade.index).rolling(rolling).mean()
    std = pd.Series(sell_order, index=trade.index).rolling(rolling).std()
    large = np.where(sell_order > (mean + 3 * std), 1, 0)

    sell_large_order_ratio = np.sqrt((pd.Series(large, index=trade.index).rolling(rolling).sum()) / rolling)
    # sell_large_order_trend = (pd.Series(large, index=trade.index).rolling(rolling).sum()) / (
    #         pd.Series(large, index=trade.index).rolling(rolling * 3).sum() / (rolling * 3))

    sell_large_order_interval_time = np.where(sell_order > (mean + 3 * std),
                                              trade['closetime'],
                                              0)
    # sell_large_order_interval_time_mean = pd.Series(sell_large_order_interval_time, index=trade.index).rolling(
    #     rolling).apply(intervel_time_mean, engine='numba', raw=True)
    sell_large_order_interval_time_std = np.sqrt(pd.Series(sell_large_order_interval_time, index=trade.index).rolling(
        rolling).apply(intervel_time_std, engine='numba', raw=True))

    return sell_large_order_ratio.fillna(0), sell_large_order_interval_time_std.fillna(0)


def buy_order_aggressiveness_level1(depth, trade, rolling, pricetick):
    v = trade['buy_qty'] + trade['sell_qty']
    a = trade['amount'].ffill()
    vo = trade['volume'].ffill()
    p = trade['price']
    biding = depth['total_bid_amount']

    avg = round(a / vo, pricetick)
    # 买家激进程度
    # avg[v < 0] = 0
    mid = (depth['traded_ask_price1'] + depth['traded_bid_price1']) / 2
    buy_price = np.where((avg > depth['traded_ask_price1']) & (abs(vo) > depth['total_ask_volume1']) & (v > 0),
                         avg, 0)
    amount = np.where((avg > depth['traded_ask_price1']) & (abs(vo) > depth['total_ask_volume1']) & (v > 0),
                      a, 0)
    buy_agg_amount_ratio = np.sqrt(
        pd.Series(amount, index=trade.index).rolling(rolling).mean() / pd.Series(amount, index=trade.index).rolling(
            rolling).std())

    buy_agg_price_max_bias = pd.Series(buy_price, index=depth.index).rolling(rolling).max() / mid
    buy_agg_price_max_bias_sign_mask = np.sign(buy_agg_price_max_bias)
    buy_agg_price_max_bias_value_mask = abs(buy_agg_price_max_bias)
    buy_agg_price_max_bias = buy_agg_price_max_bias_sign_mask * np.sqrt(buy_agg_price_max_bias_value_mask)

    buy_agg_price_interval_time = np.where(
        (avg > depth['traded_ask_price1']) & (abs(vo) > depth['total_ask_volume1']),
        depth['closetime'], 0)
    buy_agg_price_interval_time_mean = (
        pd.Series(buy_agg_price_interval_time, index=depth.index).rolling(rolling).apply(
            intervel_time_mean, engine='numba', raw=True))
    buy_agg_price_interval_time_std = (
        pd.Series(buy_agg_price_interval_time, index=depth.index).rolling(rolling).apply(
            intervel_time_std, engine='numba', raw=True))
    buy_agg_price_interval_time_mean_std = np.sqrt(
        pd.Series(buy_agg_price_interval_time_mean / buy_agg_price_interval_time_std, index=depth.index))

    # buy_agg_price_count = np.where(
    #     (avg > depth['traded_ask_price1']) & (abs(vo) > depth['traded_ask_volume1']),
    #     1, 0)
    # buy_agg_price_trend = pd.Series(buy_agg_price_count, index=trade.index).rolling(rolling).sum() / (
    #         pd.Series(buy_agg_price_count,
    #                   index=trade.index).rolling(
    #             rolling * 3).sum() / (rolling * 3))

    return buy_agg_amount_ratio, buy_agg_price_max_bias, buy_agg_price_interval_time_mean_std


def buy_order_aggressiveness_level2(depth, trade, rolling=120, pricetick=7):
    v = trade['buy_qty'] + trade['sell_qty']
    a = trade['amount'].ffill()
    vo = trade['volume'].ffill()
    biding = depth['total_bid_amount']
    p = trade['price']
    avg = round(a / vo, pricetick)
    # 买家激进程度
    # avg[v < 0] = 0
    mid = (depth['traded_ask_price1'] + depth['traded_bid_price1']) / 2
    buy_price = np.where((avg > depth['traded_ask_price1']) & (abs(vo) < depth['total_ask_volume1']) & (v > 0),
                         avg, 0)
    amount = np.where((avg > depth['traded_ask_price1']) & (abs(vo) < depth['total_ask_volume1']) & (v > 0),
                      a, 0)
    buy_agg_amount_ratio = np.sqrt(
        pd.Series(amount, index=trade.index).rolling(rolling).mean() / pd.Series(amount, index=trade.index).rolling(
            rolling).std())

    buy_agg_price_max_bias = pd.Series(buy_price, index=depth.index).rolling(rolling).max() / mid
    buy_agg_price_max_bias_sign_mask = np.sign(buy_agg_price_max_bias)
    buy_agg_price_max_bias_value_mask = abs(buy_agg_price_max_bias)
    buy_agg_price_max_bias = buy_agg_price_max_bias_sign_mask * np.sqrt(buy_agg_price_max_bias_value_mask)

    buy_agg_price_interval_time = np.where(
        (avg > depth['traded_ask_price1']) & (abs(vo) < depth['total_ask_volume1']),
        depth['closetime'], 0)
    buy_agg_price_interval_time_mean = (
        pd.Series(buy_agg_price_interval_time, index=depth.index).rolling(rolling).apply(
            intervel_time_mean, engine='numba', raw=True))
    buy_agg_price_interval_time_std = (
        pd.Series(buy_agg_price_interval_time, index=depth.index).rolling(rolling).apply(
            intervel_time_std, engine='numba', raw=True))
    buy_agg_price_interval_time_mean_std = np.sqrt(
        pd.Series(buy_agg_price_interval_time_mean / buy_agg_price_interval_time_std, index=depth.index))

    # buy_agg_price_count = np.where(
    #     (avg > depth['traded_ask_price1']) & (abs(vo) < depth['traded_ask_volume1']),
    #     1, 0)
    # buy_agg_price_trend = pd.Series(buy_agg_price_count, index=trade.index).rolling(rolling).sum() / (
    #         pd.Series(buy_agg_price_count,
    #                   index=trade.index).rolling(
    #             rolling * 3).sum() / (rolling * 3))
    return buy_agg_amount_ratio, buy_agg_price_max_bias, buy_agg_price_interval_time_mean_std


def sell_order_aggressiveness_level1(depth, trade, rolling, pricetick):
    v = trade['buy_qty'] + trade['sell_qty']
    a = trade['amount'].ffill()
    vo = trade['volume'].ffill()
    p = trade['price']
    asking = depth['total_ask_amount']
    avg = round(a / vo, pricetick)
    # 卖家激进程度
    # avg[v > 0] = 0
    mid = (depth['traded_ask_price1'] + depth['traded_bid_price1']) / 2
    sell_price = np.where((avg < depth['traded_bid_price1']) & (abs(vo) > depth['total_bid_volume1']) & (v < 0),
                          -avg, 0)
    amount = np.where((avg < depth['traded_bid_price1']) & (abs(vo) > depth['total_bid_volume1']) & (v < 0),
                      -a, 0)
    sell_agg_amount_ratio_ = (
            pd.Series(amount, index=trade.index).rolling(rolling).mean() / pd.Series(amount, index=trade.index).rolling(
        rolling).std())
    sell_agg_amount_ratio = np.sign(sell_agg_amount_ratio_) * np.sqrt(abs(sell_agg_amount_ratio_))
    sell_agg_price_min_bias_ = pd.Series(sell_price, index=depth.index).rolling(rolling).min() / mid
    sell_agg_price_min_bias = np.sign(sell_agg_price_min_bias_) * np.sqrt(abs(sell_agg_price_min_bias_))

    sell_agg_price_interval_time = np.where(
        (avg < depth['traded_bid_price1']) & (abs(vo) > depth['total_bid_volume1']),
        depth['closetime'], 0)
    sell_agg_price_interval_time_mean = (
        pd.Series(sell_agg_price_interval_time, index=depth.index).rolling(rolling).apply(
            intervel_time_mean, engine='numba', raw=True))
    sell_agg_price_interval_time_std = (
        pd.Series(sell_agg_price_interval_time, index=depth.index).rolling(rolling).apply(
            intervel_time_std, engine='numba', raw=True))
    sell_agg_price_interval_time_mean_std = np.sqrt(
        pd.Series(sell_agg_price_interval_time_mean / sell_agg_price_interval_time_std, index=depth.index)) * -1

    # sell_agg_price_count = np.where(
    #     (avg < depth['traded_bid_price1']) & (abs(vo) > depth['traded_bid_volume1']),
    #     1, 0)
    # sell_agg_price_trend = pd.Series(sell_agg_price_count, index=trade.index).rolling(rolling).sum() / (pd.Series(
    #     sell_agg_price_count,
    #     index=trade.index).rolling(
    #     rolling * 3).sum() / (rolling * 3))

    return sell_agg_amount_ratio, sell_agg_price_min_bias, sell_agg_price_interval_time_mean_std


def sell_order_aggressiveness_level2(depth, trade, rolling=120, pricetick=7):
    v = trade['buy_qty'] + trade['sell_qty']
    a = trade['amount'].ffill().copy()
    vo = trade['volume'].ffill().copy()
    p = trade['price']
    asking = depth['total_ask_amount']
    avg = round(a / vo, pricetick)
    # 卖家激进程度
    # avg[v > 0] = 0
    mid = (depth['traded_ask_price1'].ffill() + depth['traded_bid_price1'].ffill()) / 2
    sell_price = np.where((avg < depth['traded_bid_price1']) & (abs(vo) < depth['total_bid_volume1']) & (v < 0),
                          -avg, 0)
    amount = np.where((avg < depth['traded_bid_price1']) & (abs(vo) < depth['total_bid_volume1']) & (v < 0),
                      -a, 0)
    sell_agg_amount_ratio_ = (
            pd.Series(amount, index=trade.index).rolling(rolling).mean() / pd.Series(amount, index=trade.index).rolling(
        rolling).std())
    sell_agg_amount_ratio = np.sign(sell_agg_amount_ratio_) * np.sqrt(abs(sell_agg_amount_ratio_))
    sell_agg_price_min_bias_ = pd.Series(sell_price, index=depth.index).rolling(rolling).min() / mid
    sell_agg_price_min_bias = np.sign(sell_agg_price_min_bias_) * np.sqrt(abs(sell_agg_price_min_bias_))

    sell_agg_price_interval_time = np.where(
        (avg < depth['traded_bid_price1'].ffill()) & (abs(vo) < depth['total_bid_volume1'].ffill()),
        depth['closetime'], 0)
    sell_agg_price_interval_time_mean = (
        pd.Series(sell_agg_price_interval_time, index=depth.index).rolling(rolling).apply(
            intervel_time_mean, engine='numba', raw=True))
    sell_agg_price_interval_time_std = (
        pd.Series(sell_agg_price_interval_time, index=depth.index).rolling(rolling).apply(
            intervel_time_std, engine='numba', raw=True))
    sell_agg_price_interval_time_mean_std = np.sqrt(
        pd.Series(sell_agg_price_interval_time_mean / sell_agg_price_interval_time_std, index=depth.index)) * -1

    # sell_agg_price_count = np.where(
    #     (avg < depth['traded_bid_price1']) & (abs(vo) < depth['traded_bid_volume1']),
    #     1, 0)
    # sell_agg_price_trend = pd.Series(sell_agg_price_count, index=trade.index).rolling(rolling).sum() / (pd.Series(
    #     sell_agg_price_count,
    #     index=trade.index).rolling(
    #     rolling * 3).sum() / (rolling * 3))

    return sell_agg_amount_ratio, sell_agg_price_min_bias, sell_agg_price_interval_time_mean_std


def QUA(depth, trade, rolling=120):
    single_trade_amount = (trade['price'].ffill()) * (abs(trade['volume'].ffill()))
    QUA = (single_trade_amount.rolling(rolling).quantile(0.2) - single_trade_amount.rolling(rolling).min()) / (
            single_trade_amount.rolling(rolling).max() - single_trade_amount.rolling(rolling).min())
    sign_mask = np.sign(QUA)
    value_mask = abs(QUA)
    return sign_mask * np.sqrt(value_mask)


def flowInRatio(depth, trade, rolling=120, pricetick=7):
    a = trade['amount'].ffill()
    p = trade['price'].ffill()
    v = trade['volume'].ffill()
    avg = round(a.rolling(rolling).sum() / v.rolling(rolling).sum(), pricetick)
    flowInRatio = ((v.rolling(rolling).sum()) * avg * (
            (p - p.shift(rolling)) / abs(p - p.shift(rolling)))) / a.rolling(rolling).sum()
    # flowInRatio2 = (trade['open_interest']-trade['open_interest'].shift(1))*trade['price']*((trade['price']-trade['price'].shift(1))/abs(trade['price']-trade['price'].shift(1)))
    sign_mask = np.sign(flowInRatio)
    value_mask = abs(flowInRatio)
    return sign_mask * np.sqrt(value_mask)


def large_order_qunatile(depth, trade, rolling=120 * 10):
    '''
    大单买入卖出因子
    '''
    s = trade['buy_qty'] + trade['sell_qty']
    a = trade['amount'].ffill()
    buy = np.where(s > 0, a, 0)
    sell = np.where(s < 0, a, 0)
    large_buy = np.where(
        pd.Series(buy, index=trade.index) > pd.Series(buy, index=trade.index).rolling(rolling).quantile(0.9),
        pd.Series(buy, index=trade.index), 0)
    large_sell = np.where(
        pd.Series(sell, index=trade.index) > pd.Series(sell, index=trade.index).rolling(rolling).quantile(0.1),
        pd.Series(sell, index=trade.index), 0)
    large_buy_ratio = pd.Series(large_buy, index=trade.index).rolling(rolling).sum() / (
            pd.Series(buy, index=trade.index).rolling(rolling).sum() + pd.Series(sell, index=trade.index).rolling(
        rolling).sum())
    large_sell_ratio = pd.Series(large_sell, index=trade.index).rolling(rolling).sum() / (
            pd.Series(buy, index=trade.index).rolling(rolling).sum() + pd.Series(sell, index=trade.index).rolling(
        rolling).sum())
    return large_sell_ratio.ffill(), large_buy_ratio.ffill()


def game(depth, trade, rolling=120, pricetick=7):
    a = trade['amount'].ffill()
    v = trade['volume'].ffill()
    avg_price = round((a) / (v), pricetick)
    vol_buy = np.where(avg_price > depth['traded_bid_price1'], a, 0)
    vol_sell = np.where(avg_price < depth['traded_ask_price1'], a, 0)
    game = (pd.Series(vol_buy, index=depth.index).rolling(rolling).sum() - pd.Series(vol_sell,
                                                                                     index=depth.index).rolling(
        rolling).sum()) / (pd.Series(vol_buy, index=depth.index).rolling(rolling).sum() + pd.Series(vol_sell,
                                                                                                    index=depth.index).rolling(
        rolling).sum())
    return np.sign(game) * np.sqrt(abs(game))


# 资金流向因子
# def flow_amount(trade, depth, rolling):
#     a = trade['amount'].copy().ffill()
#     p = trade['price'].copy().ffill()
#     flow_amount = (a-a.shift(1))*((p-p.shift(1))/abs(p-p.shift(1)))
#     factor = pd.Series(flow_amount, index=trade.index).rolling(rolling).sum()/(a-a.shift(rolling))
#     return factor
# 批量成交划分
def multi_active_buying(trade, depth, rolling=120):
    # 朴素主动占比因子
    a = trade['amount'].ffill()
    v = trade['volume'].ffill()
    p = trade['price'].ffill()
    active_buying_1 = (a) * (st.t.cdf((p - p.shift(1)) / (p.rolling(rolling).std()), df=3))
    active_buying = pd.Series(active_buying_1, index=trade.index).rolling(rolling).sum() / (a.rolling(rolling).sum())
    std = np.std(safe_log(p / p.shift(rolling)))
    # t分布主动占比因子
    active_buying_2 = a * (st.t.cdf((safe_log(p / p.shift(1))) / std, df=3))
    t_active_buying = pd.Series(active_buying_2, index=trade.index).rolling(rolling).sum() / (a.rolling(rolling).sum())
    # 标准正太分布主动占比因子
    active_buying_3 = (a) * (st.norm.cdf((safe_log(p / p.shift(1))) / std))
    norm_active_buying = pd.Series(active_buying_3, index=trade.index).rolling(rolling).sum() / (
        a.rolling(rolling).sum())
    # 置信正态分布主动占比因子
    active_buying_4 = (a) * (st.norm.cdf((safe_log(p / p.shift(1))) / 0.1 * 1.96))
    confi_norm_active_buying = pd.Series(active_buying_4, index=trade.index).rolling(rolling).sum() / (
        a.rolling(rolling).sum())
    return (active_buying), (t_active_buying), (norm_active_buying), (confi_norm_active_buying)


def multi_active_selling(trade, depth, rolling=120):
    # 朴素主动占比因子
    a = trade['amount'].ffill()
    v = trade['volume'].ffill()
    p = trade['price'].ffill()
    # p = a/v
    active_buying_1 = a * (st.t.cdf((p - p.shift(1)) / (p.rolling(rolling).std()), df=3))
    active_selling_1 = a - active_buying_1
    active_selling = (
                             pd.Series(active_selling_1, index=trade.index).rolling(rolling).sum() / (
                         a.rolling(rolling).sum())) * -1
    std = np.std(safe_log(p / p.shift(rolling)))
    # t分布主动占比因子
    active_buying_2 = (a) * (st.t.cdf((safe_log(p / p.shift(1))) / std, df=3))
    active_selling_2 = (a - active_buying_2)
    t_active_selling = pd.Series(active_selling_2, index=trade.index).rolling(rolling).sum() / (
        a.rolling(rolling).sum()) * -1
    # 标准正太分布主动占比因子
    active_buying_3 = a * (st.norm.cdf((safe_log(p / p.shift(1))) / std))
    active_selling_3 = a - active_buying_3
    norm_active_selling = pd.Series(active_selling_3, index=trade.index).rolling(rolling).sum() / (
        a.rolling(rolling).sum()) * -1
    # 置信正态分布主动占比因子
    active_buying_4 = (a) * (
        st.norm.cdf((safe_log(p / p.shift(1))) / 0.1 * 1.96))
    active_selling_4 = (a) - active_buying_4
    confi_norm_active_selling = pd.Series(active_selling_4, index=trade.index).rolling(rolling).sum() / (
        a.rolling(rolling).sum()) * -1
    return (active_selling), (t_active_selling), (norm_active_selling), (confi_norm_active_selling)


# todo
def regret_factor(depth, trade, rolling):
    a = trade['amount'].ffill().copy()
    v = trade['volume'].ffill().copy()
    p = trade['price'].ffill().copy()
    # s = trade['size'].ffill().copy()
    s = v - v.shift(1)
    avg_price = ((a - a.shift(1)) / (
            v - v.shift(1))).fillna(
        (depth['ask_price1'].shift(1) + depth['bid_price1'].shift(1)) / 2)

    vol_buy = np.where((avg_price > depth['bid_price1'].shift(1)) & (avg_price > p), v - v.shift(1), 0)
    price_buy = np.where((avg_price > depth['bid_price1'].shift(1)) & (avg_price > p), avg_price, 0)
    HCVOL = pd.Series(vol_buy, index=trade.index).rolling(rolling).sum() / v
    HCP = (pd.Series(price_buy, index=trade.index).rolling(rolling).mean() / p) - 1
    vol_sell = np.where((avg_price < depth['ask_price1'].shift(1)) & (avg_price < p),
                        v - v.shift(1), 0)
    price_sell = np.where((avg_price < depth['ask_price1'].shift(1)) & (avg_price < p), avg_price, 0)
    LCVOL = pd.Series(vol_sell, index=trade.index).rolling(rolling).sum() / v
    LCP = (pd.Series(price_sell, index=trade.index).rolling(rolling).mean() / p) - 1

    return HCVOL, HCP, LCVOL, LCP


def large_order_regret_factor(depth, trade, rolling, pricetick=7):
    a = trade['amount'].ffill()
    v = trade['volume'].ffill()
    s = trade['buy_qty'] + trade['sell_qty']
    large_order_threshold = v.rolling(rolling).quantile(0.9)
    avg_price = round(a / v, pricetick)
    mid = (depth['traded_ask_price1'] + depth['traded_bid_price1']) / 2
    x = pd.Series(np.arange(len(trade)), index=trade.index)
    x = x % rolling + 1
    # large order
    # buy
    large_vol_buy = np.where(
        (avg_price > depth['traded_ask_price1']) & (v >
                                                    large_order_threshold),
        v, 0)
    large_price_buy = np.where(
        (avg_price > depth['traded_ask_price1']) & (v >
                                                    large_order_threshold), avg_price, 0)
    large_HCVOL = np.sqrt(
        pd.Series(large_vol_buy, index=trade.index).rolling(rolling).sum() / (v.rolling(rolling).sum()))
    large_HCP_ = (pd.Series(large_price_buy, index=trade.index).rolling(rolling).mean() / mid)
    large_HCP = np.sign(large_HCP_) * np.sqrt(abs(large_HCP_))

    large_HCP_max_ = (pd.Series(large_price_buy, index=trade.index).rolling(rolling).max() / mid)
    large_HCP_max = np.sign(large_HCP_max_) * np.sqrt(abs(large_HCP_max_))
    large_HCP_max_beta_ = large_HCP_max_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    large_HCP_max_beta = np.sqrt(abs(large_HCP_max_beta_)) * np.sign(large_HCP_max_beta_)
    # 时序特征
    large_price_vol_buy_time = np.where(
        (avg_price > depth['traded_ask_price1']) & (v >
                                                    large_order_threshold), depth['closetime'], 0)
    large_HC_mean = np.sqrt(
        pd.Series(large_price_vol_buy_time, index=depth.index).rolling(rolling).apply(intervel_time_mean,
                                                                                      engine='numba',
                                                                                      raw=True))
    large_HC_std = np.sqrt(pd.Series(large_price_vol_buy_time, index=depth.index).rolling(rolling).apply(
        intervel_time_std, engine='numba', raw=True))

    # sell
    large_vol_sell = np.where(
        (avg_price < depth['traded_bid_price1']) & (v >
                                                    large_order_threshold),
        -v, 0)
    large_price_sell = np.where(
        (avg_price < depth['traded_bid_price1']) & (v >
                                                    large_order_threshold), -avg_price, 0)
    large_LCVOL_ = pd.Series(large_vol_sell, index=trade.index).rolling(rolling).sum() / (v.rolling(rolling).sum())
    large_LCVOL = np.sign(large_LCVOL_) * np.sqrt(abs(large_LCVOL_))
    large_LCP_ = (pd.Series(large_price_sell, index=trade.index).rolling(rolling).mean() / mid)
    large_LCP = np.sign(large_LCP_) * np.sqrt(abs(large_LCP_))

    large_LCP_min_ = (pd.Series(large_price_sell, index=trade.index).rolling(rolling).min() / mid)
    large_LCP_min = np.sign(large_LCP_min_) * np.sqrt(abs(large_LCP_min_))
    large_LCP_min_beta_ = large_LCP_min_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    large_LCP_min_beta = np.sqrt(abs(large_LCP_min_beta_)) * np.sign(large_LCP_min_beta_)
    # 时序特征
    large_price_vol_sell_time = np.where(
        (avg_price < depth['traded_bid_price1']) & (v >
                                                    large_order_threshold),
        depth['closetime'], 0)
    large_LC_mean = np.sqrt(pd.Series(large_price_vol_sell_time, index=depth.index).rolling(rolling).apply(
        intervel_time_mean, engine='numba', raw=True)) * -1
    large_LC_std = np.sqrt(pd.Series(large_price_vol_sell_time, index=depth.index).rolling(rolling).apply(
        intervel_time_std, engine='numba', raw=True)) * -1
    # large_price_sell_1 = np.where(
    #     (avg_price < depth['traded_bid_price1']) & (v >
    #                                                 large_order_threshold), 1, 0)
    # large_LC_trend = np.sqrt(pd.Series(large_price_sell_1, index=depth.index).rolling(rolling).sum() / (pd.Series(
    #     large_price_sell_1, index=depth.index).rolling(rolling * 3).sum() / (rolling * 3))) * -1

    return (large_HCVOL,
            large_HCP,
            large_HCP_max,
            large_HC_mean,
            large_HC_std,
            large_LCVOL,
            large_LCP,
            large_LCP_min,
            large_LC_mean,
            large_LC_std,
            large_HCP_max_beta,
            large_LCP_min_beta,
            )


def small_order_regret_factor(depth, trade, rolling, pricetick=7):
    a = trade['amount'].ffill().copy()
    v = trade['volume'].ffill().copy()
    # s = trade['buy_qty'] + trade['sell_qty']
    small_order_threshold = abs(v).rolling(rolling).quantile(0.1)
    avg_price = round(a / v, pricetick)
    mid = (depth['traded_ask_price1'] + depth['traded_bid_price1']) / 2
    x = pd.Series(np.arange(len(trade)), index=trade.index)
    x = x % rolling + 1
    # small order
    # buy
    small_vol_buy = np.where(
        (avg_price > depth['traded_ask_price1']) & (v <
                                                    small_order_threshold),
        v, 0)
    small_price_buy = np.where(
        (avg_price > depth['traded_ask_price1']) & (v <
                                                    small_order_threshold), avg_price, 0)
    small_HCVOL = np.sqrt(
        pd.Series(small_vol_buy, index=trade.index).rolling(rolling).sum() / (v.rolling(rolling).sum()))
    small_HCP_ = (pd.Series(small_price_buy, index=trade.index).rolling(rolling).mean() / mid)
    small_HCP = np.sign(small_HCP_) * np.sqrt(abs(small_HCP_))

    small_HCP_max_ = (pd.Series(small_price_buy, index=trade.index).rolling(rolling).max() / mid)
    small_HCP_max = np.sign(small_HCP_max_) * np.sqrt(abs(small_HCP_max_))
    small_HCP_max_beta_ = small_HCP_max_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    small_HCP_max_beta = np.sqrt(abs(small_HCP_max_beta_)) * np.sign(small_HCP_max_beta_)
    # 时序特征
    small_price_vol_buy_time = np.where(
        (avg_price > depth['traded_ask_price1']) & (v <
                                                    small_order_threshold),
        depth['closetime'], 0)
    small_HC_mean = np.sqrt(pd.Series(small_price_vol_buy_time, index=depth.index).rolling(rolling).apply(
        intervel_time_mean, engine='numba', raw=True))
    small_HC_std = np.sqrt(pd.Series(small_price_vol_buy_time, index=depth.index).rolling(rolling).apply(
        intervel_time_std, engine='numba', raw=True))

    # sell
    small_vol_sell = np.where(
        (avg_price < depth['traded_bid_price1']) & (v <
                                                    small_order_threshold),
        -v, 0)
    small_price_sell = np.where(
        (avg_price < depth['traded_bid_price1']) & (v <
                                                    small_order_threshold), -avg_price, 0)
    small_LCVOL_ = (pd.Series(small_vol_sell, index=trade.index).rolling(rolling).sum() / (v.rolling(rolling).sum()))
    small_LCVOL = np.sign(small_LCVOL_) * np.sqrt(abs(small_LCVOL_))
    small_LCP_ = (pd.Series(small_price_sell, index=trade.index).rolling(rolling).mean() / mid)
    small_LCP = np.sign(small_LCP_) * np.sqrt(abs(small_LCP_))
    small_LCP_min_ = (pd.Series(small_price_sell, index=trade.index).rolling(rolling).min() / mid)
    small_LCP_min = np.sign(small_LCP_min_) * np.sqrt(abs(small_LCP_min_))
    small_LCP_min_beta_ = small_LCP_min_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    small_LCP_min_beta = np.sqrt(abs(small_LCP_min_beta_)) * np.sign(small_LCP_min_beta_)

    # 时序特征
    small_price_vol_sell_time = np.where(
        (avg_price < depth['traded_bid_price1']) & (v <
                                                    small_order_threshold),
        depth['closetime'], 0)
    small_LC_mean = np.sqrt(pd.Series(small_price_vol_sell_time, index=depth.index).rolling(rolling).apply(
        intervel_time_mean, engine='numba', raw=True)) * -1
    small_LC_std = np.sqrt(pd.Series(small_price_vol_sell_time, index=depth.index).rolling(rolling).apply(
        intervel_time_std, engine='numba', raw=True)) * -1

    return (small_HCVOL,
            small_HCP,
            small_HCP_max,
            small_HC_mean,
            small_HC_std,
            small_LCVOL,
            small_LCP,
            small_LCP_min,
            small_LC_mean,
            small_LC_std,
            small_HCP_max_beta,
            small_LCP_min_beta,
            )


# todo
def high_low_price_volume_tail(depth, trade, rolling=100):
    v = trade['volume'].ffill().copy()
    logvol = safe_log(abs(v))
    # high_rolling_percentile_80 = trade['price'].rolling(rolling).apply(lambda x: pd.Series(x).quantile(0.8))
    high_rolling_percentile_90 = logvol.rolling(rolling).quantile(0.9)
    high_price_volume_90 = np.where(logvol >= high_rolling_percentile_90, 1, 0)
    high_price_volume_tail_90 = pd.Series(high_price_volume_90, index=trade.index).rolling(rolling).sum() / rolling

    low_rolling_percentile_10 = logvol.rolling(rolling).quantile(0.1)
    low_price_volume_10 = np.where(logvol <= low_rolling_percentile_10, 1, 0)
    low_price_volume_tail_10 = pd.Series(low_price_volume_10, index=trade.index).rolling(rolling).sum() / rolling
    return np.sqrt(high_price_volume_tail_90), np.sqrt(low_price_volume_tail_10)


@nb.njit
def first_row(rows):
    return rows[0]


@nb.njit
def last_row(rows):
    return rows[-1]


def hf_trend_str(depth, trade, rolling=120, pricetick=7):
    a = trade['amount'].ffill().copy()
    v = trade['volume'].ffill().copy()

    avg_price = round(a / v, pricetick)

    first_avg_price = pd.Series(avg_price, index=trade.index).rolling(rolling).apply(first_row, engine='numba',
                                                                                     raw=True)
    last_avg_price = pd.Series(avg_price, index=trade.index).rolling(rolling).apply(last_row, engine='numba', raw=True)
    # first_close = p.rolling(rolling).apply(first_row, engine='numba', raw=True)
    # last_close = p.rolling(rolling).apply(last_row, engine='numba', raw=True)

    # diff_close = abs(p - p.shift(rolling))
    diff_avg_price = abs(avg_price - avg_price.shift(rolling))
    # ht_trend_str_close = (last_close - first_close) / pd.Series(diff_close, index=trade.index).rolling(rolling).sum()
    ht_trend_str_avg = (last_avg_price - first_avg_price) / pd.Series(diff_avg_price, index=trade.index).rolling(
        rolling).sum()

    return ht_trend_str_avg


# todo
def corr_pv(depth, trade, rolling=120):
    '''
    高频量价相关性
    '''
    p = trade['price'].ffill() / trade['price'].ffill().shift(1)
    # corr_pvpm = trade['price'].rolling(rolling).corr(trade['volume']/abs(trade['size']))
    corr_rm = pd.Series(p - 1, index=trade.index).rolling(rolling).corr(abs(trade['size'].ffill()))
    corr_rv = pd.Series(p - 1, index=trade.index).rolling(rolling).corr(trade['volume'].ffill())
    # corr_rvpm = pd.Series(p-1, index=trade.index).rolling(rolling).corr(trade['volume']/abs(trade['size']))

    return corr_rm.fillna(0), corr_rv.fillna(0)


import statsmodels.api as sm


#
def scaling_z_score(ser: pd.Series, rolling: int) -> pd.Series:
    """标准分

    Args:
        ser (pd.Series): 因子值

    Returns:
        pd.Series: 标准化后的因子
    """
    return (ser - ser.rolling(rolling).mean()) / ser.rolling(rolling).std()


def calc_ols(x: pd.Series, y: pd.Series,
             method: str = 'resid') -> pd.Series:
    result = sm.OLS(y, sm.add_constant(np.nan_to_num(x))).fit()

    return getattr(result, method)


# @nb.njit(nopython=True)
def calc_ols_numba(x, y, method='resid'):
    result = sm.OLS(y.fillna(0), sm.add_constant(np.nan_to_num(x))).fit()

    return getattr(result, method)


def CPV_corr(depth, trade, rolling=120):
    p = trade['price'].ffill()
    s = trade['volume']
    pv_corr_ = pd.Series(p, index=trade.index).rolling(rolling).corr(s)
    # 平均数因子
    pv_corr_avg = pd.Series(pv_corr_, index=trade.index).rolling(rolling).mean()
    # 波 动性因子
    pv_corr_std = pd.Series(pv_corr_, index=trade.index).rolling(rolling).std()
    # pv_beta_trend = pd.Series(pv_corr_, index=trade.index).rolling(rolling).apply(
    #     lambda x: calc_ols_numba(np.arange(1, len(x) + 1), x, 'params')[1])
    pv_corr = scaling_z_score(pv_corr_avg, rolling) + scaling_z_score(pv_corr_std, rolling)
    # CPV = scaling_z_score(pv_corr) + scaling_z_score(pv_beta_trend)

    return pv_corr


def MTS(depth, trade, rolling=60):
    p = trade['price'].ffill().copy()
    a = trade['amount'].ffill().copy()
    s = trade['buy_qty'] + trade['sell_qty']
    signal_amount = p * abs(s)
    MTS = pd.Series(signal_amount, index=trade.index).rolling(rolling).corr(a.rolling(rolling).sum())

    return MTS.fillna(0)


def MTE(depth, trade, rolling=60):
    p = trade['price'].ffill().copy()
    s = trade['buy_qty'] + trade['sell_qty']
    signal_amount = p * abs(s)
    MTE = pd.Series(signal_amount, index=trade.index).rolling(rolling).corr(p)
    return MTE.fillna(0)


# def reverse_factor(depth, trade, rolling=120):
#     p = trade['price'].ffill().copy()
#     vo = trade['volume'].ffill().copy()
#     size = vo-vo.shift(1)
#     log = safe_log(p/p.shift(1))
#     size_criterion = pd.Series(size, index=trade.index).rolling(rolling).quantile

#
def reverse_factor_w(depth, trade, rolling=60):
    p = trade['price'].ffill()
    # s = trade['buy_qty'] + trade['sell_qty']
    signal_amount = trade['amount'].ffill()
    log_return = safe_log(p / p.shift(1))
    high = pd.Series(signal_amount, index=trade.index).rolling(rolling).quantile(0.9)
    low = pd.Series(signal_amount, index=trade.index).rolling(rolling).quantile(0.1)
    M_high = np.where(signal_amount >= high, log_return, 0)
    M_low = np.where(signal_amount <= low, log_return, 0)
    reverse_factor_w = pd.Series(M_high, index=trade.index).rolling(rolling).sum() - pd.Series(M_low,
                                                                                               index=trade.index).rolling(
        rolling).sum()
    sign_mask = np.sign(reverse_factor_w)
    value_mask = abs(reverse_factor_w)
    return sign_mask * np.sqrt(value_mask)


def SR_factor(depth, trade, rolling=60):
    p = trade['price'].ffill()
    s = trade['volume'].ffill()
    signal_amount = trade['amount'].ffill()
    log_return = safe_log(p / p.shift(1))
    criterion = pd.Series(signal_amount, index=trade.index).rolling(rolling).quantile(0.9)
    filter = np.where(signal_amount >= criterion, log_return, 0)
    SR = pd.Series(filter, index=trade.index).rolling(rolling).sum()
    filter_interval_time = np.where(signal_amount >= criterion, trade['closetime'],
                                    0)
    SR_interval_time_mean = pd.Series(filter_interval_time, index=trade.index).rolling(rolling).apply(
        intervel_time_mean, engine='numba', raw=True)
    SR_interval_time_std = pd.Series(filter_interval_time, index=trade.index).rolling(rolling).apply(intervel_time_std,
                                                                                                     engine='numba',
                                                                                                     raw=True)
    filter_interval_count = np.where(signal_amount >= criterion, 1,
                                     0)
    SR_interval_time_trend = pd.Series(filter_interval_count, index=trade.index).rolling(rolling).sum() / (pd.Series(
        filter_interval_count, index=trade.index).rolling(rolling * 3).sum() / (rolling * 3))
    return np.sqrt(SR), safe_log(SR_interval_time_mean), np.sqrt(SR_interval_time_std), np.sqrt(SR_interval_time_trend)


def q_money(depth, trade, rolling=120, pricetick=7):
    a = trade['amount'].ffill()
    v = trade['volume'].ffill()
    p = round(a / v, pricetick)

    high = p.rolling(rolling).max()
    low = p.rolling(rolling).min()
    log_return = abs(safe_log(high / low))

    ss = log_return / np.log(v.rolling(rolling).sum())
    criterion = pd.Series(ss, index=trade.index).rolling(rolling).quantile(0.9)
    q_money_price = np.where(ss >= criterion, p, 0)
    q_money_volume = np.where(ss >= criterion, v, 0)
    q_vvwap = pd.Series((q_money_price * q_money_volume), index=trade.index).rolling(rolling).sum() / pd.Series(
        q_money_volume, index=trade.index).rolling(rolling).sum()
    vwap = (a.rolling(rolling).sum()) / (v.rolling(rolling).sum())
    factor = q_vvwap / vwap

    return np.sqrt(factor)


@nb.njit
def vol_entropy(row):
    # vol_array = np.asarray(df)
    # print(vol_array)
    # print(row)
    hist, bin_edges = np.histogram(row, 10)
    probs = hist / row.size
    probs[probs == 0] = 1
    vol_entropy = -np.sum(probs * np.log(probs))

    return vol_entropy


# def volume_entropy(depth, trade, rolling=120):
#     vol = (trade['volume']).ffill()
#     vol_rolling_entropy = vol.rolling(rolling).apply(vol_entropy, raw=True, engine='numba')
#     vol_entropy_std = pd.Series(vol_rolling_entropy, index=trade.index).rolling(rolling).std()
#     # print(vol_entropy_std)
#     return np.sqrt(abs(vol_entropy_std)) * np.sign(vol_entropy_std)
@nb.njit
def calculate_entropy(data, num_bins=10):
    hist, _ = np.histogram(data, bins=num_bins)
    probs = hist / data.size
    probs = probs[probs > 0]  # Remove zero probabilities
    return -np.sum(probs * np.log(probs))


@nb.njit
def rolling_entropy(data, window):
    result = np.empty(len(data))
    for i in range(len(data)):
        if i < window:
            result[i] = np.nan
        else:
            result[i] = calculate_entropy(data[i - window + 1:i + 1])
    return result


def volume_entropy(depth, trade, rolling=120):
    # Data preprocessing
    volume = trade['volume']

    # Calculate rolling entropy
    entropy_values = rolling_entropy(volume.values, rolling)
    entropy_series = pd.Series(entropy_values, index=trade.index)

    # Calculate the standard deviation of entropy
    entropy_std = entropy_series.rolling(rolling).std()

    # Apply exponential weighted moving average for smoothing
    smoothed_entropy_std = entropy_std.ewm(span=rolling, adjust=False).mean()

    # Normalize the factor
    normalized_factor = (smoothed_entropy_std - smoothed_entropy_std.rolling(
        rolling).mean()) / smoothed_entropy_std.rolling(rolling).std()

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(normalized_factor, [2.5, 97.5])
    winsorized_factor = np.clip(normalized_factor, lower, upper)

    # Final transformation
    final_factor = np.sign(winsorized_factor) * np.sqrt(np.abs(winsorized_factor))

    return final_factor


@nb.njit
def vol_bootstrap(row):
    # vol_array = np.asarray(df)
    # print(vol_array)
    # print(row)
    data = (row,)
    vol_bootstrap = bootstrap(data, np.max, confidence_level=0.95,
                              random_state=1, method='percentile')
    vol_max = vol_bootstrap.confidence_interval.high

    return vol_max


def volume_max_distrbution(depth, trade, rolling=120):
    vol = trade['buy_qty'] + trade['sell_qty']
    vol_max = vol.swifter.rolling(rolling).apply(vol_bootstrap)
    vol_max_mean = pd.Series(vol_max, index=trade.index).rolling(rolling).mean()
    vol_max_std = pd.Series(vol_max, index=trade.index).rolling(rolling).std()

    return vol_max_mean, vol_max_std


@nb.njit
def variation_coefficient(x):
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan


def amount_variation_coefficient(depth, trade, rolling=120):
    signal_amount = (trade['amount']).ffill().copy()
    amount_variation_coefficient = pd.Series(signal_amount, index=trade.index).rolling(rolling).apply(
        variation_coefficient, engine='numba', raw=True)
    sign_mask = np.sign(amount_variation_coefficient)
    value_mask = abs(amount_variation_coefficient)
    return sign_mask * np.sqrt(value_mask)


def up_down_realized_volatility(depth, trade, rolling=120):
    price = (trade['price']).ffill().copy()
    log_return = safe_log(price / price.shift(1))
    realized_vol = pd.Series(log_return, index=trade.index).rolling(rolling).apply(realized_volatility, engine='numba',
                                                                                   raw=True)
    up_realized_volatility = np.where(realized_vol * log_return > 0, realized_vol, 0)
    down_realized_volatility = np.where(realized_vol * log_return < 0, realized_vol, 0)
    up_realized_volatility_ratio = pd.Series(up_realized_volatility, index=trade.index).rolling(
        rolling).sum() / pd.Series(realized_vol, index=trade.index).rolling(rolling).sum()
    down_realized_volatility_ratio = pd.Series(down_realized_volatility, index=trade.index).rolling(
        rolling).sum() / pd.Series(realized_vol, index=trade.index).rolling(rolling).sum()
    # up_realized_volatility_variation_coefficient = pd.Series(up_realized_volatility, index=trade.index).rolling(
    #     rolling).apply(variation_coefficient, engine='numba', raw=True)
    # down_realized_volatility_variation_coefficient = pd.Series(down_realized_volatility, index=trade.index).rolling(
    #     rolling).apply(
    #     variation_coefficient, engine='numba', raw=True)
    # up_sign_mask = np.sign(up_realized_volatility_ratio)
    # up_value_mask = abs(up_realized_volatility_ratio)
    # down_sign_mask = np.sign(down_realized_volatility_ratio)
    # down_value_mask = abs(down_realized_volatility_ratio)
    return np.sqrt(up_realized_volatility_ratio), np.sqrt(down_realized_volatility_ratio)


def return_realized_absvar(depth, trade, rolling=120):
    price = (trade['price']).ffill().copy()
    log_return = safe_log(price / price.shift(1))
    realized_absvar_ = pd.Series(log_return, index=trade.index).rolling(rolling).apply(realized_absvar, engine='numba',
                                                                                       raw=True)
    sign_mask = np.sign(realized_absvar_)
    value_mask = abs(realized_absvar_)
    return sign_mask * np.sqrt(value_mask)


# todo
def return_realized_skew(depth, trade, rolling=120):
    price = (trade['price']).ffill().copy()
    log_return = safe_log(price / price.shift(1))
    realized_skew_ = pd.Series(log_return, index=trade.index).rolling(rolling).apply(realized_skew, engine='numba',
                                                                                     raw=True)
    return np.sqrt(abs(realized_skew_)) * np.sign(realized_skew_)


# todo
def return_realized_kurtosis(depth, trade, rolling=120):
    price = (trade['price']).ffill().copy()
    log_return = safe_log(price / price.shift(1))
    realized_kurtosis_ = pd.Series(log_return, index=trade.index).rolling(rolling).apply(realized_kurtosis,
                                                                                         engine='numba', raw=True)
    sign_mask = np.sign(realized_kurtosis_)
    value_mask = abs(realized_kurtosis_)
    return sign_mask * safe_log(value_mask)


# todo
def return_realized_quarticity(depth, trade, rolling=120):
    price = (trade['price']).ffill().copy()
    log_return = safe_log(price / price.shift(1))
    realized_quarticity_ = pd.Series(log_return, index=trade.index).rolling(rolling).apply(realized_quarticity,
                                                                                           engine='numba', raw=True)
    sign_mask = np.sign(realized_quarticity_)
    value_mask = abs(realized_quarticity_)
    return sign_mask * np.sqrt(value_mask)


def durbin_watson_test(x):
    from statsmodels.stats.stattools import durbin_watson
    # x = np.array(x)
    result = durbin_watson(x)
    return result


def DW_test_amount(depth, trade, rolling=120):
    single_amount = trade['amount'].ffill()
    DW_test_amount = pd.Series(single_amount, index=trade.index).rolling(rolling).apply(durbin_watson_test)
    sign_mask = np.sign(DW_test_amount)
    value_mask = abs(DW_test_amount)
    return sign_mask * np.sqrt(value_mask)


# todo
def highStdRtn_mean(depth, trade, rolling=120):
    price = (trade['price']).ffill().copy()
    log_return = (safe_log(price / price.shift(1))).fillna(0)
    realized_vol = pd.Series(log_return, index=trade.index).rolling(rolling).apply(realized_volatility,
                                                                                   engine='numba', raw=True)
    realized_volatility_std = pd.Series(realized_vol, index=trade.index).rolling(rolling).std()
    filter = pd.Series(realized_volatility_std, index=trade.index).rolling(rolling).quantile(0.9)
    realized_volatility_std_filter = np.where(realized_volatility_std >= filter, log_return, 1)
    highStdRtn_mean = pd.Series(realized_volatility_std_filter, index=trade.index).rolling(rolling).mean()
    sign_mask = np.sign(highStdRtn_mean)
    value_mask = abs(highStdRtn_mean)
    return sign_mask * np.sqrt(value_mask)


def get_roll_impact(depth, trade, rolling=120):
    """
    Get Roll Measure (p.282, Roll Model). Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.
    Get Roll Impact. Derivate from Roll Measure which takes into account dollar volume traded.
    """
    close_prices = trade['price'].ffill()
    dollar_volume = trade['amount'].ffill()
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return (2 * np.sqrt(abs(price_diff.rolling(rolling).cov(price_diff_lag)))) / dollar_volume.rolling(rolling).sum()


# Corwin-Schultz algorithm
def _get_beta(depth, trade, rolling=120):
    """
    Get beta estimate from Corwin-Schultz algorithm (p.285, Snippet 19.1).

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) estimation window
    :return: (pd.Series) of beta estimates

    """
    price = trade['price'].ffill().copy()
    high = price.rolling(rolling).max()
    low = price.rolling(rolling).min()
    ret = safe_log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(rolling).mean()
    return beta


def _get_gamma(depth, trade, rolling=120):
    """
    Get gamma estimate from Corwin-Schultz algorithm (p.285, Snippet 19.1).

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :return: (pd.Series) of gamma estimates
    """
    price = trade['price'].ffill().copy()
    high = price.rolling(rolling).max()
    low = price.rolling(rolling).min()
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = safe_log(high_max / low_min) ** 2
    return gamma


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Get alpha from Corwin-Schultz algorithm, (p.285, Snippet 19.1).

    :param beta: (pd.Series) of beta estimates
    :param gamma: (pd.Series) of gamma estimates
    :return: (pd.Series) of alphas
    """
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # Set negative alphas to 0 (see p.727 of paper)
    return alpha


def get_corwin_schultz_estimator(depth, trade, rolling=120):
    """
    Get Corwin-Schultz spread estimator using high-low prices, (p.285, Snippet 19.1).

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) estimation window
    :return: (pd.Series) of Corwin-Schultz spread estimators
    """
    # Note: S<0 iif alpha<0
    price = trade['price'].ffill().copy()
    high = price.rolling(rolling).max()
    low = price.rolling(rolling).min()

    beta = _get_beta(depth=depth, trade=trade, rolling=rolling)
    gamma = _get_gamma(depth=depth, trade=trade, rolling=rolling)
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    return spread.fillna(0)


def get_bekker_parkinson_vol(depth, trade, rolling=120):
    """
    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm, (p.286, Snippet 19.2).

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) estimation window
    :return: (pd.Series) of Bekker-Parkinson volatility estimates
    """
    # pylint: disable=invalid-name

    beta = _get_beta(depth=depth, trade=trade, rolling=rolling)
    gamma = _get_gamma(depth=depth, trade=trade, rolling=rolling)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma


# pylint: disable=invalid-name
def get_bar_based_kyle_lambda(depth, trade, rolling=120):
    """
    Get Kyle lambda from bars data, p.286-288.

    :param close: (pd.Series) Close prices
    :param volume: (pd.Series) Bar volume
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Kyle lambdas
    """
    close = trade['price'].ffill().copy()
    vo = trade['volume'].ffill().copy()
    close_diff = close.diff()
    close_diff_sign = np.sign(close_diff)
    close_diff_sign.replace(0, method='pad', inplace=True)  # Replace 0 values with previous
    volume_mult_trade_signs = vo * close_diff_sign  # bt * Vt
    return (close_diff / volume_mult_trade_signs).rolling(rolling).mean()


def get_bar_based_amihud_lambda(depth, trade, rolling=120):
    """
    Get Amihud lambda from bars data, p.288-289.

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Amihud lambda
    """
    close = trade['price'].ffill().copy()
    dollar_volume = trade['amount'].ffill().copy()
    returns_abs = safe_log(close / close.shift(1)).abs()
    return (returns_abs / dollar_volume).rolling(rolling).mean()


def get_bar_based_hasbrouck_lambda(depth, trade, rolling=120):
    """
    Get Hasbrouck lambda from bars data, p.289-290.

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Hasbrouck lambda
    """
    close = trade['price'].ffill().copy()
    dollar_volume = trade['amount'].ffill().copy()
    log_ret = safe_log(close / close.shift(1))
    log_ret_sign = np.sign(log_ret).replace(0, method='pad')

    signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)
    return (log_ret / signed_dollar_volume_sqrt).rolling(rolling).mean()


@nb.njit
def abs_energy(x):
    x = np.asarray(x)
    return np.dot(x, x)


def abs_energy_ask_price(depth, trade, rolling=100):
    return depth['traded_ask_price1'].rolling(rolling).apply(abs_energy, engine='numba', raw=True)


def abs_energy_bid_price(depth, trade, rolling=100):
    return depth['traded_bid_price1'].rolling(rolling).apply(abs_energy, engine='numba', raw=True)


def abs_energy_avg_price(depth, trade, rolling=100, pricetick=7):
    avg_price = round(trade['amount'] / trade['volume'], pricetick)
    return avg_price.rolling(rolling).apply(abs_energy, engine='numba', raw=True)


@nb.njit
def longest_strike_below_mean(x):
    # x = np.asarray(x)  # 将输入的Series转换为numpy数组
    below_mean = x < np.mean(x)
    max_strike = 0
    current_strike = 0

    for i in range(len(below_mean)):
        if below_mean[i]:
            current_strike += 1
            max_strike = max(max_strike, current_strike)
        else:
            current_strike = 0

    return max_strike


@nb.njit
def longest_strike_above_mean(x):
    # x = np.asarray(x)  # 将输入的Series转换为numpy数组
    below_mean = x > np.mean(x)
    max_strike = 0
    current_strike = 0

    for i in range(len(below_mean)):
        if below_mean[i]:
            current_strike += 1
            max_strike = max(max_strike, current_strike)
        else:
            current_strike = 0

    return max_strike


def longest_strike_below_mean_price(depth, trade, rolling=100):
    return trade['price'].ffill().rolling(rolling).apply(longest_strike_below_mean, engine='numba',
                                                         raw=True)


def longest_strike_above_mean_price(depth, trade, rolling=100):
    return trade['price'].ffill().rolling(rolling).apply(longest_strike_above_mean, engine='numba',
                                                         raw=True)


def longest_strike_above_mean_avg_price(depth, trade, rolling=100, pricetick=7):
    avg_price = round(trade['amount'] / trade['volume'], pricetick)
    return avg_price.rolling(rolling).apply(longest_strike_above_mean, engine='numba',
                                            raw=True)


@nb.njit
def count_above_mean(x):
    m = np.mean(x)
    return np.where(x > m)[0].size


@nb.njit
def count_below_mean(x):
    m = np.mean(x)
    return np.where(x < m)[0].size


def count_above_mean_price(depth, trade, rolling=100):
    return trade['price'].ffill().rolling(rolling).apply(count_above_mean, engine='numba', raw=True)


def count_below_mean_price(depth, trade, rolling=100):
    return trade['price'].ffill().rolling(rolling).apply(count_below_mean, engine='numba', raw=True)


def count_above_mean_avg_price(depth, trade, rolling=100, pricetick=7):
    avg_price = round(trade['amount'] / trade['volume'], pricetick)
    return avg_price.rolling(rolling).apply(count_above_mean, engine='numba', raw=True)


# LargeSmallOrder
# largeorder_amtp_bias: the amtp bias between this second and mean of period
# amtp = resample 1s, calculate the percent of large/small trades sum amt of the whole amt
# largeOrder: the trade amt > 80% quantile
# SmallOrder: the trade amt < 50% quantile

@nb.njit
def cal_hhi_diff(x):
    DENOM_EPSILON = 1e-5
    return np.dot(x, x.T) / (x.sum() ** 2 + DENOM_EPSILON)


def largeorder_amtp_bias(depth, trade, rolling=100):
    p = trade['price'].ffill()
    amt = trade['amount'].ffill()
    large_amount_thre = pd.Series(amt, index=trade.index).rolling(rolling).quantile(0.7)
    large_amt = np.where(amt >= large_amount_thre, amt, 0)
    large_amtp = pd.Series(large_amt, index=trade.index).rolling(rolling).sum() / (amt.rolling(rolling).sum())
    meanamtp = pd.Series(large_amtp, index=trade.index).rolling(rolling).mean()
    largeorder_amtp_bias = pd.Series(large_amtp / meanamtp).rolling(rolling).std()

    sign_mask = np.sign(largeorder_amtp_bias)
    value_mask = abs(largeorder_amtp_bias)
    return sign_mask * np.sqrt(value_mask)


def smallorder_amtp_bias(depth, trade, rolling=100):
    p = trade['price'].ffill()
    amt = trade['amount'].ffill()
    small_amount_thre = pd.Series(amt, index=trade.index).rolling(rolling).quantile(0.3)
    small_amt = np.where(amt <= small_amount_thre, amt, 0)
    small_amtp = pd.Series(small_amt, index=trade.index).rolling(rolling).sum() / (amt.rolling(rolling).sum())
    meanamtp = pd.Series(small_amtp, index=trade.index).rolling(rolling).mean()
    smallorder_amtp_bias = pd.Series(small_amtp / meanamtp).rolling(rolling).std()

    sign_mask = np.sign(smallorder_amtp_bias)
    value_mask = abs(smallorder_amtp_bias)
    return sign_mask * np.sqrt(value_mask)


def largeorder_price_bias(depth, trade, rolling=100, pricetick=7):
    p = trade['price'].ffill()
    amt = trade['amount'].ffill()
    avg = round(trade['amount'] / trade['volume'], pricetick)
    large_amount_thre = pd.Series(amt, index=trade.index).rolling(rolling).quantile(0.7)
    large_price = np.where(amt >= large_amount_thre, avg, 0)
    large_pricep = pd.Series(large_price, index=trade.index).rolling(rolling).mean() / p.rolling(rolling).mean()
    large_price_ratio_bias = (large_pricep).rolling(rolling).std()
    sign_mask = np.sign(large_price_ratio_bias)
    value_mask = abs(large_price_ratio_bias)
    return sign_mask * np.sqrt(value_mask)


def smallorder_price_bias(depth, trade, rolling=100, pricetick=7):
    p = trade['price'].ffill()
    amt = trade['amount'].ffill()
    avg = round(trade['amount'] / trade['volume'], pricetick)
    small_amount_thre = pd.Series(amt, index=trade.index).rolling(rolling).quantile(0.3)
    small_price = np.where(amt <= small_amount_thre, avg, 0)
    small_pricep = pd.Series(small_price, index=trade.index).rolling(rolling).mean() / p.rolling(rolling).mean()
    smallorder_price_bias = (small_pricep).rolling(rolling).std()
    sign_mask = np.sign(smallorder_price_bias)
    value_mask = abs(smallorder_price_bias)
    return sign_mask * np.sqrt(value_mask)


@nb.njit
def rolling_dot_func(x):
    return np.dot(x, x.T)


def qtyhhi_bias(depth, trade, rolling=100):
    s = trade['volume']
    rolling_dot = pd.Series(s, index=trade.index).rolling(rolling).apply(rolling_dot_func, engine='numba', raw=True)
    qtyhhi = rolling_dot / s.rolling(rolling).sum() ** 2
    hhi_bias = qtyhhi / pd.Series(qtyhhi, index=trade.index).rolling(rolling).mean()
    sign_mask = np.sign(hhi_bias)
    value_mask = abs(hhi_bias)
    return sign_mask * np.sqrt(value_mask)


@nb.njit
def cal_hhi_diff(x):
    DENOM_EPSILON = 1e-5
    return np.dot(x, x.T) / (x.sum() ** 2 + DENOM_EPSILON)


def qtyhhi_diff(depth, trade, rolling=100):
    s = trade['buy_qty'] + trade['sell_qty']
    v = trade['volume']
    buyer = np.where(s > 0, v, 0)
    seller = np.where(s < 0, v, 0)
    hhi_b = pd.Series(buyer, index=trade.index).rolling(rolling).apply(cal_hhi_diff, engine='numba', raw=True)
    hhi_s = pd.Series(seller, index=trade.index).rolling(rolling).apply(cal_hhi_diff, engine='numba', raw=True)
    hhi_diff = hhi_b / hhi_s
    hhi_diff_mean = pd.Series(hhi_diff, index=trade.index).rolling(rolling).mean()
    return np.sqrt(hhi_diff_mean)


def rollols_resid(y: pd.Series, x: pd.Series, win: int) -> pd.Series:
    beta = y.rolling(win).cov(x) / x.rolling(win).var()
    alpha = y.rolling(win).mean() - beta * x.rolling(win).mean()
    ebisilon = y - (beta * x + alpha)
    return ebisilon


def qtyhhi_residual(depth, trade, rolling=100):
    s = trade['buy_qty'] + trade['sell_qty']
    v = trade['volume']
    buyer = np.where(s > 0, v, 0)
    seller = np.where(s < 0, v, 0)
    hhi_b = pd.Series(buyer, index=trade.index).rolling(rolling).apply(cal_hhi_diff, engine='numba', raw=True)
    hhi_s = pd.Series(seller, index=trade.index).rolling(rolling).apply(cal_hhi_diff, engine='numba', raw=True)
    hhi_resid = rollols_resid(hhi_b, hhi_s, rolling)
    sign_mask = np.sign(hhi_resid)
    value_mask = abs(hhi_resid)
    return sign_mask * np.sqrt(value_mask)


def active_large_amtp_diff(depth, trade, rolling=100):
    p = trade['price'].ffill()
    s = trade['buy_qty'] + trade['sell_qty']
    amt = trade['amount'].ffill()
    large_amount_thre = pd.Series(amt, index=trade.index).rolling(rolling).quantile(0.9)

    large_buyer_amt = np.where((amt >= large_amount_thre) & (s > 0), amt, 0)
    large_seller_amt = np.where((amt >= large_amount_thre) & (s < 0), amt, 0)
    lbp = pd.Series(large_buyer_amt, index=trade.index).rolling(rolling).sum() / (amt.rolling(rolling).sum())
    lsp = pd.Series(large_seller_amt, index=trade.index).rolling(rolling).sum() / (amt.rolling(rolling).sum())
    active_large_amtp = lbp - lsp
    active_large_mt_amtp_diff = pd.Series(active_large_amtp, index=trade.index).rolling(rolling).mean()
    sign_mask = np.sign(active_large_mt_amtp_diff)
    value_mask = abs(active_large_mt_amtp_diff)
    return sign_mask * np.sqrt(value_mask)


def active_large_amount_diff(depth, trade, rolling=100):
    # Fill NaN values
    p = trade['price']
    amt = trade['amount']
    side = trade['buy_qty'] + trade['sell_qty']
    # Calculate buy and sell volumes
    buy_vol = np.where(side > 0, trade['buy_qty'] * p, 0)
    sell_vol = np.where(side < 0, trade['sell_qty'] * p, 0)

    # Define large amount threshold
    large_amount_thre = amt.rolling(rolling).quantile(0.9)

    # Calculate large buyer and seller amounts
    large_buyer_amt = np.where((amt >= large_amount_thre) & (buy_vol > 0), buy_vol, 0)
    large_seller_amt = np.where((amt >= large_amount_thre) & (sell_vol < 0), sell_vol, 0)

    # Calculate proportions of large trades
    total_vol = buy_vol + (sell_vol)
    lbp = pd.Series(large_buyer_amt, index=trade.index).rolling(rolling).sum() / pd.Series(total_vol,
                                                                                           index=trade.index).rolling(
        rolling).sum()
    lsp = pd.Series(large_seller_amt, index=trade.index).rolling(rolling).sum() / pd.Series(total_vol,
                                                                                            index=trade.index).rolling(
        rolling).sum()

    # Calculate difference and apply exponential moving average
    active_large_amt_diff = lbp - lsp
    ema_diff = active_large_amt_diff.ewm(span=rolling, adjust=False).mean()
    # Normalize the factor
    normalized_diff = (ema_diff - ema_diff.rolling(rolling).mean()) / ema_diff.rolling(rolling).std()

    # Apply sigmoid function for scaling
    scaled_diff = 2 / (1 + np.exp(-normalized_diff)) - 1

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(scaled_diff, [2.5, 97.5])
    winsorized_diff = np.clip(scaled_diff, lower, upper)

    return winsorized_diff


def active_small_amount_diff(depth, trade, rolling=100):
    # Fill NaN values
    p = trade['price']
    amt = trade['amount']
    side = trade['buy_qty'] + trade['sell_qty']
    # Calculate buy and sell volumes
    buy_vol = np.where(side > 0, trade['buy_qty'] * p, 0)
    sell_vol = np.where(side < 0, trade['sell_qty'] * p, 0)

    # Define large amount threshold
    small_amount_thre = amt.rolling(rolling).quantile(0.9)

    # Calculate large buyer and seller amounts
    small_buyer_amt = np.where((amt <= small_amount_thre) & (buy_vol > 0), buy_vol, 0)
    small_seller_amt = np.where((amt <= small_amount_thre) & (sell_vol < 0), sell_vol, 0)

    # Calculate proportions of large trades
    total_vol = buy_vol + (sell_vol)
    sbp = pd.Series(small_buyer_amt, index=trade.index).rolling(rolling).sum() / pd.Series(total_vol,
                                                                                           index=trade.index).rolling(
        rolling).sum()
    ssp = pd.Series(small_seller_amt, index=trade.index).rolling(rolling).sum() / pd.Series(total_vol,
                                                                                            index=trade.index).rolling(
        rolling).sum()

    # Calculate difference and apply exponential moving average
    active_small_amt_diff = sbp - ssp
    ema_diff = active_small_amt_diff.ewm(span=rolling, adjust=False).mean()
    # Normalize the factor
    normalized_diff = (ema_diff - ema_diff.rolling(rolling).mean()) / ema_diff.rolling(rolling).std()

    # Apply sigmoid function for scaling
    scaled_diff = 2 / (1 + np.exp(-normalized_diff)) - 1

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(scaled_diff, [2.5, 97.5])
    winsorized_diff = np.clip(scaled_diff, lower, upper)

    return winsorized_diff


def active_small_amtp_diff(depth, trade, rolling=100):
    p = trade['price'].ffill()
    s = trade['buy_qty'] + trade['sell_qty']
    amt = trade['amount'].ffill()
    small_amount_thre = pd.Series(amt, index=trade.index).rolling(rolling).quantile(0.1)

    small_buyer_amt = np.where((amt <= small_amount_thre) & (s > 0), amt, 0)
    small_seller_amt = np.where((amt <= small_amount_thre) & (s < 0), amt, 0)
    sbp = pd.Series(small_buyer_amt, index=trade.index).rolling(rolling).sum() / (amt.rolling(rolling).sum())
    ssp = pd.Series(small_seller_amt, index=trade.index).rolling(rolling).sum() / (amt.rolling(rolling).sum())
    active_small_amtp = sbp - ssp
    active_small_mt_amtp_diff = pd.Series(active_small_amtp, index=trade.index).rolling(rolling).mean()
    sign_mask = np.sign(active_small_mt_amtp_diff)
    value_mask = abs(active_small_mt_amtp_diff)
    return sign_mask * np.sqrt(value_mask)


def active_price_bias(depth, trade, rolling=100):
    # Fill NaN values in price and amount
    p = trade['price']
    side = trade['buy_qty'] + trade['sell_qty']
    # Calculate buyer and seller initiated volumes
    buy_vol = np.where(side > 0, trade['buy_qty'] * p, 0)
    sell_vol = np.where(side < 0, (trade['sell_qty']) * p, 0)

    # Calculate volume-weighted average prices
    vwap_buy = (pd.Series(buy_vol, index=trade.index).rolling(rolling).sum() / trade['buy_qty'].rolling(
        rolling).sum()).fillna(p)
    vwap_sell = (pd.Series(sell_vol, index=trade.index).rolling(rolling).sum() / (trade['sell_qty']).rolling(
        rolling).sum()).fillna(p)

    # Calculate price bias
    price_bias = (vwap_buy - vwap_sell) / ((vwap_buy + vwap_sell) / 2)

    # Apply statistical normalization
    normalized_bias = (price_bias - price_bias.rolling(rolling).mean()) / price_bias.rolling(rolling).std()

    # Apply sigmoid function for better scaling
    scaled_bias = 2 / (1 + np.exp(-normalized_bias)) - 1

    # Apply Winsorization to handle outliers
    lower, upper = np.percentile(scaled_bias, [2.5, 97.5])
    winsorized_bias = np.clip(scaled_bias, lower, upper)

    return winsorized_bias


def momentum_residual(depth, trade, rolling):
    price = trade['price']
    amount = trade['amount']

    logreturn = safe_log(price / price.shift(1))
    flow = amount * logreturn
    cum_flow = flow.rolling(rolling).sum()

    momentum_residual = rollols_resid(logreturn, cum_flow, rolling)
    sign_mask = np.sign(momentum_residual)
    value_mask = abs(momentum_residual)
    return sign_mask * np.sqrt(value_mask)


def reverse_mom(trade, depth, rolling):
    price = trade['price'].ffill()
    volume = trade['volume'].ffill()
    log_returns = safe_log(price / price.shift(1))
    rev_mom_weights = safe_log(1 / volume)

    rev_mom = pd.Series(log_returns * rev_mom_weights, index=trade.index).rolling(rolling).sum()
    return np.sqrt(abs(rev_mom)) * np.sign(rev_mom)


def structural_reversal(trade, depth, rolling):
    price = trade['price'].ffill()
    volume = trade['volume'].ffill()
    log_returns = safe_log(price / price.shift(1))
    rev_mom_weights = 1 / volume
    rev_rev_weights = volume
    rev_rev = pd.Series(log_returns * rev_rev_weights, index=trade.index).rolling(rolling).sum()
    rev_mom = pd.Series(log_returns * rev_mom_weights, index=trade.index).rolling(rolling).sum()
    structural_reversal = rev_rev - rev_mom
    sign_mask = np.sign(structural_reversal)
    value_mask = abs(structural_reversal)
    return sign_mask * np.sqrt(value_mask)


def reverse_mom_structural_reversal_resid(trade, depth, rolling):
    price = trade['price'].ffill()
    volume = trade['volume'].ffill()
    log_returns = safe_log(price / price.shift(1))
    rev_mom_weights = 1 / volume
    rev_rev_weights = volume
    rev_rev = pd.Series(log_returns * rev_rev_weights, index=trade.index).rolling(rolling).sum()
    rev_mom = pd.Series(log_returns * rev_mom_weights, index=trade.index).rolling(rolling).sum()
    structural_reversal = rev_rev - rev_mom

    reverse_mom_structural_reversal_resid = rollols_resid(rev_mom, structural_reversal, rolling)
    sign_mask = np.sign(reverse_mom_structural_reversal_resid)
    value_mask = abs(reverse_mom_structural_reversal_resid)
    return sign_mask * np.sqrt(value_mask)


'''
超大单(单笔成交额大于10万美元)、大单(单笔成交额大于1万美元,小于10万美元)、
中单(单笔成交额大于1000美元,小于1万美元)、
小单(单笔成交额大于100美元,小于1000美元)、
微型单(单笔成交额小于100美元,大于10美元)以及纳米单(单笔成交额小于10美元)

资金流波动：主动买入比例、主动买入额、主动卖出比例、主动卖出额、净买入比例、净买入额标准差；

资金流振幅：主动买入比例、主动买入额、主动卖出比例、主动卖出额、净买入比例、净买入额极差；

买卖力量差：主动买入成交额均值-主动卖出成交额均值；

买卖波动差：主动买入成交额标准差-主动卖出成交额标准差；
'''


# 计算标准分
def superlarge_amt(depth, trade, rolling=1000):
    size = trade["buy_qty"] + trade["sell_qty"]
    amount = trade["amount"]
    x = pd.Series(np.arange(len(trade)), index=trade.index)
    x = x % rolling + 1
    buylarge_threshold = amount.rolling(rolling).quantile(0.9)
    # buysuperlarge = np.where((size > 0) & (amount >= 100_000), amount, 0)
    # sellsuperlarge = np.where((size < 0) & (amount >= 100_000), amount, 0)
    buysuperlarge = np.where((size > 0) & (amount >= buylarge_threshold), amount, 0)
    sellsuperlarge = np.where((size < 0) & (amount >= buylarge_threshold), -amount, 0)
    # 波动因子
    buysuperlarge_ratio = np.sqrt(
        pd.Series(buysuperlarge, index=trade.index).rolling(rolling).sum()
        / amount.rolling(rolling).sum()
    )
    sellsuperlarge_ratio_ = (
            pd.Series(sellsuperlarge, index=trade.index).rolling(rolling).sum()
            / amount.rolling(rolling).sum()
    )
    sellsuperlarge_ratio = np.sign(sellsuperlarge_ratio_) * np.sqrt(abs(sellsuperlarge_ratio_))

    superlarge_net_flow_ = (
                               pd.Series((buysuperlarge + sellsuperlarge), index=trade.index).rolling(rolling).mean()
                           ) / (
                               pd.Series((buysuperlarge + sellsuperlarge), index=trade.index).rolling(rolling).std()
                           )
    superlarge_net_flow = np.sign(superlarge_net_flow_) * np.sqrt(abs(superlarge_net_flow_))

    superlarge_net_flow_ratio_ = (
            pd.Series(abs(buysuperlarge + sellsuperlarge), index=trade.index).rolling(rolling).sum()
            / amount.rolling(rolling).sum()
    )

    superlarge_net_flow_ratio = np.sign(superlarge_net_flow_ratio_) * np.sqrt(abs(superlarge_net_flow_ratio_))

    # 振幅因子
    buysuperlarge_max = pd.Series(buysuperlarge, index=trade.index).rolling(rolling).max()
    buysuperlarge_mean = pd.Series(buysuperlarge, index=trade.index).rolling(rolling).mean()
    buysuperlarge_ampt_ = (buysuperlarge_max - buysuperlarge_mean) / (buysuperlarge_max + buysuperlarge_mean)
    buysuperlarge_ampt = np.sqrt(abs(buysuperlarge_ampt_)) * np.sign(buysuperlarge_ampt_)
    buysuperlarge_ampt_beta_ = buysuperlarge_ampt_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    buysuperlarge_ampt_beta = np.sqrt(abs(buysuperlarge_ampt_beta_)) * np.sign(buysuperlarge_ampt_beta_)

    sellsuperlarge_mean = pd.Series(sellsuperlarge, index=trade.index).rolling(rolling).mean()
    sellsuperlarge_min = pd.Series(sellsuperlarge, index=trade.index).rolling(rolling).min()
    sellsuperlarge_ampt_ = (sellsuperlarge_min - sellsuperlarge_mean) / (sellsuperlarge_min + sellsuperlarge_mean) * -1
    sellsuperlarge_ampt = np.sqrt(abs(sellsuperlarge_ampt_)) * np.sign(sellsuperlarge_ampt_)
    sellsuperlarge_ampt_beta_ = sellsuperlarge_ampt_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    sellsuperlarge_ampt_beta = np.sqrt(abs(sellsuperlarge_ampt_beta_)) * np.sign(sellsuperlarge_ampt_beta_)

    superlarge_net_flow_max = (
        pd.Series(abs(buysuperlarge + sellsuperlarge), index=trade.index).rolling(rolling).max()
    )
    superlarge_net_flow_min = (
        pd.Series(abs(buysuperlarge + sellsuperlarge), index=trade.index).rolling(rolling).mean()
    )
    superlarge_net_flow_ampt_ = (superlarge_net_flow_max - superlarge_net_flow_min) / (
                superlarge_net_flow_max + superlarge_net_flow_min)
    superlarge_net_flow_ampt = np.sqrt(abs(superlarge_net_flow_ampt_)) * np.sign(superlarge_net_flow_ampt_)
    superlarge_net_flow_ampt_beta_ = superlarge_net_flow_ampt_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    superlarge_net_flow_ampt_beta = np.sqrt(abs(superlarge_net_flow_ampt_beta_)) * np.sign(
        superlarge_net_flow_ampt_beta_)

    # # 极差因子
    # superlarge_net_flow_range_ = (
    #     pd.Series(buysuperlarge - sellsuperlarge, index=trade.index).rolling(rolling).max()
    #     - pd.Series(buysuperlarge - sellsuperlarge, index=trade.index).rolling(rolling).min()
    # )
    # superlarge_net_flow_range = np.sqrt(abs(superlarge_net_flow_range_)) * np.sign(superlarge_net_flow_range_)
    # superlarge_net_flow_range_beta = superlarge_net_flow_range_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    # superlarge_net_flow_range_beta = np.sqrt(abs(superlarge_net_flow_range_beta)) * np.sign(superlarge_net_flow_range_beta)

    # 买卖力量差
    # superlargebuysell_diff_ = (
    #     pd.Series(buysuperlarge, index=trade.index).rolling(rolling).mean()
    #     + pd.Series(sellsuperlarge, index=trade.index).rolling(rolling).mean()
    # )
    superlargebuysell_diff_ = (
                                      pd.Series(buysuperlarge, index=trade.index).rolling(rolling).max()
                                      - pd.Series(sellsuperlarge, index=trade.index).rolling(rolling).min()
                              ) / (
                                      pd.Series(buysuperlarge, index=trade.index).rolling(rolling).max()
                                      + pd.Series(sellsuperlarge, index=trade.index).rolling(rolling).min()
                              )
    superlargebuysell_diff = np.sign(superlargebuysell_diff_) * np.sqrt(abs(superlargebuysell_diff_))
    superlargebuysell_diff_beta_ = superlargebuysell_diff.rolling(rolling).cov(x) / x.rolling(rolling).var()
    superlargebuysell_diff_beta = np.sqrt(abs(superlargebuysell_diff_beta_)) * np.sign(superlargebuysell_diff_beta_)

    # # 买卖波动差
    # superlargebuysell_std_ = (
    #     pd.Series(buysuperlarge, index=trade.index).rolling(rolling).std()
    #     + pd.Series(sellsuperlarge, index=trade.index).rolling(rolling).std()
    # )
    # superlargebuysell_std = np.sign(superlargebuysell_std_) * np.sqrt(abs(superlargebuysell_std_))
    # superlargebuysell_std_beta_ = superlargebuysell_std_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    # superlargebuysell_std_beta = np.sqrt(abs(superlargebuysell_std_beta_)) * np.sign(superlargebuysell_std_beta_)

    return (
        buysuperlarge_ratio,
        sellsuperlarge_ratio,
        superlarge_net_flow,
        superlarge_net_flow_ratio,
        buysuperlarge_ampt,
        sellsuperlarge_ampt,
        superlarge_net_flow_ampt,
        superlargebuysell_diff,
        buysuperlarge_ampt_beta,
        sellsuperlarge_ampt_beta,
        superlarge_net_flow_ampt_beta,
        superlargebuysell_diff_beta,
    )


def small_amt(depth, trade, rolling=1000):
    size = trade["buy_qty"] + trade["sell_qty"]
    amount = trade["amount"]
    buysmall_threshold = amount.rolling(rolling).quantile(0.1)
    x = pd.Series(np.arange(len(trade)), index=trade.index)
    x = x % rolling + 1
    buysmall = np.where((size > 0) & (amount <= buysmall_threshold), amount, 0)
    sellsmall = np.where((size < 0) & (amount <= buysmall_threshold), -amount, 0)
    # 波动因子
    buysmall_ratio = np.sqrt(
        pd.Series(buysmall, index=trade.index).rolling(rolling).sum()
        / amount.rolling(rolling).sum()
    )
    sellsmall_ratio_ = (
            pd.Series(sellsmall, index=trade.index).rolling(rolling).sum()
            / amount.rolling(rolling).sum()
    )
    sellsmall_ratio = np.sign(sellsmall_ratio_) * np.sqrt(abs(sellsmall_ratio_))
    small_net_flow_ = (
                          pd.Series((buysmall + sellsmall), index=trade.index).rolling(rolling).mean()
                      ) / (
                          pd.Series((buysmall + sellsmall), index=trade.index).rolling(rolling).std()
                      )
    small_net_flow = np.sign(small_net_flow_) * np.sqrt(abs(small_net_flow_))

    small_net_flow_ratio_ = (
            pd.Series(abs(buysmall + sellsmall), index=trade.index).rolling(rolling).sum()
            / amount.rolling(rolling).sum()
    )
    small_net_flow_ratio = np.sign(small_net_flow_ratio_) * np.sqrt(abs(small_net_flow_ratio_))

    # 振幅因子
    buysmall_max = pd.Series(buysmall, index=trade.index).rolling(rolling).max()
    buysmall_mean = pd.Series(buysmall, index=trade.index).rolling(rolling).mean()
    buysmall_ampt_ = (buysmall_max - buysmall_mean) / (buysmall_max + buysmall_mean)
    buysmall_ampt = np.sqrt(abs(buysmall_ampt_)) * np.sign(buysmall_ampt_)
    buysmall_ampt_beta_ = buysmall_ampt_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    buysmall_ampt_beta = np.sqrt(abs(buysmall_ampt_beta_)) * np.sign(buysmall_ampt_beta_)

    sellsmall_mean = pd.Series(sellsmall, index=trade.index).rolling(rolling).mean()
    sellsmall_min = pd.Series(sellsmall, index=trade.index).rolling(rolling).min()
    sellsmall_ampt_ = (sellsmall_min - sellsmall_mean) / (sellsmall_min + sellsmall_mean) * -1
    sellsmall_ampt = np.sqrt(abs(sellsmall_ampt_)) * np.sign(sellsmall_ampt_)
    sellsmall_ampt_beta_ = sellsmall_ampt_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    sellsmall_ampt_beta = np.sqrt(abs(sellsmall_ampt_beta_)) * np.sign(sellsmall_ampt_beta_)

    small_net_flow_max = pd.Series(abs(buysmall + sellsmall), index=trade.index).rolling(rolling).max()
    small_net_flow_mean = pd.Series(abs(buysmall + sellsmall), index=trade.index).rolling(rolling).mean()
    small_net_flow_ampt_ = (small_net_flow_max - small_net_flow_mean) / (small_net_flow_max + small_net_flow_mean)
    small_net_flow_ampt = np.sqrt(abs(small_net_flow_ampt_)) * np.sign(small_net_flow_ampt_)
    small_net_flow_ampt_beta_ = small_net_flow_ampt_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    small_net_flow_ampt_beta = np.sqrt(abs(small_net_flow_ampt_beta_)) * np.sign(small_net_flow_ampt_beta_)

    # 极差因子
    # small_net_flow_range_ = (
    #     pd.Series(buysmall - sellsmall, index=trade.index).rolling(rolling).max()
    #     - pd.Series(buysmall - sellsmall, index=trade.index).rolling(rolling).min()
    # )
    # small_net_flow_range = np.sqrt(abs(small_net_flow_range_)) * np.sign(small_net_flow_range_)
    # small_net_flow_range_beta = small_net_flow_range_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    # small_net_flow_range_beta = np.sqrt(abs(small_net_flow_range_beta)) * np.sign(small_net_flow_range_beta)

    # 买卖力量差
    smallbuysell_diff_ = (
                                 pd.Series(buysmall, index=trade.index).rolling(rolling).max()
                                 - pd.Series(sellsmall, index=trade.index).rolling(rolling).min()
                         ) / (
                                 pd.Series(buysmall, index=trade.index).rolling(rolling).max()
                                 + pd.Series(sellsmall, index=trade.index).rolling(rolling).min()
                         )
    smallbuysell_diff = np.sqrt(abs(smallbuysell_diff_)) * np.sign(smallbuysell_diff_)
    smallbuysell_diff_beta_ = smallbuysell_diff_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    smallbuysell_diff_beta = np.sqrt(abs(smallbuysell_diff_beta_)) * np.sign(smallbuysell_diff_beta_)

    # # 买卖波动差
    # smallbuysell_std_ = (
    #     pd.Series(buysmall, index=trade.index).rolling(rolling).std()
    #     + pd.Series(sellsmall, index=trade.index).rolling(rolling).std()
    # )
    # smallbuysell_std = np.sqrt(abs(smallbuysell_std_)) * np.sign(smallbuysell_std_)
    # smallbuysell_std_beta_ = smallbuysell_std_.rolling(rolling).cov(x) / x.rolling(rolling).var()
    # smallbuysell_std_beta = np.sqrt(abs(smallbuysell_std_beta_)) * np.sign(smallbuysell_std_beta_)

    return (
        buysmall_ratio,
        sellsmall_ratio,
        small_net_flow,
        small_net_flow_ratio,
        buysmall_ampt,
        sellsmall_ampt,
        small_net_flow_ampt,
        smallbuysell_diff,
        buysmall_ampt_beta,
        sellsmall_ampt_beta,
        small_net_flow_ampt_beta,
        smallbuysell_diff_beta,
    )


# new
def abs_volume_kurt_ols(depth, trade, rolling=500):
    size = trade['buy_qty'] + trade['sell_qty']
    abs_volume_kurt = size.rolling(rolling).kurt()
    x = pd.Series(np.arange(len(trade)), index=trade.index)
    x = x % rolling + 1
    beta = abs_volume_kurt.rolling(rolling).cov(x) / x.rolling(rolling).var()
    sign_mask = np.sign(beta)
    value_mask = abs(beta)
    return sign_mask * np.sqrt(value_mask)


# new
def abs_volume_skew_ols(depth, trade, rolling=500):
    size = trade['buy_qty'] + trade['sell_qty']
    abs_volume_skew = size.rolling(rolling).skew()
    x = pd.Series(np.arange(len(trade)), index=trade.index)
    x = x % rolling + 1
    beta = abs_volume_skew.rolling(rolling).cov(x) / x.rolling(rolling).var()
    sign_mask = np.sign(beta)
    value_mask = abs(beta)
    return sign_mask * np.sqrt(value_mask)


# new
def avg_rank_ols(depth, trade, rolling=100, pricetick=6):
    a = trade['amount'].copy().ffill()
    v = trade['volume'].copy().ffill()
    avg = round(a / v, pricetick)
    avg_rank = pd.Series(avg, index=trade.index).rolling(rolling).rank() / rolling * 2 - 1
    x = pd.Series(np.arange(len(trade)), index=trade.index)
    x = x % rolling + 1
    avg_rank_ols = rollols_resid(avg_rank, x, rolling)
    return avg_rank_ols


# new
def avg_price_ols(depth, trade, rolling=100, pricetick=6):
    a = trade['amount'].copy().ffill()
    v = trade['volume'].copy().ffill()
    avg = round(a / v, pricetick)
    x = pd.Series(np.arange(len(trade)), index=trade.index)
    x = x % rolling + 1
    avg_price_ols = avg.rolling(rolling).cov(x) / x.rolling(rolling).var()
    value_mask = abs(avg_price_ols)
    sign_mask = np.sign(avg_price_ols)
    return np.sqrt(value_mask) * sign_mask


def dnl2_ratio(trade, depth, rolling=20, pricetick=7):
    # Calculate log returns
    avg = round(trade['amount'] / trade['volume'], pricetick)
    # log_returns = np.log(trade['price'] / trade['price'].shift(1))
    log_returns = np.log(avg / avg.shift(rolling))

    # Calculate log volume relative to its rolling sum
    log_volume_ratio = np.log(trade['volume'] / (trade['volume'].rolling(rolling).mean()))

    # Function to standardize series using expanding window
    def rolling_standardize(series, window):
        mean = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        return (series - mean) / std

    # Standardize both series using expanding window
    a = rolling_standardize(log_returns, rolling)
    b = rolling_standardize(log_volume_ratio, rolling)
    # a = log_returns
    # b = log_volume_ratio

    # Calculate l1_ratio using standardized values
    # l1_ratio = (standardized_returns - standardized_volume) / (standardized_returns + standardized_volume)
    l2_ratio = (a - b) / np.sqrt(a ** 2 + b ** 2)
    # Apply final transformation
    return l2_ratio


def ob_amount(depth, trade):
    amount = (depth['traded_ask_amount1'] + depth['traded_bid_amount1'])
    # amount_mean = pd.Series(amount, index=depth.index).rolling(rolling).quantile(0.99)
    return amount


def trade_ewam_amount(depth, trade, rolling=1000, q=0.99):
    size = trade['volume'].ffill()
    price = trade['price'].ffill()
    diff = price * size
    dollar_bar_amount = pd.Series(abs(diff), index=trade.index).ewm(span=rolling).mean()
    # dollar_bar_amount =  diff.ewm(span=rolling).std() * np.sqrt(rolling)
    dollar_bar_amount_percentile = dollar_bar_amount.rolling(rolling).quantile(q)
    return dollar_bar_amount_percentile


def vwap_price(depth, trade, rolling=120, pricetick=3):
    size = trade['volume'].ffill()
    price = trade['price'].ffill()
    vwap_price = pd.Series((price * size), index=trade.index).rolling(rolling).sum() / pd.Series(size,
                                                                                                 index=trade.index).rolling(
        rolling).sum()
    return round(vwap_price, pricetick)


def trade_amount_quantile(depth, trade, rolling=1000, q=0.99):
    amount = trade['amount'].ffill()
    dollar_bar_amount = pd.Series(amount, index=trade.index).rolling(rolling).quantile(q)
    # dollar_bar_amount =  diff.ewm(span=rolling).std() * np.sqrt(rolling)
    return dollar_bar_amount


def trade_amount_ewma(depth, trade, rolling=1000):
    amount = trade['amount'].ffill()
    dollar_bar_amount = pd.Series(amount, index=trade.index).ewm(span=rolling).mean()
    return dollar_bar_amount


def ewma_price(depth, trade, rolling=120, pricetick=3):
    price = trade['price']
    # alpha = 1 - safe_log(2) / rolling  # This is ewma's decay factor.
    alpha = 1 - np.exp(-safe_log(2) / rolling)
    weights = list(reversed([(1 - alpha) ** n for n in range(rolling)]))
    ewma = partial(np.average, weights=weights)
    ewma_price = price.swifter.rolling(rolling).apply(ewma)
    return round(ewma_price, pricetick)


def weights_price(depth, trade, level=10, pricetick=3):
    w = [1 - (i - 1) / level for i in range(1, level + 1)]
    w = np.array(w) / sum(w)
    ask_bid, ask_bid_v, = 0, 0
    for i in range(1, level + 1):
        ask_bid += (depth[f'ask_price{i}'] * depth[f'ask_size{i}'] + depth[f'bid_price{i}'] * depth[f'bid_size{i}']) * \
                   w[i - 1]
        ask_bid_v += (depth[f'ask_size{i}'] + depth[f'bid_size{i}']) * w[i - 1]
    weights_price = ask_bid / ask_bid_v
    return round(weights_price, pricetick)


def vwap_weights_price(depth, trade, rolling=120, pricetick=3, level=10):
    size = trade['volume'].ffill()
    w = [1 - (i - 1) / 10 for i in range(1, level + 1)]
    w = np.array(w) / sum(w)
    ask_bid, ask_bid_v, _ask_bid_v = 0, 0, 0
    for i in range(1, level + 1):
        ask_bid += (depth[f'ask_price{i}'] * depth[f'ask_size{i}'] + depth[f'bid_price{i}'] * depth[f'bid_size{i}']) * \
                   w[i - 1]
        ask_bid_v += (depth[f'ask_size{i}'] + depth[f'bid_size{i}']) * w[i - 1]
        _ask_bid_v += (depth[f'ask_size{i}'] + depth[f'bid_size{i}'])
    weights_price = ask_bid / ask_bid_v
    vwap_weights_price = pd.Series((weights_price * size), index=depth.index).rolling(rolling).sum() / pd.Series(size,
                                                                                                                 index=trade.index).rolling(
        rolling).sum()
    return round(vwap_weights_price, pricetick)


def ewma_weights_price(depth, trade, rolling=120, pricetick=3, level=10):
    w = [1 - (i - 1) / level for i in range(1, level + 1)]
    w = np.array(w) / sum(w)
    ask_bid, ask_bid_v = 0, 0
    for i in range(1, level + 1):
        ask_bid += (depth[f'ask_price{i}'] * depth[f'ask_size{i}'] + depth[f'bid_price{i}'] * depth[f'bid_size{i}']) * \
                   w[i - 1]
        ask_bid_v += (depth[f'ask_size{i}'] + depth[f'bid_size{i}']) * w[i - 1]
    weights_price = ask_bid / ask_bid_v
    alpha = 1 - safe_log(2) / 3  # This is ewma's decay factor.
    weights = list(reversed([(1 - alpha) ** n for n in range(rolling)]))
    ewma = partial(np.average, weights=weights)
    ewma_weights_price = weights_price.swifter.rolling(rolling).apply(ewma)
    return round(ewma_weights_price, pricetick)
