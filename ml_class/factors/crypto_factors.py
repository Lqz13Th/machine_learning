import pandas as pd
import numpy as np
from numba import njit


# 使用 Numba 优化计算 drawdown 和 rebound
@njit
def calculate_trend(prices, side):
    """
    计算回撤或反弹，side=1 表示回撤（drawdown），side=-1 表示反弹（rebound）
    """
    max_trend = np.max(prices)
    min_trend = np.min(prices)
    max_range = max_trend - min_trend

    if side == 1:
        # 计算 drawdown
        drawdown = (max_trend - prices[-1]) / max_range if max_range != 0 else 0
        return drawdown
    elif side == -1:
        # 计算 rebound
        rebound = (prices[-1] - min_trend) / max_range if max_range != 0 else 0
        return rebound
    else:
        return 1.0


# 计算 rolling 窗口内的趋势（drawdown 或 rebound）
def calc_trend_rolling(depth, rolling=100, side=1):
    """
    对给定 DataFrame 深度数据进行 rolling 计算趋势 drawdown 或 rebound
    :param depth: 包含价格数据的 DataFrame，列名应包括 'ask_price1' 和 'bid_price1'
    :param rolling: rolling 窗口大小
    :param side: 方向 (1 表示多头计算 drawdown, -1 表示空头计算 rebound)
    :return: 计算结果的 Series
    """
    if side == 1:
        # 使用 ask_price1 计算回撤 (drawdown)
        prices = depth['ask_price1']
    elif side == -1:
        # 使用 bid_price1 计算反弹 (rebound)
        prices = depth['bid_price1']
    else:
        raise ValueError("side 必须是 1 (多头) 或 -1 (空头)")

    # 使用 rolling 窗口并 apply 自定义的 trend 计算
    trend_changes = prices.rolling(rolling).apply(lambda x: calculate_trend(x, side), engine='numba', raw=True).fillna(
        0)

    return trend_changes


# 示例用法
df = pd.DataFrame({
    'ask_price1': [101, 102, 103, 100, 104, 105, 106],
    'bid_price1': [100, 101, 102, 99, 103, 104, 105]
})

# 计算 rolling 窗口内的 drawdown 和 rebound
drawdown_result = calc_trend_rolling(df, rolling=3, side=1)
rebound_result = calc_trend_rolling(df, rolling=3, side=-1)

print("Drawdown 结果:", drawdown_result)
print("Rebound 结果:", rebound_result)
