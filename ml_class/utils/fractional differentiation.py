import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def fractional_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])


def fractional_differentiation(series, d, threshold=1e-8):  # 调整threshold值
    weights = fractional_weights(d, len(series))
    weights = weights[abs(weights) > threshold]  # 移除小于阈值的权重
    diff_series = np.convolve(series, weights, mode='same')
    print(diff_series)
    return pd.Series(diff_series, index=pd.RangeIndex(start=0, stop=len(diff_series)))


# 示例数据
time = np.arange(100)
trend_data = time + np.random.normal(0, 5, 100)  # 线性趋势加随机噪声
data = pd.Series(trend_data)

d = 0.5  # 分数阶

# 调用分数阶差分函数
diff_data = fractional_differentiation(data, d)

# 打印原始数据和分数阶差分结果
print("原始数据:")
print(data)
print("\n分数阶差分结果:")
print(diff_data)


def test_stationarity(data):
    result = adfuller(data, autolag='AIC')  # 仅运行一次 adfuller
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    if result[1] < 0.05:
        print("数据是平稳的")
    else:
        print("数据是非平稳的")

    # 显示临界值
    for key, value in result[4].items():
        print(f'Critical Values {key}: {value}')


# 检验示例数据
test_stationarity(data)
test_stationarity(diff_data)

first_diff = data.diff().dropna()
print(first_diff)
test_stationarity(first_diff)

second_diff = first_diff.diff().dropna()
print(second_diff)
test_stationarity(second_diff)

