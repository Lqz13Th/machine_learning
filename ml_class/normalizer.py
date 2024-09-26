import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """标准sigmoid函数"""
    return 1 / (1 + np.exp(-x))


def scaled_sigmoid(x, start, end):
    """当`x`落在`[start,end]`区间时，函数值为[0,1]且在该区间有较好的响应灵敏度"""
    n = np.abs(start - end)
    score = 2 / (1 + np.exp(-np.log(40_000) * (x - start - n) / n + np.log(5e-3)))
    return score / 2


def plot_scaled_sigmoid():
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

    x = np.linspace(-1, 1)
    ax1.plot(x, [scaled_sigmoid(i, x[0], x[-1]) for i in x])
    ax1.set_title("fit (0,1)")
    ax1.grid()

    x = np.linspace(0, 100)
    print(x)
    ax2.plot(x, [scaled_sigmoid(i, x[0], x[-1]) for i in x])
    ax2.set_title("fit (0, 100)")
    ax2.grid()

    x = np.linspace(18, 38)
    ax3.plot(x, [scaled_sigmoid(i, x[0], x[-1]) for i in x])
    ax3.set_title("fit (18, 38)")
    ax3.grid()

    x = np.linspace(0, 100)
    ax4.plot(x, [sigmoid(i) for i in x])
    ax4.set_title("fit (0,100) with original")
    ax4.grid()

    plt.tight_layout()
    plt.show()


# 绘制图形
plot_scaled_sigmoid()
