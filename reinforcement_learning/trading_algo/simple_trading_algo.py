import numpy as np
import pandas as pd


class SimpleTradingEnv:
    def __init__(self, data_frame, short_window=5, long_window=20):
        self.df = data_frame
        self.current_step = 0
        self.done = False

        self.action_space = 3  # 三种动作：保持，买入，卖出
        self.observation_space = 7  # 五个特征：开盘价、最高价、最低价、收盘价、交易量

        self.df['short_mavg'] = self.df['Close'].rolling(window=short_window, min_periods=1).mean()
        self.df['long_mavg'] = self.df['Close'].rolling(window=long_window, min_periods=1).mean()

        self.position = 0  # 持仓状态：0=空仓，1=持有
        self.initial_balance = 1000  # 初始资金
        self.balance = self.initial_balance

    def reset(self):
        self.current_step = 0
        self.done = False
        self.balance = self.initial_balance
        self.position = 0
        return self._next_observation()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.done = True

        reward = self._calculate_reward()
        obs = self._next_observation()

        return obs, reward, self.done

    def _next_observation(self):
        obs = self.df.loc[self.current_step, ['Open', 'High', 'Low', 'Close', 'Volume']].values
        obs = obs / obs.max()  # 归一化
        return obs

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'Close']

        if action == 1:  # 买入
            if self.position == 0:
                self.position = self.balance / current_price
                self.balance = 0
        elif action == 2:  # 卖出
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0

    def _calculate_reward(self):
        current_price = self.df.loc[self.current_step, 'Close']
        total_value = self.balance + self.position * current_price
        reward = total_value - self.initial_balance
        return reward

# 示例数据
data = {
    'Open': np.random.rand(100),
    'High': np.random.rand(100),
    'Low': np.random.rand(100),
    'Close': np.random.rand(100),
    'Volume': np.random.rand(100)
}
df = pd.DataFrame(data)

# 创建环境
env = SimpleTradingEnv(df)

# 运行环境的示例
obs = env.reset()
print(f'Initial observation: {obs}')

for _ in range(len(df)):
    action = np.random.choice(env.action_space)  # 随机选择一个动作
    obs, reward, done = env.step(action)
    print(f'Observation: {obs}, Reward: {reward}, Done: {done}')
    if done:
        print('Episode finished')
        break
