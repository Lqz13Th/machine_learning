import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import crypto_data_parser


class CryptoTradingEnv(gym.Env):
    def __init__(self, data_frame):
        super(CryptoTradingEnv, self).__init__()
        self.df = data_frame
        self.current_step = 0

        assert all(col in df.columns for col in [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ]), "DataFrame should have columns: 'Open', 'High', 'Low', 'Close', 'Volume'"

        self.action_space = spaces.Discrete(3)  # 三种动作：保持，买入，卖出
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def reset(self, seed=None):  # 添加 seed 参数
        if seed is not None:
            np.random.seed(seed)  # 例如，使用 NumPy 设置随机种子
        self.current_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        observation_value = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']]
        return observation_value.values

    def step(self, action_state):
        self.current_step += 1

        reward = 0
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # 可以根据需要添加截断逻辑
        current_obs = self._next_observation()

        return current_obs, reward, terminated, truncated, {}

    def render(self, mode="rgb_array", close=False):
        pass


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # plt.style.use('seaborn-v0_8')
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    psd = crypto_data_parser.ParseCryptoData()
    df = psd.parse_candle_data_okx('C:/Work Files/data/backtest/candle/candle1m/FIL-USDT1min.csv')
    print(df)

    input_dim = 5  # 假设观测空间是(5,)，即Open, High, Low, Close, Volume
    output_dim = 3

    # 创建交易环境
    env = make_vec_env(lambda: CryptoTradingEnv(df), n_envs=1)
    env.seed(seed=1)
    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
    )

    # 训练模型
    model.learn(total_timesteps=10000)

    # 保存模型
    model.save("ppo_crypto_trading")

    # 加载模型
    model = PPO.load("ppo_crypto_trading")

    env.seed(seed=1)

    # 运行和评估模型
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # env.render()
