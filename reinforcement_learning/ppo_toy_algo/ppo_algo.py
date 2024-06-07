import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class CryptoTradingEnv(gym.Env):
    def __init__(self, data_frame):
        super(CryptoTradingEnv, self).__init__()
        self.df = data_frame
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # 三种动作：保持，买入，卖出
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self, seed=1, options=None):
        self.current_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        observation_value = self.df.iloc[self.current_step]
        return observation_value.values

    def step(self, action_state):
        self.current_step += 1

        reward = 0
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # 可以根据需要添加截断逻辑
        current_obs = self._next_observation()

        return current_obs, reward, terminated, truncated, {}

    def render(self, mode='human', close=False):
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


# 创建交易环境
df = pd.read_csv('C:/Work Files/data/backtest/candle/candle1m/FET-USDT-SWAP1min.csv')
env = make_vec_env(lambda: CryptoTradingEnv(df), n_envs=1)

# 创建PPO模型
model = PPO('MlpPolicy', env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_crypto_trading")

# 加载模型
model = PPO.load("ppo_crypto_trading")

# 运行和评估模型
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
