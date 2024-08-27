import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

import crypto_data_parser


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Collect rewards from the environment
        if self.num_timesteps % self.model.n_steps == 0:
            env = self.model.get_env()
            for _ in range(self.model.n_envs):
                obs = env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, reward, done, _ = env.step(action)
                    total_reward += reward
                self.episode_rewards.append(total_reward)
            print(f"Step {self.num_timesteps}: Average Reward: {np.mean(self.episode_rewards)}")
        return True


class CryptoTradingEnv(gym.Env):
    def __init__(self, data_frame):
        super(CryptoTradingEnv, self).__init__()
        self.df = data_frame
        self.current_step = 0
        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.position = 0  # 持仓状态：0=空仓，1=持有

        assert all(col in df.columns for col in [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ]), "DataFrame should have columns: 'Open', 'High', 'Low', 'Close', 'Volume'"

        self.action_space = spaces.Discrete(3)  # 三种动作：保持，买入，卖出
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def reset(self, seed=1, a=None):  # 添加 seed 参数
        if seed is not None:
            np.random.seed(seed)  # 例如，使用 NumPy 设置随机种子
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        return self._next_observation(), {}

    def _next_observation(self):
        observation_value = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']]
        return observation_value.values

    def step(self, action_state):
        self._take_action(action_state)
        self.current_step += 1

        reward = self._calculate_reward()
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        current_obs = self._next_observation()
        return current_obs, reward, terminated, truncated, {}

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


if __name__ == '__main__':
    plt.style.use('seaborn-v0_8')
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    psd = crypto_data_parser.ParseCryptoData()
    df = psd.parse_candle_data_okx('C:/Work Files/data/backtest/candle/candle1m/FIL-USDT1min.csv')
    print(df)
    max_steps = df.index.max()

    env = make_vec_env(lambda: CryptoTradingEnv(df), n_envs=1)
    env.seed(seed=1)
    model = PPO(
        'MlpPolicy',
        env,
        verbose=2,
        n_steps=int(max_steps * 0.005),  # Number of steps to collect in each environment before updating
        batch_size=256,  # Batch size used for optimization
        n_epochs=10,
        tensorboard_log="./ppo_crypto_trading_tensorboard/",
    )

    reward_logger = RewardLoggerCallback()

    model.learn(
        total_timesteps=int(max_steps * 0.7),
        # callback=reward_logger,
    )

    model.save("ppo_crypto_trading")
    del model

    model = PPO.load("ppo_crypto_trading")

    env.seed(seed=1)

    obs = env.reset()

    px_lst = []
    pnl_lst = []
    for i in range(max_steps):
        ppo_action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(ppo_action)
        print(obs, rewards, i, max_steps)
        if i % 10 == 0:
            px_lst.append(obs[0][3])
            pnl_lst.append(rewards[0])

        if dones[0]:
            break  # 完成后跳出

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(px_lst)
    axs[0].set_title('Price (Close)')

    axs[1].plot(pnl_lst)
    axs[1].set_title('PnL (Reward)')

    plt.tight_layout()
    plt.show()



