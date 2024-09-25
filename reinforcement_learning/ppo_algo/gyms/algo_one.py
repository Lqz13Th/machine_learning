import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from reinforcement_learning.ppo_algo.datas import candle_data_parser


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 在每一步记录自定义信息, 比如奖励
        reward = self.locals['rewards'][0]  # 当前环境的奖励
        self.logger.record('train/reward', reward)  # 记录奖励到 TensorBoard
        return True  # 如果返回 False 会提前停止训练


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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):  # 添加 seed 参数
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0

        return self._next_observation(), {}

    def step(self, action_state):
        self._take_action(action_state)
        self.current_step += 1

        reward = self._calculate_reward()
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        current_obs = self._next_observation()
        return current_obs, reward, terminated, truncated, {}

    def _next_observation(self):
        observation_value = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']]
        return observation_value.values

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


def linear_schedule(initial_value, final_value=1e-5, steps=1e6):
    """
    Slower linear learning rate schedule that decays less aggressively.
    """
    def func(progress_remaining):
        return max(final_value, initial_value * progress_remaining + (1 - progress_remaining) * final_value)
    return func


if __name__ == '__main__':
    plt.style.use('seaborn-v0_8')
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    psd = candle_data_parser.ParseCandleData()
    df = psd.parse_candle_data_okx('C:/Work Files/data/backtest/candle/candle1m/FIL-USDT1min.csv')
    print(df)
    old_max_steps = df.index.max()

    df = df.tail(int(old_max_steps * 1)).reset_index(drop=True)
    max_steps = df.index.max()
    print(df)

    env = make_vec_env(lambda: CryptoTradingEnv(df), n_envs=1)
    model = PPO(
        'MlpPolicy',
        env,
        verbose=2,
        learning_rate=linear_schedule(initial_value=3e-4, final_value=3e-4, steps=max_steps),
        n_steps=max(int(max_steps * 0.005), 60*24*7),  # Number of steps to collect in each environment before updating
        batch_size=64,  # Batch size used for optimization
        n_epochs=10,
        gamma=0.99,
        clip_range=linear_schedule(initial_value=0.2, final_value=0.2, steps=max_steps),
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # False for discrete actions
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        seed=1,
        device="auto",
        tensorboard_log="./ppo_crypto_trading_tensorboard/",
    )

    model.learn(
        total_timesteps=int(max_steps * 0.7),
        callback=TensorboardCallback(),
    ).save("ppo_crypto_trading")

    del model

    model = PPO.load("ppo_crypto_trading")
    obs = env.reset()

    px_lst = []
    pnl_lst = []
    for i in range(max_steps):
        ppo_action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(ppo_action)
        # print(obs, rewards, i, max_steps)
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
