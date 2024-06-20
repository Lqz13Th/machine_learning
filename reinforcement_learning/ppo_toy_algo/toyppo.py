import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class SimpleCryptoTradingEnv(gym.Env):
    def __init__(self):
        super(SimpleCryptoTradingEnv, self).__init__()
        self.price = 100  # 初始价格
        self.balance = 1000  # 初始资金
        self.inventory = 0  # 持有的加密货币数量
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # 买入或卖出
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, -1]), high=np.array([1000, 1000, 1]), dtype=np.float32)

    def reset(self, seed=None):  # 确保添加了 seed 参数
        # 重置环境到初始状态
        self.price = 100  # 初始价格
        self.balance = 1000  # 初始资金
        self.inventory = 0  # 持有的加密货币数量
        # 返回观察值和空信息字典
        return self.observation_space.sample(), {}  # 假设 self.observation_space.sample() 能正确生成初始观察值

    def step(self, action):
        transaction_cost = 0.001  # 交易费用
        self.balance -= transaction_cost * self.inventory  # 卖出时扣除交易费用
        if action > 0:  # 买入
            amount = (self.balance / self.price) * min(1, action)
            self.inventory += amount
            self.balance -= amount * self.price
        elif action < 0:  # 卖出
            amount = self.inventory * min(-1, action)
            self.inventory -= amount
            self.balance += amount * self.price
        self.price *= 1 + 0.05 * (np.random.rand() - 0.5)  # 价格波动
        done = False
        reward = self.balance + self.inventory * self.price - 1000  # 计算奖励

        terminated = False  # 或者根据你的环境逻辑设置
        truncated = False
        return np.array([self.balance, self.inventory, self.price]), reward, terminated, truncated, {}

    def render(self, mode='human', close=False):
        pass  # 可以添加可视化逻辑


env = SimpleCryptoTradingEnv()
check_env(env)  # 检查环境是否符合要求

env = DummyVecEnv([lambda: SimpleCryptoTradingEnv()])

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

model.save("simple_crypto_trading_model")
del model
model = PPO.load("simple_crypto_trading_model")


obs = env.reset()
for _ in range(500):  # 运行500个步骤
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f"Balance: {info['balance']}, Inventory: {info['inventory']}, Price: {info['price']}, Reward: {rewards[0]}")
    if dones:
        break