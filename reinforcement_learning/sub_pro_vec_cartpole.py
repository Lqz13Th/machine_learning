import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    return _init


if __name__ == '__main__':
    # 设置环境名称，例如 "CartPole-v1"
    env_id = "CartPole-v1"
    num_cpu = 4  # 定义环境的进程数

    # 创建多进程环境
    env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # 初始化模型（例如使用 PPO 算法）
    model = PPO("MlpPolicy", env, verbose=1)

    # 训练模型
    model.learn(total_timesteps=100000)

    # 测试训练后的模型
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

    env.close()
