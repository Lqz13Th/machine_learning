import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

print("CUDA available:", torch.cuda.is_available())  # 检查是否可用
print("CUDA version:", torch.version.cuda)  # 查看CUDA版本
print("Torch version:", torch.__version__)  # 查看 PyTorch 版本
print("Number of GPUs:", torch.cuda.device_count())  # 查看 GPU 数量
print("Current device:", torch.cuda.current_device())  # 当前使用的设备
print("Device name:", torch.cuda.get_device_name(0))  # 获取 GPU 名称

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=(64, 64)):
        super(PolicyNetwork, self).__init__()
        layers = []
        prev_size = obs_dim
        for size in hidden_size:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, act_dim))
        layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


def compute_loss(policy, obs, actions, rewards):
    log_probs = torch.log(policy(obs))
    actions = actions.long()  # 显式转换为 long 类型
    selected_log_probs = log_probs[range(len(actions)), actions]
    loss = -(selected_log_probs * rewards).mean()
    return loss


def collect_trajectories(env, policy):
    obs, info = env.reset()
    obs_list, act_list, rew_list = [], [], []

    while True:
        obs_list.append(obs)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action_probs = policy(obs_tensor)
        action = torch.multinomial(action_probs, num_samples=1).item()

        next_obs, reward, done, truncated, _ = env.step(action)
        act_list.append(action)
        rew_list.append(reward)

        obs = next_obs
        if done or truncated:
            break

    return np.array(obs_list), np.array(act_list), np.array(rew_list)


def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted_rewards[t] = running_sum
    return discounted_rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    epochs = 500
    batch_size = 1024  # 指定 batch 内总步数
    gamma = 0.99

    for epoch in range(epochs):
        all_obs, all_acts, all_disc_rews, all_raw_rews = [], [], [], []
        total_steps = 0

        # 收集多个 trajectory，直到总步数达到 batch_size
        while total_steps < batch_size:
            traj_obs, traj_acts, traj_rews = collect_trajectories(env, policy)
            traj_disc_rews = compute_discounted_rewards(traj_rews, gamma)

            all_obs.append(traj_obs)
            all_acts.append(traj_acts)
            all_disc_rews.append(traj_disc_rews)
            all_raw_rews.append(traj_rews)

            total_steps += len(traj_rews)

        if total_steps > batch_size:
            print(f"Warning: Total steps {total_steps} exceeded batch size {batch_size}.")

        # 拼接所有 trajectory 的数据
        obs_con = np.concatenate(all_obs)
        acts_con = np.concatenate(all_acts)
        disc_rews_con = np.concatenate(all_disc_rews)
        raw_rews_con = np.concatenate(all_raw_rews)

        # 对折扣奖励进行归一化处理
        disc_rews_con = (disc_rews_con - disc_rews_con.mean()) / (disc_rews_con.std() + 1e-8)

        optimizer.zero_grad()
        loss = compute_loss(
            policy,
            torch.as_tensor(obs_con, dtype=torch.float32).to(device),
            torch.as_tensor(acts_con, dtype=torch.float32).to(device),
            torch.as_tensor(disc_rews_con, dtype=torch.float32).to(device),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.1)

        optimizer.step()

        # 计算每个 episode 的累计原始奖励，并取平均作为评价指标
        episode_returns = [np.sum(traj) for traj in all_raw_rews]
        avg_return = np.mean(episode_returns)

        print(f"Epoch {epoch + 1}: Loss={loss.item():.3f}, Avg Return={avg_return:.3f}")

