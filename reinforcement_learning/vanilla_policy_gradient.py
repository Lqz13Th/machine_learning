import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


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
        print(layers)
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


def compute_loss(policy, obs, actions, rewards):
    log_probs = torch.log(policy(obs))
    actions = actions.long()  # 显式转换为 long 类型
    selected_log_probs = log_probs[range(len(actions)), actions]
    loss = -(selected_log_probs * rewards).mean()
    return loss


def collect_trajectories(env, policy, max_steps_per_epoch=1000):
    obs, info = env.reset()
    obs_list, act_list, rew_list = [], [], []

    for _ in range(max_steps_per_epoch):
        obs_list.append(obs)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
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
    print(obs_dim, act_dim)
    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    epochs = 500
    batch_size = 128
    gamma = 0.99

    for epoch in range(epochs):
        obs, acts, rews = [], [], []
        total_steps = 0

        while total_steps < batch_size:
            batch_obs, batch_acts, batch_rews = collect_trajectories(env, policy)
            obs.append(batch_obs)
            acts.append(batch_acts)
            rews.append(batch_rews)
            total_steps += len(batch_rews)

        obs_con = np.concatenate(obs)
        acts_con = np.concatenate(acts)
        rews_con = np.concatenate(rews)

        disc_rews = compute_discounted_rewards(rews_con, gamma)

        optimizer.zero_grad()
        loss = compute_loss(
            policy,
            torch.as_tensor(obs_con, dtype=torch.float32),
            torch.as_tensor(acts_con, dtype=torch.float32),
            torch.as_tensor(disc_rews, dtype=torch.float32),
        )

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}: Loss={loss.item():.3f}, Return={np.sum(rews_con):.3f}")













