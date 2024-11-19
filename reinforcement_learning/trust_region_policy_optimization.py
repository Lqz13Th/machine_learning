import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=(64, 64)):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.output_layer = nn.Linear(hidden_size[1], act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.output_layer(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs


def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """use generalized advantage estimation (GAE)"""

    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = np.zeros_like(deltas)
    running_advantage = 0
    for t in reversed(range(len(deltas))):
        running_advantage = deltas[t] + gamma * lam * running_advantage
        advantages[t] = running_advantage

    return advantages


def compute_loss(policy, old_policy_probs, obs, actions, advantages):
    new_policy_probs = policy(obs)
    log_probs = torch.log(new_policy_probs + 1e-10)
    old_log_probs = torch.log(old_policy_probs + 1e-10)

    actions = actions.long()
    selected_new_log_probs = log_probs[range(len(actions)), actions]
    selected_old_log_probs = old_log_probs[range(len(actions)), actions]

    ratio = torch.exp(selected_new_log_probs - selected_old_log_probs)
    surrogate_loss = -(ratio * advantages).mean()

    return surrogate_loss


def compute_value_loss(value_net, obs, returns):
    predicted_values = value_net(obs).squeeze(-1)
    loss = F.mse_loss(predicted_values, returns)
    return loss


def compute_kl_divergence(policy, old_policy_probs, obs):
    new_policy_probs = policy(obs)
    kl = old_policy_probs * (torch.log(old_policy_probs + 1e-10) - torch.log(new_policy_probs + 1e-10))
    return kl.sum(dim=1).mean()


def collect_trajectories(env, policy, max_steps_per_epoch=1000):
    obs, info = env.reset()
    obs_list, act_list, rew_list, val_list = [], [], [], []

    for _ in range(max_steps_per_epoch):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        action_probs = policy(obs_tensor)

        action = torch.multinomial(action_probs, num_samples=1).item()

        next_obs, reward, done, truncated, _ = env.step(action)

        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)

        obs = next_obs
        if done or truncated:
            break

    return np.array(obs_list), np.array(act_list), np.array(rew_list)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    print(obs_dim, act_dim)

    epochs = 500
    batch_size = 120
    gamma = 0.99
    lam = 0.95
    kl_threshold = 0.01

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

        obs_tensor = torch.as_tensor(obs_con, dtype=torch.float32)
        advantages = compute_advantages(rews_con, vals_con, gamma, lam)

        old_policy_probs = policy(obs_tensor).detach()

        for _ in range(10):
            optimizer.zero_grad()

            loss = compute_loss(
                policy,
                old_policy_probs,
                obs_tensor,
                torch.as_tensor(acts_con, dtype=torch.int64),
                torch.as_tensor(advantages, dtype=torch.float32)
            )
            kl = compute_kl_divergence(policy, old_policy_probs, obs_tensor)

            if kl > kl_threshold:
                print(f"KL divergence exceeded threshold: {kl.item()}")
                break

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}: Loss={loss.item():.3f}, Return={np.sum(rews_con):.3f}")
















