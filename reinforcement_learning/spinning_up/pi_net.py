import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

obs_dim = 12
act_dim = 3

pi_net = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, act_dim),
)

obs = np.random.rand(obs_dim)  # 随机生成一个 12 维向量
obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
actions = pi_net(obs_tensor)
print(actions)


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


# 设置参数
obs_dim = 12  # 观测维度
n_acts = 3    # 动作空间维度
hidden_sizes = [64, 64]  # 隐藏层大小

# 构建策略网络
logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])


# 定义函数
def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)


def get_action(obs):
    return get_policy(obs).sample().item()


# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()


# 示例：单个观测
obs = torch.randn(obs_dim)  # 随机生成一个观测
action = get_action(obs)
print("Sampled Action:", action, obs)

# 示例：批量观测
batch_obs = torch.randn(5, obs_dim)  # 随机生成 5 个观测
policy = get_policy(batch_obs)  # 获取批量策略
batch_actions = policy.sample()  # 批量采样动作
print("Batch Sampled Actions:", batch_actions)


# for training policy
def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    # obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:

        # rendering
        # if (not finished_rendering_this_epoch) and render:
        #     env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(torch.as_tensor(obs, dtype=torch.float32))
        # obs, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        # ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            # obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            # if len(batch_obs) > batch_size:
            #     break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                              act=torch.as_tensor(batch_acts, dtype=torch.int32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                              )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens
