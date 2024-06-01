import numpy as np
np.random.seed(0)
# 定义状态转移概率矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]  # 定义奖励函数
gamma = 0.5  # 定义折扣因子


# 给定一条序列,计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G


# 一个状态序列,s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print("根据本序列计算得到回报为：%s。" % G)


def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数 '''
    rewards = np.array(rewards).reshape((-1, 1))  #将rewards写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   rewards)
    return value


V = compute(P, rewards, gamma, 6)
print("MRP中每个状态价值分别为\n", V)

# 强化学习中的基本概念和贝尔曼期望方程

# 状态价值函数 (State Value Function)
# 这是在策略 π 下，状态 s 的价值，即从状态 s 开始，遵循策略 π，预期能获得的总回报。
# 定义为:
# V^π(s) = E_π [ ∑_{t=0}^{∞} γ^t * R_{t+1} | S_0 = s ]
# 其中，γ 是折扣因子，表示未来奖励的现值。R_{t+1} 是时间步 t+1 的奖励。

# 状态-动作价值函数 (State-Action Value Function)
# 这是在策略 π 下，状态-动作对 (s, a) 的价值，即从状态 s 选择动作 a，然后遵循策略 π，预期能获得的总回报。
# 定义为:
# Q^π(s, a) = E_π [ ∑_{t=0}^{∞} γ^t * R_{t+1} | S_0 = s, A_0 = a ]

# 贝尔曼期望方程 (Bellman Expectation Equation)
# 将当前状态的价值表示为其后继状态的价值的期望。

# 状态价值函数的贝尔曼期望方程:
# V^π(s) = ∑_a π(a | s) ∑_{s'} P(s' | s, a) [ R(s, a, s') + γ * V^π(s') ]
# 解释:
# V^π(s) 是状态 s 在策略 π 下的价值。
# π(a | s) 是策略 π 在状态 s 选择动作 a 的概率。
# P(s' | s, a) 是状态转移概率，从状态 s 采取动作 a 转移到状态 s' 的概率。
# R(s, a, s') 是奖励函数，从状态 s 采取动作 a 到达状态 s' 所获得的奖励。
# γ 是折扣因子，表示未来奖励的现值。
# V^π(s') 是状态 s' 的价值。

# 状态-动作价值函数的贝尔曼期望方程:
# Q^π(s, a) = ∑_{s'} P(s' | s, a) [ R(s, a, s') + γ ∑_{a'} π(a' | s') Q^π(s', a') ]
# 解释:
# Q^π(s, a) 是状态-动作对 (s, a) 在策略 π 下的价值。
# P(s' | s, a) 是状态转移概率，从状态 s 采取动作 a 转移到状态 s' 的概率。
# R(s, a, s') 是即时奖励，从状态 s 采取动作 a 到达状态 s' 所获得的奖励。
# γ 是折扣因子。
# π(a' | s') 是策略 π 在状态 s' 选择动作 a' 的概率。
# Q^π(s', a') 是状态-动作对 (s', a') 的价值。

# 总结:
# - 状态价值函数 V(s) 表示在给定状态下的长期回报。
# - 状态-动作价值函数 Q(s, a) 表示在给定状态下采取特定动作的长期回报。
# - 贝尔曼期望方程提供了一种递归关系，用于计算状态或状态-动作对的价值。

# 通过理解这些方程，我们可以更好地设计和优化强化学习算法，使智能体能够在复杂环境中学习和做出最优决策。

# 定义网格世界的大小
grid_size = 4
state_space = grid_size * grid_size
action_space = 4

# 定义动作 (0=上, 1=右, 2=下, 3=左)
actions = ['U', 'R', 'D', 'L']

# 定义折扣因子
gamma = 0.9

# 初始化状态价值函数 V(s)
V = np.zeros(state_space)

# 定义奖励矩阵 (在这里我们假设除了终点外，其他地方的奖励都是-1)
rewards = -np.ones(state_space)
rewards[0] = 0  # 终点的奖励是0

# 定义状态转移概率矩阵 P(s'|s,a)
P = np.zeros((state_space, action_space, state_space))

# 填充状态转移概率矩阵 (假设所有动作都以概率1确定性地导致转移)
for s in range(state_space):
    row, col = divmod(s, grid_size)
    for a in range(action_space):
        if a == 0:  # 上
            next_state = (row - 1) * grid_size + col if row > 0 else s
        elif a == 1:  # 右
            next_state = row * grid_size + (col + 1) if col < grid_size - 1 else s
        elif a == 2:  # 下
            next_state = (row + 1) * grid_size + col if row < grid_size - 1 else s
        elif a == 3:  # 左
            next_state = row * grid_size + (col - 1) if col > 0 else s
        P[s, a, next_state] = 1.0

# 定义策略 π (随机策略)
policy = np.ones((state_space, action_space)) / action_space

# 定义一个函数来计算状态价值函数 V(s)
def compute_value_function(policy, P, rewards, gamma, theta=1e-6):
    V = np.zeros(state_space)
    while True:
        delta = 0
        for s in range(state_space):
            v = V[s]
            V[s] = sum(policy[s, a] * sum(P[s, a, s_prime] * (rewards[s_prime] + gamma * V[s_prime])
                         for s_prime in range(state_space))
                       for a in range(action_space))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# 计算状态价值函数 V(s)
V = compute_value_function(policy, P, rewards, gamma)

print("状态价值函数 V(s):")
print(V.reshape((grid_size, grid_size)))

# 定义一个函数来计算状态-动作价值函数 Q(s, a)
def compute_action_value_function(policy, P, rewards, V, gamma):
    Q = np.zeros((state_space, action_space))
    for s in range(state_space):
        for a in range(action_space):
            Q[s, a] = sum(P[s, a, s_prime] * (rewards[s_prime] + gamma * V[s_prime])
                          for s_prime in range(state_space))
    return Q

# 计算状态-动作价值函数 Q(s, a)
Q = compute_action_value_function(policy, P, rewards, V, gamma)

print("状态-动作价值函数 Q(s, a):")
print(Q)


S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2


gamma = 0.5
# 转化后的MRP的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
print("MDP中每个状态价值分别为\n", V)


def sample(MDP, Pi, timestep_max, number):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环
        episodes.append(episode)
    return episodes


# 采样5次,每个序列最长不超过20步
episodes = sample(MDP, Pi_1, 20, 5)
print('第一条序列\n', episodes[0])
print('第二条序列\n', episodes[1])
print('第五条序列\n', episodes[4])

# 对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]


timestep_max = 20
# 采样1000次,可以自行修改
episodes = sample(MDP, Pi_1, timestep_max, 1000)
gamma = 0.5
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
MC(episodes, V, N, gamma)
print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)


def occupancy(episodes, s, a, timestep_max, gamma):
    ''' 计算状态动作对（s,a）出现的频率,以此来估算策略的占用度量 '''
    rho = 0
    total_times = np.zeros(timestep_max)  # 记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


gamma = 0.5
timestep_max = 1000

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(rho_1, rho_2)


