import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from collections import deque
import random

# 确认 TensorFlow 和 Keras 版本
print(tf.__version__)
print(tf.keras.__version__)

# 状态空间大小（假设包括价格、MA和RSI）
state_size = 3
# 动作空间大小（买入、卖出、持有）
action_size = 3


def build_dqn_model(state_size, action_size):
    model = tf.keras.Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = build_dqn_model(state_size, action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=10, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.reset()

    def reset(self):
        self.current_step = 0
        self.done = False
        self.total_profit = 0
        self.position = 0  # 持仓量
        return self._get_state()

    def _get_state(self):
        state = self.data[self.current_step]
        return np.array(state).reshape((1, state_size))

    def step(self, action):
        reward = 0
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_state = self._get_state()

        # 动作：0=买入，1=卖出，2=持有
        if action == 0:  # 买入
            self.position = 1
        elif action == 1:  # 卖出
            if self.position == 1:
                reward = 1  # 获得收益
                self.total_profit += 1
                self.position = 0
        # 如果动作是持有，reward为0

        return next_state, reward, self.done


# 示例数据：每行表示一个状态（价格、MA、RSI）
data = [
    [100, 105, 30],
    [102, 106, 35],
    [101, 104, 25],
    [105, 108, 40],
    [104, 107, 45],
    # 添加更多数据
]

env = TradingEnvironment(data)
agent = DQNAgent(state_size, action_size)
batch_size = 32

for e in range(1000):  # 训练1000个回合
    state = env.reset()
    for time in range(500):  # 每个回合最多500步
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

print(f"训练完成，总收益: {env.total_profit}")

