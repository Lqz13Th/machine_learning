import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 生成示例数据
data = {
    'Close': [100, 101, 102, 103, 105, 108, 110, 111, 115, 120, 125, 130]
}
df = pd.DataFrame(data)

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# 创建数据集
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 3
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 划分训练集和测试集
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=1)

# 测试模型
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # 反归一化

# 预测未来值
future_steps = 5
last_input = X_test[-1].reshape(1, time_step, 1)

for _ in range(future_steps):
    next_prediction = model.predict(last_input)
    predictions = np.append(predictions, scaler.inverse_transform(next_prediction), axis=0)
    last_input = np.append(last_input[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Original Price', color='blue')
plt.plot(range(time_step + train_size, len(predictions) + time_step + train_size), predictions, label='Predicted Price', color='orange')
plt.title('LSTM Price Prediction with Future Steps')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
