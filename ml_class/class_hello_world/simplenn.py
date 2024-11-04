import torch
import torch.nn as nn
import torch.optim as optim


# 定义神经网络结构
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = self.fc2(x)  # 输出层
        return x


# 创建网络实例
input_size = 3  # 输入维度（例如 3 个特征）
hidden_size = 64  # 隐藏层大小
output_size = 2  # 输出维度（例如 2 个类别）
model = SimpleNN(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

# 假设我们有一些数据
data = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]], dtype=torch.float32)  # 输入数据
labels = torch.tensor([0, 1], dtype=torch.long)  # 标签

# 训练模型
for epoch in range(100):  # 迭代 100 次
    # 前向传播
    outputs = model(data)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
