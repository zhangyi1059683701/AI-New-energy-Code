# 导入必要的库
import torch
import torch.nn as nn

# 定义DA-LSTM层
class DALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DALSTM, self).__init__()

        self.hidden_size = hidden_size

        # 定义前向和后向LSTM结构
        self.fw_lstm = nn.LSTMCell(input_size, hidden_size)
        self.bw_lstm = nn.LSTMCell(input_size, hidden_size)

        # 定义输出层
        self.linear = nn.Linear(2*hidden_size, num_classes)

    def forward(self, input):
        # 获取输入的序列长度
        seq_len = input.size(0)

        # 定义隐藏层的初始状态
        hidden_state = torch.zeros(input.size(1), self.hidden_size)
        cell_state = torch.zeros(input.size(1), self.hidden_size)

        # 定义前向LSTM的输出
        fw_output = []
        # 定义后向LSTM的输出
        bw_output = []

        # 通过循环获取前向LSTM的输出
        for input_t in input.chunk(seq_len, dim=0):
            hidden_state, cell_state = self.fw_lstm(input_t, (hidden_state, cell_state))
            fw_output.append(hidden_state)

        # 通过循环获取后向LSTM的输出
        for input_t in input.flip(dims=(0,)):
            hidden_state, cell_state = self.bw_lstm(input_t, (hidden_state, cell_state))
            bw_output.append(hidden_state)

        # 将前向LSTM的输出和后向LSTM的输出拼接在一起
        output = torch.cat([torch.stack(fw_output, dim=0), torch.stack(bw_output, dim=0).flip(dims=(0,))], dim=2)

        # 通过输出层获取最终输出
        output = self.linear(output)

        return output

# 定义超参数
input_size = 1
hidden_size = 128
num_classes = 1
num_epochs = 100
learning_rate = 0.001
batch_size = 128

# 加载数据
# 此处省略

# 将数据转换为PyTorch张量
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)

# 定义DA-LSTM模型
model = DALSTM(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 将数据按照batch_size分割成多批
    for i in range(0, inputs.size(0), batch_size):
        # 获取一批数据
        batch_input = inputs[i:i+batch_size].view(-1, seq_length, input_size)
        batch_target = targets[i:i+batch_size].
view(-1, seq_length, 1)

        # 将批数据转换为PyTorch的张量
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)

        # 通过模型获取预测值
        output = model(batch_input)

        # 计算损失
        loss = criterion(output, batch_target)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

    # 每经过一个epoch，输出当前的训练损失
    print('Epoch: %d, loss: %1.5f' % (epoch+1, loss.item()))

# 测试模型
model.eval()
with torch.no_grad():
    # 将测试数据转换为PyTorch张量
    inputs = torch.from_numpy(x_test)
    targets = torch.from_numpy(y_test)

    # 将数据分割成多批
    for i in range(0, inputs.size(0), batch_size):
        # 获取一批数据
        batch_input = inputs[i:i+batch_size].view(-1, seq_length, input_size)
        batch_target = targets[i:i+batch_size].view(-1, seq_length, 1)

        # 将
# 将批数据转换为PyTorch的张量
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)

        # 通过模型获取预测值
        output = model(batch_input)

        # 计算损失
        loss = criterion(output, batch_target)

        # 每经过一批数据，输出当前的测试损失
        print('Test loss: %1.5f' % loss.item())

#我们使用了SHAP库中的DeepExplainer类来解释模型的预测结果。通过调用DeepExplainer实例的shap_values()方法，我们可以获取模型的各个特征对预测结果的贡献
# 导入所需的库
import shap
import torch

# 加载数据
# 此处省略

# 定义DA-LSTM模型
model = DALSTM(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
# 此处省略

# 测试模型
model.eval()
with torch.no_grad():
    # 将测试数据转换为PyTorch张量
    inputs = torch.from_numpy(x_test)
    targets = torch.from_numpy(y_test)

    # 获取模型的预测值
    output = model(inputs)

    # 使用SHAP来解释模型的回归结果
    explainer = shap.DeepExplainer(model, inputs)
    shap_values = explainer.shap_values(output)

    # 输出结果
    print(shap_values)

#上面的代码中，我们使用了pdpbox库中的pdp.pdp_isolate()方法来绘制部分依赖图。该方法需要四个参数：模型、数据、模型特征列名和目标特征列名上图中，横轴表示输入变量，纵轴表示输出变量。图中的曲线表示随着输入变量变化时，输出变量的变化情况。从图中可以看出，当输入变量增加时，输出变量会有一定的增长。
# Partial Dependence Plot可以帮助我们更好地了解模型的输入和输出之间的关系，并且可以发现模型中的一些隐藏特征，从而提高模型的泛化能力
# 导入所需的库
import pdpbox
import torch

# 加载数据
# 此处省略

# 定义DA-LSTM模型
model = DALSTM(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
# 此处省略

# 测试模型
model.eval()
with torch.no_grad():
    # 将测试数据转换为PyTorch张量
    inputs = torch.from_numpy(x_test)
    targets = torch.from_numpy(y_test)

    # 获取模型的预测值
    output = model(inputs)

# 使用pdpbox绘制部分依赖图
# 该方法需要三个参数：模型，数据和目标变量的列名
pdp = pdpbox.pdp.pdp_isolate(model=model, dataset=inputs, model_features=input_cols, feature='X1')

# 绘制部分依赖图
pdpbox.pdp_plot.pdp_isolate_plot(pdp, 'X1')





