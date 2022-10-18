from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch

# 准备数据
x=np.load("symm.npy", allow_pickle=True)
y=np.squeeze(np.load("energy.npy"))

x=np.reshape(x,(len(x),81))

X=np.expand_dims(x,axis=1)
Y=y.flatten()
print(X.shape,Y.shape)

# 使用批训练方式
dataset=TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
dataloader=DataLoader(dataset,batch_size=81,shuffle=True)

# 神经网络主要结构，这里就是一个简单的线性结构

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(81, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        x = x.squeeze(-1)
        return x

net=Net()

# 定义优化器和损失函数
optim=torch.optim.Adam(Net.parameters(net),lr=0.001)
Loss=nn.MSELoss()


# 下面开始训练：
# 一共训练 1000次
for epoch in range(1000):
    loss=None
    for batch_x,batch_y in dataloader:
        y_predict=net(batch_x)
        loss=Loss(y_predict,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # 每100次 的时候打印一次日志
    if (epoch+1)%100==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))

"""
# 使用训练好的模型进行预测
predict=net(torch.tensor(X,dtype=torch.float))
# 绘图展示预测的和真实数据之间的差异
import matplotlib.pyplot as plt
plt.plot(x,y,label="fact")
plt.plot(x,predict.detach().numpy(),label="predict")
plt.title("sin function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
#plt.savefig(fname="result.png",figsize=[10,10])
#plt.show()
"""