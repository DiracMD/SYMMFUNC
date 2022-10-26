from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
import random
"""

x=np.load("symm.npy", allow_pickle=True)
y=np.squeeze(np.load("energy.npy"))

print(x.shape,y.shape)
x=np.reshape(x,(len(x),63))

X=np.expand_dims(x,axis=1)
Y=y.flatten()
print(X.shape,Y.shape)

"""

x_train_list = []
y_train_list = []
for i in range(1, 50):
    x = i*random.choice([0.7,0.8,0.9])
    y = np.sin(i)+np.random.choice(1,1)
    x_train_list.append(["%.2f" % x])
    y_train_list.append(["%.2f" % y])
 
x_train = np.array(x_train_list, dtype=np.float32) #将数据列表转为np.array
y_train = np.array(y_train_list, dtype=np.float32)
X = torch.from_numpy(x_train)
Y = torch.from_numpy(y_train)

print(X)
print(Y)

dataset=TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
dataloader=DataLoader(dataset,batch_size=81,shuffle=True)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_features=1,out_features=10),
            nn.ReLU(),
            nn.Linear(10,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
    def forward(self, input:torch.FloatTensor):
        return self.net(input)

net=Net()


optim=torch.optim.Adam(Net.parameters(net),lr=0.001)
Loss=nn.MSELoss()


for epoch in range(1000):
    loss=None
    for batch_x,batch_y in dataloader:
        y_predict=net(batch_x)
        #print(y_predict)
        loss=Loss(y_predict,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    if (epoch+1)%100==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))
    