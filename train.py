from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch


x=np.load("symm.npy", allow_pickle=True)
y=np.squeeze(np.load("energy.npy"))

x=np.reshape(x,(len(x),81))

X=np.expand_dims(x,axis=1)
Y=y.flatten()
print(X.shape,Y.shape)


dataset=TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
dataloader=DataLoader(dataset,batch_size=81,shuffle=True)


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

optim=torch.optim.Adam(Net.parameters(net),lr=0.001)
Loss=nn.MSELoss()



for epoch in range(1000):
    loss=None
    for batch_x,batch_y in dataloader:
        y_predict=net(batch_x)
        loss=Loss(y_predict,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    if (epoch+1)%100==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))

