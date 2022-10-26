from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt


x=np.load("symm.npy", allow_pickle=True)
y=np.squeeze(np.load("energy.npy"))

print(x.shape,y.shape)
x=np.reshape(x,(len(x),63))

X=np.expand_dims(x,axis=1)
X.astype(np.float32)
Y=y.flatten()
print(X.shape,Y.shape)


dataset=TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
dataloader=DataLoader(dataset,batch_size=81,shuffle=True)




"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(63, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = x.squeeze(-1)
        return x

net=Net()

"""
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(63,10),
            nn.ReLU(),
            nn.Linear(10,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Linear(10,1)
        )
    def forward(self, input:torch.FloatTensor):
        return self.net(input)

net=Net()
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(63,10),
            nn.ReLU(),
            nn.Linear(10,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Linear(10,1)
        )
    def forward(self, input:torch.FloatTensor):
        return self.net(input)
net=Net()



optim=torch.optim.Adam(Net.parameters(net),lr=0.001)
Loss=nn.MSELoss()


losse=[]
for epoch in range(10000):
    loss=None
    for batch_x,batch_y in dataloader:
        y_predict=net(batch_x)
        #for i in range(len(y_predict)):
        #    print(y_predict[i],batch_y[i],y_predict[i]-batch_y[i])
        #print(y_predict)
        loss=Loss(y_predict,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    if (epoch+1)%100==0:
        losse.append([epoch+1,loss.item()])
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))
G=[]
H=[]
for batch_x,batch_y in dataloader:
    y_predict=net(batch_x)
    for i in range(len(y_predict)):
        ypre=y_predict[i].detach().numpy()[0][0]+(2*0.5004966690+75.0637742413)
        real=batch_y[i].detach().numpy()+(2*0.5004966690+75.0637742413)
        H.append([real,ypre])
        delta=ypre-real
        G.append(abs(delta))
        print(ypre,real,delta)
print("MAE eV",np.mean(np.array(G))*27.21338)


plt.plot(np.array(losse)[:,0],np.array(losse)[:,1])
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("loas(MSE)")
plt.savefig("loss.png")


result = np.array(H)*627.51
plt.scatter(result[:,0], result[:,1])
print(result)
plt.xlabel("$E_{tot}-E_{atom}$(kcal/mol)")
plt.ylabel("Deleta E/(eV)")
plt.savefig("Delta.png")