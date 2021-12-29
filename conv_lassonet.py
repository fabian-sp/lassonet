"""
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from module import hier_prox

from torchvision import datasets
import torchvision.transforms as transforms


#%%
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers=0)


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

#%%

class ConvLassoNet(nn.Module):
    def __init__(self, lambda_ = 1., M = 1., D_in = 784, D_out = 10, out_channels = 32):
        
        super(ConvLassoNet, self).__init__()
        
        self.D_in = D_in
        self.D_out = D_out
        self.out_channels = out_channels # number of output channels of first convolutional layer
        
        self.conv1 = nn.Conv2d(1, self.out_channels, kernel_size=5, stride=1, padding=2)
        self.skip = nn.Linear(self.out_channels*self.D_in, self.D_out)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.drop_out = nn.Dropout()
        # downsampling twice by factor 2 --> 7x7 output, 64 channels
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000, self.D_out)
        
    def forward(self, x):
        out = self.conv1(x)
        z1 = self.skip(out.view(-1, self.out_channels*self.D_in))
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        z2 = self.fc2(out)
        return z1+z2

    
#%% the actual model

l1 = 5.
M = 1.

model = ConvLassoNet(lambda_ = l1, M = M, D_in = 784, D_out = 10, out_channels = 32)

loss_fn = torch.nn.CrossEntropyLoss()

# test forward method, reshape input to vector with view
model.forward(images).size()


# params of G are already included in params of model!
for param in model.parameters():
    print(param.size())
    
#%% TRAINING

n_epoch = 20
alpha0 = 1e-3

all_loss = list()

#optimizer = torch.optim.Adam(model.parameters(), lr=alpha0)
optimizer = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)

scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

for j in np.arange(n_epoch):
    print(f"EPOCH {j}")
    for data, target in train_loader:
            
        # forward pass
        y_pred = model.forward(data.view(-1,28*28))    
        # compute loss.
        loss = loss_fn(y_pred, target)           
        # zero gradients
        optimizer.zero_grad()    
        # backward pass
        loss.backward()    
        # iteration
        optimizer.step()
        # step size
        alpha = optimizer.state_dict()['param_groups'][0]['lr']
        # prox step
        model.skip.weight.data, model.G.W1.weight.data = hier_prox(model.skip.weight.data, model.G.W1.weight.data,\
                                                                   lambda_=model.lambda_*alpha, lambda_bar=0, M = model.M)
        
    # decrease step size
    if j%10 ==0:
        scheduler.step()
    
    print("loss:", loss.item())
    all_loss.append(loss.item())
    
for param in model.parameters():
    print(param.data)

print("theta: ", model.skip.weight.data)

plt.figure()
plt.plot(all_loss)

plt.figure()
plt.imshow(G.W1.weight.data, cmap = "coolwarm")


importance = model.skip.weight.data.mean(dim=0).view(28,28).numpy()
importance = model.skip.weight.data[3,:].view(28,28).numpy()
plt.figure()
plt.imshow(importance, cmap = "coolwarm")#, vmin = -0.0001, vmax = 0.0001)

