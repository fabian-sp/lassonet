"""
adapted from this tutorial: https://github.com/ozx1812/MNIST-PyTorch/blob/master/mnist_mlp_pytorch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR

from module import LassoNet

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

class myG(torch.nn.Module):
    """
    2-layer NN with RelU
    """
    def __init__(self, D_in = 784, D_out = 10, H = 512):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.W1 = torch.nn.Linear(D_in, H, bias = True)
        self.relu = torch.nn.ReLU()
        self.W2 = torch.nn.Linear(H, H)
        self.W3 = torch.nn.Linear(H, D_out)
        return
    
    def forward(self, x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        x = self.relu(x)
        x = self.W3(x)
        return x
    
#%% the actual model

l1 = 5.
M = 1.

G = myG()
model = LassoNet(G, lambda_ = l1, M = M)

loss_fn = torch.nn.CrossEntropyLoss()

# test forward method, reshape input to vector with view
G(images.view(-1,28*28))
model.forward(images.view(-1,28*28))

# params of G are already included in params of model!
for param in model.parameters():
    print(param.size())
    
#%% training

alpha0 = 1e-3

#optimizer = torch.optim.Adam(model.parameters(), lr=alpha0)
optimizer = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)

scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

prep = lambda x: x.reshape(-1,28*28)

all_loss = model.do_training(loss_fn, train_loader, opt = optimizer, lr_schedule = scheduler, n_epochs = 20,\
                             preprocess = prep, verbose = True)

for param in model.parameters():
    print(param.data)

print("theta: ", model.skip.weight.data)

#%%
plt.figure()
plt.plot(all_loss)

plt.figure()
plt.imshow(G.W1.weight.data, cmap = "coolwarm")


#importance = model.skip.weight.data.mean(dim=0).view(28,28).numpy()
importance = model.skip.weight.data[7,:].view(28,28).numpy()
plt.figure()
plt.imshow(importance, cmap = "coolwarm")#, vmin = -0.0001, vmax = 0.0001)

