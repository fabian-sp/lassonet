"""
LassoNet example for the MNIST dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms

from models import LassoNet
from models.utils import eval_on_dataloader

#%% prepare Dataset and DataLoader

batch_size = 128

# convert data to torch.FloatTensor, Normalize, and reshape to (784,)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                transforms.Lambda(lambda x: x.view(-1).view(784))
                              ])
                               

# choose the training and test datasets
train_set = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)
test_set = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

# prepare data loaders
train_dl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                       drop_last=True, shuffle=True)

test_dl = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                      drop_last=True, shuffle=True)

# obtain one batch of training images
dataiter = iter(train_dl)
images, labels = dataiter.next()

    
#%% Define LassoNet model with 2layer-FeedForward

from models.utils import FeedForward

l1 = 0.1
M = 10.

G = FeedForward(D_in=784, D_out=10, H=512)
model = LassoNet(G, lambda_ = l1, M = M)

loss = torch.nn.CrossEntropyLoss() # loss function for multiclass classification

# test forward method, reshape input to vector with view
G(images.view(-1,28*28))
model.forward(images.view(-1,28*28))

print(model)

#%% Training

n_epochs = 5
lr = 1e-2 # initial learning rate

opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True) # optimizer object
lr_schedule = StepLR(opt, step_size=10, gamma=0.5) # learning rate scheduler

info = list()

print("Epoch | \t train loss \t | test loss")
for j in np.arange(n_epochs): 
    
    # training
    epoch_info = model.train_epoch(loss, train_dl, opt)
    lr_schedule.step()
    
    # validation
    model.eval()
    train_loss = eval_on_dataloader(model, loss, train_dl)
    test_loss = eval_on_dataloader(model, loss, test_dl)
    
    info.append({'train_loss': train_loss, 'test_loss': test_loss})
    
    print(f"{j+1} \t \t  {np.round(train_loss,4)}  \t \t {np.round(test_loss,4)}.")
 
     

#%% Plotting

info = pd.DataFrame(info)

fig, ax = plt.subplots()
ax.plot(info.train_loss, c='#002635', marker='o', label='Train loss')
ax.plot(info.test_loss, c='#002635', marker='o', ls='--', alpha=0.6, label='Test loss')
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.legend()

fig, ax = plt.subplots()
ax.imshow(G.W1.weight.data, cmap="coolwarm", vmin=-.05, vmax=.05)
fig.suptitle("First layer weight heatmap after training")

