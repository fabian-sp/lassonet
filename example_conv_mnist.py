"""
This is experimental code!!

We use snippets from this tutorial: https://github.com/ozx1812/MNIST-PyTorch/blob/master/mnist_mlp_pytorch.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
import torchvision.transforms as transforms

from models.conv_lassonet import ConvLassoNet
from models.utils import eval_on_dataloader

#%%
batch_size = 128

transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

# prepare data loaders
train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=True)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, drop_last=True)

# obtain one batch of training images
dataiter = iter(test_dl)
images, labels = dataiter.next()

# loss function
loss = torch.nn.CrossEntropyLoss()

#%% the actual model

# LassoNet penalty parameters
l1 = 3.
M = 1.

# (28,28) images, 10 classes
model = ConvLassoNet(lambda_=l1, M=M, D_in=(28,28), D_out=10)

# test forward method, reshape input to vector with view
model.forward(images).size()

print(model)

#%% training

def train_model(model, opt, loss, lr_schedule=None, n_epochs=5):
    
    info = list()
    for j in range(n_epochs): 
        
        # training
        epoch_info = model.train_epoch(loss, train_dl, opt=opt)
        
        if lr_schedule is not None:
            lr_schedule.step()
        
        # validation
        model.eval()
        train_loss = eval_on_dataloader(model, loss, train_dl)
        test_loss = eval_on_dataloader(model, loss, test_dl)
        
        info.append({'train_loss': train_loss, 'test_loss': test_loss})
        
        print(f"epoch {j+1}: \t \t train loss={np.round(train_loss,4)},  \t \t test_loss={np.round(test_loss,4)}.")

    return info

#%% train ConvLassoNet

n_epochs = 10
lr = 1e-2

opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
sched = StepLR(opt, step_size=1, gamma=0.9)
    
info = train_model(model, opt, loss, lr_schedule=sched, n_epochs=n_epochs)

info = pd.DataFrame(info)
 
#%% training the same architecture without LassoNet penalty

model_ = ConvLassoNet(lambda_=None, M=M, D_in=(28,28), D_out=10)

opt_ = torch.optim.SGD(model_.parameters(), lr=lr, momentum=0.9, nesterov=True)
sched_ = StepLR(opt_, step_size=1, gamma=0.9)

info_ = train_model(model_, opt_, loss, lr_schedule=sched_, n_epochs=n_epochs)
info_ = pd.DataFrame(info_)

#%% plot loss

fig, ax = plt.subplots(figsize=(6,5))


ax.plot(info['train_loss'], c='#002635', marker='o', label='Train loss ConvLassoNet')
ax.plot(info['test_loss'], c='#002635', marker='x', ls='--', label='Test loss ConvLassoNet')

ax.plot(info_['train_loss'], c='#AB1A25', marker='o', label='Train loss unconstrained')
ax.plot(info_['test_loss'], c='#AB1A25', marker='x', ls='--', label='Test loss unconstrained')

ax.set_xlabel('Epoch')
ax.set_ylim(0,)
ax.legend()

fig.suptitle('Loss with and without LassoNet constraint')

#fig.savefig('plots/conv_loss.png', dpi = 400)
##013440
##D97925

#%% plot Conv2 filter norm

def plot_filter_norm(model, ax, color):
    """plot norm per filter
    """
    X = model.conv2.weight.data.numpy()
    norm_ = np.linalg.norm(X, axis = (2,3)).T
    ax.plot(norm_, lw=0.5, alpha=0.2, c=color, marker='o')    

    return 

fig, ax = plt.subplots()
plot_filter_norm(model, ax, color = '#002635')
plot_filter_norm(model_, ax, color = '#AB1A25')

ax.set_ylabel('Norm of each filter channel')
ax.set_xlabel('Conv1 output channel')
ax.set_ylim(0,)

p1 = mpatches.Patch(color='#002635', label='ConvLassoNet')
p2 = mpatches.Patch(color='#AB1A25', label='unconstrained')
ax.legend(handles=[p1,p2])

#fig.savefig('plots/conv2_filter_norm.png', dpi = 400)

#%% plot Conv1 filter weights

def plot_filter1(model, cmap=plt.cm.cividis, vmin=None, vmax=None):
    """ plot convolutional filter weights
    """
    conv_filter = model.conv1.weight.data
    
    for j in np.arange(conv_filter.size(0)):
        ax = axs.ravel()[j]
        ax.imshow(conv_filter[j,0,:,:], cmap=cmap, vmin=vmin, vmax=vmax) 
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize=8)

    return 

v_ = 1.

fig, axs = plt.subplots(4,4)
plot_filter1(model, cmap=plt.cm.RdBu_r, vmin=-v_, vmax=v_)
fig.suptitle('Filter weights ConvLassoNet')
#fig.savefig('plots/conv_filter.png', dpi = 400)

fig, axs = plt.subplots(4,4)
plot_filter1(model_, cmap=plt.cm.RdBu_r, vmin=-v_, vmax=v_)
fig.suptitle('Filter weights unconstrained')
#fig.savefig('plots/conv_filter_unc.png', dpi = 400)

#%% plot skip layer weights

def plot_skip(model, label, cmap=plt.cm.cividis, vmin=None, vmax=None):
    """ plot skip layer weights
    """
    skip_weight = model.skip.weight.data.numpy().copy()
    T = skip_weight[label,:]
        
    for j in np.arange(model.out_channels1):
        ax = axs.ravel()[j]
        ax.imshow(T[model.h_out*model.w_out*j:model.h_out*model.w_out*(j+1)].reshape(model.h_out,model.w_out),
                  cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize = 8)

    return

label = 8 # for which digit
v_ = 1e-3

fig, axs = plt.subplots(4,4)
fig.suptitle(f'Linear weights of convolution output for digit {label}')
plot_skip(model, label, cmap=plt.cm.cividis, vmin=-v_, vmax=v_)
#fig.savefig(f'plots/conv_skip_{label}.png', dpi = 400)
