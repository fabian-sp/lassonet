"""
adapted from this tutorial: https://github.com/ozx1812/MNIST-PyTorch/blob/master/mnist_mlp_pytorch.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.optim.lr_scheduler import StepLR

from conv_lassonet import ConvLassoNet

from torchvision import datasets
import torchvision.transforms as transforms

#%%
batch_size = 128

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 512, num_workers=0)


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()


#%% the actual model
l1 = 4.
M = 1.

model = ConvLassoNet(lambda_ = l1, M = M, D_in = (28,28), D_out = 10)

# test forward method, reshape input to vector with view
model.forward(images).size()

# params of G are already included in params of model!
for param in model.parameters():
    print(param.size())

#%%
n_epochs = 10
loss = torch.nn.CrossEntropyLoss()

alpha0 = 0.01

#%% training

opt = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
sched = StepLR(opt, step_size=1, gamma=0.5)

train_info = model.do_training(loss, train_loader, opt = opt, n_epochs = n_epochs, lr_schedule = None, valid_dl = test_loader,\
                               verbose = True)


#%% training of model without penalty

model2 = ConvLassoNet(lambda_ = None, M = 1., D_in = (28,28), D_out = 10)

opt2 = torch.optim.SGD(model2.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
sched2 = StepLR(opt2, step_size=1, gamma=0.5)

train_info2 = model2.do_training(loss, train_loader, opt = opt2, n_epochs = n_epochs, lr_schedule = None, valid_dl = test_loader,\
                                 verbose = True)


#%%    

# for param in model.parameters():
#     print(param.data)

fig, ax = plt.subplots()
ax.plot(train_info['train_loss'], c = '#002635', marker = 'o', label = 'Train loss ConvLassoNet')
ax.plot(train_info['valid_loss'], c = '#002635', marker = 'x', ls = '--', label = 'Validation loss ConvLassoNet')

ax.plot(train_info2['train_loss'], c = '#AB1A25', marker = 'o', label = 'Train loss Unconstrained')
ax.plot(train_info2['valid_loss'], c = '#AB1A25', marker = 'x', ls = '--', label = 'Validation loss Unconstrained')

ax.set_xlabel('Epoch')
ax.set_ylim(0,)
ax.legend()
fig.suptitle('Loss with and without LassoNet constraint')

#fig.savefig('plots/conv_loss.png', dpi = 400)
##013440
##D97925

#%% plot Conv2 filter norm

def plot_filter_norm(mod, ax, color):
    
    X = mod.conv2.weight.data.numpy()
    infnorm_ = np.linalg.norm(X, axis = (2,3)).T
    ax.plot(infnorm_, lw = 0.5, alpha = 0.2, c = color, marker = 'o')    

    return 

fig, ax = plt.subplots()
plot_filter_norm(model, ax, color = '#002635')
plot_filter_norm(model2, ax, color = '#AB1A25')

ax.set_ylabel('Max-norm of each filter channel')
ax.set_xlabel('Conv1 output channel')
ax.set_ylim(0,)

p1 = mpatches.Patch(color='#002635', label='ConvLassoNet')
p2 = mpatches.Patch(color='#AB1A25', label='Unconstrained')
ax.legend(handles=[p1,p2])

#fig.savefig('plots/conv2_filter_norm.png', dpi = 400)

#%% plot Conv1 filter weights

def plot_filter1(model, cmap = plt.cm.cividis, vmin = None, vmax = None):
    
    conv_filter = model.conv1.weight.data
    
    for j in np.arange(conv_filter.size(0)):
        ax = axs.ravel()[j]
        ax.imshow(conv_filter[j,0,:,:], cmap = cmap, vmin = vmin, vmax = vmax) 
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize = 8)

    return 

v_ = 0.5

fig, axs = plt.subplots(4,4)
plot_filter1(model, cmap=plt.cm.RdBu_r, vmin=-v_, vmax=v_)
#fig.savefig('plots/conv_filter.png', dpi = 400)

fig, axs = plt.subplots(4,4)
plot_filter1(model2, cmap=plt.cm.RdBu_r, vmin=-v_, vmax=v_)
#fig.savefig('plots/conv_filter_unc.png', dpi = 400)

#%% plot skip layer weights

def plot_skip(model, label, cmap = plt.cm.cividis, vmin = None, vmax = None):
    skip_weight = model.skip.weight.data.numpy().copy()
    T = skip_weight[label,:]
        
    for j in np.arange(model.out_channels1):
        ax = axs.ravel()[j]
        ax.imshow(T[model.h_out*model.w_out*j:model.h_out*model.w_out*(j+1)].reshape(model.h_out,model.w_out),\
                  cmap = cmap, vmin = vmin, vmax = vmax)
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize = 8)

    return

label = 8
v_ = 1e-3

fig, axs = plt.subplots(4,4)
fig.suptitle(f'Linear weights of convolution output for digit {label}')
plot_skip(model, label, cmap = plt.cm.cividis, vmin = -v_, vmax = v_)
#fig.savefig(f'plots/conv_skip_{label}.png', dpi = 400)
