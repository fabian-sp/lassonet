"""
adapted from this tutorial: https://github.com/ozx1812/MNIST-PyTorch/blob/master/mnist_mlp_pytorch.py
"""

import numpy as np
import matplotlib.pyplot as plt
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
M = 20.

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
ax.plot(train_info['train_loss'], c = '#002635', marker = 'o', label = 'Training loss')
ax.plot(train_info['valid_loss'], c = '#002635', marker = 'x', ls = '--', label = 'Validation loss')

ax.plot(train_info2['train_loss'], c = '#AB1A25', marker = 'o', label = 'Training loss (uncon.)')
ax.plot(train_info2['valid_loss'], c = '#AB1A25', marker = 'x', ls = '--', label = 'Validation loss (uncon.)')

ax.set_xlabel('Epoch')
ax.legend()

##013440
##D97925

#%% plot Conv filter weights

def plot_filter(model, cmap = plt.cm.cividis, vmin = None, vmax = None):
    
    conv_filter = model.conv1.weight.data
    
    for j in np.arange(conv_filter.size(0)):
        ax = axs.ravel()[j]
        ax.imshow(conv_filter[j,0,:,:], cmap = cmap, vmin = vmin, vmax = vmax)
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize = 8)

    return 

fig, axs = plt.subplots(4,8)
plot_filter(model, cmap=plt.cm.viridis, vmin=-0.3, vmax=0.3)

#%% plot skip layer weights


def plot_skip(model, label, cmap = plt.cm.cividis, vmin = None, vmax = None):
    skip_weight = model.skip.weight.data.numpy().copy()
    T = skip_weight[label,:]
        
    for j in np.arange(model.out_channels):
        ax = axs.ravel()[j]
        ax.imshow(T[model.h_out*model.w_out*j:model.h_out*model.w_out*(j+1)].reshape(model.h_out,model.w_out), cmap = cmap)
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize = 8)

    return

label = 8

fig, axs = plt.subplots(4,8)
fig.suptitle(f'Linear weights of convolution output for digit {label}')
plot_skip(model, label, cmap = plt.cm.cividis, vmin = None, vmax = None)
