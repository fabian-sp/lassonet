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
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 200, num_workers=0)


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()


#%% the actual model
l1 = 5.
M = 10.

model = ConvLassoNet(lambda_ = l1, M = M, D_in = 784, D_out = 10)

# test forward method, reshape input to vector with view
model.forward(images).size()

# params of G are already included in params of model!
for param in model.parameters():
    print(param.size())

#%%
n_epochs = 2
loss = torch.nn.CrossEntropyLoss()

alpha0 = 0.001

#%% training

opt = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
sched = StepLR(opt, step_size=1, gamma=0.5)

train_info = model.do_training(loss, train_loader, opt = opt, n_epochs = n_epochs, lr_schedule = sched, valid_dl = test_loader,\
                               verbose = True)


#%% training of model without penalty

model2 = ConvLassoNet(lambda_ = None, M = 1., D_in = 784, D_out = 10)

opt = torch.optim.SGD(model2.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
sched = StepLR(opt, step_size=1, gamma=0.5)

train_info2 = model2.do_training(loss, train_loader, opt = opt, n_epochs = n_epochs, lr_schedule = sched, valid_dl = test_loader,\
                                 verbose = True)


#%%    

# for param in model.parameters():
#     print(param.data)

fig, ax = plt.subplots()
ax.plot(train_info['train_loss'], c = '#002635', markerstyle = 'o', label = 'Training loss')
ax.plot(train_info['valid_loss'], c = '#002635', markerstyle = 'x', ls = '--', label = 'Validation loss')

ax.plot(train_info2['train_loss'], c = '#AB1A25', markerstyle = 'o', label = 'Training loss (uncon.)')
ax.plot(train_info2['valid_loss'], c = '#AB1A25', markerstyle = 'x', ls = '--', label = 'Validation loss (uncon.)')

ax.set_xlabel('Epoch')
ax.legend()



#%% plot Conv filter weights

def plot_filter(model, axs = None, cmap = plt.cm.cividis, vmin = -0.5, vmax = 0.5):
    
    conv_filter = model.conv1.weight.data
    if axs is None:
        fig, axs = plt.subplots(4,8)
    
    for j in np.arange(conv_filter.size(0)):
        ax = axs.ravel()[j]
        ax.imshow(conv_filter[j,0,:,:], cmap = cmap, vmin = vmin, vmax = vmax)
        ax.axis('off')
        ax.set_title(f'Filter {j}')

    return fig


fig = plot_filter(model2, axs = None, cmap = plt.cm.cividis, vmin = -0.5, vmax = 0.5)

#%% plot skip layer weights





