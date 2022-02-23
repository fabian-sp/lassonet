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

# convert data to torch.FloatTensor (including scaling to [0,1])
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)


# filter on 1,0 digits
#ixx = (dataset.targets == 0) | (dataset.targets == 1) 
#ds.data, ds.targets = ds.data[ixx], dss.targets[ixx]

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 512, num_workers=0)


# obtain one batch of training images
dataiter = iter(test_loader)
images, labels = dataiter.next()

loss = torch.nn.CrossEntropyLoss()

#%% the actual model
l1 = 3.
M = 1.

model = ConvLassoNet(lambda_ = l1, M = M, D_in = (28,28), D_out = 10)

# test forward method, reshape input to vector with view
model.forward(images).size()

# params of G are already included in params of model!
for param in model.parameters():
    print(param.size())

#%% training

def train_model(model, opt, loss, lr_schedule=None, n_epochs = 5):
    
    loss_hist = {'train_loss':[], 'valid_loss':[], 'train_acc':[], 'valid_acc':[]}
    
    for j in range(n_epochs): 
        print(f"================== Epoch {j+1}/{n_epochs} ================== ")
        print(opt)  
        
        ### TRAINING
        epoch_info = model.train_epoch(loss, train_loader, opt=opt)
        loss_hist['train_loss'].append(np.mean(epoch_info['train_loss']))
        loss_hist['train_acc'].append(np.mean(epoch_info['train_acc']))
        
        if lr_schedule is not None:
            lr_schedule.step()
        
        ### VALIDATION
        valid_loss = 0; valid_acc = 0;
        model.eval()
        for inputs, targets in test_loader:
            output = model.forward(inputs)          
            valid_loss += loss(output, targets).item()
            _, predictions = torch.max(output.data, 1)
            valid_acc += (predictions == targets).float().mean().item()
        
        loss_hist['valid_loss'].append(valid_loss/len(test_loader))
        loss_hist['valid_acc'].append(valid_acc/len(test_loader))
                 
        print(f"\t  train loss: {np.mean(epoch_info['train_loss'])}.")
        print(f"\t  validation loss: {valid_loss/len(test_loader)}.")    

    return loss_hist

#%% train ConvLassoNet

n_epochs = 10
alpha0 = 1e-2

opt = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
sched = StepLR(opt, step_size=1, gamma=0.9)
    
hist = train_model(model, opt, loss, lr_schedule=sched, n_epochs=n_epochs)
 
#%% training of model without penalty

model_ = ConvLassoNet(lambda_ = None, M=M, D_in=(28,28), D_out=10, out_channels1=5)

opt_ = torch.optim.SGD(model_.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
sched_ = StepLR(opt_, step_size=1, gamma=0.9)

hist_ = train_model(model_, opt_, loss, lr_schedule=sched_, n_epochs=n_epochs)


#%%    

# for param in model.parameters():
#     print(param.data)

fig, axs = plt.subplots(1,2)

ax = axs[0]
ax.plot(hist['train_loss'], c = '#002635', marker = 'o', label = 'Train loss ConvLassoNet')
ax.plot(hist['valid_loss'], c = '#002635', marker = 'x', ls = '--', label = 'Validation loss ConvLassoNet')

ax.plot(hist_['train_loss'], c = '#AB1A25', marker = 'o', label = 'Train loss unconstrained')
ax.plot(hist_['valid_loss'], c = '#AB1A25', marker = 'x', ls = '--', label = 'Validation loss unconstrained')

ax.set_xlabel('Epoch')
ax.set_ylim(0,)
ax.legend()

ax = axs[1]
ax.plot(hist['train_acc'], c = '#002635', marker = 'o', label = 'Train accuracy ConvLassoNet')
ax.plot(hist['valid_acc'], c = '#002635', marker = 'x', ls = '--', label = 'Validation accuracy ConvLassoNet')

ax.plot(hist_['train_acc'], c = '#AB1A25', marker = 'o', label = 'Train accuracy unconstrained')
ax.plot(hist_['valid_acc'], c = '#AB1A25', marker = 'x', ls = '--', label = 'Validation accuracy unconstrained')

ax.set_xlabel('Epoch')
ax.set_ylim(0.9,1)
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
plot_filter_norm(model_, ax, color = '#AB1A25')

ax.set_ylabel('Max-norm of each filter channel')
ax.set_xlabel('Conv1 output channel')
ax.set_ylim(0,)

p1 = mpatches.Patch(color='#002635', label='ConvLassoNet')
p2 = mpatches.Patch(color='#AB1A25', label='unconstrained')
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
plot_skip(model, label, cmap=plt.cm.cividis, vmin=-v_, vmax=v_)
#fig.savefig(f'plots/conv_skip_{label}.png', dpi = 400)
