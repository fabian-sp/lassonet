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
M = 1.

model = ConvLassoNet(lambda_ = l1, M = M, D_in = 784, D_out = 10)

loss = torch.nn.CrossEntropyLoss()
# test forward method, reshape input to vector with view
model.forward(images).size()


# params of G are already included in params of model!
for param in model.parameters():
    print(param.size())

#%%
alpha0 = 0.001
n_epochs = 10

opt = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)

sched = StepLR(opt, step_size=1, gamma=0.5)

train_info = model.do_training(loss, train_loader, opt = opt, n_epochs = n_epochs, lr_schedule = sched, valid_dl = test_loader,\
                  verbose = True)


#%%    

for param in model.parameters():
    print(param.data)

print("theta: ", model.skip.weight.data)



#%%
# Plot Conv filter weights
conv_filter = model.conv1.weight.data
fig, axs = plt.subplots(4,8)

for j in np.arange(conv_filter.size(0)):
    ax = axs.ravel()[j]
    ax.imshow(conv_filter[j,0,:,:], cmap = plt.cm.cividis, vmin = -0.5, vmax = 0.5)
    ax.axis('off')

plt.show()
