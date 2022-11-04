"""
LassoNet for a simple toy example.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from models import LassoNet


torch.manual_seed(42)
np.random.seed(42)

#%% Generate a simple dataset

D_in = 10 # input dimension
D_out = 1 # output dimension
H = 30 # hidden layer size

N_train = 5000 # training samples
N_test = 100 # test samples
batch_size = 32 

def generate_toy_example(N):
    X = torch.randn(N, D_in)  
    y = 2.*X[:,3] - 1.*X[:,3]**2 + 1.*X[:,1] + 0.5*X[:,2] + 2*X[:,4]*X[:,5]
    return X, y.reshape(-1,1)

#%% create Pytorch Dataset and DataLoader

X_train, Y_train = generate_toy_example(N_train)
X_test, Y_test = generate_toy_example(N_test)

    
train_set = TensorDataset(X_train, Y_train)
dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = TensorDataset(X_test, Y_test)

#%% Define non-linear part of LassoNet

"""
Define a simple Feed forward RelU network.
It is important, that we the first layer has attribute name ``W1``.
"""

from models.utils import FeedForward

l1 = 6.
M = 1.

G = FeedForward(D_in, D_out, H)
model = LassoNet(G, lambda_=l1, M=M, skip_bias=True)

loss = torch.nn.MSELoss(reduction='mean')

# We see that parameters of G are already included in the LassoNet object.
print(model)

#%% Training

n_epochs = 100
lr = 1e-3 # initial learning rate

opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True) # optimizer object
lr_schedule = StepLR(opt, step_size=20, gamma=0.5) # learning rate scheduler

info = list()

for j in np.arange(n_epochs): 
    
    # training
    epoch_info = model.train_epoch(loss, dl, opt)
    lr_schedule.step()
    
    # validation
    model.eval()
    train_loss = loss(model.forward(train_set.tensors[0]), train_set.tensors[1]).item()    
    test_loss = loss(model.forward(test_set.tensors[0]), test_set.tensors[1]).item()
    
    info.append({'train_loss': train_loss, 'test_loss': test_loss})
    
    print(f"epoch {j+1}: \t \t train loss={np.round(train_loss,4)},  \t \t test_loss={np.round(test_loss,4)}.")
    
#%% Plotting

info = pd.DataFrame(info)

fig, ax = plt.subplots()
ax.plot(info.train_loss, c='#002635', marker='o', label='Train loss')
ax.plot(info.test_loss, c='#002635', marker='o', ls='--', alpha=0.6, label='Test loss')
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.legend()
#fig.savefig('plots/example_loss.png', dpi = 400)

fig, ax = plt.subplots()
ax.imshow(G.W1.weight.data, cmap="coolwarm", vmin=-.1, vmax=.1)
fig.suptitle("First layer weight heatmap after training")
#fig.savefig('plots/example_weights.png', dpi = 400)


