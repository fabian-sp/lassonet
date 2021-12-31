"""
implementation of LassoNet where the hierarchical penalty is applied to the convolutional filters.  

some snippets from: https://medium.com/dataseries/visualizing-the-feature-maps-and-filters-by-convolutional-neural-networks-e1462340518e
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
    def __init__(self, lambda_ = 1., M = 1., D_in = 784, D_out = 10):
        
        super(ConvLassoNet, self).__init__()
        
        # LassoNet penalty
        self.lambda_ = lambda_
        self.M = M
        
        assert self.lambda_ > 0, "lambda_ needs to be positive"
        assert self.M > 0, "M needs to be positive (possibly np.inf)"
        
        # hyperparameters
        self.kernel_size = 5
        self.out_channels = 32
        
        self.D_in = D_in
        self.D_out = D_out
        
        # first conv layer and skip layer
        self.conv1 = nn.Conv2d(1, self.out_channels, kernel_size=5, stride=1, padding=2)
        self.skip = nn.Linear(self.out_channels*self.D_in, self.D_out)
        
        # remaining nonlinear part (after conv1)
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
        
        return
        
    def forward(self, x):
        out = self.conv1(x)
        self.m1 = out.size(2)
        self.m2 = out.size(3)
        z1 = self.skip(out.reshape(-1, self.out_channels*self.m1*self.m2))
        
        # rest of non-linear part: can be adapted freely
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        z2 = self.fc2(out)
        return z1+z2
    
    def prox(self, lr):
        # loop through output channels / filters
        for j in range(self.out_channels):
            theta_j = self.skip.weight.data[:,self.m1*self.m2*j:self.m1*self.m2*(j+1)].reshape(-1)
            filter_j = self.conv1.weight[j,0,:,:].reshape(-1)
            
            theta_j, filter_j = hier_prox(theta_j, filter_j, lambda_=self.lambda_*lr, lambda_bar=0, M = self.M)
            
            self.skip.weight.data[:,self.m1*self.m2*j:self.m1*self.m2*(j+1)] = theta_j.reshape(self.D_out, -1)
            self.conv1.weight.data[j,0,:,:] = filter_j.reshape(self.kernel_size, self.kernel_size) #replace with self.kernel_size
                      
        return
    
    def do_training(self, loss, dl, opt = None, n_epochs = 10, lr_schedule = None, valid_dl = None,\
                    preprocess = None, verbose = True):
        """
        """
        if opt is None:
            opt = torch.optim.SGD(self.parameters(), lr = 0.001, momentum = 0.9, nesterov = True)
        
        info = {'train_loss':[],'valid_loss':[],'train_acc':[],'valid_acc':[]}
        
        if valid_dl is not None:
            valid_iter = iter(valid_dl)
            assert len(valid_iter) >= n_epochs, "Validation DataLoader needs to have more items than number of epochs."

        for j in np.arange(n_epochs):
              
            ################### START OF EPOCH ###################
            self.train()
            for inputs, targets in dl:
                if preprocess is not None:
                    inputs = preprocess(inputs)
                
                # forward pass
                y_pred = self.forward(inputs)
                # compute loss
                loss_val = loss(y_pred, targets)           
                # zero gradients
                opt.zero_grad()    
                # backward pass
                loss_val.backward()    
                # iteration
                opt.step()
                # step size
                alpha = opt.state_dict()['param_groups'][0]['lr']
                # prox step
                if self.M < np.inf:
                    self.prox(alpha)
            
                print(f"Epoch {j+1}/{n_epochs}: \t  train loss: {loss_val.item()}")
            
            ################### END OF EPOCH ###################
            if lr_schedule is not None:
                lr_schedule.step()    
                          
            ### VALIDATION
            if valid_dl is not None:
                self.eval()
                v_inputs, v_targets = valid_iter.next()                
                output = self.forward(v_inputs)
                v_loss = loss(output, v_targets)
                v_scores, v_predictions = torch.max(output.data, 1)
                v_correct = (v_predictions == v_targets).float().mean()
            
            ### STORE
            scores, predictions = torch.max(y_pred.data, 1)
            acc = (predictions == targets).float().mean()
            
            info['train_loss'].append(loss_val.item())
            info['train_acc'].append(acc.item())
            info['valid_loss'].append(v_loss.item())
            info['valid_acc'].append(v_correct.item())
            
            if verbose:
                print(f"Epoch {j+1}/{n_epochs}: \t  train loss: {loss_val.item()}, \t train accuracy: {acc.item()}.")
                
            
        return info
        

        
#%% TRAINING

# n_epoch = 1
# alpha0 = 1e-3

# all_loss = list()
# #optimizer = torch.optim.Adam(model.parameters(), lr=alpha0)
# optimizer = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)

# scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

# for j in np.arange(n_epoch):
#     print(f"EPOCH {j}")
#     for images, labels in train_loader:
            
#         # forward pass
#         y_pred = model.forward(images)    
#         # compute loss.
#         loss = loss_fn(y_pred, labels)           
#         # zero gradients
#         optimizer.zero_grad()    
#         # backward pass
#         loss.backward()    
#         # iteration
#         optimizer.step()
#         # step size
#         alpha = optimizer.state_dict()['param_groups'][0]['lr']
#         # prox step
#         model.prox(alpha)
        
#     # decrease step size
#     if j%10 ==0:
#         scheduler.step()
    
#     print("loss:", loss.item())
#     all_loss.append(loss.item())


#%% reshaping
#batch,out_channels,m1,m2
# H=torch.randn(3, 5, 4, 4)

# h=H.reshape(3, -1)

# j = 2

# H[0,j,:,:]

# h[0, 16*j:16*(j+1)].reshape(4,4)

