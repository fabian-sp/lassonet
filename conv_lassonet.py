"""
implementation of LassoNet where the hierarchical penalty is applied to the convolutional filters.  

some snippets from: https://medium.com/dataseries/visualizing-the-feature-maps-and-filters-by-convolutional-neural-networks-e1462340518e
"""

import numpy as np
import torch
import torch.nn as nn

from module import hier_prox


class ConvLassoNet(nn.Module):
    def __init__(self, lambda_ = 1., M = 1., D_in = 784, D_out = 10):
        """
        LassoNet applied after a first layer of convolutions. See https://jmlr.org/papers/volume22/20-848/20-848.pdf for details.

        Parameters
        ----------
        lambda_ : float, optional
            Penalty parameter for the skip layer. The default is 1.
            By setting to ``None``, the LassoNet penalty is deactivated.
        M : float, optional
            Penalty parameter for the hierarchical constraint. The default is 1.
        D_in : int, optional
            input dimension of the model. The default is 784.
        D_out : int, optional
            output dimension of the model. The default is 10.

        Returns
        -------
        None.

        """
        
        super(ConvLassoNet, self).__init__()
        
        # LassoNet penalty
        self.lambda_ = lambda_
        self.M = M
        
        assert (self.lambda_ is None) or (self.lambda_ > 0), "lambda_ must be None or positive"
        assert self.M > 0, "M needs to be positive (possibly np.inf)"
        
        # hyperparameters
        self.kernel_size = 5
        self.out_channels = 32
        
        self.D_in = D_in
        self.D_out = D_out
        
        # first conv layer and skip layer
        self.conv1 = nn.Conv2d(1, self.out_channels, kernel_size=5, stride=1, padding=2)
        
        if self.lambda_ is not None:
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
        
        if self.lambda_ is not None:
            z1 = self.skip(out.reshape(-1, self.out_channels*self.m1*self.m2))
        else: 
            z1 = 0.
            
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

        Parameters
        ----------
        loss : ``torch.nn`` loss function
            Loss function for the model.
        dl : ``torch.utils.data.DataLoader``
            DataLoader with the training data.
        opt : from ``torch.optim``, optional
            Pytorch optimizer. The default is SGD with Nesterov momentum and learning rate 0.001.
        n_epochs : int, optional
            Number of epochs for training. The default is 10.
        lr_schedule : from ``torch.optim.lr_scheduler``, optional
            Learning rate schedule. Step is taken after each epoch. The default is None.
        valid_dl : ``torch.utils.data.DataLoader``, optional
            DataLoader for validation loss. One sample is taken at end of each epoch. The default is None.
        preprocess : function, optional
            A function for preprocessing the inputs for the model. The default is None.
        verbose : boolean, optional
            Verbosity. The default is True.

        Returns
        -------
        info : dict
            Training and validation loss and accuracy history.

        """
        if opt is None:
            opt = torch.optim.SGD(self.parameters(), lr = 0.001, momentum = 0.9, nesterov = True)
        
        if verbose:
            print(opt)    
        
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
                if self.lambda_ is not None:
                    self.prox(alpha)
                    
                print(loss_val.item())
            
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
            if valid_dl is not None:
                info['valid_loss'].append(v_loss.item())
                info['valid_acc'].append(v_correct.item())
            
            if verbose:
                print(f"Epoch {j+1}/{n_epochs}: \t  train loss: {loss_val.item()}, \t train accuracy: {acc.item()}.")
                print(opt)    
            
        return info
        



