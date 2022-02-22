import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Callable
 
# The following code is copied (with small adaptations) from https://github.com/lasso-net/lassonet/blob/master/lassonet/prox.py
# Copyright of Louis Abraham, Ismael Lemhadri


def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)

def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)

def hier_prox(v, u, lambda_, lambda_bar, M):
    """
    v has shape (k,) or (k, d)
    u has shape (K,) or (K, d)
    
    standard case described in the paper: v has size (1,d), u has size (K,d)
    
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values
    k, d = u.shape
    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, d).to(u)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)
    x = F.relu(1 - a_s / norm_v) / (1 + s * M ** 2)
    w = M * x * norm_v
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    x_star = torch.gather(x, 0, idx).view(1, d)
    w_star = torch.gather(w, 0, idx).view(1, d)
    
    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star

#%% own implementation of LassoNet

class LassoNet(torch.nn.Module):
    def __init__(self, G: torch.nn.Module, lambda_: float=0.01, M: float=10, skip_bias: bool=False):
        """
        Implementation of LassoNet for arbitrary architecture. See https://jmlr.org/papers/volume22/20-848/20-848.pdf for details.

        Parameters
        ----------
        G : ``torch.nn.Module``
            The nonlinear part of LassoNet. Needs the following attributes:
                * ``self.W1`` : the linear layer applied to the inputs. This is called W^(1) in the LassoNet paper.
                * ``self.D_in`` : dimension of input
                * ``self.D_out`` : dimension of output
        lambda_ : float, optional
            Penalty parameter for the skip layer. The default is 1.
        M : float, optional
            Penalty parameter for the hierarchical constraint. The default is 1.
        skip_bias : boolean, optional
            Whether the skip connection has a bias.
        
        Returns
        -------
        None.

        """
        super().__init__()
        
        self.G = G
        self.lambda_ = lambda_
        self.M = M
        self.D_in = self.G.D_in
        self.D_out = self.G.D_out
        
        self.skip = torch.nn.Linear(self.D_in, self.D_out, bias = skip_bias) # skip connection aka theta
        return
    
    def forward(self, x):
        y1 = self.G(x)
        y2 = self.skip(x)
        return y1+y2
    
    def do_training(self, loss: torch.nn.Module, dl: DataLoader, opt: torch.optim.Optimizer=None, n_epochs: int=10, lr_schedule: torch.optim.lr_scheduler.StepLR=None,\
                    valid_ds: Dataset=None, preprocess: Callable=None, verbose: bool=True):
        """

        Parameters
        ----------
        loss : ``torch.nn`` loss function
            Loss function for the model.
        dl : ``torch.utils.data.DataLoader``
            DataLoader with the training data.
        opt : from ``torch.optim.Optimizer``, optional
            Pytorch optimizer. The default is SGD with Nesterov momentum and learning rate 0.001.
        n_epochs : int, optional
            Number of epochs for training. The default is 10.
        lr_schedule : from ``torch.optim.lr_scheduler.StepLR``, optional
            Learning rate schedule. Step is taken after each epoch. The default is None.
        valid_dl : ``torch.utils.data.Dataset``, optional
            Dataset for validation loss. The default is None.
        preprocess : function, optional
            A function for preprocessing the inputs for the model. The default is None.
        verbose : boolean, optional
            Verbosity. The default is True.

        Returns
        -------
        info : dict
            Training and validation loss and accuracy history. Each entry is the loss/accuracy averaged over one epoch.

        """
        if opt is None:
            opt = torch.optim.SGD(self.parameters(), lr = 0.001, momentum = 0.9, nesterov = True)
        
        if verbose:
            print(opt)    
        
        info = {'train_loss':[],'valid_loss':[],'train_acc':[],'valid_acc':[]}
        
        for j in np.arange(n_epochs):
            
            ################### SETUP FOR EPOCH ##################
            all_loss = list(); all_acc = list()
            all_vl_loss = list(); all_vl_acc = list()
            
            ################### START OF EPOCH ###################
            self.train()
            for inputs, targets in dl:
                #if preprocess is not None:
                #    inputs = preprocess(inputs)
                
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
                self.skip.weight.data, self.G.W1.weight.data = hier_prox(self.skip.weight.data, self.G.W1.weight.data,\
                                                                            lambda_=self.lambda_*alpha, lambda_bar=0, M = self.M)
                
                ## COMPUTE ACCURACY AND STORE 
                scores, predictions = torch.max(y_pred.data, 1)
                all_loss.append(loss_val.item())
                all_acc.append((predictions == targets).float().mean())
                
            
            ################### END OF EPOCH ###################
            if lr_schedule is not None:
                lr_schedule.step()    
                
            ### VALIDATION
            if valid_ds is not None:
                self.eval()
                               
                output = self.forward(valid_ds.data)
                this_vl_loss = loss(output, valid_ds.targets).item()
                v_scores, v_predictions = torch.max(output.data, 1)
                this_vl_acc = (v_predictions == valid_ds.targets).float().mean().item()
                
                
            ### STORE
            
            info['train_loss'].append(np.mean(all_loss))
            info['train_acc'].append(np.mean(all_acc))
            if valid_ds is not None:
                info['valid_loss'].append(this_vl_loss)
                info['valid_acc'].append(this_vl_acc)
            
            if verbose:
                print(f"================== Epoch {j+1}/{n_epochs} ================== ")
                print(f"\t  train loss: {np.mean(all_loss)}.")
                if valid_ds is not None:
                    print(f"\t  validation loss: {this_vl_loss}.")    
                print(opt)    
            
        return info
    