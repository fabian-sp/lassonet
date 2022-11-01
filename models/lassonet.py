"""
@author: Fabian Schaipp

Implementation of the LassoNet module.
"""
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import warnings
 
#%% The code of this section is copied (with small adaptations) from https://github.com/lasso-net/lassonet/blob/master/lassonet/prox.py
# Copyright of Louis Abraham, Ismael Lemhadri

def soft_threshold(l: float, x: torch.Tensor) -> torch.Tensor:
    """
    Soft-thresholding operator.
    
    Parameters
    ----------
    l : float
        Threshold value.
    x : torch.Tensor
        Input argument.

    Returns
    -------
    torch.Tensor
        Output of soft-thresholding.
        

    """
    return torch.sign(x) * torch.relu(torch.abs(x) - l)

def sign_binary(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)

def hier_prox(v: torch.Tensor, u: torch.Tensor, lambda_: float, lambda_bar: float, M: float):
    """
    Copied (with minor adaptions) from:
        Louis Abraham, Ismael Lemhadri: https://github.com/lasso-net/lassonet/blob/master/lassonet/prox.py
    
    
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

#%% Our own implementation

class LassoNet(torch.nn.Module):
    def __init__(self, G: torch.nn.Module, lambda_: float=0.01, M: float=10, skip_bias: bool=False):
        """
        Implementation of LassoNet for an arbitrary module G. 
        See https://jmlr.org/papers/volume22/20-848/20-848.pdf for details.

        Parameters
        ----------
        G : torch.nn.Module
            The nonlinear part of LassoNet. Needs the following attributes:
                * ``self.W1`` : the linear layer applied to the inputs. This is called W^(1) in the LassoNet paper.
                * ``self.D_in`` : input dimension of ``G``.
                * ``self.D_out`` : output dimension of ``G``.
        lambda_ : float, optional
            Penalty parameter for the (linear) skip layer. The default is 0.01.
        M : float, optional
            Penalty parameter for the hierarchical constraint. The default is 10.
        skip_bias : boolean, optional
            Whether the skip connection has a bias.
        
        """
        super().__init__()
        
        self.G = G
        
        assert hasattr(G, 'W1'), "The first linear layer of G needs to be called G.W1."
        assert hasattr(G, 'D_in'), "G needs to have the attribute D_in, the input dimension."
        assert hasattr(G, 'D_out'), "G needs to have the attribute D_out, the output dimension."
        
        self.lambda_ = lambda_
        self.M = M
        self.D_in = self.G.D_in
        self.D_out = self.G.D_out
        
        self.skip = torch.nn.Linear(self.D_in, self.D_out, bias=skip_bias) # skip connection (denoted as :math:`\theta` in the paper).
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.G(x)
        y2 = self.skip(x)
        return y1+y2
    
    def train_epoch(self, loss: torch.nn.Module, dl: DataLoader, opt: torch.optim.Optimizer=None):
        """
        Trains one epoch.

        Parameters
        ----------
        loss : torch.nn.Module
            The loss function, e.g. ``torch.nn.MSELoss``.
        dl : DataLoader
            The Dataloader object.
        opt : torch.optim.Optimizer, optional
            An optimizer object. If not specified, we use SGD with Nesterov momentum.

        Returns
        -------
        state : dict
            Measuring diagnostics (e.g. loss function value) at each iteration.

        """
        if opt is None:
            self.opt = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        else:
            if len(opt.param_groups) > 1:
                warnings.warn("The optimizer object has more than one param_group. For the proximal operator, we use the learning rate of opt.param_groups[0].")
            self.opt = opt
        
        state = {'train_loss': list(), }
                    
        self.train()
        for inputs, targets in dl:
            
            opt.zero_grad() # zero gradients    
            y_pred = self.forward(inputs) # forward pass
            loss_val = loss(y_pred, targets) # compute loss           
            loss_val.backward() # backward pass       
            opt.step() # iteration   
            alpha = opt.param_groups[0]['lr'] # step size
            self.skip.weight.data, self.G.W1.weight.data = hier_prox(self.skip.weight.data, 
                                                                     self.G.W1.weight.data,
                                                                     lambda_=self.lambda_*alpha, 
                                                                     lambda_bar=0, 
                                                                     M = self.M) # prox step
            
            state['train_loss'].append(loss_val.item())
                  
        return state
    