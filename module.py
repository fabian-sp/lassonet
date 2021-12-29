import numpy as np
import torch
from torch.nn import functional as F

 
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
    def __init__(self, G, lambda_ = 0.01, M = 10, theta_bias = False):
        """
        G is a torch object itself
        G needs to have a linear layer at the beginning called G.W1
        """
        super().__init__()
        
        self.G = G
        self.lambda_ = lambda_
        self.M = M
        self.D_in = self.G.D_in
        self.D_out = self.G.D_out
        
        self.skip = torch.nn.Linear(self.D_in, self.D_out, bias = theta_bias) # skip connection aka theta
        return
    
    def forward(self, x):
        y1 = self.G(x)
        y2 = self.skip(x)
        return y1+y2
    
    def do_training(self, loss, dl, opt = None, lr_schedule = None, n_epochs = 10, verbose = True):
        """
        dl: PyTorch DataLoader
        loss: loss function
        opt: PyTorch optimizer
        lr_schedule: PyTorch learning rate scheduler
        """
        if opt is None:
            opt = torch.optim.SGD(self.parameters(), lr = 0.001, momentum = 0.9, nesterov = True)
        
        all_loss = list()

        for j in np.arange(n_epochs):
            
            for data, target in dl:
                
                # forward pass
                y_pred = self.forward(data)
                # compute loss
                loss_val = loss(y_pred, target)           
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
                
            if lr_schedule is not None:
                lr_schedule.step()
                
            if verbose:
                print(f"Epoch {j}, loss:", loss_val.item())
                
            all_loss.append(loss_val.item())
            
        return all_loss