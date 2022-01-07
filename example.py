import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from module import LassoNet


torch.manual_seed(42)
np.random.seed(42)

#%% Generate data

D_in = 10 # input dimension
D_out = 1 # output dimension
H = 30 # hidden layer size

N = 5000 # training samples
batch_size = 32 

def generate_toy_example(N):
    X = torch.randn(N, D_in)  
    y = 2.*X[:, 3] - 1.*X[:, 3]**2 + 1.*X[:, 1] + 0.5*X[:, 2] + 2 * X[:, 4] * X[:, 5]

    return X, y.reshape(-1,1)

X, Y = generate_toy_example(N)
X_test, Y_test = generate_toy_example(4000)

#%% Create DataLoader (see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return x, y
    
ds = MyDataset(X,Y)
dl = DataLoader(ds, batch_size = batch_size, shuffle = True)

valid_ds = MyDataset(X_test,Y_test)
valid_dl = DataLoader(valid_ds, batch_size = batch_size, shuffle = True)

#%% Define non-linear part of LassoNet

class FeedForward(torch.nn.Module):
    """
    2-layer NN with RelU
    """
    def __init__(self, D_in, D_out):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        
        self.W1 = torch.nn.Linear(D_in, H, bias = True)
        self.relu = torch.nn.ReLU()
        self.W2 = torch.nn.Linear(H, H)
        self.W3 = torch.nn.Linear(H, D_out)
        return
    
    def forward(self, x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        x = self.relu(x)
        x = self.W3(x)
        return x
    
#%% Initialize the model

l1 = 6.
M = 1.

G = FeedForward(D_in, D_out)
model = LassoNet(G, lambda_ = l1, M = M, skip_bias = True)

loss = torch.nn.MSELoss(reduction='mean')

# params of G are already included in params of model!
for param in model.parameters():
    print(param.size())


#%% Training

n_epochs = 100
alpha0 = 1e-3 #initial step size/learning rate

all_loss = list()

#opt = torch.optim.Adam(model.parameters(), lr = alpha0)
opt = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
sched = StepLR(opt, step_size = 20, gamma = 0.7)


train_info = model.do_training(loss, dl, opt = opt, lr_schedule = sched, valid_dl = valid_dl, n_epochs = n_epochs, verbose = True)

#%% Evaluation

print("Skip layer weights: ", model.skip.weight.data)

fig, ax = plt.subplots()
ax.plot(train_info['train_loss'], c = '#002635', marker = 'o', label = 'Training loss')
ax.plot(train_info['valid_loss'], c = '#002635', marker = 'x', ls = '--', label = 'Validation loss')
ax.set_yscale('log')

fig, ax = plt.subplots()
ax.imshow(G.W1.weight.data, cmap = "coolwarm", vmin = -.1, vmax = .1)
#fig.savefig('plots/example_weights.png', dpi = 400)


#%% Define HierNet

# class myG(torch.nn.Module):
#     """
#     2-layer NN with RelU
#     """
#     def __init__(self, D_in, D_out):
#         super().__init__()
#         self.D_in = D_in
#         self.D_out = D_out
        
#         self.W1 = torch.nn.Linear(D_in, D_in, bias = False)
#         return
        
#     def forward(self, x):       
#         # compute W^Tx
#         y1 = torch.matmul(self.W1.weight.t(), x.t()).t()
#         # compute Wx
#         y2 = self.W1(x)
#         y = (y2+y1)/2
#         # compute x^T(W+W^T)/2 x
#         z = torch.einsum('ij,ij->i',x,y).reshape(-1,1)
        
#         return z
