"""

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR

from module import LassoNet

from torchvision import datasets
import torchvision.transforms as transforms

#%%
batch_size = 50

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

class FeedForward(torch.nn.Module):
    """
    2-layer NN with RelU
    """
    def __init__(self, D_in = 784, D_out = 10, H = 512):
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
    
#%% the actual model

l1 = 5.
M = 1.

G = FeedForward()
model = LassoNet(G, lambda_ = l1, M = M)

loss = torch.nn.CrossEntropyLoss()

# test forward method, reshape input to vector with view
G(images.view(-1,28*28))
model.forward(images.view(-1,28*28))

# params of G are already included in params of model!
for param in model.parameters():
    print(param.size())
    

#%% Training

n_epochs = 5
alpha0 = 1e-2 #initial step size/learning rate

prep = lambda x: x.reshape(-1,28*28)

#opt = torch.optim.Adam(model.parameters(), lr = alpha0)
opt = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
lr_schedule = StepLR(opt, step_size = 1, gamma = 0.5)


loss_hist = {'train_loss':[], 'valid_loss':[], 'train_acc':[], 'valid_acc':[]}

for j in range(n_epochs): 
    print(f"================== Epoch {j+1}/{n_epochs} ================== ")
    print(opt)  
    
    ### TRAINING
    epoch_info = model.train_epoch(loss, train_loader, opt=opt, preprocess=prep)
    loss_hist['train_loss'].append(np.mean(epoch_info['train_loss']))
    loss_hist['train_acc'].append(np.mean(epoch_info['train_acc']))
    
    if lr_schedule is not None:
        lr_schedule.step()
    
    ### VALIDATION
    valid_loss = 0; valid_acc = 0;
    model.eval()
    for inputs, targets in test_loader:
        output = model.forward(prep(inputs))          
        valid_loss += loss(output, targets).item()
        _, predictions = torch.max(output.data, 1)
        valid_acc += (predictions == targets).float().mean().item()
    
    loss_hist['valid_loss'].append(valid_loss/len(test_loader))
    loss_hist['valid_acc'].append(valid_acc/len(test_loader))
             
    print(f"\t  train loss: {np.mean(epoch_info['train_loss'])}.")
    print(f"\t  validation loss: {valid_loss/len(test_loader)}.")    

#%% evaluation

fig, ax = plt.subplots()
ax.plot(loss_hist['train_loss'], c = '#002635', marker = 'o', label = 'Training loss')
ax.plot(loss_hist['valid_loss'], c = '#002635', marker = 'x', ls = '--', label = 'Validation loss')
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

plt.figure()
plt.imshow(G.W1.weight.data, cmap = "coolwarm")


importance = model.skip.weight.data[8,:].view(28,28).numpy()
plt.figure()
plt.imshow(importance, cmap = "coolwarm")#, vmin = -0.0001, vmax = 0.0001)

