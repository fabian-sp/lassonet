import torch
import tqdm

class FeedForward(torch.nn.Module):
    """
    2-layer NN with RelU
    """
    def __init__(self, D_in: int, D_out: int, H: int):
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
    
def eval_on_dataloader(model, loss, dl):
    """
    Evaluates a model on a given dataloader (could be from train or test set).
    """
    model.eval()
    
    L = 0
    pbar = tqdm.tqdm(dl)
    
    with torch.no_grad():
        for batch in pbar: 
            inputs, targets = batch           
            y_pred = model.forward(inputs) # forward pass
            L += loss(y_pred, targets).item() * inputs.shape[0] # compute loss*batch_size  
        
    return L/len(dl.dataset)
    