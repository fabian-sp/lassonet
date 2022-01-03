# LassoNet implementation

The LassoNet has been implemented by their authors, however only for feed-forward neural networks with ReLU activation. Here, we try to implement the idea more generally.

## How to use

Define a PyTorch network `G`  (i.e. some class inheriting from `torch.nn.Module`) with arbitrary architecture (i.e. a `forward`-method). The only constraint on `G` is that its first layer is linear and called `G.W1`.

Hence, make sure that `G.W1` exists and is of type `torch.nn.Linear`. The `LassoNet` based on `G` is then initialized via

	model = LassoNet(G, lambda_, M)

where `lambda_` and `M` are as in the paper. See `example.py` for a simple example on how to define `G` and how to train LassoNet.

## References:

* [Original paper](https://jmlr.org/papers/volume22/20-848/20-848.pdf)
* [Original repo](https://github.com/lasso-net/lassonet)
