# LassoNet implementation

## How to use

The LassoNet has been implemented by their authors, however only for feed-forward neural networks with RelU activation. Here, we try to implement the idea more generally.

You can create a `LassoNet` specifying for `G` any Pytorch module/architecture you want. The only constraint is that the first module of `G` is a linear layer, called `W1`.
Hence, make sure that `G.W1` exists and is of type `torch.nn.Linear`. The `LassoNet` is then initialized via


	model = LassoNet(G, lambda_ = l1, M = M)

where `lambda_` and `M` are as in the paper. See `example.py` for a simple example how to train the network.

## References:

* [Original paper](https://jmlr.org/papers/volume22/20-848/20-848.pdf)
* [Original repo](https://github.com/lasso-net/lassonet)
