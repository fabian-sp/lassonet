# LassoNet implementation

The LassoNet has been implemented by their authors, however only for feed-forward neural networks with ReLU activation. Here, we try to implement the idea more generally.

## How to use

Define a PyTorch network `G`  (i.e. some class inheriting from `torch.nn.Module`) with arbitrary architecture (i.e. a `forward`-method). `G` must fulfill

* first layer is of type `torch.nn.Linear` and called `G.W1`.
* needs the attributes `G.D_in` and `G.D_out` which are input and output dimension of the network.

The `LassoNet` based on `G` is then initialized via

	model = LassoNet(G, lambda_, M)

where `lambda_` and `M` are as in the paper. 

## Examples

* See `example.py` for a simple example on how to define `G` and how to train LassoNet.
* See `example_mnist.py` for an example using the MNIST datatset.


## References:

* [Original paper](https://jmlr.org/papers/volume22/20-848/20-848.pdf)
* [Original repo](https://github.com/lasso-net/lassonet)
