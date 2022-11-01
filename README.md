# A prototype LassoNet 

The LassoNet (Lemhadri et al.) has been implemented by their authors for PyTorch, however only for feed-forward neural networks with ReLU activations. Here, we implement it for an arbitrary architecture instead.

**NOTE:** This repository is only a prototype implementation.

## How to use

The core idea of LassoNet is to use a hierarchical penalty on all input features which have no (direct) linear effect on the output. Hence, the first layer of the model should be linear as the weight of this layer are penalized columnwise.

Define a PyTorch network `G`  (i.e. some class inheriting from `torch.nn.Module`) with arbitrary architecture (i.e. a `forward`-method). `G` must fulfill

* that its first layer is of type `torch.nn.Linear` and called `G.W1`.
* that it has the attributes `G.D_in` and `G.D_out`, the input and output dimension of the network.

The `LassoNet` based on `G` is then initialized simply via

	model = LassoNet(G, lambda_, M)

where `lambda_` and `M` are penalty parameters as described in the paper. 


## Examples

* See `example.py` for a simple example on how to define `G` and how to train LassoNet.
* See `example_mnist.py` for an example using the MNIST datatset.
* See `example_conv_mnist.py` for an **experimental** model applying the LassoNet penalty to convolutional layers.


## References:

* [Original paper](https://jmlr.org/papers/volume22/20-848/20-848.pdf)
* [Original repo](https://github.com/lasso-net/lassonet)
