# LassoNet for convolutions

## Introduction

The LassoNet paper (Lemhadri et al., 2021) combines two fundamental ideas of statistical learning: sparse feature selection as in the Lasso and arbitrary nonlinear modelling using (deep) neural networks. The authors of the paper describe it like this:

> Our approach achieves feature sparsity by adding a skip (residual) layer
 and allowing a feature to participate in any hidden layer only if its skip-layer representative
 is active.

I will briefly summarize the mathematical formulation (though looking into the paper provides all the details): let $x\in\mathbb{R}^n, y\in\mathbb{R}^m$ be input and output for a prediction/classification task. We have a neural network $g(x;\omega):\mathbb{R}^n \to \mathbb{R}^m$ where $\omega$ defines the parameters/weights and the linear (aka skip) layer $\theta \in\mathbb{R}^n$. The first layer of $g$ is a linear layer which weight matrix is denoted by $W^{(1)}$. LassoNet is then given by

$$   
  \min_{\theta,\omega} L(\theta,\omega) + \lambda \|\theta\|_1 \\
  \text{s.t.} \quad \|W_j^{(1)}\|_\infty \leq M |\theta_j|
$$

where $\lambda$ and $M$ are given positive parameters. Note that the constraint has the following effect: if feature $j$ is not selected in the skip layer, then $\theta_j=0$ and thus the $j$-th column of $W^{(1)}$ is zero. This means that the $j$-th feature effectively has no effect for the network.

Importantly, learning the parameters of the network (i.e. the nonlinear model) and the feature selection is done simultaneously by solving the above optimization problem.

## Implementation

The authors of the LassoNet paper also provide a PyTorch implementation of their model. Their code only covers networks with arbitrary linear hidden layers and ReLU activation functions, even though their model is quite flexible in terms of the network architecture (the only restriction is that the network needs a linear layer as first module). Thus, I implemented LassoNet (rather to learn more about PyTorch myself) for any network architecture (starting with a linear layer). The code can be found on [Github](https://github.com/fabian-sp/lassonet).

## Extending to convolutions

So far, the LassoNet constraint was applied directly to the input and is required to start with a linear layer. In modern deep learning, convolutions have proven to be very successfull, in particular for image applications. Adapting LassoNet to this setting is adressed in the paper as in the following: 

> For example in computer vision, the inputs are pixels and without perfect registration,
  a given pixel does not even have the same meaning across different images. In this
  case, one may instead want to select visual shapes or patterns. [...] In this setting, it would be 
  desirable to achieve “filter sparsity”, that is, select the most relevant convolutional filters. This
  can improve the interpretability of the model and alleviate the need for architecture
  search. In future work, we plan to achieve this by applying LassoNet to the output of 
  one or more convolutional layers.


### Implementation details

Essentially, you only need the following additional code in your PyTorch model in order to use LassoNet:

1) Define a skip layer: we map linearly from the output of the convolutions to the output. Note that the output of the convolutions has dimension `out_channels*h_out*w_out` where `h_out, w_out` depend on the hyperparameters of the convolutional kernels.

```
self.skip = nn.Linear(self.out_channels*self.h_out*self.w_out, self.D_out)
```

2) Adapt the `forward` method: we apply the skip layer and add the result to the result of the remaining nonlinear part at the end

```
z1 = self.skip(out.reshape(-1, self.out_channels*self.h_out*self.w_out))
```

3) Defining a prox method which is executed after `.step()` in the training process. This changes the weights of the convolutional filters and the skip layer inplace. All other parameters are unchanged.

```
def prox(self, lr):
```