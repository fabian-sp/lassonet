# LassoNet for convolutions

## Introduction

In my own words, I would say that LassoNet combines two fundamental ideas: sparse feature selection as in the Lasso and arbitrary nonlinear modelling using (deep) neural networks. The authors of the paper describe it like this:


> Our approach achieves feature sparsity by adding a skip (residual) layer
 and allowing a feature to participate in any hidden layer only if its skip-layer representative
 is active.

I will briefly summarize the mathematical formulation (though looking into the paper provides all the details): let $x\in\mathbb{R}^n, y\in\mathbb{R}^m$ be input and output for a prediction/classification task. We have a neural network $g(x;\omega):\mathbb{R}^n \to \mathbb{R}^m$ where $\omega$ defines the parameters/weights and the linear (aka skip) layer $\theta \in\mathbb{R}^n$. The first layer of $g$ is a linear layer which weight matrix is denoted by $W^{(1)}$. LassoNet is then given by

$$   
	\min_{\theta,\omega} L(\theta,\omega) + \lambda \|\theta\|_1 \\
	\text{s.t.} \quad \|W_j^{(1)}\|_\infty \leq M |\theta_j|
$$

where $\lambda$ and $M$ are given positive parameters. Note that the constraint has the following effect: if feature $j$ is not selected in the skip layer, then $\theta_j=0$ and thus the $j$-th column of $W^{(1)}$ is zero. This means that the $j$-th feature effectively has no effect for the network.

Learning the parameters of the network (i.e. the nonlinear model) and the feature selection is done simultaneously.


## Implementation

The authors of the LassoNet paper also provide a PyTorch implementation of their model. They only cover networks with arbitrary linear hidden layers and RelU activation functions. Their modelling however is quite flexible in terms of the network architecture, the only restriction is that the network needs a linear layer as first module. Thus, I implemented LassoNet (rather to learn more about PyTorch myself) for any network architecture (starting with a linear layer). The code can be found on [Github](https://github.com/fabian-sp/lassonet).

## Extending to convolutions

So far, 
In the LassoNet paper (Lemhadri et al., 2021), the authors explain  

> For example in computer vision, the inputs are pixels and without perfect registration,
  a given pixel does not even have the same meaning across different images. In this
  case, one may instead want to select visual shapes or patterns. The current practice
  of convolutional neural networks shies away from hand-engineered features and learns
  these from the data instead (Neyshabur, 2020). Currently, the state-of-the-art is
  based on learning these convolutional filters. In this setting, it would be desirable to
  achieve “filter sparsity”, that is, select the most relevant convolutional filters. This
  can improve the interpretability of the model and alleviate the need for architecture
  search. In future work, we plan to achieve this by applying LassoNet to the output of 
  one or more convolutional layers.