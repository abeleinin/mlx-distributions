# MLX Distributions

Unofficial probability distribution library for [Apple's MLX framework](https://github.com/ml-explore/mlx). This package generally follows the [torch.distributions](https://pytorch.org/docs/stable/distributions.html) interface.

## Install

Use the following commands to install the package:

```
git clone git@github.com:abeleinin/mlx-distributions.git
cd mlx-distributions/
pip install mlx
pip install -e .
```

## Distributions

Documentation on the available distributions in the library.

- [Normal](#normal)
- More coming soon!

### Normal

Creates a normal (also called Gaussian) distribution parameterized by `loc` and `scale`.

```python
import mlx_distributions as mx_dist
m = mx_dist.Normal(mx.array([0.0]), mx.array([1.0]))
m.sample() # Returns a value sampled from a normal distrbution (mu=0.0, sigma=1.0)
```

Parameters:

- `loc` (float or mx.array) – mean or µ (mu) the distribution
- `scale` (float or mx.array) – standard deviation or σ (sigma) of the distribution