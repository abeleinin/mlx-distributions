import mlx.core as mx

class Normal:
    f"""
    Normal (or Gaussian) distribution parameterized by ``loc`` and ``scale``.
    """

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, sample_shape):
        return mx.random.normal(shape=sample_shape, dtype=self.mean.dtype, loc=self.loc, scale=self.scale)

    def log_prob(self, value: mx.array) -> mx.array:
        variance = self.scale ** 2
        return -0.5 * (mx.log(2 * mx.pi * variance) + ((value - self.mean) ** 2) / variance)

    def entropy(self):
        return 0.5 * (1 + mx.log(2 * mx.pi * self.scale ** 2))
