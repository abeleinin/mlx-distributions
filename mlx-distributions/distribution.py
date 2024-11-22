import mlx.core as mx

__all__ = ["Distribution"]

class Distribution:
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    def log_prob(self, value: mx.array):
        raise NotImplementedError

    def sample(self, num_samples=1):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError
