import unittest
import mlx_distributions_tests

# Array libraries
import mlx.core as mx
import numpy as np
import torch

import torch.distributions as torch_dist
import mlx_distributions as mx_dist

class TestDistributions(mlx_distributions_tests.DistributionsTestCase):
    def test_normal(self):
        loc = 4
        std = 2

        torch_normal = torch_dist.Normal(loc=loc, scale=std)
        mx_normal = mx_dist.Normal(loc=loc, scale=std)

        torch_log_prob = torch_normal.log_prob(torch.arange(5)/4)
        mx_log_prob = mx_normal.log_prob(mx.arange(5)/4)

        self.assertEqualArray(mx_log_prob, mx.array(torch_log_prob))

if __name__ == "__main__":
    np.random.seed(1337)
    unittest.main()
