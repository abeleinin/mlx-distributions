import unittest

import mlx.core as mx

class DistributionsTestCase(unittest.TestCase):
    def assertEqualArray(
        self,
        mx_res: mx.array,
        expected: mx.array,
        atol=1e-2,
        rtol=1e-2,
    ):
        self.assertEqual(
            tuple(mx_res.shape),
            tuple(expected.shape),
            msg=f"shape mismatch expected={expected.shape} got={mx_res.shape}",
        )
        self.assertEqual(
            mx_res.dtype,
            expected.dtype,
            msg=f"dtype mismatch expected={expected.dtype} got={mx_res.dtype}",
        )
        if not isinstance(mx_res, mx.array):
            mx_res = mx.array(mx_res)
        elif not isinstance(expected, mx.array):
            expected = mx.array(expected)
        self.assertTrue(mx.allclose(mx_res, expected, rtol=rtol, atol=atol))
