#!/usr/bin/env python3
import unittest
import numpy as np
from tensorfit import TensorFunction


class TestClass(unittest.TestCase):

    def test_class(self):
        tfunc = TensorFunction()
        assert tfunc.Fitted is False
        assert tfunc.Function is None
        assert tfunc.Params == {}
        assert tfunc.Summary == {}

    def test_init(self):
        tfunc = TensorFunction()
        func = "tf.multiply(self.m, self.X) + self.b"
        params = {"m": 0., "b": 0.}
        tfunc.initialize(func, params)
        assert tfunc.Function is not None


class TestLinearFit(unittest.TestCase):

    def test_fit(self):
        tfunc = TensorFunction()
        func = "self.m * self.X + self.b"
        params = {"m": 0.1, "b": 0.1}
        tfunc.initialize(func, params)

        x = np.linspace(-1, 1, 100)
        y = 5. * x + 0.33 * np.random.randn(*x.shape)
        tfunc.fit(x, y, metric="mse", num_rounds=10000, early_stopping_rounds=10, verbose_eval=1000)
        assert np.isclose(tfunc.Params.get("m"), 5, atol=0.3)
        assert np.isclose(tfunc.Params.get("b"), 0, atol=0.3)


class TestNonlinearFit(unittest.TestCase):

    def test_fit(self):
        tfunc = TensorFunction()
        func = "self.A * tf.sin(self.X) * tf.cos(self.X) ** 2 + self.B"
        params = {"A": 0.1, "B": 0.1}
        tfunc.initialize(func, params)

        # y = m * x + b
        x = np.linspace(0, 30, 1000)
        y = 8 * np.sin(x) * np.cos(x) ** 2 + 0.5 * np.random.randn(*x.shape)
        tfunc.fit(x, y, metric="mse", num_rounds=10000, early_stopping_rounds=10, verbose_eval=1000)
        fitted_params = tfunc.Params
        assert np.isclose(fitted_params["A"], 8, atol=0.3)
        assert np.isclose(fitted_params["B"], 0, atol=0.3)


if __name__ == "__main__":
    unittest.main()
