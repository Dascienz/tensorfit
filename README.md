# TensorFit

TensorFit is an open source package for curve fitting. This package was designed with the intention of allowing students and researchers to quickly assess parametric functions for explaining experimental data and trends. The package currently only supports univariate functions, <i>i.e.</i> functions with a single independent variable.


## Install

```bash
pip install tensorfit
```

## Usage

Import packages as needed.
```python
>>> import numpy as np
>>> from tensorfit import TensorFunction
```

Generate from fake experimental data for the purpose of demonstration.
```python
>>> x = np.linspace(-1, 1, 100)
>>> y = 9.8 * x ** 2 + 6.1 * x + 0.87 * np.random.randn(*x.shape)
```

Create and initialize TensorFunction instance using a parametric model for your fitting function and a set of starting parameters.
```python
>>> tfunc = TensorFunction()
>>> my_func = "self.a * self.X ** 2 + self.b * self.X + self.c"
>>> init_params = {"a": 0.1, "b": 0.1, "c": 0.1}
>>> tfunc.initialize(func=my_func, params=init_params)
```

After initialization, you can make a call to `.fit()` to fit your `TensorFunction()` to the experimental data.
```python
>>> tfunc.fit(x, y, num_rounds=10000, early_stopping_rounds=10, verbose_eval=0)
Early stopping, best iteration is:
[Episode - 6046] mse: 0.81566346
```

Fitted parameters and a summary of your fit can then be looked at.
```python
>>> tfunc.Params
{'a': 9.560993, 'b': 6.0437393, 'c': 0.11265278}
>>> tfunc.Summary
{'mse': 0.81566346, 'r2': 0.9623992666602135}
```

## License
[MIT License](./LICENSE)

This library uses:
* [numpy](https://github.com/numpy/numpy), which is distributed under the BSD 3-Clause license.
* [tensorflow](https://github.com/tensorflow/tensorflow), which is distributed under the Apache 2.0 license.