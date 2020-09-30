#!/usr/bin/env python3
import re
import numpy as np
import collections
import tensorflow as tf
from .utils import r2_score

if not hasattr(tf, "placeholder"):
    import tensorflow.compat.v1 as tf
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.disable_v2_behavior()


class TensorFunction(object):

    def __init__(self) -> None:
        """ Initialize TensorFunction class. """
        self.Fitted = False
        self.Function = None
        self.X = tf.placeholder(dtype=tf.float32)
        self.Y = tf.placeholder(dtype=tf.float32)
        self.Params = {}
        self.Summary = {}

    def _set_parameters(self, params: dict) -> None:
        """ Dynamically set function parameters. """
        variable_code = "self.{0} = tf.Variable({1}, name='{0}', dtype=tf.float32)"
        for k, v in params.items():
            byte_code = compile(variable_code.format(k, v), filename="<inline code>", mode="exec")
            exec(byte_code)
        return

    def _set_function(self, func: str) -> None:
        """ Dynamically set TensorFunction. """
        func = re.sub(" +", " ", str(func).strip())
        function_code = "self.Function = {0}"
        byte_code = compile(function_code.format(func), filename="<inline code>", mode="exec")
        exec(byte_code)
        return

    def initialize(self, func: str, params: dict) -> object:
        """ Initialize Function Parameters. """
        if self.Function is not None:
            raise TypeError("TensorFunction has already been initialized. Please reset before initializing again.")
        if not params:
            raise ValueError("params cannot be an empty dict.")
        if not self.Fitted:
            self._set_parameters(params)
            self._set_function(func)
        return self

    def reset(self) -> object:
        """ Reset attributes. """
        for k, v in self.__dict__.copy().items():
            if isinstance(v, tf.Variable):
                del self.__dict__[k]
        self.Fitted = False
        self.Function = None
        self.X = tf.placeholder(dtype=tf.float32)
        self.Y = tf.placeholder(dtype=tf.float32)
        self.Params = {}
        self.Summary = {}
        return self

    def fit(self, x: np.ndarray, y: np.ndarray, metric: str = "mse", learning_rate: float = 0.01,
            num_rounds: int = 100, early_stopping_rounds: int = 0, verbose_eval: int = 0) -> object:
        """ Fit TensorFunction to array-like data. """
        if self.Fitted:
            raise ValueError(
                "TensorFunction has been fitted already. Please reset and initialize before fitting again.")
        try:
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            assert x.ndim == y.ndim == 1
            assert x.shape[0] == y.shape[0]
        except Exception as e:
            raise ValueError("x and y must be array-like objects: {0}".format(e))

        if self.Function is None:
            raise ValueError("TensorFunction has not been initialized. Please initialize before fitting.")

        if metric == "mse":
            loss = tf.reduce_mean(tf.square(self.Y - self.Function))
        else:
            raise ValueError("'{0}' is not a valid error metric.".format(metric))

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        if 1 < int(early_stopping_rounds) < 1000:
            print("Fitting until error does not improve for {0} rounds.".format(int(early_stopping_rounds)))
            error_rounds = collections.deque(maxlen=int(early_stopping_rounds))
        else:
            error_rounds = None

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for episode in range(int(num_rounds)):
                sess.run(train, feed_dict={self.X: x, self.Y: y})
                error = sess.run(loss, feed_dict={self.X: x, self.Y: y})
                if verbose_eval > 0 and episode % verbose_eval == 0:
                    print("[Episode - {0}] {1}: {2:,.8f}".format(episode, metric, error))
                if error_rounds is not None:
                    error_rounds.append(error)
                    if len(error_rounds) == int(early_stopping_rounds):
                        if all(e == error_rounds[0] for e in error_rounds):
                            print("Early stopping, best iteration is:")
                            print("[Episode - {0}] {1}: {2:,.8f}".format(
                                episode - int(early_stopping_rounds), metric, error))
                            break

            for k, v in self.__dict__.copy().items():
                if isinstance(v, tf.Variable):
                    self.Params[k] = sess.run(v)

            self.Summary[metric] = sess.run(loss, feed_dict={self.X: x, self.Y: y})
            self.Summary["r2"] = r2_score(y=y, yhat=sess.run(self.Function, feed_dict={self.X: x, self.Y: y}))
        self.Fitted = True
        return self
