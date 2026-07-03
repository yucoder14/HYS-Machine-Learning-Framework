"""GPU-aware activation layers.

Each class mirrors the ``Layer`` API from ``cccaatl_ml.core.layer``:
``forward(x)`` / ``__call__(x)`` / ``parameters()``.

When the input ``Tensor`` is on CUDA, the forward pass dispatches a
CUDA kernel.  When the tensor is on CPU, a NumPy fallback is used so
the layers work without a GPU.
"""

import numpy as np

from cccaatl_ml.core.layer import Layer
from cccaatl_ml.core.tensor import Tensor
from cccaatl_ml.cuda import kernels as K


class ReLU(Layer):
    def forward(self, x):
        if x.device == "cuda":
            out_dev = K.cuda_relu(x._array)
            return Tensor(out_dev, device="cuda")
        return Tensor(np.maximum(0, x._array))


class Sigmoid(Layer):
    def forward(self, x):
        if x.device == "cuda":
            out_dev = K.cuda_sigmoid(x._array)
            return Tensor(out_dev, device="cuda")
        arr = x._array
        out = np.empty_like(arr)
        pos = arr >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
        neg = ~pos
        ex = np.exp(arr[neg])
        out[neg] = ex / (1.0 + ex)
        return Tensor(out)


class Tanh(Layer):
    def forward(self, x):
        if x.device == "cuda":
            out_dev = K.cuda_tanh(x._array)
            return Tensor(out_dev, device="cuda")
        return Tensor(np.tanh(x._array))


class GELU(Layer):
    """GELU using the fast approximation: ``sigmoid(1.702 * x) * x``."""

    def forward(self, x):
        if x.device == "cuda":
            out_dev = K.cuda_gelu(x._array)
            return Tensor(out_dev, device="cuda")
        inner = 1.702 * x._array
        sig = np.where(inner >= 0,
                       1.0 / (1.0 + np.exp(-inner)),
                       np.exp(inner) / (1.0 + np.exp(inner)))
        return Tensor(sig * x._array)


class Softmax(Layer):
    """Softmax along the last axis (numerically stable)."""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        arr = x._array
        if self.dim != -1 and self.dim != arr.ndim - 1:
            raise NotImplementedError(
                "CUDA Softmax currently only supports the last axis"
            )
        if x.device == "cuda":
            out_dev = K.cuda_softmax(x._array, x.shape)
            return Tensor(out_dev, device="cuda")
        # CPU fallback
        max_vals = np.max(arr, axis=-1, keepdims=True)
        shifted = arr - max_vals
        exp_vals = np.exp(shifted)
        sums = np.sum(exp_vals, axis=-1, keepdims=True)
        return Tensor(exp_vals / sums)
