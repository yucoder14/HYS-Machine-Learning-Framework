import numpy as np


class Tensor:
    #uhh i kinda forgot how to oop in python

    def __init__(self, data, device="cpu"):
        self._array = data
        self.shape = self._array.shape
        self.size = self._array.size
        self.dtype = self._array.dtype
        self.data = data
        self.device = device
    
    #Private helper: returns a raw nparray or scalar
    def _coerce(self, other):
        if isinstance(other, Tensor):
            return other._array
        return other

    def _is_scalar(self, other):
        return isinstance(other, (int, float, np.number))

    # --- device management ---

    def to(self, device):
        """Move tensor to ``device`` ("cpu" or "cuda").

        Returns *self* for chaining.  If the tensor is already on the
        target device, this is a no-op.
        """
        if device == self.device:
            return self
        if device == "cuda":
            from numba import cuda
            self._array = cuda.to_device(self._array)
            self.device = "cuda"
        elif device == "cpu":
            self._array = self._array.copy_to_host()
            self.device = "cpu"
        else:
            raise ValueError(f"Unknown device: {device!r}")
        self.data = self._array
        self.shape = self._array.shape
        return self

    def numpy(self):
        """Return a NumPy array copy of the tensor data.

        If the tensor lives on the GPU, this physically copies it to the
        CPU.  The tensor itself is not modified.
        """
        if self.device == "cuda":
            return self._array.copy_to_host()
        return np.array(self._array)

    # --- Tensor arithmetic ---

    def __add__(self, other):
        if self.device == "cuda":
            return self._cuda_binary(other, "add")
        return Tensor(self._array + self._coerce(other))

    def __sub__(self, other):
        if self.device == "cuda":
            return self._cuda_binary(other, "sub")
        return Tensor(self._array - self._coerce(other))

    def __mul__(self, other):
        if self.device == "cuda":
            return self._cuda_binary(other, "mul")
        return Tensor(self._array * self._coerce(other))

    def __truediv__(self, other):
        if self.device == "cuda":
            return self._cuda_binary(other, "div")
        return Tensor(self._array / self._coerce(other))

    def _cuda_binary(self, other, op):
        """Dispatch a binary op to CUDA kernels with manual broadcasting."""
        from cccaatl_ml.cuda import kernels as K

        scalar = self._coerce(other)
        is_scalar = self._is_scalar(other) or (not hasattr(scalar, "shape"))

        fn_map = {
            "add": (K.cuda_scalar_add, K.cuda_add),
            "sub": (K.cuda_scalar_sub, K.cuda_sub),
            "mul": (K.cuda_scalar_mul, K.cuda_mul),
            "div": (K.cuda_scalar_div, K.cuda_div),
        }
        scalar_fn, tensor_fn = fn_map[op]

        if is_scalar:
            out_dev = scalar_fn(self._array, scalar)
        else:
            out_dev = tensor_fn(self._array, scalar, self.shape, scalar.shape)
        return Tensor(out_dev, device="cuda")

    def matmul(self,other): 
        result = Tensor(np.matmul(self._array, self._coerce(other)))
        return result 
    
    def sum(self, axis=None, keepdims=False):
        return Tensor(np.sum(self._array, axis=axis, keepdims=keepdims))

    def mean(self,axis=None, keepdims=False):
        return Tensor(np.mean(self._array, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False): 
        return Tensor(np.max(self._array, axis=axis, keepdims=keepdims))
    
    def reshape(self, *shape):
        if(len(shape) == 1 and isinstance(shape[0], (tuple,list))):
            shape = shape[0]
        self._array = self._array.reshape(shape)
        self.shape = self._array.shape
        self.data = self._array
        return self
    
    def transpose(self):
        if (self._array.ndim >= 2):
            axes_order = list(range(self._array.ndim))
            axes_order[-1], axes_order[-2] = axes_order[-2], axes_order[-1]
            self._array = self._array.transpose(axes_order)
        else:
            self._array = self._array.transpose()
        self.data = self._array
        self.shape = self._array.shape
        return self

