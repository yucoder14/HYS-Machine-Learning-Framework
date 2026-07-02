import cupy as cp
from cccaatl_ml.core.tensor import *
from cccaatl_ml.core.layer import *
from cccaatl_ml.core.activations import *

def _unbroadcast_cupy(grad, shape):
    grad = cp.asarray(grad) 
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)

def _swap_last2_cupy(x):
    x = cp.asarray(x)
    return cp.swapaxes(x, -1, -2) if x.ndim >= 2 else x.T

"""
    Dynamically modify Tensor class for GPU support
"""
def enable_gpu_Tensor():
    if hasattr(Tensor, "_cuda_enabled") and Tensor._cuda_enabled:
        return

    # Modifying init, str, repr
    _original_init = Tensor.__init__
    _original_str = Tensor.__str__
    _original_repr = Tensor.__repr__

    def cuda_aware_init(self, data, *args, device = "cpu"):
        _original_init(self, data, *args)
        self.device = device 
        if device == "gpu": 
          self._array = cp.asarray(self._array)

    def cuda_aware_str(self):
        return f"Tensor(data={self.data}, device={self.device})"

    Tensor.__init__ = cuda_aware_init
    Tensor.__str__ = cuda_aware_str
    Tensor.__repr__ = cuda_aware_str

    # New method to move data to/from gpu
    def to(self, device = "cpu"): 
        if self.device == device: 
            return 
        else: 
            self.device = device
            if device == "cpu": 
                self._array = cp.asnumpy(self._array)
            elif device == "gpu":
                self._array = cp.asarray(self._array)  
            else: 
                raise ValueError(f"Unrecognized device: {device} does not exist or yet to be supported")

    Tensor.to = to

    # modifying pairwise arithmetic functions
    _original_add = Tensor.__add__
    _original_sub = Tensor.__sub__
    _original_mul = Tensor.__mul__
    _original_truediv = Tensor.__truediv__

    def cuda_aware_add(self, other): 
        if isinstance(other, Tensor) and self.device != other.device:
            raise ValueError(f"Device mismatch: trying to add Tensors in different devices")
        # this is where you would launch as custom kernel upon checking devices
        # as of now, it just uses cupy
        result = _original_add(self, other)

        # just changing the numpy based fn to bupy based fn to be used in 
        # backward call
        if device == "gpu": 
            self._grad_fn.fn_unbroadcast = _unbroadcast_cupy  

        result.device = self.device 
        return result

    def cuda_aware_sub(self, other): 
        if isinstance(other, Tensor) and self.device != other.device:
            raise ValueError(f"Device mismatch: trying to subtract Tensors in different devices")
        result = _original_sub(self, other)

        if device == "gpu": 
            self._grad_fn.fn_unbroadcast = _unbroadcast_cupy  
            
        result.device = self.device 
        return result

    def cuda_aware_mul(self, other): 
        if isinstance(other, Tensor) and self.device != other.device:
            raise ValueError(f"Device mismatch: trying to multiply Tensors in different devices")
        result = _original_mul(self, other)

        if device == "gpu": 
            self._grad_fn.fn_unbroadcast = _unbroadcast_cupy  

        result.device = self.device 
        return result

    def cuda_aware_truediv(self, other): 
        if isinstance(other, Tensor) and self.device != other.device:
            raise ValueError(f"Device mismatch: trying to divide Tensors in different devices")
        result = _original_truediv(self, other)

        if device == "gpu": 
            self._grad_fn.fn_unbroadcast = _unbroadcast_cupy  

        result.device = self.device 
        return result

    Tensor.__add__ = cuda_aware_add
    Tensor.__sub__ = cuda_aware_sub
    Tensor.__mul__ = cuda_aware_mul
    Tensor.__truediv__ = cuda_aware_truediv

    # modifying matrix operations  
    _original_matmul = Tensor.matmul
    _original_reshape = Tensor.reshape
    _original_transpose = Tensor.transpose

    def cuda_aware_matmul(self, other):
        if isinstance(other, Tensor) and self.device != other.device:
            raise ValueError(f"Device mismatch: trying to matmul Tensors in different devices")
        result = _original_matmul(self, other)
        if device == "gpu": 
            self._grad_fn.swap = _swap_last2_cupy
            
        result.device = self.device
        return result

    def cuda_aware_reshape(self, *shape): 
        result = _original_reshape(self, *shape)

        if device == "gpu": 
            self._grad_fn.asarray = cp.asarray

        result.device = self.device
        return result

    def cuda_aware_transpose(self):
        result = _original_transpose(self)

        if device == "gpu": 
            self._grad_fn.swap = _swap_last2_cupy

        result.device = self.device
        return result 

    Tensor.matmul = cuda_aware_matmul
    Tensor.reshape = cuda_aware_reshape
    Tensor.transpose = cuda_aware_transpose

    # modifying reductions 
    _original_sum = Tensor.sum
    _original_mean = Tensor.mean
    _original_max = Tensor.max
    _original_min = Tensor.min

    def cuda_aware_sum(self, axis=None, keepdims=False): 
        result = _original_sum(self, axis=axis, keepdims=keepdims)
        if device == "gpu": 
            self._grad_fn.asarray = cp.asarray
            self._grad_fn.expand_dims = cp.expand_dims
            self._grad_fn.broadcast_to = cp.broadcast_to

        result.device = self.device 
        return result

    def cuda_aware_mean(self, axis=None, keepdims=False): 
        if self.device == "cpu": 
            return _original_mean(self, axis=axis, keepdims=keepdims)
        elif self.device == "gpu":
            return Tensor(cp.mean(self._array, axis=axis, keepdims=keepdims), device = "gpu")

    def cuda_aware_max(self, axis=None, keepdims=False): 
        if self.device == "cpu": 
            return _original_max(self, axis=axis, keepdims=keepdims)
        elif self.device == "gpu":
            return Tensor(cp.max(self._array, axis=axis, keepdims=keepdims), device = "gpu")
      
    def cuda_aware_min(self, axis=None, keepdims=False): 
        if self.device == "cpu": 
            return _original_min(self, axis=axis, keepdims=keepdims)
        elif self.device == "gpu":
            return Tensor(cp.min(self._array, axis=axis, keepdims=keepdims), device = "gpu")

    Tensor.sum = cuda_aware_sum
    Tensor.mean = cuda_aware_mean
    Tensor.max = cuda_aware_max
    Tensor.min = cuda_aware_min

    # Modifying backward

    _original_backward = Tensor.backward 

    def _backward_cupy(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self._array)  # seed d(self)/d(self) = 1
        grad = cp.asarray(grad)

        self.grad = grad if self.grad is None else self.grad + grad

        if self._grad_fn is None:
            return  # leaf
        for inp, g in zip(self._grad_fn.inputs, self._grad_fn.backward(grad)):
            if isinstance(inp, Tensor) and inp.requires_grad and g is not None:
                inp.backward(g)

    def cuda_aware_backward(self, grad=None): 
        if self.device == "cpu": 
            _original_backward(self, grad) 
        elif self.device == "gpu": 
            _backward_cupy(self, grad)
        else: 
            pass

    Tensor.backward = cuda_aware_backward

    # End of dynamic modification
    Tensor._cuda_enabled = True

def enable_gpu_Activations(): 
    if hasattr(Activation, "_cuda_enabled") and Activation._cuda_enabled: 
        return

    def to(self, device="cpu"): 
        self.device = device

    Activation.to = to

    _original_relu = ReLU.forward 
    _original_sig = Sigmoid.forward
    _original_tanh = Tanh.forward
    _original_gelu = GELU.forward
    _original_softmax = Softmax.forward

    Activation._cuda_enabled = True

def enable_gpu_Layer(): 
    if not Tensor._cuda_enabled or not Activation._cuda_enabled:
        return

    if hasattr(Layer, "_cuda_enabled") and Layer._cuda_enabled:
        return 

    def to(self, device="cpu"): 
        attrs = vars(self) 
        for name, value in attrs.items(): 
            if isinstance(value, Tensor):
                tensor = self.__getattribute__(name) 
                tensor.to(device)
        self.device = device

    Layer.to = to

    def to_layers(self, device="cpu"):
        for layer in self.layers:
            layer.to(device)

    Sequential.to = to_layers

    Layer._cuda_enabled = True

def enable_gpu():
    enable_gpu_Tensor()
    enable_gpu_Activations()
    enable_gpu_Layer()

    global CUPY_ENABLED 
    CUPY_ENABLED = True
