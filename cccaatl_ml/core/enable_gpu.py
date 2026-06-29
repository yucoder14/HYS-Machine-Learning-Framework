import cupy as cp
from cccaatl_ml.core.tensor import Tensor

def enable_gpu():
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
    result.device = self.device 
    return result

  def cuda_aware_sub(self, other): 
    if isinstance(other, Tensor) and self.device != other.device:
      raise ValueError(f"Device mismatch: trying to subtract Tensors in different devices")
    result = _original_sub(self, other)
    result.device = self.device 
    return result

  def cuda_aware_mul(self, other): 
    if isinstance(other, Tensor) and self.device != other.device:
      raise ValueError(f"Device mismatch: trying to multiply Tensors in different devices")
    result = _original_mul(self, other)
    result.device = self.device 
    return result

  def cuda_aware_truediv(self, other): 
    if isinstance(other, Tensor) and self.device != other.device:
      raise ValueError(f"Device mismatch: trying to divide Tensors in different devices")
    result = _original_truediv(self, other)
    result.device = self.device 
    return result
  
  Tensor.__add__ = cuda_aware_add
  Tensor.__sub__ = cuda_aware_sub
  Tensor.__mul__ = cuda_aware_mul
  Tensor.__truediv__ = cuda_aware_truediv

  # modifying matrix multiplication 
  _original_matmul = Tensor.matmul

  def cuda_aware_matmul(self, other):
    if isinstance(other, Tensor) and self.device != other.device:
      raise ValueError(f"Device mismatch: trying to matmul Tensors in different devices")
    result = _original_matmul(self, other)
    result.device = self.device
    return result

  Tensor.matmul = cuda_aware_matmul

  # modifying reductions 
  _original_sum = Tensor.sum
  _original_mean = Tensor.mean
  _original_max = Tensor.max
  _original_min = Tensor.min

  def cuda_aware_sum(self, axis=None, keepdims=None): 
    if self.device == "cpu": 
      return _original_sum(self, axis=axis, keepdims=keepdims)
    elif self.device == "gpu":
      return Tensor(cp.sum(self._array, axis=axis, keepdims=keepdims), device = "gpu")
    else: 
      pass # silently fail for undetected devices

  def cuda_aware_mean(self, axis=None, keepdims=None): 
    if self.device == "cpu": 
      return _original_mean(self, axis=axis, keepdims=keepdims)
    elif self.device == "gpu":
      return Tensor(cp.mean(self._array, axis=axis, keepdims=keepdims), device = "gpu")
    else: 
      pass

  def cuda_aware_max(self, axis=None, keepdims=None): 
    if self.device == "cpu": 
      return _original_max(self, axis=axis, keepdims=keepdims)
    elif self.device == "gpu":
      return Tensor(cp.max(self._array, axis=axis, keepdims=keepdims), device = "gpu")
    else: 
      pass
      
  def cuda_aware_min(self, axis=None, keepdims=None): 
    if self.device == "cpu": 
      return _original_min(self, axis=axis, keepdims=keepdims)
    elif self.device == "gpu":
      return Tensor(cp.min(self._array, axis=axis, keepdims=keepdims), device = "gpu")
    else: 
      pass

  Tensor.sum = cuda_aware_sum
  Tensor.mean = cuda_aware_mean
  Tensor.max = cuda_aware_max
  Tensor.min = cuda_aware_min

  Tensor._cuda_enabled = True
