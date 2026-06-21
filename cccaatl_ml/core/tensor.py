import numpy as np

class Tensor:
    #uhh i kinda forgot how to oop in python

    def __init__(self, data):
        self._array = data
        self.shape = self._array.shape
        self.size = self._array.size
        self.dtype = self._array.dtype
        self.data = data
    
    #Private helper: returns a raw nparray or scalar
    def _coerce(self, other):
        if isinstance(other, Tensor):
            return other._array
        return other

    #Tensor arithmetic
    def __add__(self, other):
        return Tensor(self._array + self._coerce(other))
    
    def __sub__(self, other):
        return Tensor(self._array - self._coerce(other))
    
    def __mul__(self, other):
        result = Tensor(self._array * self._coerce(other))
        return result
    
    def __truediv__(self, other):
        result = Tensor(self._array / self._coerce(other))
        return result

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

