import numpy as np

class Tensor:
    #uhh i kinda forgot how to oop in python

    def __init__(self, data):
        self._array = np.array(data) if isinstance(data, list) else data
        self.shape = self._array.shape
        self.size = self._array.size
        self.dtype = self._array.dtype
        self.data = data
    
    #Private helper: returns a raw nparray or scalar
    def _coerce(self, other):
        if isinstance(other, Tensor):
            return other._array
        return other

    #String
    def __str__(self): 
        return f"Tensor(data={self.data})"
        
    def __repr__(self): 
        return f"Tensor(data={self.data})"

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

    def __matmul__(self, other): 
        return self.matmul(other)

    def matmul(self,other): 
        result = Tensor(np.matmul(self._array, self._coerce(other)))
        return result 
    
    def sum(self, axis=None, keepdims=False):
        return Tensor(np.sum(self._array, axis=axis, keepdims=keepdims))

    def mean(self,axis=None, keepdims=False):
        return Tensor(np.mean(self._array, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False): 
        return Tensor(np.max(self._array, axis=axis, keepdims=keepdims))
        
    def min(self, axis=None, keepdims=False): 
        return Tensor(np.min(self._array, axis=axis, keepdims=keepdims))
    
    def reshape(self, *shape):
        if(len(shape) == 1 and isinstance(shape[0], (tuple,list))):
            shape = shape[0]

        return Tensor(self._array.reshape(shape))
    
    def transpose(self):
        return Tensor(self._array.transpose())

