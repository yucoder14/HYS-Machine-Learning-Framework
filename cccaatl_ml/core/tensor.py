import numpy as np

# Autograd is off until enable_autograd() flips this. Ops still run; they just
# don't record a graph, so backward() does nothing useful until it's on.
_AUTOGRAD_ENABLED = False


def enable_autograd():
    global _AUTOGRAD_ENABLED
    _AUTOGRAD_ENABLED = True


def _val(x):
    return x._array if isinstance(x, Tensor) else x


# undo numpy broadcasting: sum grad back down to the input's shape
def _unbroadcast(grad, shape):
    grad = np.asarray(grad) 
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


# transpose of the last two axes (plain .T for 1-D)
def _swap_last2(x):
    x = np.asarray(x)
    return np.swapaxes(x, -1, -2) if x.ndim >= 2 else x.T


# attach a grad_fn to out, but only while tracking and only if some
# input actually needs a gradient
def _track(out, fn_cls, inputs, **state):
    if not _AUTOGRAD_ENABLED:
        return out
    if not any(isinstance(t, Tensor) and t.requires_grad for t in inputs):
        return out
    out.requires_grad = True
    out._grad_fn = fn_cls(inputs, **state)
    return out


class Function:
    def __init__(self, inputs, **state):
        self.inputs = list(inputs)
        self.saved_tensors = self.inputs
        self.__dict__.update(state)  # axis, keepdims, etc.

    def backward(self, grad):
        raise NotImplementedError


class AddBackward(Function):
    def backward(self, grad):
        a, b = self.inputs
        return [self.fn_unbroadcast(grad, _val(a).shape),
                self.fn_unbroadcast(grad, _val(b).shape)]


class SubBackward(Function):
    def backward(self, grad):
        a, b = self.inputs
        return [self.fn_unbroadcast(grad, _val(a).shape),
                self.fn_unbroadcast(-grad, _val(b).shape)]


class MulBackward(Function):
    def backward(self, grad):
        a, b = self.inputs
        av, bv = _val(a), _val(b)
        return [self.fn_unbroadcast(grad * bv, av.shape),
                self.fn_unbroadcast(grad * av, bv.shape)]


class DivBackward(Function):
    def backward(self, grad):
        a, b = self.inputs
        av, bv = _val(a), _val(b)
        return [self.fn_unbroadcast(grad / bv, av.shape),
                self.fn_unbroadcast(-grad * av / (bv * bv), bv.shape)]


class MatmulBackward(Function):
    def backward(self, grad):
        a, b = self.inputs
        return [grad @ self.swap(_val(b)), self.swap(_val(a)) @ grad]


class SumBackward(Function):
    def backward(self, grad):
        (a,) = self.inputs
        grad = self.asarray(grad)
        if self.axis is not None and not self.keepdims:
            grad = self.expand_dims(grad, self.axis)
        return [self.broadcast_to(grad, _val(a).shape).copy()]


class ReshapeBackward(Function):
    def backward(self, grad):
        (a,) = self.inputs
        return [self.asarray(grad).reshape(_val(a).shape)]

class TransposeBackward(Function):
    def backward(self, grad):
        return [self.swap(grad)]


class Tensor:
    #uhh i kinda forgot how to oop in python

    def __init__(self, data, requires_grad=False):
        self._array = np.asarray(data) if isinstance(data, list) else data
        self.shape = self._array.shape
        self.size = self._array.size
        self.dtype = self._array.dtype
        self.data = data
        # autograd bookkeeping (harmless when autograd is off)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    #Private helper: returns a raw nparray or scalar
    def _coerce(self, other):
        if isinstance(other, Tensor):
            return other._array
        return other

    #Tensor arithmetic
    def __add__(self, other):
        out = Tensor(self._array + self._coerce(other))
        return _track(out, AddBackward, (self, other), fn_unbroadcast=_unbroadcast)

    def __sub__(self, other):
        out = Tensor(self._array - self._coerce(other))
        return _track(out, SubBackward, (self, other), fn_unbroadcast=_unbroadcast)

    def __mul__(self, other):
        out = Tensor(self._array * self._coerce(other))
        return _track(out, MulBackward, (self, other), fn_unbroadcast=_unbroadcast)

    def __truediv__(self, other):
        out = Tensor(self._array / self._coerce(other))
        return _track(out, DivBackward, (self, other), fn_unbroadcast=_unbroadcast)
    
    def __matmul__(self, other): 
        return self.matmul(other)

    def matmul(self, other):
        out = Tensor(np.matmul(self._array, self._coerce(other)))
        return _track(out, MatmulBackward, (self, other), swap=_swap_last2)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self._array, axis=axis, keepdims=keepdims))
        return _track(out, SumBackward, (self,), axis=axis, keepdims=keepdims,
                      asarray=np.asarray, expand_dims=np.expand_dims, broadcast_to=np.broadcast_to)

    def mean(self,axis=None, keepdims=False):
        return Tensor(np.mean(self._array, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        return Tensor(np.max(self._array, axis=axis, keepdims=keepdims))

    def min(self, axis=None, keepdims=False):
        return Tensor(np.min(self._array, axis=axis, keepdims=keepdims))

    # functional (returns a new Tensor) so the result can be a graph node
    def reshape(self, *shape):
        if(len(shape) == 1 and isinstance(shape[0], (tuple,list))):
            shape = shape[0]
        out = Tensor(self._array.reshape(shape))
        return _track(out, ReshapeBackward, (self,), asarray=np.asarray)

    def transpose(self):
        if (self._array.ndim >= 2):
            out = Tensor(np.swapaxes(self._array, -1, -2))
        else:
            out = Tensor(self._array.transpose())
        return _track(out, TransposeBackward, (self,), swap=_swap_last2)

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self._array)  # seed d(self)/d(self) = 1
        grad = np.asarray(grad)

        self.grad = grad if self.grad is None else self.grad + grad

        if self._grad_fn is None:
            return  # leaf
        for inp, g in zip(self._grad_fn.inputs, self._grad_fn.backward(grad)):
            if isinstance(inp, Tensor) and inp.requires_grad and g is not None:
                inp.backward(g)

    def zero_grad(self):
        self.grad = None
