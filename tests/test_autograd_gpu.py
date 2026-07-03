from cccaatl_ml.core.tensor import Tensor, enable_autograd
import numpy as np
import pytest

enable_autograd()
cuda_enable = pytest.importorskip("cccaatl_ml.cuda.enable_gpu")
cuda_enable.enable_gpu()

def test_add_chain():
    # y = (a + b) * a  ->  dy/da = 2a + b, dy/db = a
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True, device="gpu")
    b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True, device="gpu")
    y = (a + b) * a
    y.backward(np.ones(3))
    assert np.allclose(a.grad, 2 * a._array + b._array)
    assert np.allclose(b.grad, a._array)


def test_sub_div():
    # y = (a - b) / b  ->  dy/da = 1/b, dy/db = -a/b^2
    a = Tensor(np.array([6.0, 8.0]), requires_grad=True, device="gpu")
    b = Tensor(np.array([2.0, 4.0]), requires_grad=True, device="gpu")
    y = (a - b) / b
    y.backward(np.ones(2))
    assert np.allclose(a.grad, 1.0 / b._array)
    assert np.allclose(b.grad, -a._array / (b._array ** 2))


def test_matmul():
    # y = sum(A @ B)  ->  dA = ones @ B.T, dB = A.T @ ones
    A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True, device="gpu")
    B = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True, device="gpu")
    (A.matmul(B)).sum().backward()
    g = np.ones((2, 2))
    assert np.allclose(A.grad, g @ B._array.T)
    assert np.allclose(B.grad, A._array.T @ g)


def test_sum_then_scalar_seed():
    # scalar output seeds grad = 1 automatically
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True, device="gpu")
    (x * x).sum().backward()
    assert np.allclose(x.grad, 2 * x._array)


def test_bias_broadcast():
    # adding a (2,) bias to a (3, 2) batch: bias grad sums over the batch axis
    x = Tensor(np.ones((3, 2)), requires_grad=True, device="gpu")
    b = Tensor(np.array([1.0, 2.0]), requires_grad=True, device="gpu")
    (x + b).sum().backward()
    assert np.allclose(b.grad, [3.0, 3.0])
    assert b.grad.shape == (2,)


def test_zero_grad_resets_accumulation():
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True, device="gpu")
    (x * x).sum().backward()
    first = x.grad.copy()
    # without zeroing, a second backward accumulates on top
    (x * x).sum().backward()
    assert np.allclose(x.grad, 2 * first)
    x.zero_grad()
    assert x.grad is None
    (x * x).sum().backward()
    assert np.allclose(x.grad, first)


def test_requires_grad_false_skips_graph():
    a = Tensor(np.array([1.0, 2.0]), requires_grad=False, device="gpu")
    b = Tensor(np.array([3.0, 4.0]), requires_grad=False, device="gpu")
    out = (a + b) * a
    assert out.requires_grad is False
    assert out._grad_fn is None
    out.backward(np.ones(2))  # no-op, must not raise
    assert a.grad is None and b.grad is None
