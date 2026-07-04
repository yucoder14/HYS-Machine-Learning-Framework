from numba import cuda
import numpy as np
import cupy as cp

import time
import math
from functools import partial

from cccaatl_ml.core.tensor import *
from cccaatl_ml.cuda.enable_gpu import *

BROADCAST_MAX_DIM = 4
STRIDES_ARR_DIM = BROADCAST_MAX_DIM * 3 # BROADCAST_MAX_DIM * 3
import numpy as np
import cupy as cp

def benchmark(func, warmup = 10):
  times = []
  for _ in range(warmup):
    start = time.perf_counter()
    func()
    end = time.perf_counter()
    times.append(end - start)

  return sorted(times)[warmup//2]

@cuda.jit(device=True)
def _modulo(i, mod):
    # there is a method to make integer division a lot faster,
    # by using magic numbers; rn this is the main bottle neck
    # but that may be out of scope for this project
    divisor = i // mod
    remainder = i - divisor * mod
    return divisor, remainder

@cuda.jit(device=True)
def _calculate_index(i, strides):
    rem = i
    a_index = b_index = 0

    for j in range(0, STRIDES_ARR_DIM, 3):
        if strides[j + 2] == 0:
            break
        coord, rem = _modulo(rem, strides[j + 2])
        a_index += coord * strides[j]
        b_index += coord * strides[j + 1]

    return a_index, b_index

@cuda.jit
def _add_broadcast(a_flat, b_flat, c_flat, strides):
    """
        Taking the approach of flattening the array
        and using strides to calculate appropriate indices
    """
    i = cuda.grid(1)
    tid = cuda.threadIdx.x
    stride = cuda.gridsize(1)

    if i < c_flat.size:
        for j in range(i, c_flat.size, stride):
            a_index, b_index = _calculate_index(j, strides)
            c_flat[j] = a_flat[a_index] + b_flat[b_index]

# a, b should already be at GPU memory
# assumes a and b are contiguous in memory
def numba_add(a: Tensor, b: Tensor):
    c_shape = cp.broadcast_shapes(a.shape, b.shape)
    ndim = len(c_shape)
    assert ndim <= BROADCAST_MAX_DIM, f"{ndim} exceeds {BROADCAST_MAX_DIM}. Change BROADCAST_MAX_DIM to be higher"

    _orig_a_strides = np.array(a._array.strides) // a._array.itemsize
    _orig_b_strides = np.array(b._array.strides) // b._array.itemsize

    a_strides = np.zeros(ndim, dtype=np.int32)
    b_strides = np.zeros(ndim, dtype=np.int32)
    c_strides = np.zeros(ndim, dtype=np.int32)

    curr = 1
    for i in range(ndim - 1, -1, -1):
        c_strides[i] = curr
        curr *= c_shape[i]

    a_pad = ndim - a._array.ndim
    for d in range(ndim):
        if d >= a_pad and a.shape[d - a_pad] != 1:
            a_strides[d] = _orig_a_strides[d - a_pad]

    b_pad = ndim - b._array.ndim
    for d in range(ndim):
        if d >= b_pad and b.shape[d - b_pad] != 1:
            b_strides[d] = _orig_b_strides[d - b_pad]

    a_flat = a._array.ravel()
    b_flat = b._array.ravel()
    c_flat = cp.zeros(c_shape).ravel()
    zero_pads = np.zeros(BROADCAST_MAX_DIM - ndim, dtype=np.int32)

    magics, shifts = compute_magic_numbers(c_strides)
    a_strides = np.hstack((a_strides, zero_pads))
    b_strides = np.hstack((b_strides, zero_pads))
    c_strides = np.hstack((c_strides, zero_pads))
    strides = cp.asarray(
        np.vstack(
            (a_strides, b_strides, c_strides)
        ).ravel(order="F")
    ).astype(np.int32)

    _add_broadcast.forall(len(c_flat))(a_flat, b_flat, c_flat, strides_combined)
    cuda.synchronize()

    # `ncu --page details python scriptname.py` to see profiling of the kernel
    cuda.profile_start()
    _add_broadcast.forall(len(c_flat))(a_flat, b_flat, c_flat, strides_combined)
    cuda.synchronize()
    cuda.profile_stop()

    return Tensor(c_flat.reshape(c_shape), device="gpu")

if __name__ == "__main__":
    enable_gpu()

    b_shape = (128, 1, 50)
    c_shape = (64, 1, 300, 1)
    b = Tensor(np.random.rand(*b_shape))
    c = Tensor(np.random.rand(*c_shape))

    b.to("gpu")
    c.to("gpu")

    print(benchmark(partial(numba_add, b, c)))
    print(benchmark(partial(b.__add__, c)))
    assert np.allclose(numba_add(b,c)._array, (b + c)._array)
