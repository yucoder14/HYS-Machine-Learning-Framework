"""CUDA kernels for element-wise arithmetic and activation functions.

Each kernel operates on flat 1-D arrays. Broadcasting is handled by the
launcher functions, which compute the broadcast shape on the CPU and pass
the per-dimension sizes and strides so each thread can resolve its own
source indices.
"""

import math
import numpy as np

# --------------------------------------------------------------------------- #
#  Broadcast helpers
# --------------------------------------------------------------------------- #

def _broadcast_shapes(a_shape, b_shape):
    """Return the broadcasted shape of two shapes (numpy semantics)."""
    ndim = max(len(a_shape), len(b_shape))
    a = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b = (1,) * (ndim - len(b_shape)) + tuple(b_shape)
    out = []
    for ad, bd in zip(a, b):
        if ad == 1:
            out.append(bd)
        elif bd == 1 or ad == bd:
            out.append(ad)
        else:
            raise ValueError(
                f"operands could not be broadcast together with shapes "
                f"{a_shape} {b_shape}"
            )
    return tuple(out)


def _compute_strides(shape):
    """Row-major (C-order) strides for a given shape."""
    strides = []
    s = 1
    for dim in reversed(shape):
        strides.append(s)
        s *= dim
    return tuple(reversed(strides))


# --------------------------------------------------------------------------- #
#  Element-wise binary kernels (add, sub, mul, div)
# --------------------------------------------------------------------------- #

try:
    from numba import cuda

    @cuda.jit
    def _add_kernel(a, b, out, a_strides, b_strides, out_strides,
                    a_shape, b_shape, out_shape, ndim):
        tid = cuda.grid(1)
        if tid >= out.size:
            return
        out_multi = 0
        remaining = tid
        for d in range(ndim):
            dim = out_shape[d]
            out_multi += (remaining % dim) * out_strides[d]
            remaining //= dim
        a_idx = 0
        rem = tid
        for d in range(ndim):
            dim = out_shape[d]
            coord = rem % dim
            rem //= dim
            if a_shape[d] > 1:
                a_idx += coord * a_strides[d]
        b_idx = 0
        rem = tid
        for d in range(ndim):
            dim = out_shape[d]
            coord = rem % dim
            rem //= dim
            if b_shape[d] > 1:
                b_idx += coord * b_strides[d]
        out[out_multi] = a[a_idx] + b[b_idx]

    @cuda.jit
    def _sub_kernel(a, b, out, a_strides, b_strides, out_strides,
                    a_shape, b_shape, out_shape, ndim):
        tid = cuda.grid(1)
        if tid >= out.size:
            return
        out_multi = 0
        remaining = tid
        for d in range(ndim):
            dim = out_shape[d]
            out_multi += (remaining % dim) * out_strides[d]
            remaining //= dim
        a_idx = 0
        rem = tid
        for d in range(ndim):
            dim = out_shape[d]
            coord = rem % dim
            rem //= dim
            if a_shape[d] > 1:
                a_idx += coord * a_strides[d]
        b_idx = 0
        rem = tid
        for d in range(ndim):
            dim = out_shape[d]
            coord = rem % dim
            rem //= dim
            if b_shape[d] > 1:
                b_idx += coord * b_strides[d]
        out[out_multi] = a[a_idx] - b[b_idx]

    @cuda.jit
    def _mul_kernel(a, b, out, a_strides, b_strides, out_strides,
                    a_shape, b_shape, out_shape, ndim):
        tid = cuda.grid(1)
        if tid >= out.size:
            return
        out_multi = 0
        remaining = tid
        for d in range(ndim):
            dim = out_shape[d]
            out_multi += (remaining % dim) * out_strides[d]
            remaining //= dim
        a_idx = 0
        rem = tid
        for d in range(ndim):
            dim = out_shape[d]
            coord = rem % dim
            rem //= dim
            if a_shape[d] > 1:
                a_idx += coord * a_strides[d]
        b_idx = 0
        rem = tid
        for d in range(ndim):
            dim = out_shape[d]
            coord = rem % dim
            rem //= dim
            if b_shape[d] > 1:
                b_idx += coord * b_strides[d]
        out[out_multi] = a[a_idx] * b[b_idx]

    @cuda.jit
    def _div_kernel(a, b, out, a_strides, b_strides, out_strides,
                    a_shape, b_shape, out_shape, ndim):
        tid = cuda.grid(1)
        if tid >= out.size:
            return
        out_multi = 0
        remaining = tid
        for d in range(ndim):
            dim = out_shape[d]
            out_multi += (remaining % dim) * out_strides[d]
            remaining //= dim
        a_idx = 0
        rem = tid
        for d in range(ndim):
            dim = out_shape[d]
            coord = rem % dim
            rem //= dim
            if a_shape[d] > 1:
                a_idx += coord * a_strides[d]
        b_idx = 0
        rem = tid
        for d in range(ndim):
            dim = out_shape[d]
            coord = rem % dim
            rem //= dim
            if b_shape[d] > 1:
                b_idx += coord * b_strides[d]
        out[out_multi] = a[a_idx] / b[b_idx]

    # ----------------------------------------------------------------------- #
    #  Scalar binary kernels (when one operand is a Python scalar)
    # ----------------------------------------------------------------------- #

    @cuda.jit
    def _scalar_add_kernel(a, scalar, out):
        tid = cuda.grid(1)
        if tid < out.size:
            out[tid] = a[tid] + scalar

    @cuda.jit
    def _scalar_sub_kernel(a, scalar, out):
        tid = cuda.grid(1)
        if tid < out.size:
            out[tid] = a[tid] - scalar

    @cuda.jit
    def _scalar_mul_kernel(a, scalar, out):
        tid = cuda.grid(1)
        if tid < out.size:
            out[tid] = a[tid] * scalar

    @cuda.jit
    def _scalar_div_kernel(a, scalar, out):
        tid = cuda.grid(1)
        if tid < out.size:
            out[tid] = a[tid] / scalar

    # ----------------------------------------------------------------------- #
    #  Activation kernels (element-wise)
    # ----------------------------------------------------------------------- #

    @cuda.jit
    def _relu_kernel(a, out):
        tid = cuda.grid(1)
        if tid < out.size:
            val = a[tid]
            out[tid] = val if val > 0.0 else 0.0

    @cuda.jit
    def _sigmoid_kernel(a, out):
        tid = cuda.grid(1)
        if tid < out.size:
            x = a[tid]
            if x >= 500.0:
                out[tid] = 1.0
            elif x <= -500.0:
                out[tid] = 0.0
            else:
                out[tid] = 1.0 / (1.0 + math.exp(-x))

    @cuda.jit
    def _tanh_kernel(a, out):
        tid = cuda.grid(1)
        if tid < out.size:
            out[tid] = math.tanh(a[tid])

    @cuda.jit
    def _gelu_kernel(a, out):
        tid = cuda.grid(1)
        if tid < out.size:
            x = a[tid]
            # sigmoid(1.702 * x) * x
            inner = 1.702 * x
            if inner >= 500.0:
                sig = 1.0
            elif inner <= -500.0:
                sig = 0.0
            else:
                sig = 1.0 / (1.0 + math.exp(-inner))
            out[tid] = sig * x

    # ----------------------------------------------------------------------- #
    #  Softmax kernel (per-row, last axis)
    # ----------------------------------------------------------------------- #

    @cuda.jit
    def _softmax_kernel(a, out, n_rows, row_len):
        """Each thread block handles one row.
        Shared memory is used for the per-row max and sum.
        """
        row = cuda.blockIdx.x
        if row >= n_rows:
            return

        tid = cuda.threadIdx.x
        blockSize = cuda.blockDim.x

        # Phase 1: find row max
        local_max = -math.inf
        for i in range(tid, row_len, blockSize):
            val = a[row * row_len + i]
            if val > local_max:
                local_max = val

        # Reduce max in shared memory
        sm = cuda.shared.array(1024, cuda.float64)
        sm[tid] = local_max
        cuda.syncthreads()

        s = blockSize // 2
        while s > 0:
            if tid < s:
                if sm[tid + s] > sm[tid]:
                    sm[tid] = sm[tid + s]
            cuda.syncthreads()
            s //= 2

        row_max = sm[0]
        cuda.syncthreads()

        # Phase 2: exp(x - max) and sum
        local_sum = 0.0
        for i in range(tid, row_len, blockSize):
            val = math.exp(a[row * row_len + i] - row_max)
            out[row * row_len + i] = val
            local_sum += val

        sm[tid] = local_sum
        cuda.syncthreads()

        s = blockSize // 2
        while s > 0:
            if tid < s:
                sm[tid] += sm[tid + s]
            cuda.syncthreads()
            s //= 2

        row_sum = sm[0]
        cuda.syncthreads()

        # Phase 3: divide
        for i in range(tid, row_len, blockSize):
            out[row * row_len + i] /= row_sum

except ImportError:
    # numba not installed — kernels remain None, CPU fallback used
    pass


# --------------------------------------------------------------------------- #
#  Launcher functions (called from Python, handle grid/block config)
# --------------------------------------------------------------------------- #

_THREADS_PER_BLOCK = 256


def _launch_binary(kernel, a_dev, b_dev, a_shape, b_shape):
    """Launch a binary element-wise kernel with manual broadcasting."""
    out_shape = _broadcast_shapes(a_shape, b_shape)
    ndim = len(out_shape)

    # Pad shapes to ndim
    a_padded = (1,) * (ndim - len(a_shape)) + tuple(a_shape)
    b_padded = (1,) * (ndim - len(b_shape)) + tuple(b_shape)

    a_strides = _compute_strides(a_padded)
    b_strides = _compute_strides(b_padded)
    out_strides = _compute_strides(out_shape)

    out_size = int(np.prod(out_shape)) if out_shape else 1
    out_dev = cuda.device_array(out_shape, dtype=a_dev.dtype)

    blocks = (out_size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK

    kernel[blocks, _THREADS_PER_BLOCK](
        a_dev, b_dev, out_dev,
        cuda.to_device(np.array(a_strides, dtype=np.int64)),
        cuda.to_device(np.array(b_strides, dtype=np.int64)),
        cuda.to_device(np.array(out_strides, dtype=np.int64)),
        cuda.to_device(np.array(a_padded, dtype=np.int64)),
        cuda.to_device(np.array(b_padded, dtype=np.int64)),
        cuda.to_device(np.array(out_shape, dtype=np.int64)),
        np.int32(ndim),
    )
    return out_dev


def _launch_scalar(kernel, a_dev, scalar):
    """Launch a scalar element-wise kernel."""
    out_size = a_dev.size
    out_dev = cuda.device_array(a_dev.shape, dtype=a_dev.dtype)
    blocks = (out_size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    kernel[blocks, _THREADS_PER_BLOCK](a_dev, scalar, out_dev)
    return out_dev


def _launch_unary(kernel, a_dev):
    """Launch a unary element-wise kernel."""
    out_size = a_dev.size
    out_dev = cuda.device_array(a_dev.shape, dtype=a_dev.dtype)
    blocks = (out_size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    kernel[blocks, _THREADS_PER_BLOCK](a_dev, out_dev)
    return out_dev


# --------------------------------------------------------------------------- #
#  Public arithmetic launchers
# --------------------------------------------------------------------------- #

def cuda_add(a_dev, b_dev, a_shape, b_shape):
    return _launch_binary(_add_kernel, a_dev, b_dev, a_shape, b_shape)

def cuda_sub(a_dev, b_dev, a_shape, b_shape):
    return _launch_binary(_sub_kernel, a_dev, b_dev, a_shape, b_shape)

def cuda_mul(a_dev, b_dev, a_shape, b_shape):
    return _launch_binary(_mul_kernel, a_dev, b_dev, a_shape, b_shape)

def cuda_div(a_dev, b_dev, a_shape, b_shape):
    return _launch_binary(_div_kernel, a_dev, b_dev, a_shape, b_shape)

def cuda_scalar_add(a_dev, scalar):
    return _launch_scalar(_scalar_add_kernel, a_dev, scalar)

def cuda_scalar_sub(a_dev, scalar):
    return _launch_scalar(_scalar_sub_kernel, a_dev, scalar)

def cuda_scalar_mul(a_dev, scalar):
    return _launch_scalar(_scalar_mul_kernel, a_dev, scalar)

def cuda_scalar_div(a_dev, scalar):
    return _launch_scalar(_scalar_div_kernel, a_dev, scalar)


# --------------------------------------------------------------------------- #
#  Public activation launchers
# --------------------------------------------------------------------------- #

def cuda_relu(a_dev):
    return _launch_unary(_relu_kernel, a_dev)

def cuda_sigmoid(a_dev):
    return _launch_unary(_sigmoid_kernel, a_dev)

def cuda_tanh(a_dev):
    return _launch_unary(_tanh_kernel, a_dev)

def cuda_gelu(a_dev):
    return _launch_unary(_gelu_kernel, a_dev)

def cuda_softmax(a_dev, shape):
    """Softmax along the last axis.

    Expects a 1-D or 2-D array.  For 1-D, treats the whole array as a
    single row.  For 2-D, each row is processed independently by one
    thread block.
    """
    if len(shape) == 1:
        n_rows = 1
        row_len = shape[0]
    else:
        n_rows = int(np.prod(shape[:-1]))
        row_len = shape[-1]

    out_dev = cuda.device_array(shape, dtype=a_dev.dtype)

    # Each block handles one row; threads within block cooperate
    block_size = min(_THREADS_PER_BLOCK, row_len)
    # round up to power of 2 for reduction simplicity
    block_size = 1 << (block_size - 1).bit_length()
    block_size = min(block_size, 1024)

    _softmax_kernel[n_rows, block_size](a_dev, out_dev, n_rows, row_len)
    return out_dev
