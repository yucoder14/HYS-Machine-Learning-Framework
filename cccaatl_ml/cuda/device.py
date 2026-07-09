"""GPU device detection utilities."""

_CUDA_AVAILABLE = None


def detect():
    """Return True if a CUDA-capable GPU is available via numba.cuda.

    Safe to call even when the CUDA toolkit or driver is missing —
    returns False instead of raising.
    """
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE
    try:
        from numba import cuda
        _CUDA_AVAILABLE = cuda.is_available()
    except Exception:
        _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE
