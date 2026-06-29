"""CUDA GPU support for the HYS ML framework.

Public API:
    detect()          — check for CUDA GPU availability
    ReLU, GELU,       — GPU-aware activation layers
    Sigmoid, Tanh,
    Softmax
"""

from cccaatl_ml.cuda.device import detect
from cccaatl_ml.cuda.activations import ReLU, GELU, Sigmoid, Tanh, Softmax

__all__ = ["detect", "ReLU", "GELU", "Sigmoid", "Tanh", "Softmax"]
