from cccaatl_ml.core.tensor import Tensor
from cccaatl_ml.core.layer import Linear, Dropout, Sequential
from cccaatl_ml.core.activations import * 

import numpy as np
import pytest

cuda_enable = pytest.importorskip("cccaatl_ml.cuda.enable_gpu")
cuda_enable.enable_gpu()

@pytest.fixture 
def create_tensor():
    return Tensor(np.array([1,2,3]), device="cpu") 

@pytest.fixture 
def create_tensor_gpu():
    return Tensor(np.array([1,2,3]), device="gpu") 

def test_device_correct(create_tensor_gpu):
    assert create_tensor_gpu.device == "gpu"
    assert isinstance(create_tensor_gpu._array, cp.ndarray)

def test_forward(create_net, create_batch_data): 
    try:
        net = Sequential([
            Linear(10, 10), 
            ReLU(),
            Linear(10, 10), 
            GELU(), 
            Linear(10, 10), 
            Dropout(0.1), 
            Tanh(),
            Linear(10, 10), 
            Sigmoid(),
            Linear(10, 10),
            Dropout(0.1), 
            Softmax()
        ]) 
        net.to("gpu")
        data = Tensor(np.random.rand((32, 10)), device="gpu")
        assert net(data).sum() == 1.0
    except Exception as e:
        pytest.fail(e)

def test_device_error(create_tensor, create_tensor_gpu): 
    with pytest.raises(ValueError): 
        create_tensor + create_tensor_gpu 

def test_matmul_correct(create_tensor_gpu):
    assert create_tensor_gpu.matmul(create_tensor_gpu.transpose())._array == 14

def test_matmul_error(create_tensor_gpu):
    with pytest.raises(ValueError):
        create_tensor_gpu.matmul(np.array([1,2]))

def test_sum(create_tensor_gpu):
    assert create_tensor_gpu.sum()._array == 6

def test_mean(create_tensor_gpu):
    assert create_tensor_gpu.mean()._array == 2

def test_max(create_tensor_gpu):
    assert create_tensor_gpu.max()._array == 3
    
def test_min(create_tensor_gpu):
    assert create_tensor_gpu.min()._array == 1

