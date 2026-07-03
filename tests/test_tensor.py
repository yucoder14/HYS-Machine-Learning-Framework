from cccaatl_ml.core.tensor import Tensor
import numpy as np
import pytest

@pytest.fixture 
def create_tensor():
    return Tensor(np.array([1,2,3])) 

def test_matmul_correct(create_tensor):
    assert create_tensor.matmul(create_tensor.transpose())._array == 14

def test_matmul_error(create_tensor):
    with pytest.raises(ValueError):
        create_tensor.matmul(np.array([1,2]))

def test_sum(create_tensor):
    assert create_tensor.sum()._array == 6

def test_mean(create_tensor):
    assert create_tensor.mean()._array == 2

def test_max(create_tensor):
    assert create_tensor.max()._array == 3
    
def test_min(create_tensor):
    assert create_tensor.min()._array == 1

