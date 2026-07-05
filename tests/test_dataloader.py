from cccaatl_ml.core.tensor import Tensor
from cccaatl_ml.core.dataloader import DataLoader,TensorDataset,Dataset
import numpy as np
import pytest


# ---------- Dataset / TensorDataset ----------
 
def test_dataset_is_abstract():
    with pytest.raises(TypeError):
        Dataset()
 
 
def test_tensor_dataset_mismatched_dims():
    x = Tensor(np.random.randn(10, 3))
    y = Tensor(np.random.randn(9))  # mismatched first dimension
    with pytest.raises(ValueError):
        TensorDataset(x, y)
 
 
def test_tensor_dataset_len_and_getitem():
    x = Tensor(np.arange(20).reshape(10, 2))
    y = Tensor(np.arange(10))
    ds = TensorDataset(x, y)
 
    assert len(ds) == 10
 
    xi, yi = ds[3]
    assert isinstance(xi, Tensor) and isinstance(yi, Tensor)
    assert (xi._array == np.array([6, 7])).all()
    assert yi._array == 3
 
 
# ---------- DataLoader ----------
 
def test_last_batch_smaller_than_batch_size():
    x = Tensor(np.random.randn(10, 3))
    y = Tensor(np.random.randn(10))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=3)
 
    batches = list(loader)
    assert len(batches) == 4  # 3, 3, 3, 1
    sizes = [b[0].shape[0] for b in batches]
    assert sizes == [3, 3, 3, 1]
    assert len(loader) == 4
 
 
def test_batches_cover_all_samples_without_shuffle():
    x = Tensor(np.arange(10).reshape(10, 1))
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
 
    seen = np.concatenate([b[0]._array.flatten() for b in loader])
    assert (seen == np.arange(10)).all()  # exact order preserved
 
 
def test_shuffle_preserves_elements_but_changes_order():
    np.random.seed(0)
    x = Tensor(np.arange(100).reshape(100, 1))
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=100, shuffle=True)
 
    batch = next(iter(loader))
    values = batch[0]._array.flatten()
 
    assert sorted(values.tolist()) == list(range(100))       # same elements
    assert not (values == np.arange(100)).all()               # different order
 
 
def test_collate_batch_shapes():
    x = Tensor(np.random.randn(7, 4))
    y = Tensor(np.random.randn(7))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=5)
 
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (5, 4)
    assert y_batch.shape == (5,)
 
 
def test_invalid_batch_size():
    ds = TensorDataset(Tensor(np.arange(5)))
    with pytest.raises(ValueError):
        DataLoader(ds, batch_size=0)

print("all tests passed!")