import abc
import numpy as np
from cccaatl_ml.core.tensor import Tensor

#Common interface so DataLoader can work with any dataset type.
class Dataset(abc.ABC):
    def __init__(self):
        pass
    @abc.abstractmethod
    def __len__(self) -> int:
        pass
    @abc.abstractmethod
    def __getitem__(self, index:int) -> any:
        pass

#Wrap N tensors (e.g. features and labels) as a single indexable dataset.
class TensorDataset:
    def __init__(self, *tensors: Tensor):
        if len(tensors) == 0:
            raise ValueError("At least one tensor is required")
        shared_dim = tensors[0].shape[0]
        for t in tensors:
            if t.shape[0] != shared_dim:
                raise ValueError("First dimension size differs")
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]
    
    def __getitem__(self, index):
        return tuple(Tensor(t._array[index]) for t in self.tensors)

#Dataloader class
class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        indicies = np.arrange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indicies)
        for i in range(0, len(indicies), self.batch_size):
            batch_indicies = indicies[i:i + self.batch_size]
            samples = [self.dataset[i] for i in batch_indicies]
            yield self._collate_batch(samples)

    def __len__(self) -> int:
        # Number of batches per epoch (ceil division).
        return -(-len(self.dataset) // self.batch_size)

    def _collate_batch(self, samples):
        num_tensors = len(samples[0])
        batch = []
        for i in range(num_tensors):
            stacked = np.stack([sample[i]._array for sample in samples])
            batch.append(Tensor(stacked))
        return tuple(batch)