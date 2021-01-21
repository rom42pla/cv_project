from modules.HDF5Store import HDF5Store
import torch
from torch.utils.data.dataset import Dataset


class ASLDataset(Dataset):

    def __init__(self, h5filepath: str):
        self.store = HDF5Store(h5filepath, mode="r")
        self.keys = list(self.store.keys())
        assert all(self.store[self.keys[0]].shape[0] == self.store[k].shape[0] for k in self.keys[1:])

    def __len__(self):
        return self.store[list(self.store.keys())[0]].shape[0]

    def get(self, dsname, idx=None):
        if idx is None: return self.store[dsname][:]
        return self.store[dsname][idx]

    def __getitem__(self, idx):
        return tuple([torch.from_numpy(self.store[k][idx]) if not len(self.store[k].shape) == 1
                      else torch.tensor(self.store[k][idx], dtype=torch.long)
                      for k in self.keys])


class KeysOrderException(Exception): pass
