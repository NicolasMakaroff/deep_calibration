from torch import from_numpy
import torch
from torch.utils.data import Dataset
import numpy as np

class VolatilityDataset(Dataset):
    """ Volatility dataset.
    
    Args:
        :data: (numpy_array)
    """

    # Initialize your data, download, etc.
    def __init__(self,data):
        xy = data
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:8])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class LiftedHestonDataset(Dataset):
    """ Lifted Heston dataset.
    
    Args:
        :data_x: (numpy_array) inputs
        :data_y: (numpy_array) outputs
    """

    # Initialize your data, download, etc.
    def __init__(self,data_x,data_y):
        x = data_x
        y = data_y
        self.len = x.shape[0]
        self.x_data = x[:,:]
        self.y_data = y[:,:]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        input_ = self.x_data[index,:]
        vol_imp = self.y_data[index,:]
        vol_imp = np.array([vol_imp])
        vol_imp = vol_imp.astype('float').reshape(-1,88)
        sample = {'input': from_numpy(input_), 'output': from_numpy(vol_imp)}

        return from_numpy(input_),from_numpy(vol_imp)
        

    def __len__(self):
        return self.len
