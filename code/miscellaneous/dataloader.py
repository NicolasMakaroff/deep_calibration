from torch import from_numpy
import torch
from torch.utils.data import Dataset

class SobolevDataset(Dataset):
    """ Sobolev Dataset.
    
    Args:
        :data_x: (numpy_array) inputs
    """

    # Initialize your data, download, etc.
    def __init__(self,data):
        x = data[:,:]
        self.len = x.shape[0]
        self.x_data = x


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        input_ = self.x_data[index,:]

        return from_numpy(input_) 
        

    def __len__(self):
        return self.len
