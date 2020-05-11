import torch
from torch import nn, optim
import torch.nn.functional as F

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SobolevRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 256)

        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256, 256)

        self.fc4 = nn.Linear(256,1)
    
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        x = self.fc4(x)
        
        return x