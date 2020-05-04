from torch import nn, optim
import torch.nn.functional as F

class RegressorHeston(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, 400)
        self.bn3 = nn.BatchNorm1d(400)
        self.fc4 = nn.Linear(400, 400)
        self.bn4 = nn.BatchNorm1d(400)
        self.fc5 = nn.Linear(400,400)
        self.bn5 = nn.BatchNorm1d(400)
        self.fc6 = nn.Linear(400,1)
        
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        x = self.fc6(x)
        
        return x

class RegressorLiftedHeston(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 60)
        #self.bn1 = nn.BatchNorm1d(600)
        self.fc2 = nn.Linear(60, 60)
        #self.bn2 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(60, 60)
        #self.bn3 = nn.BatchNorm1d(400)
        self.fc4 = nn.Linear(60, 60)
        #self.bn5 = nn.BatchNorm1d(200)
        self.fc5 = nn.Linear(60,88)
        
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))

        x = self.fc5(x)
            

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)