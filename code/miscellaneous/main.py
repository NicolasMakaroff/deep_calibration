# Standard library imports
import os
import sys

# Important directories
code_dir = os.path.dirname(os.getcwd())
deep_cal_dir = os.path.dirname(os.path.dirname(os.getcwd()))
# Allows to import my own module
sys.path.insert(0, code_dir)

from miscellaneous.helpers import StyblinskiTang, StyblinskiTangNN, PlotStyblinskiTang, plot_results
from miscellaneous.dataloader import SobolevDataset
from miscellaneous.models import SobolevRegressor
from miscellaneous.train_sobolev import train_sobolev

from ann.helpers import open_data

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

x_min = -5
x_max = 5
y_min = -5
y_max = 5
features = np.random.uniform(low=[x_min,y_min], high=[x_max,y_max], size=(20,2))

value_ = {'x':features[:,0], 'y':features[:,1]}
database = pd.DataFrame(value_)
print(database.tail())
database.to_csv('data/StyblinskiTang.csv',index=False)

data = open_data('data/StyblinskiTang.csv')
data = data.to_numpy()
train_data = data[:]
test_data = data[:]

train_dataset =SobolevDataset(train_data)
test_dataset =SobolevDataset(test_data)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=5,
                          shuffle=True,
                          num_workers=1)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=20,
                          shuffle=True,
                          num_workers=1)



for k in [50000]:
    model = SobolevRegressor()
    out_S, out_sobolev_S = train_sobolev(model,train_loader,test_loader,1.0,nb_epochs=k,seed=123,save_model_dir='results/tang.pt',log_df='results/log_df.csv')
    model = SobolevRegressor()
    out1, out_sobolev1 = train_sobolev(model,train_loader,test_loader,0.0,nb_epochs=k,seed=123,save_model_dir='results/tangNN.pt',log_df='results/log_dfNN.csv')

    x,y = np.arange(-5,5,0.25),np.arange(-5,5,0.25)

    reg = SobolevRegressor()
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reg.load_state_dict(torch.load('results/tangNN.pt',map_location=torch.device(_device)))

    z = []
    for i in x:
        for j in y:
            c = reg(torch.tensor([i,j])).detach().numpy()
            z.append(c)

    prediction = np.array(z,dtype=np.float32)        
    prediction = np.reshape(prediction,(40,40))

    reg = SobolevRegressor()
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reg.load_state_dict(torch.load('results/tang.pt',map_location=torch.device(_device)))

    z_S = []
    for i in x:
        for j in y:
            c = reg(torch.tensor([i,j])).detach().numpy()
            z_S.append(c)

    prediction_S = np.array(z_S,dtype=np.float32)        
    prediction_S = np.reshape(prediction_S,(40,40))
    
    out = [prediction,out1]
    out_sobolev = [prediction_S,out_S,out_sobolev_S]
    
    plot_results(20,k,'1e-4','1', out, out_sobolev)

    