import os
import sys

code_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, code_dir)

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import pandas as pd

from torch import nn, optim
import torch
from torch.autograd import Variable, grad
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from ann.model import init_weights
from miscellaneous.jacobian import Jacobian
from miscellaneous.helpers import StyblinskiTangNN

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(code_dir + '/ann/runs/lifted_heston_experiment')

f1 = StyblinskiTangNN()


def train_sobolev(model,
                  train_loader,
                  test_loader,
                  lambd,
                  nb_epochs,
                  seed, 
                  save_model_dir,
                  log_df= 'log_df.csv'):
    """
    Arguments:

        train_loader:   Dataloader. 
            train_loader.features: array-like, shape=[# samples, # features].
                Features of the data set.
            train_loader.labels: array-like, shape=[# samples, # labels].
                Labels of the data set.
        test_loader:   Dataloader.
            test_loader.features: array-like, shape=[# samples, # features].
                Features of the data set.
            test_loader.labels: array-like, shape=[# samples, # labels].
                Labels of the data set.
        lambd: double.
            Proportion for the sobolev mean squared error.
        nb_epochs: integer.
            Number of epochs to train the network.
        seed: integer.
            Random seed for PRNG, allowing reproducibility of results.
        project_dir: string.
            Project directory to write to.
        log_df: pandas dataframe, shape=[, nb_layers + 8], default = None
            Pandas df that serves as a log file. If none, df is created.
    Returns:
        log_df: pandas dataframe.
            Pandas df log file with training and validation metrics across eps.
        best_error: float
            Best error on test set among epochs.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.00003)

    train_loss_min = np.Inf
    steps = 0
    model.apply(init_weights)
    model.to(_device)
    jacobian = Jacobian()
    train_losses,train_L1losses,train_L2losses, test_losses,test_L1losses, learningRate = [], [], [], [],[],[]

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    epochs_iter = tqdm(range(nb_epochs), desc="Epoch")
    for e in epochs_iter:
        running_loss = 0
        L1Loss = 0
        L2Loss = 0
        for features in train_loader:
            
            features = features.type(torch.FloatTensor)
            features = features.to(_device)

            features = Variable(features,requires_grad = True)

            labels = f1(features)

            labels = labels.unsqueeze(1)


            MSE = model(features)

            
            #MSE.requires_grad = True

            #labels = Variable(labels,requires_grad=True)

            J_teacher = jacobian(features, labels)
            J_student = jacobian(features, MSE)

            loss = criterion(MSE, labels) 

            loss_sobolev = criterion(J_teacher.flatten(), J_student.flatten())

            loss_total =  loss + lambd * loss_sobolev

            optimizer.zero_grad() # dé-commenter si on utilise pas Adatune
            loss_total.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2000)

            optimizer.step()
 
            L1Loss += loss.item()
            L2Loss += loss_sobolev.item()
            running_loss += loss_total.item()

        else:
            test_L1loss = 0
            test_L2loss = 0
            test_loss = 0
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                for features1 in test_loader:
                    
                    
                    features1 = features1.type(torch.FloatTensor)

                    #features1= features1.to(_device)  #,dlabels1.to(_device)
                    labels1 = f1(features1)
                    MSE = model(features1)
                    labels1 = labels1.unsqueeze(1)
                    test_L1loss += criterion(MSE, labels1)
                    
                    test_loss += test_L1loss 

                    
                    
                    
            model.train()        
            train_losses.append(running_loss/len(train_loader))
            train_L1losses.append(L1Loss/len(train_loader))
            train_L2losses.append(L2Loss/len(train_loader))

            test_loss = test_loss/len(test_loader)
            test_losses.append(test_loss)
            test_L1losses.append(test_L1loss/len(test_loader))


            learningRate.append(optimizer.param_groups[0]['lr'])
            #scheduler.step(test_loss)

            """print(
                  "Training Loss: {:.7f}.. ".format(L1Loss/len(train_loader)),
                  "Training Sobolev Loss: {:.7f}..".format(L2Loss/len(train_loader)),
                  "Training Full Loss: {:.7f}..".format(running_loss/len(train_loader)),
                  "Test Loss: {:.3f}.. ".format(np.sqrt(test_loss)))"""
            # save model if validation loss has decreased
            if running_loss <= train_loss_min:
                #print('Validation loss decreased ({} --> {}).  Saving model ...'.format(test_loss_min,test_loss))
                torch.save(model.state_dict(),save_model_dir)
                train_loss_min = running_loss
            
    #learning_data = pd.DataFrame(list(zip(train_losses,train_L1losses, train_L2losses, test_losses, test_L1losses, learningRate)),columns=['Train Losses', 'Train L1 Losses', 'Train L2 Losses','Test Losses','Test L1 Losses', 'Learning Rate'])
    #learning_data.to_csv(log_df,index=False)
    
    return train_L1losses, train_L2losses