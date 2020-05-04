from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import pandas as pd
from torch import nn, optim
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .model import init_weights
import os
import sys
code_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, code_dir)

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(code_dir + '/ann/runs/lifted_heston_experiment')

def train(model,
          train_loader,
          test_loader,
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
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    test_loss_min = np.Inf
    steps = 0
    model.apply(init_weights)
    model.to(_device)
    train_losses, test_losses, learningRate = [], [], []
    #vg = ValidationGradient(test_loader, criterion, 'cuda')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    epochs_iter = tqdm_notebook(range(nb_epochs), desc="Epoch")
    for e in epochs_iter:
        running_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad() # dé-commenter si on utilise pas Adatune
            labels = labels.view(labels.size(0),-1)
            labels = labels.type(torch.FloatTensor)
            features = features.type(torch.FloatTensor)
            features,labels = features.to(_device), labels.to(_device)
            MSE = model(features)
            loss = criterion(MSE, labels)

            # a commenter si on utilise pas adatune
            ###
            """first_grad = ag.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
            hyper_optim.compute_hg(model, first_grad)
            for params, gradients in zip(model.parameters(), first_grad):
                params.grad = gradients
            optimizer.step()
            hyper_optim.hyper_step(vg.val_grad(model))
            clear_grad(model)"""
            ###

            # décommenter pour utiliser adamHD
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2000)
            optimizer.step()

            running_loss += loss.item()
        
        else:
            test_loss = 0

        
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                for features, labels in test_loader:
                    labels = labels.view(labels.size(0),-1)
                    labels = labels.type(torch.FloatTensor)
                    features = features.type(torch.FloatTensor)
                    features,labels = features.to(_device), labels.to(_device)
                    MSE = model(features)
                    test_loss += criterion(MSE, labels)

            model.train()        
            train_losses.append(running_loss/len(train_loader))
            writer.add_scalar('Train/Loss', running_loss/len(train_loader), e)
            writer.flush()
            test_loss = test_loss/len(test_loader)
            test_losses.append(test_loss)
            writer.add_scalar('Test/Loss', test_loss, e)
            writer.flush()
            learningRate.append(optimizer.param_groups[0]['lr'])
            #scheduler.step(test_loss)
            scheduler.step()
            """print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(np.log(running_loss/len(train_loader))),
                  "Test Loss: {:.3f}.. ".format(torch.log(test_loss)),
                  "Learning rate: {}..".format(optimizer.param_groups[0]['lr']))"""
            # save model if validation loss has decreased
            if test_loss <= test_loss_min:
                print('Validation loss decreased ({} --> {}).  Saving model ...'.format(test_loss_min,test_loss))
                torch.save(model.state_dict(),save_model_dir + '/modelLiftedHeston.pt')
                test_loss_min = test_loss
            
    learning_data = pd.DataFrame(list(zip(train_losses, test_losses, learningRate)),columns=['Train Losses','Test Losses', 'Learning Rate'])
    learning_data.to_csv(log_df,index=False)
    
    return learning_data, test_loss_min