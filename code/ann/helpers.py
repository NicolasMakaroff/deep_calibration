# ----------------- Utility functions ------------------------------------- #

import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go

import os
import sys
from os.path import dirname as up

# Important directories
code_dir = os.path.dirname(os.getcwd())
deep_cal_dir = os.path.dirname(os.path.dirname(os.getcwd()))

# Allows to import my own module
sys.path.insert(0, code_dir)

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = code_dir + '/ann/plots'
logdir = "{}/plot-{}.html".format(root_logdir, now)


def plot_map(z,save=False):
    """
    Plot the vol implied map
    
    Args:
    -----
        x: moneyness
        y: maturity
        z: (array_like) the implied vol
        save: (boolean) save the file to the directory plot
    """
    x, y = np.log(np.linspace(5,15,11)/10), np.linspace(1,20,8)/10
    fig = go.Figure(data=go.Surface(x=x, y=y, z=z))
    fig.update_layout(title='Implied volatility map', autosize=False,
                  width=500, height=500,margin=dict(l=65, r=50, b=65, t=90))
    if save == True:
        fig.write_html(logdir)
    else: fig.show()

def open_data(file,
              info = False):
    """ Open the data and transform it in a DataFrame
        Arguments :
            :file: CSV to read and convert into a pandas DataFrame
            :info: default = False : Boolean to get summary information on the created object
        Output :
            A pandas DataFrame with all the data from the CSV file
    """
    df = pd.read_csv(file)
    if info is True :
        print('Five first rows of the generated DataFrame : \n {}'.format(df.head()))
        print('\nDataFrame shape : {}\n'.format(df.shape))
    return df


# Utility function to standardise inputs for NN training.
def standardise_inputs(test_inputs, train_mean, train_std):
    
    logger.info("Normalizing labeled inputs for feeding in NN.")
    
    test_inputs -= train_mean
    test_inputs /= train_std
    
    return test_inputs



def create_train_test_split(dataframe,
                            train_frac,
                            test_frac,
                            target,
                            norm = False,
                            lifted = False,
                            random_state = 123):
    """ Create the train and test set for the training with a random method
        Arguments :
            :dataframe: pandas DataFrame containing the date to split
            :train_frac: float, fraction number of training data to keep
            :test_frac: float, fraction number of test data to keep
            :target: string, name of the target value
            :norm: boolean, wheter or not to apply normalization
        Outputs :
            :train_features: pandas DataFrame of the training points selected randomly
            :train_labels: pandas DataFrame, outputs for the training
            :test_features: pandas DataFrame of the test points selected randomly
            :test_labels: pandas DataFrame, outputs for the tests
    """
    train_dataset = dataframe.sample(frac = train_frac, random_state = random_state)
    tmp = dataframe.drop(train_dataset.index)
    test_dataset = tmp.sample(frac = test_frac, random_state = random_state)
    tmp.drop(test_dataset.index)
    train_labels = train_dataset.pop(target)
    train_features = train_dataset
    test_labels = test_dataset.pop(target)
    test_features =test_dataset
    if norm == True:
        train_features, test_features = norm(train_features), norm(test_features)
    if lifted == True:
        train_data = train_features.to_numpy(dtype=np.float32) 
        train_labels = np.array(list(train_labels.values))
        test_data = test_features.to_numpy(dtype=np.float32) 
        test_labels = np.array(list(test_labels.values))
        return train_data, train_labels, test_data, test_labels
    else :  
        train_dataset = pd.concat([train_features, train_labels], axis=1)
        test_dataset = pd.concat([test_features, test_labels], axis=1)
        train_dataset, test_dataset = train_dataset.dropna(), test_dataset.dropna()  
        train_data = train_dataset.to_numpy(dtype=np.float32) 
        test_data = test_dataset.to_numpy(dtype=np.float32)
        return train_data, test_data



def get_statistics(dataframe,
                   *argv):
    """ Compute some basic statistics over the data
        Arguments :
            :dataframe: pandas DataFrame
            :*argv: allows to pass multiple DataFrame in one time
        Output : None
    """
    print('Statistics Computed : \n {}'.format(dataframe.describe().transpose()))
    for arg in argv :
        print(arg.describe().transpose())



def norm(x):
    """ Standardization of a dataset
        Arguments :
            :x: pandas Dataframe contening the data to standardize
        Output :
            A pandas DataFrame with standardize values
    """
    x_stats = x.describe().transpose()
    return((x - x_stats['mean'])/x_stats['std'])




def minmaxscaler(x):
    """ MinMax scale of a dataset
        Arguments :
            :x: pandas Dataframe contening the data to standardize
        Output :
            A pandas DataFrame with scaled values
    """
    x_stats = x.describe().transpose()
    return ((x-x_stats['max'])/(x_stats['max']-x_stats['min']))




