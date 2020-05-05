# Standard library imports
import numpy as np
import pandas as pd
import ast
import os
import sys
from os.path import dirname as up
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
# Important directories
code_dir = os.path.dirname(os.getcwd())
deep_cal_dir = os.path.dirname(os.path.dirname(os.getcwd()))

# Allows to import my own module
sys.path.insert(0, code_dir)

from pricing.liftedheston import Pricer_Lifted_Heston

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error

best_mse = np.inf


def load_calibrated_data():
    
    return 0

def set_params_range(params):
    """
    Defines the model parameters to calibrate
    
    Args:
    =====
        :params: (dict) dict holding lists of two values, low and high for a parameter.
        
    Return:
    =======
        :dimensions: (list) list with initialized boundaries.
    """
    dimensions = []
    for key, value in params.items():
        param = Real(low=value[0], high=value[1], prior='uniform',name=key,transform='identity')
        dimensions.append(param)
        
    return dimensions



def fitness(r, rho,lambd, nu, theta, v0):
    """
    Fit the calibrated data
    
    Args:
    =====
        :dimensions:
        :r:
        :kappa: mean reversion speed   
        :theta: long run variance
        :sigma: Volatility of variance
        :rho: Correlation parameter
        :v0: Initial variance
        true_model: the map to fit
    
    Return:
    =======
        :mse: mean squared error between created data and expected data()
    """

    # Print the hyper-parameters.
    #print('kappa: {0:.1e},  theta:{0:.1e},  sigma:{0:.1e} ,  rho:{0:.1e} ,  v0:{0:.1e}'.format(kappa,theta, sigma,rho,v0))
    #print()
    
    # Create the neural network with these hyper-parameters.

    
    def processInput(i):
        func1 = Pricer_Lifted_Heston
        func2 = input_.append
        for j in [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0]:
            c = func1(1,r,s,j,rho=rho,lamb=lambd,theta=theta,nu=nu,V0=v0,N=20,rN=2.5,alpha=0.1+1/2,M=200,L_=50)
            func2([c])
        return input_
    
    model = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in np.arange(0.5,1.6,0.1))
    model = np.array(model)
    if not np.sum(np.isnan(model))>=1:
    # after the last training-epoch.
        mse = mean_squared_error(model,input_[:,2])

        # Print the regression error.
        print("MSE: {0:.4%}".format(mse))
        print()

        # Save the model if it improves on the best-found performance.
        # We use the global keyword so we update the variable outside
        # of this function.
        global best_mse

        # If the regressin error of the saved model is improved ...
        if mse < best_mse:
        
            # Update the regression error.
            best_mse = mse
    

        return mse
    return 1000

def fit(func,dimensions,n_calls):
    gp_result = gp_minimize(func=func,dimensions=dimensions,n_calls=n_calls,n_jobs=-1)
    return gp_result

