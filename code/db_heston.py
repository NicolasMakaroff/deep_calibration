from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

from tqdm import tqdm 
import numpy as np
import os
import sys
from os.path import dirname as up
import scipy as scp

# Important directories
code_dir = os.path.dirname(os.getcwd())
deep_cal_dir = os.path.dirname(os.path.dirname(os.getcwd()))

# Allows to import my own module
sys.path.insert(0, code_dir)

from pricing.heston_fourier import price_heston_fourier
from pricing_model import call_option_brent
import warnings
warnings.filterwarnings("ignore")


So_min = 50
So_max = 150
tau_min = 30
tau_max = 110
r_min = 0.0
r_max = 10
rho_min = -95
rho_max = 0
kappa_min = 0
kappa_max = 200.0
nu_min = 0
nu_max = 50
gamma_min = 0.0
gamma_max = 50
v_min = 5
v_max = 50
min_ = [So_min,tau_min,r_min,rho_min,kappa_min,nu_min,gamma_min,v_min]
max_ = [So_max,tau_max,r_max,rho_max,kappa_max,nu_max,gamma_max,v_max]
sample_heston = np.random.uniform(low=min_, high=max_, size=(100000,8))/100

def Loop(i):
    c = price_heston_fourier(K_=1., alpha_=1, r_ = sample_heston[i,2], tau_=sample_heston[i,1], kappa_=sample_heston[i,4], S0_=np.log(sample_heston[i,0]), theta_=sample_heston[i,5], rho_=sample_heston[i,3], sigma_=sample_heston[i,6], V0_=sample_heston[i,7], L_=50)
    
    return [np.log(sample_heston[i,0]),sample_heston[i,1],c]

heston = tqdm(range(100000), desc="Heston")
input_ = Parallel(n_jobs=num_cores)(delayed(Loop)(i) for i in heston)


def BrentInput(i):
    brent_ = scp.optimize.brentq(lambda x: call_option_brent(x,i[0],1.,0.05,i[1],i[2]),-10,10)
    return brent_

brent_loop = tqdm(input_, desc="Brent")
brent_output = Parallel(n_jobs=num_cores)(delayed(BrentInput)(i) for i in brent_loop)
brent_output = np.array(brent_output)



value_ = {'S0':sample_heston[:,0],'tau':sample_heston[:,1],'r':sample_heston[:,2],'rho':sample_heston[:,3],'kappa':sample_heston[:,4],  
                                 'theta':sample_heston[:,5],  'sigma':sample_heston[:,6], 'V0':sample_heston[:,7],'price':input_,'vol_imp':brent_output}
database = pd.DataFrame(value_)
database.to_csv(deep_cal_dir + '/data/implied-volatility-heston.csv',index=False) 