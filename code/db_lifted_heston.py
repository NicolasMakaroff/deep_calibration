from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm 
import numpy as np
import os
import sys
from os.path import dirname as up

# Important directories
code_dir = os.path.dirname(os.getcwd())
deep_cal_dir = os.path.dirname(os.path.dirname(os.getcwd()))

# Allows to import my own module
sys.path.insert(0, code_dir)

from pricing.liftedheston import Pricer_Lifted_Heston

r_min = 0.0
r_max = 10
rho_min = -95
rho_max = 0
lambda_min = 0
lambda_max = 200.0
nu_min = 0
nu_max = 50
theta_min = 0.0
theta_max = 50
v_min = 5
v_max = 50
min_ = [r_min,rho_min,lambda_min,nu_min,theta_min,v_min]
max_ = [r_max,rho_max,lambda_max,nu_max,theta_max,v_max]
sample_heston = np.random.uniform(low=min_, high=max_, size=(40000,6))/100

input_ = []
def processInput(i):
    output_heston = []
    func1 = Pricer_Lifted_Heston
    func2 = input_.append
    for s in np.arange(0.5,1.6,0.1):
        for j in [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0]:
            c = func1(1,0.03,s,j,rho=sample_heston[i,1],lamb=sample_heston[i,2],theta=sample_heston[i,4],nu=sample_heston[i,3],V0=sample_heston[i,5],N=20,rN=2.5,alpha=0.1+1/2,M=200,L_=50)
            func2([s,j,c])
    return input_

num_cores = multiprocessing.cpu_count()
heston = tqdm(range(40000), desc="Lifted Heston")
input_ = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in heston)
output_heston = np.array(output_heston)
shape_heston = np.reshape(output_heston,(40000,88))

def BrentInput(iter_map):
    brent_output = [] 
    for i in iter_map:
        brent_ = scp.optimize.brentq(lambda x: pricing_model.call_option_brent(x,i[0],1.,0.05,i[1],i[2]),-10,10)
        brent_output.append(brent_)
    return brent_output

brent_loop = tqdm(input_, desc="Brent")
brent_output = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in brent_loop)

brent_output = np.array(brent_output)

value_ = {'r':sample_heston[:,0], 'rho':sample_heston[:,1], 'lambda':sample_heston[:,2],'nu':sample_heston[:,3],'theta':sample_heston[:,4],'v0':sample_heston[:,5]}
database = pd.DataFrame(value_)
database['vol_imp'] = brent_output.tolist()
database.to_csv('heston_img.csv',index=False)
