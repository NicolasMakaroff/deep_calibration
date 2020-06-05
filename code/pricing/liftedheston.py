import scipy as scp
from scipy.special import gamma
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

num_cores = multiprocessing.cpu_count()

np.seterr(divide='ignore', invalid='ignore')

from numba import jit, njit
# Characteristic function of the Lifted Heston model see Slides 85-87
@jit
def F(u,v,var4,lamb,var5):
    return 0.5*(u**2-u)+(var4*u-lamb)*v+var5*v**2
@jit
def g(t,V0,lamb_theta,c_gammas,gammas):
    return V0 +lamb_theta*np.dot(c_gammas,1-np.exp(-t*gammas))

@jit
def Loop1(psi,mat,N,delta,gammas,M,omega,c,var4,lamb,var5):
    # Iteration for approximation of psi - see Slide 87

    var2 = 1+delta*gammas
    var3 = mat/(var2)
    for k in range (1,M+1):
        psi[k,:] = (var3)*(psi[k-1,:]+delta*F(omega,np.dot(c,psi[k-1,:]),var4,lamb,var5)*mat)
    return psi

@jit
def Loop2(t,M,g_0,T,V0,lamb_theta,c_gammas,gammas):
    for k in range(1,M+2):
        g_0[0,k-1]=g(T-t[k-1],V0,lamb_theta,c_gammas,gammas)
    return g_0

@jit
def Ch_Lifted_Heston(omega,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M):
    """
    Characteristic function for the lifted heston model.
    
    Arguments:
    ----------
        omega: argument of the ch. function
        S0: Initial price
        rho,lamb,theta,nu,V0: parameters Lifted Heston
        N: number of factors in the model
        rN: constant used to define weights and mean-reversions
        alpha: H+1/2 where H is the Hurst index
        T: maturity
        M: number of steps in the time discretization to calculate ch. function

    Returns:
    --------
        phi
    """

    # to make sure we calculate ch. function and not moment gen. function
    i=complex(0,1)
    omega=i*omega
    
    # Definition of weights and mean reversions in the approximation
    h=np.linspace(0,N-1,N)
    rpowerN=np.power(rN,h-N/2) 
    # weights
    alpha_1 = 1 - alpha
    var1 = rN**(alpha_1)-1
    c=(var1)*(rpowerN**(alpha_1))/(gamma(alpha)*gamma(alpha_1+1))
    # mean reversions 
    gammas=((alpha_1)/(alpha_1+1))*((rN**(alpha_1+1)-1)/(var1))*rpowerN

    lamb_theta = lamb*theta
    c_gammas = c/gammas
    # Definition of the initial curve
    #g = lambda t: V0 +lamb_theta*np.dot(c_gammas,1-np.exp(-t*gammas))
    
    
    # Time steps for the approximation of psi         
    delta = T/M;
    t=np.linspace(0,M,M+1)
    t = t * delta
    
    var4 = rho*nu
    var5 = .5*nu**2
    # Function F
    #F = lambda u,v : 0.5*(u**2-u)+(var4*u-lamb)*v+var5*v**2
    

    psi=np.zeros((M+1,N),dtype=complex)
    mat = np.ones(N)
    psi = Loop1(psi,mat,N,delta,gammas,M,omega,c,var4,lamb,var5)
    # Invert g_0 to calculate phi - see Slide 87
    
    #g_0[0,:] = g(T-t[:])


    g_0=np.zeros((1,M+1))

    
    g_0 = Loop2(t,M,g_0,T,V0,lamb_theta,c_gammas,gammas)

    Y=np.zeros((1,M+1),dtype=complex)
    phi=0

    Y=F(omega,np.dot(c,psi.transpose()),var4,lamb,var5)*g_0
   
    
    # Trapezoid rule to calculate phi
    weights=np.ones(M+1)*delta
    weights[0]=delta/2
    weights[M]=delta/2
    phi=np.dot(weights,Y.transpose())
    
    phi=np.exp(omega*np.log(S0)+phi)
    
    return phi

@jit
def psi_Lifted_Heston(K_,r_,omega,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M):
    k_ = np.log(K_/S0)
    phi = Ch_Lifted_Heston(omega,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M)
    F = phi*np.exp(-1j*omega.real*k_)
    aa = (1+1j*omega.real)
    d = aa*(aa+1)
    return np.exp(-r_*T-k_)/np.pi*(F/d).real

@jit
def f(K_,r_,x,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M):
    return psi_Lifted_Heston(K_,r_,x-2*1j,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M)

@jit
def Pricer_Lifted_Heston(K_,r_,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M,L_):
    """
    
    Arguments:
    ----------
        K_: strike
        r_: rate
        S0: Initial price
        rho,lamb,theta,nu,V0: parameters Lifted Heston
        N: number of factors in the model
        rN: constant used to define weights and mean-reversions
        alpha: H+1/2 where H is the Hurst index
        T: maturity
        M: number of steps in the time discretization to calculate ch. function
        L_: (int) Level of the domain that we integrate

    Returns:
    --------
        phi
    """
    a = 10
    b = L_
    #f = lambda x: psi_Lifted_Heston(K_,r_,x-2*1j,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M)
    deg = 200
    x,w = np.polynomial.legendre.leggauss(deg)
    t = 0.5*(x + 1)*(b - a) + a
    gauss = 0
    for i in range(len(t)):
        gauss = gauss + w[i] * f(K_,r_,t[i],S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M)
    #I = scp.integrate.quad(lambda x: psi_Lifted_Heston(K_,r_,x-2*1j,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M) , 0, L_)
    return (gauss* 0.5*(b - a) )

def create_vol_map():
    func1 = Pricer_Lifted_Heston
    func2 = input_.append
    for s in np.arange(0.5,1.6,0.1):
        for j in [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0]:
            c = func1(s,r_=0.05,S0=1,T=j,rho=-0.7,lamb=1.3,theta=0.3,nu=0.3,V0=0.2,N=20,rN=2.5,alpha=0.1+1/2,M=200,L_=50)
            func2([s,j,c])
    return input_
