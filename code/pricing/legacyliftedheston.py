import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy.integrate as integrate
from scipy.special import gamma

# Characteristic function of the Lifted Heston model see Slides 85-87
def Ch_Lifted_Heston(omega,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M):
    # omega = argument of the ch. function
    # S0 = Initial price
    # rho,lamb,theta,nu,V0 = parameters Lifted Heston
    # N = number of factors in the model
    # rN = constant used to define weights and mean-reversions
    # alpha = H+1/2 where H is the Hurst index
    # T = maturity
    # M = number of steps in the time discretization to calculate ch. function

    # to make sure we calculate ch. function and not moment gen. function
    i=complex(0,1)
    omega=i*omega
    
    # Definition of weights and mean reversions in the approximation
    h=np.linspace(0,N-1,N)
    rpowerN=np.power(rN,h-N/2) 
    # weights
    c=(rN**(1-alpha)-1)*(rpowerN**(1-alpha))/(gamma(alpha)*gamma(2-alpha))
    # mean reversions 
    gammas=((1-alpha)/(2-alpha))*((rN**(2-alpha)-1)/(rN**(1-alpha)-1))*rpowerN
    
    # Definition of the initial curve
    g = lambda t: V0+lamb*theta*np.dot(c/gammas,1-np.exp(-t*gammas))
    
    
    # Time steps for the approximation of psi         
    delta = T/M;
    t=np.linspace(0,M,M+1)
    t = t * delta
    
    # Function F
    F = lambda u,v : 0.5*(u**2-u)+(rho*nu*u-lamb)*v+.5*nu**2*v**2
    
    
    # Iteration for approximation of psi - see Slide 87
    psi=np.zeros((M+1,N),dtype=complex)
    
    for k in range (1,M+1):
        psi[k,:] = (np.ones(N)/(1+delta*gammas))*(psi[k-1,:]+delta*F(omega,np.dot(c,psi[k-1,:]))*np.ones(N))
        
    
    # Invert g_0 to calculate phi - see Slide 87
    g_0=np.zeros((1,M+1))
    
    for k in range(1,M+2):
        g_0[0,k-1]=g(T-t[k-1])
    
    
    Y=np.zeros((1,M+1),dtype=complex)
    phi=0
    
    Y=F(omega,np.dot(c,psi.transpose()))*g_0
   
    
    # Trapezoid rule to calculate phi
    weights=np.ones(M+1)*delta
    weights[0]=delta/2
    weights[M]=delta/2
    phi=np.dot(weights,Y.transpose())
    
    phi=np.exp(omega*np.log(S0)+phi)
    
    return phi



def psi_Lifted_Heston(K_,r_,omega,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M):
    k_ = np.log(K_)
    phi = Ch_Lifted_Heston(omega,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M)
    F = phi*np.exp(-1j*omega.real*k_)
    aa = (1+1j*omega.real)
    d = aa*(aa+1)
    return np.exp(-r_*T-k_)/np.pi*(F/d).real

def Legacy_Pricer_Lifted_Heston(K_,r_,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M,L_):
    I = integrate.quad(lambda x: psi_Lifted_Heston(K_,r_,x-2*1j,S0,T,rho,lamb,theta,nu,V0,N,rN,alpha,M) , 0, L_)
    return I[0]

