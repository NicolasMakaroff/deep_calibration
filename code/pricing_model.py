import scipy.stats as stats
from scipy import integrate
import scipy as scp
import numpy as np


def d_j(j_,S_,K_,r_,sigma_,T_):
    return (np.log(S_/K_) + (r_ + np.power(-1,j_-1))*0.5*sigma_**2)*(T_)/(sigma_*np.sqrt(T_))


def call_option(S_,K_,r_,sigma_,T_):
    return S_ * stats.norm.cdf(d_j(1,S_,K_,r_,sigma_,T_))  - K_*np.exp(-r_*(T_)) * stats.norm.cdf(d_j(2,S_,K_,r_,sigma_,T_))


def phi(u,kappa_,S0_,r_,tau_,theta_,rho_,sigma_,V0_):
    var1 = kappa_-1j*rho_*sigma_*u
    var2 = kappa_*theta_/(sigma_**2)
    gamma_ = np.sqrt((sigma_**2)*(u**2+u*1j)+(var1)**2)
    cosh = np.cosh(gamma_*tau_/2)
    a = 1j*u*np.log(S0_) + 1j*u*r_*tau_+tau_*(var1)*var2
    b = (u**2+1j*u)*V0_/(gamma_*cosh+kappa_-1j*rho_*sigma_*u)
    c = cosh+((var1)/gamma_)*np.sinh(gamma_*tau_/2)
    d = 2*var2
    return np.exp(a)*np.exp(-b)/(c**d)
  
def psi(nu_,alpha_,K_,r_,tau_,kappa_,S0_,theta_,rho_,sigma_,V0_):
    k_ = np.log(K_)
    F = phi(nu_-1j*(alpha_+1),kappa_,S0_,r_,tau_,theta_,rho_,sigma_,V0_)*np.exp(-1j*nu_*k_)
    aa = (alpha_+1j*nu_)
    d = aa*(aa+1)
    return np.exp(-r_*tau_-alpha_*k_)/np.pi*(F/d).real



def C(K_,alpha_,r_,tau_,kappa_,S0_,theta_,rho_,sigma_,V0_,L_):
    I = scp.integrate.quad(lambda nu_: psi(nu_,alpha_,K_,r_,tau_,kappa_,S0_,theta_,rho_,sigma_,V0_) , 0, L_)
    return I[0]


def d(S_, K_, r_,sigma_ ,tau_):
    d1_ = 1 / (sigma_ * np.sqrt(tau_)) * ( np.log(S_/K_) + (r_ + sigma_**2/2) * tau_)
    d2_ = d1_ - sigma_ * np.sqrt(tau_)
    return d1_, d2_

def call_option_brent(sigma_ ,S_, K_, r_,tau_,price):
    return (S_ * stats.norm.cdf(d_j(1,S_,K_,r_,sigma_,tau_))  - K_*np.exp(-r_*(tau_)) * stats.norm.cdf(d_j(2,S_,K_,r_,sigma_,tau_)) - price)

def Brent(S_,K_,r_,tau_, sigma0_,sigma1_,sigma2_ ,price_, epsilon_):
    volatility = [sigma0_,sigma1_,sigma2_]
    g = []
    d1_,d2_ = d(S_, K_, r_,sigma0_ ,tau_)
    g.append(call_option_brent(S_, K_, r_,sigma0_ ,tau_,d1_,d2_)-price_)
    d1_,d2_ = d(S_, K_, r_,sigma1_ ,tau_)
    g.append(call_option_brent(S_, K_, r_,sigma1_ ,tau_,d1_,d2_)-price_)
    d1_,d2_ = d(S_, K_, r_,sigma2_ ,tau_)
    g.append(call_option_brent(S_, K_, r_,sigma2_ ,tau_,d1_,d2_)-price_)
    i=2
    while ((np.abs(volatility[i]-volatility[i-1]) > epsilon_) & (np.abs(volatility[i-1]-volatility[i-2]) > epsilon_)) :
        if (volatility[i]==volatility[i-1]):
            sigma_ = volatility[i-1]-g[i-1]*(volatility[i-1]-volatility[i-2])/(g[i-1]-g[i-2])
        else :
            aux0_ = volatility[i]*g[i-1]*g[i-2]/((g[i]-g[i-1])*(g[i]-g[i-2]))
            aux1_ = volatility[i-1]*g[i-2]*g[i]/((g[i-1]-g[i-2])*(g[i-1]-g[i]))
            aux2_ = volatility[i-2]*g[i-1]*g[i]/((g[i-2]-g[i-1])*(g[i-2]-g[i]))
            sigma_ = aux0_ + aux1_ + aux2_
        volatility.append(sigma_)
        i=i+1
        d1_,d2_ = d(S_, K_, r_,volatility[i] ,tau_)
        g.append(call_option_brent(S_, K_, r_,volatility[i] ,tau_,d1_,d2_)-price_)
    return volatility