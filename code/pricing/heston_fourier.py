import scipy as scp
from scipy.integrate import quad
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def phi(u,kappa_,S0_,r_,tau_,theta_,rho_,sigma_,V0_):
    """
    Utility
    -------
    Calculate the caracteristic function of Heston model

    Parameters
    ----------
    u : Complex Double
        argument of the ch. function
    kappa_ : Double
        mean reversion speed
    S0_ : Double
        initial price
    r_ : Double
        the risk-free interest rate
    tau_ : Double
        maturity
    theta_ : Double
        the mean reversion level
    rho_ : Double
        correlation coefficient
    sigma_ : Double
        the volatility of the variance
    V0_ : Double
        initial Variance

    Returns
    -------
    Double
        value of the characteristic function of the Heston's model

    """
    gamma_ = np.sqrt((sigma_**2)*(u**2+u*1j)+(kappa_-1j*rho_*sigma_*u)**2)
    a = 1j*u*np.log(S0_) + 1j*u*r_*tau_+kappa_*theta_*tau_*(kappa_-1j*rho_*sigma_*u)/(sigma_**2)
    b = (u**2+1j*u)*V0_/(gamma_*np.cosh(gamma_*tau_/2)/np.sinh(gamma_*tau_/2)+kappa_-1j*rho_*sigma_*u)
    c = np.cosh(gamma_*tau_/2)+((kappa_-1j*rho_*sigma_*u)/gamma_)*np.sinh(gamma_*tau_/2)
    d = 2*kappa_*theta_/(sigma_**2)
    return np.exp(a)*np.exp(-b)/(c**d)


def psi(nu_,alpha_,K_,r_,tau_,kappa_,S0_,theta_,rho_,sigma_,V0_):
    """
    Utility
    -------
    Calculate the integrand of the Heston's model

    Parameters
    ----------
    nu_ : Complex Double
        argument of the ch. function
    alpha_ : int
        parameters of integrand (>=1)
    K_ : Double
        Strike
    r_ : Double
        the risk-free interest rate
    tau_ : Double
        maturity
    kappa_ : Double
        mean reversion speed
    S0_ : Double
        initial price
    theta_ : Double
        the mean reversion level
    rho_ : Double
        correlation coefficient
    sigma_ : Double
        the volatility of the variance
    V0_ : Double
        initial Variance

    Returns
    -------
    double
        integrand of the Heston's model

    """
    k_ = np.log(K_)
    F = phi(nu_-1j*(alpha_+1),kappa_,S0_,r_,tau_,theta_,rho_,sigma_,V0_)*np.exp(-1j*nu_*k_)
    d = (alpha_+1j*nu_)*(alpha_+1+1j*nu_)
    return np.exp(-r_*tau_-alpha_*k_)/np.pi*(F/d).real


def price_heston_fourier(K_,alpha_,r_,tau_,kappa_,S0_,theta_,rho_,sigma_,V0_,L_):
    """
    Utility
    -------
    Calculate the price of the Heston's model by Fourier

    Parameters
    ----------
    K_ : Double
        Strike
    alpha_ : int
        parameters of integrand (>=1)
    r_ : Double
        the risk-free interest rate
    tau_ : Double
        maturity
    kappa_ : Double
        mean reversion speed
    S0_ : Double
        initial price
    theta_ : Double
        the mean reversion level
    rho_ : Double
        correlation coefficient
    sigma_ : Double
        the volatility of the variance
    V0_ : Double
        initial Variance
    L_ : Double
        Level of the domain that we integrate

    Returns
    -------
    Double
        price of Heston by Fourier

    """
    x,w = np.polynomial.legendre.leggauss(156)
    f = lambda nu_: psi(nu_,alpha_,K_,r_,tau_,kappa_,S0_,theta_,rho_,sigma_,V0_)
    t = 0.5*(x + 1)*(L_ - 0) + 0
    
    x1 = np.array([t.T])
    fx = np.asarray(f(x1))

    I = np.dot(fx, w)*0.5*(L_-0)
    #I = quad(lambda nu_: psi(nu_,alpha_,K_,r_,tau_,kappa_,S0_,theta_,rho_,sigma_,V0_) , 0, L_)
    return I[0]