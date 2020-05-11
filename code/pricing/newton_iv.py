import numpy as np
import scipy.stats as scp


def d(S_, K_, r_,sigma_ ,tau_):
    """
    
    Utility
    -------
    Calculate the coefficients in the normal distribution in the model of Black Scholes

    Parameters
    ----------
    S_ : Double
        the price
    K_ : Double
        the strike
    r_ : Double
        the risk-free interest rate
    sigma_ : Double
        the volatility
    tau_ : Double
        the maturity

    Returns
    -------
    d1_,d2_ : tuple(Double)
        coefficient in the normal distribution in the model of BS

    """
    d1_ = 1 / (sigma_ * np.sqrt(tau_)) * ( np.log(S_/K_) + (r_ + sigma_**2/2) * tau_)
    d2_ = d1_ - sigma_ * np.sqrt(tau_)
    return d1_, d2_

def call_BS(S_, K_, r_,sigma_ ,tau_):
    """
    Utility
    -------
    Calculate the price by BS model
    

    Parameters
    ----------
    S_ : Double
        the price
    K_ : Double
        the strike
    r_ : Double
        the risk-free interest rate
    sigma_ : Double
        the volatility
    tau_ : Double
        the maturity
    d1_ : Double
        first coefficient in the normal distribution in the model of BS
    d2_ : Double
        second coefficient in the normal distribution in the model of BS

    Returns
    -------
    Double
        price by BS model.

    """
    d1_ = 1 / (sigma_ * np.sqrt(tau_)) * (np.log(S_/K_) + (r_ + sigma_**2/2) * tau_)
    d2_ = d1_ - sigma_ * np.sqrt(tau_)
    return scp.norm.cdf(d1_) * S_ - scp.norm.cdf(d2_) * K_ * np.exp(-r_ *tau_)

def call_vega(S_, K_, r_,sigma_ ,tau_):
    """
    Utility
    -------
    Calculate the call vega :the differentiate of call_BS

    Parameters
    ----------
    S_ : Double
        the price
    tau_ : Double
        the maturity
    d1_ : Double
        first coefficient in the normal distribution in the model of BS

    Returns
    -------
    Double
        Call Vega

    """
    d1_ = 1 / (sigma_ * np.sqrt(tau_)) * ( np.log(S_/K_) + (r_ + sigma_**2/2) * tau_)
    return S_ * scp.norm.pdf(d1_) * np.sqrt(tau_)


def newton_raphson(S_,K_,r_,tau_, sigma0_ ,price_, epsilon_):
    """
    Utility
    -------
    Calculate the implied volatility thanks to newton's method

    Parameters
    ----------
    S_ : Double
        the price
    K_ : Double
        the strike
    r_ : Double
        the risk-free interest rate
    tau_ : Double
        the maturity
    sigma0_ : Double
        initial volatility
    price_ : Double
        price in the market
    epsilon_ : double
        precision of the method

    Returns
    -------
    sigma_ : double
        the implied volatility of the model

    """
    g = call_BS(S_, K_, r_,sigma0_ ,tau_)-price_
    sigma_ = sigma0_ - g/call_vega(S_, K_, r_,sigma0_ ,tau_)
    i = 1
    while np.abs(g) > epsilon_ :
        sigma0_ = sigma_
        g = call_BS(S_, K_, r_,sigma0_ ,tau_)-price_
        sigma_ = sigma0_ - g/(call_vega(S_, K_, r_,sigma0_ ,tau_)+1e-15)
    return sigma_
