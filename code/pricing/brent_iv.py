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
    d1_ = 1 / (sigma_ * np.sqrt(tau_)) * (np.log(S_/K_) + (r_ + sigma_**2/2) * tau_)
    d2_ = d1_ - sigma_ * np.sqrt(tau_)
    return d1_, d2_

def call_BS(S_, K_, r_,sigma_ ,tau_,price):
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
    return scp.norm.cdf(d1_) * S_ - scp.norm.cdf(d2_) * K_ * np.exp(-r_ *tau_) - price

def call_vega(S_, tau_ , d1_):
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
    return S_ * scp.norm.pdf(d1_) * np.sqrt(tau_)


def brent(S_,K_,r_,tau_, sigma0_,sigma1_,price_, epsilon_):
    """
    Utility
    -------
    Calculate the implied volatility thanks to brent's method

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
    sigma1_ : Double
        second volatility (< sigma0_)
    price_ : Double
        price in the market
    epsilon_ : double
        precision of the method

    Returns
    -------
    Double
        the implied volatility of the model

    """
    volatility = [sigma0_,sigma0_,sigma1_]
    g = []
    d1_,d2_ = d(S_, K_, r_,sigma0_ ,tau_)
    g.append(call_BS(S_, K_, r_,sigma0_ ,tau_,d1_,d2_)-price_)
    g.append(g[0])
    d1_,d2_ = d(S_, K_, r_,sigma1_ ,tau_)
    g.append(call_BS(S_, K_, r_,sigma1_ ,tau_,d1_,d2_)-price_)
    i=2
    while ((np.abs(volatility[i]-volatility[i-1]) > epsilon_) and (np.abs(volatility[i]-volatility[i-2]) > epsilon_)) :
        if (volatility[i]==volatility[i-1]):
            sigma_ = volatility[i-1]-g[i-1]*(volatility[i-1]-volatility[i-2])/(g[i-1]-g[i-2])
        elif (volatility[i]==volatility[i-2]):
            sigma_ = volatility[i]-g[i]*(volatility[i]-volatility[i-1])/(g[i]-g[i-1])
        elif (volatility[i-1]==volatility[i-2]):
            sigma_ = volatility[i]-g[i]*(volatility[i]-volatility[i-2])/(g[i]-g[i-2])
        else :
            aux0_ = volatility[i]*g[i-1]*g[i-2]/((g[i]-g[i-1])*(g[i]-g[i-2]))
            aux1_ = volatility[i-1]*g[i-2]*g[i]/((g[i-1]-g[i-2])*(g[i-1]-g[i]))
            aux2_ = volatility[i-2]*g[i-1]*g[i]/((g[i-2]-g[i-1])*(g[i-2]-g[i]))
            sigma_ = aux0_ + aux1_ + aux2_
        volatility.append(sigma_)
        i=i+1
        d1_,d2_ = d(S_, K_, r_,volatility[i] ,tau_)
        g.append(call_BS(S_, K_, r_,volatility[i] ,tau_,d1_,d2_)-price_)
    return volatility[-1]
