import numpy as np

def linear(S_,K_,r_,tau_,price_):
    """
    Utility
    -------
    Calculate the implied volatility thanks to linearisation

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
    price_ : Double
        price in the market

    Returns
    -------
    Double
        implied Volatility of the model

    """
    a = tau_/(2*np.sqrt(np.pi*2)) * (S_ + K_ *np.exp(-r_*tau_))
    b = np.sqrt(tau_)*(1/2*(S_-K_*np.exp(-r_*tau_)) - price_)
    c = 1/(np.sqrt(2*np.pi))*(S_-K_*np.exp(-r_*tau_))*(np.log(S_/K_) + r_*tau_)
    return (-b + np.sqrt(max(0,b**2-4*a*c)))/(2*a)
