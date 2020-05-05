import numpy as np
import scipy.stats as scp

def s_simul(kappa_,theta_,sigma_,rho_,r_,T_,L_,V0_,S0_):
    """
    Utility
    -------
    Calculate a simulate final value of the asset's price

    Parameters
    ----------
    kappa_ : Double
        mean reversion speed
    theta_ : Double
        the mean reversion level
    sigma_ : Double
        the volatility of the variance
    rho_ : Double
        correlation coefficient
    r_ : Double
        the risk-free interest rate
    T_ : Double
        maturity
    L_ : int
        number of step until reaching the final asset price
    V0_ : Double
        initial Variance
    S0_ : Double
        initial price

    Returns
    -------
    Double
        a simulate final value of the asset's price

    """
    V = [V0_]
    X = [np.log(S0_)]
    delta_t_ = T_/L_
    delta_W_ = np.array([scp.norm.rvs(loc = 0, scale = np.sqrt(delta_t_), size = L_)]).T
    delta_W_orth_ = np.array([scp.norm.rvs(loc = 0, scale = np.sqrt(delta_t_), size = L_)]).T
    for i in range(L_):
        Vi_ = V[i]+kappa_*(theta_-V[i])*delta_t_ + sigma_*np.sqrt(max(V[i],0))*(rho_*delta_W_[i][0] + np.sqrt(1-rho_**2)*delta_W_orth_[i][0])
        V.append(Vi_)
        Xi_ = X[i] + (r_-1/2*V[i])*delta_t_ + np.sqrt(max(V[i],0)) * delta_W_[i][0]
        X.append(Xi_)
    return X[L_]

def f(X_,K_):
    """
    Utility
    -------
    Calculate the payoff

    Parameters
    ----------
    X_ : Double
        an asset's price
    K_ : Double
        the strike

    Returns
    -------
    Double
        the payoff (>= 0)

    """
    return max(np.exp(X_)-K_,0)
    
def monte_carlo(kappa_,theta_,sigma_,rho_,r_,T_,L_,V0_,S0_,K0_,N_):
    """
    Utility
    -------
    Calculate the average of the payoff

    Parameters
    ----------
    kappa_ : Double
        mean reversion speed
    theta_ : Double
        the mean reversion level
    sigma_ : Double
        the volatility of the variance
    rho_ : Double
        correlation coefficient
    r_ : Double
        the risk-free interest rate
    T_ : Double
        maturity
    L_ : int
        number of step until reaching the final asset price
    V0_ : Double
        initial Variance
    S0_ : Double
        initial price
    K0_ : TYPE
        the strike
    N_ : int
        number of simulation to do

    Returns
    -------
    Double
        the average of the payoff

    """
    X = []
    for i in range(N_):
        X.append(f(s_simul(kappa_,theta_,sigma_,rho_,r_,T_,L_,V0_,S0_),K0_))
    return sum(X)/N_
    
def price_heston_mc_legacy(kappa_,theta_,sigma_,rho_,r_,T_,L_,V0_,S0_,K0_,N_):
    """
    Utility
    -------
    Calculate the price of the Heston model by Monte Carlo

    Parameters
    ----------
    esp_ : Double
        the average of the payoff
    r_ : Double
        the risk-free interest rate
    T_ : Double
        the maturity

    Returns
    -------
    Double
        the price of the Heston model by Monte Carlo

    """
    esp_ = monte_carlo(kappa_,theta_,sigma_,rho_,r_,T_,L_,V0_,S0_,K0_,N_)
    return np.exp(-r_*T_)*esp_