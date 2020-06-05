import os
import sys

code_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, code_dir)

from complex import *
from quadrature import GaussLegendreQuadrature1D
import torch
import torch.nn as nn

class TorchHestonFourier(nn.Module):

    def __init__(self,S0, K, r, T, alpha, precision):
        super().__init__()
        self. S0 = 1.
        self.K = K
        self.r = r
        self.T = T
        self.precision = precision
        self.alpha = alpha

    def TorchPhi(u,kappa_,S0_,r_,tau_,theta_,rho_,sigma_,V0_):
        size = kappa_.size()
        gamma_ = (Complex(u ** 2,u)*((torch.pow(sigma_,2)))+(Complex(kappa_,u*-1*rho_*sigma_)**2))**(1/2)
        a = Complex(torch.zeros(size,dtype=torch.float64),u*torch.log(S0_)) + Complex(torch.zeros(size,dtype=torch.float64),u*r_*tau_) + (Complex(kappa_,torch.zeros(kappa_.size(),dtype=torch.float64)) - Complex(torch.zeros(size,dtype=torch.float64),u*rho_*sigma_))*(kappa_*theta_*tau_)*(1/(sigma_**2))
        b = Complex((u**2)*V0_,u*V0_).div(gamma_ * TorchCosh((gamma_*tau_)*(1/2)).div(TorchSinh((gamma_*tau_)*(1/2))) + (Complex(kappa_,torch.zeros(kappa_.size()))- Complex(torch.zeros(size),rho_*sigma_*u))  )
        c = TorchCosh((gamma_*tau_)*(1/2)) + (Complex(kappa_,torch.zeros(kappa_.size()))- Complex(torch.zeros(size),rho_*sigma_*u)).div(gamma_)*TorchSinh((gamma_*tau_)*(1/2)) 
        d = 2*kappa_*theta_/(sigma_**2)
        return Torchexp(a) * Torchexp(-b)*(c**d)**(-1)


    def TorchPsi(nu,alpha,K,kappa,S0,r,tau,theta,rho,sigma,V0):
        k = torch.log(K)
        F = TorchPhi(Complex(nu,torch.ones(nu.size())*-(alpha+1)),kappa,S0,r,tau,theta,rho,sigma,V0) * Torchexp(Complex(torch.zeros(nu.size()),-nu*k))
        d = Complex(alpha,nu) * Complex(alpha+1,nu)
        return ((torch.exp(-r*tau-alpha*k)/math.pi)*(F.div(d))).re


    def forward(self,kappa,theta,rho,sigma,V0):
        """
        
        Args :
        ------
            kappa: speed of mean reversion
            theta: long term vol
            rho: spot-vol correlation
            sigma: vol of vol
            V0: forward variance curve
            
        Return:
        -------
            Heston Price
        """
        f = lambda nu : TorchPsi(nu,self.alpha,self.K,self.r,self.T,kappa,self.S0,theta,rho,sigma,V0)
        gg = GaussLegendreQuadrature1D()
        res = gg.forward(f,0,self.precision)
        return res