#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn 


from gpytorch.utils.broadcasting import _pad_with_singletons


class GaussLegendreQuadrature1D(nn.Module):
    """
    Implements Gauss-Legendre quadrature for integrating a function in batch mode.

    This is implemented as a Module because Gauss-Legendre quadrature has a set of locations and weights that it
    should initialize one time, but that should obey parent calls to .cuda(), .double() etc.
    """

    def __init__(self, num_locs=156):
        super().__init__()
        self.num_locs = num_locs

        locations, weights = self._locs_and_weights(num_locs)

        self.locations = locations
        self.weights = weights

    def _apply(self, fn):
        self.locations = fn(self.locations)
        self.weights = fn(self.weights)
        return super(GaussLegendreQuadrature1D, self)._apply(fn)

    def _locs_and_weights(self, num_locs):
        """
        Get locations and weights for Gauss-Legendre quadrature.
        
        """
        locations, weights = np.polynomial.legendre.leggauss(20)
        locations = torch.Tensor(locations)
        weights = torch.Tensor(weights)
        return locations, weights

    def forward(self, func, a=0,b=1):
        """
        Runs Gauss-Legendre quadrature on the callable func.

        Args:
        -----
            - func (callable): Function to integrate
            - a and b: (int): boundaries of for the intergation
            
        Returns:
        --------
            - Result of integrating func
        """
        lower_bound = torch.Tensor([a])
        higher_bound =  torch.Tensor([b])

        locations = _pad_with_singletons(self.locations, num_singletons_before=0, num_singletons_after=lower_bound.dim())
        t = 0.5*(locations + 1)*(higher_bound - lower_bound) + lower_bound

        x1 = t
        fx = func(x1)
        weights = _pad_with_singletons(self.weights, num_singletons_before=0, num_singletons_after=fx.dim() - 1)
        res = (fx * weights)
        res = res.sum(tuple(range(self.locations.dim())))

        return res * 0.5 * (higher_bound - lower_bound)
    
