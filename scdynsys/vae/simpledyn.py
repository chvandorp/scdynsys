"""
implement a number of VAE to harmonize, dimension reduce, and cluster the TRM data
Goal: combine dynamic modeling with deep learning by making the GMM weights
time-dependent (according to some dynamical model)
"""

import torch
import pyro
from pyro.nn.module import PyroModule

from .err_model import LogNormalErrModel



class EncapsulatedDyn(PyroModule):
    """
    Fit various dynamical models (from dyn module) to cell count data
    """
    def __init__(self, num_clus: int, dyn: PyroModule) -> None:
        super().__init__()

        ## dims
        self.num_clus = num_clus
        
        ## model module
        self.dyn_model = dyn
        self.dyn_guide = dyn.guide
        
        ## error model
        self.err_model = LogNormalErrModel()
        
    def model(
        self,
        y: torch.Tensor, ## cell counts (short time series)
        ytime: torch.Tensor, ## time of each y data point
    ) -> torch.Tensor:
        logweights, logyhat = self.dyn_model(ytime)

        self.err_model(logyhat, y)
            
        pyro.deterministic("logweights", logweights)
        pyro.deterministic("logyhats", logyhat)            
            
        return logyhat
    
    def guide(
        self,
        y: torch.Tensor, ## cell counts (short time series)
        ytime: torch.Tensor, ## time of each y data point
    ) -> torch.Tensor:
        logweights, logyhat = self.dyn_guide(ytime)
        
        self.err_model.guide(logyhat, y)
        
        return logyhat

