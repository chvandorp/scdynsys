"""
The observation model for a timeseries of cell counts.
Includes the variational distribution of the scale parameter
"""
import pyro
from pyro.nn import PyroModule
import torch
import pyro.distributions as dist
from pyro.distributions import constraints


class LogNormalErrModel(PyroModule):
    def __init__(self, use_cuda=False) -> None:
        super().__init__()
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        
    def forward(self, logyhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_scale = pyro.sample(
            "y_scale",
            dist.Exponential(torch.tensor(1.0, device=self.device))
        )
        with pyro.plate("ydata", logyhat.shape[-1]):
            ysim = pyro.sample("yobs", dist.LogNormal(logyhat, y_scale), obs=y)
                
        return ysim

    def guide(self, logyhat: torch.Tensor, y: torch.Tensor) -> None:
        y_scale_vari_loc = pyro.param(
            "y_scale_vari_loc",
            torch.tensor(0.0, device=self.device)
        )
        y_scale_vari_scale = pyro.param(
            "y_scale_vari_scale",
            torch.tensor(0.1, device=self.device),
            constraint=constraints.positive
        )
        pyro.sample(
            "y_scale",
            dist.LogNormal(y_scale_vari_loc, y_scale_vari_scale)
        )   
    
    