"""
Bayesian Neural Networks
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample

from typing import Literal


class BayesianLinear(PyroModule):
    def __init__(self, dim_in: int, dim_out: int, scale: torch.Tensor, bias: bool=True) -> None:
        super().__init__()
        self.weight = PyroSample(prior=dist.Normal(torch.tensor(0.0), scale).expand([dim_in, dim_out]).to_event(2))
        self.bias = PyroSample(prior=dist.Normal(torch.tensor(0.0), scale).expand([dim_out]).to_event(1)) if bias else None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.bias:
            return x @ self.weight
        return x @ self.weight + self.bias
    
    
class RegularizedLinear(PyroModule):
    """
    The goal is to add regularization to the weights and biases of a linear layer.
    This is achieved by defining tensors `weight` and `bias`, and then sampling
    auxiliary tensors `rand_weight` and `rand_bais` from a Normal of Laplace
    distribution, with has observation `obs=weight` and `obs=bias`. That way,
    we're not actually samping random weights and biases, but are adding a penalty 
    to the optimization target for large weights or biases.
    """
    
    dist_dispatch = {
        "l1": dist.Laplace,
        "l2": dist.Normal
    }
    def __init__(
        self, 
        name: str, 
        dim_in: int, 
        dim_out: int, 
        reg_scale: float, 
        reg_norm: Literal["l1", "l2"]="l2",
        bias: bool=True,
        use_cuda: bool=False
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        self.name = name
        self.dim_in = dim_in
        self.dim_out = dim_out
        ran_unif_weight = 2*torch.rand((dim_in, dim_out), device=self.device) - 1
        unif_scale = torch.tensor(dim_in, device=self.device).sqrt()
        self.weight = PyroParam(ran_unif_weight / unif_scale)
        self.bias = None
        if bias:
            ran_unif_bias = 2*torch.rand((dim_out,), device=self.device) - 1
            self.bias = PyroParam(ran_unif_bias / unif_scale)
        # parameters for the regularization distribution
        self.scale = torch.tensor(reg_scale, device=self.device)
        self.loc = torch.tensor(0.0, device=self.device)
        self.reg_norm = reg_norm
        # put tensors on GPU?
        if use_cuda:
            self.cuda()
            
    def sample(self) -> None:
        distr = self.dist_dispatch[self.reg_norm](self.loc, self.scale)
        pyro.factor(
            f"{self.name}.rand_weight", 
            distr.expand([self.dim_in, self.dim_out]).to_event(2).log_prob(self.weight)
        )
        
        if self.bias is None:
            return 
        
        # else, also sample bias
        pyro.factor(
            f"{self.name}.rand_bias",
            distr.expand([self.dim_out]).to_event(1).log_prob(self.bias)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #self.sample() # FIXME: this may cause a pyro warning abount dependencies
        
        if self.bias is None:
            return x @ self.weight
        
        # else, also add bias to output tensor
        return x @ self.weight + self.bias
    