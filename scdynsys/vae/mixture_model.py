"""
module defining a Gaussian mixture model in Pyro, to be used in other models
"""
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer.autoguide.initialization import init_to_sample
from pyro import poutine


class GaussMix(PyroModule):
    def __init__(
        self, 
        z_dim: int,
        num_clus: int,
        weighted: bool = True,
        fixed_scales: bool = False, 
        use_cuda: bool = False
    ) -> None:
        """
        PyroModule packaging the ingredients of a Gaussian Mixture Model (GMM)
        
        Parameters
        ----------
        z_dim : int
            dimension of the domain of the GMM distribution.
        num_clus : int
            number of components of the GMM.
        weighted : bool, optional
            Should the mixture weights be included. Weights may be excluded 
            for dynamical models. The default is True.
        fixed_scales : bool, optional
            We can fix the scales (i.e. covariance matrices) of the components
            to identity matrices. The default is False.
        use_cuda : bool, optional
            move tensors to the GPU. The default is False.
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.num_clus = num_clus
        self.weighted = weighted
 
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        
        # mixture-model parameters
        self.concentration = torch.tensor(2.0, device=self.device)
        self.fixed_scales = fixed_scales
        
        #self.auto_guide = AutoDelta(self, init_loc_fn=init_to_sample)
        
        # if fixed_scales, we sample clus_scales and clus_corrs from a Delta, 
        # and they should be blocked in the guide!
        hide = ["clus_scales", "clus_corrs"] if fixed_scales else []
        self.auto_guide = AutoMultivariateNormal(poutine.block(self, hide=hide), init_loc_fn=init_to_sample)
                
        if use_cuda:
            self.cuda()
        
    def forward(self) -> tuple[torch.Tensor, ...]:
        """
        Sample locations, correlations, scales and (optionally) weights.
        
        Returns
        -------
        clus_locs : torch.Tensor
            the sampled location (i.e. mean) vectors of the components.
        clus_chol_fact : torch.Tensor
            the sampled Cholesky factors (L) of the components. The covariance
            matrix Sigma is given by L.T L
        weights : torch.Tensor
            the weights of the components. This is only returned if
            self.weighted is True
        """
            
        ## sample location vectors
        clus_locs = pyro.sample(
            "clus_locs",
            dist.Normal(
                torch.zeros((self.num_clus, self.z_dim), device=self.device),
                torch.ones((self.num_clus, self.z_dim), device=self.device)
            ).to_event(2)
        )

        if not self.fixed_scales:
            ## sample covariance matrices
            clus_scales = pyro.sample(
                "clus_scales",
                dist.Exponential(
                    torch.ones((self.num_clus, self.z_dim), device=self.device)
                ).to_event(2)
            )
            clus_corrs = pyro.sample(
                "clus_corrs",
                dist.LKJCholesky(self.z_dim, self.concentration).expand((self.num_clus,)).to_event(1)
            )
        else:    
            ## sample fixed covariance matrices
            clus_scales = pyro.sample(
                "clus_scales",
                dist.Delta(
                    torch.ones((self.num_clus, self.z_dim), device=self.device)
                ).to_event(2)
            )
            clus_corrs = pyro.sample(
                "clus_corrs",
                dist.Delta(
                    torch.eye(self.z_dim, device=self.device).expand(self.num_clus, -1, -1)
                ).to_event(3)
            )


                                                
        clus_chol_fact = torch.matmul(clus_scales.diag_embed(), clus_corrs)
                
        if not self.weighted:
            return clus_locs, clus_chol_fact
                    
        ## else...
        weights = pyro.sample(
            "weights",
            dist.Dirichlet(torch.full((self.num_clus,), 0.5, device=self.device))
        )
        return clus_locs, clus_chol_fact, weights
    
    def guide(self) -> tuple[torch.Tensor, ...]:
        """
        Guide method. Replays the model with sampled parameters and 
        returns the result.

        Returns
        -------
        tuple[torch.Tensor, ...]
            locations, Cholesky factors and (optionally) weights.
        """
        guide_trace = poutine.trace(self.auto_guide).get_trace()
        
        if self.fixed_scales:
            pyro.sample(
                "clus_scales",
                dist.Delta(
                    torch.ones((self.num_clus, self.z_dim), device=self.device)
                ).to_event(2)
            )
            pyro.sample(
                "clus_corrs",
                dist.Delta(
                    torch.eye(self.z_dim, device=self.device).expand(self.num_clus, -1, -1)
                ).to_event(3)
            )

        
        return poutine.block(poutine.replay(self, guide_trace))()
     