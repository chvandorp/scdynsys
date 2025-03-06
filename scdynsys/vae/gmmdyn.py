"""
implement a number of VAE to harmonize, dimension reduce, and cluster the TRM data
Goal: combine dynamic modeling with deep learning by making the GMM weights
time-dependent (according to some dynamical model)
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.nn.module import PyroModule
from typing import Tuple, Optional, Literal
from pyro.distributions.kl import kl_divergence
from pyro import poutine
from pyro.infer import Predictive

from .mixture_model import GaussMix
from .layers_gmm import DecoderX, RegularizedDecoderX, FullRegularizedDecoderX
from .layers_gmm import EncoderZ, RegularizedEncoderZ, FullRegularizedTimeEncoderZ
from .err_model import LogNormalErrModel


def cluster_distance_matrix(clus_loc: torch.Tensor) -> torch.Tensor:
    """
    Compute the distance matrix between components.
    This uses the Euclidean distance between the means of the components.
    """
    return torch.cdist(clus_loc, clus_loc)



class VAEgmmdyn(PyroModule):
    """
    GMM VAE model

    Use a Gaussian Mixture model on the latent space (Z) to cluster the data.
    The Gaussian mixture model consists of a number of multivariate normals.
    The variational posterior (guide) predicts location and covariance of Z | X
    and the log-odds of cluster membership
    """    
    gmm: bool = True
    dyn: bool = True

    def __init__(
        self, 
        data_dim: int, 
        z_dim: int, 
        hidden_dim: int,
        num_clus: int, 
        num_batch: int, 
        dyn: PyroModule,
        time_scale: float, 
        anchored_clusters: Optional[list[int]] = None,
        reg_scale: Optional[float] = None,
        reg_scale_batch: Optional[float] = None,
        reg_norm: Literal["l1", "l2"] = "l2",
        fixed_scales: bool = False,
        distance_guided_diff: bool = False,
        use_cuda: bool = False
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

        ## dims and settings
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.num_clus = num_clus
        self.time_scale = time_scale
        self.num_batch = num_batch        
        self.anchored_clusters = [] if anchored_clusters is None else anchored_clusters
        self.reg_scale = reg_scale
        if reg_scale is not None and reg_scale_batch is None:
            self.reg_scale_batch = reg_scale
        else:
            self.reg_scale_batch = reg_scale_batch
        self.reg_norm = reg_norm
        self.fixed_scales = fixed_scales
        # check that distance_guided_diff is compatible with the dynamic model
        if not dyn.diff and distance_guided_diff:
            raise ValueError("distance_guided_diff option is incompatible with the dynamic model")
        self.distance_guided_diff = distance_guided_diff

        # create the encoder and decoder networks
        if reg_scale is not None:
            self.decoder_x = FullRegularizedDecoderX(
                z_dim, 
                hidden_dim, 
                data_dim, 
                num_batch, 
                reg_scale, 
                reg_scale_batch,
                use_cuda=use_cuda            
            )
            self.encoder_z = FullRegularizedTimeEncoderZ(
                z_dim, 
                hidden_dim, 
                data_dim, 
                num_batch,
                time_scale,
                reg_scale, 
                reg_scale_batch,
                use_cuda=use_cuda            
            )

        elif reg_scale_batch is not None:
            self.decoder_x = RegularizedDecoderX(
                z_dim, 
                hidden_dim, 
                data_dim, 
                num_batch, 
                reg_scale_batch,
                reg_norm=reg_norm,
                use_cuda=use_cuda
            )
            self.encoder_z = RegularizedEncoderZ(
                z_dim, 
                hidden_dim, 
                data_dim, 
                num_batch, 
                reg_scale_batch,
                reg_norm=reg_norm,
                use_cuda=use_cuda
            )
        else:
            self.decoder_x = DecoderX(z_dim, hidden_dim, data_dim, num_batch)
            self.encoder_z = EncoderZ(z_dim, hidden_dim, data_dim, num_batch)

        # the dynamic model
        self.dyn_model = dyn
        
        # the mixture model
        self.mix = GaussMix(z_dim, num_clus, weighted=False, fixed_scales=fixed_scales, use_cuda=use_cuda)
        
        # the error model for total cell count
        self.err_model = LogNormalErrModel(use_cuda=use_cuda)

        if use_cuda: ## put tensors into GPU RAM
            self.cuda()

    # define the model p(x|z)p(z)
    def model(
        self,
        x: torch.Tensor, ## expression data (mini-batches)
        xtime: torch.Tensor, ## time of each x data point: index in utime
        s: torch.Tensor, ## experimental batch
        cluster_hint: torch.Tensor,
        N: torch.Tensor, ## as we're doing mini-batching, we have to re-scale the log prob
        persuasiveness: torch.Tensor,
        y: torch.Tensor, ## cell counts (short time series)
        ytime: torch.Tensor, ## time of each y data point: index in utime
        utime: torch.Tensor, ## unique observation times
    ) -> None:
        ## use GaussMix object to sample parameters for the mixture model
        clus_locs, clus_chol_fact = self.mix()

        ## compute distance matrix between components
        if self.distance_guided_diff:
            dist_mat = cluster_distance_matrix(clus_locs)
            pyro.deterministic("dist_mat", dist_mat)
            logweights, logyhat = self.dyn_model(utime, shape_Q=dist_mat)
        else:
            logweights, logyhat = self.dyn_model(utime)

        logweights = logweights[..., xtime, :]
        logyhat = logyhat[..., ytime]
            
        
        # compare y to yhat
        self.err_model(logyhat, y)
        
        ## penalty for overlapping distributions
        comp_dists = [
            dist.MultivariateNormal(
                clus_locs[...,c,:], 
                scale_tril=clus_chol_fact[...,c,:,:]
            ) for c in range(self.num_clus)
        ]
        
        kl_dists = torch.stack([
            kl_divergence(comp_dists[i], comp_dists[j]) 
            for i in range(self.num_clus)
            for j in range(self.num_clus)
            if i != j
        ])

        pyro.factor("penalty", -(1/kl_dists).sum())
        
        ## regularization of the encoder network
        self.encoder_z.sample()
        self.decoder_x.sample()

        with pyro.plate("unobserved", x.shape[-2]), pyro.poutine.scale(scale=N/x.shape[-2]):
            # integrate out cluster
            mix = dist.Categorical(logits=logweights)
            comp = dist.MultivariateNormal(clus_locs, scale_tril=clus_chol_fact)
            z = pyro.sample("latent", dist.MixtureSameFamily(mix, comp))

        ## decode the latent code z outsize of the plate (module could be Bayesian)
        x_loc, x_scale = self.decoder_x(z, s)
                
        with pyro.plate("xdata", x.shape[-2]), pyro.poutine.scale(scale=N/x.shape[-2]):
            ## score against actual expression data
            pyro.sample("xobs", dist.Normal(x_loc, x_scale).to_event(1), obs=x)
            

        # add a term to the likelihood to anchor clusters to certain groups of cells
        for i, c in enumerate(self.anchored_clusters):
            comp = dist.MultivariateNormal(clus_locs, scale_tril=clus_chol_fact)
            lps = comp.log_prob(z.unsqueeze(-2).expand((-1,self.num_clus,-1)))
            wlps = logweights + lps
            not_c = torch.arange(self.num_clus, device=self.device) != c
            logodds = torch.stack([wlps[:,c], torch.logsumexp(wlps[:,not_c], dim=-1)], dim=-1)
            with pyro.plate(f"hint_{i}", x.shape[-2]), pyro.poutine.scale(scale=N/x.shape[-2]*persuasiveness):
                # define a mixture that can handle "unknown" cases
                hint_mix = dist.Categorical(logits=logodds) ## weights
                hint_probs = torch.tensor([[0, 0.5, 0.5], [0.5, 0, 0.5]], device=self.device)
                hint_comp = dist.Categorical(probs=hint_probs)
                pyro.sample(f"cluster_hint_{i}", dist.MixtureSameFamily(hint_mix, hint_comp), obs=cluster_hint[:,i])
                

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(
        self,
        x: torch.Tensor, ## expression data (mini-batches)
        xtime: torch.Tensor, ## time of each x data point: index in utime
        s: torch.Tensor, ## experimental batch
        cluster_hint: torch.Tensor,
        N: torch.Tensor, ## as we're doing mini-batching, we have to re-scale the log prob
        persuasiveness: torch.Tensor,
        y: torch.Tensor, ## cell counts (short time series)
        ytime: torch.Tensor, ## time of each y data point: index in utime
        utime: torch.Tensor ## unique observation times
    ) -> None:
        ## use the guide method of GaussMix
        clus_locs, clus_chol_fact = self.mix.guide()

        # compute weights using the dynamic model (using its guide method)
        if self.distance_guided_diff:
            ## compute distance matrix between components
            dist_mat = cluster_distance_matrix(clus_locs)
            logweights, logyhat = self.dyn_model.guide(utime, shape_Q=dist_mat)
        else:
            logweights, logyhat = self.dyn_model.guide(utime)

        logweights = logweights[..., xtime, :]
        logyhat = logyhat[..., ytime]
                    
        # guide for the error model of y
        self.err_model.guide(logyhat, y)
            
        # use the encoder to get the parameters used to define q(z|x)
        #z_loc, z_scale = self.encoder_z(x, s)
        z_loc, z_scale = self.encoder_z(x, utime[xtime], s) # testing: use time info for encoding
        
        # sample latent vectors
        with pyro.plate("unobserved", x.shape[-2]), pyro.poutine.scale(scale=N/x.shape[-2]):
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


    # define a helper function for reconstructing samples
    def reconstruct_sample(self, x: torch.Tensor, time: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # encode sample x
        z_loc, z_scale = self.encoder_z(x, time, s) # testing: use time info for encoding
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the latent sample
        x_loc, x_scale = self.decoder_x(z, s)
                
        x = dist.Normal(x_loc, x_scale).sample()
        return x

    # use the VAE as a dimension reduction algo
    def dimension_reduction(self, x: torch.Tensor, time: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # encode sample x
        z_loc, z_scale = self.encoder_z(x, time, s)
        return z_loc

    # use the VAE as a classifier
    def classifier(
        self,
        x: torch.Tensor,
        xtime: torch.Tensor,
        utime: torch.Tensor,
        s: torch.Tensor,
        method: Literal["map", "sample"]
    ) -> torch.Tensor:
        # encode sample x
        time = utime[xtime]
        z_loc, z_scale = self.encoder_z(x, time, s)
        locs, chols = self.mix.guide()

        if self.distance_guided_diff:
            ## compute distance matrix between components
            dist_mat = cluster_distance_matrix(locs)
            logweights, logyhat = self.dyn_model.guide(utime, shape_Q=dist_mat)
        else:
            logweights, logyhat = self.dyn_model.guide(utime)

        
        loglikes = torch.t(torch.cat([
            dist.MultivariateNormal(locs[c], scale_tril=chols[c]).log_prob(z_loc).unsqueeze(0)
            for c in range(self.num_clus)
        ]))
        logodds = loglikes + logweights[xtime]
        if method == "map":
            return torch.argmax(logodds, dim=-1)
        elif method == "sample":
            return dist.Categorical(logits=logodds).sample()
        else:
            raise Exception("invalid sampling method")


    def posterior_sample(
        self,
        time: torch.Tensor, ## sampling time
        s: torch.Tensor, ## experimental batch
        n: int = 1000, ## number of samples
        clus: Optional[int] = None ## sample only from one cluster (time independent)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take a posterior data sample at a given time point.
        """

        zzs, xxs = [], []
        
        # first handle simpler case
        
        if clus is not None:
            for i in range(n):
                locs, chols = self.mix.guide()
                zz = dist.MultivariateNormal(
                    locs[clus],
                    scale_tril=chols[clus]
                ).sample()

                zzs.append(zz)
                # decode z into x
                xx_loc, xx_scale = self.decoder_x(zz, s)
                xx = dist.Normal(xx_loc, xx_scale).to_event(1).sample()
                xxs.append(xx)

            return torch.cat(zzs), torch.cat(xxs)

        # else...

        def dyn_model_wrap(time):
            logweights, logyhats = self.dyn_model(time)
            pyro.deterministic("logweights", logweights)
            pyro.deterministic("logyhats", logyhats)
            
        pred_dyn = Predictive(dyn_model_wrap, guide=self.dyn_model.guide, num_samples=n, parallel=True)
        sam_dyn = pred_dyn(time)
        
        def gmm_model_wrap():
            locs, chols = self.mix()
            pyro.deterministic("locs", locs)
            pyro.deterministic("chols", chols)

        pred_gmm = Predictive(gmm_model_wrap, guide=self.mix.guide, num_samples=n, parallel=True)
        sam_gmm = pred_gmm()

        logweights = sam_dyn["logweights"]
        clus_vec = dist.Categorical(logits=logweights).sample()
        
        locs = sam_gmm["locs"]
        chols = sam_gmm["chols"]
            
        sel_locs = locs.gather(1, clus_vec.view(-1,1,1).expand(-1,1,locs.shape[2])).squeeze(-2)
        sel_chols = chols.gather(1, clus_vec.view(-1,1,1,1).expand(-1,1,*chols.shape[2:])).squeeze(-3)

        zz = dist.MultivariateNormal(sel_locs, scale_tril=sel_chols).sample()
            
        # decode z into x
        xx_loc, xx_scale = self.decoder_x(zz, s)
        xx = dist.Normal(xx_loc, xx_scale).to_event(1).sample()
        
        return zz, xx, clus_vec
    

    def _posterior_sample_old(
        self,
        size: int,
        time: torch.Tensor, ## sampling time
        s: Optional[torch.Tensor] = None, ## batch for each sample
        clus: Optional[int] = None ## sample only from one cluster (time independent)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: remove this method in favor of the new one
        """
        logweights, logyhat = self.dyn_model.guide(time)
        locs, chols = self.mix.guide()
        if clus is None:
            clus_vec = dist.Categorical(logits=logweights).sample((size,))
            zz = torch.cat([ ## FIXME: use torch.stack
                dist.MultivariateNormal(
                    locs[c],
                    scale_tril=chols[c]
                ).sample().unsqueeze(0)
                for c in clus_vec
            ])
        else:
            zz = dist.MultivariateNormal(
                locs[clus],
                scale_tril=chols[clus]
            ).sample((size,))
        # decode z into x
        if s is None:
            s = torch.zeros((size, time.shape[0], self.num_batch), device=self.device)
        ## FIXME: this won't work when time is a scalar
        xx_loc, xx_scale = self.decoder_x(zz, s)
                
        xx = dist.Normal(xx_loc, xx_scale).to_event(1).sample()
        if clus is None:
            return zz, xx, clus_vec
        # else...
        return zz, xx
    

    def _sample_trajectories_old(
        self, 
        ts: torch.Tensor, 
        n: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sample trajectories of the dynamic model with parameters
        sampled from the variational distribution.

        TODO: remove this method in favor of the new one
        """
        wss, yss, yss_sim = [], [], []
        for i in range(n):
            # run dynamic model with posterior sample
            logweights, logyhats = self.dyn_model.guide(ts)
            weights = logweights.exp()
            yhats = logyhats.exp()
            # simulate observations with the error model
            trace = poutine.trace(self.err_model.guide).get_trace(logyhats, None)
            yhats_sim = poutine.replay(self.err_model, trace=trace)(logyhats, None)
            # append samples to list
            wss.append(weights)
            yss.append(yhats)
            yss_sim.append(yhats_sim)
        ws = torch.stack(wss)
        ys = torch.stack(yss)
        ys_sim = torch.stack(yss_sim)
        return ws, ys, ys_sim
    

    def sample_trajectories(
        self, 
        ts: torch.Tensor, 
        n: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sample trajectories of the dynamic model with parameters
        sampled from the variational distribution.

        HACK: we use Predictive to sample from the posterior.
        However, Predictive samples from the prior during the first
        few samples, which might lead to NaNs and downstream exceptions.
        Therefore we use nan_to_num to replace NaNs with zeros.
        """
        
        def traj_model(ts):
            if self.distance_guided_diff:
                locs, chols = self.mix()
                ## compute distance matrix between components
                dist_mat = cluster_distance_matrix(locs)
                logweights, logyhats = self.dyn_model(ts, shape_Q=dist_mat)
            else:
                logweights, logyhats = self.dyn_model(ts)
            logyhats = torch.nan_to_num(logyhats, nan=0.0)
            ysims = self.err_model(logyhats, None)
            pyro.deterministic("ysims", ysims)
            pyro.deterministic("yhats", logyhats.exp())
            pyro.deterministic("weights", logweights.exp())

        def traj_guide(ts):
            if self.distance_guided_diff:
                locs, chols = self.mix.guide()
                ## compute distance matrix between components
                dist_mat = cluster_distance_matrix(locs)
                logweights, logyhats = self.dyn_model.guide(ts, shape_Q=dist_mat)
            else:
                logweights, logyhats = self.dyn_model.guide(ts)
            self.err_model.guide(logyhats, None)
        
        pred = pyro.infer.Predictive(traj_model, guide=traj_guide, num_samples=n, parallel=True)
        sams = pred(ts)

        # predictive adds a batch dimension to the samples
        ws = sams["weights"].squeeze(1)
        ys = sams["yhats"].squeeze(1)
        ys_sim = sams["ysims"].squeeze(1)

        return ws, ys, ys_sim


    def settings_dict(self):
        """
        returns a dictionary witch chosen settings
        """
        penalize_growth = self.dyn_model.growth_rate_penalty is not None
        growth_rate_penalty = self.dyn_model.growth_rate_penalty.detach().item() if penalize_growth else 0.0
        settings = {
            "z_dim" : self.z_dim,
            "hidden_dim" : self.hidden_dim,
            "data_dim" : self.data_dim,
            "num_clus" : self.num_clus,
            "num_batch" : self.num_batch,
            "time_scale" : self.time_scale,
            "reg_scale" : self.reg_scale,
            "reg_norm" : self.reg_norm,
            "fixed_scales" : self.fixed_scales,
            "reg_scale_batch" : self.reg_scale_batch,
            "anchored_clusters" : self.anchored_clusters,
            "hom" : self.dyn_model.hom,
            "diff" : self.dyn_model.diff,
            "numeric_solver" : self.dyn_model.numeric_solver,
            "penalize_growth" : penalize_growth,
            "growth_rate_penalty" : growth_rate_penalty,
            "static" : False
        }
        return settings

