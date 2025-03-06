import torch
import pyro
import pyro.distributions as dist
from pyro.nn.module import PyroModule
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer.autoguide.initialization import init_to_mean
from pyro import poutine
from typing import Tuple, Optional, Callable
import functools
import torchode # for integrating ODEs with gradient backprop


# some auxiliary functions

def unflatten_triu_matrix(flatQ: torch.Tensor, n: int) -> torch.Tensor:
    """
    Create an upper-triangular matrix from a flat array.
    The flat array may have batch dimensions. The user
    must provide the size of the full matrix n. Note that the resulting
    matrix is of size [batch dims] x (n-1) x (n-1), because the diagonal
    is zero and redundant zeros are removed.
    """
    assert (n * (n-1)) // 2 == flatQ.shape[-1], "given dimension and flat array size don't match"
    triu_indices = torch.triu_indices(row=n-1, col=n-1, offset=0, device=flatQ.device)
    Q = torch.zeros((*flatQ.shape[:-1], n-1, n-1), device=flatQ.device)
    Q[..., triu_indices[0], triu_indices[1]] = flatQ
    return Q


def build_Q_mat_signed(Qsigned: torch.Tensor) -> torch.Tensor:
    """
    Take an upper-triangular matrix Qsigned, and make a Q-matrix.
    The convention is that negative rates go to the lower-triangular part of Q
    and positive rates to the upper -triangular part.

    Parameters
    ----------
    Qsigned : torch.Tensor
        Upper-triangular matrix with signed rates (negative means backwards flow).

    Returns
    -------
    torch.Tensor
        a square, stochastic Q matrix (continuous time) with either Q_{ij} = 0
        or Q_{ji} = 0 for i \neq j

    """
    n = Qsigned.shape[-1] + 1
    Q = torch.zeros((*Qsigned.shape[:-2], n, n), device=Qsigned.device)
    Q[...,:-1,1:] = Qsigned * (Qsigned > 0)
    Q[...,1:,:-1] = Q[...,1:,:-1] - (Qsigned * (Qsigned < 0)).mT
    torch.diagonal(Q, dim1=-1, dim2=-2)[...,:] = -Q.sum(axis=-2)
    return Q



def build_offdiag_Q_mat_signed(Qsigned: torch.Tensor) -> torch.Tensor:
    """ 
    Simular to build_Q_mat_signed, but only use off-diagonal elements.
    Adding the diaginal is done in downstream code.
    The function calls build_offdiag_Q_mat_signed, followed by build_Q_mat
    have the same effect as build_Q_mat_signed
    
    """
    n = Qsigned.shape[-1] + 1
    Qoffdiag = torch.zeros((*Qsigned.shape[:-2], n-1, n), device=Qsigned.device)
    Qoffdiag[...,:,1:] = Qsigned * (Qsigned > 0)
    Qoffdiag[...,:,:-1] = Qoffdiag[...,:,:-1] - (Qsigned * (Qsigned < 0)).mT
    return Qoffdiag



def build_Q_mat(Qoffdiag: torch.Tensor) -> torch.Tensor:
    """
    Construct a stochastic matrix from a matrix of entry-rates Q_ij 
    for i != j.
    
    Parameters
    ----------
        Qoffdiag: torch.Tensor
            off-diagonal elements of stochastic matrix
    
    Returns
    -------
        Q: a square, stochastic matrix (continuous time) Q
    """
    n = Qoffdiag.shape[-1]
    Q = torch.zeros((*Qoffdiag.shape[:-2], n, n), device=Qoffdiag.device)
    torch.diagonal(Q, dim1=-2, dim2=-1)[..., :] = -Qoffdiag.sum(axis=-2)
    Q[..., :-1, :] = Q[..., :-1, :] + torch.triu(Qoffdiag, diagonal=1)
    Q[..., 1:, :] = Q[..., 1:, :] + torch.tril(Qoffdiag)
    return Q



def remove_diagonal(Q: torch.Tensor) -> torch.Tensor:
    """
    Remove diagonal elements from a Q matrix.
    The result is a n-1 times n matrix with the same off-diagonal elements as Q
    This is the "inverse" of build_Q_mat.
    """
    Qoffdiag = torch.triu(Q, diagonal=1)[..., :-1, :]
    Qoffdiag = Qoffdiag + torch.tril(Q, diagonal=-1)[..., 1:, :]
    return Qoffdiag


def build_single_row_Q_mat(q: torch.Tensor, row: int) -> torch.Tensor:
    """
    Construct a stochastic matrix with only as single non-zero row.
    The off-diagonal elements (in-fluxes) are given by q.    

    Parameters
    ----------
    q : torch.Tensor
        off-diagonal elements of the non-zero row of the Q-matrix.
    row : int
        index of non-zero row

    Returns
    -------
    torch.Tensor
        Q matrix with row `row` given by `q`.
    """
    n = q.shape[-1] + 1
    Q = torch.zeros(*q.shape[:-1], n, n, device=q.device)
    Q[..., row, :row] = q[..., :row]
    Q[..., row, row+1:] = q[..., row:]
    torch.diagonal(Q)[..., :] = -Q[..., row, :]
    return Q
    



def log1p_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Compute log(1 + exp(x)) in a numerically stable way.
    This uses torch.logsumexp internally.
    
    TODO: use torch.nn.functional.softplus?

    Parameters
    ----------
    x : torch.Tensor
        input tensor.

    Returns
    -------
    torch.Tensor
        log(1+exp(x)) for input x.
    """
    return torch.logsumexp(torch.stack([torch.zeros_like(x), x]), axis=0)



def lin_interpolate(ts: torch.Tensor, knots: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """
    Make a linear interpolation between vals defined at time points knots.
    Return interpolated values at time points ts.
    """
    eps = torch.tensor(1.0, device=knots.device) # add extended knots
    knots_e = torch.cat([knots[:1]-eps, knots, knots[-1:]+eps])
    knots_t = knots_e.reshape((*knots_e.shape, 1))
    ## degree 0 basis
    H1 = torch.heaviside(knots_t[1:]-ts, torch.tensor(1.0))
    H2 = torch.heaviside(ts-knots_t[:-1], torch.tensor(0.0))
    B0 = H1 * H2
    widths = knots_t[1:] - knots_t[:-1]
    ## degree 1 basis
    omega1 = (ts - knots_t[:-1]) / widths
    omega2 = (knots_t[1:] - ts) / widths
    B1 = B0[:-1] * omega1[:-1] + B0[1:] * omega2[1:]
        
    return vals @ B1


def max_growthrate_diff(eta: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    returns the maximum real part of the eigenvalues of the matrix Q + diag(eta)
    Allows for batch dimensions in Q and eta.
    """
    A = torch.diag_embed(eta, dim1=-1, dim2=-2) + Q
    eivals = torch.linalg.eigvals(A)
    max_re_eival, _ = torch.real(eivals).max(dim=-1)
    return max_re_eival


def max_growthrate(eta: torch.Tensor) -> torch.Tensor:
    """
    returns the maximum of eta
    """
    max_eta, _ = eta.max(dim=-1)
    return max_eta



# dynamical model functions

def dynamic_model_hom(
    time: torch.Tensor,
    rho: torch.Tensor,
    logX0: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline dynamical model: populations are independent, 
    turnover rates are constant
    """
    """ old code (does not work with parallel particles)
    logXt = torch.tensordot(time, rho, dims=0) + logX0
    ## NB: the dims=0 in tensordot indicates that there is no inproduct-like 'contraction'
    """
    logXt = time.unsqueeze(-1) * rho + logX0
    logYt = torch.logsumexp(logXt, axis=-1)
    logFt = logXt - logYt.unsqueeze(-1)
    
    return logFt, logYt



def dynamic_model(
    time: torch.Tensor,
    rho: torch.Tensor,
    eta: torch.Tensor,
    u: torch.Tensor,
    logX0: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamical model: populations are independent,
    parameters are time-dependent
    """
    expt = (1-torch.exp(-u*time)) / u
    logXt = expt.unsqueeze(-1) * (rho-eta) + time.unsqueeze(-1) * eta + logX0
    logYt = torch.logsumexp(logXt, axis=-1)
    logFt = logXt - logYt.unsqueeze(-1)
        
    return logFt, logYt



def dynamic_model_diff_hom(
    time: torch.Tensor,
    rho: torch.Tensor,
    logX0: torch.Tensor,
    Qoffdiag: torch.Tensor ## off-diagonal elements of Q
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamical model allowing for differentiation. 
    Parameters are constant.

    We have to be careful to get the batch dimension right.
    Pyro adds a batch dimension for time to the parameters,
    and might also add more batch dims to the left for parallel
    particles. However, Predictive does not add a batch dim for time.
    """
    batch_shape = logX0.shape[:-1]
    batch_dim = functools.reduce(lambda a, b: a*b, batch_shape, 1)

    Q_plus_rho = torch.diag_embed(rho, dim1=-1, dim2=-2) + build_Q_mat(Qoffdiag)
    X0 = torch.exp(logX0)
    X0_almost_flat = X0.reshape(batch_dim, 1, X0.shape[-1])
    Q_plus_rho_flat = Q_plus_rho.reshape(batch_dim, *Q_plus_rho.shape[-2:])
    # solve the ODEs explicitly
    At = torch.einsum("...i,...jk->...ijk", time, Q_plus_rho_flat)
    logXt = torch.log(torch.einsum("...ij,...j->...i", torch.matrix_exp(At), X0_almost_flat))

    # reshape the output with the correct batch dims
    sol_shape = logXt.shape[-2:]
    if len(batch_shape) > 0 and batch_shape[-1] == 1:
        logXt = logXt.reshape(*batch_shape[:-1], *sol_shape)
    else:
        logXt = logXt.reshape(*batch_shape, *sol_shape)

    # transform the output to total pop and freqs
    logYt = torch.logsumexp(logXt, axis=-1)
    logFt = logXt - logYt.unsqueeze(-1)

    return logFt, logYt



def dynamic_model_diff(
    time: torch.Tensor,
    rho: torch.Tensor,
    eta: torch.Tensor,
    u: torch.Tensor,
    logX0: torch.Tensor,
    Qoffdiag: torch.Tensor ## off-diagonal elements of Q
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamical model allowing for differentiation. 
    Turnover rates are time-dependent.
    WARNING: this is only an exact solution when Q=0 or u=0 (no diff or no time-dependence)
    Otherwise this is a "Magnus approximation of order 1".
    Use the ODE version to get a better approx!

    FIXME: make this compatible with SVI and Predictive!
    """
    Q = build_Q_mat(Qoffdiag)
    expt = (1-torch.exp(-u*time))/u
    At1 = expt.view(*expt.shape, 1, 1) * torch.diag_embed(rho-eta, dim1=-1, dim2=-2)
    At2 = time.view(*time.shape, 1, 1) * (torch.diag_embed(eta, dim1=-1, dim2=-2) + Q)    
    logXt = torch.log(torch.einsum("...ij,...j->...i", torch.matrix_exp(At1 + At2), torch.exp(logX0)))
    logYt = torch.logsumexp(logXt, axis=-1)
    logFt = logXt - logYt.unsqueeze(-1)
    return logFt, logYt



def vf_dynamic_model_diff(
    t: torch.Tensor,
    x: torch.Tensor,
    rho_minus_eta: torch.Tensor,
    u: torch.Tensor,
    Q_plus_eta: torch.Tensor
) -> torch.Tensor:
    """Vector field for the ODE model"""
    Q_plus_eta_x = torch.einsum("...ij,...j->...i", Q_plus_eta, x)
    rho_minus_eta_x = rho_minus_eta * torch.exp(-u*t).unsqueeze(-1) * x
    dxdt =  Q_plus_eta_x + rho_minus_eta_x
    return dxdt


def dynamic_model_diff_ode(
    time: torch.Tensor,
    rho: torch.Tensor,
    eta: torch.Tensor,
    u: torch.Tensor,
    logX0: torch.Tensor,
    Qoffdiag: torch.Tensor ## off-diagonal elements of Q
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamical model allowing for differentiation. 
    Turnover rates are time-dependent.
    batch dimension is derived from logX0
    """
    Q = build_Q_mat(Qoffdiag)
    Q_plus_eta = torch.diag_embed(eta, dim1=-1, dim2=-2) + Q
    rho_minus_eta = rho - eta
    X0 = logX0.exp()
    # get the batch shape, replicate time, reshape params
    batch_shape = logX0.shape[:-1]
    batch_dim = functools.reduce(lambda a, b: a*b, batch_shape, 1)
    # add t0 to the time tensor (required by torchode)
    t0 = torch.full((1,), 1e-5, device=time.device, dtype=time.dtype) 
    # HACK: set t0 slightly less than 0 to make sure that time span (t0, 0.0) works!
    time_ext = torch.cat([t0, time])
    time_repl = time_ext.repeat(batch_dim, 1)
    # flatten all batch dimensions
    X0_flat = X0.reshape(batch_dim, X0.shape[-1])
    rho_minus_eta_flat = rho_minus_eta.reshape(batch_dim, rho_minus_eta.shape[-1])
    Q_plus_eta_flat = Q_plus_eta.reshape(batch_dim, *Q_plus_eta.shape[-2:])
    u_flat = u.reshape(batch_dim)
    def func(t, x):
        return vf_dynamic_model_diff(t, x, rho_minus_eta_flat, u_flat, Q_plus_eta_flat)
    term = torchode.ODETerm(func).to(logX0.device)
    step_method = torchode.Dopri5(term=term)
    step_size_controller = torchode.IntegralController(atol=1e-9, rtol=1e-6, term=term)
    solver = torchode.AutoDiffAdjoint(step_method, step_size_controller)
    ivp = torchode.InitialValueProblem(y0=X0_flat, t_eval=time_repl)
    sol = solver.solve(ivp)
    logXt = sol.ys[...,1:,:].log() # skip the added t0 point
    sol_shape = logXt.shape[-2:]
    # final batch dim is reserved for the time axis but this does not always work with Predictive
    if len(batch_shape) > 0 and batch_shape[-1] == 1:
        logXt = logXt.reshape(*batch_shape[:-1], *sol_shape)
    else:
        logXt = logXt.reshape(*batch_shape, *sol_shape)
    logYt = torch.logsumexp(logXt, axis=-1)
    logFt = logXt - logYt.unsqueeze(-1)
    return logFt, logYt



# classes wrapping model parameters and dynamical functions

class DynamicModel(PyroModule):
    """
    collect model and guide for the independent dynamical model
    The option hom determines if the parameters are constant or
    time-dependent (the default)

    TODO refactor the guide "anti-pattern" and use AutoNormalMessenger.
    The problem is that we currently have to solve the ODEs in the guide
    as well as in the model. This can be expensive.
    """
    diff = False
    
    def __init__(
        self, 
        num_clus: int, 
        hom: bool = False,
        init_fn: Callable = init_to_mean,
        init_scale: float = 0.1,
        numeric_solver: bool = False, # use numeric ODE solver, else Magnus order 1 approx
        growth_rate_penalty: Optional[float] = None, # penalty for positive growth rates (at t=inf)
        use_cuda: bool = False
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

        self.num_clus = num_clus
        self.hom = hom # time-homogeneous model
        
        self.numeric_solver = numeric_solver
        if growth_rate_penalty is not None:
            self.growth_rate_penalty = torch.tensor(growth_rate_penalty, device=self.device)
        else:
            self.growth_rate_penalty = None

        if numeric_solver:
            raise NotImplementedError("numeric ODE solver not yet implemented")
                
        self.auto_guide = AutoMultivariateNormal(self, init_loc_fn=init_fn, init_scale=init_scale)
        
        if use_cuda:
            self.cuda()
        
    def forward(self, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zero, one = torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        # sample the hyper parameters for rho
        loc_rho = pyro.sample("loc_rho", dist.Normal(zero, one))
        scale_rho = pyro.sample("scale_rho", dist.HalfNormal(one))
        # sample the vector rho
        if self.num_clus > 1:
            rho = pyro.sample("rho", dist.Normal(loc_rho.unsqueeze(-1), scale_rho.unsqueeze(-1)).expand((*loc_rho.shape, self.num_clus)).to_event(1))
        else:
            # set event_dim=1 such that batch dims can be added on the left
            rho = pyro.deterministic("rho", loc_rho.reshape(*loc_rho.shape, 1), event_dim=1) # add dimension on the right
        # sample initial population sizes
        scale_logX0 = torch.tensor(5.0, device=self.device) ## FIXME: hard-coded constant
        logX0 = pyro.sample("logX0", dist.Normal(zero, scale_logX0).expand((*loc_rho.shape, self.num_clus,)).to_event(1))
        
        if self.hom:
            # penalize positive growth rates at t=inf
            if self.growth_rate_penalty is not None:
                penalty = torch.relu(max_growthrate(rho) * self.growth_rate_penalty)
                pyro.factor("growth_rate_penalty", -penalty.square())
            # we're done sampling, now compute the predictions
            logweights, logyhat = dynamic_model_hom(time, rho, logX0)
            return logweights, logyhat
            
        # else, sample other parameters
        loc_eta = pyro.sample("loc_eta", dist.Normal(zero, one))
        scale_eta = pyro.sample("scale_eta", dist.HalfNormal(one))
        if self.num_clus > 1:
            eta = pyro.sample("eta", dist.Normal(loc_eta.unsqueeze(-1), scale_eta.unsqueeze(-1)).expand((*loc_eta.shape,self.num_clus)).to_event(1))
        else:
            # set event_dim=1 such that batch dims can be added on the left
            eta = pyro.deterministic("eta", loc_eta.reshape(*loc_eta.shape, 1), event_dim=1) # add dimension on the right
        # sample transition parameter u
        u = pyro.sample("u", dist.HalfNormal(one))

        # penalize positive growth rates at t=inf
        if self.growth_rate_penalty is not None:
            penalty = torch.relu(max_growthrate(eta) * self.growth_rate_penalty)
            pyro.factor("growth_rate_penalty", -penalty.square())

        logweights, logyhat = dynamic_model(time, rho, eta, u, logX0)
        return logweights, logyhat
            
    def guide(self, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        guide_trace = poutine.trace(self.auto_guide).get_trace(time)
        logweights, logyhat = poutine.block(poutine.replay(self, guide_trace))(time)
        return logweights, logyhat

    
class DynamicModelDiff(PyroModule):
    """
    collect model and guide for the dynamical model with differentation
    
    TODO refactor the guide anti-pattern and use AutoNormalMessenger.
    """
    diff = True
    
    def __init__(
        self, 
        num_clus: int, 
        hom: bool = False,
        init_fn: Callable = init_to_mean,
        init_scale: float = 0.1,
        numeric_solver: bool = False, # use numeric ODE solver, else Magnus order 1 approx
        growth_rate_penalty: Optional[float] = None, # penalty for positive growth rates (at t=inf)
        prior_loc_Q: float = 1.0, # loc_Q has prior Exponential(prior_loc_Q)
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        
        self.num_clus = num_clus
        self.hom = hom
        
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
           
        self.auto_guide = AutoMultivariateNormal(
            self, 
            init_loc_fn=init_fn, 
            init_scale=init_scale
        )

        self.numeric_solver = numeric_solver
        if growth_rate_penalty is not None:
            self.growth_rate_penalty = torch.tensor(growth_rate_penalty, device=self.device)
        else:
            self.growth_rate_penalty = None


        self.prior_loc_Q = torch.tensor(prior_loc_Q, device=self.device)
        
        if use_cuda:
            self.cuda()
        
    def forward(self, time: torch.Tensor, shape_Q: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        zero, one = torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        # sample the hyper parameters for rho
        loc_rho = pyro.sample("loc_rho", dist.Normal(zero, one))
        scale_rho = pyro.sample("scale_rho", dist.HalfNormal(one))
        # sample the vector rho
        rho = pyro.sample("rho", dist.Normal(loc_rho.unsqueeze(-1), scale_rho.unsqueeze(-1)).expand((*loc_rho.shape, self.num_clus)).to_event(1))
        # sample initial population sizes
        scale_logX0 = torch.tensor(5.0, device=self.device) ## FIXME: hard-coded constant
        logX0 = pyro.sample("logX0", dist.Normal(zero, scale_logX0).expand((*loc_rho.shape, self.num_clus)).to_event(1))
        # sample hyper parameter for Q
        loc_Q = pyro.sample("loc_Q", dist.Exponential(self.prior_loc_Q))
        # and sample the off-diagonal elements of Q
        # factor in shape parameter (e.g. based on distances between clusters)
        if shape_Q is not None:
            weight_shape_Q = pyro.sample("weight_shape_Q", dist.HalfNormal(one))
            weight_shape_Q = weight_shape_Q.view(*loc_Q.shape, 1, 1)
            loc_Q = loc_Q.view(*loc_Q.shape, 1, 1)
            nodiag_shape_Q = remove_diagonal(shape_Q)
            nodiag_shape_Q_centered = nodiag_shape_Q - nodiag_shape_Q.mean(dim=(-1,-2), keepdim=True)
            rate_Q = loc_Q.reciprocal() * torch.exp(weight_shape_Q * nodiag_shape_Q_centered)
            Qoffdiag = pyro.sample("Qoffdiag", dist.Exponential(rate_Q).to_event(2))
        else:       
            shp = (*loc_Q.shape, self.num_clus-1, self.num_clus)
            rate_Q = loc_Q.view(*loc_Q.shape, 1, 1).reciprocal()
            Qoffdiag = pyro.sample("Qoffdiag", dist.Exponential(rate_Q).expand(shp).to_event(2))

        if not self.hom:
            loc_eta = pyro.sample("loc_eta", dist.Normal(zero, one))
            scale_eta = pyro.sample("scale_eta", dist.HalfNormal(one))
            eta = pyro.sample("eta", dist.Normal(loc_eta.unsqueeze(-1), scale_eta.unsqueeze(-1)).expand((*loc_eta.shape, self.num_clus,)).to_event(1))
            # sample transition parameter u
            u = pyro.sample("u", dist.HalfNormal(one))

        # penalize positive growth rates at t=inf
        if self.growth_rate_penalty is not None:
            Q = build_Q_mat(Qoffdiag)
            match (self.hom, self.hom_diff):
                case (True, True):
                    penalty = torch.relu(max_growthrate_diff(rho, Q) * self.growth_rate_penalty)
                case (True, False): # limit Q(t) = 0
                    penalty = torch.relu(max_growthrate(rho) * self.growth_rate_penalty)
                case (False, True): # limit rho(t) = eta
                    penalty = torch.relu(max_growthrate_diff(eta, Q) * self.growth_rate_penalty)
                case (False, False): # limit Q(t) = 0, limit rho(t) = eta
                    penalty = torch.relu(max_growthrate(eta) * self.growth_rate_penalty)
                case _:
                    raise Exception(f"invalid combination of hom ({self.hom}) hom_diff ({self.hom_diff})")
                            
            pyro.factor("growth_rate_penalty", -penalty.square())

        # model dispatch
        
        match (self.hom, self.numeric_solver):
            case (True, False):           
                logweights, logyhat = dynamic_model_diff_hom(time, rho, logX0, Qoffdiag)
            case (False, False):
                logweights, logyhat = dynamic_model_diff(time, rho, eta, u, logX0, Qoffdiag)
            case (False, True):
                logweights, logyhat = dynamic_model_diff_ode(time, rho, eta, u, logX0, Qoffdiag)
            case _:
                raise Exception(f"invalid combination of hom ({self.hom}) and numeric_solver ({self.numeric_solver})")
        return logweights, logyhat
    
    
    def guide(self, time: torch.Tensor, shape_Q: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:       
        guide_trace = poutine.trace(self.auto_guide).get_trace(time, shape_Q=shape_Q)
        logweights, logyhat = poutine.block(poutine.replay(self, guide_trace))(time, shape_Q=shape_Q)
                
        return logweights, logyhat
    
        