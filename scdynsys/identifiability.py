import numpy as np
from typing import Literal

from . import simulate as simul


## method for getting a point estimate from Stan MCMC samples
## these samples are given as a dictionary returned by "stan_variables()"


def get_ML_idx(par_dict):
    """
    Get the index in the MCMC chain of the maximum log-likelihood value.
    """
    log_lik = np.sum(par_dict["log_lik"], axis=1)
    max_ll_idx = np.argmax(log_lik)
    return max_ll_idx

def get_MAP_idx(diag_dict):
    """
    Get the index in the MCMC chain of the maximum a posteriori value.
    This makes use of Stan "method_variables" output.
    """
    lp = np.concatenate(diag_dict["lp__"].T) # chains are separate: concatenate!
    max_lp_idx = np.argmax(lp)
    return max_lp_idx

def get_rand_idx(par_dict):
    """
    To get a random sample, we just pick a random index.
    """
    chain_len = par_dict["log_lik"].shape[0]
    return np.random.choice(chain_len)


def get_median_point_est(par_dict):
    med_par_dict = {
        k : np.median(v, axis=0) for k, v in par_dict.items()
    }
    # p might not be a simplex: rescale
    p0 = med_par_dict["p0"]
    p0 = p0 / np.sum(p0)
    pd = {
        "logx0" : med_par_dict["logY0"] + np.log(p0),
        "rho0"  : med_par_dict["rho"],
        "eta"   : med_par_dict["eta"],
        "Q"     : med_par_dict["Q"],
        "u"     : med_par_dict["u"],
        "sigma" : med_par_dict["sigma_c"],
        "phi"   : 1 / med_par_dict["phi_inv_f"],
    }
    return pd


def get_point_est(par_dict, diag_dict, method="MAP"):
    """
    Get a point estimate using the method specified.
    """
    match method:
        case "ML":
            idx = get_ML_idx(par_dict)
        case "MAP":
            idx = get_MAP_idx(diag_dict)
        case "rand":
            idx = get_rand_idx(par_dict)
        case "median":
            return get_median_point_est(par_dict)
        case _:
            raise Exception(f"invalid method {method}, should be ML or MAP")
        
    pd = {
        "logx0" : par_dict["logY0"][idx] + np.log(par_dict["p0"][idx]),
        "rho0"  : par_dict["rho"][idx],
        "eta"   : par_dict["eta"][idx],
        "Q"     : par_dict["Q"][idx],
        "u"     : par_dict["u"][idx],
        "sigma" : par_dict["sigma_c"][idx],
        "phi"   : 1 / par_dict["phi_inv_f"][idx],
    }
    return pd



## Method for generating pseudo-data usign a point estimate

SolverMethod = Literal["ODE", "magnus1", "magnus2"]

def generate_pseudo_data(model, pd_gt, t0, tobs, flow_sample_sizes, count_scaling, 
                         method: SolverMethod="ODE", n_sim: int=200):
    ts = np.linspace(t0, tobs[-1], n_sim)
    method_dispatch = {
        "ODE" : simul.solve_trm_ivp,
        "magnus1" : simul.solve_trm_ivp_magnus1_approx,
        "magnus2" : simul.solve_trm_ivp_magnus2_approx,
    }

    if method not in method_dispatch:
        raise Exception(f"invalid solve method {method}")
    
    solver = method_dispatch[method]
    
    match model:        
        case "M3" | "M1": 
            x_gt = solver(ts, t0, pd_gt["logx0"], pd_gt["rho0"], pd_gt["rho0"], pd_gt["u"], pd_gt["Q"])
            y_gt = np.sum(x_gt, axis=0)
            xobs = solver(tobs, t0, pd_gt["logx0"], pd_gt["rho0"], pd_gt["rho0"], pd_gt["u"], pd_gt["Q"])
        case "M4" | "M2":
            x_gt = solver(ts, t0, pd_gt["logx0"], pd_gt["rho0"], pd_gt["eta"], pd_gt["u"], pd_gt["Q"])
            y_gt = np.sum(x_gt, axis=0)
            xobs = solver(tobs, t0, pd_gt["logx0"], pd_gt["rho0"], pd_gt["eta"], pd_gt["u"], pd_gt["Q"])
            
    yobs, kobs = simul.gen_trm_data(xobs, pd_gt["sigma"], pd_gt["phi"], flow_sample_sizes, count_scaling)

    dset = {
        "t0" : t0,
        "ts" : ts,
        "x_gt" : x_gt,
        "y_gt" : y_gt,
        "tobs" : tobs,
        "xobs" : xobs,
        "yobs" : yobs,
        "kobs" : kobs,
    }

    return dset




def plot_pseudo_data(t0, ts, x_gt, y_gt, tobs, xobs, yobs, kobs, count_scaling=1.0, celltype_names=None, axs=None):
    import matplotlib.pyplot as plt
    if axs is None:
        fig, axs = plt.subplots(2, 5 , figsize=(14,7), sharex=True)
    else:
        fig = axs.flat[0].get_figure()
    ax = axs.flatten()[0]
    ax.plot(ts, y_gt*count_scaling, color='k', linewidth=0.5)
    ax.set_yscale('log')
    ax.scatter(tobs, yobs, s=5, color='k')

    num_pops = x_gt.shape[0]

    flow_sample_sizes = np.sum(kobs, axis=1)

    for i in range(num_pops):
        ax = axs.flatten()[i+1]
        pi_gt = x_gt[i] / y_gt
        ax.plot(ts, pi_gt, linewidth=0.5)
        ax.set_ylim(2e-3, 1.0)
        ax.set_yscale('log')
        p_obs = kobs[:,i] / flow_sample_sizes
        ax.scatter(tobs, p_obs, s=5)
        if celltype_names is not None:
            ax.set_title(celltype_names[i])


    return fig, axs




## fit the Stan model to psuedo-data

def fit_model_to_psuedo_data(model, t0, ts, x_gt, y_gt, tobs, xobs, yobs, kobs, count_scaling=1.0, stan_kwargs=None):
    match model:
        case "M1":
            time_dependence = "autonomous"
            population_structure = "independent"
        case "M2":
            time_dependence = "non_autonomous"
            population_structure = "independent"
        case "M3":
            time_dependence = "autonomous"
            population_structure = "unrestricted"
        case "M4":
            time_dependence = "non_autonomous"
            population_structure = "unrestricted"
        case _:
            raise Exception(f"invalid model {model}, must be M1, M2, M3, or M4")
    
    from .stanfit import fit_stan_model

    default_stan_kwargs = dict(
        refresh=1, chains=5,
        iter_warmup=500, iter_sampling=500, 
        adapt_delta=0.9, step_size=0.01,
    )

    if stan_kwargs is not None:
        default_stan_kwargs.update(stan_kwargs)

    fit, inits, stan_data = fit_stan_model(
        count_times = tobs,
        count_data = yobs,
        freq_times = tobs, 
        freq_data = kobs.T,
        t0 = t0,
        obs_model_freq = "dirichlet_multinomial",
        time_dependence = time_dependence,
        population_structure = population_structure,
        count_scaling = count_scaling,
        growth_rate_penalty = 100.0,
        **default_stan_kwargs
    )

    par_dict = fit.stan_variables()
    diag_dict = fit.method_variables()

    return par_dict, inits, stan_data, diag_dict




## methods for generating a range of parameter values

def generate_range(x, par_range, num, scale):
    if scale == 'log':
        fx = np.log(x)
        finv = np.exp
        fr = np.log(1+par_range)
    elif scale == 'linear':
        fx = x
        finv = lambda y: y
        fr = par_range * x
    else:
        raise Exception(f"invalid scale {scale}")
        
    fx_pert = np.linspace(fx - fr, fx + fr, num)
    return finv(fx_pert)

def perturb_Q(Q, par_range, num, i, j):
    qij = Q[i,j]
    qij_perturbed = generate_range(qij, par_range, num, 'log')

    Q_perturbed = []
    for x in qij_perturbed:
        Qp = Q.copy()
        Qp[i,j] = x
        Qp[j,j] += -x + qij
        Q_perturbed.append(Qp)

    return Q_perturbed


def perturb_param(pd_gt, parname, par_range, num, i=None, j=None):
    import copy
    pd_perts = [copy.deepcopy(pd_gt) for _ in range(num)]
    match parname:
        case "Q":
            if i is None or j is None:
                raise Exception("for Q the kwargs i and j are required")
            Q_perts = perturb_Q(pd_gt["Q"], par_range, num, i, j)
            for n, pQ in enumerate(Q_perts):
                pd_perts[n]["Q"] = pQ
        case "rho0" | "eta" | "logX0":
            if i is None:
                raise Exception(f"for {parname} the kwarg i is required")
            x = pd_gt[parname][i]
            x_perts = generate_range(x, par_range, num, 'linear')
            for n, px in enumerate(x_perts):
                pd_perts[n][parname][i] = px
        case "u" | "sigma" | "phi":
            x = pd_gt[parname]
            x_perts = generate_range(x, par_range, num, 'log')
            for n, px in enumerate(x_perts):
                pd_perts[n][parname] = px
        case _:
            raise Exception(f"invalid parameter name {parname} given")
    return pd_perts
