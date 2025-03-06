import cmdstanpy
import numpy as np
from scipy.special import logsumexp
from typing import Optional
import os
import arviz
import copy


obs_model_codes = {
    "multinomial" : 0,
    "dirichlet_multinomial" : 1,
}

time_dependence_codes = {
    "autonomous" : 0,
    "non_autonomous" : 1,
    "sigmoid" : 2,
    "diff_non_autonomous" : 3,
    "full_non_autonomous" : 4,
}

stan_model_dir = os.path.join(os.path.dirname(__file__), 'stan_models')


def prepare_stan_timeseries_data(
    count_times: np.ndarray, 
    count_data: np.ndarray, 
    freq_times: np.ndarray, 
    freq_data: np.ndarray, 
    t0: float,
    num_sim_times: int = 100,
    count_scaling: float = 1.0,
) -> dict:
    # puts observations in the right format
        
    tx_count = [(t, x) for t, x in zip(count_times, count_data) if t >= t0]
    tx_count.sort()
        
    num_count = len(tx_count)
    t_count = np.array([t for t, x in tx_count])
    x_count = np.array([x for t, x in tx_count]) / count_scaling
    
    num_celltypes = freq_data.shape[0]
    tx_freq = [(t, x) for t, x in zip(freq_times, freq_data.T) if t >= t0]
    num_freq = len(tx_freq)
    tx_freq.sort(key=lambda p: p[0])
    
    t_freq = [t for t, x in tx_freq]
    x_freq = np.array([x for t, x in tx_freq], dtype=int).T
    
    # find unique time points and indices of these points for all observations
    t_unique = sorted(list(set(t_freq).union(t_count)))
    num_unique_times = len(t_unique)
    
    idx_freq = [t_unique.index(t) + 1 for t in t_freq] ## add 1 to index!
    idx_count = [t_unique.index(t) + 1 for t in t_count]

    # simulation time points
    t_sim = np.linspace(t0, np.max(t_unique), num_sim_times)

    # combine data into a dict
    data_dict = {
        "T0" : t0,
        "T" : t_unique,
        "N" : num_unique_times,
        "Nc" : num_count,
        "Nf" : num_freq,
        "Idxc" : idx_count,
        "Idxf" : idx_freq,
        "C" : num_celltypes,
        "TotalCounts" : x_count,
        "ClusFreq" : x_freq,
        "Nsim" : num_sim_times,
        "Tsim" : t_sim,
    }
    
    return data_dict


def compile_stan_model() -> cmdstanpy.CmdStanModel:
    # find stan model
    sm_path = os.path.join(stan_model_dir, 'linear_inhom_DA_model.stan')
    #sm_path = os.path.join(stan_model_dir, 'linear_inhom_DA_model_reparam.stan') ## TESTING!!!! remove correlation?
    sm_include_path = os.path.join(stan_model_dir, 'include')
    
    # compile stan model (make sure it has a unique name)
    sm = cmdstanpy.CmdStanModel(
        stan_file=sm_path,
        stanc_options={"include-paths" : [sm_include_path]}
    )

    return sm


def fit_stan_model(
    count_times: np.ndarray, 
    count_data: np.ndarray, 
    freq_times: np.ndarray, 
    freq_data: np.ndarray, 
    t0: float,
    obs_model_freq: str,
    time_dependence: str,
    population_structure: str | np.ndarray,
    covariate_matrix: Optional[np.ndarray] = None,
    excluded_obs_freq: Optional[list[int]] = None,
    count_scaling: float = 1.0,
    growth_rate_penalty: float = 0.0,
    **kwargs
) -> tuple[cmdstanpy.CmdStanMCMC, dict, dict]:
    """ 
    population_structure must be a str or a numpy array.
    The numpy array contains all the differentation pathways that are allowed,
    and the str is either 'independent', meaning "no differentiation allowed"
    of 'unrestricted', meaning all pathways are allowed.
    If actual pathways are given in a numpy array, YOU MUST USE BASE ONE INDEXING.
    
    TODO: more documentation
    
    Returns
    -------
    tuple[cmdstanpy.CmdStanMCMC, dict, dict]
        the fitted stan model, a dict with data and a dict with initial values
        used for fitting.
    
    
    """
    
    # prepare dataset for the stan model
    
    # first format the timeseries data
    stan_data = prepare_stan_timeseries_data(
        count_times, 
        count_data, 
        freq_times, 
        freq_data, 
        t0,
        count_scaling = count_scaling
    )
    
    # add covariate matrix
    if covariate_matrix is None: # empty matrix
        covariate_matrix = np.array([[]])
    
    covar_dict = {
        "Ncovs" : covariate_matrix.shape[1],
        "Covariates" : covariate_matrix
    }
    
    stan_data.update(covar_dict)
        
    # options
    options_dict = {
        "obs_model_freq" : obs_model_codes[obs_model_freq],
        "dyn_model" : time_dependence_codes[time_dependence],
        "eta_nonzero" : 1, # legacy
        "growth_rate_penalty" : growth_rate_penalty,
    }
    
    stan_data.update(options_dict)
    
    # differentiation pathways
    match population_structure:
        case str():
            match population_structure:
                case "independent":
                    idx_diff = np.array([])
                    signed_diff = 0
                case "unrestricted":
                    n = freq_data.shape[0]
                    # must use base-one-indexing
                    idx_diff = np.array([[i+1, j+1] for i in range(n) for j in range(n) if i != j], dtype=int)
                    signed_diff = 0
                case "one_way":
                    n = freq_data.shape[0]
                    # must use base-one-indexing
                    idx_diff = np.array([[i+1, j+1] for i in range(n) for j in range(n) if i < j], dtype=int)
                    signed_diff = 1
                case _:
                    raise Exception(f"invalid population structure string {population_structure}")
        case np.ndarray():
            idx_diff = population_structure
            signed_diff = 0
        case _:
            raise Exception(f"invalid type of population structure object {type(population_structure)}")
    diff_dict = {
        "Nd" : idx_diff.shape[0],
        "IdxDiff" : idx_diff.T,
        "signed_diff" : signed_diff
    }
    
    stan_data.update(diff_dict)
    
    # prior values 
    priors_dict = {
        "tau_loc" : 15.0,
        "tau_scale" : 3.0,
        "u_loc" : 0.0,
        "u_scale" : 1.0,
        "w_loc" : 0.0,
        "w_scale" : 1.0,
        "delta_loc_prior" : 1e2 ## FIXME: add to user interface
    }
    
    stan_data.update(priors_dict)
        
    # optional excluded data points
    nf = stan_data["Nf"]
    if excluded_obs_freq is None:
        excluded_obs_freq = []
    included_freq = [0 if i in excluded_obs_freq else 1 for i in range(nf)]
    stan_data.update({
        "InclFreq" : included_freq
    })
    
    # prepare initial parameter guesses
    ## TODO: import from file!!??
    
    n = freq_data.shape[0]
    nd = idx_diff.shape[0]
    
    inits = {
        "rho" : np.full(n, -0.3),
        "eta_vec" : np.full(n, -0.05),
        "u" : 0.1,
        "w" : 0.1,
        "tau" : priors_dict["tau_loc"],
        "logY0" : 2.0, ## FIXME: depends on the dataset!
        "p0" : np.full(n, 1.0 / n),
        "sigma_c" : 1.0,
        "phi_inv_f" : 0.01,
        "delta_raw" : np.full(nd, 0.002),
    }
    
    # fit stan model
    sm = compile_stan_model()
    sam = sm.sample(data=stan_data, inits=inits, **kwargs)
    
    return sam, inits, stan_data


def get_ic_scale(name: str) -> float:
    match name:
        case "deviance":
            scale_value = -2
        case "log":
            scale_value = 1
        case "negative_log":
            scale_value = -1
        case _:
            raise Exception("unknown IC scale definition")
    return scale_value



def find_bad_pareto_k_values(loo, threshold_bad=0.7):
    pareto_k_values = loo["pareto_k"].to_numpy()
    bad_idxs = [i for i, x in enumerate(pareto_k_values) if x > threshold_bad]
    return bad_idxs

        
def robust_looic(sam, inits, stan_data, threshold_bad=0.7, verbose=False, **kwargs):
    """ 
    Compute looic and check for bad pareto-k values.
    Then re-fit the model leaving each of them out.
    
    TODO: use arviz.reloo when that's more developed
    """
    asam = arviz.from_cmdstanpy(sam, log_likelihood="log_lik")
    loo = arviz.loo(asam, pointwise=True)
    
    # find bad k values
    bad_idxs = find_bad_pareto_k_values(loo, threshold_bad=threshold_bad)

    if len(bad_idxs) == 0:
        if verbose:
            print("no bad pareto k-values")
        return loo
    
    if any([x == 0 for x in stan_data["InclFreq"]]):
        raise NotImplementedError("robust_looic does currently assume that all obervations are included initially.")
    
    # else, re-sample for each bad value
    if verbose:
        print(f"re-fitting for {len(bad_idxs)} observations...", bad_idxs)
    
    sm = compile_stan_model()
    
    # create a new elpd object with the corrected values
    loo_refitted = loo.copy()
    khats = loo_refitted.pareto_k
    loo_i = loo_refitted.loo_i
    scale = loo.scale

    scale_value = get_ic_scale(scale.lower())
    lppd_orig = loo.p_loo + loo.elpd_loo / scale_value
    n_data_points = loo.n_data_points
    
    for idx in bad_idxs:
        # modify stan_data
        nf = stan_data["Nf"]
        if idx >= nf:
            if verbose:
                print(f"skipping bad pareto-k value from count data {idx - nf}")
            continue
        
        incl_freq = np.ones(nf, dtype=int)
        incl_freq[idx] = 0 # leave bad_idx out
        
        stan_data_copy = copy.deepcopy(stan_data)
        stan_data_copy["InclFreq"] = incl_freq   

        # re-fit the model
        re_sam = sm.sample(data=stan_data_copy, inits=inits, **kwargs)
        
        # get likelihood
        log_like_idx = re_sam.stan_variable("log_lik")[:, idx]
        loo_lppd_idx = scale_value * logsumexp(log_like_idx, b=1.0/len(log_like_idx))
        khats[idx] = 0 # dummy value
        loo_i[idx] = loo_lppd_idx

    loo_refitted.elpd_loo = loo_i.values.sum()
    loo_refitted.se = (n_data_points * np.var(loo_i.values)) ** 0.5
    loo_refitted.p_loo = lppd_orig - loo_refitted.elpd_loo / scale_value
    
    return loo_refitted
        
    

    
def pairwise_loo_comparison(results, models=None, method="BB-pseudo-BMA"):
    """
    Compare each pair of models with the `compare` method from `arviz`.

    Parameters:
    -----------
    results : dict
        A dictionary of stan model fits, where the keys correspond to model names.

    models : list, optional
        A list of model names to compare. 
        If not provided, all models in the `results` dictionary will be compared.

    method : str, optional
        The method to use for model weight calculations. Default is "BB-pseudo-BMA".

    Returns:
    --------
    comps : list
        A list of comparison objects generated by the `compare` method from `arviz`.

    """
    if models is None:
        models = sorted(list(results.keys()))
    comps = []
    for i1, k1 in enumerate(models):
        for i2, k2 in enumerate(models):
            if i1 <= i2:
                continue
            pairwise_dict = {
                k1 : results[k1],
                k2 : results[k2]
            }
            pairwise_comp = arviz.compare(pairwise_dict, method=method)
            comps.append(pairwise_comp)
    return comps

    
