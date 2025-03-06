"""
Some statistical tools
"""
from typing import Callable, Optional, List, Dict, Tuple
import torch
import numpy as np
from scipy.special import logsumexp 
from pyro import poutine


def sample_from_posterior(
    model: Callable,
    guide: Callable,
    *args: torch.Tensor, 
    num_samples: int = 1000,
    ll_sites: Optional[List[str]] = None, 
    val_sites: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Sample from the (variational approximation) of a pyro model
    and compute log-likelihood of given sites.
    Result can be used to construct an arviz `inference_data` object
    """
    if ll_sites is None: ll_sites = []
    if val_sites is None: val_sites = []
    lls = {site : [] for site in ll_sites}
    vals = {site : [] for site in val_sites}
    with torch.no_grad():
        for n in range(num_samples):
            guide_trace = poutine.trace(guide).get_trace(*args)
            repl = poutine.replay(model, trace=guide_trace)
            trace = poutine.trace(repl).get_trace(*args)
            trace.compute_log_prob()
            for site in ll_sites:
                lls[site].append(trace.nodes[site]['log_prob'])
            for site in val_sites:
                vals[site].append(trace.nodes[site]['value'])
    lls = {k : torch.stack(v).unsqueeze(0).cpu().numpy() for k, v in lls.items()}
    vals = {k : torch.stack(v).unsqueeze(0).cpu().numpy() for k, v in vals.items()}
    return lls, vals


def log_posterior_likelihood(ll_vals: np.ndarray) -> Tuple[float, float]:
    num_sam, num_obs = ll_vals.shape
    lse = logsumexp(ll_vals, axis=0)
    lpl = lse - np.log(num_sam)
    total_lpl = np.sum(lpl)
    se_lpl = np.std(lpl) * np.sqrt(num_obs)
    return total_lpl, se_lpl
