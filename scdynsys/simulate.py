import numpy as np
import scipy.stats as sts
from scipy.integrate import solve_ivp
from scipy.special import expit
from scipy.linalg import expm


def rho(t, rho0, eta, u):
    return (rho0 - eta) * np.exp(-t*u) + eta

def trm_ode_model(t, x, rho0, eta, u, Q):
    rhot = rho(t, rho0, eta, u)
    return rhot * x + np.dot(Q, x)

def trm_model_magnus1_approx(t, x0, rho0, eta, u, Q):
    cumul_rhot = (1-np.exp(-u*t)) / u * (rho0-eta) + eta * t
    At = Q*t + np.diag(cumul_rhot)
    return np.dot(expm(At), x0)

def trm_model_magnus2_approx(t, x0, rho0, eta, u, Q):
    cumul_rhot = (1-np.exp(-u*t)) / u * (rho0-eta) + eta * t
    Omega1t = Q*t + np.diag(cumul_rhot)
    cumul_rho2t = -((rho0 - eta) / u**2 * (2*(1-np.exp(-u*t)) - u*t*(np.exp(-u*t)+1)))
    Omega2t = 0.5 * (Q @ np.diag(cumul_rho2t) - np.diag(cumul_rho2t) @ Q)
    return np.dot(expm(Omega1t + Omega2t), x0)

def solve_trm_ivp(ts, t0, logx0, rho0, eta, u, Q):
    x0 = np.exp(logx0)
    tts = ts - t0
    tts_unique = sorted(list(set(tts)))
    tt_index = [tts_unique.index(t) for t in tts]
    t_span = np.min(tts), np.max(tts)
    sol = solve_ivp(trm_ode_model, t_span, x0, t_eval=tts_unique, args=(rho0, eta, u, Q), rtol=1e-11, atol=1e-11)
    x = sol.y[:,tt_index]
    return x

def solve_trm_ivp_magnus1_approx(ts, t0, logx0, rho0, eta, u, Q):
    """this solution is not valid in general"""
    x0 = np.exp(logx0)
    tts = ts - t0
    tts_unique = sorted(list(set(tts)))
    tt_index = [tts_unique.index(t) for t in tts]
    x = np.array([trm_model_magnus1_approx(t, x0, rho0, eta, u, Q) for t in tts_unique]).T
    return x[:, tt_index]
    
def solve_trm_ivp_magnus2_approx(ts, t0, logx0, rho0, eta, u, Q):
    """this solution is not valid in general"""
    x0 = np.exp(logx0)
    tts = ts - t0
    tts_unique = sorted(list(set(tts)))
    tt_index = [tts_unique.index(t) for t in tts]
    x = np.array([trm_model_magnus2_approx(t, x0, rho0, eta, u, Q) for t in tts_unique]).T
    return x[:, tt_index]

def gen_trm_data(x, sigma, phi, N, scaling):
    y = np.sum(x, axis=0)
    y_obs = np.exp(np.log(y*scaling) + sts.norm.rvs(scale=sigma, size=y.shape))
    pi = x / y
    if phi is None:
        pi_obs = pi.T
    else:
        pi_obs = [sts.dirichlet.rvs(np.abs(p * phi))[0] for p in pi.T]
    k_obs = np.array([sts.multinomial.rvs(n, q) for q, n in zip(pi_obs, N)])
    return y_obs, k_obs
