import numpy as np
from scipy.integrate import solve_ivp

from . import simulate


def sens_ode(t, z, rho0, eta, u, Q):
    """
    sensitivity equations for x, rho0, eta, and u
    z is a concatenation of [x, dxdrho0, dxdeta, dxdu]
    this is the full model
    """
    
    n = len(rho0)
    x = z[:n]
    ddt_x = simulate.trm_ode_model(t, x, rho0, eta, u, Q)

    ddrho0_x = np.reshape(z[n:n+n**2], (n,n))
    ddeta_x = np.reshape(z[n+n**2:n+2*n**2], (n,n))
    ddu_x = z[n+2*n**2:2*n+2*n**2]

    ddrho0_rho = np.exp(-t*u)
    ddeta_rho = (1-np.exp(-t*u))
    ddu_rho = -t * np.exp(-t*u) * (rho0 - eta)

    ddt_ddrho0_x = np.diag(simulate.rho(t, rho0, eta, u)) @ ddrho0_x + Q @ ddrho0_x + ddrho0_rho * np.diag(x)
    ddt_ddeta_x = np.diag(simulate.rho(t, rho0, eta, u)) @ ddeta_x + Q @ ddeta_x + ddeta_rho * np.diag(x)
    ddt_ddu_x = simulate.rho(t, rho0, eta, u) * ddu_x + Q @ ddu_x + ddu_rho * x

    dzdt = np.zeros_like(z)
    dzdt[:n] = ddt_x
    dzdt[n:n+n**2] = np.reshape(ddt_ddrho0_x, (n**2,))
    dzdt[n+n**2:n+2*n**2] = np.reshape(ddt_ddeta_x, (n**2,))
    dzdt[n+2*n**2:2*n+2*n**2] = ddt_ddu_x

    return dzdt

def sens_ode_hom(t, z, rho0, Q):
    """
    sensitivity equations for x, rho0, eta, and u
    z is a concatenation of [x, dxdrho0, dxdeta, dxdu]
    this is the autonomous model
    """

    n = len(rho0)
    x = z[:n]
    ddt_x = simulate.trm_ode_model(t, x, rho0, rho0, 0.0, Q)

    ddrho0_x = np.reshape(z[n:n+n**2], (n,n))
    
    ddt_ddrho0_x = np.diag(rho0) @ ddrho0_x + Q @ ddrho0_x + np.diag(x)

    dzdt = np.zeros_like(z)
    dzdt[:n] = ddt_x
    dzdt[n:n+n**2] = np.reshape(ddt_ddrho0_x, (n**2,))

    return dzdt


def sens_ode_Q(t, z, rho0, eta, u, Q):
    """
    sensitivity equations for Q
    z is a concatenation of [x, dxdQ]
    here the diagonal elements of Q are independent of the off-diagonal elements.
    we will fix this in a function below.
    This is for the full model
    """

    n = len(rho0)
    x = z[:n]
    ddt_x = simulate.trm_ode_model(t, x, rho0, eta, u, Q)

    ddQ_x = np.reshape(z[n:n+n**3], (n,n,n))
    ddt_ddQ_x = np.zeros_like(ddQ_x)
    rho_val = simulate.rho(t, rho0, eta, u)

    # Vectorized computation
    ddt_ddQ_x = np.einsum('k,kij->kij', rho_val, ddQ_x) + np.einsum('ij, jkl -> ikl', Q, ddQ_x) + np.einsum('j, ki -> kij', x, np.eye(n))
    dzdt = np.zeros_like(z)
    dzdt[:n] = ddt_x
    dzdt[n:n+n**3] = np.reshape(ddt_ddQ_x, (n**3,))
    return dzdt


def sens_ode_Q_hom(t, z, rho0, Q):
    """
    sensitivity equations for Q
    z is a concatenation of [x, dxdQ]
    here the diagonal elements of Q are independent of the off-diagonal elements.
    we will fix this in a function below.
    This is the the autonomous model
    """

    n = len(rho0)
    x = z[:n]
    ddt_x = simulate.trm_ode_model(t, x, rho0, rho0, 0.0, Q)

    ddQ_x = np.reshape(z[n:n+n**3], (n,n,n))
    ddt_ddQ_x = np.zeros_like(ddQ_x)

    # Vectorized computation
    ddt_ddQ_x = np.einsum('k,kij->kij', rho0, ddQ_x) + np.einsum('ij, jkl -> ikl', Q, ddQ_x) + np.einsum('j, ki -> kij', x, np.eye(n))
    dzdt = np.zeros_like(z)
    dzdt[:n] = ddt_x
    dzdt[n:n+n**3] = np.reshape(ddt_ddQ_x, (n**3,))
    return dzdt




def solve_trm_ivp_sens(ts, t0, logx0, rho0, eta, u, Q):
    """full model"""
    x0 = np.exp(logx0)
    n = len(x0)
    z0 = np.concatenate([x0, np.zeros(n+2*n**2)])
    tts = ts - t0
    tts_unique = sorted(list(set(tts)))
    tt_index = [tts_unique.index(t) for t in tts]
    t_span = np.min(tts), np.max(tts)
    sol = solve_ivp(sens_ode, t_span, z0, t_eval=tts_unique, args=(rho0, eta, u, Q), rtol=1e-11, atol=1e-11)
    z = sol.y[:,tt_index]
    N = len(ts)
    x = z[0:n]
    ddrho0_x = np.reshape(z[n:n+n**2], (n,n,N))
    ddeta_x = np.reshape(z[n+n**2:n+2*n**2], (n,n,N))
    ddu_x = z[n+2*n**2:2*n+2*n**2]
    sens_result = {
        "x" : x,
        "ddrho0_x" : ddrho0_x,
        "ddeta_x" : ddeta_x,
        "ddu_x" : ddu_x,
    }
    return sens_result



def solve_trm_ivp_sens_hom(ts, t0, logx0, rho0, Q):
    """autonomous model"""
    x0 = np.exp(logx0)
    n = len(x0)
    z0 = np.concatenate([x0, np.zeros(n+2*n**2)])
    tts = ts - t0
    tts_unique = sorted(list(set(tts)))
    tt_index = [tts_unique.index(t) for t in tts]
    t_span = np.min(tts), np.max(tts)
    sol = solve_ivp(sens_ode_hom, t_span, z0, t_eval=tts_unique, args=(rho0, Q), rtol=1e-11, atol=1e-11)
    z = sol.y[:,tt_index]
    N = len(ts)
    x = z[0:n]
    ddrho0_x = np.reshape(z[n:n+n**2], (n,n,N))
    sens_result = {
        "x" : x,
        "ddrho0_x" : ddrho0_x,
    }
    return sens_result



def solve_trm_ivp_sens_Q(ts, t0, logx0, rho0, eta, u, Q):
    """full model"""
    x0 = np.exp(logx0)
    n = len(x0)
    z0 = np.concatenate([x0, np.zeros(n**3)])
    tts = ts - t0
    tts_unique = sorted(list(set(tts)))
    tt_index = [tts_unique.index(t) for t in tts]
    t_span = np.min(tts), np.max(tts)
    sol = solve_ivp(sens_ode_Q, t_span, z0, t_eval=tts_unique, args=(rho0, eta, u, Q), rtol=1e-11, atol=1e-11)
    z = sol.y[:,tt_index]
    N = len(ts)
    x = z[0:n]
    ddQ_x = np.reshape(z[n:n+n**3], (n,n,n,N))

    # Fix the diagonal elements of Q

    ddQ_x = ddQ_x - np.einsum('kjjn->kjn', ddQ_x)[:,None,:,:]

    sens_result = {
        "x" : x,
        "ddQ_x" : ddQ_x
    }
    return sens_result



def solve_trm_ivp_sens_Q_hom(ts, t0, logx0, rho0, Q):
    """autonomous model"""
    x0 = np.exp(logx0)
    n = len(x0)
    z0 = np.concatenate([x0, np.zeros(n**3)])
    tts = ts - t0
    tts_unique = sorted(list(set(tts)))
    tt_index = [tts_unique.index(t) for t in tts]
    t_span = np.min(tts), np.max(tts)
    sol = solve_ivp(sens_ode_Q_hom, t_span, z0, t_eval=tts_unique, args=(rho0, Q), rtol=1e-11, atol=1e-11)
    z = sol.y[:,tt_index]
    N = len(ts)
    x = z[0:n]
    ddQ_x = np.reshape(z[n:n+n**3], (n,n,n,N))

    # Fix the diagonal elements of Q

    ddQ_x = ddQ_x - np.einsum('kjjn->kjn', ddQ_x)[:,None,:,:]

    sens_result = {
        "x" : x,
        "ddQ_x" : ddQ_x
    }
    return sens_result



def trm_sens_y_pi(sens_result):
    """
    Compute the sensitivity of y and pi with respect to rho0, eta, and u.
    Use the sensitivity equations of x and the definition of y and pi.
    This is for the full model
    """
    x = sens_result["x"]
    ddrho0_x = sens_result["ddrho0_x"]
    ddeta_x = sens_result["ddeta_x"]
    ddu_x = sens_result["ddu_x"]

    y = np.sum(x, axis=0)
    pi = x / y

    ddrho0_y = np.sum(ddrho0_x, axis=0)
    ddeta_y = np.sum(ddeta_x, axis=0)
    ddu_y = np.sum(ddu_x, axis=0)

    ddrho0_logy = ddrho0_y / y
    ddeta_logy = ddeta_y / y
    ddu_logy = ddu_y / y

    ddrho0_pi = ddrho0_x / y - pi * ddrho0_y / y
    ddeta_pi = ddeta_x / y - pi * ddeta_y / y
    ddu_pi = ddu_x / y - pi * ddu_y / y

    logit_factor = 1 / (pi * (1-pi))

    ddrho_logitpi = np.einsum("kln,kn->kln", ddrho0_pi, logit_factor)
    ddeta_logitpi = np.einsum("kln,kn->kln", ddeta_pi, logit_factor)
    ddu_logitpi = ddu_pi * logit_factor

    sens_result = {
        "y" : y,
        "pi" : pi,
        "ddrho0_y" : ddrho0_y,
        "ddeta_y" : ddeta_y,
        "ddu_y" : ddu_y,
        "ddrho0_logy" : ddrho0_logy,
        "ddeta_logy" : ddeta_logy,
        "ddu_logy" : ddu_logy,
        "ddrho0_pi" : ddrho0_pi,
        "ddeta_pi" : ddeta_pi,
        "ddu_pi" : ddu_pi,
        "ddrho_logitpi" : ddrho_logitpi,
        "ddeta_logitpi" : ddeta_logitpi,
        "ddu_logitpi" : ddu_logitpi,
    }

    return sens_result


def trm_sens_y_pi_hom(sens_result):
    """
    Compute the sensitivity of y and pi with respect to rho0, eta, and u.
    Use the sensitivity equations of x and the definition of y and pi.
    This is for the autonomous model
    """
    x = sens_result["x"]
    ddrho0_x = sens_result["ddrho0_x"]

    y = np.sum(x, axis=0)
    pi = x / y

    ddrho0_y = np.sum(ddrho0_x, axis=0)
    ddrho0_logy = ddrho0_y / y
    ddrho0_pi = ddrho0_x / y - pi * ddrho0_y / y
    logit_factor = 1 / (pi * (1-pi))
    ddrho_logitpi = np.einsum("kln,kn->kln", ddrho0_pi, logit_factor)

    sens_result = {
        "y" : y,
        "pi" : pi,
        "ddrho0_y" : ddrho0_y,
        "ddrho0_logy" : ddrho0_logy,
        "ddrho0_pi" : ddrho0_pi,
        "ddrho_logitpi" : ddrho_logitpi,
    }

    return sens_result



def trm_sens_y_pi_Q(sens_result):
    """
    Compute the sensitivity of y and pi with respect to Q.
    Use the sensitivity equations of x and the definition of y and pi.
    This function works for both the full and autonomous model.
    """
    x = sens_result["x"]
    ddQ_x = sens_result["ddQ_x"]

    y = np.sum(x, axis=0)
    pi = x / y

    ddQ_y = np.sum(ddQ_x, axis=0)

    ddQ_logy = ddQ_y / y

    ddQ_pi = ddQ_x / y - np.einsum("kn,ijn->kijn", pi, ddQ_y / y)

    logit_factor = 1 / (pi * (1-pi))

    ddQ_logitpi = np.einsum("kijn,kn->kijn", ddQ_pi, logit_factor)

    sens_result = {
        "y" : y,
        "pi" : pi,
        "ddQ_y" : ddQ_y,
        "ddQ_logy" : ddQ_logy,
        "ddQ_pi" : ddQ_pi,
        "ddQ_logitpi" : ddQ_logitpi,
    }

    return sens_result