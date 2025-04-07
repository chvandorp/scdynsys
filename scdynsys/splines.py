from scipy.interpolate import make_smoothing_spline
import numpy as np

def fit_smoothing_spline(xs, ys, bootstrap=None, num_points=100, **kwargs):
    uxs = np.unique(xs)
    yss = [[y for x, y in zip(xs, ys) if x == ux] for ux in uxs]
    ws = [len(y) for y in yss]
    mys = [np.mean(y) for y in yss]
    spl = make_smoothing_spline(uxs, mys, w=ws, **kwargs)
    x_grid = np.linspace(np.min(xs), np.max(xs), num_points)
    y_smooth = spl(x_grid)
    
    if bootstrap is None:
        return spl, x_grid, y_smooth

    # else...
    ys_hat = spl(xs)
    res = ys - ys_hat
    ys_smooth = []
    bootstrap_spls = []
    for b in range(bootstrap):
        rand_res = np.random.choice(res, len(res), replace=True)
        ys_bs = ys_hat + rand_res
        b_spl, _, y_smooth_bs = fit_smoothing_spline(xs, ys_bs, **kwargs)
        ys_smooth.append(y_smooth_bs)
        bootstrap_spls.append(b_spl)
    y_smooth_lower, y_smooth_upper = np.percentile(ys_smooth, axis=0, q=[2.5, 97.5])

    return spl, x_grid, y_smooth, bootstrap_spls, y_smooth_lower, y_smooth_upper