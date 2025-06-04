import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster import hierarchy
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib
import torch
from typing import Mapping
from matplotlib.ticker import LogFormatterSciNotation

from . import flow
from . import utilities as util
from .vae.dyn import build_Q_mat


class SemiLogSciFormatter(LogFormatterSciNotation):
    """
    Class to format ticklabels on a log scale
    using scientific (latex) notation for very large and
    small numbers, and regular notation using numbers 
    in the range 0.01 to 100. 
    Also reduce the number of minor ticklabels (in cases where
    these are shown: see the `minor_thresholds` parameter)
    Pass an instance of this class to the `ax.yaxis.set_major_formatter` 
    to use the formatter.
    """
    def __call__(self, x, pos=None):
        xfmt = super().__call__(x, pos=pos)
        if xfmt == '' or x > 100 or x < 0.01:
            return xfmt
        # else...
        return f"{x:g}"

    def set_locs(self, locs=None):
        super().set_locs(locs=locs)
        # reduce the number of sublabels
        assert self._base == 10, "this hack only works for base 10"
        if (n := len(self._sublabels)) > 1 and n < 10:
            self._sublabels = {1, 2, 5, 10}
        

class PiecewiseNorm(Normalize):
    """
    Class extending the Normalize class from matplotlib in order to
    make divergent colorbars with different scales on the postive
    and negative half-line. This is used for a `pcolor` representation of 
    stochastic matrices Q.
    """
    def __init__(self, levels, clip=False):
        # input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))

    def inverse(self, value):
        return np.ma.masked_array(np.interp(value, self._normed, self._levels))



def density_plot(
    ax, xs, color="tab:blue", n=1000, kwargs_line=None,
    kwargs_fill=None,
):
    if kwargs_line is None:
        kwargs_line = {}
    if kwargs_fill is None:
        kwargs_fill = {}
    us = np.linspace(np.min(xs), np.max(xs), 1000)
    idxs = np.random.choice(len(xs), n)
    z = sts.gaussian_kde(xs[idxs])(us)
    ax.plot(us, z, linewidth=2, color=color, **kwargs_line)
    ax.fill_between(us, z, alpha=0.5, linewidth=0, color=color, **kwargs_fill)
    
def simple_boxplot(
    ax, pos, data, color='k', color_med='red', p=95, 
    horizontal=False, quantiles=False, s=None, **kwargs
):
    uppers = [np.nanpercentile(x, 50+p/2) for x in data]
    downers = [np.nanpercentile(x, 50-p/2) for x in data]
    if quantiles:
        q1s = [np.nanpercentile(x, 25) for x in data]
        q2s = [np.nanpercentile(x, 75) for x in data]
    medians = [np.nanmedian(x) for x in data]
    if isinstance(color, str):
        color = [color for _ in pos]
    if horizontal:
        ax.scatter(uppers, pos, marker='|', color=color, s=s, **kwargs)
        ax.scatter(downers, pos, marker='|', color=color, s=s, **kwargs)
        ax.scatter(medians, pos, marker='|', color=color_med, s=s, **kwargs)
        for c, p, d, u in zip(color, pos, downers, uppers):
            ax.plot([d, u], [p, p], color=c, **kwargs)
        if quantiles:
            for c, p, d, u in zip(color, pos, q1s, q2s):
                ax.plot([d, u], [p, p], color=c, linewidth=3, **kwargs)

    else: ## vertical
        ax.scatter(pos, uppers, marker='_', color=color, s=s, **kwargs)
        ax.scatter(pos, downers, marker='_', color=color, s=s, **kwargs)
        ax.scatter(pos, medians, marker='_', color=color_med, s=s, **kwargs)
        for c, p, d, u in zip(color, pos, downers, uppers):
            ax.plot([p, p], [d, u], color=c, **kwargs)
        if quantiles:
            for c, p, d, u in zip(color, pos, q1s, q2s):
                ax.plot([p, p], [d, u], color=c, linewidth=3, **kwargs)


                
def fancy_boxplot(ax, xs, pos=None, color='k', horizontal=False, **kwargs):
    """
    plot a boxplot with 95% and 75% confidence intervals
    and median. The 75% CI is plotted as a thicker line.
    The 95% CI is plotted as a line with markers at the end.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        the axes on which to plot the boxplot.
    xs : array-like
        the data to plot.
    pos : array-like, optional
        the positions at which to plot the boxplots.
    color : str or list, optional
        the color of the boxplots.
    horizontal : bool, optional
        whether to plot the boxplots horizontally.
    **kwargs : dict
        keyword arguments to pass to the `scatter` and `plot` method of the axes.

    Returns
    -------
    None   
    """
    if pos is None:
        pos = np.arange(len(xs))    
    q = [2.5, 25, 50, 75, 97.5]
    if isinstance(xs, np.ndarray):
        llx, lx, mx, ux, uux = np.percentile(xs, axis=1, q=q)
    else: # allow ragged data
        llx = np.array([np.percentile(x, q=q[0]) for x in xs])
        lx = np.array([np.percentile(x, q=q[1]) for x in xs])
        mx = np.array([np.percentile(x, q=q[2]) for x in xs])
        ux = np.array([np.percentile(x, q=q[3]) for x in xs])
        uux = np.array([np.percentile(x, q=q[4]) for x in xs])
    if type(color) is not list:
        color = [color for _ in pos]
    if horizontal:
        ax.scatter(llx, pos, marker="|", color=color, **kwargs)
        ax.scatter(uux, pos, marker="|", color=color, **kwargs)
        ax.scatter(mx, pos, marker="|", color=color, **kwargs)
        for i, p in enumerate(pos):
            ax.plot([llx[i], uux[i]], [p,p], color=color[i], **kwargs)
            ax.plot([lx[i], ux[i]], [p,p], color=color[i], linewidth=3, **kwargs)
    else:
        ax.scatter(pos, llx, marker="_", color=color, **kwargs)
        ax.scatter(pos, uux, marker="_", color=color, **kwargs)
        ax.scatter(pos, mx, marker="_", color=color, **kwargs)
        if type(color) is not list:
            color = [color for _ in pos]
        for i, p in enumerate(pos):
            ax.plot([p,p], [llx[i], uux[i]], color=color[i], **kwargs)
            ax.plot([p,p], [lx[i], ux[i]], color=color[i], linewidth=3, **kwargs)
                
                
            
def plot_dendrogram(vals, bx, method='average', metric='euclidean'):
    Z = hierarchy.linkage(vals, method, metric=metric)
    den = hierarchy.dendrogram(Z, ax=bx, link_color_func=lambda x: 'k')
    return den["leaves"]


def plot_expression_matrix(vals, idxs, marker_names, ax):
    M = np.array(vals)[idxs,:]
    cs = ax.pcolor(M.T, cmap="RdBu_r")
    num_markers = len(marker_names)

    ax.set_yticks(np.linspace(0, num_markers-1, num_markers)+0.5)
    ax.set_yticklabels(marker_names)

    num_clus = len(idxs)
    ax.set_xticks(np.linspace(0, num_clus-1, num_clus)+0.5)
    ax.set_xticklabels(idxs)
    return cs

def raw_scatter_plot(data_x, data_y, xlab, ylab, x_gate=None, y_gate=None):
    """scatter plot with 1 and 2D density and optionally draw gates"""
    fig, axs = plt.subplots(2, 2, figsize = (10,10))
    axs[0,0].scatter(data_x, data_y, s = 0.02, color='k')
    axs[0,0].set_xlabel(xlab)
    axs[0,0].set_ylabel(ylab)
    axs[1,1].axis('off')
    
    sns.kdeplot(data_x, y=data_y, ax=axs[0,0], levels=10)
    sns.kdeplot(data_x, ax=axs[1,0])
    sns.kdeplot(y=data_y, ax=axs[0,1])

    ## gates
    if x_gate is not None:
        axs[0,0].axvline(x_gate, linestyle='--', color='tab:red')
        axs[1,0].axvline(x_gate, linestyle='--', color='tab:red')
    if y_gate is not None:
        axs[0,0].axhline(y_gate, linestyle='--', color='tab:red')
        axs[0,1].axhline(y_gate, linestyle='--', color='tab:red')
        
    if x_gate is not None and y_gate is not None:
        fracs = flow.calc_nd_fracs([data_x, data_y], [x_gate, y_gate])
        coords = [0.05, 0.95]
        for i in range(2):
            for j in range(2):
                axs[0,0].text(coords[i], coords[j], f"{fracs[i,j]:0.2f}", 
                              transform=axs[0,0].transAxes, color='red', 
                              va='center', ha='center')
        
        
    return fig, axs


def generate_scatter_plot(df, marker1, marker2, cof_x, cof_y, nsam=None):
    """
    transform the flow cytometry data using the arcsinh transform
    and cofactors. Then make a scatter plot with density estimates
    """
    data_x = df[marker1].to_numpy()
    data_y = df[marker2].to_numpy()
    ## reduce the sample size by taking a random sample
    if nsam is not None:
        idx = np.random.choice(range(len(data_x)), size=nsam, replace=False)
        data_x, data_y = data_x[idx], data_y[idx]
    ## apply asinh transform with cofactors
    data_x = np.arcsinh(data_x/cof_x)
    data_y = np.arcsinh(data_y/cof_y)
    
    fig, axs = raw_scatter_plot(data_x, data_y, marker1, marker2)
    
    return fig, axs


def confidence_ellipse(mu, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    plot an ellipse representing the confidence region of a 2D Gaussian
    distribution. The ellipse is centered at the mean `mu` and has
    the covariance matrix `cov`. The `n_std` parameter controls the
    size of the ellipse.
    This function is adapted from the matplotlib gallery.

    Parameters
    ----------
    mu : array-like
        the mean of the distribution.
    cov : array-like
        the covariance matrix of the distribution.
    ax : matplotlib.axes.Axes
        the axes on which to plot the ellipse.
    n_std : float, optional
        the number of standard deviations to include in the ellipse.
    facecolor : str, optional
        the color to fill the ellipse with.
    **kwargs : dict
        keyword arguments to pass to the `Ellipse` object.

    Returns
    -------
    matplotlib.patches.Ellipse
        the ellipse object. This is also added to the axes.
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(*mu)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_hpd_contour(ax, sam, color='k', linewidth=3, level=0.25, label=None):
    num_grid_points = 200
    m, M = np.min(sam[:,0]), np.max(sam[:,0])
    w = M-m
    xs = np.linspace(m - 0.1*w, M + 0.1*w, num_grid_points)
    m, M = np.min(sam[:,1]), np.max(sam[:,1])
    w = M-m
    ys = np.linspace(m - 0.1*w, M + 0.1*w, num_grid_points)
    Xs, Ys = np.meshgrid(xs, ys)
    XYs = np.stack([Xs.flatten(), Ys.flatten()])
    density = sts.gaussian_kde(sam.T)(XYs).reshape(num_grid_points, num_grid_points)
    contour = ax.contour(
        xs, ys, density, levels=[level*np.max(density)], 
        linewidths=[linewidth], colors=[color]
    )
    if label is not None:
        patch = Line2D([0], [0], color=color, label=label)
        return patch

    
    
def plot_mpd(ax, pos, sam, bx=None, loc=None, boxplot_kwargs=None):
    """marginal posterior densities"""
    if boxplot_kwargs is None:
        boxplot_kwargs = {}
    if sam is not None:
        ax.violinplot(sam, positions=pos, showextrema=False)
        simple_boxplot(ax, pos, sam.T, **boxplot_kwargs)
    if loc is not None and bx is not None:
        bx.violinplot(loc, positions=pos[:1], showextrema=False)
        simple_boxplot(bx, pos[:1], [loc], **boxplot_kwargs)

def plot_mpd_array(sams, locs, pos, xlabs, ylabs, boxplot_kwargs=None):
    """array of marginal posterior densities"""
    n = len(sams)
    k = len(pos)
    gs = GridSpec(n, k+1)
    fig = plt.figure(figsize=((k+1)*0.6,2.5*n))
    axs, bxs = [], []
    for i, sam in enumerate(sams):
        ax = fig.add_subplot(gs[i,:-1])
        axs.append(ax)
        bx = fig.add_subplot(gs[i,-1], sharey=ax) if locs[i] is not None else None
        bxs.append(bx)
        plot_mpd(ax, pos, sam, bx, locs[i], boxplot_kwargs=boxplot_kwargs)
        
    # make sure x-axis is shared
    for ax in axs:
        ax.sharex(axs[0])
        
    # but remove redundant xticklabels
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # set tick for each violin
    for ax in axs:
        ax.set_xticks(pos)

    # remove redundant labels
    for ax in axs[:-1]:
        ax.set_xticklabels([])

    # add labels to the bottom graph
    axs[-1].set_xticklabels(xlabs, rotation=90)
        
    for i, bx in enumerate(bxs):
        if bx is not None:
            bx.axes.get_yaxis().set_visible(False)
            bx.set_xticks(pos[:1])
            bx.set_xticklabels(["loc"])
            
    for ax, lab in zip(axs, ylabs):
        ax.set_ylabel(lab)
    
    return fig, axs, bxs


def simple_heatmap(ax, M, features, groups):
    vmax = np.max(np.abs(M))

    ax.pcolor(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_xticks(0.5 + np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90)

    ax.set_yticks(0.5 + np.arange(len(groups)))
    ax.set_yticklabels(groups)


    
def plot_square_gates(ax, xgate, ygate, percentage=None, **kwargs):
    """
    plot a square gate on the x-y plane.

    The gate is defined by two tuples (x1, x2) and (y1, y2)
    where x1 and y1 are the lower bounds and x2 and y2 are the
    upper bounds of the gate.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        the axes on which to plot the gate.
    xgate : tuple[float, float]
        the lower and upper bounds of the gate on the x-axis.
    ygate : tuple[float, float]
        the lower and upper bounds of the gate on the y-axis.
    percentage : float, optional
        the percentage of events that fall within the gate.
    **kwargs : dict
        keyword arguments to pass to the `plot` method of the axes.

    Returns
    -------
    None
    """
    x1, x2 = xgate
    y1, y2 = ygate
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], **kwargs)
    if percentage is not None:
        ax.text(x1, y2, f"{percentage:0.2f}", va='top', ha='left')
    


def plot_1d_gates(ax: matplotlib.axes.Axes, xgate: tuple[float, float], **kwargs) -> None:
    """
    plot a 1D gate on the x-axis.

    The gate is defined by a tuple (x1, x2) where x1 is the
    lower bound and x2 is the upper bound of the gate.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        the axes on which to plot the gate.
    xgate : tuple[float, float]
        the lower and upper bounds of the gate.
    **kwargs : dict
        keyword arguments to pass to the `axvline` method of the axes.

    Returns
    -------
    None
    """
    x1, x2 = xgate
    ax.axvline(x1, **kwargs)
    ax.axvline(x2, **kwargs)


def plot_stan_fits(
    sam, 
    t_sim, 
    t_freq, 
    pop_freq, 
    t_count, 
    total_count, 
    pop_names,
    total_name,
    scaling=1.0, 
    cols=3, 
    color='tab:blue'
):
    num_pop = pop_freq.shape[1]
    nax = num_pop+1
    rows = nax // cols + (0 if nax % cols == 0 else 1)
    fig, axs = plt.subplots(rows, cols, figsize=(10, 7), sharex=True)
    
    freqs = sam.stan_variable("freqs")
    freqs_sim = sam.stan_variable("freqs_sim")
    rel_pop_freq = (pop_freq.T / np.sum(pop_freq.T, axis=0)).T
    
    if isinstance(color, str):
        color = [color for _ in range(num_pop)]
        
    for i in range(num_pop):
        ax = axs.flatten()[i+1]
        ## plot data
        ax.scatter(t_freq, rel_pop_freq[:,i], s=5, color='k')
        ax.set_title(f"{pop_names[i]}")
        ## plot prediction
        lf, mf, hf = np.percentile(freqs[:,i,:], axis=0, q=[2.5, 50, 97.5])
        ax.plot(t_sim, mf, color=color[i])
        ax.fill_between(t_sim, lf, hf, alpha=0.5, linewidth=0, color=color[i])
        ax.set_yscale('log'); ax.set_ylim(2e-3, 1)
        ax.set_ylabel("fraction of cells")
        ## plot simulations
        for j, t in enumerate(t_freq):
            l, u = np.percentile(freqs_sim[:,i,j], q=[2.5, 97.5])
            ax.plot([t,t], [l, u], color=color[i], linewidth=3, alpha=0.2)   

    ## total number of cells
    ax = axs.flatten()[0]
    counts = sam.stan_variable("counts") * scaling
    counts_sim = sam.stan_variable("counts_sim") * scaling
        
    ## plot prediction
    lc, mc, hc = np.percentile(counts, axis=0, q=[2.5, 50, 97.5])
    ax.plot(t_sim, mc, color='blue')
    ax.fill_between(t_sim, lc, hc, alpha=0.3, color='blue', linewidth=0)
    ## plot data
    ax.scatter(t_count, total_count, s=5, color='k')
    ## plot simulations
    lo, hi = np.percentile(counts_sim, q=[2.5, 97.5], axis=0)
    ax.fill_between(t_sim, lo, hi, color='blue', alpha=0.2, linewidth=0)

    ax.set_yscale('log')
    ax.set_title(total_name)
    ax.set_ylabel("number of cells")
    
    return fig, axs


def plot_stan_fits_fancy(
    count_ax,
    freq_axs,
    stan_vars,
    stan_data,
    count_scaling,
    clus_colors,
    minor_thresholds=(1.5, 0.5)
) -> None:   
    ts_all = np.array(stan_data["T"])
    ts_clus = ts_all[np.array(stan_data["Idxf"])-1] ## 0/1 indexing correction 
    xs_clus = stan_data["ClusFreq"]
    ts_counts = ts_all[np.array(stan_data["Idxc"])-1] 
    xs_counts = stan_data["TotalCounts"] * count_scaling
    
    xticks = [14, 28, 42, 56]
        
    # store the stan variables in the pickle!
    freqs = stan_vars["freqs"]
    freqs_sim = stan_vars["freqs_sim"]
    
    num_clus = xs_clus.shape[0]
    
    xs_clus_rel = xs_clus / np.sum(xs_clus, axis=0)
    
    ts_sim = stan_data["Tsim"]
            
    for i in range(num_clus):
        ax = freq_axs[i]
        ## plot data
        ax.scatter(ts_clus, xs_clus_rel[i], s=5, color='k', zorder=3)
        ## plot prediction
        lf, mf, hf = np.percentile(freqs[:,i,:], axis=0, q=[2.5, 50, 97.5])
        ax.plot(ts_sim, mf, color=clus_colors[i], zorder=2)
        ax.fill_between(ts_sim, lf, hf, alpha=0.5, linewidth=0, color=clus_colors[i], zorder=1)
        ## plot simulations
        for j, t in enumerate(ts_clus):
            l, u = np.percentile(freqs_sim[:,i,j], q=[2.5, 97.5])
            ax.plot([t,t], [l, u], color=clus_colors[i], linewidth=1, alpha=0.2, zorder=2)   
        ax.set_yscale('log')
        ax.set_xticks(xticks)

        ax.yaxis.set_major_formatter(SemiLogSciFormatter(minor_thresholds=minor_thresholds))
        ax.yaxis.set_minor_formatter(SemiLogSciFormatter(minor_thresholds=minor_thresholds))
        
            
    ## total number of cells
    ax = count_ax

    # storing the stan variables in the pickle...
    counts = stan_vars["counts"] * count_scaling
    counts_sim = stan_vars["counts_sim"] * count_scaling
        
    ## plot prediction
    lc, mc, hc = np.percentile(counts, axis=0, q=[2.5, 50, 97.5])
    ax.plot(ts_sim, mc, color='blue', zorder=2)
    ax.fill_between(ts_sim, lc, hc, alpha=0.3, color='blue', linewidth=0, zorder=1)
    ## plot data
    ax.scatter(ts_counts, xs_counts, s=5, color='k', zorder=3)
    ## plot simulations
    lo, hi = np.percentile(counts_sim, q=[2.5, 97.5], axis=0)
    ax.fill_between(ts_sim, lo, hi, color='blue', alpha=0.2, linewidth=0, zorder=1)
    ax.set_yscale('log')
    ax.set_xticks(xticks)
    ax.yaxis.set_major_formatter(SemiLogSciFormatter()) # Andy-complient ticklabels


def plot_diff_matrix(
    ax, Q, pop_names, 
    colorbar_args=None,
    num_cticks=10,
    ctick_formatter=None,
    x_rotation=90, y_rotation=None,
    tick_fontsize="medium",
):
    mQ = np.mean(Q, axis=0)
    num_pops = mQ.shape[0]
    pos = range(num_pops)
    fig = ax.get_figure()

    colorbar_args = {} if colorbar_args is None else colorbar_args

    ctick_formatter = "{:0.3f}" if ctick_formatter is None else ctick_formatter

    if any(mQ.flatten() > 0):
        norm = matplotlib.colors.TwoSlopeNorm(vmin=np.min(mQ), vcenter=0, vmax=np.max(mQ))
        cs = ax.pcolor(mQ, cmap="RdBu_r", norm=norm)
        cb = fig.colorbar(cs, **colorbar_args)
        cb.set_label("Differentiation rate ($Q_{ij}$)")
        ## get evenly-spaced ticks by using Normalization's inverse method
        cticks1 = norm.inverse(np.linspace(0, 0.5, num_cticks))
        cticks2 = norm.inverse(np.linspace(0.5, 1, num_cticks))
        cticks = np.concatenate([cticks1, cticks2[1:]])
        cb.set_ticks(cticks)
        cb.set_ticklabels([ctick_formatter.format(x) for x in cticks])

    pos = np.linspace(0, num_pops-1, num_pops) + 0.5
    ax.set_xticks(pos)
    ax.set_xticklabels(pop_names, rotation=x_rotation, ha='right', fontsize=tick_fontsize)
    ax.set_yticks(pos)
    ax.set_yticklabels(pop_names, rotation=y_rotation, fontsize=tick_fontsize)

    ax.set_xlabel("source")
    ax.set_ylabel("destination")
    
    
    
def plot_means(ax, ts, xs, fun=sts.gmean, **kwargs):
    """
    This function expects times (ts) and values (xs).
    At each unique time point, values are collected and 
    the geometric mean is calculated. The result is plotted on ax.
    """
    uts = sorted(util.unique(ts))
    xss = [[x for t, x in zip(ts, xs) if t == u] for u in uts]
    mxs = [fun(x) for x in xss]
    ax.plot(uts, mxs, **kwargs)
    
    
    
def plot_diff_matrix_vae(
    ax,
    Qsams: torch.Tensor,
    unique_clus: list[int], 
    clus_order: list[int], 
    celltypedict: Mapping[int, str],
    cbar_loc: str | None = None,
    num_cticks: int = 10,
    ctick_formatter: str | None = None
) -> matplotlib.colorbar.Colorbar:
    Qmean = torch.mean(Qsams, axis=0)
    Qmean = build_Q_mat(Qmean)
    Qmean = Qmean.numpy()
    Qmean[:,:] = Qmean[:,clus_order]
    Qmean[:,:] = Qmean[clus_order,:]

    vmin, vmax = np.min(Qmean), np.max(Qmean)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    C = ax.pcolor(Qmean, cmap='RdBu_r', norm=norm)
    fig = ax.get_figure()
    cx = fig.colorbar(C, ax=ax, location=cbar_loc)
    
    ## get evenly-spaced ticks by using Normalization's inverse method
    cticks1 = norm.inverse(np.linspace(0, 0.5, num_cticks))
    cticks2 = norm.inverse(np.linspace(0.5, 1, num_cticks))
    cticks = np.concatenate([cticks1, cticks2[1:]])
    ctick_formatter = "{:0.3f}" if ctick_formatter is None else ctick_formatter

    cx.set_ticks(cticks)
    cx.set_ticklabels([ctick_formatter.format(x) for x in cticks])
    cx.set_label("$Q_{ij}$")
    
    ax.set_xticks(np.array(unique_clus) + 0.5)
    ax.set_yticks(np.array(unique_clus) + 0.5)
    ax.set_xticklabels([celltypedict[c] for c in unique_clus], rotation=90)
    ax.set_yticklabels([celltypedict[c] for c in unique_clus])
    
    ax.set_xlabel("source population")
    ax.set_ylabel("destination")
    
    return cx
    
    
def remove_axes_keep_labels(ax: matplotlib.axes.Axes) -> None:
    """
    remove the axes from a matplotlib axes object
    but keep the labels. This is useful for making
    x and y labels that are meant to be used in a figure with
    multiple subplots. 

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        the axes to modify

    Returns
    -------
    None
    """

    for _, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])



def umap_cluster_plot(ax, zs, cs, clus, colors):
    """
    Make a UMAP plot that shows the identified clusters.
    This function is used to visualize the result of Leiden clustering.
    And can be used to further annotate clusters, or find 
    consensus clusters.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        the axes on which to plot the UMAP.
    zs : np.ndarray
        the UMAP coordinates of the cells.
    cs : np.ndarray
        the cluster labels of the cells.
    clus : list[int]
        a lsit of cluster labels (in a specific order).
        these names are used to label the clusters in the plot.
    colors : list[str]
        a list of colors to use for the clusters.

    Returns
    -------
    None
    """
    means = util.get_mean_per_cluster(zs, cs, label_order=clus)    
    for i, cl in enumerate(clus):
        sel = cs == cl
        cc = colors[i]
        ax.scatter(*zs[sel].T, s=1, linewidths=0, color=cc, alpha=0.3, rasterized=True)
        xi, yi = means[i]
        ax.text(xi, yi, cl, va='center', ha='center', bbox = dict(boxstyle="circle", color=cc))
    
    # remove x and y ticks and set labels
    [f([]) for f in [ax.set_xticks, ax.set_yticks]]
    [f(f"UMAP {i+1}") for i, f in enumerate([ax.set_xlabel, ax.set_ylabel])]
        



def heatmap_cluster_plot(ax, xs, cs, feat, clus, colors):
    """
    Make a heatmap of the mean feature values per cluster.
    This function can be used to annotate clusters based on
    marker expression.

    Parameters
    ----------

    ax : matplotlib.axes.Axes
        the axes on which to plot the heatmap.
    xs : np.ndarray
        the feature values of the cells.
    cs : np.ndarray
        the cluster labels of the cells.
    feat : list[str]
        a list of feature names (in a specific order).
        these names are used a ticks lables for the features in the heatmap.
    clus : list[int]
        a list of cluster labels (in a specific order).
        these names are used as ticklabels for the clusters in the heatmap.
    colors : list[str]
        a list of colors to use for the clusters.
        The colors are used to color the y-ticks of the heatmap.

    Returns
    -------
    None
    """
    MFI = util.get_mean_per_cluster(
        xs, cs, label_order=clus
    )
    MFI = sts.zscore(MFI, axis=0)
    simple_heatmap(ax, MFI, feat, clus)
    
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(colors[i])
