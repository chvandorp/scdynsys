import matplotlib
import colorsys
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.stats as sts
from collections import Counter
import pandas as pd

from typing import List, Dict, Callable, Optional

def transformed_sample(
    mu: np.ndarray, 
    cov: np.ndarray, 
    transform: Callable = lambda x: x, 
    n: int = 1000
) -> np.ndarray:
    """
    sample from the multivariate normal distribution and then 
    transform the sampled values
    """
    xs = sts.multivariate_normal.rvs(mean=mu, cov=cov, size=n)
    return transform(xs)


def density_2d(xs: np.ndarray, ys: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """
    Compute density of points (x, y) with all the x-coords given in xs
    and all the y coords in ys. This is just a wrapper around
    scipy.stats.gaussian_kde

    Parameters
    ----------
    xs : np.ndarray
        x-coords of 2D points.
    ys : np.ndarray
        y-coords of 2D points.
    n: Optional[int]
        The maximum number of points to use for KDE fitting.
        Default is None.

    Returns
    -------
    np.ndarray
        density of points (x, y)

    """
    xys = np.stack([xs, ys])
    if n is not None and n < xys.shape[1]:
        idx = np.random.choice(xys.shape[1], n, replace=False)
        return sts.gaussian_kde(xys[:, idx])(xys)
    return sts.gaussian_kde(xys)(xys)
    
    
def select_square_gate(xss: list[np.ndarray], gates: list[tuple[float, float]]) -> np.ndarray:
    selected = np.array([True for _ in xss[0]])
    for xs, gate in zip(xss, gates):
        lb, ub = gate
        selected = selected & (xs > lb) & (xs < ub)
    return selected
    


    

def scale_color(col, scale) -> tuple[float]:
    """
    Scale a color to combine density of points with color to indicate a cluster label
    """
    rgb = matplotlib.colors.ColorConverter.to_rgb(col)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    l_new = l + (1-l) * (1-scale) * 0.9
    return colorsys.hls_to_rgb(h, l_new, s)


def get_mean_per_cluster(xs, labels, label_order=None, fun=np.mean) -> np.ndarray:
    """
    compute the mean value of points in given clusters.
    All points are given by the numpy array xs 
    and the clusters are defined by the list labels.
    """
    if label_order is None:
        label_order = sorted(list(set(labels)))
    num_clusters = len(label_order)
    means = np.zeros((num_clusters, xs.shape[1]))
    for j, ID in enumerate(label_order):
        idxs = [i for i, c in enumerate(labels) if c == ID]
        ys = xs[idxs,:]
        means[j,:] = fun(ys, axis=0)
    return means


def compute_DA_matrix(cell_type_per_sample, unit_per_sample, subset=None) -> pd.DataFrame:
    """
    Take a list of cell types and a list of individual IDs, and compile a 
    data frame containing the counts for each cell type and each individual.
    Optionally, the function takes a parameter `subset` (a boolean list)
    to subset the data (e.g to only select tetramer positive cells).
    The data frame has colums indexed by the individuals and rows indexed by
    the cell types.

    Remark: DA stands for Differential Abundance.


    Parameters
    ----------
    cell_type_per_sample : TYPE
        list of cell types
    unit_per_sample : TYPE
        list of units (or individual IDs).
    subset : TYPE, optional
        list of True/False used for subsetting the data. The default is None.

    Returns
    -------
    count_df : pd.DataFrame
        pandas data frame with all counts.

    """
    cell_type_sizes = Counter(cell_type_per_sample)
    cell_types = sorted(list(cell_type_sizes.keys()), key=lambda x : cell_type_sizes[x], reverse=True)

    num_cell_types = len(cell_type_sizes)
    
    units = sorted(list(set(unit_per_sample)))
    num_units = len(units)

    count_df = pd.DataFrame(
        np.zeros((num_cell_types, num_units), dtype=int), 
        columns=units, index=cell_types
    )

    for unit in units:
        if subset is None:
            counts = Counter(cell_type_per_sample[unit_per_sample == unit])
        else:
            counts = Counter(cell_type_per_sample[(unit_per_sample == unit) & subset])
        count_df[unit] = np.array([counts.get(k, 0) for k in cell_types], dtype=int)
        
    return count_df


def relabel(labs: List[int]) -> List[int]:
    """
    Some clustering algorithms return non-consecutive labels.
    Replace these labels with 0, 1, 2, ...
    """
    ulabs = sorted(list(set(labs)))
    mapping = dict((l, i) for i, l in enumerate(ulabs))
    new_labs = [mapping[l] for l in labs]
    return new_labs


organ_map = {
    "Lung" : "Lung",
    "Jej" : "Jejenum",
    "MLN" : "MLN",
    "LLN" : "LLN",
    "dLN" : "dLN",
    "LN" : "dLN",
    "Spl" : "Spleen",
    "Spleen" : "Spleen"
}

def find_organ_name(sample_name):
    for k, v in organ_map.items():
        if k in sample_name:
            return v
    return "NA"


def unique(xs) -> List:
    """
    find unique elements in a list `xs`. Sort in order of
    appearence in `xs`.
    """
    uxs = list(set(xs))
    uxs.sort(key=lambda x: list(xs).index(x))
    return uxs


def count_unique_values(xs) -> Dict:
    """
    make a histogram of the unique values in `xs`
    """
    us = set(xs)
    counts = [sum([x == u for x in xs]) for u in us]
    return dict(zip(us, counts))


def sign_stars(p: float) -> str:
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    return ''

def nn_analysis(xs, labels, unique_labels, n_neighbors=50, n_jobs=-1, relative=False):
    """
    For each point in xs, use the kNN to compute entropy 
    of the distributio of local labels. Also do this for a random
    (uniform) sample, taking label frequencies into account.
    Entropy is higher when distribution is more uniform.
    This function can be used to verify the effect of batch correction.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=n_jobs).fit(xs)
    _, indices = nbrs.kneighbors(xs)

    ## we want to create a historgram of batch counts for each cell
    
    n_points = xs.shape[0]
    points = np.arange(n_points)
    n_labels = len(unique_labels)
    label_index_dict = {b : i for i, b in enumerate(unique_labels)}
    label_indices = np.vectorize(label_index_dict.get)(labels)

    p_null = np.bincount(label_indices)
    p_null = p_null / len(labels)
    print("background label distribution:", p_null)

    # in order to use histogram2d, we have to label each point (i.e. to separate points)
    ss, x_edges, y_edges = np.histogram2d(
        label_indices[indices].flatten(), 
        np.repeat(points, n_neighbors),
        bins=(n_labels, n_points),
        range=np.array([[0, n_labels], [0, n_points]])
    )
    qk = np.expand_dims(p_null, 1) if relative else None
    hs = sts.entropy(ss, qk=qk, axis=0)
    random = sts.multinomial.rvs(n_neighbors, p=p_null, size=len(points)).T
    hs_rand = sts.entropy(random, qk=qk, axis=0)

    return hs, hs_rand



