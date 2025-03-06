from typing import Optional, Literal

import numpy as np
import scanpy as sc

def bhattacharyya_dist_multi_norm(mu1, mu2, Sigma1, Sigma2):
    """
    compute the Bhattacharayya distance two normal distributions 
    with mean vectors mu1 and mu2 and covarniance 
    matrices Sigma1 and Sigma2.
    """
    Sigmabar = 0.5 * (Sigma1 + Sigma2)
    Deltamu = mu1 - mu2
    z = np.dot(Deltamu, np.linalg.solve(Sigmabar, Deltamu))
    c = np.linalg.det(Sigmabar) / np.sqrt(np.linalg.det(Sigma1) * np.linalg.det(Sigma2))
    return z / 8 + np.log(c) / 2
    
    
def bhattacharyya_dist_sample(Y1, Y2):
    """
    compute Bhattacharayya distance based on two samples
    assuming normal distribtions. The first dimension (rows)
    represent different observations.
    """
    mu1 = np.mean(Y1, axis=0)
    mu2 = np.mean(Y2, axis=0)
    Sigma1 = np.cov(Y1, rowvar=False)
    Sigma2 = np.cov(Y2, rowvar=False)
    return bhattacharyya_dist_multi_norm(mu1, mu2, Sigma1, Sigma2)


def leiden_clus(
        X: np.ndarray, 
        n_neighbors: int, 
        resolution: float, 
        seed: Optional[int] = None, 
        scale: bool = True,
        umap: bool = False,
        init_pos: Literal["paga", "spectral"] = "paga"
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Leiden-cluster data matrix X.
    Internally, the scanpy method is used

    Parameters
    ----------
    X : np.ndarray
        Data matrix with batch dimension on the left, feature dimension on the right.
    n_neighbors : int
        number of neightbors used for the KNN algorithm.
    resolution : float
        Leiden cluster resolution parameter.
    seed : Optional[int], optional
        Random seed. The default is None.
    scale : bool
        should the data X be scaled?. The default is True.
    umap : bool
        compute a UMAP embedding. The default is False
    init_pos : Literal["paga", "spectral"]
        method for choosing initial position for UMAP embedding.
        The default is "paga"

    Returns
    -------
    leiden_clus : np.ndarray
        Assigned Leiden cluster per sample.
    embedding : np.ndarray
        the UMAP embedding. Only returned if umap argument is True

    """
    ann = sc.AnnData(X, dtype=np.float32) # create AnnData object
    if scale:
        sc.pp.scale(ann) # scale the data
    # apply KNN algo
    sc.pp.neighbors(ann, n_neighbors=n_neighbors, random_state=seed)
    # apply Leiden algo
    sc.tl.leiden(ann, resolution=resolution, random_state=seed)
    # extract cluster assignment
    leiden_clus = ann.obs["leiden"].to_numpy()
    
    if not umap:
        return leiden_clus        

    # apply UMAP algo
    if init_pos == "paga":
        sc.tl.paga(ann)
        sc.pl.paga(ann, plot=False, random_state=seed)
    sc.tl.umap(ann, init_pos=init_pos, random_state=seed)
    embedding = ann.obsm["X_umap"]
    return leiden_clus, embedding


