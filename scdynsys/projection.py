"""
Some tools for low-dimensional projections / dimension reduction
and visualizations
"""

import umap
import torch
import numpy as np


def umap_embedding(xs, dim=2, random_state=None):
    reducer = umap.UMAP(random_state=random_state, n_components=dim)
    reducer.fit(xs)
    return reducer.transform(xs), reducer

class Identity():
    def transform(self, x):
        return x

def apply_umap(raw_latent, dim=2, random_state=None):
    print("doing UMAP...")
    if raw_latent.shape[1] > 2:
        raw_latent_umap, reducer = umap_embedding(raw_latent, dim=dim, random_state=random_state)
    else:
        raw_latent_umap = raw_latent
        reducer = Identity()
    return raw_latent_umap, reducer


def align_maps(X, Y, mirror=False, lr=1e-2, epochs=100):
    """
    look for a rotation matrix R so that R X is most similar to Y
    first dimension is batch dim, second dimension should have size 2.
    Some manual gradient descent in pytorch for fun!
    """
    assert X.shape == Y.shape and X.shape[-1] == 2
    ## rotation method
    def rotate_and_translate(X, theta, b):
        u, v = torch.cos(theta), torch.sin(theta)
        R = torch.stack([torch.stack([u, v]), torch.stack([-v, u])])
        return (X + b) @ R
        
    def mirror_func(X):
        M = np.array([[-1, 0], [0, 1]])
        return X @ M
        
    def target(theta, b, X, Y):
        RXpb = rotate_and_translate(X, theta, b)
        return torch.mean(torch.square(RXpb - Y))
    
    theta = torch.tensor(0.0, requires_grad=True)
    b = torch.zeros((2,), requires_grad=True)
    
    if mirror:
        X_tens = torch.tensor(mirror_func(X), dtype=torch.float, requires_grad=True)
    else:
        X_tens = torch.tensor(X, dtype=torch.float, requires_grad=True)
    
    Y_tens = torch.tensor(Y, dtype=torch.float, requires_grad=True)
    
    losses = []
    for i in range(epochs):
        loss = target(theta, b, X_tens, Y_tens)
        loss.backward()
        with torch.no_grad():
            theta -= theta.grad * lr
            b -= b.grad * lr
        
        theta.grad = None
        b.grad = None
        losses.append(loss.detach().numpy())
        
    result = {
        "theta" : theta.detach().numpy(),
        "b" : b.detach().numpy(),
        "aligned_data" : rotate_and_translate(X_tens, theta, b).detach().numpy(),
        "losses" : losses
    }
        
    return result

def O2_transform(xy, alpha=0.0, reflection=False):
    """
    planar orthogonal transformation O(2).
    First apply a rotation of angle alpha,
    then an (optional) reflection in the y-axis.
    """
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    rxy = np.dot(R, xy)
    if not reflection:
        return rxy
    # else
    T = np.array([[-1,0],[0,1]])
    return np.dot(T, rxy)
