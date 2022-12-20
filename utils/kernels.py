import math
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.metrics.pairwise import euclidean_distances


def sigma2gamma(sigma):
    gamma = 1 / (sigma ** 2)
    return gamma


def normalize_linear_kernel(data):
    """Linear kernel with normalization.

    Args:
        data ():

    Returns:

    """
    data = np.divide(data, np.tile(np.sqrt(np.sum(data * data, axis=1)), (data.shape[1], 1)).T)
    a = linear_kernel(data)
    return a


def dense_rbf_kernel(x, y, sigma):
    gamma = sigma2gamma(sigma)
    return rbf_kernel(x, y, gamma=gamma)


def sparse_rbf_kernel(x, y, sigma):
    """

    A_{i j}^{(\sigma, \nu, C)}=
    \left[\left(1-\frac{\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}}{C}\right)^{\nu}\right]^{+}
    \cdot
    \exp \left(\frac{-\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}^{2}}{\sigma^{2}}\right)
    Args:
        x ():
        y ():
        sigma ():

    Returns:

    """
    c = 3 * sigma
    gamma = sigma2gamma(sigma)
    v = math.ceil((x.shape[1] + 1) / 2)
    return ((1 - euclidean_distances(x) / c) ** v).clip(min=0) * rbf_kernel(x, y, gamma=gamma)


def matern_kernel(x, y, sigma, nu):
    """

    Args:
        x ():
        y ():
        sigma (float): The length scale of the kernel. We keep using sigma for consistency.
        nu (float):

    Returns:

    """
    matern = Matern(length_scale=sigma, nu=nu)
    return matern.__call__(x, y)
