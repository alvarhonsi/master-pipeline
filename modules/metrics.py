import numpy as np
from scipy.spatial import KDTree

# https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py
def knn_distance(point, sample, k):
    """ Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample` """
    dist = np.linalg.norm(sample - point, axis=1)
    return np.sort(dist)[k]

def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert(len(s1.shape) == len(s2.shape) == 2)
    # Check dimensionality of sample is identical
    assert(s1.shape[1] == s2.shape[1])

def KL_div_knn_estimation(s1, s2, k=1):
    """ KL-Divergence estimator using scipy's KDTree
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    nu_d,  nu_i   = KDTree(s2).query(s1, k)
    rho_d, rhio_i = KDTree(s1).query(s1, k+1)


    # KTree.query returns different shape in k==1 vs k > 1
    if k > 1:
        D += (d/n)*np.sum(np.log(nu_d[::, -1]/rho_d[::, -1]))
    else:
        D += (d/n)*np.sum(np.log(nu_d/rho_d[::, -1]))

    return D


# https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
        Samples from distribution P, which typically represents the true
        distribution.
    y : 2D array (m,d)
        Samples from distribution Q, which typically represents the approximate
        distribution.
    Returns
    -------
    out : float
        The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    print(x.shape, y.shape)

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.

    # -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

    a = r/(s)
    print("a", a)
    b = m / (n - 1.)
    print("b", b)
    c = n + np.log(b)
    print("c", c)

    return -np.log(a).sum() * d / n + np.log(b)

def difference_mean(s1, s2):
    """ Metric that compares the mean of the samples
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        return: difference between the means
    """
    s1_mean, s2_mean = np.mean(s1, axis=0), np.mean(s2, axis=0)
    return np.mean(np.abs(s1_mean - s2_mean))

def difference_std(s1, s2):
    """ Metric that compares the std of the samples
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        return: difference between the stds
    """
    s1_std, s2_std = np.std(s1, axis=0), np.std(s2, axis=0)
    return np.mean(np.abs(s1_std - s2_std))

def difference_var(s1, s2):
    """ Metric that compares the variances of the samples
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        return: difference between the variances
    """
    s1_var, s2_var = np.var(s1, axis=0), np.var(s2, axis=0)
    return np.mean(np.abs(s1_var - s2_var))