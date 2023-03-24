import numpy as np

#Assumes normal distributions
def KL_divergance_normal(dist1, dist2):
    '''
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    '''
    mu1, mu2 = np.mean(dist1, axis=0), np.mean(dist2, axis=0)
    sigma1, sigma2 = np.std(dist1, axis=0), np.std(dist2, axis=0)
    kl_div = np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5

    return kl_div


def difference_mean(s1, s2):
    """ Metric that compares the mean of the samples
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        return: mean difference between means of each distribution
    """
    s1_mean, s2_mean = np.mean(s1, axis=0), np.mean(s2, axis=0)
    return np.abs(s1_mean - s2_mean)

def difference_std(s1, s2):
    """ Metric that compares the std of the samples
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        return: mean difference between std of each distribution
    """
    s1_std, s2_std = np.std(s1, axis=0), np.std(s2, axis=0)
    return np.abs(s1_std - s2_std)