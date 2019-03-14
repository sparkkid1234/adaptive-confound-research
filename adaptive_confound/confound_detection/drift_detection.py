"""
Each function must accept the following parameters:
  - X_train, y_train, z_train, z_cmp_train the data at training time
  - X_test, y_test, z_test, z_cmp_test the data at testing time
  - A logistic regression classifier fitted on the training data.
  
Not all parameters need to be used by every method but for
compatibility, they must still be present in the list of parameters.

Each function must return a real number between 0 and 1
indicating the amount of drift detected.
"""
import numpy as np
import pandas as pd
from copy import copy
from tqdm import tqdm_notebook
from scipy.stats import entropy, norm
from sklearn.linear_model import LogisticRegression


# Method 1: compute the difference between p(x|z) at training time
# and p(x|z) at testing time using the Hellinger distance or the
# Jensen-Shannon divergence.

SQRT2 = np.sqrt(2)


def hd(p, q):
    """
    Computes the Hellinger distance between two probability distributions.
    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / SQRT2


def jsd(p, q, base=np.e):
    """
    Computes the Jensen-Shannon divergence between two distributions.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    m = 1. / 2 * (p + q)
    return entropy(p, m, base=base) / 2. + entropy(q, m, base=base) / 2.


def compute_pxgz(X, z, num_z):
    """
    Computes p(X|z) for the given data sample where `X` is a term-document matrix
    and `z` is a vector of integers representing the value of a covariate.
    """
    nrows = num_z
    ncols = X.shape[1]
    r = np.ones((nrows, ncols))
    for zval in range(num_z):
        keep = (z == zval).flatten()
        r[zval] += X[keep].sum(axis=0).A1
        r[zval] += 1 # smoothing and avoid division by zero
        r[zval] /= r[zval].sum()
    return r


def pxgz_diff(X_tr, y_tr, z_tr, X_te, y_te, z_te, dist, num_z):
    """
    Compute drift value as a change in p(X|z) using a statistical distance `dist`.

    Computes p(X|z) at training time and testing time. Then measures the distance
    between the two probability distributions using `dist` and returns the 
    average distance.
    """
    tr_pxgz = compute_pxgz(X_tr, z_tr, num_z)
    te_pxgz = compute_pxgz(X_te, z_te, num_z)
    dists = [dist(p_tr, p_te) for p_tr, p_te in zip(tr_pxgz, te_pxgz)]
    return np.mean(dists)


def pxgz_diff_hd(X_tr, y_tr, z_tr, X_te, y_te, z_te, num_z):
    """
    Compute drift value as a change in p(X|z) using Hellinger distance.
    """
    return pxgz_diff(X_tr, y_tr, z_tr, X_te, y_te, z_te, hd, num_z)


def pxgz_diff_jsd(X_tr, y_tr, z_tr, X_te, y_te, z_te, num_z):
    """
    Compute drift value as a change in p(X|z) using Jensen-Shannon divergence.
    """
    return pxgz_diff(X_tr, y_tr, z_tr, X_te, y_te, z_te, jsd, num_z)