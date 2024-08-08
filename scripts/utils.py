import numpy as np
import math
from scipy.linalg import svd
import random

def sample_k_movies(A ,k, m):
    """
    This function takes a matrix A (each row represents a movie)
    and returns the matrix of just the first k movies with the m forst columns (the ones with more ratings)
    """
    indices = random.sample([i for i in range(A.shape[0])], k)
    return A[indices][:,:m], indices




def matrix_rank(A, tol=None):
    """ Compute matrix rank using SVD with a given tolerance. """
    u, s, vh = svd(A, full_matrices=False)
    if tol is None:
        tol = np.max(A.shape) * np.amax(s) * np.finfo(s.dtype).eps
    return np.sum(s > tol)



