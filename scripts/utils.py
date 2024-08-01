import numpy as np
import math

def sample_k_movies(A ,k):
    """
    This function takes a matrix A (each row represents a movie)
    and returns the matrix of just the first k movies(the ones with more ratings)
    """
    return A[:k]



