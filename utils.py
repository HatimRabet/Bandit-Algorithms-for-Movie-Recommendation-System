import numpy as np
import math



def V(A, pi):
    """
    This function will return the V_pi matrix of a design
    Parameters : 
    A : numpy array representing the actions
    pi : Array of probabilities associated to each action
    """

    V_pi = 0
    for a, pi_a in zip(A,pi):
        V_pi += pi_a * (a @ a.T)

    return V_pi

def g(A, pi):
    """
    """

    V_pi = V(A,pi)
    inv_V_pi = np.linalg.inv(V_pi)
    max_g = -float("inf")

    for a in A:
        g_a = a.T @ inv_V_pi @ a
        max_g = max(max_g, g_a)

    return max_g


def na(A, pi, epsilon, delta):

    common_term = (2 * np.log(1/delta)) / epsilon**2
    g_pi = g(A,pi)
    Na = []

    for pi_a in pi:
        na = common_term * pi_a * g_pi
        na = math.ceil(na)
        Na.append(na)

    return Na



