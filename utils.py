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
        V_pi += pi_a * (a.T @ a)

    return V_pi

def g(A, pi):
    """
    """

    V_pi = V(A,pi)
    inv_V_pi = np.linalg.inv(V_pi)
    max_g = -float("inf")
    a_max = None

    for a in A:
        g_a = a @ inv_V_pi @ a.T
        if g_a >= max_g:
            max_g = max(max_g, g_a)
            a_max = a

    return max_g, a_max


def na(A, pi, epsilon, delta):

    common_term = (2 * np.log(1/delta)) / epsilon**2
    g_pi = g(A,pi)
    Na = []

    for pi_a in pi:
        na = common_term * pi_a * g_pi
        na = math.ceil(na)
        Na.append(na)

    return Na


def frank_wolfe_algo(A, epsilon):

    d, card_A = A.shape[1], A.shape[0]
    pi_k = np.ones(card_A)
    pi_k /= card_A
    n_iterations = int(d * np.log(np.log(card_A)) + d/epsilon)

    for _ in range(n_iterations):

        g_pi_k, a_k = g(A,pi_k)
        gamma_k = ((1 / d) * g_pi_k - 1) / (g_pi_k - 1)
        pi_k = (1-gamma_k) * pi_k + gamma_k * (A == a_k)
    
    return pi_k

    

def g_optimal_design_exploration_algo(A, delta, epsilon):
    l = 1
    A_i = A.copy()





