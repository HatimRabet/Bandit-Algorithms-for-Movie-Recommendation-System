import gurobipy
import numpy as np



def V(A, pi):
    """
    This function will return the V_pi matrix of a design
    Parameters : 
    A : numpy array representing the actions
    pi : Array of probabilities associated to each action
    """

    V_pi = A @ pi @ A.T
    return V_pi

 
