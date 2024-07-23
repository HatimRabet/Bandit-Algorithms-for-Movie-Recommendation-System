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


def na(A, pi, epsilon, delta, l, k=2):
    
    d = A.shape[1]
    common_term = (2 * d * np.log(k*(l*(l+1))/delta)) / epsilon**2
    Na = []

    for pi_a in pi:
        na = common_term * pi_a
        na = math.ceil(na)
        Na.append(na)

    return Na


def frank_wolfe_algo(A, epsilon):

    d, card_A = A.shape[1], A.shape[0]
    pi_k = np.ones(card_A)
    pi_k /= card_A
    # n_iterations = int(d * np.log(np.log(card_A)) + d/epsilon)
    g_pi_k = g(A,pi_k)[0]

    while g_pi_k > (1+epsilon) * d :

        g_pi_k, a_k = g(A,pi_k)
        gamma_k = ((1 / d) * g_pi_k - 1) / (g_pi_k - 1)
        pi_k = (1-gamma_k) * pi_k + gamma_k * (A == a_k)
    
    return pi_k

    

def g_optimal_design_exploration_algo(A, delta, epsilon, n_rounds, true_rewards):
    l = 1
    A_i = A.copy()
    d = A.shape[0]
    theta = np.random.rand(d)
    max_reward = max(true_rewards.values())
    cumulative_regret = [0]

    t = 0
    while t < n_rounds:
        pi_optimal = frank_wolfe_algo(A,epsilon)
        epsilon_l = 2**(-l)
        N_a = na(A, pi_optimal, epsilon_l, delta, l)
        N = sum(N_a)
        rewards, X_t = np.zeros(N), np.zeros(N)
        tracker = 0
        V_l = np.zeros((d,d))
        sum_reward = np.zeros(d)
        
        for a, n_a in zip(A_i, N_a):
            times = 0
            while times < n_a:
                r = a @ theta
                rewards[tracker] = r
                true_reward = true_rewards[a] 
                cumulative_regret.append(cumulative_regret[-1] + (max_reward - true_reward))
                X_t[tracker] = true_reward
                sum_reward += a.T * true_reward
                tracker += 1
                times += 1
                
            
            V_l += n_a * a.T @ a
            
        
        theta = np.linalg.inv(V_l) @ sum_reward
        t += N
        l += 1
    
    return cumulative_regret, theta
        
        
        

                








