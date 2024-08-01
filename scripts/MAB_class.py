import numpy as np
import math
import random



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
    max_reward = np.max(true_rewards)
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
        
        
        

import numpy as np
import math

class MABAgent:
    def __init__(self, A, delta, epsilon, n_rounds, true_rewards, rewardsXmovie_indices, actionXmovie_indices):
        """
        Initializes the MAB Agent
        Parameters:
        A : numpy array representing the actions
        delta : float, parameter for na function
        epsilon : float, precision parameter for the Frank-Wolfe algorithm
        n_rounds : int, number of rounds for exploration
        true_rewards : dict, mapping actions to their true rewards
        """
        self.A = A
        self.delta = delta
        self.epsilon = epsilon
        self.n_rounds = n_rounds
        self.true_rewards = true_rewards
        self.d = A.shape[1]
        self.theta = np.random.rand(self.d)
        self.cumulative_regret = [0]
        self.V_l = np.zeros((self.d, self.d))
        self.sum_reward = np.zeros(self.d)
        self.max_reward = np.max(true_rewards)
        self.rewardsXmovie_indices = rewardsXmovie_indices  
        self.actionXmovie_indices = actionXmovie_indices
    
    def V(self, pi):
        """
        This function will return the V_pi matrix of a design
        Parameters:
        pi : Array of probabilities associated to each action
        """
        V_pi = np.zeros((self.d, self.d))
        for a, pi_a in zip(self.A, pi):
            V_pi += pi_a * (a.T @ a)
        return V_pi

    def g(self, pi):
        """
        Calculates the g function
        """
        V_pi = self.V(pi)
        inv_V_pi = np.linalg.inv(V_pi)
        max_g = -float("inf")
        a_max = None

        for a in self.A:
            g_a = a @ inv_V_pi @ a.T
            if g_a >= max_g:
                max_g = max(max_g, g_a)
                a_max = a

        return max_g, a_max

    def na(self, pi, epsilon, delta, l, k=2):
        """
        Calculate the number of samples required for each action
        """
        common_term = (2 * self.d * np.log(k * (l * (l + 1)) / delta)) / epsilon ** 2
        Na = []

        for pi_a in pi:
            na = common_term * pi_a
            na = math.ceil(na)
            Na.append(na)

        return Na

    def frank_wolfe_algo(self, epsilon):
        """
        Frank-Wolfe optimization algorithm
        """
        card_A = self.A.shape[0]
        pi_k = np.ones(card_A) / card_A
        g_pi_k = self.g(pi_k)[0]

        while g_pi_k > (1 + epsilon) * self.d:
            g_pi_k, a_k = self.g(pi_k)
            gamma_k = ((1 / self.d) * g_pi_k - 1) / (g_pi_k - 1)
            pi_k = (1 - gamma_k) * pi_k + gamma_k * (self.A == a_k)
        
        return pi_k

    def select_action(self, pi):
        """
        Selects an action based on the given probability distribution pi
        """
        return np.random.choice(len(self.A), p=pi)

    def update(self, action, reward):
        """
        Updates the agent's parameters based on the received reward
        """
        a = self.A[action]
        self.V_l += a.T @ a
        self.sum_reward += a.T * reward
        self.theta = np.linalg.inv(self.V_l) @ self.sum_reward

    def reward(self, action_idx):
        """
        Returns randomly a reward from a user for this action
        """
        movie_id = self.actionXmovie_indices[action_idx]
        # rewards_indices = [i for i in range(self.rewardsXmovie_indices.size) if self.rewardsXmovie_indices[i] == movie_id]
        rewards_indices = np.where(self.rewardsXmovie_indices == movie_id)[0]
        if rewards_indices.size == 0:
            raise ValueError(f"No rewards found for movie_id {movie_id}")
        reward_index = np.random.choice(rewards_indices)
        return self.true_rewards[reward_index]
    

    def run(self):
        """
        Runs the G-optimal design exploration algorithm
        """
        l = 1
        t = 0

        while t < self.n_rounds:
            pi_optimal = self.frank_wolfe_algo(self.epsilon)
            epsilon_l = 2 ** (-l)
            N_a = self.na(pi_optimal, epsilon_l, self.delta, l)
            N = sum(N_a)
            rewards, X_t = np.zeros(N), np.zeros(N)
            tracker = 0

            for action_idx, n_a in enumerate(N_a):
                for _ in range(n_a):
                    true_reward = self.reward(action_idx)
                    reward = self.A[action_idx] @ self.theta
                    self.cumulative_regret.append(self.cumulative_regret[-1] + (self.max_reward - true_reward))
                    self.update(action_idx, true_reward)
                    tracker += 1

            t += N
            l += 1
        
        return self.cumulative_regret, self.theta

                








