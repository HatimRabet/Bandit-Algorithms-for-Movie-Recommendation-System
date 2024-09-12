import numpy as np
import math
from utils import matrix_rank


# G-Optimal Design Agent Class

class MABAgent:
    def __init__(self, A, delta, epsilon, n_rounds, true_rewards, rewardsXmovie_indices, actionXmovie_indices):
        self.A = A
        self.delta = delta
        self.epsilon = epsilon
        self.n_rounds = n_rounds
        self.d = A.shape[1]
        self.true_rewards = true_rewards
        self.m = matrix_rank(A)
        self.theta = np.random.rand(self.d,1)
        self.cumulative_regret = [0]
        self.regret = [0]
        self.V_l = np.zeros((self.d, self.d))
        self.sum_reward = np.zeros(self.d)
        self.rewardsXmovie_indices = rewardsXmovie_indices  
        self.actionXmovie_indices = actionXmovie_indices
        self.pi = np.ones(A.shape[0]) / A.shape[0]

    def get_pi(self):
        return self.pi
    
    
    def V(self):
        shape = self.A.shape[1]
        V_pi = np.zeros((shape, shape))

        for a, pi_a in zip(self.A, self.pi.flatten()):
            V_pi += pi_a * np.outer(a, a)

        return V_pi

    def g(self):
        V_pi = self.V()

        inv_V_pi = np.linalg.inv(V_pi + 1*np.eye(V_pi.shape[0]))
        max_g = -float("inf")
        a_max = None
        for a in self.A:
            a_col = a[:,np.newaxis]
            g_a = a_col.T @ inv_V_pi @ a_col
            if g_a > max_g:
                max_g = g_a
                a_max = a
        return max_g, a_max.T

    def na(self, l):
        epsilon = 2**(-l)
        common_term = (2 * self.d * np.log(self.A.shape[0] * (l * (l + 1)) / self.delta)) / epsilon ** 2
        Na = [math.ceil(common_term * pi_a) for pi_a in self.pi]
        return Na
    
    def compute_max(self, action_idx):
        a = self.A[action_idx]
        b = self.A
        max_value = np.max(self.theta.T @ (b.T - a[:, np.newaxis]))
        return max_value

    
    def elimination_phase(self, epsilon):
        remaining = [self.A[action_idx] for action_idx in range(self.A.shape[0]) if self.compute_max(action_idx) < 2*epsilon]
        self.A = np.array(remaining)

    def frank_wolfe_algo(self):
        g_pi_k = self.g()[0]
        self.pi = np.ones(self.A.shape[0]) / self.A.shape[0]
        while g_pi_k > (1 + self.epsilon) * self.m:
            g_pi_k, a_k = self.g()
            gamma_k = ((1 / self.d) * g_pi_k - 1) / (g_pi_k - 1)
            indicator_vector = np.zeros(self.A.shape[0])
            row_index = np.flatnonzero(np.all(self.A == a_k, axis=1))[0]
            indicator_vector[row_index] = 1
            self.pi = (1 - gamma_k) * self.pi + gamma_k * indicator_vector
        
        self.pi = self.pi.flatten()
        return self.pi

    def select_action(self):
        return np.random.choice(len(self.A), p=self.pi)

    def update(self, action_indices, true_rewards):
        self.sum_reward = 0
        for action, reward in zip(action_indices, true_rewards):
            a = self.A[action][:,np.newaxis] # row to column vector
            self.V_l +=  np.outer(a, a)
            self.sum_reward += a * reward

        self.theta = np.linalg.inv(self.V_l + 1*np.eye(self.V_l.shape[0])) @ self.sum_reward
        

    def reward(self, action_idx):
        movie_id = self.actionXmovie_indices[action_idx]
        rewards_indices = np.where(self.rewardsXmovie_indices == movie_id)[0]
        if rewards_indices.size == 0:
            raise ValueError(f"No rewards found for movie_id {movie_id}")
        reward_index = np.random.choice(rewards_indices)
        return self.true_rewards[reward_index]
    
    def run(self):
        l = 1
        t = 0
        while t < self.n_rounds:
            self.pi = self.frank_wolfe_algo()
            N_a = self.na(l)
            N = sum(N_a)
            action_indices = []
            true_rewards = []

            for action_idx, n_a in enumerate(N_a):
                for _ in range(n_a):

                    true_reward = np.max(self.A @ self.theta)
                    reward = self.A[action_idx] @ self.theta
                    regret = (true_reward - reward)
                    self.cumulative_regret.append(self.cumulative_regret[-1] + regret)
                    self.regret.append(regret)
                    action_indices.append(action_idx)
                    true_rewards.append(true_reward)

            self.update(action_indices, true_rewards)

            epsilon = 2**(-l)
            self.elimination_phase(epsilon)

            t += N
            l += 1
        return self.cumulative_regret, self.regret, self.theta





# LinUCB Agent Class


class LinUCB_MABAgent:
    def __init__(self, A, delta, lambda_, n_rounds, true_rewards, rewardsXmovie_indices, actionXmovie_indices):
        self.A = A
        self.delta = delta
        self.n_rounds = n_rounds
        self.true_rewards = true_rewards
        self.rewardsXmovie_indices = rewardsXmovie_indices
        self.actionXmovie_indices = actionXmovie_indices
        self.lambda_ = lambda_
        self.beta = np.sqrt(lambda_) + np.sqrt(2 * np.log(1/delta))
        self.d = A.shape[1]
        self.n_actions = A.shape[0]
        self.V = lambda_*np.eye(self.d)  # Identity matrix for the initial V
        self.b = np.zeros(self.d)  # Mean rewards vector
        self.theta = np.zeros(self.d)
        self.ucb_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)
        self.cumulative_regret = [0]
        self.regret = [0]
        self.time = 0

    def ucb_values_fn(self):
        A_inv = np.linalg.inv(self.V)

        for i in range(self.n_actions):
            a = self.A[i]
            ucb = a @ self.theta 
            upper_conf = self.beta * np.sqrt(a @ A_inv @ a.T)
            self.ucb_values[i] = ucb + upper_conf
        
        return 

    def select_action(self):
            
        return np.argmax(self.ucb_values)
    
    def update(self, action_idx, reward):
        a = self.A[action_idx]
        self.V += np.outer(a, a)
        self.b += reward * a.T
        self.theta = np.linalg.inv(self.V) @ self.b
        self.beta = np.sqrt(self.lambda_) + np.sqrt(2 * np.log(1/self.delta) + self.d*np.log(1 + (self.time-1)/(self.lambda_*self.d)))
        self.action_counts[action_idx] += 1

    def reward(self, action_idx):
        movie_id = self.actionXmovie_indices[action_idx]
        rewards_indices = np.where(self.rewardsXmovie_indices == movie_id)[0]
        if rewards_indices.size == 0:
            raise ValueError(f"No rewards found for movie_id {movie_id}")
        reward_index = np.random.choice(rewards_indices)
        return self.true_rewards[reward_index]

    def run(self):
        while self.time < self.n_rounds:
            self.ucb_values_fn()
            action_idx = self.select_action()
            reward = self.reward(action_idx)
            true_reward = np.max(self.A @ self.theta)

            self.update(action_idx, reward)
            regret = (true_reward - reward)

            self.cumulative_regret.append(self.cumulative_regret[-1] + regret)
            self.regret.append(regret)
            self.time += 1
        return self.cumulative_regret, self.regret, self.theta




# UCB Agent Class

class UCB_MABAgent:
    def __init__(self, n_rounds, n_actions, max_reward, true_rewards, rewardsXmovie_indices, actionXmovie_indices):
        self.n_rounds = n_rounds
        self.true_rewards = true_rewards
        self.max_reward = max_reward
        self.cumulative_regret = [0]
        self.rewardsXmovie_indices = rewardsXmovie_indices  
        self.actionXmovie_indices = actionXmovie_indices
        self.n_actions = n_actions
        self.action_counts = np.zeros(self.n_actions)
        self.total_reward = np.zeros(self.n_actions)
        self.UCB_values = np.zeros(self.n_actions)
        self.time = 0

    def select_action(self):
        # Compute UCB values
        if self.time < self.n_actions:
            # Select each action at least once
            return self.time
        self.UCB_values = (self.total_reward / (self.action_counts + 1)) + np.sqrt(2 * np.log(self.time + 1) / (self.action_counts + 1))
        return np.argmax(self.UCB_values)

    def update(self, action_idx, reward):
        self.action_counts[action_idx] += 1
        self.total_reward[action_idx] += reward

    def reward(self, action_idx):
        movie_id = self.actionXmovie_indices[action_idx]
        rewards_indices = np.where(self.rewardsXmovie_indices == movie_id)[0]
        if rewards_indices.size == 0:
            raise ValueError(f"No rewards found for movie_id {movie_id}")
        reward_index = np.random.choice(rewards_indices)
        return self.true_rewards[reward_index]

    def run(self):
        while self.time < self.n_actions:
            action_idx = self.time
            reward = self.reward(action_idx)
            self.update(action_idx, reward)

            self.cumulative_regret.append(self.cumulative_regret[-1] + (self.max_reward - reward))
            self.time += 1

        while self.time < self.n_rounds:
            action_idx = self.select_action()
            reward = self.reward(action_idx)
            self.update(action_idx, reward)

            self.cumulative_regret.append(self.cumulative_regret[-1] + (self.max_reward - reward))
            self.time += 1
        return self.cumulative_regret




# Thompson Sampling Agent Class


class TS_MABAgent:
    def __init__(self, n_rounds, n_actions, sigma, max_reward, true_rewards, rewardsXmovie_indices, actionXmovie_indices):
        self.n_rounds = n_rounds
        self.sigma = sigma
        self.true_rewards = true_rewards
        self.max_reward = max_reward
        self.cumulative_regret = [0]
        self.rewardsXmovie_indices = rewardsXmovie_indices  
        self.actionXmovie_indices = actionXmovie_indices
        self.n_actions = n_actions
        self.action_counts = np.zeros(self.n_actions)
        self.total_reward = np.zeros(self.n_actions)
        self.UCB_values = np.zeros(self.n_actions)
        self.time = 0

    def select_action(self):
        # Sample from Beta distribution for each action
        sampled_values = np.random.normal(self.total_reward / self.action_counts, (self.sigma)**2 / self.action_counts)
        return np.argmax(sampled_values)

    def update(self, action_idx, reward):
        self.action_counts[action_idx] += 1
        self.total_reward[action_idx] += reward

    def reward(self, action_idx):
        movie_id = self.actionXmovie_indices[action_idx]
        rewards_indices = np.where(self.rewardsXmovie_indices == movie_id)[0]
        if rewards_indices.size == 0:
            raise ValueError(f"No rewards found for movie_id {movie_id}")
        reward_index = np.random.choice(rewards_indices)
        return self.true_rewards[reward_index]

    def run(self):

        while self.time < self.n_actions:
            action_idx = self.time
            reward = self.reward(action_idx)
            self.update(action_idx, reward)

            self.cumulative_regret.append(self.cumulative_regret[-1] + (self.max_reward - reward))
            self.time += 1

        while self.time < self.n_rounds:
            action_idx = self.select_action()
            reward = self.reward(action_idx)
            self.update(action_idx, reward)

            self.cumulative_regret.append(self.cumulative_regret[-1] + (self.max_reward - reward))
            self.time += 1
        return self.cumulative_regret



# EXP3 (Exponential-weight algorithm for Exploration and Exploitation) Agent Class


class EXP3_MABAgent:
    def __init__(self, A, gamma, n_rounds, n_actions, max_reward, true_rewards, rewardsXmovie_indices, actionXmovie_indices):
        self.gamma = gamma  # Exploration parameter
        self.n_rounds = n_rounds
        self.true_rewards = true_rewards
        self.max_reward = max_reward
        self.cumulative_regret = [0]
        self.rewardsXmovie_indices = rewardsXmovie_indices  
        self.actionXmovie_indices = actionXmovie_indices
        self.n_actions = n_actions
        self.weights = np.ones(self.n_actions)  # Initialize weights to 1 for each action
        self.probabilities = np.ones(self.n_actions) / self.n_actions  # Initial uniform probability distribution
        self.time = 0

    def select_action(self):
        # Normalize weights to get a probability distribution over actions
        total_weight = np.sum(self.weights)
        self.probabilities = (1 - self.gamma) * (self.weights / total_weight) + self.gamma / self.n_actions
        return np.random.choice(self.n_actions, p=self.probabilities)

    def update(self, action_idx, reward):
        estimated_reward = reward / self.probabilities[action_idx]
        self.weights[action_idx] *= np.exp(self.gamma * estimated_reward / self.n_actions)

    def reward(self, action_idx):
        movie_id = self.actionXmovie_indices[action_idx]
        rewards_indices = np.where(self.rewardsXmovie_indices == movie_id)[0]
        if rewards_indices.size == 0:
            raise ValueError(f"No rewards found for movie_id {movie_id}")
        reward_index = np.random.choice(rewards_indices)
        return self.true_rewards[reward_index]

    def run(self):
        while self.time < self.n_rounds:
            action_idx = self.select_action()
            reward = self.reward(action_idx)
            self.update(action_idx, reward)
        
            self.cumulative_regret.append(self.cumulative_regret[-1] + (self.max_reward - reward))
            self.time += 1
        return self.cumulative_regret, self.theta
