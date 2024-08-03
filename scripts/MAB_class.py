# import numpy as np
# import math

# class MABAgent:
#     def __init__(self, A, delta, epsilon, n_rounds, true_rewards, rewardsXmovie_indices, actionXmovie_indices):
#         """
#         Initializes the MAB Agent
#         Parameters:
#         A : numpy array representing the actions
#         delta : float, parameter for na function
#         epsilon : float, precision parameter for the Frank-Wolfe algorithm
#         n_rounds : int, number of rounds for exploration
#         true_rewards : dict, mapping actions to their true rewards
#         """
#         self.A = A
#         self.delta = delta
#         self.epsilon = epsilon
#         self.n_rounds = n_rounds
#         self.true_rewards = true_rewards
#         self.d = A.shape[1]
#         self.theta = np.random.rand(self.d)
#         self.cumulative_regret = [0]
#         self.V_l = np.zeros((self.d, self.d))
#         self.sum_reward = np.zeros(self.d)
#         self.max_reward = np.max(true_rewards)
#         self.rewardsXmovie_indices = rewardsXmovie_indices  
#         self.actionXmovie_indices = actionXmovie_indices
#         self.pi = np.ones(A.shape[0]) / A.shape[0]

#     def get_pi(self):
#         return self.pi
    
#     def V(self):
#         """
#         This function will return the V_pi matrix of a design
#         Parameters:
#         pi : Array of probabilities associated to each action
#         """
#         V_pi = np.zeros((self.d, self.d))
#         for a, pi_a in zip(self.A, self.pi):
#             V_pi += pi_a * (a.T @ a)
#         return V_pi

#     def g(self):
#         """
#         Calculates the g function
#         """
#         V_pi = self.V()
#         # inv_V_pi = np.linalg.inv(V_pi+0.01*np.eye(V_pi.shape[0]))
#         inv_V_pi = np.linalg.pinv(V_pi)
#         max_g = -float("inf")
#         a_max = None

#         for a in self.A:
#             g_a = a @ inv_V_pi @ a.T
#             if g_a >= max_g:
#                 max_g = max(max_g, g_a)
#                 a_max = a

#         return max_g, a_max

#     def na(self, epsilon, delta, l, k=2):
#         """
#         Calculate the number of samples required for each action
#         """
#         common_term = (2 * self.d * np.log(k * (l * (l + 1)) / delta)) / epsilon ** 2
#         Na = []

#         for pi_a in self.pi:
#             na = common_term * pi_a
#             na = math.ceil(na)
#             Na.append(na)

#         return Na

#     def frank_wolfe_algo(self, epsilon):
#         """
#         Frank-Wolfe optimization algorithm
#         """
#         # card_A = self.A.shape[0]
#         # pi_k = np.ones(card_A) / card_A
#         g_pi_k = self.g()[0]

#         while g_pi_k > (1 + epsilon) * self.d:
#             g_pi_k, a_k = self.g()
#             gamma_k = ((1 / self.d) * g_pi_k - 1) / (g_pi_k - 1)
#             a_k = np.array(a_k)

#             # Find the index of the row that matches `a_k`
#             row_index = np.flatnonzero(np.all(self.A == a_k, axis=1))

#             # We assume a_k is unique, so we take the first match
#             row_index = row_index[0]

#             # Create the indicator vector
#             indicator_vector = np.zeros(self.A.shape[0], dtype=int)
#             indicator_vector[row_index] = 1
#             self.pi = (1 - gamma_k) * self.pi + gamma_k * indicator_vector
        
#         return self.pi

#     def select_action(self):
#         """
#         Selects an action based on the given probability distribution pi
#         """
#         return np.random.choice(len(self.A), p=self.pi)

#     def update(self, action_indices, true_rewards):
#         """
#         Updates the agent's parameters based on the received reward
#         """
#         for action, reward in zip(action_indices,true_rewards):
#             a = self.A[action]
#             self.V_l += a.T @ a
#             self.sum_reward += a.T * reward
            

#         # self.theta = np.linalg.inv(self.V_l + 0.01*np.eye(self.V_l.shape[0])) @ self.sum_reward
#         self.theta = np.linalg.pinv(self.V_l) @ self.sum_reward
        

#     def reward(self, action_idx):
#         """
#         Returns randomly a reward from a user for this action
#         """
#         movie_id = self.actionXmovie_indices[action_idx]
#         rewards_indices = np.where(self.rewardsXmovie_indices == movie_id)[0]
#         if rewards_indices.size == 0:
#             raise ValueError(f"No rewards found for movie_id {movie_id}")
#         reward_index = np.random.choice(rewards_indices)
#         return self.true_rewards[reward_index]
    

#     def run(self):
#         """
#         Runs the G-optimal design exploration algorithm
#         """
#         l = 1
#         t = 0

#         while t < self.n_rounds:
#             self.pi = self.frank_wolfe_algo(self.epsilon)
#             epsilon_l = 2 ** (-l)
#             N_a = self.na(self.pi, epsilon_l, self.delta, l)
#             N = sum(N_a)
#             rewards, X_t = np.zeros(N), np.zeros(N)
#             tracker = 0
#             action_indices = []
#             true_rewards = []
#             for action_idx, n_a in enumerate(N_a):
#                 for _ in range(n_a):
#                     true_reward = np.max(np.dot(self.A, self.theta))
#                     reward = self.A[action_idx] @ self.theta
#                     self.cumulative_regret.append(self.cumulative_regret[-1] + (true_reward - reward))
#                     action_indices.append(action_idx)
#                     true_rewards.append(true_reward)
#             self.update(action_indices, true_rewards)
#             tracker += 1

#             t += N
#             l += 1
        
#         return self.cumulative_regret, self.theta

                








import numpy as np
import math

class MABAgent:
    def __init__(self, A, delta, epsilon, n_rounds, true_rewards, rewardsXmovie_indices, actionXmovie_indices):
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
        self.rewardsXmovie_indices = rewardsXmovie_indices  
        self.actionXmovie_indices = actionXmovie_indices
        self.pi = np.ones(A.shape[0]) / A.shape[0]

    def get_pi(self):
        return self.pi
    
    def V(self):
        V_pi = np.zeros((self.d, self.d))
        for a, pi_a in zip(self.A, self.pi):
            V_pi += pi_a * np.outer(a, a)
        return V_pi

    def g(self):
        V_pi = self.V()
        inv_V_pi = np.linalg.inv(V_pi + np.eye(V_pi.shape[0]))
        max_g = -float("inf")
        a_max = None
        for a in self.A:
            g_a = a @ inv_V_pi @ a
            if g_a > max_g:
                max_g = g_a
                a_max = a
        return max_g, a_max

    def na(self, l):
        common_term = (2 * self.d * np.log(self.A.shape[0] * (l * (l + 1)) / self.delta)) / self.epsilon ** 2
        Na = [math.ceil(common_term * pi_a) for pi_a in self.pi]
        return Na

    def frank_wolfe_algo(self):
        g_pi_k = self.g()[0]
        while g_pi_k > (1 + self.epsilon) * self.d:
            g_pi_k, a_k = self.g()
            gamma_k = ((1 / self.d) * g_pi_k - 1) / (g_pi_k - 1)
            indicator_vector = np.zeros(self.A.shape[0])
            row_index = np.flatnonzero(np.all(self.A == a_k, axis=1))[0]
            indicator_vector[row_index] = 1
            self.pi = (1 - gamma_k) * self.pi + gamma_k * indicator_vector
        return self.pi

    def select_action(self):
        return np.random.choice(len(self.A), p=self.pi)

    def update(self, action_indices, true_rewards):
        for action, reward in zip(action_indices, true_rewards):
            a = self.A[action]
            self.V_l += np.outer(a, a)
            self.sum_reward += a * reward
        self.theta = np.linalg.inv(self.V_l + np.eye(self.V_l.shape[0])) @ self.sum_reward

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
                    self.cumulative_regret.append(self.cumulative_regret[-1] + (true_reward - reward))
                    action_indices.append(action_idx)
                    true_rewards.append(true_reward)
            self.update(action_indices, true_rewards)
            t += N
            l += 1
        return self.cumulative_regret, self.theta
