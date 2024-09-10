# Bandit Algorithms for Movie Recommendation System


## Introduction

### Bandit Algorithms

![Bandit Algorithms](images/bandits_image.png)


Bandit algorithms are a class of decision-making algorithms used in scenarios where an agent must choose between multiple actions to maximize cumulative rewards. The "bandit" analogy comes from the "multi-armed bandit" problem, where each arm of the bandit machine represents an action with unknown reward distributions. The challenge is to balance exploration (trying new actions to learn more about them) and exploitation (choosing actions known to provide high rewards).

Key concepts in bandit algorithms include:

- **Exploration**: Trying out different actions to gather information about their rewards.
- **Exploitation**: Leveraging the knowledge gained to maximize rewards by choosing the best-known action.

### Linear Bandits

Linear bandits are a special case of bandit algorithms where the reward of an action is assumed to be a linear function of its features. This assumption allows for more efficient computation and better theoretical guarantees. The objective of linear bandits is to learn the optimal linear model that predicts rewards based on the action features, and to balance exploration and exploitation within this linear framework.

Key concepts in linear bandits include:

- **Feature Vectors**: Each action is represented by a feature vector, and the reward is modeled as a linear function of these features.
- **Regret Minimization**: The goal is to minimize the cumulative regret, which is the difference between the rewards obtained by the optimal action and the rewards obtained by the chosen actions.


## Project Overview

In this project, I implemented and compared the performance of various Multi-Armed Bandit (MAB) algorithms on a movie recommendation dataset. The goal was to explore how different exploration-exploitation strategies perform in selecting movies based on reward feedback.

### Implemented Bandit Algorithms:

1. **UCB (Upper Confidence Bound)**: 
   - UCB explores actions by balancing the estimated reward and the uncertainty of each action. It calculates an upper confidence bound for each action and selects the action with the highest bound.

2. **Thompson Sampling**:
   - Thompson Sampling selects actions based on sampling from a posterior distribution of the expected reward, balancing exploration and exploitation using a probabilistic approach.

3. **EXP3**:
   - EXP3 is designed for adversarial bandit problems and selects actions based on an exponentially weighted average of the rewards. It maintains a probability distribution over the actions, updating weights using the observed rewards.

4. **LINUCB**:
   - LINUCB is a linear bandit model that assumes the expected reward of an action is a linear function of its features. It uses an upper confidence bound in a linear setting to explore actions.

5. **G-Optimal Design Exploration**:
   - G-Optimal design focuses on reducing the maximum prediction variance over possible actions. It ensures exploration is directed towards actions that provide the most information about the true underlying model, minimizing the worst-case prediction error.
   - In a linear bandit setting, the agent selects the action that maximizes the minimum eigenvalue of the design matrix, ensuring that the exploration is focused on the directions of greatest uncertainty.

### Dataset:
The dataset used for the experiments contains movie ratings and features. The goal of the bandit agents is to select movies that will maximize the cumulative reward (e.g., positive user ratings), given contextual information such as user and movie features.

### Comparison Methodology:

1. **Linear Bandits**:
   - I compared the performance of linear bandit models (LINUCB and G-Optimal design exploration) based on their ability to balance exploration and exploitation in a linear reward setting.

2. **Normal Agents**:
   - I compared the performance of standard multi-armed bandit agents (UCB, Thompson Sampling, and EXP3) on the same dataset, evaluating their exploration-exploitation strategies in the absence of linear assumptions.

### Performance Metrics:

- **Cumulative Regret**: The difference between the reward of the optimal action and the chosen action at each time step. Lower cumulative regret indicates better exploration-exploitation balance.
- **Regret**: The evoltuion of regret over time by each agent.

## Results:

- **Linear Models**: LinUCB significantly outperforms the G-optimal design agent across the tested dimensions. Throughout the evaluation period, the regret for linUCB remains relatively small, while the G-optimal design exhibits a sharp increase in regret from the outset.
- **Normal Agents**: Thompson Sampling showed strong performance in probabilistic exploration, while UCB balanced exploration and exploitation with tighter confidence bounds. EXP3's performance was consistent in adversarial settings but performed slightly worse in this specific context.

## References:

- [Bandit Algorithms](https://tor-lattimore.com/downloads/book/book.pdf) (Pages 102-111 for UCB, Pages 460-475 for Thompson Sampling, Pages 148-160 for EXP3, Pages 238-248 for LINUCB, Pages 267-276 for G-Optimal Design)
  - **UCB Algorithm**: Pages 102-111
  - **Thompson Sampling**: Pages 460-475
  - **EXP3 Algorithm**: Pages 148-160
  - **LINUCB**: Pages 238-248
  - **G-Optimal Design**: Pages 267-276


