import numpy as np
import random
from scipy import special
seed = 0
np.random.seed(seed)
random.seed(seed)
d = 100
true_theta = np.random.randn(d)

n_actions = 20

features = np.random.randn(n_actions*true_theta.shape[0]).reshape(n_actions, -1)

expert = special.softmax(features.dot(true_theta))

trajectories = np.random.choice(range(n_actions), p=expert, size=(10))

efev = np.sum([features[t] for t in trajectories], axis = 0)

policy = np.ones(n_actions)/n_actions
learner_trajectories = []
covariance = np.eye(d)
w = np.zeros(d)
for k in range(100):

    learner_trajectories.append(np.random.choice(range(n_actions), p=policy))
    
    covariance += np.outer(features[learner_trajectories[-1]],
                            features[learner_trajectories[-1]])
    bonus = np.array([np.sqrt(np.dot(features[a], np.linalg.solve(covariance, features[a])))
                         for a in range(n_actions)])
    Q = features.dot(w) + bonus
    index = np.where(Q == np.max(Q))[0]
    policy = np.zeros(n_actions)
    policy[index]=1
    
    w 

