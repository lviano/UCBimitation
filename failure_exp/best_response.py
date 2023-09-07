import numpy as np
import random
from scipy import special
import pickle
br = np.zeros((100,20))
d = 100
true_theta = np.zeros(d)
true_theta[:-1:2] = 1
n_actions = 20
np.random.seed(0)
random.seed(0)
features = np.random.randn(n_actions*true_theta.shape[0]).reshape(n_actions, -1)
policy = np.ones(n_actions)/n_actions
expert = special.softmax(features.dot(true_theta))
escore = expert.dot(features).dot(true_theta)
rscore = policy.dot(features).dot(true_theta)
Q=0
for seed in range(100):
    np.random.seed(seed)
    random.seed(seed)
    expert = special.softmax(features.dot(true_theta))
    #expert = np.zeros(n_actions)
    #r = features.dot(true_theta)
    #expert[r == np.max(r)] = 1
    trajectories = np.random.choice(range(n_actions), p=expert, size=(10))

    efev = np.sum([0.999**t*features[t] for t in trajectories], axis = 0)

    policy = np.ones(n_actions)/n_actions
    learner_trajectories = []
    covariance = np.eye(d)
    w = np.zeros(d)
    escore = expert.dot(features).dot(true_theta)
    rscore = policy.dot(features).dot(true_theta)
    policies = np.zeros((100,n_actions))
    for k in range(100):
        learner_trajectories.append(np.random.choice(range(n_actions), p=policy))
        w = w + 0.001*(efev - features[learner_trajectories[-1]])
        if np.linalg.norm(w) > np.linalg.norm(true_theta):
            w = w/np.linalg.norm(w)
        covariance += np.outer(features[learner_trajectories[-1]],
                                features[learner_trajectories[-1]])
        bonus = np.array([np.sqrt(np.dot(features[a], np.linalg.solve(covariance, features[a])))
                            for a in range(n_actions)])
        Q = features.dot(w) + 0.09*bonus + 0.999*np.max(Q)
        Q = np.clip(Q,-1/(1-0.999),1/(1-0.999))
        index = np.where(Q == np.max(Q))[0][0]
        policy = np.zeros(n_actions)
        policy[index]=1
        policies[k,:] = policy
        if not k % 5:
            mean_policy = policies[:(k+1)].mean(axis=0)
            br[seed][np.int(k/5)] = mean_policy.dot(features).dot(true_theta)/(1 - 0.999)
with open("br.p","wb") as f:
    pickle.dump(br,f)
    
    

