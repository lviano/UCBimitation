import numpy as np
import random
from scipy import special
import pickle
ilarl = np.zeros((100,20))
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
for seed in range(100):
    np.random.seed(seed)
    random.seed(seed)
    #expert = np.zeros(n_actions)
    #r = features.dot(true_theta)
    #expert[r == np.max(r)] = 1
    trajectories = np.random.choice(range(n_actions), p=expert, size=(10))

    efev = np.sum([0.999**t*features[t] for t in trajectories], axis = 0)
    
    w = np.zeros(d)
    Qs = []
    policies = np.zeros((20,n_actions))
    Q_batch = 0
    for j in range(20):
        learner_trajectories = []
        covariance = np.eye(d)
        for k in range(20):
            learner_trajectories.append(np.random.choice(range(n_actions), p=policy))
            covariance += np.outer(features[learner_trajectories[-1]],
                                    features[learner_trajectories[-1]])
        bonus = np.array([np.sqrt(np.dot(features[a], np.linalg.solve(covariance, features[a])))
                            for a in range(n_actions)])
        lfev = np.sum([0.999**t*features[t] for t in learner_trajectories], axis = 0)
        
        for k in range(20):
            w = w + 0.001*(efev - lfev)
            if np.linalg.norm(w) > np.linalg.norm(true_theta):
                w = w/np.linalg.norm(w)
            Q_batch += features.dot(w) + 0.09*bonus + 0.999*policy.dot(Q_batch)
            Q_batch =  np.clip(Q_batch,-1/(1-0.999),1/(1-0.999))
        Qs.append(Q_batch/5)

        policy = special.softmax(np.sum(np.array(Qs),axis=0))
        policies[j,:] = policy
        mean_policy = policies[:(j+1)].mean(axis=0)
        ilarl[seed][j] = policy.dot(features).dot(true_theta)/(1 - 0.999)


with open("ilarl.p","wb") as f:
    pickle.dump(ilarl,f)


    