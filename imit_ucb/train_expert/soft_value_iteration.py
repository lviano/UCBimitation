import argparse
import gym
import gym_simple
import my_gym
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *

env = gym.make("LinMDP-v0", feature_dim=10,  n_states=100, n_actions = 10)
subfolder = "env"+str("LinMDP-v0")
if not os.path.isdir(assets_dir(subfolder)):
    os.makedirs(assets_dir(subfolder))
    os.makedirs(assets_dir(subfolder+"/learned_models"))
def softmax(vec,axis):
    vec = np.exp(vec - np.max(vec,axis=axis,keepdims=True))
    return vec / np.sum(vec,axis=axis,keepdims=True)
def logsumexp(vec,axis):
    return 1/1e5*np.log(np.sum(np.exp(1e5*vec - np.max(1e5*vec,axis=axis,keepdims=True)),axis=axis))
    

def compute_optimal_v(env):
    V = np.zeros(env.n_states)
    for k in range(50):
        Q = env.reward + env.gamma*env.transition@V
        V = logsumexp(Q,axis=1)
    return V

def get_expert(env):
    V = compute_optimal_v(env)
    Q = env.reward + env.gamma*env.transition@V
    print(softmax(1e5*Q,axis=1)[softmax(1e5*Q,axis=1) < 0],"expert_policy")
    return softmax(1e5*Q,axis=1)


