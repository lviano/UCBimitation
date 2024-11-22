import argparse
import gym
import my_gym
from scipy import special
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
from copy import deepcopy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from train_expert.soft_value_iteration import get_expert
from itertools import product
parser = argparse.ArgumentParser(description='UCB')
parser.add_argument('--env-name', default="LinMDP-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-trajs', metavar='G',
                    help='path to expert data')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--beta', type=float, default=100.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--eta', type=float, default=1.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot masses")
parser.add_argument('--len-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot lengths")
parser.add_argument('--friction', default=False, action='store_true')
parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='N')
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
env = gym.make(args.env_name, feature_dim=20,  n_states=100, n_actions = 30)
env.seed(args.seed)
subfolder = "env"+str(args.env_name)
if not os.path.isdir(assets_dir(subfolder+f"/optail_lin/learned_models")):
    os.makedirs(assets_dir(subfolder+f"/optail_lin/learned_models"))
if not os.path.isdir(assets_dir(subfolder+f"/optail_lin/reward_history")):
    os.makedirs(assets_dir(subfolder+f"/optail_lin/reward_history"))


def evaluate_policy(env,policy):
    V = np.zeros(env.n_states)
    for k in range(50):
        Q = env.reward + env.gamma*env.transition@V
        V = np.diag(policy.dot(Q.T))
    return np.mean(V)

def collect_trajectories(policy,n=1):
    states_list = []
    action_list = []
    next_states_list = []
    for _ in range(n):
        state = env.reset()
        h = 0
        states = []
        next_states = []
        actions = []
        rewards = []
        done = False
        while not done:
            action = np.random.choice(env.action_space.n, p=policy[state])
            
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            state = next_state 
            h = h + 1
        #print(done)
        if done:
            states.append(state)
            actions.append(np.random.choice(env.action_space.n))
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            next_states.append(next_state)
        states_list.append(states)
        action_list.append(actions)
        next_states_list.append(next_states)
    if n==1:
        return states, actions, rewards, next_states
    return states_list, action_list, rewards, next_states_list

expert_policy = get_expert(env)

expert_states, expert_actions, expert_rewards, _ = collect_trajectories(expert_policy,n=args.n_expert_trajs)
expert_value = evaluate_policy(env,expert_policy)

print(expert_value)


def compute_features_expectation(states,actions, env):
    features = []
    for traj_states, traj_actions in zip(states[:args.n_expert_trajs], actions[:args.n_expert_trajs]):
        h = 0
        features_exp = 0
        for state,action in zip(traj_states, traj_actions):
            features_exp = features_exp + \
                args.gamma**h * env.features_reward[state,action]
            h = h + 1
        features.append(features_exp)
    return np.mean(features, axis=0)
if args.n_expert_trajs == 1:
    expert_fev = compute_features_expectation([expert_states], [expert_actions],env)
else:
    expert_fev = compute_features_expectation(expert_states, expert_actions,env)


def softmax(vec,axis):
    vec = np.exp(vec - np.max(vec,axis=axis,keepdims=True))
    return vec / np.sum(vec,axis=axis,keepdims=True)

def run_imitation_learning(K, eta=1e-1, tau=5):
    policy_list=[]
    theta = np.zeros(env.features.shape[2])
    w = np.zeros(env.features_reward.shape[2])
    Q = np.zeros((env.observation_space.n,env.action_space.n))
    V = np.zeros((env.observation_space.n))
    reward_weights = [w]
    policy = np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n
    policy_list.append(policy)
    rs = [(expert_value - evaluate_policy(env,policy))/expert_value]
    for _ in range(tau-1):
        rs.append((expert_value - evaluate_policy(env,policy))/expert_value)
    for k in range(K):
        Qs = []
        states_traj_data, actions_traj_data, _, next_states_traj_data = collect_trajectories(policy,n=tau)

        # for states,actions,next_states in zip(states_traj_data, actions_traj_data,next_states_traj_data):
        #     states_dataset = states_dataset + states
        #     actions_dataset = actions_dataset + actions
        #     next_states_dataset = next_states_dataset + next_states
        reward_weights = []
        w = w - (compute_features_expectation(states_traj_data,actions_traj_data,env) - expert_fev)
        w = np.clip(w, -10 , 10)
        reward_weights.append(w)
        
        #for _ in range(10):
        grad=0
        for states,actions,next_states in zip(states_traj_data, actions_traj_data,next_states_traj_data):
            grad-=deepcopy(env.features[states[1],actions[1]])
            for state,action,state_prime in zip(states[1:],actions[1:],next_states[1:]):
                grad += deepcopy(env.features[state,action])*( 
                    Q[state,action] - deepcopy(env.features_reward[state,action]).dot(w) - env.gamma*V[state_prime])/len(states)
        theta = theta - 0.001*grad
        Q = env.features.dot(theta)
        r_max = np.max(deepcopy(env.features_reward).dot(w))
        r_min = np.min(deepcopy(env.features_reward).dot(w))
        Q = np.clip(Q, r_min/(1-env.gamma),r_max/(1-env.gamma))
        V = np.diag(policy.dot(Q.T))
        Qs.append(Q)
        
        policy = softmax(eta*np.mean(Qs,axis=0) + np.log(policy),axis=1)
        
        for _ in range(tau):
            policy_list.append(policy)
            rs.append((expert_value - evaluate_policy(env,policy))/expert_value)
            print("Episode Last" + str(k) + ": " + str(rs[-1]))
        
    with open(assets_dir(subfolder+f"/optail_lin/reward_history/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
        pickle.dump(np.array(rs), f)
    with open(assets_dir(subfolder+f"/optail_lin/learned_models/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
        pickle.dump({"policies": policy_list}, 
                    f)
        
run_imitation_learning(args.max_iter_num)
