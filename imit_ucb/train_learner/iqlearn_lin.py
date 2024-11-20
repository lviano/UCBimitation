import argparse
import gym
import my_gym
from scipy import special
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from itertools import product
from train_expert.soft_value_iteration import get_expert

parser = argparse.ArgumentParser(description='UCB')
parser.add_argument('--env-name', default="LinMDP-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-trajs', metavar='G',
                    help='path to expert data')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
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
if not os.path.isdir(assets_dir(subfolder+f"/iqlearn_lin/learned_models")):
    os.makedirs(assets_dir(subfolder+f"/iqlearn_lin/learned_models"))
if not os.path.isdir(assets_dir(subfolder+f"/iqlearn_lin/reward_history")):
    os.makedirs(assets_dir(subfolder+f"/iqlearn_lin/reward_history"))
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
    next_action_list = []
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

            last_action = np.random.choice(env.action_space.n, p=policy[state])
            next_actions = actions[1:] 
            next_actions.append(last_action)
        states_list.append(states)
        action_list.append(actions)
        next_states_list.append(next_states)
        next_action_list.append(next_actions)
    if n==1:
        return states, actions, rewards, next_states, next_actions
    return states_list, action_list, rewards, next_states_list, next_action_list


expert_policy = get_expert(env)

expert_states, expert_actions, expert_rewards, _, _ = collect_trajectories(expert_policy,n=args.n_expert_trajs)
expert_value = evaluate_policy(env,expert_policy)

if args.n_expert_trajs == 1:
    data_expert_states = [expert_states]
    data_expert_actions = [expert_actions]
else:
    data_expert_states = expert_states
    data_expert_actions = expert_actions



def run_iqlearn(K, tau=5):
    theta = np.zeros(env.features.shape[2])
    """create agent"""
    policy_list = []
    policy = np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n
    rs = []
    for k in range(K):
        states_dataset = []
        actions_dataset = []
        next_states_dataset = []
        next_actions_dataset = []
        for i in range(tau):
            states, actions, true_rewards, next_states, next_actions = collect_trajectories(policy, n=1 
                                                                 )
            if i == 0:
                states_traj_data = [states]
                actions_traj_data = [actions]
            else:
                states_traj_data.append(states)
                actions_traj_data.append(actions)

            policy_list.append(policy)
            rs.append((expert_value - evaluate_policy(env,policy))/expert_value)
            print("Episode Last" + str(k) + ": " + str(rs[-1]))
            states_dataset = states_dataset + states
            actions_dataset = actions_dataset + actions
            next_states_dataset = next_states_dataset + next_states
            next_actions_dataset = next_actions_dataset + next_actions
        ### Approxiately solve logistic Bellman error minimization
        for _ in range(20): # before was 100
            gradient=0
            n = 0
            Q = env.features.dot(theta)
            V = special.logsumexp(Q, axis=1)

            policy = special.softmax(Q, axis=1)
            policy_list.append(policy)
            for traj_state, traj_actions in zip(data_expert_states,
                                                 data_expert_actions):
                for state, action, next_state in zip(traj_state[:-1],
                            traj_actions[:-1],
                            traj_state[1:]):
                    n = n + 1
                    feature_s_a = env.features[state, action]
                    features_next_state = env.features[next_state]
                    
                    
                    gradient += (feature_s_a - args.gamma*features_next_state.T.dot(policy[next_state]))\
                        *(0.5*Q[state,action] - 0.5*args.gamma*V[next_state])
            gradient = gradient/n
            
            gradient_2 = 0
            n = len(states_dataset)
            for b in zip(states_dataset,
                        actions_dataset,
                        next_states_dataset):
                state, action, next_state = b
                features_state = env.features[state]
                
                features_next_state = env.features[next_state]
                
                # Q_next_state = features_next_state.dot(theta)
                # Q_state = features_state.dot(theta)
                
                # probs_next_state = special.softmax(Q_next_state, axis=0)
                # probs_state = special.softmax(Q_state, axis=0)

                # value_state = special.logsumexp(Q_state)
                # value_next_state = special.logsumexp(Q_next_state)
                gradient_2 += (features_state.T.dot(policy[state]) 
                - args.gamma*features_next_state.T.dot(policy[next_state]))*(V[state] - args.gamma*V[next_state])
            gradient = gradient - gradient_2/n
        
            theta = theta - 0.005*gradient
        
        # plt.figure(k)
        # plt.scatter(np.stack(states)[:,0], np.stack(states)[:,1], color="blue" )
        # plt.scatter(np.stack(data["states"][0])[:,0], np.stack(data["states"][0])[:,1],color="red")
        # plt.savefig("figs/"+ str(k) + "iqlearn.png")

    with open(assets_dir(subfolder+f"/iqlearn_lin/reward_history/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
        pickle.dump(np.array(rs), f)
    with open(assets_dir(subfolder+f"/iqlearn_lin/learned_models/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
        pickle.dump(policy_list, f)
        
run_iqlearn(args.max_iter_num)
