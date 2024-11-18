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

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
env = gym.make(args.env_name, feature_dim=10,  n_states=100, n_actions = 10)
subfolder = "env"+str(args.env_name)+"type"+str(args.grid_type)+"noiseE"+str(args.noiseE)
with open(assets_dir(subfolder+"/expert_trajs/"+args.expert_trajs), "rb") as f:
    data = pickle.load(f)
if not os.path.isdir(assets_dir(subfolder+f"/fra/learned_models")):
    os.makedirs(assets_dir(subfolder+f"/fra/learned_models"))
if not os.path.isdir(assets_dir(subfolder+f"/fra/reward_history")):
    os.makedirs(assets_dir(subfolder+f"/fra/reward_history"))
state_dim = env.observation_space.n
is_disc_action = len(env.action_space.shape) == 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

def compute_features_expectation(states,actions, env):
    features = []
    for traj_states, traj_actions in zip(states[:args.n_expert_trajs], actions[:args.n_expert_trajs]):
        h = 0
        features_exp = 0
        for state,action in zip(traj_states, traj_actions):
            features_exp = features_exp + \
                args.gamma**h * env.features[state,action]
            h = h + 1
        features.append(features_exp)
    return np.mean(features, axis=0)

expert_fev = compute_features_expectation(data["states"][:args.n_expert_trajs],
                                             data["actions"][:args.n_expert_trajs],env)

def collect_trajectories(policy):
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
    print(done)
    if done:
        states.append(state)
        actions.append(np.random.choice(env.action_space.n))
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        next_states.append(next_state)
    return states, actions, rewards, next_states

def compute_covariance(states_dataset, actions_dataset):
    features = []
    covariance = 1/8e-2*np.eye(env.features.shape[2])
    for state,action in zip(states_dataset, actions_dataset):
        features.append(env.feature[state,action])
    for feature in features:
        covariance += np.outer(feature, feature)
    return covariance
    
def compute_bonus(covariance_inv, beta = args.beta):
    norm_feat = np.sqrt(env.features.dot(covariance_inv).dot(env.features.transpose(2,0,1)))
    contraction_factor = special.expit(-beta*norm_feat + np.log(args.max_iter_num))
    bonus = norm_feat*contraction_factor
    return bonus, contraction_factor

def softmax(vec,axis):
    vec = np.exp(vec - np.max(vec,axis=axis,keepdims=True))
    return vec / np.sum(vec,axis=axis,keepdims=True)

def run_imitation_learning(K, eta=1):
    policy_list=[]
    zeta = np.zeros(env.features.shape[2])
    w = np.zeros(env.features_reward.shape[1])
    Q = np.zeros((env.observation_space.n,env.action_space.n))
    V = np.zeros((env.observation_space.n))
    reward_weights = [w]
    covariance_inv = 1/8e-2*np.eye(env.features.shape[2])
    covariance_e = 8e-2*np.eye(env.features.shape[2])
    policy = np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n
    policy_list.append(policy)
    rs=[]
    for k in range(K):
        states_dataset = []
        actions_dataset = []
        next_states_dataset = []
        for i in range(tau):
            states, actions, true_rewards, next_states = collect_trajectories(policy, 
                                                                reward_weights, 
                                                                env, 
                                                                covariance_inv )
            if i == 0:
                states_traj_data = [states]
                actions_traj_data = [actions]
            else:
                states_traj_data.append(states)
                actions_traj_data.append(actions)

            rs.append(np.sum(np.array([1**h for h in range(len(true_rewards))])*true_rewards))
            print("Episode " + str(k) + ": " + str(rs[-1]))
            states_dataset = states_dataset + states
            actions_dataset = actions_dataset + actions
            next_states_dataset = next_states_dataset + next_states
        reward_weights = []
            
        w = w - \
            0.001*(compute_features_expectation(states_traj_data,actions_traj_data,env) - expert_fev)
        reward_weights.append(w)
        covariance = compute_covariance(states_dataset, actions_dataset)
        covariance_inv = np.linalg.inv(covariance)
        if np.linalg.det(covariance) >= 2*np.linalg.det(covariance_e) or k == 0:
            covariance_e = covariance
            bonus_e, contraction_factor = compute_bonus(covariance_e)
            features_e = env.features*contraction_factor
            features_reward_e = env.features_reward*contraction_factor
            policy = np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n
        for state,action,state_prime in zip(states_dataset,actions_dataset,next_states_dataset):
            target_vec += env.features[state,action]*V[state_prime]
        zeta = covariance_inv.dot(target_vec)
        Q = features_reward_e.dot(w) + env.gamma*features_e.dot(zeta) + bonus_e
        V = policy.dot(Q.T)
        policy = softmax(eta*Q + np.log(policy))
        policy_list.append(policy)
        
        # plt.figure(k)
        # plt.scatter(np.stack(states)[:,0], np.stack(states)[:,1], color="blue" )
        # plt.scatter(np.stack(data["states"][0])[:,0], np.stack(data["states"][0])[:,1],color="red")
        # plt.savefig("figs/"+ str(k) + "imit.png")
    with open(assets_dir(subfolder+f"/fra/reward_history/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
        pickle.dump(np.array(rs), f)
    with open(assets_dir(subfolder+f"/fra/learned_models/{args.seed}_{args.n_expert_trajs}.p"), "wb") as f:
        pickle.dump({"policies": policy_list,
                     "covariance": covariance_inv}, 
                    f)
        
run_imitation_learning(args.max_iter_num)
