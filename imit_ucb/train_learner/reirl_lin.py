import argparse
import gym
import my_gym
import os
import sys
import pickle
import time
import copy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_critic import Value
from utils.reirl_tools import AdamOptimizer, Weights
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from train_expert.soft_value_iteration import get_expert
parser = argparse.ArgumentParser(description='PyTorch REIRL example')
parser.add_argument('--env-name', default="GaussianGridworld-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-trajs', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=6144, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--exp-type', type=str, default="mismatch", metavar='N',
                    help="experiment type: noise, friction or mismatch")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
parser.add_argument('--noiseL', type=float, default=0.0, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mulL', type=float, default=1.0, metavar='G',
                    help="Mass Multiplier for learner environment")
parser.add_argument('--len-mulL', type=float, default=1.0, metavar='G',
                    help="Lenght Multiplier for learner environment")
parser.add_argument('--mass-mulE', type=float, default=1.0, metavar='G',
                    help="Mass multiplier for expert environment")
parser.add_argument('--len-mulE', type=float, default=1.0, metavar='G',
                    help="Lenght multiplier for expert environment")
parser.add_argument('--scheduler-lr', action='store_true', default=False,
                    help='Use discriminator lr scheduler')
parser.add_argument('--warm-up', action='store_true', default=False,
                    help='Discriminator Warm UP')
parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='N')

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda',
                      index=args.gpu_index) if torch.cuda.is_available() else torch.device(
    'cpu')
print(device, "device")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
max_grad = 40
env = gym.make(args.env_name, feature_dim=20,  n_states=100, n_actions = 30)
env.seed(args.seed)
subfolder = "env"+str(args.env_name)
if not os.path.isdir(assets_dir(subfolder+f"/reirl_lin/learned_models")):
    os.makedirs(assets_dir(subfolder+f"/reirl_lin/learned_models"))
if not os.path.isdir(assets_dir(subfolder+f"/reirl_lin/reward_history")):
    os.makedirs(assets_dir(subfolder+f"/reirl_lin/reward_history"))

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

reirl_weights = Weights(env.features_reward.shape[2])

optim_epochs = 3  # 10
optim_batch_size = 64


"""create agent"""

policy_net = DiscretePolicy(env.observation_space.n, env.action_space.n)
optimizer_policy = torch.optim.Adam(policy_net.parameters(),
                                    lr=args.learning_rate)

import multiprocessing
from utils.replay_memory import Memory, TwoPlayerMemory
from utils.torch import *
import math
import time
import copy


def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, state_only = False,
                    opponent_policy = None, alpha = None, reward_type=None):
    torch.randn(pid)
    log = dict()
    if opponent_policy is None:
        memory = Memory()
    else:
        memory = TwoPlayerMemory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size:
        state = env.reset()
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0

        for t in range(100000): #range(10000):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    if opponent_policy is not None:
                        opponent_plays = np.random.choice(2, p=[alpha, 1 - alpha])
                        opponent_action = opponent_policy.select_action(state_var)[0].numpy()
                        player_action = policy.select_action(state_var)[0].numpy()
                        if opponent_plays:
                            action = copy.deepcopy(opponent_action)
                        else:
                            action = copy.deepcopy(player_action)

                        player_action = int(player_action) if policy.is_disc_action else player_action.astype(
                            np.float64)
                        opponent_action = int(opponent_action) if policy.is_disc_action else opponent_action.astype(
                            np.float64)
                        """if np.isnan(player_action).any():
                            print("Player Nan")
                            player_action = np.zeros_like(player_action)
                        if np.isnan(opponent_action).any():
                            print("Opponent Nan")
                            opponent_action = np.zeros_like(opponent_action)
                        action = (1 - alpha)*opponent_action.clip(-1.0, 1.0) + alpha*player_action.clip(-1.0, 1.0)"""
                    else:
                        action = policy.select_action(
                            torch.Tensor(to_categorical(state,env.observation_space.n)).unsqueeze(0))

            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            if not policy.is_disc_action:
                action_to_play = action.clip(-1.0, 1.0)
                next_state, reward, done, _ = env.step(action_to_play)
            else:
                next_state, reward, done, _ = env.step(action)
            reward_episode += reward #env.gamma**t*reward
            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:

                if state_only:
                    reward = custom_reward(state, next_state)
                else:
                    reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1
            if opponent_policy is not None:
                memory.push(state, player_action, opponent_action, action, mask, next_state, reward)
            else:
                memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
            if done:
                print("done")
                if opponent_policy is not None:
                    memory.push(next_state, player_action, opponent_action, action,
                                mask, next_state, reward)
                else:
                    memory.push(next_state, action, mask, next_state, reward)
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1, state_only=False,
                 opponent_net = None, alpha= None, reward_type=None):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads
        self.state_only = state_only
        self.opponent_net = opponent_net
        self.alpha = alpha
        self.reward_type = reward_type

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size, self.state_only)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size, self.state_only,
                                      self.opponent_net, self.alpha, self.reward_type)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        if self.opponent_net is None:
            log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
            log['action_min'] = np.min(np.vstack(batch.action), axis=0)
            log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        else:
            log['action_mean'] = np.mean(np.vstack(batch.player_action), axis=0)
            log['action_min'] = np.min(np.vstack(batch.player_action), axis=0)
            log['action_max'] = np.max(np.vstack(batch.player_action), axis=0)
            log['opponent_action_mean'] = np.mean(np.vstack(batch.opponent_action), axis=0)
            log['opponent_action_min'] = np.min(np.vstack(batch.opponent_action), axis=0)
            log['opponent_action_max'] = np.max(np.vstack(batch.opponent_action), axis=0)
        return batch, log
def expert_reward(state, next):
    weights = torch.from_numpy(reirl_weights.read())
    feats = torch.from_numpy(env.features_reward[state,next].T)
    return torch.matmul(weights, feats).detach().numpy()

running_state = lambda x : x
ppo_agent = Agent(env, policy_net, device, custom_reward=expert_reward,
                  running_state=running_state, render=args.render,
                  num_threads=args.num_threads,
                  alpha=1)
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
            states.append(to_categorical(state,env.observation_space.n))
            actions.append(action)
            next_states.append(to_categorical(next_state,env.observation_space.n))
            rewards.append(reward)
            state = next_state 
            h = h + 1
        #print(done)
        if done:
            states.append(to_categorical(state,env.observation_space.n))
            actions.append(np.random.choice(env.action_space.n))
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            next_states.append(to_categorical(next_state,env.observation_space.n))

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

actions = []
states = []
if args.n_expert_trajs > 1:
    for l in expert_actions:
        actions = actions + l
    for l in expert_states:
        states = states + l


one_hot_actions = to_categorical(actions, env.action_space.n)
one_hot_states = np.array(states) #to_categorical(states, env.observation_space.n)

expert_traj = np.concatenate([one_hot_states, one_hot_actions], axis=1)

def reirl(expert_feature_expectations,
          random_feature_expectations,
          opt,
          max_iter=500,
          learning_rate=0.005,
          verbose=False):
    # Compute features expectations

    expert_feature_expectations_mean = np.mean(expert_feature_expectations,
                                               axis=0)

    # importance_sampling = np.zeros(n_random_trajectories)
    # Weights initialization
    w = reirl_weights.read().flatten()
    # Gradient descent
    for i in range(max_iter):
        if verbose:
            print('Iteration %s/%s' % (i + 1, max_iter))
        import pdb; pdb.set_trace()
        #Do not use concatenation of tabular as features but the features 
        #evaluate dat the state action pairs
        to_exp = np.dot(random_feature_expectations, w)
        to_exp -= np.max(to_exp)
        importance_sampling = np.exp(to_exp)
        importance_sampling /= np.sum(importance_sampling, axis=0)
        weighted_sum = np.sum(np.multiply(
            np.array([importance_sampling, ] * random_feature_expectations.shape[1]).T,
                                          random_feature_expectations
                                          ), 
                                axis=0)
        grad = expert_feature_expectations_mean - weighted_sum
        w += opt.update(grad)
        w = w / np.linalg.norm(w, keepdims=True)
    return w

def update_params(batch, i_iter, opt):
    """update discriminator"""
    act = to_categorical(np.stack(batch.action).astype(int), env.action_space.n)
    
    learner_traj = np.concatenate([to_categorical(np.array(batch.state),
    env.observation_space.n), act],axis=1)
    
    reirl_weights.write(
        reirl(expert_traj, learner_traj, opt))
    value_net = Value(env.observation_space.n)
    optimizer_value = torch.optim.Adam(value_net.parameters(),
                                       lr=args.learning_rate)
    if i_iter > 0:
        j_max = 3 #if i_iter < 20 else 15
        for j in range(j_max): #3):
            batch, log = ppo_agent.collect_samples(3000)
            # print(
            #     '{}\tT_sample {}\texpert_R_avg {}\tR_avg {}'.format(
            #         i_iter*j_max + j, log['sample_time'], log['avg_c_reward'],
            #         log['avg_reward']))
            states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(
                device)
            actions = torch.from_numpy(np.stack(batch.action)).to(
                dtype).to(device)
            rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(
                device)
            masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
            with torch.no_grad():
                values = value_net(states)
                fixed_log_probs = policy_net.get_log_prob(states,
                                                          actions)
            """get advantage estimation from the trajectories"""
            advantages, returns = estimate_advantages(rewards, masks, values,
                                                      args.gamma, args.tau,
                                                      device
                                                    )
            """perform mini-batch PPO update"""
            optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
            for _ in range(optim_epochs):
                perm = np.arange(states.shape[0])
                np.random.shuffle(perm)
                perm = LongTensor(perm).to(device)
                states, actions, returns, advantages, fixed_log_probs = \
                    states[perm].clone(), actions[perm].clone(), \
                     returns[perm].clone(), \
                    advantages[perm].clone(), \
                    fixed_log_probs[perm].clone()

                for i in range(optim_iter_num):
                    ind = slice(i * optim_batch_size,
                                min((i + 1) * optim_batch_size,
                                    states.shape[0]))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                        states[ind], actions[ind], \
                        advantages[ind], returns[ind], fixed_log_probs[ind]
                    # Update the player
                    ppo_step(policy_net, value_net, optimizer_policy,
                             optimizer_value, 1, states_b, actions_b,
                             returns_b,
                             advantages_b, fixed_log_probs_b, args.clip_epsilon,
                             args.l2_reg, max_grad=max_grad)
                    
def main_loop():
    rewards = []
    episodes = []
    best_reward = -10000

    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = ppo_agent.collect_samples(args.min_batch_size)
        opt = AdamOptimizer(env.features_reward.shape[2], 5e-3)
        t0 = time.time()
        update_params(batch, i_iter, opt)
        t1 = time.time()
        if i_iter % args.log_interval == 0:
            print(
                '{}\tT_sample {}\tT_update {}\texpert_R_avg {}\tR_avg {}'.format(
                    i_iter, log['sample_time'], t1 - t0, log['avg_c_reward'],
                    log['avg_reward']))
            rewards.append(log['avg_reward'])
            episodes.append(log['num_episodes'])
            to_save = {"rewards": rewards,
                        "episodes": episodes}
            pickle.dump(to_save, open(
                os.path.join(assets_dir(subfolder),
                             'reirl_lin/reward_history/{}_{}.p'.format(str(
                                 args.seed), str(args.n_expert_trajs))), 'wb'))
        if args.save_model_interval > 0 and (
                i_iter + 1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net)
            pickle.dump(policy_net,
                        open(os.path.join(assets_dir(subfolder),
                                          'reirl_lin/learned_models/{}_{}.p'.format(
                                              str(args.seed), 
                                              str(args.n_expert_trajs))), 'wb'))
            if log['avg_reward'] > best_reward:
                print(best_reward)
                pickle.dump(policy_net,
                            open(os.path.join(assets_dir(subfolder),
                                              'reirl_lin/learned_models/{}_{}_best.p'.format(
                                                str(args.seed),
                                                str(args.n_expert_trajs))), 'wb'))
                best_reward = copy.deepcopy(log['avg_reward'])

            to_device(device, policy_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
