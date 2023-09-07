import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import *

plt.style.use('seaborn')
parser = argparse.ArgumentParser(description='Grid search hyperparameters')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='G')
args = parser.parse_args()
subfolder = "envDiscreteGaussianGridworld-v0type1noiseE"+str(args.noiseE)
algs = ["ppil","iqlearn", "gail", "airl", "reirl","ilarl"]
to_plot_x = []
to_plot_y = []
means = []
stds = []
xs = []

if args.noiseE == 0.0:
    max_expert = -1253.113259858868
    random_p = -356525.46
    bc = -1253.113259858868
    bc_std = 0
elif args.noiseE == 0.05:
    max_expert = -4741.92237394391
    random_p = -356525.46
    bc = -1165.5682994141093
    bc_std = 18.57952907393339
elif args.noiseE == 0.1:
    max_expert = -2500
    random_p = -356525.46
    if args.n_expert_trajs == 1:
        bc = -27635.4115080747433
        bc_std = 30000.7501572488752
    elif args.n_expert_trajs == 2:
        bc = -5427.4115080747433
        bc_std = 4438.7501572488752
    elif args.n_expert_trajs == 3:
        bc = -2054.4819211391077
        bc_std =  829.6770278814829
    elif args.n_expert_trajs == 5:
        bc = -1522.5870483696021
        bc_std =  120
    
colors = {"ppil": "green",
            "iqlearn":"goldenrod",
            "gail":"brown",
            "airl":"gray",
            "reirl":"darkcyan",
            "ilarl":"blue",}
for alg in algs:
    to_plot_x = []
    to_plot_y = []
    for seed in range(0,5):
        with open(assets_dir(subfolder+f"/{alg}/reward_history/{seed}_{args.n_expert_trajs}.p"), "rb") as f:
            data = pickle.load(f)
        if alg in ["ppil","ilarl","iqlearn"]:
            tau = 5 if alg in ["ppil","ilarl","iqlearn"] else 1
            to_plot_y.append(np.array([np.mean(data[tau*i:tau*i+tau])
                for i in range(np.int(len(data)/tau))]))
            to_plot_x.append(np.array([tau*i 
                for i in range(np.int(len(data)/tau))]))
        else:
            to_plot_x.append(np.cumsum(data["episodes"]))
            to_plot_y.append(data["rewards"])
    print(alg)
    means.append(np.mean(to_plot_y,axis=0))
    if alg in ["iqlearn","ilarl"]:
        means[-1][0] = random_p
    stds.append(np.std(to_plot_y, axis=0))
    xs.append(np.mean(to_plot_x,axis=0))
#import pdb; pdb.set_trace()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
m_norm = (bc - random_p)/(max_expert - random_p)
s_norm = bc_std/(max_expert - random_p)
ax.plot(xs[0],m_norm*np.ones_like(xs[0]),"-", color="black", label="BC")
ax.fill_between(xs[0],(m_norm-s_norm)*np.ones_like(xs[0]),
                             (m_norm+s_norm)*np.ones_like(xs[0]),
                             facecolor = "black", 
                             alpha=0.1)
for m,s,x,alg in zip(means, #[3:], 
                        stds, #[3:], 
                        xs, #[3:], 
                        algs, #[3:]
                        ):
    print(alg)
    m_norm = (m - random_p)/(max_expert - random_p)
    s_norm = s/(max_expert - random_p)
    ax.plot(x,m_norm,"-o", color=colors[alg], label=alg)
    ax.fill_between(x,m_norm-s_norm,
                             m_norm+s_norm,
                             facecolor = colors[alg], 
                             alpha=0.1)
#plt.legend(fontsize=20)
ax.xaxis.set_major_locator(MaxNLocator(5)) 
ax.yaxis.set_major_locator(MaxNLocator(2)) 
ax.set_yticks([0,1])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim([-0.1, 1.1])
plt.xlim([-1,200])
plt.xlabel("MDP trajectories", fontsize=30)
#plt.ylabel("Normalized Return", fontsize=30)
plt.tight_layout()
plt.savefig(f"noisy_noiseE{args.noiseE}_n_traj{args.n_expert_trajs}.pdf")

for m,s,x,alg in zip(means, #[3:], 
                        stds, #[3:], 
                        xs, #[3:], 
                        algs, #[3:]
                        ):
    print(alg)
    m_norm = (m - random_p)/(max_expert - random_p)
    s_norm = s/(max_expert - random_p)
    ax.plot(x,m_norm,"-o", color=colors[alg], label=alg)
    ax.fill_between(x,m_norm-s_norm,
                             m_norm+s_norm,
                             facecolor = colors[alg], 
                             alpha=0.1)
#plt.legend(fontsize=20)
ax.xaxis.set_major_locator(MaxNLocator(5)) 
ax.yaxis.set_major_locator(MaxNLocator(2))
plt.xticks(fontsize=30)
plt.yticks(fontsize=20)
plt.xlim([-1,200])
plt.xlabel("MDP trajectories", fontsize=30)
#plt.ylabel("Normalized Return", fontsize=30)
#plt.show()
if args.n_expert_trajs == 1:
    ax.set_yticks([0.9,1.0])
    plt.ylim([0.85, 1.01])
else:
    ax.set_yticks([0.97,1.0])
    plt.ylim([0.95, 1.01])
plt.tight_layout()
plt.savefig(f"noisy_noiseE{args.noiseE}_zoom.pdf")
plt.show()


        
