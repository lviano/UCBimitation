import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import *

plt.style.use('seaborn')
parser = argparse.ArgumentParser(description='Grid search hyperparameters')
parser.add_argument('--noiseE', type=float, default=0.0, metavar='G')
args = parser.parse_args()
subfolder = "envDiscreteGaussianGridworld-v0type1noiseE"+str(args.noiseE)
algs = ["ppil","iqlearn", "gail", "airl", "reirl","ilarl"]
to_plot_x = []
to_plot_y = []
means = []
stds = []
xs = []


max_expert = -2500
random_p = -356525.46
bc = np.array([-27635.4115080747433,
                -5427.4115080747433,
                -2054.4819211391077,
                -1522.5870483696021])
bc_std = np.array([30000.7501572488752,
            4438.7501572488752,
             829.6770278814829,
             120])
colors = {"ppil": "green",
            "iqlearn":"goldenrod",
            "gail":"brown",
            "airl":"gray",
            "reirl":"darkcyan",
            "ilarl":"blue",}
for alg in algs:
    to_plot_x = []
    to_plot_y = np.zeros((5,4))
    for seed in range(0,5):
        for jj, n_expert_trajs in enumerate([1,2,3,5]):
            with open(assets_dir(subfolder+f"/{alg}/reward_history/{seed}_{n_expert_trajs}.p"), "rb") as f:
                data = pickle.load(f)
            if alg in ["ppil","ilarl","iqlearn"]:
                tau = 5 
                to_plot_y[seed][jj] = np.max([np.mean(data[tau*i:tau*i+tau])
                    for i in range(np.int(len(data)/tau))])
            else:
                to_plot_y[seed][jj] = np.max(data["rewards"])
    
    means.append(np.mean(to_plot_y,axis=0))
    stds.append(np.std(to_plot_y, axis=0))
    xs.append(np.array([1,2,3,5]))
#import pdb; pdb.set_trace()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
m_norm = (bc - random_p)/(max_expert - random_p)
s_norm = bc_std/(max_expert - random_p)
ax.plot(xs[0],m_norm,"-", color="black", label="BC")
ax.fill_between(xs[0],(m_norm-s_norm),
                             (m_norm+s_norm),
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
plt.ylim([0.85, 1.1])
plt.xlim([-1,6])
plt.xlabel("Expert trajectories", fontsize=30)
#plt.ylabel("Normalized Return", fontsize=30)
plt.tight_layout()
plt.savefig(f"expert_noisy_noiseE{args.noiseE}_n_traj{n_expert_trajs}.pdf")
