import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
plt.style.use('seaborn')
parser = argparse.ArgumentParser(description='Grid search hyperparameters')
parser.add_argument('--n-expert-trajs', type=int, default=2, metavar='G')
args = parser.parse_args()
subfolder = "envLinMDP-v0"
algs = ["fra","ilarl_lin"]
to_plot_x = []
to_plot_y = []
means = []
stds = []
xs = []
colors = {"fra": "green",
# """             "iqlearn":"goldenrod",
#             "gail":"brown",
#             "airl":"gray",
#             "reirl":"darkcyan", """
            "ilarl_lin":"blue",}

alg_name = {"fra": "FRA", "ilarl_lin": "ILARL"}
for alg in algs:
    to_plot_x = []
    to_plot_y = []
    for seed in range(0,10):
        with open(assets_dir(subfolder+f"/{alg}/reward_history/{seed}_{args.n_expert_trajs}.p"), "rb") as f:
            data = data = pickle.load(f)

            
        #data = np.cumsum(data)/np.arange(1,len(data)+1)

        data = data[:15]
            

        to_plot_x.append(np.arange(len(data)))
        to_plot_y.append(data)
        means.append(np.mean(to_plot_y,axis=0))
    stds.append(np.std(to_plot_y, axis=0))
    xs.append(np.mean(to_plot_x,axis=0))


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for m,s,x,alg in zip(means, #[3:], 
                        stds, #[3:], 
                        xs, #[3:], 
                        algs, #[3:]
                        ):
    print(alg)
    m_norm = m
    s_norm = s
    ax.semilogy(x,m_norm,"-o", color=colors[alg], label=alg_name[alg])
    ax.fill_between(x,m_norm-s_norm,
                             m_norm+s_norm,
                             facecolor = colors[alg], 
                             alpha=0.1)
plt.legend(fontsize=20)
#ax.xaxis.set_major_locator(MinNLocator(3)) 
#ax.yaxis.set_major_locator(MaxNLocator(5)) 
#ax.set_yticks([0,1])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim([-0.1, 1.1])
#plt.xlim([-1,200])
plt.xlabel("MDP trajectories", fontsize=30)
#plt.ylabel("Normalized Return", fontsize=30)
plt.tight_layout()
plt.savefig(f"lin{args.n_expert_trajs}.pdf")


        