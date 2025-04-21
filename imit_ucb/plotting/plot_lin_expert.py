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

subfolder = "envLinMDP-v0"
algs = ["fra","ilarl_lin","ppil_lin","iqlearn_lin","optail_lin"]
x = [1,5,10,50,100]
to_plot_y = []
means = []
stds = []
colors = {"ppil_lin": "green",
            "fra":"red",
            "iqlearn_lin":"goldenrod",
             "optail_lin":"brown",
#             "airl":"gray",
#             "reirl":"darkcyan",
            "ilarl_lin":"blue",}

alg_name = {"fra": "FRA", "ilarl_lin": "ILARL","ppil_lin":"PPIL","iqlearn_lin":"IQ-Learn",
"optail_lin":"OPTAIL"}
for alg in algs:
    var_y = []
    to_plot_y = []
    for n in x:
        vec = []
        for seed in range(0,10):
            with open(assets_dir(subfolder+f"/{alg}/reward_history/{seed}_{n}.p"), "rb") as f:
                data = data = pickle.load(f)

                
            data = np.cumsum(data)/np.arange(1,len(data)+1)

            

            
            vec.append(np.min(data))
        to_plot_y.append(np.mean(vec))
        var_y.append(np.std(vec))
    means.append(np.array(to_plot_y))
    stds.append(np.array(var_y))

    


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for m,s,alg in zip(means, #[3:], 
                        stds, #[3:],
                        algs, #[3:]
                        ):
    print(alg)
    m_norm = m
    s_norm = s
    ax.plot(x,m_norm,"-o", color=colors[alg], label=alg_name[alg])
    ax.fill_between(x,m_norm-s_norm,
                             m_norm+s_norm,
                             facecolor = colors[alg], 
                             alpha=0.1)
plt.legend(fontsize=20)
ax.xaxis.set_major_locator(MaxNLocator(4)) 
ax.yaxis.set_major_locator(MaxNLocator(5)) 
#ax.set_yticks([0,1])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-1, 110])
plt.ylim([-0.1,1.2])
plt.xlabel("Expert trajectories", fontsize=30)
#plt.ylabel("Normalized Return", fontsize=30)
plt.tight_layout()
plt.savefig(f"lin_expert_plot.pdf")


        