import pickle
import numpy as np
import random
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use('seaborn')
with open("br.p","rb") as f:
    br = pickle.load(f)

with open("ilarl.p","rb") as f:
    ilarl = pickle.load(f)

d = 100
true_theta = np.zeros(d)
true_theta[:-1:2] = 1

n_actions = 20
np.random.seed(0)
random.seed(0)
features = np.random.randn(n_actions*true_theta.shape[0]).reshape(n_actions, -1)
policy = np.ones(n_actions)/n_actions
expert = special.softmax(features.dot(true_theta))
escore = expert.dot(features).dot(true_theta)/(1 - 0.999)
rscore = policy.dot(features).dot(true_theta)/(1-0.999)
br = np.hstack([np.zeros((100,1)),br[:,:-1]])
ilarl = np.hstack([np.zeros((100,1)),ilarl[:,:-1]])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

xs = np.arange(0,100,5)
m = (br.mean(axis=0) - rscore) /(escore - rscore)
s = br.std(axis=0)/(escore - rscore)
ax.plot(xs, m,"-o",color="blue", label="BRIL")
ax.fill_between(xs, m - s,
                m + s,
                alpha=0.1,color="blue")
m = (ilarl.mean(axis=0) - rscore) /(escore - rscore)
s = ilarl.std(axis=0)/(escore - rscore)
ax.plot(xs,m, "-o", color="green", label="ILARL")
ax.fill_between(xs, m - s,
                m + s,
                alpha=0.1,color="green")

ax.xaxis.set_major_locator(MaxNLocator(5)) 
ax.yaxis.set_major_locator(MaxNLocator(2)) 
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#plt.ylim([-0.1, 1.6])
plt.xlim([-1,110])
plt.xlabel("MDP trajectories", fontsize=30)
#plt.ylabel("Normalized Return", fontsize=30)
#plt.show()
plt.legend(fontsize=20)
ax.set_yticks([0.0,1.0])
plt.tight_layout()
plt.savefig(f"bandits.pdf")
plt.show()

