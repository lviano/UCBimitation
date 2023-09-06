import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
 
labels = ["ILARL (Ours)", "PPIL","IQLearn", "GAIL", "AIRL", "REIRL","BC"]
colors = ["blue", "green",
            "goldenrod",
            "brown",
            "gray",
            "darkcyan", "black"]
 
fig = plt.figure()
fig_legend = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[3])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[4])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[5])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True)
plt.tight_layout()
plt.savefig("figs/legend.pdf")


fig = plt.figure()
fig_legend = plt.figure(figsize=(10, 1))
ax = fig.add_subplot(111)
lines = ax.plot(range(2), range(2), '-o', c=colors[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[1])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[2])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[3])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[4])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[5])[0])
lines.append(ax.plot(range(2), range(2), '-o', c=colors[6])[0])
fig_legend.legend(lines, labels, loc='center', frameon=True, ncol=7)
plt.savefig("figs/legend_horizontal.pdf")

