from optimizer.metrics import hyper_volume
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

pareto_x = (np.linspace(0, 1, 5))
pareto_y = -pareto_x + 1
pareto = np.column_stack((pareto_x, pareto_y))

print("Pareto front: \n", pareto)
ref_point = [1,1]
print("Hyper volume ", hyper_volume(pareto, ref_point))

fig, ax = plt.subplots()
plt.scatter(pareto_x, pareto_y, label = "Pareto front")
plt.scatter(1,1, color = 'r', label = "Reference point")
for x, y in zip(pareto_x, pareto_y):
    width = ref_point[0] - x
    height = ref_point[1] - y
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=None,  hatch='/', facecolor='green', alpha=0.5)
    ax.add_patch(rect)
    ref_point[1] = y

ax.set_xlabel("f1")
ax.set_ylabel("f2")
legend_patch = patches.Patch(facecolor='green', edgecolor=None, hatch='/')
plt.legend()
handles, labels = ax.get_legend_handles_labels()
handles.append(legend_patch)
labels.append('Hyper volume')
ax.legend(handles=handles, labels=labels)
plt.show()