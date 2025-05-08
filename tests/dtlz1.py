# implementation of dtlz1 problem
import numpy as np
import matplotlib.pyplot as plt
import os
import optimizer

num_agents = 200
num_iterations = 100
num_params = 30
num_obj = 3

k = num_params - num_obj + 1
lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('DEBUG')

def dtlz1(x):
    x = np.asarray(x, dtype=np.float64)
    D = len(x)
    g = 100 * (D - num_obj + 1 + np.sum((x[num_obj-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[num_obj-1:] - 0.5))))

    f = []
    for m in range(num_obj):
        prod = np.prod(x[:num_obj - m - 1]) if m < num_obj - 1 else 1.0
        if m > 0:
            prod *= (1 - x[num_obj - m - 1])
        f.append(0.5 * (1 + g) * prod)

    return np.array(f)

def true_pareto_dltz1(x):
    f_vals = []

    for f1 in x:
        for f2 in x:
            f3 = 0.5 - f1 - f2
            if f3 >= 0:
                f_vals.append([f1, f2, f3])

    return np.array(f_vals).T

objective = optimizer.ElementWiseObjective(dtlz1, num_obj, true_pareto=true_pareto_dltz1)
optimizer.FileManager.working_dir = "tmp/dtlz1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.5, cognitive_coefficient=2, social_coefficient=0.5)

# run the optimization algorithm
pso.optimize(num_iterations)

fig, ax = pso.scatter(fig=None)

linspace = np.linspace(0, 1, 100)
ref_x, ref_y, ref_z = true_pareto_dltz1(linspace)
ax.scatter(ref_x, ref_y, ref_z, color='red', label='True Pareto Front', alpha=0.5)
plt.show()
plt.savefig(optimizer.FileManager.working_dir + 'sc.png')
