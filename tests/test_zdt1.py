import optimizer
import numpy as np
from optimizer.tester import test, test_time, test_random, test_num_evaluations
from matplotlib import pyplot as plt

# for the cool plot in the slides
num_agents = 100
num_iterations = 200

# num_agents = 100
# num_iterations = 100

num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('INFO')

def zdt1_objective1(x):
    return x[0]

def zdt1_objective2(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2

optimizer.FileManager.working_dir = "tmp/policy_easy/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2], sleep_time = 0.01)

# test(objective, num_agents, num_iterations, lb, ub)
model = "./models/model_3_rewards"
test_random(model, objective, num_agents, num_iterations, lb, ub, max_time= 90)
# test_time(objective, num_agents, num_iterations, lb, ub, [0.1, 0.5, 1, 2, 5, 10])

# evaluations = []
# hvs = []
# iterations = [10, 50, 100, 200, 300, 500, 800, 1000, 10000]
# iterations = [1000]
# for i in iterations:
#     print("NUM ITERATIONS " + str(i))
#     plot_index = i
#     evaluations, bad_points_per_iteration, pareto_points_per_iteration, hv = test_num_evaluations(objective, num_agents, i, lb, ub, plot_index)
#     # evaluations.append(eval / (num_agents * i))
#     # hvs.append(hv)

# # fig, axs = plt.subplots(nrows=2, ncols=1)
# # axs[0].plot(iterations, evaluations)
# # axs[1].plot(iterations, hvs)
# # axs[0].set_xlabel("Iterations")
# # axs[1].set_xlabel("Iterations")
# # axs[0].set_ylabel("Evaluations")
# # axs[1].set_ylabel("Hyper Volume")
# x = np.linspace(0, iterations[0], 10*iterations[0])
# y = 5 * x

# ratios = [pareto_points_per_iteration[i] / bad_points_per_iteration[i] if bad_points_per_iteration[i] != 0 else 100 for i in range(len(bad_points_per_iteration))]
# has_evaluated = [evaluations[i] > 0 for i in range(len(evaluations))]
# ratios = [evaluations[i] / num_agents for i in range(len(evaluations))]

# # Creare il grafico
# fig, axs = plt.subplots(nrows=2, ncols=1)
# axs[0].plot(ratios, label = "Ratio")
# axs[1].scatter(list(range(iterations[0])), has_evaluated, label = "Has evaluated", s=5)
# axs[0].set_ylabel("Evaluated agents / agents")
# axs[1].set_xlabel("Iterations")
# axs[1].set_ylabel("Evaluated agents > 0")
# # plt.plot(x, y, label='y = 5x', color='blue')
# # plt.legend()
# plt.savefig("ratios_iterations.png")
# plt.close()
# print(evaluations)
