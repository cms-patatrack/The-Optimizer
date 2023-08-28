import numpy as np
import matplotlib.pyplot as plt
import optimizer
# the optimize function is wrong, the 1D does not work anymore, there is no test case so idk if the code works, 

def objective_function_1(x):
    return np.sin(x[0]) + np.sin(x[1])
    # return x**2


def objective_function_2(x):
    return np.cos(x[0])+np.cos(x[1])
    # return (x-2)**2


if __name__ == "__main__":
  # define the lower and upper bounds
  lb = [-10,-10] #!!!
  ub = [10,10] #!!!

  # create the PSO object
  pso = optimizer.MOPSO(objective_functions=[objective_function_1, objective_function_2], 
            lower_bound=lb, upper_bound=ub, num_particles=100, num_iterations=50, intertia_weight=0.9, 
            cognitive_coefficient=2, social_coefficient=2, max_iter_no_improv=5)

  # run the optimization algorithm
  pareto_front = pso.optimize()

  # print the results
  print("Global best position:", pso.global_best_position)
  print("Global best fitness:", pso.global_best_fitness)

  # plot the Pareto front
  pareto_x = [particle.fitness[0] for particle in pareto_front]
  print(pareto_x)
  pareto_y = [particle.fitness[1] for particle in pareto_front]
  plt.scatter(pareto_x, pareto_y)
  plt.xlabel("Objective 1")
  plt.ylabel("Objective 2")
  plt.title("Pareto Front")
  plt.savefig("pareto")

  # plot the convergence history
  plt.plot(pso.history)
  plt.xlabel("Iteration")
  plt.ylabel("Global Best Fitness")
  plt.title("Convergence History")
  plt.savefig("convergence")