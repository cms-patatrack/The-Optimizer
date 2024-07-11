import optimizer
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
import time

def test(objective, num_agents, num_iterations, lower_bounds, upper_bounds, radius = None, plot = True):

    print("Starting MOPSO withot RL")
    optimizer.Randomizer.rng = np.random.default_rng(43)
    pso_no_rl = optimizer.MOPSO(objective=objective, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                        num_particles=num_agents,
                        inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                        rl_model=None, radius=radius)

    start_time_no_rl = time.time()                    
    pso_no_rl.optimize(num_iterations=num_iterations)
    end_time_no_rl = time.time()

    print("Starting MOPSO with RL")
    optimizer.Randomizer.rng = np.random.default_rng(43)
    pso_rl = optimizer.MOPSO(objective=objective, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                        num_particles=num_agents,
                        inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                        rl_model="./models/masked_model", radius=radius)

    start_time_rl = time.time() 
    pso_rl.optimize(num_iterations=num_iterations)
    end_time_rl = time.time() 

    if plot:
        plt.figure()
        pareto_x_no_rl = [particle.fitness[0] for particle in pso_no_rl.pareto_front]
        pareto_y_no_rl = [particle.fitness[1] for particle in pso_no_rl.pareto_front]
        pareto_x_rl = [particle.fitness[0] for particle in pso_rl.pareto_front]
        pareto_y_rl = [particle.fitness[1] for particle in pso_rl.pareto_front]
        plt.scatter(pareto_x_no_rl, pareto_y_no_rl, s=5, c='red', label = "MOPSO")
        plt.scatter(pareto_x_rl, pareto_y_rl, s=5, label = "Reinforcement Learning")
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.legend()
        plt.savefig("pareto_comparison.png")
        plt.close()
    print("Evaluations ", pso_rl.evaluations / (num_agents * num_iterations) * 100, "%")
    ref_point = [2,2]
    ind = HV(ref_point=ref_point)
    print("Pareto points ", ind(np.array([p.fitness for p in pso_rl.pareto_front])) / ind(np.array([p.fitness for p in pso_no_rl.pareto_front])) * 100, "%")
    delta_rl = end_time_rl - start_time_rl
    delta_no_rl = end_time_no_rl - start_time_no_rl
    time_diff = (delta_no_rl - delta_rl) / delta_no_rl
    print("Time saved ", time_diff * 100, "%")
    return time_diff

def test_time(objective, num_agents, num_iterations, lower_bounds, upper_bounds, times):
    time_decreases = []

    for t in times:
        objective.sleep_time = t
        time_diff = test(objective, num_agents, num_iterations, lower_bounds, upper_bounds, plot = False)
        time_decreases.append(time_diff * 100)

    np.save("times.npy", time_decreases)
    plt.figure()
    plt.plot(times, time_decreases)
    plt.xlabel("Sleep time [s]")
    plt.ylabel("Time decrease [%]")
    plt.savefig("time_decrease.png")
    plt.close()

def test_random(model, objective, num_agents, num_iterations, lower_bounds, upper_bounds, radius=None, plot = True, max_time = np.inf):
    seed = 40
    # OPTIMAL MOPSO
    print("Starting Optimal MOPSO ")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_optimal = optimizer.MOPSO(objective=objective, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                        num_particles=num_agents * 2,
                        inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True)
    start_time_optimal = time.time()                    
    pso_optimal.optimize(num_iterations=num_iterations * 2, max_time = max_time * 2)
    end_time_optimal = time.time()
    
    # MOPSO
    print("Starting MOPSO ")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_no_rl = optimizer.MOPSO(objective=objective, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                        num_particles=num_agents,
                        inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True)
    start_time_no_rl = time.time()                    
    pso_no_rl.optimize(num_iterations=num_iterations, max_time = max_time)
    end_time_no_rl = time.time()
    
    # MOPSO WITH RANDOM POLICY
    print("Starting MOPSO with random policy ")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_rl_random = optimizer.MOPSO(objective=objective, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                        num_particles=num_agents,
                        inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                        rl_model="Random", radius=radius)

    start_time_rl_random = time.time()                    
    pso_rl_random.optimize(num_iterations=num_iterations, max_time = max_time)
    end_time_rl_random = time.time()

    # MOPSO WITH TRAINED POLICY
    print("Starting MOPSO with trained policy")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_rl = optimizer.MOPSO(objective=objective, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                        num_particles=num_agents,
                        inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                        rl_model=model, radius=radius)

    start_time_rl = time.time() 
    pso_rl.optimize(num_iterations=num_iterations, max_time = max_time)
    end_time_rl = time.time()

    if plot:
        alpha = 0.7
        pareto_x_optimal = [particle.fitness[0] for particle in pso_optimal.pareto_front]
        pareto_y_optimal = [particle.fitness[1] for particle in pso_optimal.pareto_front]
        pareto_x_no_rl = [particle.fitness[0] for particle in pso_no_rl.pareto_front]
        pareto_y_no_rl = [particle.fitness[1] for particle in pso_no_rl.pareto_front]
        pareto_x_rl = [particle.fitness[0] for particle in pso_rl.pareto_front]
        pareto_y_rl = [particle.fitness[1] for particle in pso_rl.pareto_front]
        pareto_x_rl_random = [particle.fitness[0] for particle in pso_rl_random.pareto_front]
        pareto_y_rl_random = [particle.fitness[1] for particle in pso_rl_random.pareto_front]
        plt.figure()
        plt.scatter(pareto_x_optimal, pareto_y_optimal, s=5, c='red', label = "Optimal MOPSO", alpha = alpha)
        plt.scatter(pareto_x_no_rl, pareto_y_no_rl, s=5, c='blue', label = "MOPSO", alpha = alpha)
        plt.scatter(pareto_x_rl, pareto_y_rl, s=5, c='green', label = "Trained policy", alpha = alpha)
        plt.scatter(pareto_x_rl_random, pareto_y_rl_random, s=5, c='orange', label = "Random policy", alpha = alpha)
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.legend()
        plt.savefig("pareto_comparison_policy_max_time" + str(max_time) + "_seed_" + str(seed) + ".png")
        plt.close()
    print("Evaluations Trained / MOPSO", np.sum(pso_rl.evaluations) / (num_agents * num_iterations) * 100, "%")
    print("Evaluations Random / MOPSO", np.sum(pso_rl_random.evaluations) / (num_agents * num_iterations) * 100, "%")
    print("Evaluations Trained / Random", np.sum(pso_rl.evaluations) / np.sum(pso_rl_random.evaluations) * 100, "%")
    ref_point = [2,2]
    ind = HV(ref_point=ref_point)
    print("HV Trained / MOPSO", ind(np.array([p.fitness for p in pso_rl.pareto_front])) / ind(np.array([p.fitness for p in pso_no_rl.pareto_front])) * 100, "%")
    print("HV Random / MOPSO", ind(np.array([p.fitness for p in pso_rl_random.pareto_front])) / ind(np.array([p.fitness for p in pso_no_rl.pareto_front])) * 100, "%")
    print("HV Trained / Random", ind(np.array([p.fitness for p in pso_rl.pareto_front])) / ind(np.array([p.fitness for p in pso_rl_random.pareto_front])) * 100, "%")
    
    delta_rl = end_time_rl - start_time_rl
    delta_no_rl = end_time_no_rl - start_time_no_rl
    time_diff = (delta_no_rl - delta_rl) / delta_no_rl
    # print("Time decrease ", time_diff * 100, "%")
    return time_diff

def test_num_evaluations(objective, num_agents, num_iterations, lower_bounds, upper_bounds, plot_index, radius = None,):
    print("Starting MOPSO with RL")
    optimizer.Randomizer.rng = np.random.default_rng(43)
    pso_rl = optimizer.MOPSO(objective=objective, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                        num_particles=num_agents,
                        inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                        rl_model="./models/masked_model", radius=radius)
    pso_rl.optimize(num_iterations=num_iterations)
    evaluations = pso_rl.evaluations
    plt.figure()
    plt.hist(evaluations, bins = list(range(num_agents)))
    plt.xticks(list(range(0, num_agents + 1, 5)), ha='center')
    plt.xlabel("Evaluations per iteration")
    plt.ylabel("Counts")
    plt.savefig("./plots/evaluation_histogram_" + str(plot_index) + ".png")
    plt.close()
    plt.figure()
    pareto_x_rl = [particle.fitness[0] for particle in pso_rl.pareto_front]
    pareto_y_rl = [particle.fitness[1] for particle in pso_rl.pareto_front]
    plt.scatter(pareto_x_rl, pareto_y_rl, s=5, c='red', label = "Trained policy")
    plt.savefig("./plots/evaluation_pareto_" + str(plot_index) + ".png")
    plt.close()
    print("Tot evaluations: ", np.sum(evaluations) / (num_agents * num_iterations) * 100)
    ref_point = [5, 5]
    ind = HV(ref_point=ref_point)
    hv = ind(np.array([p.fitness for p in pso_rl.pareto_front]))
    return pso_rl.evaluations,pso_rl.bad_points_per_iteration, pso_rl.pareto_points_per_iteration, hv


    