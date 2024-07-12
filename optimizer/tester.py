import optimizer
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
import time

def test(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, plot_paretos = True, print_results = True, known_pareto=None, verbose = 0):
    results = {}
    for seed in seeds:
        result = test_seed(objective, mopso_parameters, num_iterations, rl_model, ref_point, seed, verbose = verbose)
        results[f'{seed}'] = result

    if print_results: print_results(results)
    if plot_paretos: plot_paretos(results, known_pareto)    
    return results

def test_seed(objective, mopso_parameters, num_iterations, rl_model, ref_point, seed, verbose = 0):

    if verbose > 0 : print(f"SEED {seed}")

    #Optimizers
    optimizers = []

    if verbose > 1 : print("Starting MOPSO withot RL")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso = optimizer.MOPSO(objective=objective, lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'],
                                num_particles=mopso_parameters['num_particles'],
                                inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                                rl_model=None, radius=mopso_parameters['radius'])

    start_time_pso = time.time()                    
    pso.optimize(num_iterations=num_iterations)
    end_time_pso = time.time()
    optimizers.append(pso)

    if verbose > 1 : print("Starting MOPSO with trained policy")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_trained_policy = optimizer.MOPSO(objective=objective, lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'],
                                num_particles=mopso_parameters['num_particles'],
                                inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                                rl_model=rl_model, radius=mopso_parameters['radius'])

    start_time_pso_trained_policy = time.time() 
    pso_trained_policy.optimize(num_iterations=num_iterations)
    end_time_pso_trained_policy = time.time() 
    optimizers.append(pso_trained_policy)

    if verbose > 1 : print("Starting MOPSO with random policy")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_random_policy = optimizer.MOPSO(objective=objective, lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'],
                                num_particles=mopso_parameters['num_particles'],
                                inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                                rl_model='random', radius=mopso_parameters['radius'])

    start_time_pso_random_policy = time.time() 
    pso_random_policy.optimize(num_iterations=num_iterations)
    end_time_pso_random_policy = time.time()
    optimizers.append(pso_random_policy)

    # Results
    evaluations_pso = pso.evaluations
    evaluations_pso_trained_policy = pso_trained_policy.evaluations
    evaluations_pso_random_policy = pso_random_policy.evaluations

    pareto_pso = [p.fitness for p in pso.pareto_front]
    pareto_pso_trained_policy = [p.fitness for p in pso_trained_policy.pareto_front]
    pareto_pso_random_policy = [p.fitness for p in pso_random_policy.pareto_front]

    ind = HV(ref_point=ref_point)
    hv_pso = ind(np.array(pareto_pso))
    hv_pso_trained_policy= ind(np.array(pareto_pso_trained_policy))
    hv_pso_random_policy= ind(np.array(pareto_pso_random_policy))

    time_pso = start_time_pso - end_time_pso
    time_pso_trained_policy = start_time_pso_trained_policy - end_time_pso_trained_policy
    time_random_policy = start_time_pso_random_policy - end_time_pso_random_policy

    result = {'seed'   : seed,
              'models' : {'pso':                {'evaluations'  : evaluations_pso,
                                                 'pareto_front' : pareto_pso,
                                                 'hyper_volume' : hv_pso,
                                                 'time'         : time_pso,
                                                },

                          'pso_trained_policy': {'evaluations'  : evaluations_pso_trained_policy,
                                                 'pareto_front' : pareto_pso_trained_policy,
                                                 'hyper_volume' : hv_pso_trained_policy,
                                                 'time'         : time_pso_trained_policy
                                                },

                          'pso_random_policy':  {'evaluations'  : evaluations_pso_random_policy,
                                                 'pareto_front' : pareto_pso_random_policy,
                                                 'hyper_volume' : hv_pso_random_policy,
                                                 'time'         : time_random_policy
                                                }
                        }
              }
    
    return result

def print_results(results):
    models = results[next(iter(results))]['models']
    models_keys = models.keys()
    metrics_keys = list(models[next(iter(models))].keys()).remove('pareto_front')

    num_metrics = len(metrics_keys)
    num_models = len(models_keys)
    num_seeds = len(results.keys())

    metrics = np.zeros((num_models, num_metrics))

    for s, seed in enumerate(results.keys()):
        for i, mod in enumerate(models_keys):
            for j, met in enumerate(metrics_keys):
                metrics[s][i][j] = results[seed]['models'][mod][met]

    means = np.mean(metrics, axis = 1)
    stds = np.std(metrics, axis = 1)

    for i, mod in enumerate(models_keys):
        print(f"Model {mod}:")
        for j, met in enumerate(metrics_keys):
            print(f"\t{met}: {round(means[i][j], 2)} +- {round(stds[i][j] / np.sqrt(num_seeds), 2)}")

def plot_paretos(results, known_pareto = None):
    for r in results.keys():
        result = results[r]
        paretos = {}
        for mod in result['models'].keys():
            model = result['models'][mod]
            pareto = model['pareto_front']
            axs = []
            num_objectives = len(pareto[0])
            for obj in range(num_objectives):
                axs.append([fitness[obj] for fitness in pareto])
            paretos[mod] = axs
        if num_objectives == 2: plot_pareto_2d(paretos, result['seed'], known_pareto)
        elif num_objectives == 3: plot_pareto_3d(paretos, result['seed'], known_pareto)
        else: print(f"No implementation of plot fuction for {num_objectives} objectives")

def plot_pareto_2d(paretos, seed, known_pareto = None):
    plt.figure()
    if known_pareto is not None:
        plt.scatter(known_pareto[0], known_pareto[1], c='red', s=5, label = mod)
    for mod in paretos.keys():
        pareto = paretos[mod] 
        plt.scatter(pareto[0], pareto[1], s=5, label = mod)

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"Seed: {seed}")
    plt.legend()
    plt.savefig(f"Pareto_front_seed_{seed}.png")
    plt.close()

def plot_pareto_3d(paretos, known_pareto = None):
    pass    