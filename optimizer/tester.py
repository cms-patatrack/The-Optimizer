import optimizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.colors as mcolors
from pymoo.indicators.hv import HV
import time
from tqdm import tqdm
import json
from stable_baselines3 import PPO

def test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, name, plot_paretos_enabled = False, print_results_enabled = True, known_pareto=None, verbose = 0):
    results = {}
    for seed in tqdm(seeds, desc="Testing seed", unit="iter"):
        result = test_seed(objective, mopso_parameters, num_iterations, rl_model, ref_point, seed, verbose = verbose)
        results[f'{seed}'] = result

    if print_results_enabled: print_results(results)
    if plot_paretos_enabled: plot_paretos(results, known_pareto)

    out_file = open(f"{name}.json", "w")

    json.dump(results, out_file, indent = 6)

    out_file.close()

    return results

def test_seed(objective, mopso_parameters, num_iterations, rl_model, ref_point, seed, verbose = 0):

    if verbose > 0 : print(f"SEED {seed}")

    #Optimizers
    optimizers = []

    if verbose > 1 : print("Starting MOPSO withot RL")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso = optimizer.MOPSO(objective=objective, 
                          lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                          inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                          initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                          rl_model=None, radius=mopso_parameters['radius'])

    start_time_pso = time.time()                    
    pso.optimize(num_iterations=num_iterations)
    end_time_pso = time.time()
    optimizers.append(pso)

    if verbose > 1 : print("Starting MOPSO with trained policy")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_trained_policy = optimizer.MOPSO(objective=objective, 
                          lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                          inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                          initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                          rl_model = rl_model, radius=mopso_parameters['radius'])

    start_time_pso_trained_policy = time.time() 
    pso_trained_policy.optimize(num_iterations=num_iterations)
    end_time_pso_trained_policy = time.time() 
    optimizers.append(pso_trained_policy)

    if verbose > 1 : print("Starting MOPSO with random policy")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_random_policy = optimizer.MOPSO(objective=objective, 
                          lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                          inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                          initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                          rl_model='random', radius=mopso_parameters['radius'])

    start_time_pso_random_policy = time.time() 
    pso_random_policy.optimize(num_iterations=num_iterations)
    end_time_pso_random_policy = time.time()
    optimizers.append(pso_random_policy)

    if verbose > 1 : print("Starting MOPSO with explainable policy")
    optimizer.Randomizer.rng = np.random.default_rng(seed)
    pso_explainable_policy = optimizer.MOPSO(objective=objective, 
                          lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                          inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                          initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                          rl_model='explainable', radius=mopso_parameters['radius'])

    start_time_pso_explainable_policy = time.time() 
    pso_explainable_policy.optimize(num_iterations=num_iterations)
    end_time_pso_explainable_policy = time.time()
    optimizers.append(pso_explainable_policy)

    # Results
    evaluations_pso = int(sum(pso.evaluations))
    evaluations_pso_trained_policy = int(sum(pso_trained_policy.evaluations))
    evaluations_pso_random_policy = int(sum(pso_random_policy.evaluations))
    evaluations_pso_explainable_policy = int(sum(pso_explainable_policy.evaluations))

    pareto_pso = [p.fitness.tolist() for p in pso.pareto_front]
    pareto_pso_trained_policy = [p.fitness.tolist() for p in pso_trained_policy.pareto_front]
    pareto_pso_random_policy = [p.fitness.tolist() for p in pso_random_policy.pareto_front]
    pareto_pso_explainable_policy = [p.fitness.tolist() for p in pso_explainable_policy.pareto_front]

    ind = HV(ref_point=ref_point)
    hv_pso = ind(np.array(pareto_pso))
    hv_pso_trained_policy= ind(np.array(pareto_pso_trained_policy))
    hv_pso_random_policy= ind(np.array(pareto_pso_random_policy))
    hv_pso_explainable_policy= ind(np.array(pareto_pso_explainable_policy))

    time_pso = end_time_pso - start_time_pso
    time_pso_trained_policy = end_time_pso_trained_policy - start_time_pso_trained_policy
    time_random_policy = end_time_pso_random_policy - start_time_pso_random_policy
    time_explainable_policy = end_time_pso_explainable_policy - start_time_pso_explainable_policy

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
                                                },

                          'pso_explainable_policy':  {'evaluations'  : evaluations_pso_explainable_policy,
                                                 'pareto_front' : pareto_pso_explainable_policy,
                                                 'hyper_volume' : hv_pso_explainable_policy,
                                                 'time'         : time_explainable_policy
                                                }
                        }
              }
    
    return result

def print_results(results):
    results = check_dict(results)
    models = results[next(iter(results))]['models']
    models_keys = models.keys()
    metrics_keys = list(models[next(iter(models))].keys())
    metrics_keys.remove('pareto_front')

    num_metrics = len(metrics_keys)
    num_models = len(models_keys)
    num_seeds = len(results.keys())

    metrics = np.zeros((num_seeds, num_models, num_metrics))

    for s, seed in enumerate(results.keys()):
        for i, mod in enumerate(models_keys):
            for j, met in enumerate(metrics_keys):
                metrics[s][i][j] = results[seed]['models'][mod][met]

    means = np.mean(metrics, axis = 0)
    stds = np.std(metrics, axis = 0)

    for i, mod in enumerate(models_keys):
        print(f"Model {mod}:")
        for j, met in enumerate(metrics_keys):
            print(f"\t{met}: {means[i][j]} +- {stds[i][j] / np.sqrt(num_seeds)}")

def plot_paretos(results, known_pareto = None):
    results = check_dict(results)
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
    markers = list(MarkerStyle('').markers.keys())
    plt.figure()
    if known_pareto is not None:
        plt.scatter(known_pareto[0], known_pareto[1], c='red', s=5, label = 'Known pareto')
    for i, mod in enumerate(paretos.keys()):
        pareto = paretos[mod] 
        plt.scatter(pareto[0], pareto[1], s=5, label = mod, marker=markers[i])

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"Seed: {seed}")
    plt.legend()
    plt.savefig(f"./tests/paretos/Pareto_front_seed_{seed}.png")
    plt.close()

def plot_pareto_3d(paretos, known_pareto = None):
    pass 

def check_dict(results):
    if type(results) is not dict:
        print(f"Loading file {results}")
        f = open(results) 
        results = json.load(f)
    return results

def explainability(rl_model, num_points):
    model = PPO.load(rl_model)
    res = np.empty((num_points,num_points))
    for x in range(num_points):
        for y in range(num_points):
            res[y,x] = model.predict([x, y], deterministic=True)[0].tolist()

    cmap = mcolors.ListedColormap(['red', 'limegreen'])
    plt.imshow(res, cmap=cmap, origin='lower')
    ax = plt.gca();
    ax.set_xticks(np.arange(0, num_points, 5));
    ax.set_yticks(np.arange(0, num_points, 5));
    plt.xlabel("Bad points")
    plt.ylabel("Pareto points")

    x = np.linspace(0, num_points)
    y = 5 * x
    plt.plot(x, y, label=f'y = {5}x', color = 'black')
    plt.legend()
    plt.ylim(top=num_points)
    name = rl_model.replace("./models/", '')
    name = name.replace("/model", '')
    plt.savefig(f"explainability_{name}.png")