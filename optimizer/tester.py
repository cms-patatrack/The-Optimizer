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
import plotly.graph_objs as go
import plotly.io as pio
from copy import deepcopy

VALID_MODELS = ['pso', 'pso_trained_policy', 'pso_random_policy', 'pso_explainable_policy']

class results_container:
    def __init__(self, res):
        self.res = self.check_dict(res)
        models = self.res[next(iter(res))]['models']
        self.models_keys = models.keys()
        self.metrics_keys = list(models[next(iter(models))].keys())
        self.metrics_keys.remove('pareto_front')
        self.num_metrics = len(self.metrics_keys)
        self.num_models = len(self.models_keys)
        self.num_seeds = len(self.res.keys())

        self.means = None
        self.stds = None

        self.metric_to_index_map = {}
        for i, met in enumerate(self.metrics_keys):
            self.metric_to_index_map[met] = i

    def check_dict(self, res):
        if type(res) is not dict:
            if type(res) is not str:
                 raise ValueError(f"Results must be a dictionary or a string")
            print(f"Loading file {res}")
            f = open(res) 
            res = json.load(f)
        return res

    def print_results(self):
        for i, mod in enumerate(self.models_keys):
            print(f"Model {mod}:")
            for j, met in enumerate(self.metrics_keys):
                print(f"\t{met}: {self.get_metric_means(met)[mod]} +- {self.get_metric_stds(met)[mod] / np.sqrt(self.num_seeds)}")

    def calculate_metrics_momenta(self):
        metrics = np.zeros((self.num_seeds, self.num_models, self.num_metrics))
        for s, seed in enumerate(self.res.keys()):
            for i, mod in enumerate(self.models_keys):
                for j, met in enumerate(self.metrics_keys):
                    metrics[s][i][j] = self.res[seed]['models'][mod][met]

        # means has models on rows and metrics on cols
        self.means = np.mean(metrics, axis = 0)
        self.stds = np.std(metrics, axis = 0)
        return self.means, self.stds

    def get_metric_momentum(self, metric, matrix):
        metric_id = self.metric_to_index_map[metric]
        metric_means = {}
        for i, mod in enumerate(self.models_keys):
            metric_means[mod] = matrix[i][metric_id]
        return metric_means
    
    def get_metric_means(self, metric):
        if self.means is None:
            self.calculate_metrics_momenta()
        return self.get_metric_momentum(metric, self.means)
    
    def get_metric_stds(self, metric):
        if self.stds is None:
            self.calculate_metrics_momenta()
        return self.get_metric_momentum(metric, self.stds)

    def plot_paretos(self, known_pareto = None):
        for r in self.res.keys():
            result = self.res[r]
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

def test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, name, plot_paretos_enabled = False, print_results_enabled = True, known_pareto=None, time_limit = np.inf, models_to_test = VALID_MODELS, verbose = 0):
    results = dict()
    for seed in tqdm(seeds, desc="Testing seed", unit="iter"):
        res = test_seed(objective, mopso_parameters, num_iterations, rl_model, ref_point, seed, time_limit, models_to_test = models_to_test, verbose = verbose)
        results[f"{seed}"] = res

    out_file = open(f"{name}.json", "w")
    json.dump(results, out_file, indent = 6)
    out_file.close()

    results_obj = results_container(results)
    if print_results_enabled: results_obj.print_results()
    if plot_paretos_enabled: results_obj.plot_paretos(known_pareto)

    return results_obj

def test_seed(objective, mopso_parameters, num_iterations, rl_model, ref_point, seed, time_limit = np.inf, models_to_test = VALID_MODELS, verbose = 0):

    print(f"Testing models: {models_to_test}")
    res =    {'seed'   : seed,
              'models' : {}
             }
    ind = HV(ref_point=ref_point)
    
    if verbose > 0 : print(f"SEED {seed}")

    #Optimizers
    optimizers = []

    if VALID_MODELS[0] in models_to_test:
        if verbose > 1 : print("Starting MOPSO withot RL")
        optimizer.Randomizer.rng = np.random.default_rng(seed)
        pso = optimizer.MOPSO(objective=objective, 
                            lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                            initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                            rl_model=None, radius_scaler=mopso_parameters['radius_scaler'])

        start_time_pso = time.time()                    
        pso.optimize(num_iterations=num_iterations, time_limit=time_limit)
        end_time_pso = time.time()
        optimizers.append(pso)

        evaluations_pso = int(sum(pso.evaluations))
        pareto_pso = [p.fitness.tolist() for p in pso.pareto_front]
        hv_pso = ind(np.array(pareto_pso))
        time_pso = end_time_pso - start_time_pso

        res['models']['pso'] = {'evaluations'  : evaluations_pso,
                                'pareto_front' : pareto_pso,
                                'hyper_volume' : hv_pso,
                                'time'         : time_pso,
                            }
        
    if VALID_MODELS[1] in models_to_test:
        if verbose > 1 : print("Starting MOPSO with trained policy")
        optimizer.Randomizer.rng = np.random.default_rng(seed)
        pso_trained_policy = optimizer.MOPSO(objective=objective, 
                            lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                            initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                            rl_model = rl_model, radius_scaler=mopso_parameters['radius_scaler'])

        start_time_pso_trained_policy = time.time() 
        pso_trained_policy.optimize(num_iterations=num_iterations, time_limit=time_limit)
        end_time_pso_trained_policy = time.time() 
        optimizers.append(pso_trained_policy)

        evaluations_pso_trained_policy = int(sum(pso_trained_policy.evaluations))
        pareto_pso_trained_policy = [p.fitness.tolist() for p in pso_trained_policy.pareto_front]
        hv_pso_trained_policy= ind(np.array(pareto_pso_trained_policy))
        time_pso_trained_policy = end_time_pso_trained_policy - start_time_pso_trained_policy

        res['models']['pso_trained_policy'] = {'evaluations'  : evaluations_pso_trained_policy,
                                            'pareto_front' : pareto_pso_trained_policy,
                                            'hyper_volume' : hv_pso_trained_policy,
                                            'time'         : time_pso_trained_policy
                                            }
        
    if VALID_MODELS[2] in models_to_test:
        if verbose > 1 : print("Starting MOPSO with random policy")
        optimizer.Randomizer.rng = np.random.default_rng(seed)
        pso_random_policy = optimizer.MOPSO(objective=objective, 
                            lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                            initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                            rl_model='random', radius_scaler=mopso_parameters['radius_scaler'])

        start_time_pso_random_policy = time.time() 
        pso_random_policy.optimize(num_iterations=num_iterations, time_limit=time_limit)
        end_time_pso_random_policy = time.time()
        optimizers.append(pso_random_policy)

        evaluations_pso_random_policy = int(sum(pso_random_policy.evaluations))
        pareto_pso_random_policy = [p.fitness.tolist() for p in pso_random_policy.pareto_front]
        hv_pso_random_policy= ind(np.array(pareto_pso_random_policy))
        time_random_policy = end_time_pso_random_policy - start_time_pso_random_policy

        res['models']['pso_random_policy'] = {'evaluations'  : evaluations_pso_random_policy,
                                            'pareto_front' : pareto_pso_random_policy,
                                            'hyper_volume' : hv_pso_random_policy,
                                            'time'         : time_random_policy
                                            }
    if VALID_MODELS[3] in models_to_test:
        if verbose > 1 : print("Starting MOPSO with explainable policy")
        optimizer.Randomizer.rng = np.random.default_rng(seed)
        pso_explainable_policy = optimizer.MOPSO(objective=objective, 
                            lower_bounds=mopso_parameters['lower_bounds'], upper_bounds=mopso_parameters['upper_bounds'], num_particles=mopso_parameters['num_particles'],
                            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, topology = mopso_parameters['topology'],
                            initial_particles_position='random', exploring_particles = mopso_parameters['exploring_particles'],
                            rl_model='explainable', radius_scaler=mopso_parameters['radius_scaler'])

        start_time_pso_explainable_policy = time.time() 
        pso_explainable_policy.optimize(num_iterations=num_iterations, time_limit=time_limit)
        end_time_pso_explainable_policy = time.time()
        optimizers.append(pso_explainable_policy)

        evaluations_pso_explainable_policy = int(sum(pso_explainable_policy.evaluations))
        pareto_pso_explainable_policy = [p.fitness.tolist() for p in pso_explainable_policy.pareto_front]
        hv_pso_explainable_policy= ind(np.array(pareto_pso_explainable_policy))
        time_explainable_policy = end_time_pso_explainable_policy - start_time_pso_explainable_policy

        res['models']['pso_explainable_policy'] = {'evaluations'  : evaluations_pso_explainable_policy,
                                                'pareto_front' : pareto_pso_explainable_policy,
                                                'hyper_volume' : hv_pso_explainable_policy,
                                                'time'         : time_explainable_policy
                                                }

    # Results
    # evaluations_pso = int(sum(pso.evaluations))
    # evaluations_pso_trained_policy = int(sum(pso_trained_policy.evaluations))
    # evaluations_pso_random_policy = int(sum(pso_random_policy.evaluations))
    # evaluations_pso_explainable_policy = int(sum(pso_explainable_policy.evaluations))

    # pareto_pso = [p.fitness.tolist() for p in pso.pareto_front]
    # pareto_pso_trained_policy = [p.fitness.tolist() for p in pso_trained_policy.pareto_front]
    # pareto_pso_random_policy = [p.fitness.tolist() for p in pso_random_policy.pareto_front]
    # pareto_pso_explainable_policy = [p.fitness.tolist() for p in pso_explainable_policy.pareto_front]

    # ind = HV(ref_point=ref_point)
    # hv_pso = ind(np.array(pareto_pso))
    # hv_pso_trained_policy= ind(np.array(pareto_pso_trained_policy))
    # hv_pso_random_policy= ind(np.array(pareto_pso_random_policy))
    # hv_pso_explainable_policy= ind(np.array(pareto_pso_explainable_policy))

    # time_pso = end_time_pso - start_time_pso
    # time_pso_trained_policy = end_time_pso_trained_policy - start_time_pso_trained_policy
    # time_random_policy = end_time_pso_random_policy - start_time_pso_random_policy
    # time_explainable_policy = end_time_pso_explainable_policy - start_time_pso_explainable_policy

    # res =    {'seed'   : seed,
    #           'models' : {'pso':                    {'evaluations'  : evaluations_pso,
    #                                                  'pareto_front' : pareto_pso,
    #                                                  'hyper_volume' : hv_pso,
    #                                                  'time'         : time_pso,
    #                                                 },

    #                       'pso_trained_policy':     {'evaluations'  : evaluations_pso_trained_policy,
    #                                                  'pareto_front' : pareto_pso_trained_policy,
    #                                                  'hyper_volume' : hv_pso_trained_policy,
    #                                                  'time'         : time_pso_trained_policy
    #                                                 },

    #                       'pso_random_policy':      {'evaluations'  : evaluations_pso_random_policy,
    #                                                  'pareto_front' : pareto_pso_random_policy,
    #                                                  'hyper_volume' : hv_pso_random_policy,
    #                                                  'time'         : time_random_policy
    #                                                 },

    #                       'pso_explainable_policy': {'evaluations'  : evaluations_pso_explainable_policy,
    #                                                 'pareto_front' : pareto_pso_explainable_policy,
    #                                                 'hyper_volume' : hv_pso_explainable_policy,
    #                                                 'time'         : time_explainable_policy
    #                                                 }
    #                     }
    #           }
    
    return res

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

def plot_pareto_3d(paretos, seed, known_pareto = None):

    colors = ['red', 'blue', 'green', 'yellow']
    model_paretos = {'pareto_obj1' : [],
                      'pareto_obj2' : [],
                      'pareto_obj3' : [],
                      }
    ranges = np.empty((3,2))
    ranges[:, 0] = np.inf
    ranges[:, 1] = -np.inf

    fig = go.Figure()

    # if known_pareto is not None:
    #     plt.scatter(known_pareto[0], known_pareto[1], c='red', s=5, label = 'Known pareto')

    for i, mod in enumerate(paretos.keys()):
        pareto = paretos[mod]
        model_paretos ['pareto_obj1'] = [fitness[0] for fitness in pareto]
        model_paretos ['pareto_obj2'] = [fitness[1] for fitness in pareto]
        model_paretos ['pareto_obj3'] = [fitness[2] for fitness in pareto]
        for j, k in enumerate(model_paretos.keys()):
            min_pareto = np.min(model_paretos[k])
            max_pareto = np.max(model_paretos[k])
            if min_pareto < ranges[j][0]: ranges[j][0] = min_pareto
            if max_pareto > ranges[j][1]: ranges[j][1] = max_pareto
        
        fig.add_trace(go.Scatter3d(
                        x=pareto[0],
                        y=pareto[1],
                        z=pareto[2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=colors[i],
                            opacity=0.8
                            )
                        )
                    #   go.Layout(
                    #         scene=dict(
                    #         xaxis=dict(title='Objective 1', range=ranges[0]),
                    #         yaxis=dict(title='Objective 2', range=ranges[1]),
                    #         zaxis=dict(title='Objective 3', range=ranges[2])
                    #         )
                    #     )

                    )

    pio.write_html(fig, file=f"./tests/paretos/Pareto_front_seed_{seed}.html", auto_open=True)

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
    plt.savefig(f"explainability.png")