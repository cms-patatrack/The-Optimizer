# The Optimizer

The Optimizer is a python package that provides a collection of powerful optimization algorithms, including MOPSO (Multi-Objective Particle Swarm Optimization). The primary purpose of this package is to facilitate running optimization tasks using user-defined Python functions as the optimization target.
The package is developed with the objectives of CMS and Patatrack in mind.

- [The Optimizer](#the-optimizer)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Objective Function](#objective-function)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

To use The Optimizer package, you can install it using pip and the provided setup.py script.

1. Clone this repository
1. Navigate to the project directory
1. Install the package and its dependencies using pip:

    ```bash
    pip install .
    ```

You can install a project in “editable” or “develop” mode while you’re working on it. When installed as editable, a project can be edited in-place without reinstallation:

```bash
python3 -m pip install -e .
```

## Usage

Currently the package provides the `optimizer` module that defines an optimization algorithm: `MOPSO`.
The optimizer relies on a few helper classes to handle configuration and the objective functions. To use this module in your Python projects:

1. Import the required modules:

    ```python
    import optimizer
    ```

2. Define the objective function to be optimized. I.e.:

    ```python
    def f1(x):
        return 4 * x[0]**2 + 4 * x[1]**2


    def f2(x):
        return (x[0] - 5)**2 + (x[1] - 5)**2

    objective = optimizer.ElementWiseObjective([f1, f2])
    ```

3. Define the boundaries of the parameters:

    ```python
    lb = [0.0, 0.0]
    ub = [5.0, 3.0]
    ```

4. Create the MOPSO object with the configuration of the algorithm

    ```python
    mopso = optimizer.MOPSO(objective=objective,
                            lower_bounds=lb, upper_bounds=ub,
                            num_particles=50,
                            inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2,
                            initial_particles_position='random', exploring_particles=True,
                            max_pareto_lenght=100)
    ```

5. Run the optimization algorithm

    ```python
    pareto_front = mopso.optimize(num_iterations = 100)
    ```

### Objective Function

Depending on the optimization mode, the Objective function can be defined in two way:

1. As `Objective`, the objective function is evaluated once per iteration and is called as:

    ```python
    f([particle.position for particle in self.particles])
    ```

    the argument of the optimization function is a list of arrays: an array of parameters for each particle.  
    The output is a list of fitnesses: the value(s) to minimize for each particle.

2. As `ElementWiseObjective`, the objective function is evaluated particle by particle at every iteration and is called as:

    ```python
    f(self.position) # self is a Particle object
    ```

    the argument of the optimization function is an array of elements corresponding to the parameters to optimize.  
    The output is the fitness of the particle: the value(s) to minimize in order to solve the optimization problem.  

See the `tests` and `examples` folders for examples.

### MOPSO

The Multi-Objective Particle Swarm Optimization (MOPSO) algorithm is a versatile optimization tool designed for solving multi-objective problems. It leverages the concept of swarm to navigate the search space and find optimal solutions.

- **Objective**: MOPSO can optimize virtually any objective function defined by the user.
- **Boundary Constraints**: Users can specify lower and upper bounds for each parameter, and uses the definition of the boundaries to detect the variable types (i.e. `0.0` for floating points, `0` for integers, `False` for booleans)
- **Swarm Size**: Adjusting the number of particles in the swarm allows to balance convergence speed and computation intensity.
- **Inertia Weight**: Control the inertia of the particle velocity to influence the global and local search capabilities.
- **Cognitive and Social Coefficients**: Fine-tune the cognitive and social components of the velocity update equation to steer the search process.
- **Initial Particle Position**: Offers multiple strategies for initializing particle positions:

  - `random` uniform distribution
  - `gaussian` distribution around a given point
  - all in the `lower_bounds` or `upper_bounds` of the parameter space
- **Exploration Mode**: An optional exploration mode enables particles to scatter from their position when they don't improve for a given number of iterations
- **Swarm Topology**: Supports different swarm topologies, affecting how particles chose the `global_best` to follow.

See the docstring for additional information on the parameters.

## Contributing

Contributions are welcome. If you want to contribute, please follow the [Contribution guidelines](CONTRIBUTING.md).

## License

The Optimizer is distributed under the [MPL 2.0 License](LICENSE). Feel free to use, modify, and distribute the code following the terms of the license.  
