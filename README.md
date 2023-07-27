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

Currently the package provides the `mopso.py` module that defines two classes: `MOPSO` and `Particle`. To use this module in your Python projects:

1. Import the required modules:
    ```python
    from optimizer import MOPSO
    ```
2. Define the objective function to be optimized.
    ```python
    def objective_function(x):
        return np.sin(x[0]) + np.sin(x[1])
    ```
3. Create the MOPSO object with the configuration of the algorithm

    ```python
    mopso = MOPSO(
        objective_functions=[objective_function],
        lower_bound=-5,
        upper_bound=5,
        num_particles=50,
        num_iterations=100
    )
    ```
4. Run the optimization algorithm

    ```python
    pareto_front = mopso.optimize()
    ```

### Objective Function

Depending on the optimization mode (see [next section](#mopso-arguments)), the Objective function can be defined in two way:

In `individual` mode, the objective function is evaluated particle by particle at every iteration and is called as:

```python
objective_funciton(self.position) # self is a Particle object
```

the argument of the optimization function is an array of elements corresponding to the parameters to optimize. The output is the fitness of the particle: the value(s) to minimize in order to solve the optimization problem. 

See [run_mopso.py](example/run_mopso.py) for an example.

in `global` mode, the objective function is evaluated once per iteration and is called as:

```python
objective_function([particle.position for particle in self.particles])
```
the argument of the optimization function is a list of arrays: nn array of parameters for each particle. The output is a list of fitnesses: the value(s) to minimize for each particle.

See [run_mopso.py](example/run_track_mopso.py) for an example.

## Contributing
Contributions are welcome. If you want to contribute, please follow the [Contribution guidelines](CONTRIBUTING.md).

## License

The Optimizer is distributed under the [MPL 2.0 License](LICENSE). Feel free to use, modify, and distribute the code following the terms of the license.  
