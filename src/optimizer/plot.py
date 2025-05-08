import numpy as np
import matplotlib.pyplot as plt

def _parallel_coordinates(self, ax, highlighted_particle):
    if ax is None:
        ax = plt.gca()
    n = len(self.pareto_front)
    parameters = np.array(
        [particle.position for particle in self.pareto_front])
    parameters_columns = [f'Parameter {i}' for i in range(len(parameters[0]))]
    npar_cols = len(parameters_columns)

    # Scale parameters to range [0, 1]
    min_vals = parameters.min(axis=0)
    max_vals = parameters.max(axis=0)
    scaled_parameters = (parameters - min_vals) / (max_vals - min_vals + 1e-9)

    fitnesses = np.array([particle.fitness for particle in self.pareto_front])
    fitness_columns = [f'Fitness {i}' for i in range(len(fitnesses[0]))]
    nfit_cols = len(fitness_columns)

    for i in range(n):
        y = np.concatenate([scaled_parameters[i], fitnesses[i]])
        x = np.arange(npar_cols + nfit_cols)
        if highlighted_particle is not None and i == highlighted_particle:
            continue
        ax.plot(x, y)

    if highlighted_particle is not None:
        y = np.concatenate(
            [scaled_parameters[highlighted_particle], fitnesses[highlighted_particle]])
        x = np.arange(npar_cols + nfit_cols)
        ax.plot(x, y, linewidth=2, color='red')

    for i in range(npar_cols):
        ax.axvline(i, color='grey')
        ax.text(i, 0, f'{min_vals[i]:.2f}', ha='center',
                va='bottom', fontsize=8, color='black')
        ax.text(i, 1, f'{max_vals[i]:.2f}', ha='center',
                va='top', fontsize=8, color='black')

    ax.set_xticks(np.arange(npar_cols + nfit_cols))
    ax.set_xticklabels(parameters_columns + fitness_columns)

    if len(parameters_columns) > 10:
        plt.xticks(rotation=90)

    ax.set_xlim(0, npar_cols + nfit_cols - 1)
    ax.grid()

    # Put y axis on the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    return ax


def _tight_plot(ax, x, y, ref_x, ref_y, label, ref_label, **kwargs):
    if ax is None:
        ax = plt.gca()  # Get current axes if none is provided
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Extract styling options from kwargs with defaults
    linecolor = kwargs.get('linecolor', 'blue')
    markercolor = kwargs.get('markercolor', 'red')
    markersize = kwargs.get('markersize', 10)
    linestyle = kwargs.get('linestyle', '--')
    ref_linecolor = kwargs.get('ref_linecolor', 'grey')

    ax.step(x, y, where='post', color=linecolor)
    ax.scatter(x, y, s=markersize, facecolors='none', edgecolors=markercolor, label=label)
    if ref_x is not None and ref_y is not None:
        ax.plot(ref_x, ref_y, color=ref_linecolor, label=ref_label, linestyle=linestyle)
    if label:
        ax.legend()
    return ax

# define scatter plot that takes minimum 2d objective but also 3d or Nd. It takes in input the figure

def _scatter(fig, data, **kwargs):
    if fig is None:
        fig = plt.figure()
    if len(data) == 2:
        ax = fig.add_subplot(111)
        ax.scatter(data[0], data[1], **kwargs)
    elif len(data) == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[0], data[1], data[2], **kwargs)
    else:
        # grid of n * n subplots with empty plots on the diagonal and scatter plots on the rest
        n = len(data)
        ax = fig.subplots(n, n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax[i, j].set_visible(False)
                else:
                    ax[i, j].scatter(data[i], data[j], **kwargs)
    return fig, ax
