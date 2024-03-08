import optimizer

lb = [0., False]
ub = [2., True]

optimizer.FileManager.working_dir = "tmp/bool/"
optimizer.FileManager.saving_enabled = True


def func(x):
    if x[1] == False:
        print(x, x[0] >= 1)
        return int(x[0] >= 1)
    else:
        print(x, x[0] < 1)
        return int(x[0] < 1)


objective = optimizer.ElementWiseObjective([func])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      initial_particles_position='random', num_particles=5)

pso.optimize(10)
