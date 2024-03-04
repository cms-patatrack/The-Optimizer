import optimizer

lb1 = [0., 1, 0., 5.]
lb2 = [0., 1, 0., "lower"]
ub1 = [1, 2, 3, 4]
ub2 = [2., 1, 1.]


print([type(lb) for lb in lb1])
print([type(lb) for lb in ub1])
def always_true(x):
  return True

objective = optimizer.Objective([always_true])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb1, upper_bounds=ub1 )

print([type(lb) for lb in pso.lower_bounds])
print([type(ub) for ub in pso.upper_bounds])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb1, upper_bounds=ub2 )

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb2, upper_bounds=ub1 )

print([type(lb) for lb in pso.lower_bounds])
print([type(ub) for ub in pso.upper_bounds])