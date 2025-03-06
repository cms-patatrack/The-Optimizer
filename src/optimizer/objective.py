import numpy as np


class Objective():
    def __init__(self, objective_functions, num_objectives=None) -> None:
        if not isinstance(objective_functions, list):
            self.objective_functions = [objective_functions]
        else:
            self.objective_functions = objective_functions

        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives

    def evaluate(self, items):
        result = [objective_function(items)
                  for objective_function in self.objective_functions]
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in r:
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        return np.array(solutions)

    def type(self):
        return self.__class__.__name__


class ElementWiseObjective(Objective):
    def evaluate(self, items):
        result = [[obj_func(item) for item in items]
                  for obj_func in self.objective_functions]
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in (np.array(r).T):
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        return np.array(solutions).T


class BatchObjective(Objective):
    pass
