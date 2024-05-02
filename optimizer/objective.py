import numpy as np


class Objective():
    def __init__(self, objective_functions, num_objectives=None) -> None:
        self.objective_functions = objective_functions
        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives
        pass

    def evaluate(self, items):
        return np.array([objective_function(items) for objective_function in self.objective_functions]).T

    def type(self):
        return self.__class__.__name__


class ElementWiseObjective(Objective):
    def __init__(self, objective_functions, num_objectives=None) -> None:
        super().__init__(objective_functions, num_objectives)

    def evaluate(self, items):
        return np.array([[obj_func(item) for obj_func in self.objective_functions] for item in items] )


class BatchObjective(Objective):
    def __init__(self, objective_functions, num_objectives=None) -> None:
        super().__init__(objective_functions, num_objectives)
