import numpy as np


class Objective():
    def __init__(self, objective_functions, num_objectives=None, evaluation_mask=None) -> None:
        self.objective_functions = objective_functions
        self.evaluation_mask = evaluation_mask
        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives

    def evaluate(self, items, iteration):
        mask = self.evaluation_mask[iteration] if self.evaluation_mask is not None else [True]*len(items)
        result = []
        for objective_function in self.objective_functions:
            evaluation = objective_function(items)
            for i, mask_value in enumerate(mask):
                if not mask_value:
                    evaluation[i] = [np.inf]*self.num_objectives
            result.append(evaluation)
        return np.array(result).T

    def type(self):
        return self.__class__.__name__


class ElementWiseObjective(Objective):
    def __init__(self, objective_functions, num_objectives=None, evaluation_mask=None) -> None:
        super().__init__(objective_functions, num_objectives, evaluation_mask)

    def evaluate(self, items, iteration):
        mask = self.evaluation_mask[iteration] if self.evaluation_mask is not None else [True]*len(items)
        print([int(x) for x in mask])
        result = []
        for i, item in enumerate(items):
            evaluations = []
            if mask[i]:
                for obj_func in self.objective_functions:
                    evaluation = obj_func(item)
                    evaluations.append(evaluation)
            else:
                evaluations = [None]*len(self.objective_functions)
            result.append(evaluations)
        return np.array(result)


class BatchObjective(Objective):
    def __init__(self, objective_functions, num_objectives=None) -> None:
        super().__init__(objective_functions, num_objectives)
