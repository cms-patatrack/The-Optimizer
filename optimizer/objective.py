import numpy as np
import time


class Objective():
    def __init__(self, objective_functions, num_objectives=None, sleep_time=0) -> None:
        if not isinstance(objective_functions, list):
            self.objective_functions = [objective_functions]
        else:
            self.objective_functions = objective_functions

        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives
        self.sleep_time = sleep_time

    def populate_matrix(self, outputs, mask):
        result = []
        output_id = 0
        for m in mask:
            if m:
                result.append(outputs[output_id])
                output_id += 1
            else:
                # Maybe len(self.objective_function) depending on how self.num_objectives is defined
                result.append([np.inf] * self.num_objectives)
        return result

    def evaluate(self, items, mask=None):
        if mask is None:
            mask = np.full((len(items)), True, dtype=bool)
        result = [objective_function(np.array(items)[mask])
                  for objective_function in self.objective_functions]
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in r:
                    solutions.append(sub_r)
            else:
                solutions.append(r)
        return np.array(self.populate_matrix(np.array(solutions), mask))

    def type(self):
        return self.__class__.__name__


class ElementWiseObjective(Objective):

    def evaluate(self, items, mask=None):
        if mask is None:
            mask = np.full((len(items)), True, dtype=bool)
        result = [[obj_func(item) for item in np.array(items)[mask]]
                  for obj_func in self.objective_functions]
        solutions = []
        for r in result:
            if len(np.shape(r)) > 1:
                for sub_r in (np.array(r).T):
                    solutions.append(sub_r)
            else:
                solutions.append(r)

        for _ in range(np.sum(mask)):
            time.sleep(self.sleep_time * self.num_objectives)

        return np.array(self.populate_matrix(np.array(solutions).T, mask))


class BatchObjective(Objective):
    pass
