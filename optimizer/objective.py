class Objective():
    def __init__(self, function_list, num_objectives = None) -> None:
        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives
        pass

    def evaluate(self, items, worker_id=0):
        return np.array([objective_function([item for item in items], worker_id) for objective_function in self.objective_functions])

    def type(self):
        return self.__class__.__name__

class ElementWiseObjective(Objective):
    def __init__(self, objective_functions, num_objectives=None) -> None:
        super().__init__(objective_functions, num_objectives)

    def evaluate(self, items, worker_id=0):
        return np.array([[obj_func(item, worker_id) for item in items] for obj_func in self.objective_functions])

class BatchObjective(Objective):
    def __init__(self, objective_functions, num_objectives=None) -> None:
        super().__init__(objective_functions, num_objectives)
