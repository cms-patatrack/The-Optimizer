class Objective():
    def __init__(self, function_list, num_objectives = None) -> None:
        if num_objectives is None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives
        pass
    def type(self):
        return self.__class__.__name__

class ElementWiseObjective(Objective):
    def __init__(self) -> None:
        pass
    
class BatchObjective(Objective):
    def __init__(self) -> None:
        pass