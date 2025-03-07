import numpy as np

# Generational distance
def generational_distance(pareto_front, reference_front):
    """
    This function calculates the generational distance metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_front : numpy array
        Represents the true pareto front.
    Returns:
    generational_distance : float
        The generational distance metric value.
    """
    return np.mean(np.min(np.linalg.norm(pareto_front - reference_front, axis=1), axis=0))

# Inverted generational distance
def inverted_generational_distance(pareto_front, reference_front):
    """
    This function calculates the inverted generational distance metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_front : numpy array
        Represents the true pareto front.
    Returns:
    inverted_generational_distance : float
        The inverted generational distance metric value.
    """
    return np.mean(np.min(np.linalg.norm(reference_front - pareto_front, axis=1), axis=0))

def generational_distance_plus(pareto_front, reference_front):
    """
    This function calculates the generational distance plus metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_front : numpy array
        Represents the true pareto front.
    Returns:
    generational_distance_plus : float
        The generational distance plus metric value.
    """
    return np.mean(np.min(np.linalg.norm(pareto_front - reference_front, axis=1), axis=0)) + np.mean(np.min(np.linalg.norm(reference_front - pareto_front, axis=1), axis=0))

def inverted_generational_distance_plus(pareto_front, reference_front):
    """
    This function calculates the inverted generational distance plus metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_front : numpy array
        Represents the true pareto front.
    Returns:
    inverted_generational_distance_plus : float
        The inverted generational distance plus metric value.
    """
    return np.mean(np.min(np.linalg.norm(reference_front - pareto_front, axis=1), axis=0)) + np.mean(np.min(np.linalg.norm(pareto_front - reference_front, axis=1), axis=0))