
import autosim.car
import autosim.estimation
import autosim.simulation
import autosim.training
import copy
import sys


def simulate(parameters: autosim.simulation.SimulationParameters):
    """Perform simulation. Applies side effects!
    """
    simulation = autosim.simulation.Simulation(parameters)
    simulation.simulate()


def fitness(target: autosim.car.Car, simulation_parameters: autosim.simulation.SimulationParameters, strategy: autosim.estimation.EstimationStrategy) -> float:
    """Evaluates fitness as 1 / fine. Does not apply any side effects.
    """
    estimator = autosim.estimation.EstimatorObject(target=target, strategy=strategy)

    simulation_parameters = copy.deepcopy(simulation_parameters)
    simulation_parameters.objects += [target, estimator]

    simulate(simulation_parameters)

    if estimator.fine == 0:
        return sys.float_info.max
    else:
        return 1 / estimator.fine

def train(strategy: autosim.training.TrainingStrategy, population: autosim.training.Population):
    """Train population. Does not provide apply any side effect.
    """
    return autosim.training.train(strategy=strategy, population=population)
