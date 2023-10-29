
import autosim.car
import autosim.estimation
import autosim.simulation
import copy
import sys


def simulate(parameters: autosim.simulation.SimulationParameters):
    simulation = autosim.simulation.Simulation(parameters)
    simulation.simulate()


def fitness(target: autosim.car.Car, simulation_parameters: autosim.simulation.SimulationParameters, strategy: autosim.estimation.Strategy) -> float:
    estimator = autosim.estimation.EstimatorObject(target=target, strategy=strategy)

    simulation_parameters = copy.copy(simulation_parameters)
    simulation_parameters.objects += [estimator]

    simulate(simulation_parameters)

    if estimator.fine == 0:
        return sys.float_info.max
    else:
        return 1 / estimator.fine

