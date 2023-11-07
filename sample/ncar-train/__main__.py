
from time import time
import autosim
from autosim.car.ncar import NetworkArchitecture
from autosim.nn import DenseLayerArchitecture, ActivationFunction, NeuralNetwork
import autosim.training as t
import autosim.estimation as e
import autosim.simulation as s
import autosim.car as c
from helpers.functions import kph_to_mps
from renderer import Renderer
from simulation.location import Circle, CircleSpace, Line
import os

import logging


SOLUTION_FILE = 'solution.json'
L = 500
T_TRAINING = 60
ARCHITECTURE = NetworkArchitecture([DenseLayerArchitecture(4, ActivationFunction.SIGM)] * 1, ActivationFunction.SIGM)

def measure_time(func, /, level=logging.INFO):
    def wrap(func):
        def wrapper(*args, **kwargs):
            nonlocal level

            logging.log(level=level, msg=f"{func.__name__}...")

            start = time()
            ret = func(*args, **kwargs)
            end = time()

            logging.log(level=level, msg=f"finished {func.__name__} in {end - start:.2f} seconds")

            return ret
        return wrapper

    if func is None:
        return wrap
    
    return wrap(func)

def get_solution():
    if os.path.isfile(SOLUTION_FILE):
        return load()
    else:
        return train()

def load():
    logging.info(f"load solution from {os.path.realpath(SOLUTION_FILE)}...")
    return NeuralNetwork.from_file(SOLUTION_FILE)

@measure_time
def train():
    POPULATION = 16
    GENERATIONS = 10
    PARENTS_MATING = 8
    
    estimation_strategy = e.EstimationStrategy(collision=e.Criteria(10000), distance=e.ReferenceCriteria(1, 100))

    suites = [
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(f"{kph_to_mps(60)} * dt", c.ACar.Mode.MOVEMENT, location=Line(100), name='trainer')]
                            )
                        ),
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(f"{kph_to_mps(40)} * dt", c.ACar.Mode.MOVEMENT, location=Line(100), name='trainer')]
                            )
                        ),
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(f"{0.5}", c.ACar.Mode.ACCELERATION, location=Line(100), name='trainer')]
                            )
                        ),
        ]

    ga = t.GeneticAlgorithmParameters(
        num_generations=GENERATIONS, num_parents_mating=PARENTS_MATING,
        on_start=lambda ga:logging.info(f"genetic algorithm started"),
        on_stop=lambda ga,fitnesses:logging.info(f"genetic algorithm stopped"),
        on_generation=lambda ga:logging.info(f"finished {ga.generations_completed}/{GENERATIONS} generations"),
        )
    strategy = t.TrainingStrategy(suites=suites, ga=ga, aggregation=t.SuiteAggregationStrategy.SUM_FINES)


    session = t.train(strategy, population=t.Population.random(n=POPULATION, architecture=ARCHITECTURE, bound=(-5, 5)))

    solution, solution_fitness, solution_idx = session.ga.best_solution()
    logging.info(f"trained {session.ga.generations_completed} generations of {POPULATION}")
    logging.info(f"best solution found in generation {session.ga.best_solution_generation} with fitness {solution_fitness} (fine {1 / solution_fitness if solution_fitness > 0 else '+inf'})")

    network = NeuralNetwork.from_vector(architecture=ARCHITECTURE, vector=solution)
    network.to_file(SOLUTION_FILE)

    return network

@measure_time
def simulate(network):
    CARS_TESTED = 5
    T_RENDER = 3 * T_TRAINING
    FPS = 24
    W = 768
    H = 768
    renderer = Renderer(fps=FPS, width=W, height=H)
    parameters = s.SimulationParameters(timeout=T_RENDER, objects=[
        c.NCar(network=network,
               name=f"student-{i}",
               location = Circle(CircleSpace(L), i * L / (CARS_TESTED + 1)))
        for i in range(CARS_TESTED)] + [c.ACar(function=f"0.5 + 0.5 * sin(2 * pi * t / {10})",
                                        mode = c.ACar.Mode.ACCELERATION,
                                        location=Circle(CircleSpace(L), CARS_TESTED * L / (CARS_TESTED + 1)),
                                        name='automatic'), renderer]
        )
    autosim.simulate(parameters)
    logging.info( f"simulated {parameters.timeout}s ({FPS * parameters.timeout} {W}x{H} frames)")
    return renderer

@measure_time
def render(renderer):
    renderer.render('output')

def everything():
    render(simulate(get_solution()))

@measure_time
def main():
    for handler in logging.getLogger().handlers:
        handler.setFormatter(logging.Formatter(fmt='%(levelname)s\t%(message)s'))
    logging.getLogger().setLevel(logging.INFO)
    everything()
    return 0

if __name__ == '__main__':
    ret = main()
    exit(ret)
