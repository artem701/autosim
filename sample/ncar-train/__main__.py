
from time import time
import autosim
from autosim.car.ncar import NetworkArchitecture
from autosim.nn import DenseLayerArchitecture, ActivationFunction, NeuralNetwork
import autosim.training as t
import autosim.estimation as e
import autosim.simulation as s
import autosim.car as c
from renderer import Renderer
from simulation.location import Circle, CircleSpace, Line

import logging


logging.getLogger().setLevel(logging.INFO)
logging.info('initializing...')

L = 500
T = 60
FPS = 24
W = 768
H = 768
ARCHITECTURE = NetworkArchitecture([DenseLayerArchitecture(4, ActivationFunction.SIGM)] * 1, ActivationFunction.SIGM)

POPULATION = 16
GENERATIONS = 25
PARENTS_MATING = 8

estimation_strategy = e.EstimationStrategy(collision=e.Criteria(1000), distance=e.ReferenceCriteria(1, 1))
simulation_parameters = s.SimulationParameters(
    timeout=T,
    objects=[c.ACar(f"{c.kph_to_mps(60)} * dt", c.ACar.Mode.MOVEMENT, location=Line(100), name='trainer')]
    )

suites = [t.TrainingSuite(estimation=estimation_strategy, simulation=simulation_parameters)]
ga = t.GeneticAlgorithmParameters(
    num_generations=GENERATIONS, num_parents_mating=PARENTS_MATING,
    on_start=lambda ga:logging.info(f"genetic algorithm started"),
    on_stop=lambda ga,fitnesses:logging.info(f"genetic algorithm stopped"),
    on_generation=lambda ga:logging.info(f"finished {ga.generations_completed}/{GENERATIONS} generations"),
    )
strategy = t.TrainingStrategy(suites, ga)


logging.info('training...')
start = time()
session = t.train(strategy, population=t.Population.random(POPULATION, ARCHITECTURE))
end = time()
training_time = end - start

solution, solution_fitness, solution_idx = session.ga.best_solution()
logging.info(f"Trained {len(session.ga.generations_completed)} generations of {POPULATION} cars for {training_time:.2f}s")
logging.info(f"Best solution found in generation {session.ga.best_solution_generation} with fitness {solution_fitness} (fine {1 / solution_fitness if solution_fitness > 0 else '+inf'})")

logging.info('simulating...')
cars = [c.NCar(network=NeuralNetwork.from_vector(architecture=ARCHITECTURE, vector=solution),
               spec=strategy.spec,
               f=strategy.friction,
               name=f"student-{i}",
               location = Circle(CircleSpace(L), i * L / 5))
        for i in range(5)]
renderer = Renderer(fps=FPS, width=W, height=H)
simulation_parameters.objects = [*cars, renderer]
simulation_parameters.timeout = 2 * T

start = time()
autosim.simulate(simulation_parameters)
end = time()
simulation_time = end - start

logging.info('rendering...')
start = time()
renderer.render('output')
end = time()
rendering_time = end - start

logging.info(f"Trained {len(session.generations)} generations of {POPULATION} cars for {training_time:.2f}s")
logging.info(
    f"Simulated {simulation_parameters.timeout}s ({FPS*simulation_parameters.timeout} {W}x{H} frames) in {simulation_time:.2f} real seconds, render for {rendering_time:.2f} seconds")
