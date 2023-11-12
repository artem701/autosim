
from time import time
import autosim
from autosim.car import specs
from autosim.car.ncar import NCar, NeuralNetwork, NetworkArchitecture
from autosim.nn import  ActivationFunction
from eventloop import Event, Listener
from eventloop.events import Iteration
from helpers.functions import kph_to_mps
from renderer import Renderer
from renderer.events import FrameRendered
from simulation.environment.events import UpdateRequest, Tick
from simulation.location import Circle, CircleSpace
from timer.timer import Timer

import logging

logging.getLogger().setLevel(logging.INFO)
logging.info('initializing...')


L = 500
T = 60
CARS = 10
# ARCHITECTURE = NetworkArchitecture([DenseLayerArchitecture(4, ActivationFunction.SIGM)] * 1, ActivationFunction.SIGM)
ARCHITECTURE = NetworkArchitecture([], ActivationFunction.SIGM)

cars = [NCar(NeuralNetwork.random(ARCHITECTURE), Circle(CircleSpace(L), -
             i * (L // CARS)), spec=specs.TEST, name=f"car-{i+1}") for i in range(CARS)]

logging.info('estimating...')
start = time()
fitness = autosim.fitness(cars[0], autosim.SimulationParameters(T, [*cars]), autosim.EstimationStrategy(
    collision=autosim.Criteria(1000),
    speed=autosim.ReferenceCriteria(10, kph_to_mps(60)),
    distance=autosim.ReferenceCriteria(10, 20)
))
end = time()
sim_time = end - start

logging.info(f"Estimated {T}s in {sim_time:.2f} real seconds, target fitness: {fitness}")
