
from time import time
import autosim
from autosim.car import specs
from autosim.car.ncar import NCar, NeuralNetwork, NetworkArchitecture
from autosim.nn import  DenseLayerArchitecture, ActivationFunction
from eventloop import Event, EventLoop, Listener
from eventloop.events import Iteration
from helpers.functions import kph_to_mps
from renderer import Renderer
from renderer.events import FrameRendered
from simulation import Environment
from simulation.environment.events import UpdateRequest, Tick
from simulation.location import Circle, CircleSpace
from timer.timer import Timer

import logging

logging.getLogger().setLevel(logging.INFO)
logging.info('initializing...')


class Watcher(Listener):
    def __init__(self, logging_interval=1):
        self.last_time = 0
        self.iterations = 0
        self.logging_interval = logging_interval

    def input_events(self) -> set:
        return Iteration, Tick, FrameRendered

    def accept(self, event: Event) -> list[Event]:
        if isinstance(event, Iteration):
            self.iterations += 1

        if isinstance(event, Tick):
            if self.logging_interval > 0 and event.environment.time >= self.last_time + self.logging_interval:
                logging.info(f"time: {event.environment.time:.2f} seconds")
                self.last_time += self.logging_interval

        if isinstance(event, FrameRendered):
            pass

L = 500
T = 60
FPS = 24
W = 768
H = 768
CARS = 10
# ARCHITECTURE = NetworkArchitecture([DenseLayerArchitecture(4, ActivationFunction.SIGM)] * 1, ActivationFunction.SIGM)
ARCHITECTURE = NetworkArchitecture([], ActivationFunction.SIGM)

cars = [NCar(NeuralNetwork.random(ARCHITECTURE), Circle(CircleSpace(L), -
             i * (L // CARS)), spec=specs.TEST, name=f"car-{i+1}") for i in range(CARS)]
renderer = Renderer(fps=FPS, width=W, height=H)
watcher = Watcher(10)

logging.info('estimating...')
start = time()
fitness = autosim.fitness(cars[0], autosim.SimulationParameters(T, [*cars, renderer, watcher]), autosim.EstimationStrategy(
    collision=autosim.Criteria(1000),
    speed=autosim.ReferenceCriteria(10, kph_to_mps(60)),
    distance=autosim.ReferenceCriteria(10, 20)
))
end = time()
sim_time = end - start

logging.info('rendering...')
start = time()
renderer.render('output')
end = time()
render_time = end - start

logging.info(
    f"Simulated {T}s ({FPS*T} {W}x{H} frames, {watcher.iterations} iterations) in {sim_time:.2f} real seconds, render for {render_time:.2f} seconds, target fitness: {fitness}")
