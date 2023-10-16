
from helpers import coalesce, not_implemented
from eventloop import Listener
from simulation.object import Object
from simulation.location import Path
from simulation.environment.events import Tick
from simulation.moveable.events import Move


class Environment(Listener):

    DEFAULT_DT = 0.01

    Objects = set[Object]

    def __init__(self, dt=None):
        self._dt = coalesce(dt, Environment.DEFAULT_DT)
        self._time = 0
        self._objects = Environment.Objects()

    def input_events(self):
        return {Tick, Move}

    @not_implemented
    def accept(self, event):
        pass

    def time(self) -> float:
        return self._time

    def objects(self):
        return self._objects
