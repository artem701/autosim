
from helpers import Cached, coalesce
from simulation import Moveable
from simulation import Location
from simulation.math.rk2a import rk2a
from simulation.moveable.events import Move
from typing import Callable
import numpy


class Body(Moveable):
    def __init__(self, location: Location, mass: float, v: float = 0, name: str = None):
        super().__init__(location, name=name)
        self.mass = mass
        self.v = v
        # self._next = Cached[Body]

    def move(self, dt) -> Move:
        return Move(dt * self.v)

    def push(self, dt: float, force: Callable[[float, float, float], float]) -> Move:
        """Push object for dt time with force, depending on (t, x, v).
        """
        def x_v_diff(xv, t):
            return numpy.asarray([
                # x'
                xv[1],
                # v'
                force(t, xv[0], xv[1]) / self.mass
            ])

        xv = rk2a(x_v_diff, [0, self.v], [0, dt])[-1]
        self.v = xv[1]
        return Move(max(0, xv[0]))

    def next(self, environment) -> 'Body':
        return self.next_in(environment.bodies)