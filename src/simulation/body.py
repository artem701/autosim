
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
        return Move(xv[0])

    # @not_implemented
    # def next(self, environment: Environment):
    #     # TODO: test, finish implementation
    #     def getter():
    #         nonlocal environment
    #         i = 0
    #         min_distance = None
    #         min_distance_index = None
    #         for body in environment.bodies:
    #             if body is self:
    #                 return environment.bodies[(i + 1) % len(environment.bodies)]
    #             distance = self.location.distance(body.location)
    #             if distance >= 0 and distance < coalesce(min_distance, distance + 1):
    #                 min_distance = distance
    #                 min_distance_index = i
    #             i += 1
    #         if min_distance_index is None:
    #             return self
    #         else:
    #             return environment.bodies[min_distance_index]
    #     next = self._next.get(getter=getter) 
    #     if next is None:
    #         self._next.is_set = False
    #     return next