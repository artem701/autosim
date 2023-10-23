from dataclasses import dataclass
from eventloop.eventloop import Event
from simulation import Body
from simulation.environment.events import Tick
from simulation.moveable.events import Move
from helpers import not_implemented
from simulation.math.rk2a import rk2a
import autosim.car.specs as specs


@dataclass
class Characteristics:
    mass: float
    thrust: float
    front_area: float
    streamlining: float


@dataclass
class Friction:
    ASPHALT = 0.015
    STONE = 0.02
    GROUND = 0.03
    SAND = 0.15


class Car(Body):

    AIR_DENSITY = 1.25
    g = 9.8
    F_BREAK = 0.8

    def __init__(self, spec: specs.LADA_GRANTA, f: Friction = Friction.ASPHALT):
        if f >= Car.F_BREAK:
            raise ValueError(
                f"f is expected to be less than f_break = {Car.F_BREAK}")
        self.spec = spec
        self.f = f

        N = spec.mass * Car.g
        self.__F_break_max = N * Car.F_BREAK
        self.__F_friction = N * f
        self.__K_air = spec.front_area * \
            spec.streamlining * Car.AIR_DENSITY / 2

        self.v = 0

    def input_events(self) -> set:
        return Tick

    def accept(self, event: Event) -> list[Event]:
        self.update(event)

    def accelerate(self, d: float, dt: float) -> Move:
        d_pos = d if d > 0 else 0
        d_neg = d if d < 0 else 0

        v = self.v

        def force(t, x, v): return d_pos * self.spec.thrust \
            - (self.__F_friction - d_neg * (self.__F_break_max - self.__F_friction)) \
            - self.__K_air * (v ** 2)

        return self.push(dt, force)

    @not_implemented
    def update(self, tick: Tick):
        pass
