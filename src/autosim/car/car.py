from dataclasses import dataclass
from eventloop.eventloop import AddListener, RemoveListener, Event, RemoveListener
from simulation import Body
from simulation.location import Location, Line
from simulation import Environment
from simulation.environment.events import Tick, Collision
from simulation.moveable.events import Move
from helpers import not_implemented, curry, Cached
from simulation.math.rk2a import rk2a
import autosim.car.specs as specs


@dataclass
class Friction:
    ASPHALT = 0.0015
    STONE = 0.02
    GROUND = 0.03
    SAND = 0.15


class Car(Body):

    AIR_DENSITY = 1.25
    g = 9.8
    F_BREAK = 0.05

    def __init__(self, location: Location = Line(0), spec: specs.Characteristics = specs.TEST, f: Friction = Friction.ASPHALT, name: str = None):
        super().__init__(location=location, mass=spec.mass, name=name)
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

        self._next = Cached[Body]()

    def input_events(self) -> set:
        return [Tick, Collision, AddListener, RemoveListener]

    def accept(self, event: Event) -> list[Event]:

        if isinstance(event, Tick):
            return self.update(event.environment)

        if isinstance(event, Collision) and event.collider is self:
            return self.on_collision(event)
        
        if isinstance(event, AddListener) and isinstance(event.listener, Body):
            self._next.unset()
        
        if isinstance(event, RemoveListener) and isinstance(event.listener, Body):
            self._next.unset()

        
    def accelerate(self, d: float, dt: float) -> Move:
        d_pos = d if d > 0 else 0
        d_neg = -d if d < 0 else 0

        def force(t, x, v):
            # TODO: ensure formula is right, fix if needed, check if "v * ..." is needed in friction section
            result = d_pos * self.spec.thrust \
                - v * (self.__F_friction + d_neg * (self.__F_break_max - self.__F_friction)) \
                - self.__K_air * (v ** 2)
            return result

        return self.push(dt, force)

    @not_implemented
    def update(self, environment: Environment) -> Move:
        pass

    def on_collision(self, collision: Collision):
        return RemoveListener(self)

    def next(self, environment) -> Body:
        return self._next.get(curry(super().next, environment))
