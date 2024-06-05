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
    ICE = 0.02
    ASPHALT = 0.015
    STONE = 0.02
    GROUND = 0.03
    SAND = 0.15


class Car(Body):

    AIR_DENSITY = 1.25
    g = 9.8

    def __init__(self, location: Location = Line(0), spec: specs.Characteristics = specs.TEST, f: Friction = Friction.ASPHALT, name: str = None):
        super().__init__(location=location, mass=spec.mass, name=name)
        if f >= spec.mbreak:
            raise ValueError(
                f"f is expected to be less than mbreak = {spec.mbreak}")
        self.spec = spec
        self.f = f

        self.N = spec.mass * Car.g
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

    def get_force(self, d: float):
        d_pos = d if d > 0 else 0
        d_neg = -d if d < 0 else 0

        def force(t, x, v):
            # ref https://studref.com/596310/tehnika/sily_deystvuyuschie_avtomobil_pryamolineynom_dvizhenii
            F_thrust = d_pos * self.spec.thrust
            F_frict = self.f * (1 + (v**2) / 1500) * self.N
            F_break = d_neg * self.spec.mbreak * self.N * self.f / Friction.ASPHALT
            F_air = self.spec.front_area * self.spec.streamlining * self.AIR_DENSITY * (v**2)
            
            result = F_thrust - F_frict - F_break - F_air
            
            return result

        return force
        
    def accelerate(self, d: float, dt: float) -> Move:
        return self.push(dt, self.get_force(d))

    @not_implemented
    def update(self, environment: Environment) -> Event:
        pass

    def on_collision(self, collision: Collision):
        return RemoveListener(self)

    def next(self, environment) -> Body:
        return self._next.get(curry(super().next, environment))
