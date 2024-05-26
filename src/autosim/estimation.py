
import autosim.car
import autosim.simulation
import eventloop
import helpers
import simulation.environment.events
import simulation.location
import logging

from dataclasses import dataclass, field
from eventloop.eventloop import Event, Terminate
from simulation.environment.events import Collision, Tick
from typing import Callable

@dataclass
class Criteria:
    fine: float
    def get_fine(self):
        return self.fine
    def __add__(self, other):
        return LambdaCriteria(lambda *args: self.get_fine(*args) + other.get_fine(*args))

class LambdaCriteria(Criteria):
    fine: Callable
    def get_fine(self, *args):
        return self.fine(*args)

@dataclass
class ValueCriteria(Criteria):
    def get_fine(self, value):
        return super().get_fine()

@dataclass
class ReferenceCriteria(ValueCriteria):
    reference: float
    def get_fine(self, value: float):
        return super().get_fine(value) * abs(value - self.reference)

@dataclass
class LessCriteria(ReferenceCriteria):
    def get_fine(self, value: float):
        return super().get_fine(value) if not value < self.reference else 0 

@dataclass
class MoreCriteria(ReferenceCriteria):
    def get_fine(self, value: float):
        return super().get_fine(value) if not value > self.reference else 0 

@dataclass
class EstimationStrategy:
    collision: Criteria = field(default_factory=lambda:Criteria(0))
    speed: ValueCriteria = field(default_factory=lambda:ReferenceCriteria(0, 0))
    distance: ValueCriteria = field(default_factory=lambda:ReferenceCriteria(0, 0))

class EstimatorObject(eventloop.Listener):
    def __init__(self, target: autosim.car.Car, strategy: EstimationStrategy):
        self.strategy = strategy
        self.target = target
        self.first = True
        self.fine = 0
    
    def input_events(self):
        return Tick, Collision

    def accept(self, event: Event):
        if isinstance(event, Tick):
            self.fine += self.periodical_fine(event.environment)

        if isinstance(event, Collision):
            if event.collider is not self.target:
                return

            assert not self.first
            self.fine += self.strategy.collision.get_fine()
            return Terminate(immediate=False)

    def periodical_fine(self, env: simulation.Environment):
        if self.first:
            self.dt = env.dt
            self.first = False
            return 0

        fine = 0
        fine += self.strategy.speed.get_fine(self.target.v)

        front = self.front(env)
        if front is None:
            return fine

        distance = self.target.location.distance(front.location)

        fine += self.strategy.distance.get_fine(distance)
        fine *= self.dt
        return fine

    def front(self, env: simulation.Environment) -> simulation.Body:
        # TODO: optimize. Can remember front and then recalc on add/remove listener.
        # consider implementation in Body class to use inside both estimation and decision.

        target_index = helpers.index_if(env.bodies, lambda body: body is self.target)
        if target_index is None:
            return None

        front_index = (target_index + 1) % len(env.bodies)
        front = env.bodies[front_index]

        if front is self.target:
            return None
        
        return front
