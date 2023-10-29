
import autosim.car
import autosim.simulation
import eventloop
import helpers
import simulation.environment.events
import simulation.location

from dataclasses import dataclass
from eventloop.eventloop import Event
from simulation.environment.events import Collision, Tick

@dataclass
class Criteria:
    fine: float
    def get_fine(self):
        return self.fine

@dataclass
class ReferenceCriteria(Criteria):
    reference: float
    def get_fine(self, value: float):
        return super().get_fine() * abs(value - self.reference)

@dataclass
class Strategy:
    collision: Criteria
    speed: ReferenceCriteria
    distance: ReferenceCriteria

class EstimatorObject(eventloop.Listener):
    def __init__(self, target: autosim.car.Car, strategy: Strategy):
        self.strategy = strategy
        self.target = target
        self.first = True
        self.fine = 0
    
    def input_events(self):
        return Tick, Collision

    def accept(self, event: Event):
        if isinstance(event, Tick):
            env = event.environment
            if self.first:
                self.dt = env.dt
                self.first = False
            else:
                self.fine += self.periodical_fine(env)

        if isinstance(event, Collision):
            assert not self.first
            self.fine += self.strategy.collision.get_fine()

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
        if len(env.bodies) == 1:
            return None

        target_index = helpers.index_if(env.bodies, lambda body: body is self.target)
        if target_index is None:
            return None

        front_index = (target_index + 1) % len(env.bodies)
        front = env.bodies[front_index]
        return front
