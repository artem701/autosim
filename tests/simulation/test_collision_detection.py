import logging
import pytest

from autosim.car import ACar
from eventloop.eventloop import Event, Terminate
from eventloop.events import Iteration
import simulation
from simulation.location import Circle, CircleSpace, Line
from simulation import Environment
from simulation.environment.events import Tick, Collision
from simulation.object import Object


def mps(speed):
    return {'function': f"{speed} * dt", 'mode': ACar.Mode.MOVEMENT}


def car(speed, location):
    return ACar(location=location, **mps(speed))


class Watcher(Object):

    def __init__(self, carl, carr, sim_time, log):
        self.carl = carl
        self.carr = carr
        self.need_log = log
        self.sim_time = sim_time
        self.collider = None
        self.collidee = None

    def log(self, *args):
        if self.need_log:
            logging.info(args)

    def input_events(self):
        return [Tick, Collision]

    def accept(self, event: Event):

        if isinstance(event, Tick):
            self.log(
                f"time: {event.environment.time:.2f}, lx: {self.carl.location.x():.2f}, rx: {self.carr.location.x():.2f}")
            if event.environment.time >= self.sim_time:
                self.log("simulation time exceeded, terminate")
                return Terminate(False)

        if isinstance(event, Collision) and event.time < self.sim_time:
            self.log("collision, terminate")
            self.collider = event.collider
            self.collidee = event.collidee
            return Terminate(True)


def simulate(carl, carr, sim_time):

    environment = Environment(driver=simulation.Driver(type=simulation.Driver.Type.FAST))
    watcher = Watcher(carl, carr, sim_time, False)
    environment.subscribe(watcher, carr, carl)
    environment.simulate()

    assert (watcher.collider is None and watcher.collidee is None) or (
        watcher.collider is not watcher.collidee)

    return watcher.collider


def perform_testing(carl, carr, sim_time, expected_collider):
    if expected_collider == 'l':
        expected_collider = carl
    if expected_collider == 'r':
        expected_collider = carr
    assert simulate(carl, carr, sim_time) is expected_collider


@pytest.mark.parametrize(
    'posl,speedl,posr,speedr,sim_time,collider',
    [
        # Both cars are far from each other
        #                                ...and their trajectories never intersect during one update.
        (0, 50, 100, 100,  3, None),
        #                                ...but their trajectories touch during first update.
        (0, 50,  50, 100,  3, None),
        #                                ...but their trajectories intersect during first update.
        (0, 50,  10, 100,  3, None),
        #                                ...but their trajectories intersect during each update.
        (0, 50,  10,  50,  3, None),
        #                                ...and their trajectories meet
        #                                                            ...but not during simulation.
        (0, 50,  50,  45,  9, None),
        #                                                            ...at the very end of simulation.
        # TODO: 10.1 must be 10. This is workaround for floating precision. Need to fix.
        (0, 50,  50,  45, 10.1, 'l'),
        #                                                            ...in the middle of simulation.
        (0, 90,  50,  45,    5, 'l'),

        # Fast approach.
        (0, 10000, 50,  50, 10, 'l'),
    ]
)
def test_line(posl, speedl, posr, speedr, sim_time, collider):
    carl = car(speedl, Line(posl))
    carr = car(speedr, Line(posr))
    perform_testing(carl, carr, sim_time, collider)


@pytest.mark.parametrize(
    'posl,speedl,posr,speedr,sim_time,collider',
    [
        # Both cars are far from each other
        #                                ...and their trajectories never intersect during one update.
        (0, 450, 500, 450,  3, None),
        #                                ...but their trajectories touch during first update.
        (0, 500, 500, 500,  3, None),
        #                                ...but their trajectories intersect during each update.
        (0, 900, 500, 900,  3, None),
        #                                ...and their trajectories meet
        #                                                            ...but not during simulation.
        (0, 500, 500, 450,  9, None),
        #                                                            ...at the very end of simulation.
        # TODO: 10.1 must be 10. This is workaround for floating precision. Need to fix.
        (0, 500, 500, 450, 10.1, 'l'),
        #                                                            ...in the middle of simulation.
        (0, 500, 500, 450, 15, 'l'),
        #                                                            ...in the middle of simulation, but collider is not carl.
        (0, 500, 500, 550, 15, 'r'),
    ]
)
def test_circle(posl, speedl, posr, speedr, sim_time, collider):
    carl = car(speedl, Circle(CircleSpace(1000), posl))
    carr = car(speedr, Circle(CircleSpace(1000), posr))
    perform_testing(carl, carr, sim_time, collider)
