
import logging
import autosim
import autosim.car

from enum import IntEnum, auto
from eventloop.events import Event, Terminate
from helpers import measure_time
from helpers.functions import mps_to_kph
from renderer.renderer import Renderer
from simulation import Environment
from simulation.location import Circle, CircleSpace

class State(IntEnum):
    REACH_MAX_SPEED_1 = auto()
    RELEASE_UNTIL_STOP = auto()
    REACH_MAX_SPEED_2 = auto()
    BREAK_UNTIL_STOP = auto()
    TERMINATE = auto()

class TestCar(autosim.car.Car):
    
    def __init__(self):
        super().__init__(location=Circle(CircleSpace(1000)),
                         spec=autosim.car.specs.TEST,
                         f=autosim.car.Friction.ASPHALT,
                         name='car')
        self.state = State.REACH_MAX_SPEED_1
        self.v_previous = None
        self.t_start = None
        self.breaking_distance = None
    
    def update(self, environment: Environment) -> Event:
        EPS = 0.001 # 1 mm/s
        if self.state in {State.REACH_MAX_SPEED_1, State.REACH_MAX_SPEED_2}:
            if self.t_start is None:
                self.t_start = environment.time

            if self.v_previous is not None and abs(self.v - self.v_previous) <= EPS:
                acc_time = environment.time - self.t_start
                self.t_start = None
                if self.state == State.REACH_MAX_SPEED_1:
                    logging.info(f"max speed = {self.v:.2f} m/s ({mps_to_kph(self.v):.2f} km/h), reach in {acc_time:.2f} s")
                self.state += 1 
                self.v_previous = None
            else:
                self.v_previous = self.v
                self.d = 1

        if self.state == State.RELEASE_UNTIL_STOP:
            if self.t_start is None:
                self.t_start = environment.time
            if self.breaking_distance is None:
                self.breaking_distance = 0

            if self.v <= EPS:
                dec_time = environment.time - self.t_start
                self.t_start = None
                logging.info(f"release til 0 in {dec_time:.2f} s, breaking distance: {self.breaking_distance:.2f} m")
                self.breaking_distance = None
                self.state += 1
            else:
                self.d = 0

        if self.state == State.BREAK_UNTIL_STOP:
            if self.t_start is None:
                self.t_start = environment.time
            if self.breaking_distance is None:
                self.breaking_distance = 0

            if self.v <= EPS:
                dec_time = environment.time - self.t_start
                self.t_start = None
                logging.info(f"break til 0 in {dec_time:.2f} s, breaking distance: {self.breaking_distance:.2f} m")
                self.breaking_distance = None
                self.state += 1
            else:
                self.d = -1
    
        if self.state == State.TERMINATE:
            del self.d
            return Terminate(immediate=False)

        move = self.accelerate(self.d, environment.dt)

        if self.breaking_distance is not None:
            self.breaking_distance += move.dx

        return move

@measure_time
def main():
    logging.getLogger().setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(logging.Formatter(fmt='%(levelname)s\t%(message)s'))


    FPS=24
    W=768
    H=768

    car = TestCar()
    renderer = Renderer(fps=FPS, width=W, height=H)
    simulation = autosim.SimulationParameters(timeout=600, objects=[car, renderer])

    environment = simulate(parameters=simulation)
    logging.info(f"simulated {environment.time:.2f} seconds ~ {round(environment.time / 60)} minutes")
    render(renderer=renderer)

@measure_time
def simulate(parameters):
    return autosim.simulate(parameters=parameters)
    
@measure_time
def render(renderer):
    renderer.render('output')

if __name__ == '__main__':
    main()
