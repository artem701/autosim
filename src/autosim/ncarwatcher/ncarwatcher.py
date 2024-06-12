
from eventloop import Listener, Event
from autosim.car import NCar
from simulation.environment.events import Tick
from autosim.ncarwatcher.events import NCarWatcherUpdate

class NCarWatcher(Listener):

    first: bool = True
    cars: list[NCar] = []
    v_sum: float = 0
    u_pos_dt_sum: float = 0
    ticks: int = 0

    def __init__(self, filter=None):
        custom_filter = filter or (lambda obj: True)
        self.filter = lambda obj: isinstance(obj, NCar) and custom_filter(obj)

    def input_events(self) -> set:
        return Tick, 

    def accept(self, event: Event) -> list[Event]:
        
        assert type(event) is Tick

        self.ticks += 1

        if self.first:
            self.first = False
            self.cars = [body for body in event.environment.bodies if self.filter(body)]

        for car in self.cars:
            self.v_sum += car.v
            if car.u > 0:
                self.u_pos_dt_sum += car.u * event.environment.dt
        
        return NCarWatcherUpdate(v_avg=self.v_sum / (self.ticks * len(self.cars)), u_pos_dt_int=self.u_pos_dt_sum)
