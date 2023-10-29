
from enum import Enum
from eventloop import Event, Listener
from eventloop.eventloop import Iteration
from helpers.functions import not_implemented
from simulation.environment.events import UpdateRequest


class Type(Enum):
    FAST = 'fast'
    REALTIME = 'realtime'

class Driver(Listener):
    def __init__(self, driver_type: Type = Type.FAST):
        if driver_type == Type.FAST:
            self.init_fast()
        elif driver_type == Type.REALTIME:
            self.init_realtime()
        else:
            raise RuntimeError(f"Unknown driver type {driver_type}")
    
    @not_implemented
    def init_realtime(self):
        pass

    def init_fast(self):
        self.handler = lambda _: UpdateRequest
    
    def input_events(self) -> set:
        return Iteration

    def accept(self, event: Event) -> list[Event]:
        return self.handler(event)
