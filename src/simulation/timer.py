
from eventloop import Event
from simulation import Environment
from simulation.environment.events import Tick
import timer.timer as timer


class Timer(timer.Timer):
    def __init__(self, environment: Environment, timeout: float, oneshot: bool = True, accumulate=None, event: type(Event) = timer.Tick, args=[], kwargs={}):
        super().__init__(timeout, oneshot, accumulate, event, args, kwargs, time_source=lambda: environment.time, trigger_event=Tick)
