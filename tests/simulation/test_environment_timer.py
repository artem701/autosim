
from math import ceil
from eventloop import EventLoop, Listener, Event
from eventloop.events import Terminate, AddListener
from simulation import Driver
from simulation.timer import Timer
from timer.events import Tick
from simulation import Environment
import pytest


class StartTimer(Event):
    pass


class Watcher(Listener):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.start = None
        self.end = None
        self.time = None

    def input_events(self):
        return {StartTimer, Tick}

    def accept(self, event):
        if isinstance(event, StartTimer):
            self.timer = Timer(*self.args, **self.kwargs)
            self.start = self.timer.time_source()
            self.timer.start()
            return AddListener(self.timer)

        assert isinstance(event, Tick)
        assert self.start is not None

        self.end = self.timer.time_source()
        self.time = self.end - self.start
        return Terminate(True)

@pytest.mark.parametrize('timeout_dts', [
    (0.0), (1.0), (2.0), (10.0),
    (0.5), (1.5), (2.5), (10.5),
    (0.1), (1.1), (2.1), (10.1),
    (0.9), (1.9), (2.9), (10.9),
])
@pytest.mark.timeout(1.0)
def test(timeout_dts):
    driver = Driver(Driver.Type.FAST)
    environment = Environment(driver=driver)
    watcher = Watcher(environment=environment, timeout=environment.dt*timeout_dts)
    environment.subscribe(watcher)
    environment.put(StartTimer())
    environment.simulate()
    assert watcher.time == environment.dt * ceil(timeout_dts)
