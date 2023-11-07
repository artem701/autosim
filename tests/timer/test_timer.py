from math import ceil
from eventloop import EventLoop, Listener, Event
from eventloop.eventloop import Iteration
from eventloop.events import Terminate, AddListener
from timer.timer import Timer
from timer.events import Tick
from time import time
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


@pytest.mark.timeout(1.0)
@pytest.mark.parametrize('timeout', [
    (0.01),
    (0.10),
    (0.15),
    (0.20),
    (0.50),
])
def test_basic(timeout):
    # 0.1 us
    watcher = Watcher(timeout)
    loop = EventLoop()
    loop.subscribe(watcher)
    loop.put(StartTimer())
    loop.loop()
    assert watcher.time > timeout


def test_zero():
    watcher = Watcher(0)
    loop = EventLoop()
    loop.subscribe(watcher)
    loop.put(StartTimer())
    assert loop.iterate()
    assert not loop.iterate()
