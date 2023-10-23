
from eventloop import EventLoop, Listener, Event
from eventloop.events import Terminate, AddListener
from timer import Timer
from timer.events import Tick
from time import time
import logging
import pytest


class StartTimer(Event):
    pass


class Watcher(Listener):
    def __init__(self, timeout):
        self.timeout = timeout
        self.start = None
        self.end = None
        self.time = None

    def input_events(self):
        return {StartTimer, Tick}

    def accept(self, event):
        if isinstance(event, StartTimer):
            self.start = time()
            timer = Timer(self.timeout)
            logging.info(
                f"Register timer. Current time: {self.start:10f}, timer ring: {timer._when}")
            return AddListener(timer)

        assert isinstance(event, Tick)
        assert self.start is not None

        self.end = time()
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
    EPSILON = 0.0001
    watcher = Watcher(timeout)
    loop = EventLoop()
    loop.subscribe(watcher)
    loop.put(StartTimer())
    loop.loop()
    assert abs(watcher.time - timeout) < EPSILON


def test_zero():
    watcher = Watcher(0)
    loop = EventLoop()
    loop.subscribe(watcher)
    loop.put(StartTimer())
    assert loop.iterate()
    assert not loop.iterate()
