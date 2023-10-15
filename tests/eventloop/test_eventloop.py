
from eventloop import EventLoop, Event, Listener
from eventloop.events import Iteration, Terminate
import pytest


class Terminator(Listener):
    def __init__(self, countdown, immediate):
        self.countdown = countdown
        self.immediate = immediate

    def input_events(self):
        return {Iteration}

    def accept(self, event):
        assert isinstance(event, Iteration)
        product = None
        self.countdown -= 1
        if self.countdown == 0:
            product = {Terminate(self.immediate)}
        return product


@pytest.mark.timeout(0.1)
@pytest.mark.parametrize('countdown,immediate', [
    (1, True),
    (1, False),
    (2, True),
    (2, False),
    (10, True),
    (10, False),
])
def test_terminate(countdown, immediate):
    terminator = Terminator(countdown, immediate)
    loop = EventLoop()
    loop.subscribe(terminator)
    loop.loop()
    assert terminator.countdown == 0


class Loopback(Event):
    pass


class Watcher(Listener):
    def __init__(self):
        self.count = 0

    def input_events(self):
        return {Iteration, Loopback}

    def accept(self, event):
        if isinstance(event, Iteration):
            return {Loopback()}
        if isinstance(event, Loopback):
            self.count += 1


@pytest.mark.timeout(0.1)
@pytest.mark.parametrize('countdown,immediate,expected_count', [
    (1,  True,   0),
    (1,  False,  1),
    (2,  True,   1),
    (2,  False,  2),
    (10, True,   9),
    (10, False, 10),
])
@pytest.mark.timeout(0.1)
def test_terminate_immediate_tw(countdown, immediate, expected_count):
    terminator = Terminator(countdown, immediate)
    watcher = Watcher()
    loop = EventLoop()
    loop.subscribe(terminator)
    loop.subscribe(watcher)
    loop.loop()
    assert watcher.count == expected_count


@pytest.mark.timeout(0.1)
@pytest.mark.parametrize('countdown,immediate,expected_count', [
    (1,  True,   1),
    (1,  False,  1),
    (2,  True,   2),
    (2,  False,  2),
    (10, True,  10),
    (10, False, 10),
])
@pytest.mark.timeout(0.1)
def test_terminate_immediate_wt(countdown, immediate, expected_count):
    watcher = Watcher()
    terminator = Terminator(countdown, immediate)
    loop = EventLoop()
    loop.subscribe(watcher)
    loop.subscribe(terminator)
    loop.loop()
    assert watcher.count == expected_count
