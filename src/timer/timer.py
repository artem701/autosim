
import logging
from eventloop import EventLoop, Listener, Event
from eventloop.events import Iteration, RemoveListener
from timer.events import Tick
from time import time
from math import floor


class Timer(Listener):
    def __init__(self, timeout: float, oneshot: bool = True, accumulate=None, event: type(Event) = Tick, ctx=None):
        if accumulate is None:
            accumulate = not oneshot
        if accumulate and oneshot:
            raise ValueError("Oneshot timer cannot accumulate ticks")

        self.timeout = timeout
        self.oneshot = oneshot
        self.accumulate = accumulate
        self.event = event
        self.ctx = ctx
        self._start = time()
        self._active = True
        self._set_when()

    def input_events(self):
        return {Iteration}

    def accept(self, event):
        assert isinstance(event, Iteration)
        product = EventLoop.EventsQueue()
        t = time()
        logging.debug(f"active: {self._active}, time: {t}, when: {self._when}")
        if self._active and t >= self._when:
            ticks = floor((t - self._when) /
                          self.timeout) if self.accumulate else 1
            production = self.event(
                self.ctx) if self.ctx is not None else self.event()
            product += [production] * ticks
            if self.oneshot:
                self._active = False
                product += [RemoveListener(self)]
            else:
                self._set_when()
        return product

    def _set_when(self):
        self._when = self._start + \
            floor((time() - self._start) / self.timeout + 1) * self.timeout
