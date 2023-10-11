
from eventloop import Listener
from eventloop.events import Iteration, RemoveListener
from timer.events import Tick
from time import time
from math import floor


class Timer(Listener):
    def __init__(self, timeout: float, oneshot: bool = True, accumulate = None):
        if accumulate is None:
            accumulate = not oneshot
        if accumulate and oneshot:
            raise ValueError("Oneshot timer cannot accumulate ticks")

        self.timeout = timeout
        self.oneshot = oneshot
        self.accumulate = accumulate
        self._start = time()
        self._active = False
        self._set_when()

    def inputEvents(self):
        return {Iteration}

    def accept(self, event):
        product = set()
        t = time()
        if self._active and t >= self._when:
            ticks = floor((t - self._when) /
                          self.timeout) if self.accumulate else 1
            product.add([Tick()] * ticks)
            if self.oneshot:
                self._active = False
                product.add(RemoveListener(self))
            else:
                self._set_when()
        return product

    def _set_when(self):
        self._when = self._start + \
            floor((time() - self._start) / self.timeout + 1) * self.timeout
