
import logging
from eventloop import EventLoop, Listener, Event
from eventloop.events import Iteration, RemoveListener
from timer.events import Tick
from time import time
from math import floor


class Timer(Listener):
    def __init__(self, timeout: float, oneshot: bool = True, accumulate=None, event: type(Event)=Tick, args=[], kwargs={}, time_source=time, trigger_event=Iteration):
        if accumulate is None:
            accumulate = not oneshot
        if accumulate and oneshot:
            raise ValueError("Oneshot timer cannot accumulate ticks")

        self.timeout = timeout
        self.oneshot = oneshot
        self.accumulate = accumulate
        self.event = event
        self.args = args
        self.kwargs = kwargs
        self.time_source = time_source
        self.trigger_event = trigger_event

    def start(self):
        self._start = self.time_source()
        self._active = True
        self._set_when(self._start)
        

    def input_events(self):
        return self.trigger_event

    def accept(self, event):
        assert isinstance(event, self.trigger_event)
        product = EventLoop.EventsQueue()
        t = self.time_source()
        if self._active and t >= self._when:
            ticks = floor((t - self._when) /
                          self.timeout) if self.accumulate else 1
            production = self.event(*self.args, **self.kwargs)
            product += [production] * ticks
            if self.oneshot:
                self._active = False
                product = product + [RemoveListener(self)]
            else:
                self._set_when(t)
        return product

    def _set_when(self,current_time):
        if self.timeout == 0:
            self._when = current_time
        else:
            self._when = self._start + \
                floor((current_time - self._start) /
                      self.timeout + 1) * self.timeout
