
from typing import Callable
from dataclasses import dataclass


class Event:
    def __init__(self, sender):
        self.sender = sender


@dataclass
class Listner:
    handle: Callable[[Event], list[Event]]


class EventLoop:

    def __init__(self):
        self._listners = set[Listner]()
        self._queue = list[Event]()

    def subscribe(self, listner: Listner):
        self._listners.add(listner)

    def put(self, event: Event):
        self._queue.put(event)

    def iterate(self):
        q = list[Event]()
        for e in self._queue:
            for l in self._listners:
                q.append(l.handle(e))
        self._queue = q

    def loop_until_empty(self):
        while len(self._queue) > 0:
            self.iterate()
