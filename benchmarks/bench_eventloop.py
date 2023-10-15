
from eventloop import Event, EventLoop, Listener
from eventloop.events import Iteration, Terminate


class Countdown(Listener):
    def __init__(self, count: int):
        self.count = count

    def input_events(self):
        return Iteration

    def accept(self, event: Event) -> list[Event]:
        self.count -= 1
        if self.count <= 0:
            return Terminate(True)


def bench(count):
    loop = EventLoop()
    loop.subscribe(Countdown(count))
    loop.loop()


def iteration_1_listener_1_000():
    bench(1_000)


def iteration_1_listener_10_000():
    loop = EventLoop()
    loop.subscribe(Countdown(10_000))
    loop.loop()


def iteration_10_000_listener_100():
    loop = EventLoop()
    for _ in range(100):
        loop.subscribe(Countdown(10_000))
    loop.loop()


__benchmarks__ = [
    (iteration_1_listener_1_000, "1'000 iterations, 1 listener"),
    (iteration_1_listener_10_000, "10'000 iterations, 1 listener"),
    (iteration_10_000_listener_100, "10'000 iterations time, 100 listeners"),
]
