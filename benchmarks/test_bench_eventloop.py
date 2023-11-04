
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


def bench(iteration, listener):
    loop = EventLoop()
    for _ in range(listener):
        loop.subscribe(Countdown(iteration))
    loop.loop()


def test_iteration_1_listener_1_000(benchmark):
    benchmark(bench, 1, 1_000)

def test_iteration_1_listener_10_000(benchmark):
    benchmark(bench, 1, 10_000)

def test_iteration_10_000_second_100_listener_100(benchmark):
    benchmark(bench, 10_000, 100)

def test_iteration_100_second_1_listener_100(benchmark):
    benchmark(bench, 100, 100)
