
from eventloop import Event


class Tick(Event):
    def __init__(self, time):
        self.time = time
