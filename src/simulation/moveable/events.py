
from eventloop import Event


class Move(Event):
    def __init__(self, dx: float):
        self.dx = dx
