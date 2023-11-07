
from eventloop import Event


class Move(Event):
    def __init__(self, dx: float):
        self.dx = dx
    
    def __str__(self):
        return f"Move({self.dx})"
