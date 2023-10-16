
from simulation import Object
from simulation import Location
from simulation.moveable.events import Move


class Moveable(Object):
    def __init__(self, location: Location):
        self.location = location

    def move(self, dx: float) -> Move:
        return Move(dx)
