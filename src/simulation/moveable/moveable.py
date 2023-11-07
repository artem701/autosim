
from simulation import Object, Location
from simulation.moveable.events import Move


class Moveable(Object):
    def __init__(self, location: Location, name: str = None):
        super().__init__(name=name)
        self.location = location

    def move(self, dx: float) -> Move:
        return Move(dx)
    
    def __str__(self):
        return f"{self.location}"
