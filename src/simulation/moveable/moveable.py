
from simulation import Object, Location
from simulation.moveable.events import Move


class Moveable(Object):
    def __init__(self, location: Location, name: str = None):
        super().__init__(name=name)
        self.location = location

    def move(self, dx: float) -> Move:
        return Move(dx)
    
    def __str__(self):
        return f"{self.__class__} (Moveable at {self.location})"
    
    def next_in(self, array: list['Moveable']):
        self_index = 0
        for moveable in array:
            if moveable is self:
                break
            self_index += 1
        
        if self_index == len(array):
            return None

        next_index = (self_index + 1) % len(array)
        if next_index == self_index:
            return None
        
        next_candidate = array[next_index]
        if self.location.distance(array[next_index].location) < 0:
            return None
        
        return next_candidate

    def next(self, environment) -> 'Moveable':
        return self.next_in(environment.moveables)
    
    def distance(self, other: 'Moveable'):
        return self.location.distance(other.location)
