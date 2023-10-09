
from math import ceil, floor


class Location:

    def __init__(self):
        pass

    def moved(self, dx: float) -> 'Location':
        raise NotImplementedError

    def x(self) -> float:
        raise NotImplementedError
    
    def collides(self, dx: float, other: 'Path') -> bool:
        """Check if current point collides other moving point. Does not check if other collides current!

        Args:
            dx (float): Current point transition.
            other (Path): Other point transition.

        Returns:
            bool: Whether current point collides other during transition.
        """
        raise NotImplementedError


class Path:

    def __init__(self, a: Location, dx: float):
        assert dx >= 0
        self._a = a
        self._b = a.moved(dx)
        self._dx = dx

    def dx(self):
        return self._dx

    def length(self):
        return abs(self._dx)

    def a(self):
        return self._a

    def ax(self):
        return self._a.x()

    def b(self):
        return self._b

    def bx(self):
        return self._b.x()
    
    def collides(self, other: 'Path') -> bool:
        """Whether current path collides other. Does not check reverse!

        Args:
            other (Path): Other path.

        Returns:
            bool: If current collides other.
        """
        return self._a.collides(self._dx, other)


class Line(Location):

    def __init__(self, x: float = 0):
        super().__init__()
        self._x = x
    
    def moved(self, dx):
        return Line(self._x + dx)

    def x(self) -> float:
        return self._x

    def collides(self, dx: float, other: 'Path') -> bool:
        s = self._x - other.ax()
        e = self._x + dx - other.bx()
        return (s < 0 and e > 0) or (e == 0 and s <= 0)

class CircleSpace:

    def __init__(self, length: float):
        self._length = length

    def length(self):
        return self._length

class Circle(Location):
    
    def __init__(self, space: CircleSpace, x: float = 0):
        super().__init__()

        if x >= 0:
            self._x = x - space.length() * floor(x / space.length())
        else:
            self._x = x + space.length() * ceil(-x / space.length())

        self._space = space

    def moved(self, dx: float):
        return Circle(self._space, self._x + dx)

    def x(self) -> float:
        return self._x
    
    def collides(self, dx: float, other: 'Path') -> bool:
        x = self._x

        # TODO: refactor

        if x == other.ax() and dx == other.dx():
            return True

        if x >= other.ax():
            x = x - self._space.length()

        return x + dx >= other.ax() + other.dx()
