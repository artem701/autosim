from enum import IntEnum, auto
from autosim.car.car import Car
from cexprtk import Expression, Symbol_Table
from simulation.moveable.events import Move


class ACar(Car):
    class Mode(IntEnum):
        COORDINATE = auto()
        ACCELERATION = auto()

    def __init__(self, function, mode: Mode = Mode.ACCELERATION):
        self.mode = mode
        if type(function) is str:
            st = Symbol_Table({'t': 0.0}, add_constants=True)
            expression = Expression(function, st)

            def f(t: float):
                st['t'] = t
                return expression()
            self.function = f
        else:
            self.function = function

    def update(self, tick):
        if self.mode == ACar.Mode.COORDINATE:
            return Move(self.function(tick.time + tick.dt))
        elif self.mode == ACar.Mode.ACCELERATION:
            return self.accelerate(self.function(tick.time))
        else:
            raise RuntimeError(f"Unhandeled mode {self.mode.name}")
