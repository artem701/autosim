from enum import IntEnum, auto
from autosim.car import specs
from autosim.car.car import Car, Friction
from cexprtk import Expression, Symbol_Table
from simulation.location import Location, Line
from simulation.moveable.events import Move


class ACar(Car):
    class Mode(IntEnum):
        MOVEMENT = auto()
        ACCELERATION = auto()

    def __init__(self, function, mode: Mode = Mode.ACCELERATION, location: Location = Line(0), spec: specs.Characteristics = specs.LADA_GRANTA, f: Friction = Friction.ASPHALT):
        super().__init__(location=location, spec=spec, f=f)
        self.mode = mode
        if type(function) is str:
            st = Symbol_Table({'t': 0.0, 'dt': 0.0}, add_constants=True)
            expression = Expression(function, st)

            def f(t: float, dt: float):
                st.variables['t'] = t
                st.variables['dt'] = dt
                return expression()
            self.function = f
        else:
            self.function = function

    def update(self, environment):
        if self.mode == ACar.Mode.MOVEMENT:
            return Move(self.function(environment.time, environment.dt))
        elif self.mode == ACar.Mode.ACCELERATION:
            return self.accelerate(self.function(environment.time, environment.dt))
        else:
            raise RuntimeError(f"Unhandeled mode {self.mode.name}")
