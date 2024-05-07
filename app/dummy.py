
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable
from helpers import coalesce
from eventloop import Event
from simulation import Environment
from simulation.environment.events import Collision
import autosim.car as c
import math

def make_logistic(x0, y0, x1, y1, eps = 0.01):
    if x0 == x1:
        return lambda _: (y0 + y1) / 2
    if y0 == y1:
        return lambda _: y0

    if x0 > x1:
        (x0, y0), (x1, y1) = (x1, y1), (x0, y0)

    xm = (x0 + x1) / 2
    dx = x1 - x0
    dy = y1 - y0
    l = abs(dy) + 2 * eps
    t = None
    if dy > 0:
        t = (dy + eps) / eps
    else:
        t = eps / (-dy + eps)
    k = 2 * math.log(t) / dx
    b = y0 - l / (1 + t)

    return lambda x: l / (1 + math.exp(-k * (x - xm))) + b

ATTACK = 3
SUSTAIN = 0.8
HOLD = 12
RELEASE = 1
WAIT = 14

@dataclass
class State:
    t: float
    f: Callable[[float], float]

class DummyCar(c.Car):
    
    def __init__(self, attack=ATTACK, sustain=SUSTAIN, hold=HOLD, release=RELEASE, wait=WAIT, **kwargs):
        super().__init__(**kwargs)

        class States(Enum):
            ACCELERATE  = State(t=attack,   f=make_logistic(0, 0, attack, sustain))
            HOLD        = State(t=hold,     f=lambda _: sustain)
            BREAK       = State(t=release,  f=make_logistic(0, sustain, release, -1))
            WAIT        = State(t=wait,     f=lambda _: -1)
            
            def next(self):
                values = [element for element in States]
                return values[(values.index(self) + 1) % len(values)]

        self.state = States.ACCELERATE
        self.t0 = None
    
    def next_state(self):
        self.state = self.state.next()
        self.t0 = None
    
    def update(self, environment: Environment) -> Event:
        t = environment.time
        self.t0 = coalesce(self.t0, t)
        dt = t - self.t0

        state = self.state.value
        if dt < state.t:
            self.d = state.f(dt)
        else:
            self.next_state()

        return self.accelerate(self.d, environment.dt)
    
    def accept(self, event: Event) -> list[Event]:
        if isinstance(event, Collision):
            # ignore collisions
            return
        return super().accept(event)