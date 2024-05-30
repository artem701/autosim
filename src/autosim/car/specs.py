
from dataclasses import dataclass


@dataclass
class Characteristics:
    mass: float
    thrust: float
    mbreak: float
    front_area: float
    streamlining: float


LADA_GRANTA = Characteristics(
    mass=1100,
    thrust=750,
    mbreak=1.2,
    front_area=2.25,
    streamlining=0.36
)

TEST = Characteristics(
    mass=1000,
    thrust=2250,
    mbreak=1.2,
    front_area=2.25,
    streamlining=0.36
)
