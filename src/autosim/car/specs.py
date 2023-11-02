
from dataclasses import dataclass


@dataclass
class Characteristics:
    mass: float
    thrust: float
    front_area: float
    streamlining: float


LADA_GRANTA = Characteristics(
    mass=1100,
    thrust=750,
    front_area=2.25,
    streamlining=0.36
)

TEST = Characteristics(
    mass=1000,
    thrust=1500,
    front_area=2.25,
    streamlining=0.36
)
