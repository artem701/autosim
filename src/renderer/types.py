
from dataclasses import dataclass

from renderer.frame import Frame

@dataclass
class Drawable:
    x: float
    y: float
    name: str
    color: tuple[int, int, int]
    skip: bool = False
