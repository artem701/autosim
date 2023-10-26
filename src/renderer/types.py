
from dataclasses import dataclass

from renderer.frame import Frame

@dataclass
class Drawable:
    x: int
    y: int
    name: str
    color: tuple[int, int, int]
    skip: bool = False
