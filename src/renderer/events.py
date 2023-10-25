
from dataclasses import dataclass
from renderer.types import Frame


@dataclass
class FrameRendered:
    image: Frame
