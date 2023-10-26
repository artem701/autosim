
from typing import Callable
from renderer.types import Drawable
from simulation.location import Location
from simulation.object import Object
import distinctipy


SpaceMapper = Callable[[Object], tuple[float, float]]


class Mapper:

    COLORS_N = 36
    COLORS = [(round(255 * r), round(255 * g), round(255 * b))
              for r, g, b in distinctipy.get_colors(COLORS_N)]

    def __init__(self, space_mapper: SpaceMapper, color_key=id):
        self.space_mapper = space_mapper
        self.color_index = 0
        self.color_mapping = {}
        self.color_key = color_key

    def map(self, object: Object) -> Drawable:
        x, y = self.space_mapper(object)
        return Drawable(x=x, y=y, name=object.name, color=self.get_color(object))

    def get_color(self, object):
        key = self.color_key(object)
        if key not in self.color_mapping:
            self.color_mapping[key] = self.next_color()
        return self.color_mapping[key]

    def next_color(self):
        color = Mapper.COLORS[self.color_index]
        self.color_index = (self.color_index + 1) % Mapper.COLORS_N
        return color
