
import math
from eventloop.eventloop import Event, Listener
from helpers.functions import not_implemented
from renderer.frame import copy_frame
from simulation.environment.environment import Environment
from simulation.environment.events import Tick
from renderer.types import Drawable, Frame
from renderer.mapper import Mapper
from renderer.events import FrameRendered
import numpy as np
import cv2

from simulation.location import CircleSpace, LineSpace
from simulation.moveable.moveable import Moveable
from simulation.object import Object


class Renderer(Listener):

    def __init__(self, fps=24, width=1024, height=1024):
        self.frames = list[Frame]()
        self.fps = fps
        self.frame_dt = 1 / fps
        self.width = width
        self.height = height
        self.start_time = None
        self.last_frame_time = None
        self.space = None
        self.mapper = None

    def render(self, path: str):

        size = self.width, self.height

        if not path.endswith('.avi'):
            path += '.avi'

        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, size)
        for frame in self.frames:
            writer.write(frame)
        writer.release()

    def input_events(self) -> set:
        return Tick

    def accept(self, event: Event) -> list[Event]:
        if isinstance(event, Tick):
            return self.handle_tick(event.environment)

    def handle_tick(self, environment: Environment) -> list[FrameRendered]:
        frames_needed = 0

        if self.start_time is None:
            self.start_time = environment.time
            self.last_frame_time = self.start_time
            frames_needed = 1
        else:
            frames_needed = math.floor(
                (environment.time - self.last_frame_time) / self.frame_dt)
            self.last_frame_time = self.last_frame_time + frames_needed * self.frame_dt

        return self.create_frames(frames_needed, environment)

    def create_frames(self, n, environment: Environment) -> list[FrameRendered]:
        if n == 0:
            return []

        frames = []

        if n > 1:
            # Create n - 1 lag frames
            last_frame = self.frames[-1]
            lag_frame = copy_frame(last_frame)
            cv2.rectangle(lag_frame, (0, 0), (self.width - 1,
                          self.height - 1), (255, 0, 0), 5)
            frames += [lag_frame] * (n - 1)

        frames += [self.capture_frame(environment)]
        self.frames += frames
        return [FrameRendered(frame) for frame in frames]

    def capture_frame(self, environment: Environment):
        canvas = self.create_canvas()

        if self.get_space() is None:
            return canvas
        
        self.draw_space()
        mapper = self.get_mapper(environment)
        for obj in environment.objects:
            drawable = mapper.map(obj)
            if drawable.skip:
                continue
            cv2.circle(canvas, (drawable.x, drawable.y), 2, drawable.color, 1)
            if drawable.name is not None:
                cv2.putText(canvas, drawable.name, (drawable.x + 3, drawable.y - 3), cv2.FONT_HERSHEY_PLAIN, 0.1, (0,0,0))
        return canvas


    def create_canvas(self):
        return np.full((self.height, self.width, 3), 255)

    def get_space(self, environment: Environment):
        if self.space is None and len(environment.moveables) > 0:
            location = environment.moveables[0].location
            self.space = location.space

        if len(environment.moveables) == 0:
            return None

        return self.space

    def get_mapper(self, environment: Environment) -> Mapper:
        if self.mapper is None:

            space = self.get_space(environment)
            space_mapper = self.none_mapper()

            if isinstance(space, LineSpace):
                space_mapper = self.line_mapper

            if isinstance(space, CircleSpace):
                space_mapper = self.circle_mapper(space)

            self.mapper = Mapper(space_mapper)

        return self.mapper

    @not_implemented
    def line_mapper(self):
        pass

    def circle_mapper(self, space: CircleSpace):
        cx, cy, r = self.circle()
        def map(object: Object):
            if not isinstance(object, Moveable):
                return Drawable(skip=True)
            x = object.location.x()
            l = space.length()
            angle = 2 * math.pi * x / l
            return cx + math.cos(angle) * r, cy + math.sin(angle)
        return map

    def circle(self, margin=10):
        r = math.floor(min(self.width, self.height)) / 2 - margin
        cx = math.floor(self.width / 2)
        cy = math.floor(self.height / 2)
        return cx, cy, r

    def none_mapper(self):
        return lambda _: Drawable(skip=True)

    def draw_space(self, canvas):
        space = self.get_space()
        if isinstance(space, LineSpace):
            self.draw_line(canvas)

        if isinstance(space, CircleSpace):
            self.draw_circle(canvas)

    @not_implemented
    def draw_line(self):
        pass

    def draw_circle(self, canvas):
        cx, cy, r = self.circle()
        cv2.circle(canvas, (cx, cy), r, (0,0,0), 2)
