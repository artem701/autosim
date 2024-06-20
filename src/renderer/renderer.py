
import math
from autosim.car.ncar import NCar
from autosim.ncarwatcher.events import NCarWatcherUpdate
from eventloop.eventloop import Event, Listener
from helpers.functions import mps_to_kph, not_implemented
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

# TODO: major refactoring


class Renderer(Listener):

    def __init__(self, fps=24, width=640, height=480, text: str='n,x,u,vms,vkh'):
        self.frames = list[Frame]()
        self.fps = fps
        self.frame_dt = 1 / fps
        self.width = width
        self.height = height
        self.start_time = None
        self.last_frame_time = None
        self.space = None
        self.mapper = None
        self.texts = [t.strip() for t in text.split(',')]
        self.last_ncarwatcher_update: NCarWatcherUpdate = None

    def render(self, path: str):

        size = self.width, self.height

        if not path.endswith('.avi'):
            path += '.avi'

        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, size)
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        
        return path

    def input_events(self) -> set:
        return Tick, NCarWatcherUpdate

    def accept(self, event: Event) -> list[Event]:
        if isinstance(event, Tick):
            self.handle_tick(event.environment)
        if isinstance(event, NCarWatcherUpdate):
            self.last_ncarwatcher_update = event

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

    def capture_frame(self, environment: Environment) -> Frame:
        canvas = self.create_canvas()

        text_duration = 5
        
        def text_common(body, drawable):
            # text = f"{drawable.name}: x={body.location.x():.2f} m"
            texts = []

            if self.must_show_text('n'):
                texts += [f"{drawable.name}"]

            if self.must_show_text('x'):
                texts += [f"x: {body.location.x():.2f} m"]


            if self.must_show_text('u') and hasattr(body, 'u'):
                texts += [f"u: {body.u:.2f}"]

            return texts

        text_mapper = []
        show_speed=False
        if self.must_show_text('vms'):
            text_mapper.append(lambda body, drawable: [*text_common(body, drawable), f"v: {body.v:.2f} m/s"])
            show_speed=True
        if self.must_show_text('vkh'):
            text_mapper.append(lambda body, drawable: [*text_common(body, drawable), f"v: {mps_to_kph(body.v):.2f} km/h"])
            show_speed=True
        if not show_speed:
            text_mapper.append(lambda body, drawable: [*text_common(body, drawable)])

        pick_text = lambda t: text_mapper[math.floor(t / text_duration) % len(text_mapper)]

        if self.get_space(environment) is None:
            return canvas

        self.draw_space(environment, canvas)
        mapper = self.get_mapper(environment)
        for obj in environment.bodies:
            drawable = mapper.map(obj)
            if drawable.skip:
                continue
            R = 5
            cv2.circle(canvas, (drawable.x, drawable.y),
                       R, drawable.color, cv2.FILLED)
            if drawable.name is not None:
                texts = pick_text(environment.time)(obj, drawable)
                text = ', '.join(texts)
                self.print(canvas,
                           drawable.x + math.ceil(R / 2),
                           drawable.y - math.ceil(R / 2),
                           text)

        self.print(canvas, 1, 1, f"t = {environment.time:.2f} s")
        if self.last_ncarwatcher_update:
            self.print(canvas, 1, 16 + 20, f"Vavg  = {mps_to_kph(self.last_ncarwatcher_update.v_avg):.2f} km/h")
            self.print(canvas, 1, 16 + 40, f"U+int = {self.last_ncarwatcher_update.u_pos_dt_int:.2f}")
        return canvas

    def create_canvas(self):
        return np.full((self.height, self.width, 3), 255).astype(np.uint8)

    def get_space(self, environment: Environment):
        if self.space is None and len(environment.bodies) > 0:
            location = environment.bodies[0].location
            self.space = location.space

        return self.space

    def get_mapper(self, environment: Environment) -> Mapper:
        if self.mapper is None:

            space = self.get_space(environment)

            if isinstance(space, LineSpace):
                self.mapper = self.line_mapper
            elif isinstance(space, CircleSpace):
                self.mapper = self.circle_mapper(space)
            else:
                self.mapper = self.none_mapper()

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
            return int(round(cx + math.cos(angle) * r)), int(round(cy - math.sin(angle) * r))
        return Mapper(space_mapper=map)

    def circle(self, margin=10):
        r = int(math.floor(min(self.width, self.height)) / 2 - margin)
        cx = int(math.floor(self.width / 2))
        cy = int(math.floor(self.height / 2))
        return cx, cy, r

    def none_mapper(self):
        return lambda _: Drawable(skip=True)

    def draw_space(self, environment, canvas):
        space = self.get_space(environment)
        if isinstance(space, LineSpace):
            self.draw_line(canvas)

        if isinstance(space, CircleSpace):
            self.draw_circle(canvas)

    @not_implemented
    def draw_line(self):
        pass

    def draw_circle(self, canvas):
        cx, cy, r = self.circle()
        cv2.circle(canvas, (cx, cy), r, (0, 0, 0), 2)
        if self.space:
            assert isinstance(self.space, CircleSpace)
            self.print(canvas, 1, self.height - 2, f"R = {self.space.length()/2/math.pi:.2f}m")


    def print(self, canvas, x, y, text):
        font = cv2.FONT_HERSHEY_PLAIN
        scale = max(self.width, self.height) / 768
        color = (0, 0, 0)
        thickness = 1

        (w, h), b = cv2.getTextSize(text, font, scale, thickness)
        h += b

        x = max(0, min(x, self.width - w - 1))
        y = max(h, min(y, self.height - 1))

        cv2.putText(canvas, text, (x, y), font, scale, color, thickness)

    def must_show_text(self, text: str):
        return text in self.texts
