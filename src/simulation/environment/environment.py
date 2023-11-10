
from dataclasses import dataclass
from enum import Enum
from eventloop.eventloop import Iteration
from eventloop.events import AddListener, RemoveListener
from helpers import IdentitySet, coalesce, not_implemented, remove_by_identity
from eventloop import Listener, CallbackListener, Event, EventLoop
from simulation import Object, Moveable
from simulation.body import Body
from simulation.location import Path
from simulation.moveable.events import Move
from bisect import insort


@dataclass
class Tick(Event):
    """Request from Environment to Objects to calculate their state
       in moment time + dt.
    """
    environment: 'Environment'


class UpdateRequest(Event):
    """Request from Driver to Environment to increment time by dt
       and update its state accordingly.
    """
    pass

@dataclass
class Collision(Event):
    # Who hit
    collider: Moveable
    # Who was hit
    collidee: Moveable
    time: float = None

class Driver(Listener):

    class Type(Enum):
        FAST = 'fast'
        REALTIME = 'realtime'

    def __init__(self, type: Type = Type.FAST):
        if type == Driver.Type.FAST:
            self.init_fast()
        elif type == Driver.Type.REALTIME:
            self.init_realtime()
        else:
            raise RuntimeError(f"Unknown driver type {type}")
    
    @not_implemented
    def init_realtime(self):
        pass

    def init_fast(self):
        self.handler = lambda _: UpdateRequest
    
    def input_events(self) -> set:
        return Iteration

    def accept(self, event: Event) -> list[Event]:
        return self.handler(event)

class Environment:

    DEFAULT_DT = 0.01
    INPUT_EVENTS = {UpdateRequest, Move, AddListener, RemoveListener}

    # Unordered cache.
    Objects = IdentitySet[Object]
    # Ordered.
    Moveables = list[Moveable]
    # Ordered.
    Bodies = list[Moveable]

    def __init__(self, dt: float = None, driver: Driver = Driver(type = Driver.Type.FAST)):
        # Environmental parameters.
        self.dt = coalesce(dt, Environment.DEFAULT_DT)

        # Objects' cache.
        self.objects = Environment.Objects()
        self.moveables = Environment.Moveables()
        self.bodies = Environment.Bodies()

        # Number of updates already done.
        # During each update objects are asked to calculate their state in the next point of time.
        # Time is incremented when the next update starts.
        self.time = 0
        self._updates = 0
        self._moves = {}
        self.loop = EventLoop()
        self.loop.subscribe(CallbackListener(accept_callback=self._accept, input_events=Environment.INPUT_EVENTS))
        self.loop.subscribe(driver)
    
    def subscribe(self, *listeners):
        for listener in listeners:
            self.loop.put(AddListener(listener))

    def put(self, event):
        self.loop.put(event=event)

    def simulate(self):
        self.loop.loop()

    def iterate(self):
        self.loop.iterate()

    def _add_object(self, object: Object):
        object.environment = self
        self.objects.add(object)

    def _add_moveable(self, moveable: Moveable):
        self._add_object(moveable)
        insort(self.moveables, moveable, key=lambda m: m.location.x())

    def _add_body(self, body: Body):
        self._add_moveable(body)
        insort(self.bodies, body, key=lambda b: b.location.x())

    def _accept(self, event):

        if isinstance(event, UpdateRequest):
            self.time = self._updates * self.dt

            collisions = self.detect_collision()
            self._moves = {}
            self._updates += 1
            return collisions + [Tick(self)]

        if isinstance(event, AddListener):

            if isinstance(event.listener, Body):
                self._add_body(event.listener)
            elif isinstance(event.listener, Moveable):
                self._add_moveable(event.listener)
            elif isinstance(event.listener, Object):
                self._add_object(event.listener)
            else:
                pass
            
            return None

        if isinstance(event, RemoveListener):
            if event.listener in self.objects:
                self.objects.remove(event.listener)
                remove_by_identity(self.moveables, event.listener)
                remove_by_identity(self.bodies, event.listener)
            return None

        if isinstance(event, Move) and isinstance(event.sender, Body):
            return self.handle_move(event)

        raise RuntimeError(f"Unhandeled event: {event}")

    def handle_move(self, move: Move) -> Event | None:
        if move.sender not in self.bodies:
            raise RuntimeError(f"Unregistered body moved: {move.sender}")
        self._moves[move.sender] = Path(move.sender.location, move.dx)
        move.sender.location = move.sender.location.moved(move.dx)

    def detect_collision(self) -> list[Collision]:
        collisions = []
        N = len(self.bodies)
        if N < 2:
            return []

        def get_path(i) -> Path:
            return self._moves.setdefault(self.bodies[i], Path(self.bodies[i].location, 0))

        current_path = get_path(0)
        for i in range(N):
            next_idx = (i + 1) % N
            next_path = get_path(next_idx)
            if current_path.collides(next_path):
                collisions.append(
                    Collision(self.bodies[i], self.bodies[next_idx], time=self.time))
            current_path = next_path

        return collisions
