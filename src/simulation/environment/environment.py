
from dataclasses import dataclass
from eventloop.events import RemoveListener
from helpers import IdentitySet, coalesce, not_implemented, remove_by_identity
from eventloop import Listener, Event
from simulation import Object, Moveable
from simulation.location import Path
from simulation.moveable.events import Move


@dataclass
class Tick(Event):
    environment: 'Environment'


class UpdateRequest(Event):
    pass


@dataclass
class Collision(Event):
    # Who hit
    collider: Moveable
    # Who was hit
    collidee: Moveable
    time: float = None


class Environment(Listener):

    DEFAULT_DT = 0.01

    Objects = IdentitySet[Object]

    def __init__(self, dt: float = None):
        self.dt = coalesce(dt, Environment.DEFAULT_DT)
        self.time = 0
        self.moveables = list[Moveable]()
        self._moves = {}
        self._first_update = True

    def input_events(self):
        return {UpdateRequest, Move, RemoveListener}

    def accept(self, event):

        if isinstance(event, UpdateRequest):

            if not self._first_update:
                self.time += self.dt
            self._first_update = False

            collisions = self.detect_collision()
            self._moves = {}
            return collisions + [Tick(self)]

        if isinstance(event, Move):
            return self.handle_move(event)

        if isinstance(event, RemoveListener):
            remove_by_identity(self.moveables, event.listener)
            return None

        raise RuntimeError(f"Unhandeled event: {event}")

    def handle_move(self, move: Move) -> Event | None:
        if move.sender not in self.moveables:
            self.moveables.append(move.sender)
            self.moveables.sort(key=lambda moveable: moveable.location.x())
        self._moves[move.sender] = Path(move.sender.location, move.dx)
        move.sender.location = move.sender.location.moved(move.dx)

    def detect_collision(self) -> list[Collision]:
        collisions = []
        N = len(self.moveables)
        if N < 2:
            return []

        def get_path(i) -> Path:
            return self._moves.setdefault(self.moveables[i], Path(self.moveables[i].location, 0))

        for i in range(N):
            next_idx = (i + 1) % N
            current_path = get_path(i)
            next_path = get_path(next_idx)
            if current_path.collides(next_path):
                collisions.append(
                    Collision(self.moveables[i], self.moveables[next_idx], time=self.time))

        return collisions
