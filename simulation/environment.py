
from helpers import coalesce, not_implemented
from simulation.object import Object
from simulation.location import Path
from dataclasses import dataclass
from typing import Callable

OID = int

@dataclass
class EnvironmentInterface:
    time: Callable[[], float]           = not_implemented
    objects: Callable[[], list[Object]] = not_implemented
    register: Callable[[Object], OID]   = not_implemented


class Environment(EnvironmentInterface):

    DEFAULT_DT = 0.01

    def __init__(self, dt = None):
        self._dt = coalesce(dt, Environment.DEFAULT_DT)
        self._time = 0
        self._objects = dict[OID, Object]()
        self.next_oid = 0

        self._interface = EnvironmentInterface()
        self._interface.time = self.time
        self._interface.objects = self.objects
        self._interface.register = self._register

    def interface(self) -> EnvironmentInterface:
        return self._interface

    def time(self) -> float:
        return self._time

    def objects(self) -> [Object]:
        return self._objects.values()

    def _register(self, object: Object) -> OID:
        oid = self.next_oid
        self.next_oid += 1
        self._objects[oid] = object
        return oid
    
    def create_object(self, cls: type, *args, **kwargs) -> Object:
        object = cls(args, kwargs, env = self.interface())
        return object

    def update(self, dt = None):
        dt = coalesce(dt, self._dt)
        
        # TODO parallel
        paths = {}
        for oid, object in self._objects.items():
            paths[oid] = Path(object.location(), object.update())

        

        self._time += dt
        raise NotImplementedError

    def __detect_collisions(self, paths: dict[OID, Path]) -> tuple[OID, OID]:
        paths
