
from simulation.environment import EnvironmentInterface
from simulation.location import Location


class Object:

    def __init__(self, env: EnvironmentInterface, location: Location):
        self._env = env
        self.__oid = env.register(self)
        self.__location = location

    def env(self):
        return self._env

    def oid(self):
        return self.__oid

    def location(self):
        return self.__location

    def update(dt: float) -> float:
        pass
