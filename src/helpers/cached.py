
from typing import TypeVar, Generic, Callable

T = TypeVar('T')

class Cached(Generic[T]):
    def __init__(self):
        self.is_set = False

    def set(self, value: T):
        self.value = value
        self.is_set = True

    def get(self, getter: Callable[[], T]):
        if not self.is_set:
            self.value = getter()
            self.is_set = True
        return self.value
        
