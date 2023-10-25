
from eventloop import Listener


class Object(Listener):
    def __init__(self, name: str = None):
        self.name = name
