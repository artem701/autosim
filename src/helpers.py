
from typing import Callable


def coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def not_implemented():
    raise NotImplementedError
