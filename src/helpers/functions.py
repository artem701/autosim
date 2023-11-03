
import functools
import inspect
from typing import Any, Callable


def coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def not_implemented(func):
    def wrapper(*args, **kwargs):
        owner = get_class_that_defined_method(func)
        owner_str = '' if owner is None else f" of {owner.__name__}"
        raise NotImplementedError(
            f"function {func.__name__}{owner_str} is not implemented")
    return wrapper


def remove_by_identity(list: list, element):
    for i in range(len(list)):
        if list[i] is element:
            list.pop(i)
            return True
    return False


def to_array(arg):
    if arg is None:
        return []

    try:
        return list(arg)
    except:
        return [arg]


def get_class_that_defined_method(meth):
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (inspect.isbuiltin(meth) and getattr(meth, '__self__', None) is not None and getattr(meth.__self__, '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        # fallback to __qualname__ parsing
        meth = getattr(meth, '__func__', meth)
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[
            0].rsplit('.', 1)[0],
            None)
        if isinstance(cls, type):
            return cls
    # handle special descriptor objects
    return getattr(meth, '__objclass__', None)


def index_if(array: list, predicate: Callable[[Any], bool]):
    i = 0
    for element in array:
        if predicate(element):
            return i
        i += 1
    return None


def bound(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def kph_to_mps(kph: float) -> float:
    return kph * 1000 / 3600

def mps_to_kph(mps: float) -> float:
    return mps * 3600 / 1000
