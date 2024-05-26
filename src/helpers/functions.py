
import functools
import inspect
import logging
import time
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


def to_iterable(arg):
    if arg is None:
        return []
    
    if hasattr(arg, '__iter__'):
        return arg
    
    return [arg]


def to_array(arg):
    return list(to_iterable(arg))

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

def curry(f, *args, **kwargs):
    def curried(*new_args, **new_kwargs):
        nonlocal f, args, kwargs
        return f(*args, *new_args, **kwargs, **new_kwargs)
    return curried

def bound(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def kph_to_mps(kph: float) -> float:
    return kph * 1000 / 3600

def mps_to_kph(mps: float) -> float:
    return mps * 3600 / 1000

def indent(func, /, logger: logging.Logger = None):
    def wrap(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            logger = coalesce(logger, logging.getLogger())

            handlers = []
            for h in logger.handlers:
                if hasattr(h.formatter, 'up') and hasattr(h.formatter, 'down'):
                    h.formatter.up()
                    handlers += [h]

            ret = func(*args, **kwargs)

            for h in handlers:
                h.formatter.down()

            return ret

        return wrapper

    if func is None:
        return wrap
    
    return wrap(func)

def time_to_str(time: float):
    h = 0
    if time > 3600:
        h = int(time / 3600)
        time = time - h * 3600
    
    m = 0
    if time > 60:
        m = int(time / 60)
        time = time - m * 60
    
    string = f"{h}h {m}m {time:.3f}s"
    string = string.removeprefix('0h ')
    string = string.removeprefix('0m ')
    return string

def measure_time(func, /, level=logging.INFO):
    def wrap(func):
        def measure_time(*args, **kwargs):
            nonlocal level

            @indent
            def call():
                nonlocal args, kwargs
                return func(*args, **kwargs)


            logging.log(level=level, msg=f"┬ start {func.__name__}...")

            start = time.time()
            ret = call()
            end = time.time()

            logging.log(level=level, msg=f"┴ finished {func.__name__} in {time_to_str(end - start)}")

            return ret
        return measure_time

    if func is None:
        return wrap
    
    return wrap(func)
