
import functools
import inspect


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
    if isinstance(arg, set):
        return list(arg)
    if not isinstance(arg, list):
        return [arg]
    return arg


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
