
def coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def not_implemented(func):
    def wrapper(*args, **kwargs):
        raise NotImplementedError
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
