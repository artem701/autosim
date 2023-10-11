
def coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def not_implemented():
    raise NotImplementedError


def remove_by_identity(list: list, element):
    for i in range(len(list)):
        if list[i] is element:
            list.pop(i)
            return True
    return False
