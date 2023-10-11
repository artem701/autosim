from collections.abc import MutableSet


class Ref(object):
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value is other.value

    def __hash__(self):
        return id(self.value)


class IdentitySet(MutableSet):
    key = id  # should return a hashable object

    def __init__(self, iterable=()):
        self.map = {}  # id -> object
        self |= iterable  # add elements from iterable to the set (union)

    def __len__(self):  # Sized
        return len(self.map)

    def __iter__(self):  # Iterable
        return iter(self.map.values())

    def __contains__(self, x):  # Container
        return self.key(x) in self.map

    def add(self, value):  # MutableSet
        """Add an element."""
        self.map[self.key(value)] = value

    def discard(self, value):  # MutableSet
        """Remove an element.  Do not raise an exception if absent."""
        self.map.pop(self.key(value), None)

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))
