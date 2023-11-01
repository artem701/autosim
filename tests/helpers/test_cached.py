
from helpers import Cached

def test_get():
    VALUE = 123
    cached = Cached[int]()
    assert not cached.is_set
    is_called = False
    def getter():
        nonlocal is_called
        assert not is_called
        is_called = True
        return VALUE
    assert cached.get(getter=getter) == VALUE
    assert is_called
    assert cached.is_set
    assert cached.get(getter=getter) == VALUE
    assert cached.is_set

def test_set():
    VALUE_1 = 123
    VALUE_2 = 321
    cached = Cached[int]()
    cached.set(value=VALUE_1)
    is_called = False
    def getter():
        nonlocal is_called
        is_called = True
        return VALUE_2
    assert cached.get(getter=getter) == VALUE_1
    assert not is_called
    cached.set(value=VALUE_2)
    assert cached.get(getter=getter) == VALUE_2
    assert not is_called
