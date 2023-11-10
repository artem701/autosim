
from itertools import permutations
from simulation import Environment, Moveable
from simulation.location import Line, Circle, CircleSpace
import pytest

def make_permutations(n):
    return [(p) for p in permutations(range(n))]

def environment(moveables: list[Moveable], p=None):
    if p is None:
        p = range(len(moveables))
    moveables = [moveables[i] for i in p]

    environment = Environment()
    environment.subscribe(*moveables)
    # Initialize environment.moveables
    environment.iterate()
    return environment

def line(n, p=None):
    ms = [Moveable(location=Line(i)) for i in range(n)]
    env = environment(ms, p)
    return env, ms

def circle(n, p=None):
    ms = [Moveable(location=Circle(CircleSpace(n), i)) for i in range(n)]
    env = environment(ms, p)
    return env, ms

def test_next_line_none():
    e, ms = line(1)
    assert ms[0].next(e) is None

def test_next_line_behind():
    e, ms = line(2)
    assert ms[1].next(e) is None

def test_next_line_one():
    e, ms = line(2)
    assert ms[0].next(e) is ms[1]

@pytest.mark.parametrize('p', make_permutations(5))
def test_next_line_many(p):
    N = len(p)
    assert N > 2

    e, ms = line(N, p)
    for i in range(N - 1):
        assert ms[i].next(e) is ms[i + 1]
    assert ms[-1].next(e) is None

def test_next_circle_one():
    e, ms = circle(2)
    assert ms[0].next(e) is ms[1]
    assert ms[1].next(e) is ms[0]

@pytest.mark.parametrize('p', make_permutations(5))
def test_next_circle_many(p):
    N = len(p)
    assert N > 2

    e, ms = circle(N, p)
    for i in range(N):
        assert ms[i].next(e) is ms[(i + 1) % N]
