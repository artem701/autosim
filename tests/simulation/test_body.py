
from simulation import Body
from simulation.location import Line
import pytest


@pytest.fixture
def body():
    return Body(Line(0), 1000, 0)


def test_acceleration_no_push(body):
    assert body.push(100, lambda t, x, v: 0).dx == 0
    assert body.v == 0


def test_acceleration_push_const(body):
    assert body.push(100, lambda t, x, v: 100).dx == 500
    assert body.v == 10


def test_acceleration_start_velocity_no_push(body):
    body.v = 100
    assert body.move(10).dx == 1000
    assert body.v == 100


def test_acceleration_start_velocity_push_const(body):
    body.v = 10
    assert body.push(10, lambda t, x, v: 100).dx == 105
    assert body.v == 11
