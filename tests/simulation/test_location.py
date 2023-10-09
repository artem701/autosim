
from simulation.location import Path, Line, Circle, CircleSpace
import pytest

fixture = pytest.fixture
parametrize = pytest.mark.parametrize

class TestCollisionDetection:

    @fixture
    @staticmethod
    def unit_circle():
        return  CircleSpace(1)

    @staticmethod
    def template(expected: bool, first: Path, second: Path):
        assert first.collides(second) == expected

    @staticmethod
    def line_template(expected, ax, adx, bx, bdx):
        def path(x, dx):
            return Path(Line(x), dx)
        TestCollisionDetection.template(expected, path(ax, adx), path(bx, bdx))

    @staticmethod
    def circle_template(expected, ax, adx, bx, bdx, unit_circle):
        assert 0 <= ax <= 1 
        assert 0 <= bx <= 1 
        def path(x, dx):
            return Path(Circle(unit_circle, x), dx)
        TestCollisionDetection.template(expected, path(ax, adx), path(bx, bdx))

    @parametrize('ax,adx,bx,bdx', [
        (0,     0, 100,   0),
        (0,    50, 100,  50),
        (0,   100, 100,  50),
        (0,   149, 100,  50),
        (0,    99, 100,   0),
        (100,  50,   0, 150), # collided into us
        (100,  50,   0, 151), # collided into us
        (100,  50,   0, 149), # not collided into us
        (0,    100,  0, 101),
        (0,    101,  0, 100),
    ])
    def test_line_negative(self, ax, adx, bx, bdx):
        TestCollisionDetection.line_template(False, ax, adx, bx, bdx)

    @parametrize('ax,adx,bx,bdx', [
        (0, 150, 100,   50),
        (0, 151, 100,   50),
        (0, 100, 100,    0),
        (0, 150, 100,    0),
        (0, 100, 0,    100),
    ])
    def test_line_positive(self, ax, adx, bx, bdx):
        TestCollisionDetection.line_template(True, ax, adx, bx, bdx)

    @parametrize('ax,adx,bx,bdx', [
        (0.0, 0.0, 0.5, 0.0), # moving or not, not reaching other's start
        (0.0, 0.0, 0.5, 0.1),
        (0.0, 0.1, 0.5, 0.0),
        (0.0, 0.1, 0.5, 0.1),

        (0.0, 0.6, 0.5, 0.2), # reaching other's start, but not end

        (0.0, 1.0, 0.5, 1.0), # making full circle together
        (0.0, 5.0, 0.5, 5.0),
        (0.0, 1.4, 0.5, 1.0),
        (0.0, 1.0, 0.9, 0.2),
        (0.0, 2.0, 0.9, 1.2),

        (0.5, 0.0, 0, 0.5), # collides into us
        (0.5, 0.0, 0, 0.6),
        (0.5, 0.1, 0, 0.6),
        (0.5, 0.1, 0, 0.7),
        (0.5, 5.1, 0, 5.7), # collides into us after 5 circles

        (0.5, 0.0, 0, 0.4), # not collides into us

        (0.8, 1.5, 0.0, 1.4),
        (0.8, 1.5, 0.0, 1.5),
        (0.8, 1.5, 0.0, 1.6),

        (0.0, 0.1, 0.0, 0.2),
        (0.0, 0.2, 0.0, 0.1),
        (0.0, 1.1, 0.0, 1.2),
        (0.0, 1.2, 0.0, 1.1),
    ])
    def test_circle_negative(self, ax, adx, bx, bdx, unit_circle):
        TestCollisionDetection.circle_template(False, ax, adx, bx, bdx, unit_circle)

    @parametrize('ax,adx,bx,bdx', [
        (0.0, 0.5, 0.5, 0.0),
        (0.0, 0.6, 0.5, 0.0),
        (0.0, 0.6, 0.5, 0.1),
        (0.0, 0.7, 0.5, 0.1),
        (0.0, 5.7, 0.5, 5.1),

        (0.9, 0.1, 0.0, 0.0),
        (0.9, 0.2, 0.0, 0.0),
        (0.9, 0.2, 0.0, 0.1),
        (0.9, 0.3, 0.0, 0.1),
        (0.9, 1.0, 0.0, 0.0),
        (0.9, 1.0, 0.0, 0.1),
        (0.9, 1.0, 0.0, 0.1),

        (0.0, 1.0, 0.0, 0.0),
        (0.0, 1.5, 0.0, 0.0),
        (0.0, 1.5, 0.0, 0.5),
        (0.0, 1.6, 0.0, 0.5),

        (0.0, 0.1, 0.0, 0.1),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 1.1, 0.0, 1.1),

    ])
    def test_circle_positive(self, ax, adx, bx, bdx, unit_circle):
        TestCollisionDetection.circle_template(True, ax, adx, bx, bdx, unit_circle)


