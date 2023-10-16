
import pytest
from helpers import not_implemented


@not_implemented
def func(x: float, y: float, z: float) -> float:
    pass


def test():
    with pytest.raises(NotImplementedError):
        func()
