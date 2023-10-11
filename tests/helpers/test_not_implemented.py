
import pytest
from helpers import not_implemented


def test():
    with pytest.raises(NotImplementedError):
        not_implemented()
