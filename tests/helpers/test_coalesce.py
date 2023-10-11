
import pytest
from helpers import coalesce


@pytest.mark.parametrize("expected,args", [
                         (None, [None, None, None]),
                         (1, [1, 2, 3]),
                         (1, [1, None, 3]),
                         (1, [1, 2, None]),
                         (1, [None, 1, 2, 3]),
                         (2, [None, 2, 3, 1]),
                         ])
def test(expected, args):
    assert expected == coalesce(*args)
