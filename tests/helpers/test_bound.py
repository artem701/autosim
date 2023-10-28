from helpers import bound

def test_less():
    assert bound(-2, -1, 1) == -1

def test_low():
    assert bound(-1, -1, 1) == -1

def test_between():
    assert bound(0, -1, 1) == 0

def test_high():
    assert bound(1, -1, 1) == 1

def test_more():
    assert bound(2, -1, 1) == 1
