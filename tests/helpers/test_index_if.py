from dataclasses import dataclass
from typing import Any
from helpers import index_if
import pytest

@dataclass
class Class:
    value: Any

@pytest.fixture
def array():
    return [Class(i) for i in range(10)]

def eq_val(x):
    return lambda y: y == x

def eq_ref(x):
    return lambda y: y is x

def test_val(array):
    for i in range(len(array)):
        assert index_if(array, eq_val(Class(i))) == i

def test_val_not(array):
    assert index_if(array, eq_val(Class(-1))) is None
    assert index_if(array, eq_val(Class(len(array)))) is None

def test_ref(array):
    for i in range(len(array)):
        assert index_if(array, eq_ref(array[i])) == i

def test_ref_not(array):
    for i in range(len(array)):
        assert index_if(array, eq_ref(Class(array[i].value))) is None
