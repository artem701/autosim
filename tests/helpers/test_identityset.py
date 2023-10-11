
from helpers import IdentitySet
from helpers.testing import v
import pytest


def test_add_new(v):
    expected = IdentitySet([v.a, v.b, v.c, v.d, v.e])
    iset = IdentitySet([v.a, v.b, v.c, v.d])
    iset.add(v.e)
    assert iset == expected


def test_add_old(v):
    expected = IdentitySet([v.a, v.b, v.c, v.d, v.e])
    iset = IdentitySet([v.a, v.b, v.c, v.d, v.e])
    iset.add(v.e)
    assert iset == expected


def test_rm_existing(v):
    expected = IdentitySet([v.a, v.b, v.d, v.e])
    iset = IdentitySet([v.a, v.b, v.c, v.d, v.e])
    iset.remove(v.c)
    assert iset == expected


def test_rm_not_existing(v):
    with pytest.raises(KeyError):
        iset = IdentitySet([v.a, v.b, v.c, v.d, v.e])
        iset.remove(v.f)
