
from helpers import get_class_that_defined_method


class SomeClass():
    def a(self):
        pass

    @staticmethod
    def b():
        pass

    @classmethod
    def c(cls):
        pass


def test_method():
    assert get_class_that_defined_method(SomeClass.a) == SomeClass


def test_static_method():
    assert get_class_that_defined_method(SomeClass.b) == SomeClass


def test_class_method():
    assert get_class_that_defined_method(SomeClass.c) == SomeClass
