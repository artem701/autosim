from types import SimpleNamespace
import pytest


class Class:
    """For identity testing
    """

    def __eq__(self, other):
        return True


@pytest.fixture
def v():
    """Namespace of value-equal objects, different by identity
    """
    return SimpleNamespace(
        a=Class(), b=Class(), c=Class(), d=Class(), e=Class(), f=Class(),
        g=Class(), h=Class(), i=Class(), j=Class(), k=Class(), l=Class(),
        m=Class(), n=Class(), o=Class(), p=Class(), q=Class(), r=Class(),
        s=Class(), t=Class(), u=Class(), v=Class(), w=Class(), x=Class(),
        y=Class(), z=Class()
    )
