from helpers import remove_by_identity
from helpers.testing import v


def helper_remove_a_cmp(v, input, expected):
    remove_by_identity(input, v.a)
    assert len(input) == len(expected)
    for i in range(len(input)):
        assert input[i] is expected[i]


def test(v):
    helper_remove_a_cmp(v,
                        [v.b, v.c, v.d],
                        [v.b, v.c, v.d]
                        )
    helper_remove_a_cmp(v,
                        [v.a, v.b, v.c],
                        [v.b, v.c]
                        )
    helper_remove_a_cmp(v,
                        [v.b, v.a, v.c],
                        [v.b, v.c]
                        )
    helper_remove_a_cmp(v,
                        [v.b, v.c, v.a],
                        [v.b, v.c]
                        )
