from helpers.testing import v


def test(v):
    values = list(v.__dict__.values())
    for i in range(len(values)):
        for j in range(len(values)):
            assert values[i] == values[j]
            assert (values[i] is values[j]) == (i == j)
