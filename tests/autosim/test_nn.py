
from autosim.nn import InputLayerArchitecture, DenseLayerArchitecture, NetworkArchitecture, NeuralNetwork
import pytest

@pytest.mark.parametrize('architecture,compare_against', [
    # Only input.
    (NetworkArchitecture(InputLayerArchitecture(10), []), None),
    # Input and one output.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, 'relu')]), None),
    # Input and multiple output.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, 'relu'), DenseLayerArchitecture(10, 'relu')]), None),
    # Input and multiple output (use asterisc operator).
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, 'relu')] * 2), None),
    # Test previous two are same.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, 'relu'), DenseLayerArchitecture(10, 'relu')]),
     NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, 'relu')] * 2)),
])
def test_architecture_extraction(architecture, compare_against):
    if compare_against is None:
        compare_against = architecture
    assert NeuralNetwork.random(architecture).architecture() == architecture
