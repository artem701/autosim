
import logging
from autosim.nn import InputLayerArchitecture, DenseLayerArchitecture, NetworkArchitecture, NeuralNetwork, ActivationFunction
import pytest

@pytest.mark.parametrize('architecture,compare_against', [
    # Only input.
    (NetworkArchitecture(InputLayerArchitecture(10), []), None),
    # Input and one output.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU)]), None),
    # Input and multiple output.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU), DenseLayerArchitecture(10, ActivationFunction.RELU)]), None),
    # Input and multiple output (use asterisc operator).
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU)] * 2), None),
    # Test previous two are same.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU), DenseLayerArchitecture(10, ActivationFunction.RELU)]),
     NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU)] * 2)),
])
def test_architecture_extraction(architecture, compare_against):
    if compare_against is None:
        compare_against = architecture
    assert NeuralNetwork.random(architecture).architecture() == architecture

@pytest.mark.parametrize('architecture', [
    # Only input.
    (NetworkArchitecture(InputLayerArchitecture(10), [])),
    # Input and one output.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU)])),
    # Input and multiple output.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU), DenseLayerArchitecture(10, ActivationFunction.RELU)])),
    # Input and multiple output (use asterisc operator).
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU)] * 2)),
    # Test previous two are same.
    (NetworkArchitecture(InputLayerArchitecture(10), [DenseLayerArchitecture(10, ActivationFunction.RELU)])),
])
def test_weights_extraction(architecture):
    network = NeuralNetwork.random(architecture=architecture)
    assert network == NeuralNetwork.from_vector(architecture=architecture, vector=network.as_vector())

@pytest.mark.parametrize('string,architecture', [
    ('1 x 2 (none) x 3 (relu) x 4 (sigm) x 5 (soft)', NetworkArchitecture(
        InputLayerArchitecture(1), [
            DenseLayerArchitecture(2, ActivationFunction.NONE),
            DenseLayerArchitecture(3, ActivationFunction.RELU),
            DenseLayerArchitecture(4, ActivationFunction.SIGM),
            DenseLayerArchitecture(5, ActivationFunction.SOFT),
    ])),
    ('7 x 7 (none) x 7 (relu) x 7 (sigm) x 7 (soft)', NetworkArchitecture(
        InputLayerArchitecture(7), [
            DenseLayerArchitecture(7, ActivationFunction.NONE),
            DenseLayerArchitecture(7, ActivationFunction.RELU),
            DenseLayerArchitecture(7, ActivationFunction.SIGM),
            DenseLayerArchitecture(7, ActivationFunction.SOFT),
    ])),
])
def test_from_string(string, architecture):
    assert NetworkArchitecture.from_string(string) == architecture
