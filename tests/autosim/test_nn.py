
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