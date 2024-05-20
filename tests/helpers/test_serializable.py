
import logging
from autosim.nn import *
from dataclasses import dataclass
from helpers import Serializable
import pygad.pygad.nn as pgnn
import pytest

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
def test_nn(architecture: NetworkArchitecture):
    network = NeuralNetwork.random(architecture=architecture)
    for layer in network.layers_reversed():
        if isinstance(layer, pgnn.InputLayer):
            continue
        weights = [10 / (i + 1) for i in range(layer.trained_weights.size)]
        layer.trained_weights = np.array(weights).reshape(layer.trained_weights.shape)
    assert network == NeuralNetwork.from_json(network.to_json())

@dataclass
class SomeClassA(Serializable):
    a: int
    b: str
    c: float
    d: list[int]

@dataclass
class SomeClassB(Serializable):
    a: int
    b: str
    c: float
    d: list[int]

def test_type():
    obj_a = SomeClassA(1, 'one', 1.0, [1])
    logging.error(f"{obj_a.to_json()}")
    with pytest.raises(TypeError):
        SomeClassB.from_json(obj_a.to_json())