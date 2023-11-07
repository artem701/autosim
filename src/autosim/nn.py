from dataclasses import dataclass, field
import enum
from typing import Any, Generator
import numpy as np
import pygad.nn as pgnn
from helpers import Serializable
from helpers.cached import Cached

@dataclass
class LayerArchitecture:
    n: int

@dataclass
class InputLayerArchitecture(LayerArchitecture):
    pass

class ActivationFunction(enum.Enum):
    NONE = 'None'
    RELU = 'relu'
    SIGM = 'sigmoid'
    SOFT = 'softmax'

@dataclass
class DenseLayerArchitecture(LayerArchitecture):
    f: ActivationFunction

@dataclass
class NetworkArchitecture:
    input_layer: InputLayerArchitecture
    dense_layers: list[DenseLayerArchitecture]

@dataclass
class NeuralNetwork(Serializable):
    output_layer: pgnn.DenseLayer
    _architecture: Cached[NetworkArchitecture] = field(default_factory=Cached)
    
    def __eq__(self, other: 'NeuralNetwork'):
        if self.architecture() != other.architecture():
            return False
        a = self.as_vector()
        b = other.as_vector()
        
        return a.shape == b.shape and (a == b).all()
    
    @staticmethod
    def random(architecture: NetworkArchitecture, bound: tuple[float, float] = (-0.5, 0.5)) -> 'NeuralNetwork':
        input_layer = pgnn.InputLayer(architecture.input_layer.n)
        output_layer = input_layer
        
        for arch in architecture.dense_layers:
            output_layer = pgnn.DenseLayer(arch.n, output_layer, arch.f.value)
            output_layer.trained_weights = np.random.uniform(low=bound[0], high=bound[1], size=output_layer.trained_weights.size)
        
        return NeuralNetwork(output_layer)
    
    def predict_many(self, inputs: np.ndarray, problem_type: str = 'regression'):
        return pgnn.predict(last_layer=self.output_layer, data_inputs=np.array(inputs, np.double), problem_type=problem_type)

    def predict(self, inputs: np.ndarray, problem_type: str = 'regression'):
        prediction = self.predict_many(inputs=[inputs], problem_type=problem_type)[0]
        if type(prediction) is not np.ndarray:
            prediction = np.array([prediction])
        return prediction

    def layers_reversed(self) -> Generator[pgnn.DenseLayer | pgnn.InputLayer, Any, None]:
        layer = self.output_layer
        while isinstance(layer, pgnn.DenseLayer):
            yield layer
            layer = layer.previous_layer
        yield layer
        assert isinstance(layer, pgnn.InputLayer)

    def as_vector(self):
        vector = []
        for layer in self.layers_reversed():
            if isinstance(layer, pgnn.DenseLayer):
                vector = layer.trained_weights.flatten().tolist() + vector
        return np.array(vector)
    
    @staticmethod
    def from_vector(architecture: NetworkArchitecture, vector: np.ndarray):
        input_layer = pgnn.InputLayer(architecture.input_layer.n)
        output_layer = input_layer
        weight_index = 0
        for arch in architecture.dense_layers:
            previous_n = output_layer.num_neurons
            assert weight_index + arch.n * previous_n <= vector.shape[0]

            output_layer = pgnn.DenseLayer(arch.n, output_layer, arch.f.value)
            for i in range(previous_n):
                output_layer.trained_weights[i,:] = vector[weight_index + arch.n * i : weight_index + arch.n * (i + 1)]

            weight_index += arch.n * previous_n
        assert weight_index == vector.shape[0]

        network = NeuralNetwork(output_layer=output_layer)
        network._architecture.set(architecture)
        return network

    def architecture(self) -> NetworkArchitecture:
        def getter():
            layers = reversed(list(self.layers_reversed()))
            input_layer = next(layers)
            return NetworkArchitecture(
                input_layer=InputLayerArchitecture(n=input_layer.num_neurons),
                dense_layers=[
                    DenseLayerArchitecture(n=layer.num_neurons, f=ActivationFunction(layer.activation_function))
                    for layer in layers
                    ])
        return self._architecture.get(getter)
