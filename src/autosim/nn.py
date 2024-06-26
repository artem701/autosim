from dataclasses import dataclass, field
from typing import Any, Generator
import math
import numpy as np
import pygad.pygad.nn as pgnn
import re
from helpers import Serializable, SmartEnum
from helpers.cached import Cached
import pyvis.network

@dataclass
class LayerArchitecture:
    n: int
    
    def __str__(self):
        return str(self.n)

@dataclass
class InputLayerArchitecture(LayerArchitecture):
    pass

class ActivationFunction(SmartEnum):
    NONE = 'None'
    RELU = 'relu'
    SIGM = 'sigmoid'
    SOFT = 'softmax'

@dataclass
class DenseLayerArchitecture(LayerArchitecture):
    f: ActivationFunction
    
    def __str__(self):
        return super().__str__() + f" ({self.f.name.lower()})"

@dataclass
class NetworkArchitecture:
    input_layer: InputLayerArchitecture
    dense_layers: list[DenseLayerArchitecture]
    
    def __str__(self):
        return ' x '.join([str(layer) for layer in [self.input_layer, *self.dense_layers]])

    @staticmethod
    def from_string(string: str):
        string = string.replace(' ', '')
        string.lower()
        if string == 'linear':
            string = ''
        input, rest = re.compile(r'(?P<input>\d+)(?P<rest>.*)').fullmatch(string).groups()
        input_layer = InputLayerArchitecture(int(input))
        dense_layers = []
        denses = rest.split('x')
        assert denses[0] == ''
        for dense in denses[1:]:
            neurons, activation = re.compile(r'(?P<neurons>\d+)\((?P<activation>[a-z]+)\)').fullmatch(dense).groups()
            dense_layers += [DenseLayerArchitecture(int(neurons), ActivationFunction.from_string(activation.upper()))]
        return NetworkArchitecture(input_layer, dense_layers)

@dataclass
class NeuralNetwork(Serializable):
    output_layer: pgnn.DenseLayer
    _architecture: Cached[NetworkArchitecture] = field(default_factory=Cached)
    fitness: list[float] = field(default_factory=lambda:[])
    
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

    def layers(self) -> Generator[pgnn.DenseLayer | pgnn.InputLayer, Any, None]:
        return reversed(list(self.layers_reversed()))

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
            previous_n = output_layer.num_neurons + 1
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
            layers = self.layers()
            input_layer = next(layers)
            return NetworkArchitecture(
                input_layer=InputLayerArchitecture(n=input_layer.num_neurons),
                dense_layers=[
                    DenseLayerArchitecture(n=layer.num_neurons, f=ActivationFunction(layer.activation_function))
                    for layer in layers
                    ])
        return self._architecture.get(getter)

    def draw(self, output: str, inputs = None):
        if inputs:
            inputs = list(inputs) + ['1']
        def get_input(i):
            if inputs and len(inputs) > i:
                return inputs[i]
            else:
                return ''
        def chunks(array, n):
            k = len(array) // n
            for i in range(n):
                yield array[k * i:k * i + k]

        g = pyvis.network.Network()
        layers = list(self.layers())
        for i in range(len(layers)):
            H = 250
            V = 100
            x = i * H
            layer = layers[i]
            def y(j):
                num = layer.num_neurons + 1
                if i == len(layers) - 1:
                    num -= 1

                amp = num * V / 2
                return -amp + j * V

            for j in range(layer.num_neurons):
                if i == 0:
                    g.add_node(f"{(i, j)}", label=get_input(j), physics=False, x=x, y=y(j))
                else:
                    g.add_node(f"{(i, j)}", label=layer.activation_function, physics=False, x=x, y=y(j))
                    for k in range(layer.previous_layer.num_neurons + 1):
                        w = layer.trained_weights[k][j]
                        g.add_edge(f"{(i, j)}", f"{(i - 1, k)}", title=f"{w}", value=abs(w), color='blue' if w <= 0 else 'orange', physics=False)
            if i < len(layers) - 1:
                g.add_node(f"{(i, layer.num_neurons)}", label='1', physics=False, x=x, y=y(layer.num_neurons))
        g.write_html(output, notebook=False, open_browser=False)


