from dataclasses import dataclass
import enum
import numpy as np
import pygad.nn as pgnn
from helpers import Serializable

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
    
    @staticmethod
    def random(architecture: NetworkArchitecture) -> 'NeuralNetwork':
        input_layer = pgnn.InputLayer(architecture.input_layer.n)
        output_layer = input_layer
        
        for arch in architecture.dense_layers:
            output_layer = pgnn.DenseLayer(arch.n, output_layer, arch.f.value)
        
        return NeuralNetwork(output_layer)
    
    def predict_many(self, inputs: np.ndarray, problem_type: str = 'regression'):
        return pgnn.predict(last_layer=self.output_layer, data_inputs=np.array(inputs, np.double), problem_type=problem_type)

    def predict(self, inputs: np.ndarray, problem_type: str = 'regression'):
        return self.predict_many(inputs=[inputs], problem_type=problem_type)[0]
