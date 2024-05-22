import sys
from autosim.car import Car
from autosim.car import specs, Friction
from helpers import index_if, bound
from simulation.environment.environment import Environment
from simulation.environment.events import Tick
from simulation.location import Location, Line
import autosim.nn as nn

INPUTS = 'dx', 'dv', 'v'
OUTPUTS = 'd'

class NetworkArchitecture(nn.NetworkArchitecture):
  
    def __init__(self, dense_layers: list[nn.DenseLayerArchitecture], output_activation: str):
        input_layer = nn.InputLayerArchitecture(n=len(INPUTS))
        output_layer = nn.DenseLayerArchitecture(n=len(OUTPUTS), f=output_activation)
        super().__init__(input_layer=input_layer, dense_layers=dense_layers+[output_layer])

    @staticmethod
    def from_string(string: str, output_activation: str):
        string = string.strip()
        if string == 'linear' or string == '':
            string = ''
        else:
            string = f"x {string} "
        return nn.NetworkArchitecture.from_string(f"{len(INPUTS)} {string} x 1 ({output_activation})")

NeuralNetwork = nn.NeuralNetwork

class NCar(Car):

    def __init__(self, network: NeuralNetwork, location: Location = Line(0), spec: specs.Characteristics = specs.TEST, f: Friction = Friction.ASPHALT, name: str = None):
        super().__init__(location=location, spec=spec, f=f, name=name)
        assert isinstance(network, NeuralNetwork)
        self.network = network
        self.d = 0

    def update(self, environment: Environment):
        next = self.next(environment=environment)
        if next is not None:
            dx = self.location.distance(next.location)
            dv = next.v - next.v
        else:
            # TODO: something better? How to find front if you are not on the body list yet?
            dx = sys.float_info.max
            dv = sys.float_info.max
        decision = self.network.predict([dx, dv, self.v])[0]
        d = bound(decision * 2 - 1, -1, 1)
        self.d = d
        return self.accelerate(d, environment.dt)
