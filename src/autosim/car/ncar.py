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

NeuralNetwork = nn.NeuralNetwork

class NCar(Car):

    def __init__(self, network: NeuralNetwork, location: Location = Line(0), spec: specs.Characteristics = specs.TEST, f: Friction = Friction.ASPHALT, name: str = None):
        super().__init__(location=location, spec=spec, f=f, name=name)
        assert isinstance(network, NeuralNetwork)
        self.network = network

    def update(self, environment: Environment):
        bodies = environment.bodies
        self_index = index_if(bodies, lambda body: body is self)
        front_index = None
        if self_index is not None:
            front_index = (self_index + 1) % len(bodies)
            front = bodies[front_index]
            dx = self.location.distance(front.location)
            dv = front.v - self.v
        else:
            # TODO: something better? How to find front if you are not on the body list yet?
            dx = sys.float_info.max
            dv = sys.float_info.max
        decision = self.network.predict([dx, dv, self.v])[0]
        d = bound(decision * 2 - 1, -1, 1)
        return self.accelerate(d, environment.dt)
