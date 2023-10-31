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

class NeuralNetwork(nn.NeuralNetwork):
    def __init__(self, output_layer):
        super().__init__(output_layer)

    @staticmethod
    def random(architecture: NetworkArchitecture):
        return NeuralNetwork(nn.NeuralNetwork.random(architecture).output_layer)

class NCar(Car):

    def __init__(self, network: NeuralNetwork, location: Location = Line(0), spec: specs.Characteristics = specs.LADA_GRANTA, f: Friction = Friction.ASPHALT, name: str = None):
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
        else:
            # TODO: something better? How to find front if you are not on the body list yet?
            front = self
        dx = front.location.x() - self.location.x()
        dv = front.v - self.v
        decision = self.network.predict([dx, dv, self.v])[0]
        d = bound(decision, -1, 1)
        return self.accelerate(d, environment.dt)
