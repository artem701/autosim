
from dataclasses import dataclass, field
from autosim.car import Car
from helpers import not_implemented
import eventloop
import eventloop.events
import simulation

@dataclass
class SimulationParameters:
    timeout: float
    objects: list[eventloop.Listener] = field(default_factory=list)
    dt: float = simulation.Environment.DEFAULT_DT
    driver: simulation.Driver = simulation.Driver(simulation.Driver.Type.FAST)

class Simulation:

    def __init__(self, parameters: SimulationParameters):
        self.parameters = parameters
    
    def simulate(self):
        p = self.parameters

        environment = simulation.Environment(dt=p.dt, driver=p.driver)
        terminator = simulation.Timer(environment=environment, timeout=p.timeout, event=eventloop.events.Terminate, kwargs={'immediate': False})
        terminator.start()
        
        environment.subscribe(terminator, *p.objects)
        environment.simulate()
        
        return environment
