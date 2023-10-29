
from dataclasses import dataclass, field
from autosim.car import Car
from helpers import not_implemented
import eventloop
import eventloop.events
import simulation
import simulation.driver

@dataclass
class SimulationParameters:
    timeout: float
    objects: list[eventloop.Listener] = field(default_factory=list)
    dt: float = simulation.Environment.DEFAULT_DT
    driver: simulation.Driver = simulation.Driver(simulation.driver.Type.FAST)

class Simulation:

    def __init__(self, parameters: SimulationParameters):
        self.parameters = parameters
    
    def simulate(self):
        p = self.parameters

        loop = eventloop.EventLoop()
        environment = simulation.Environment(dt=p.dt)
        terminator = simulation.Timer(environment=environment, timeout=p.timeout, event=eventloop.events.Terminate, kwargs={'immediate': False})
        terminator.start()
        
        loop.subscribe(p.driver)
        loop.subscribe(environment)
        loop.subscribe(terminator)
        for object in p.objects:
            loop.subscribe(object)
        
        loop.loop()
