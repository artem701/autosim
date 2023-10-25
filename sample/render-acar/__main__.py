
from autosim.car import ACar
from eventloop import EventLoop
from eventloop.events import Terminate
from renderer import Renderer
from simulation import Environment
from simulation.location import Circle, CircleSpace
from timer.timer import Timer


loop = EventLoop()

environment = Environment()
car = ACar('50 * dt', ACar.Mode.MOVEMENT, Circle(CircleSpace(1000)))
renderer = Renderer()
timer = Timer(60, event=Terminate, ctx=False)

loop.subscribe(environment)
loop.subscribe(car)
loop.subscribe(renderer)
loop.subscribe(timer)

loop.loop()

renderer.render('output')
