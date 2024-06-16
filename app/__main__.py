
from dummy import DummyCar
from functools import partial
import autosim
from autosim.car.ncar import NetworkArchitecture, NeuralNetwork, INPUTS
from autosim import NCarWatcher
from autosim.nn import DenseLayerArchitecture, ActivationFunction
import autosim.training as t
import autosim.estimation as e
import autosim.simulation as s
import autosim.car as c
from dataclasses import dataclass
from eventloop.eventloop import Event, Listener
from helpers import kph_to_mps, measure_time, not_implemented, coalesce, indent, SmartEnum, Serializable
from renderer import Renderer
from simulation import Environment, Body
from simulation.environment.events import Tick
from simulation.location import Circle, CircleSpace, Line
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import sys
import warnings

import logging

L = 500
T_TRAINING = 60
BOUND = (-1, 1)

class Watcher(Listener):

    def __init__(self, target: c.Car):
        self.target = target
        self.t  = []
        self.dx = []
        self.dv = []
        self.v  = []
        self.u  = []

    def input_events(self) -> set:
        return Tick
    
    def accept(self, event: Event) -> list[Event]:
        
        assert isinstance(event, Tick)

        t = self.target
        n = t.next(event.environment)

        self.dx.append(t.location.distance(n.location))
        self.dv.append(t.v - n.v)
        self.v.append(t.v)
        self.u.append(t.u)
        self.t.append(event.environment.time)

    def plot(self, path, what='dx,dv,v,u'):
        what = set(what.replace(' ', '').split(','))

        plt.clf()
        plt.title('Target trajectory over time')

        if 'dx' in what:
            plt.plot(self.t, self.dx, label='ds')
        if 'dv' in what:
            plt.plot(self.t, self.dv, label='dv')
        if 'v' in what:
            plt.plot(self.t, self.v, label='v')
        if 'u' in what:
            plt.plot(self.t, self.u, label='u')
        
        plt.legend()
        plt.grid(which='both')
        plt.savefig(path)
        logging.info(f'saved watcher plot to {path}')


class SolutionArray(Serializable):
    def __init__(self):
        self.array = list[NeuralNetwork]()

    def best(self):
        b = self.array[0]
        for s in self.array:
            if s.fitness > b.fitness:
                b = s
        return b

def get_testing_simulation_parameters(solutions: SolutionArray, renderer, watcher, strategy, no_dummy):
    # DISTANCE = 20
    # l = 200 if no_dummy else L
    l = 100 if no_dummy else L
    CARS_TESTED = len(solutions.array) if no_dummy else 5
    GAP = 0 if no_dummy else 1
    T_RENDER = 3 * T_TRAINING
    objects = []
    students = []
    if no_dummy:
        students = [c.NCar(network=solutions.array[i],
                       name=f"student-{i}",
                       location = Circle(CircleSpace(l), i * l / (CARS_TESTED + GAP)))
                for i in range(CARS_TESTED)]
    else:
        students = [c.NCar(network=solutions.best(),
                        name=f"student-{i}",
                        location = Circle(CircleSpace(l), i * l / (CARS_TESTED + GAP)))
                    for i in range(CARS_TESTED)]            
    watcher.target = students[-1]
    objects += students
    if strategy != 'const60kph' and not no_dummy:
        dummy = DummyCar(location = Circle(CircleSpace(l), CARS_TESTED * l / (CARS_TESTED + GAP)), name = 'dummy')
        objects += [dummy]
    objects += [renderer, NCarWatcher(), watcher]
    return s.SimulationParameters(timeout=T_RENDER, objects = objects)

def strategy_value(func):
    def wrapper(ga, agg):
        return t.TrainingStrategy(func(), ga, agg)
    return wrapper

@strategy_value
def get_null_training_strategy():
    return []

@strategy_value
def get_const60kph_training_strategy():
    estimation_strategy = e.EstimationStrategy(
        collision=e.Criteria(10000),
        speed=e.ReferenceCriteria(0.01, kph_to_mps(60))
        )
    return [
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[]
                            )
                        ),
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(function='1', mode=c.ACar.Mode.ACCELERATION, location=Line(100))]
                            )
                        ),
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(function=f"{kph_to_mps(60)} * dt", mode=c.ACar.Mode.MOVEMENT, location=Line(100))]
                            )
                        ),
        ]

@strategy_value
def get_cautious_training_strategy():
    optimal_distance = 40
    estimation_strategy = e.EstimationStrategy(
        collision=e.Criteria(10000),
        distance=
            e.LessCriteria(fine=0.001, reference=optimal_distance + 20) +
            e.MoreCriteria(fine=0.001, reference=optimal_distance - 20),
        speed=e.LessCriteria(1, kph_to_mps(60))
        )
    return [
        # learn to stop
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar('0', c.ACar.Mode.MOVEMENT, location=Line(optimal_distance * 5), name='trainer')]
                            )
                        ),

        # follow constantly moving target
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(f"{kph_to_mps(60)} * dt", c.ACar.Mode.MOVEMENT, location=Line(optimal_distance), name='trainer')]
                            )
                        ),
        ]

@strategy_value
def get_speedy_training_strategy():
    # estimation_strategy = e.EstimationStrategy(collision=e.Criteria(10000), speed=e.ReferenceCriteria(fine=0.01, reference=kph_to_mps(120)))
    optimal_distance = 20
    estimation_strategy = e.EstimationStrategy(
        collision=e.Criteria(10000),
        distance=
         e.LessCriteria(fine=0.1, reference=optimal_distance + 10) + 
         e.MoreCriteria(fine=0.5, reference=optimal_distance - 10),
         speed=e.MoreCriteria(fine=0.001, reference=60)
        )
    return [
        # learn to stop
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar('0', c.ACar.Mode.MOVEMENT, location=Line(L), name='trainer')]
                            )
                        ),

        # follow constantly moving target
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(f"{kph_to_mps(60)} * dt", c.ACar.Mode.MOVEMENT, location=Line(L), name='trainer')]
                            )
                        ),
        ]

class Strategy(SmartEnum):
    null = partial(get_null_training_strategy)
    cautious = partial(get_cautious_training_strategy)
    speedy = partial(get_speedy_training_strategy)
    const60kph = partial(get_const60kph_training_strategy)

def plot_fitness(path, network: NeuralNetwork):
    fitness_min = [1 / fitness for fitness in network.fitness]
    plt.clf()
    plt.title('Fitness over generations')
    plt.plot(fitness_min)
    plt.xlabel('generation')
    plt.ylabel('fine')
    plt.yscale('log')
    plt.grid(which='both')
    plt.figtext(0.01, 0.01, f"generations: {len(network.fitness)}\narchitecture: {network.architecture()}", fontsize=12, ha='left')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(path)

@indent
def get_solution(strategy: Strategy, path: str, new: bool, cont: bool, population: int = None, generations: int = None, architecture_string: str = None, number = 1):
    architecture_presented = architecture_string is not None
    architecture = NetworkArchitecture.from_string(architecture_string or 'linear', 'sigm')

    solution_path = f"{path}\\{strategy.name}.json"
    fitness_path = lambda suffix: f"{path}\\{strategy.name}-{suffix}.jpg"
    network_html_path = lambda suffix: f"{path}\\{strategy.name}-{suffix}.html"

    logging.info(f"get solution for {strategy.name} strategy")
    
    solution = None
    if cont or not new:
        solution = lambda: load(solution_path, True)
    
    if solution is not None and architecture_presented and solution().architecture() != architecture:
        raise RuntimeError(f"Architecture is provided, but solution {solution()} with incompatible architecture exists!")

    if solution is not None:
        architecture = solution().architecture()
    else:
        logging.info('continue with random base solution')
        solution = lambda: random_solution(architecture)
        new = True
        cont = False
    
    if cont or new:
        if cont:
            logging.info(f"train solution from base: {solution_path}")

        os.makedirs(pathlib.Path(solution_path).parent, exist_ok=True)

        solutions = SolutionArray()
        for i in range(number):
            logging.info(f"training solution {i+1}/{number}")
            new_solution = train(solution(), strategy, population, generations, architecture)
            solutions.array.append(new_solution)

            plot_fitness(fitness_path(i), new_solution)        
            logging.info(f"saved {strategy.name} fitness plot to \"{fitness_path(i)}\"")

            new_solution.draw(network_html_path(i), INPUTS)
            logging.info(f"saved {strategy.name} network graph to \"{network_html_path(i)}\"")

        solutions.to_file(solution_path)
        logging.info(f"written {strategy.name} solutions to \"{solution_path}\"")

    return solutions, pathlib.Path(solution_path).parent

def load(path: str, best = False):
    if not os.path.isfile(path):
        logging.warning(f"solution file \"{path}\" is not found!")
        return None
    else:
        logging.info(f"load solution from \"{os.path.realpath(path)}\"...")

        try:
            solution = SolutionArray.from_file(path)
            logging.info(f"solution array contains {len(solution.array)} solutions")
            if best:
                return solution.best()
            return solution
        except:
            ret = SolutionArray()
            ret.array = [NeuralNetwork.from_file(path)]
            return ret

def random_solution(architecture):
    return NeuralNetwork.from_vector(architecture, NeuralNetwork.random(architecture, BOUND).as_vector())

@measure_time
def train(base: NeuralNetwork, strategy: Strategy, population: int, generations: int, architecture: NetworkArchitecture):
    
    population, generations = estimate_population_generations(base.architecture(), population, generations)
    logging.info(f"Algorithm genes = {NeuralNetwork.random(architecture=base.architecture()).as_vector().size}, population = {population}, generations = {generations}. Network architecture = {{ {base.architecture()} }}")

    ga = t.GeneticAlgorithmParameters(
        num_generations=generations, num_parents_mating=population//2,
        on_start=lambda ga:logging.info(f"genetic algorithm started"),
        on_stop=lambda ga,fitnesses:logging.info(f"genetic algorithm stopped"),
        on_generation=lambda ga:logging.info(f"finished {ga.generations_completed:3>}/{generations} generations (fine: {1 / np.max(ga.last_generation_fitness):.3f})"),
        # parallel_processing=('thread', 4),
        parallel_processing=None,
        crossover_type='uniform',
        suppress_warnings=True,
        )
    warnings.filterwarnings('ignore')
    # agg = t.SuiteAggregationStrategy.SUM_FINES    
    agg = t.SuiteAggregationStrategy.MAX_FINE    
    
    training_strategy = strategy.value(ga, agg)
    initial_population = t.Population(networks=[base] + t.Population.random(n=population - 1, architecture=base.architecture(), bound=BOUND).networks)
    session = t.train(training_strategy, population=initial_population)

    solution, solution_fitness, solution_idx = session.ga.best_solution()
    logging.info(f"trained {session.ga.generations_completed} generations of {generations}")
    logging.info(f"best solution found in generation {session.ga.best_solution_generation} with fitness {solution_fitness} (fine {1 / solution_fitness if solution_fitness > 0 else '+inf'})")

    network = NeuralNetwork.from_vector(architecture=architecture, vector=solution)
    network.fitness = base.fitness + session.ga.best_solutions_fitness[1:]

    return network

def estimate_population_generations(architecture: NetworkArchitecture, population: int = None, generations: int = None):
    K = 1.25
    genes = NeuralNetwork.random(architecture=architecture).as_vector().size
    population = population or int(K * 4 * genes)
    generations = generations or population // 4
    return population, generations

@measure_time
def simulate(solutions: SolutionArray, renderer: Renderer, watcher: Watcher, strategy: str, no_dummy: bool):
    parameters = get_testing_simulation_parameters(solutions, renderer, watcher, strategy, no_dummy)
    autosim.simulate(parameters)
    logging.info( f"simulated {parameters.timeout}s ({renderer.fps * parameters.timeout} {renderer.width}x{renderer.height} frames)")

@measure_time
def render(renderer: Renderer, dir: str, name: str):
    path = os.path.join(dir, name)
    path = renderer.render(path)
    logging.info(f"rendered file written to \"{path}\"")

def action_train(args):
    for strategy in args.strategies:
        get_solution(strategy, args.solutions, args.new, args.cont, args.population, args.generations, args.architecture, args.number)
    action_render(args)

def action_info(args):
    path = pathlib.Path(args.solutions)
    if path.is_file():
        files = [path]
    else:
        files = path.glob('**/*.json')

    def print_solution(file):
        @indent
        def print_network(file):
            try:
                network = NeuralNetwork.from_file(file)
                fitness = network.fitness[-1] if isinstance(network.fitness, list) else network.fitness
                logging.info(f"Architecture:   {network.architecture()}")
                logging.info(f"Fitness:        {fitness:.3f} (fine {1 / fitness if fitness > 0 else '+inf':.3f})")
            except:
                logging.warning('Failed to parse!')

        logging.info(f"Found solution: {file.absolute()}")
        print_network(file)

    for file in files:
        print_solution(file)

def action_render(args):

    if args.no_render:
        return

    path: str = args.solutions
    if not path:
        path = '.'

    files = None
    path = pathlib.Path(path)
    if path.is_file():
        files = [path]
    else:
        files = []
        for strategy in args.strategies:
            files += list(path.glob(f"{strategy.name}.json"))

    solutions = [load(file) for file in files]

    if len(solutions) == 0:
        logging.error(f"No solutions found in \"{path}\"")

    for solution, file in zip(solutions, files):
        renderer = Renderer(args.fps, args.width, args.height, args.text)
        watcher = Watcher(None)
        simulate(solution, renderer, watcher, file.stem, args.no_dummy)
        watcher.plot(file.parent / (file.stem + '-watcher' + ('-no-dummy' if args.no_dummy else '') + '.jpg'))
        render(renderer, file.parent, file.stem + ('-no-dummy' if args.no_dummy else ''))

def action_help(parser):
    parser.print_help()
    exit(0)

class Action(SmartEnum):
    train = partial(action_train)
    render = partial(action_render)
    info = partial(action_info)
    help = partial(action_help)

@measure_time
def everything(args):
    try:
        action = Action.from_string(args.action)
    except:
        logging.error(f"Unknown action: {args.action}")
        return
    action.value(args)

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s\t%(levelname)s\t"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def __init__(self):
        self.tab_level = 0

    def format(self, record):
        v = '│  ' * max(0, self.tab_level - 1)
        h = '├─ ' * (1 if self.tab_level > 0 else 0)
        log_fmt = self.FORMATS.get(record.levelno) + v + h + '%(message)s' + CustomFormatter.reset
        if record.filename != '__main__.py' and record.funcName != 'measure_time':
            log_fmt += ' (%(filename)s:%(lineno)d)'
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
    def up(self):
        self.tab_level += 1
    
    def down(self):
        assert self.tab_level > 0
        self.tab_level -= 1

def setup_logging(level):
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    ch.setLevel(level)
    
    logger = logging.getLogger()
    logger.addHandler(ch)
    logger.setLevel(level)

def make_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('action', default=Action.train.name, choices=[item.name for item in Action], help='Action to perform.')
    parser.add_argument('solutions', nargs='?', default='.', help='Folder to search/store solutions.')

    parser.add_argument('-h', '--help', action='store_true', default=False,
                        help='Show this help message and exit.')

    training = parser.add_argument_group('train')
    training.add_argument('-a', '--architecture', default=None,
                          help=
                          'Network architecture in following format: <neurons> (<activation>) [x ...]. '
                          'For example: 4 (relu) x 3 (sigm). '
                          f"Available activations: {', '.join([item.name.lower() for item in ActivationFunction])}. "
                          'If -c is specified and base solution exists, error is emitted. '
                          'Default architecture is linear (sigmoid at the end is always presented for normalization). '
                          )
    training.add_argument('-n', '--new', action='store_true', default=False,
                          help='Generate new solutions instead of using existing.')
    training.add_argument('-c', '--continue', action='store_true', default=False, dest='cont',
                          help='Use existing solutions as initial population.')
    training.add_argument('-s', '--strategies', action='append', default=None, choices=[item.name for item in Strategy],
                          help='Comma-separated list of strategies to train. All strategies except null are enabled by default.')
    training.add_argument('-p', '--population', default=None, type=int,
                          help='Size of population.')
    training.add_argument('-g', '--generations', default=None, type=int,
                          help='Number of generations.')
    training.add_argument('-m', '--number', default=1, type=int,
                           help='Number of different solutions to generate.')

    rendering = parser.add_argument_group('render')
    rendering.add_argument('-d', '--no-render', action='store_true', default=False,
                           help='Skip rendering solutions.')
    rendering.add_argument('--fps', default=24, type=int,
                           help='Frames per second.')
    rendering.add_argument('--width', default=768, type=int,
                           help='Picture width.')
    rendering.add_argument('--height', default=None,
                           help='Picture height. Equals to width by default.')
    rendering.add_argument('-t', '--text', default='n,x,u,vkh',
                           help='Rendered text. Available options: n,x,u,vms,vkh.')
    rendering.add_argument('--no-dummy', default=False, action='store_true',
                           help='Do not use dummy for simulations.')

    other = parser.add_argument_group('other')
    other.add_argument('-l', '--log-level', default='info',
                           choices=['critical', 'fatal', 'error',
                                    'warning', 'info', 'debug'],
                           help='Logging level.')

    return parser

def parse_args(argv):
    parser = make_parser()
    args = parser.parse_args(argv[1:])

    if args.help or args.action == 'help':
        action_help(parser)

    args.solutions = os.path.abspath(args.solutions)

    args.strategies = coalesce(args.strategies, [item.name for item in Strategy if item != Strategy.null])
    args.strategies = [Strategy.from_string(string) for string in args.strategies]

    if args.height is None:
        args.height = args.width

    args.log_level = logging.getLevelNamesMapping()[args.log_level.upper()]

    if args.architecture is not None:
        args.architecture = args.architecture.strip()

    return args

def main(argv):
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        everything(args)
    except Exception as e:
        raise RuntimeError('Uncaught exception!') from e

    return 0

if __name__ == '__main__':
    ret = main(sys.argv)
    exit(ret)
