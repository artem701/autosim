
from dummy import DummyCar
from enum import Enum
from functools import partial
import autosim
from autosim.car.ncar import NetworkArchitecture, NeuralNetwork
from autosim.nn import DenseLayerArchitecture, ActivationFunction
import autosim.training as t
import autosim.estimation as e
import autosim.simulation as s
import autosim.car as c
from eventloop.eventloop import Event
from helpers import kph_to_mps, measure_time, not_implemented, coalesce, indent
from renderer import Renderer
from simulation import Environment, Body
from simulation.location import Circle, CircleSpace, Line
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import sys

import logging

L = 500
T_TRAINING = 60
# ARCHITECTURE = NetworkArchitecture([DenseLayerArchitecture(4, ActivationFunction.SIGM)] * 1, ActivationFunction.SIGM)
ARCHITECTURE = NetworkArchitecture([], ActivationFunction.SIGM)
BOUND = (-1, 1)


def get_testing_simulation_parameters(solution, renderer):
    CARS_TESTED = 5
    T_RENDER = 3 * T_TRAINING
    students = [c.NCar(network=solution,
                       name=f"student-{i}",
                       location = Circle(CircleSpace(L), i * L / (CARS_TESTED + 1)))
                for i in range(CARS_TESTED)]            
    
    dummy = DummyCar(location = Circle(CircleSpace(L), CARS_TESTED * L / (CARS_TESTED + 1)), name = 'dummy')
    return s.SimulationParameters(timeout=T_RENDER, objects = [*students, dummy, renderer])

def strategy_value(func):
    def wrapper(ga, agg):
        return t.TrainingStrategy(func(), ga, agg)
    return wrapper

@strategy_value
def get_null_training_strategy():
    return []

@strategy_value
def get_cautious_training_strategy():
    optimal_distance = L // 6
    estimation_strategy = e.EstimationStrategy(collision=e.Criteria(10000), distance=e.ReferenceCriteria(fine=0.01, reference=optimal_distance), speed=e.LessCriteria(0.01, kph_to_mps(60)))
    return [
        # learn to stop
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar('0', c.ACar.Mode.MOVEMENT, location=Line(optimal_distance // 6), name='trainer')]
                            )
                        ),

        # follow constantly moving target
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(f"{kph_to_mps(60)} * dt", c.ACar.Mode.MOVEMENT, location=Line(optimal_distance), name='trainer')]
                            )
                        ),

        # # just follow trainer
        # t.TrainingSuite(estimation=estimation_strategy,
        #                 simulation=s.SimulationParameters(
        #                     timeout=T_TRAINING,
        #                     objects=[DummyCar(location=Line(optimal_distance), name='trainer')]
        #                     )
        #                 ),
        ]

@strategy_value
def get_speedy_training_strategy():
    # estimation_strategy = e.EstimationStrategy(collision=e.Criteria(10000), speed=e.ReferenceCriteria(fine=0.01, reference=kph_to_mps(120)))
    estimation_strategy = e.EstimationStrategy(collision=e.Criteria(10000), distance=e.LessCriteria(fine=0.01, reference=10) + e.MoreCriteria(fine=0.05, reference=5), speed=e.LessCriteria(1, kph_to_mps(60)))
    return [
        # learn to stop
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar('0', c.ACar.Mode.MOVEMENT, location=Line(L // 6), name='trainer')]
                            )
                        ),

        # follow constantly moving target
        t.TrainingSuite(estimation=estimation_strategy,
                        simulation=s.SimulationParameters(
                            timeout=T_TRAINING,
                            objects=[c.ACar(f"{kph_to_mps(60)} * dt", c.ACar.Mode.MOVEMENT, location=Line(L // 6), name='trainer')]
                            )
                        ),
        
        # t.TrainingSuite(estimation=estimation_strategy,
        #                 simulation=s.SimulationParameters(
        #                     timeout=T_TRAINING,
        #                     objects=[c.ACar('0', c.ACar.Mode.MOVEMENT, location=Line(1000), name='trainer')]
        #                     )
        #                 ),
        # t.TrainingSuite(estimation=estimation_strategy,
        #                 simulation=s.SimulationParameters(
        #                     timeout=T_TRAINING,
        #                     objects=[c.ACar(f"{kph_to_mps(60)} * dt", c.ACar.Mode.MOVEMENT, location=Line(100), name='trainer')]
        #                     )
        #                 ),
        # t.TrainingSuite(estimation=estimation_strategy,
        #                 simulation=s.SimulationParameters(
        #                     timeout=T_TRAINING,
        #                     objects=[]
        #                     )
        #                ),
        # t.TrainingSuite(estimation=estimation_strategy,
        #                 simulation=s.SimulationParameters(
        #                     timeout=T_TRAINING,
        #                     objects=[DummyCar(location=Line(100), name='trainer')]
        #                     )
        #                 ),
        ]

class Strategy(Enum):
    null = partial(get_null_training_strategy)
    cautious = partial(get_cautious_training_strategy)
    speedy = partial(get_speedy_training_strategy)

    @staticmethod
    def from_string(string: str):
        for item in Strategy:
            if item.name == string:
                return item
        return None

def plot_fitness(path, session: t.TrainingSession):
    fitness = [1 / fitness for fitness in session.ga.best_solutions_fitness[1:]]
    plt.clf()
    plt.title('Fitness over generations')
    plt.plot(fitness)
    plt.xlabel('generation')
    plt.ylabel('fine')
    plt.figtext(0.01, 0.01, f"population: {session.ga.sol_per_pop}\ngenerations: {session.ga.generations_completed}\narchitecture: {session.architecture}", fontsize=12, ha='left')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(path)

@indent
def get_solution(strategy: Strategy, path: str, new: bool, cont: bool, population: int = None, generations: int = None):
    solution_path = f"{path}/{strategy.name}.json"
    fitness_path = f"{path}/{strategy.name}.jpg"

    logging.info(f"get solution for {strategy.name} strategy")
    
    solution = None
    if cont or not new:
        solution = load(solution_path)
    
    if solution is None:
        logging.info('continue with random base solution')
        solution = random_solution()
        new = True
    
    if cont or new:
        if cont:
            logging.info(f"train solution from base: {solution_path}")
        solution, session = train(solution, strategy, population, generations)
        os.makedirs(path, exist_ok=True)
        solution.to_file(solution_path)
        logging.info(f"written {strategy.name} solution to {solution_path}")

        plot_fitness(fitness_path, session)        
        logging.info(f"saved {strategy.name} fitness plot to {fitness_path}")

    return solution

def load(path: str):
    if not os.path.isfile(path):
        logging.warning(f"solution file {path} is not found!")
        return None
    else:
        logging.info(f"load solution from {os.path.realpath(path)}...")
        return NeuralNetwork.from_file(path)

def random_solution():
    return NeuralNetwork.from_vector(ARCHITECTURE, NeuralNetwork.random(ARCHITECTURE, BOUND).as_vector())

@measure_time
def train(base: NeuralNetwork, strategy: Strategy, population: int = None, generations: int = None):
    
    population, generations = estimate_population_generations(base.architecture(), population, generations)
    logging.info(f"Algorithm genes = {NeuralNetwork.random(architecture=base.architecture()).as_vector().size}, population = {population}, generations = {generations}. Network architecture = {{ {base.architecture()} }}")

    ga = t.GeneticAlgorithmParameters(
        num_generations=generations, num_parents_mating=population//2,
        on_start=lambda ga:logging.info(f"genetic algorithm started"),
        on_stop=lambda ga,fitnesses:logging.info(f"genetic algorithm stopped"),
        on_generation=lambda ga:logging.info(f"finished {ga.generations_completed:3>}/{generations} generations (fine: {1 / np.max(ga.last_generation_fitness):.3f})"),
        parallel_processing=('thread', 4),
        )
    # agg = t.SuiteAggregationStrategy.SUM_FINES    
    agg = t.SuiteAggregationStrategy.MAX_FINE    
    
    training_strategy = strategy.value(ga, agg)
    initial_population = t.Population(networks=[base] + t.Population.random(n=population - 1, architecture=base.architecture(), bound=BOUND).networks)
    session = t.train(training_strategy, population=initial_population)

    solution, solution_fitness, solution_idx = session.ga.best_solution()
    logging.info(f"trained {session.ga.generations_completed} generations of {generations}")
    logging.info(f"best solution found in generation {session.ga.best_solution_generation} with fitness {solution_fitness} (fine {1 / solution_fitness if solution_fitness > 0 else '+inf'})")

    network = NeuralNetwork.from_vector(architecture=ARCHITECTURE, vector=solution)
    network.fitness = solution_fitness

    return network, session

def estimate_population_generations(architecture: NetworkArchitecture, population: int = None, generations: int = None):
    K = 1.25
    genes = NeuralNetwork.random(architecture=architecture).as_vector().size
    population = population or int(K * 4 * genes)
    generations = generations or population // 4
    return population, generations

@measure_time
def simulate(network: NeuralNetwork, renderer: Renderer):
    parameters = get_testing_simulation_parameters(network, renderer)
    autosim.simulate(parameters)
    logging.info( f"simulated {parameters.timeout}s ({renderer.fps * parameters.timeout} {renderer.width}x{renderer.height} frames)")

@measure_time
def render(renderer: Renderer, dir: str, name: str):
    path = os.path.join(dir, name)
    path = renderer.render(path)
    logging.info(f"rendered file written to {path}")

def action_train(args):
    for strategy in args.strategies:
        get_solution(strategy, args.solutions, args.new, args.cont, args.population, args.generations)
    action_render(args)

def action_info(args):
    path = pathlib.Path(args.solutions)
    if path.is_file():
        files = [path]
    else:
        files = path.glob('**/*.json')

    def print_solution(file):
        @indent
        def print_network(network):
            logging.info(f"Architecture:   {network.architecture()}")
            logging.info(f"Fitness:        {network.fitness}")

        logging.info(f"Found solution: {file.absolute()}")
        print_network(NeuralNetwork.from_file(file))

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
        files = list(path.glob('*.json'))

    solutions = [load(file) for file in files]

    for solution, file in zip(solutions, files):
        renderer = Renderer(args.fps, args.width, args.height)
        simulate(solution, renderer)
        render(renderer, file.parent, file.stem)

class Action(Enum):
    train = partial(action_train)
    render = partial(action_render)
    info = partial(action_info)

    @staticmethod
    def from_string(string: str):
        for item in Action:
            if item.name == string:
                return item
        raise RuntimeError(f"No action {string}")

@measure_time
def everything(args):
    action = Action.from_string(args.action)
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

    rendering = parser.add_argument_group('render')
    rendering.add_argument('-d', '--no-render', action='store_true', default=False,
                           help='Skip rendering solutions.')
    rendering.add_argument('--fps', default=24,
                           help='Frames per second.')
    rendering.add_argument('--width', default=768,
                           help='Picture width.')
    rendering.add_argument('--height', default=None,
                           help='Picture height. Equals to width by default.')

    other = parser.add_argument_group('other')
    other.add_argument('-l', '--log-level', default='info',
                           choices=['critical', 'fatal', 'error',
                                    'warning', 'info', 'debug'],
                           help='Logging level.')

    return parser

def parse_args(argv):
    parser = make_parser()
    args = parser.parse_args(argv[1:])

    if args.help:
        parser.print_help()
        exit(0)

    args.solutions = os.path.abspath(args.solutions)

    args.strategies = coalesce(args.strategies, [item.name for item in Strategy if item != Strategy.null])
    args.strategies = [Strategy.from_string(string) for string in args.strategies]

    if args.height is None:
        args.height = args.width

    args.log_level = logging.getLevelNamesMapping()[args.log_level.upper()]

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
