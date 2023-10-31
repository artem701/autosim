from dataclasses import dataclass, field
from typing import Any

from autosim.car.ncar import NetworkArchitecture, NeuralNetwork, NCar, Friction
from autosim.car.specs import Characteristics, TEST
from autosim.estimation import EstimationStrategy
from autosim.simulation import SimulationParameters
from helpers import Serializable
import pygad.gann


@dataclass
class GeneticAlgorithmParameters:
   """For detailed documentation refer to pygad docs.
   """
   num_generations: int
   """Number of generations.
   """
   num_parents_mating: int
   """Number of solutions to be selected as parents.
   """
   parent_selection_type: str = "sss",
   """The parent selection type. Supported types are:
      sss (for steady-state selection)
      rws (for roulette wheel selection)
      sus (for stochastic universal selection)
      rank (for rank selection)
      random (for random selection)
      tournament (for tournament selection)
   """
   keep_parents: int = -1,
   """Number of parents to keep in the current population. -1 (default) means to keep all parents in the next population.
   """
   keep_elitism: int = 1,
   """It can take the value 0 or a positive integer that satisfies (0 <= keep_elitism <= sol_per_pop). It defaults to 1 which means only the best solution in the current generation is kept in the next generation. If assigned 0, this means it has no effect. If assigned a positive integer K, then the best K solutions are kept in the next generation. It cannot be assigned a value greater than the value assigned to the sol_per_pop parameter. If this parameter has a value different than 0, then the keep_parents parameter will have no effect.
   """
   K_tournament: int = 3,
   """In case that the parent selection type is tournament, the K_tournament specifies the number of parents participating in the tournament selection.
   """
   crossover_type: str = "single_point",
   """Type of the crossover operation. Supported types are:
      single_point (for single-point crossover)
      two_points (for two points crossover)
      uniform (for uniform crossover)
      scattered (for scattered crossover)
   """
   crossover_probability: Any | None = None,
   """The probability of selecting a parent for applying the crossover operation. Its value must be between 0.0 and 1.0 inclusive. For each parent, a random value between 0.0 and 1.0 is generated. If this random value is less than or equal to the value assigned to the crossover_probability parameter, then the parent is selected."""
   mutation_type: str = "random",
   """Type of the mutation operation. Supported types are:
      random (for random mutation)
      swap (for swap mutation)
      inversion (for inversion mutation)
      scramble (for scramble mutation)
      adaptive (for adaptive mutation).
   """
   mutation_probability: Any | None = None,
   """The probability of selecting a gene for applying the mutation operation. Its value must be between 0.0 and 1.0 inclusive. For each gene in a solution, a random value between 0.0 and 1.0 is generated. If this random value is less than or equal to the value assigned to the mutation_probability parameter, then the gene is selected. If this parameter exists, then there is no need for the 2 parameters mutation_percent_genes and mutation_num_genes."""
   mutation_by_replacement: bool = False,
   """An optional bool parameter. It works only when the selected type of mutation is random (mutation_type="random"). In this case, mutation_by_replacement=True means replace the gene by the randomly generated value. If False, then it has no effect and random mutation works by adding the random value to the gene."""
   mutation_percent_genes: str = 'default',
   """Percentage of genes to mutate. It defaults to the string "default" which is later translated into the integer 10 which means 10% of the genes will be mutated. It must be >0 and <=100. Out of this percentage, the number of genes to mutate is deduced which is assigned to the mutation_num_genes parameter. The mutation_percent_genes parameter has no action if mutation_probability or mutation_num_genes exist."""
   mutation_num_genes: Any | None = None,
   """Number of genes to mutate which defaults to None meaning that no number is specified. The mutation_num_genes parameter has no action if the parameter mutation_probability exists."""
   random_mutation_min_val: float = -1,
   """For random mutation, the random_mutation_min_val parameter specifies the start value of the range from which a random value is selected to be added to the gene. It defaults to -1."""
   random_mutation_max_val: float = 1,
   """For random mutation, the random_mutation_max_val parameter specifies the end value of the range from which a random value is selected to be added to the gene. It defaults to +1."""
   gene_space: Any | None = None,
   """It is used to specify the possible values for each gene in case the user wants to restrict the gene values. It is useful if the gene space is restricted to a certain range or to discrete values. It accepts a list, range, or numpy.ndarray. When all genes have the same global space, specify their values as a list/tuple/range/numpy.ndarray. For example, gene_space = [0.3, 5.2, -4, 8] restricts the gene values to the 4 specified values. If each gene has its own space, then the gene_space parameter can be nested like [[0.4, -5], [0.5, -3.2, 8.2, -9], ...] where the first sublist determines the values for the first gene, the second sublist for the second gene, and so on. If the nested list/tuple has a None value, then the gene’s initial value is selected randomly from the range specified by the 2 parameters init_range_low and init_range_high and its mutation value is selected randomly from the range specified by the 2 parameters random_mutation_min_val and random_mutation_max_val. gene_space is added in PyGAD 2.5.0. Check the Release History of PyGAD 2.5.0 section of the documentation for more details. In PyGAD 2.9.0, NumPy arrays can be assigned to the gene_space parameter. In PyGAD 2.11.0, the gene_space parameter itself or any of its elements can be assigned to a dictionary to specify the lower and upper limits of the genes. For example, {'low': 2, 'high': 4} means the minimum and maximum values are 2 and 4, respectively. In PyGAD 2.15.0, a new key called "step" is supported to specify the step of moving from the start to the end of the range specified by the 2 existing keys "low" and "high"."""
   allow_duplicate_genes: bool = True,
   """If True, then a solution/chromosome may have duplicate gene values. If False, then each gene will have a unique value in its solution."""
   on_start: Any | None = None,
   on_fitness: Any | None = None,
   on_parents: Any | None = None,
   on_crossover: Any | None = None,
   on_mutation: Any | None = None,
   on_generation: Any | None = None,
   on_stop: Any | None = None,
   suppress_warnings: bool = False,
   """Suppress warning messages.
   """
   stop_criteria: Any | None = None,
   """Some criteria to stop the evolution. Added in PyGAD 2.15.0. Each criterion is passed as str which has a stop word. The current 2 supported words are reach and saturate. reach stops the run() method if the fitness value is equal to or greater than a given fitness value. An example for reach is "reach_40" which stops the evolution if the fitness is >= 40. saturate means stop the evolution if the fitness saturates for a given number of consecutive generations. An example for saturate is "saturate_7" which means stop the run() method if the fitness does not change for 7 consecutive generations.
   """
   parallel_processing: Any | None = None,
   """If None (Default), this means no parallel processing is applied. It can accept a list/tuple of 2 elements [1) Can be either 'process' or 'thread' to indicate whether processes or threads are used, respectively., 2) The number of processes or threads to use.]. For example, parallel_processing=['process', 10] applies parallel processing with 10 processes. If a positive integer is assigned, then it is used as the number of threads. For example, parallel_processing=5 uses 5 threads which is equivalent to parallel_processing=["thread", 5]. For more information, check the Parallel Processing in PyGAD section."""
   random_seed: Any | None = None,
   """It defines the random seed to be used by the random function generators (we use random functions in the NumPy and random modules). This helps to reproduce the same results by setting the same random seed (e.g. random_seed=2). If given the value None, then it has no effect."""
   logger: Any | None = None
   """Accepts an instance of the logging.Logger class to log the outputs."""


