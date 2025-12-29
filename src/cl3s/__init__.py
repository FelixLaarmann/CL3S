"""
cl3s: Combinatory Logic Search Space Synthesizer

This package provides utilities for working with derivation trees,
search spaces, and genetic programming operations such as crossover and mutation.
"""
from .tree import DerivationTree
from .search_space import SearchSpace
from .synthesizer import SearchSpaceSynthesizer

from cosy.specification_builder import SpecificationBuilder
from cosy.subtypes import Subtypes, Taxonomy
from cosy.synthesizer import Specification
from cosy.types import Arrow, Constructor, Intersection, Literal, Omega, Type, Var, Group, DataGroup

from .evolutionary_search import TournamentSelection
from .bayesian_optimization import BayesianOptimization
from .acquisition_function import (SimplifiedExpectedImprovement, ExpectedImprovement,
                                   EvolutionaryAcquisitionFunctionOptimization)
from .graph_kernel import WeisfeilerLehmanKernel

__version__ = "0.0.1"

__all__ = [
    "SpecificationBuilder",
    "Literal",
    "Var",
    "Subtypes",
    "Type",
    "Omega",
    "Constructor",
    "Arrow",
    "Intersection",
    "SearchSpace",
    "DerivationTree",
    "SearchSpaceSynthesizer",
    "Specification",
    "Taxonomy",
    "Group",
    "DataGroup",
    "TournamentSelection",
    "BayesianOptimization",
    "ExpectedImprovement",
    "EvolutionaryAcquisitionFunctionOptimization",
    "SimplifiedExpectedImprovement",
    "WeisfeilerLehmanKernel",
]

