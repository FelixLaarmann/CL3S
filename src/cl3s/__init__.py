"""
cl3s: Combinatory Logic Search Space Synthesizer

This package provides utilities for working with derivation trees,
search spaces, and genetic programming operations such as crossover and mutation.
"""
from cosy.solution_space import SolutionSpace

from .tree import DerivationTree
from .search_space import SearchSpace
from .synthesizer import SearchSpaceSynthesizer

from .genetic_programming.evolutionary_search import TournamentSelection
from .scikit.acquisition_function import ExpectedImprovement
from .scikit.graph_kernel import WeisfeilerLehmanKernel
from .scikit.bayesian_optimization import BayesianOptimization

from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Generic, TypeVar, Hashable

from cosy.specification_builder import SpecificationBuilder
from cosy.subtypes import Subtypes, Taxonomy
from cosy.synthesizer import Specification
from cosy.types import Arrow, Constructor, Intersection, Literal, Omega, Type, Var, Group, DataGroup


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
    "ExpectedImprovement",
    "WeisfeilerLehmanKernel",
    "BayesianOptimization",
]

