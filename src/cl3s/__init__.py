"""
cl3s: Combinatory Logic Search Space Synthesizer

This package provides utilities for working with derivation trees,
search spaces, and genetic programming operations such as crossover and mutation.
"""
from cosy.solution_space import SolutionSpace

from .tree import DerivationTree
from .search_space import SearchSpace
from .synthesizer import SearchSpaceSynthesizer

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
]


T = TypeVar("T", bound=Hashable)

class CL3S(Generic[T]):
    component_specifications: Mapping[T, Specification]
    taxonomy: Taxonomy | None = None
    _synthesizer: SearchSpaceSynthesizer

    def __init__(
        self,
        component_specifications: Mapping[T, Specification],
        taxonomy: Taxonomy | None = None,
    ) -> None:
        self.component_specifications = component_specifications
        self.taxonomy = taxonomy if taxonomy is not None else {}
        self._synthesizer = SearchSpaceSynthesizer(component_specifications, None, self.taxonomy)

    def solve(self, query: Type, max_count: int = 100) -> Iterable[Any]:
        """
        Solves the given query by constructing a solution space and enumerating and interpreting the resulting trees.

        :param query: The query to solve.
        :param max_count: The maximum number of trees to enumerate.
        :return: An iterable of interpreted trees.
        """
        if not isinstance(query, Type):
            msg = "Query must be of type Type"
            raise TypeError(msg)
        search_space = self._synthesizer.construct_search_space(query).prune()

        trees = search_space.enumerate_trees(query, max_count=max_count)
        for tree in trees:
            yield tree.interpret()

    def sample(self, query: Type, max_count: int = 100) -> Iterable[Any]:
        """
        Samples the given query by constructing a solution space and sampling the resulting trees.

        :param query: The query to sample.
        :param max_count: The maximum number of trees to sample.
        :return: An iterable of interpreted trees.
        """
        if not isinstance(query, Type):
            msg = "Query must be of type Type"
            raise TypeError(msg)
        search_space = self._synthesizer.construct_search_space(query).prune()

        trees = search_space.sample(max_count, query)
        for tree in trees:
            yield tree.interpret()
