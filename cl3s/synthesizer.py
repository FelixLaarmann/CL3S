from __future__ import annotations

from collections.abc import (
    Callable,
    Container,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from re import search
from typing import (
    Any,
    Generic,
    TypeVar,
)

from cosy.synthesizer import Synthesizer, Specification, ParameterSpace, Taxonomy, Type

from .search_space import SearchSpace

# type of components
C = TypeVar("C", bound=Hashable)

class SearchSpaceSynthesizer(Synthesizer[C], Generic[C]):
    """
    A synthesizer for search spaces, which can be used to generate
    combinatory logic expressions or other structures based on a search space.
    """

    def __init__(
            self,
            component_specifications: Mapping[C, Specification],
            parameter_space: ParameterSpace | None = None,
            taxonomy: Taxonomy | None = None,
    ):
        super().__init__(
            component_specifications=component_specifications,
            parameter_space=parameter_space,
            taxonomy=taxonomy,
        )

    def construct_search_space(self, *targets: Type) -> SearchSpace[Type, C, str]:
        """Constructs a logic program in the current environment for the given target types."""

        search_space: SearchSpace[Type, C, str] = SearchSpace()
        for nt, rule in self.construct_solution_space_rules(*targets):
            search_space.add_rule(nt, rule.terminal, rule.arguments, rule.predicates)

        return search_space
