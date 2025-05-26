# cl3s/search_space.py

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, Optional, TypeVar, overload
import typing
from dataclasses import dataclass, field

from clsp.solution_space import SolutionSpace, RHSRule

from .core import SearchSpace, DerivationTree, NT, T

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals


@dataclass
class SearchSpace(SolutionSpace[NT, T], Generic[NT, T]):
    """
    This class extends the SolutionSpace to provide additional methods
    for searching and manipulating the solution space.
    It relies on DerivationTrees instead of Trees.
    """

    # TODO: Most methods must be overwritten for DerivationTrees.

    def sample_tree(self, non_terminal: NT, max_depth: int | None = None) -> DerivationTree[NT, T]:
        """
        Sample a random tree from the search space.
        """
        # TODO
        raise NotImplementedError("This method still needs to be implemented.")

    def sample(self, size: int, non_terminal: NT, max_depth: int | None = None) -> list[DerivationTree[NT, T]]:
        """
        Sample a list of random trees from the search space.
        """
        # TODO
        raise NotImplementedError("This method still needs to be implemented.")

