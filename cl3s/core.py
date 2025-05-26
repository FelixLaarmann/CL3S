# cl3s/core.py

from collections.abc import Hashable
from typing import TypeVar, Generic

NT = TypeVar("NT", bound=Hashable)  # type of non-terminals
T = TypeVar("T", bound=Hashable)    # type of terminals

class SearchSpace(Generic[NT, T]):
    pass

class DerivationTree(Generic[NT, T]):
    pass
