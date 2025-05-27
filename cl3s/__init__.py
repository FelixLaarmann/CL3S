"""
cl3s: Combinatory Logic Search Space Synthesizer

This package provides utilities for working with derivation trees,
search spaces, and genetic programming operations such as crossover and mutation.
"""

from .tree import DerivationTree
from .search_space import SearchSpace
from .synthesizer import SearchSpaceSynthesizer

__version__ = "0.0.1"

__all__ = [
    "SearchSpace",
    "DerivationTree",
    "SearchSpaceSynthesizer",
]
