"""
cl3s: Combinatory Logic Search Space Synthesizer

This package provides utilities for working with derivation trees,
search spaces, and genetic programming operations such as crossover and mutation.
"""

from .core import SearchSpace, DerivationTree, NT, T

__version__ = "0.0.1"

__all__ = [
    "SearchSpace",
    "DerivationTree",
    "NT",
    "T",
]
