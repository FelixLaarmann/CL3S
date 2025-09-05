from typing import Any, Generic, Optional, TypeVar, Union, Generator
import typing

from collections.abc import Callable, Hashable, Sequence

from ..tree import DerivationTree
from ..search_space import SearchSpace

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names

class EvolutionarySearch(Generic[NT, T, G]):
    def __init__(self, search_space, fitness_function, population_size=100, reproduction_rate=0.5, generation_limit=100):
        self.search_space = search_space
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.reproduction_rate = reproduction_rate
        self.generation_limit = generation_limit

    def initialize_population(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def selection(self, fitnesses):
        raise NotImplementedError("Subclasses must implement this method.")

    def reproduction(self, fitnesses, selected):
        raise NotImplementedError("Subclasses must implement this method.")

    def check_termination_criteria(self, old_fitnesses, new_fitnesses):
        raise NotImplementedError("Subclasses must implement this method.")

    def optimize(self, greater_is_better=False):
        current_generation = self.initialize_population()
        fitnesses = {t: self.fitness_function(t) for t in current_generation}
        n = 1
        while n <= self.generation_limit:
            selected = self.selection(fitnesses)
            next_generation = self.reproduction(fitnesses, selected)
            new_fitnesses = {t: self.fitness_function(t) for t in next_generation}
            if self.check_termination_criteria(fitnesses, new_fitnesses):
                break
            current_generation = next_generation
            fitnesses = new_fitnesses
            n += 1
        if greater_is_better:
            best_tree = max(fitnesses, key=fitnesses.get)
        else:
            best_tree = min(fitnesses, key=fitnesses.get)
        return best_tree
