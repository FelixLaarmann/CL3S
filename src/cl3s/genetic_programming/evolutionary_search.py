import random
from copy import deepcopy, copy

from typing import Any, Generic, Optional, TypeVar, Union, Generator
import typing

from collections.abc import Callable, Hashable, Sequence

from ..tree import DerivationTree
from ..search_space import SearchSpace

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names

class EvolutionarySearch(Generic[NT, T, G]):
    def __init__(self, search_space: SearchSpace[NT, T, G], request: NT,
                 fitness_function,
                 population_size: int=100, reproduction_rate: float=0.5, generation_limit: int=5,
                 greater_is_better: bool=False):
        self.search_space = search_space
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.reproduction_rate = reproduction_rate
        self.generation_limit = generation_limit
        self.greater_is_better = greater_is_better
        self.request = request
        self.stale_generations = 0

    def initialize_population(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def selection(self, fitnesses):
        raise NotImplementedError("Subclasses must implement this method.")

    def reproduction(self, fitnesses):
        raise NotImplementedError("Subclasses must implement this method.")

    def check_termination_criteria(self, old_fitnesses, new_fitnesses):
        raise NotImplementedError("Subclasses must implement this method.")

    def optimize(self):
        current_generation = self.initialize_population()
        fitnesses = {t: self.fitness_function(t) for t in current_generation}
        n = 1
        while n <= self.generation_limit:
            # next_generation = self.reproduction(fitnesses)
            new_fitnesses = self.reproduction(fitnesses)  # {t: self.fitness_function(t) for t in next_generation}
            if self.check_termination_criteria(fitnesses, new_fitnesses):
                break
            fitnesses = new_fitnesses
            n += 1
        if self.greater_is_better:
            best_tree = max(fitnesses, key=fitnesses.get)
        else:
            best_tree = min(fitnesses, key=fitnesses.get)
        return best_tree

class TournamentSelection(EvolutionarySearch[NT, T, G], Generic[NT, T, G]):
    def __init__(self, search_space: SearchSpace[NT, T, G], request: NT, fitness_function,
                 population_size: int = 20, reproduction_rate: float = 0.1, generation_limit: int = 5,
                 tournament_size: int = 3, greater_is_better: bool = False):
        super().__init__(search_space, request, fitness_function, population_size, reproduction_rate, generation_limit,
                         greater_is_better)
        self.tournament_size = tournament_size

    def initialize_population(self):
        return self.search_space.sample(self.population_size, self.request)

    def selection(self, fitnesses: dict[DerivationTree[NT, T, G], float]):
        selected: set[DerivationTree[NT, T, G]] = set()
        #for _ in range(int(self.population_size * self.reproduction_rate)):
        while len(selected) < 2:
            tournament = random.sample(list(fitnesses.items()), self.tournament_size)
            if self.greater_is_better:
                winner = max(tournament, key=lambda x: x[1])[0]
            else:
                winner = min(tournament, key=lambda x: x[1])[0]
            selected.add(winner)
        parent1 = selected.pop()
        parent2 = selected.pop()
        return parent1, parent2

    def reproduction(self, fitnesses: dict[DerivationTree[NT, T, G], float]):
        fit = copy(fitnesses)
        next_generation: set[DerivationTree[NT, T, G]] = set()
        current_generation = list(fit.keys())
        while len(next_generation) < int(self.population_size * self.reproduction_rate):
            parent1, parent2 = self.selection(fit)
            child = parent1.crossover(parent2, self.search_space)
            if child is not None:
                child = child.mutate(self.search_space)
            while child is None:
                parent1, parent2 = self.selection(fit)
                child = parent1.crossover(parent2, self.search_space)
                if child is not None:
                    child = child.mutate(self.search_space)
            if child not in current_generation:
                next_generation.add(child)
        # replace the worst len(next_generation) individuals from current generation
        # natural selection
        for _ in range(len(next_generation)):
            if self.greater_is_better:
                worst = min(fit.keys(), key=lambda t: fit[t])
            else:
                worst = max(fit.keys(), key=lambda t: fit[t])
            fit.pop(worst)
        new_fitnesses = {t: self.fitness_function(t) for t in next_generation}
        result = fit | new_fitnesses
        return result

    def check_termination_criteria(self, old_fitnesses, new_fitnesses):
        old_best = max(old_fitnesses.values())
        new_best = max(new_fitnesses.values())
        if self.greater_is_better:
            if new_best <= old_best:
                self.stale_generations += 1
        else:
            if new_best >= old_best:
                self.stale_generations += 1
        if self.stale_generations >= int(self.generation_limit/10 + 1):
            return True
        return False

