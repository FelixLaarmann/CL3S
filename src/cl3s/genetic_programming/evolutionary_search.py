from numpy import random as nrandom
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
                 population_size: int=100, crossover_rate: float=0.8, mutation_rate: float= 0.05,
                 generation_limit: int=100, elitism: int=2,
                 greater_is_better: bool=False):
        self.search_space = search_space
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation_limit = generation_limit
        self.greater_is_better = greater_is_better
        self.elitism = elitism
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
    def __init__(self, search_space: SearchSpace[NT, T, G], request: NT,
                 fitness_function,
                 population_size: int=100, tournament_size: int = 5, crossover_rate: float=0.8, mutation_rate: float= 0.05,
                 generation_limit: int=100, elitism: int=2,
                 greater_is_better: bool=False, enforce_diversity: bool = False):
        super().__init__(search_space, request, fitness_function, population_size, crossover_rate, mutation_rate,
                         generation_limit, elitism, greater_is_better)
        self.tournament_size = tournament_size
        self.enforce_diversity = enforce_diversity

    def initialize_population(self):
        return self.search_space.sample(self.population_size, self.request)

    def selection(self, fitnesses: dict[DerivationTree[NT, T, G], float]):
        if self.enforce_diversity:
            selected: set[DerivationTree[NT, T, G]] = set()
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
        else:
            selected: list[DerivationTree[NT, T, G]] = []
            for _ in range(2):
                tournament = random.sample(list(fitnesses.items()), self.tournament_size)
                if self.greater_is_better:
                    winner = max(tournament, key=lambda x: x[1])[0]
                else:
                    winner = min(tournament, key=lambda x: x[1])[0]
                selected.append(winner)
            parent1 = selected.pop()
            parent2 = selected.pop()
            return parent1, parent2

    def reproduction(self, fitnesses: dict[DerivationTree[NT, T, G], float]):
        fit = copy(fitnesses)
        if self.enforce_diversity:
            next_generation: set[DerivationTree[NT, T, G]] = set()
            current_generation = list(fit.keys())
            # elitism
            elite = sorted(fit.items(), key=lambda x: x[1], reverse=self.greater_is_better)[:self.elitism]
            for e in elite:
                next_generation.add(e[0])  # invariant diversity: there are no duplicates in population
            while len(next_generation) < int(self.population_size):
                parent1, parent2 = self.selection(fit)
                cx = nrandom.choice([True, False], p=[self.crossover_rate, 1 - self.crossover_rate], size=1).item()
                assert isinstance(cx, bool)
                if cx:
                    child = parent1.crossover(parent2, self.search_space)
                    while child is None:
                        parent1, parent2 = self.selection(fit)  # if parents are impotent... TODO: is this possible for crossover?
                        child = parent1.crossover(parent2, self.search_space)
                    mt = nrandom.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate], size=1).item()
                    assert isinstance(mt, bool)
                    if mt:
                        child = child.mutate(self.search_space)
                    if child not in current_generation:
                        next_generation.add(child)
                else:
                    mt = nrandom.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate], size=1).item()
                    assert isinstance(mt, bool)
                    if mt:
                        mutant1 = parent1.mutate(self.search_space)
                        mutant2 = parent2.mutate(self.search_space)
                        if mutant1 not in current_generation:
                                next_generation.add(mutant1)
                        if len(next_generation) < int(self.population_size):
                                if mutant2 not in current_generation:
                                    next_generation.add(mutant2)
                    else:
                        if len(next_generation) < int(self.population_size):
                            next_generation.add(parent1)
                        if len(next_generation) < int(self.population_size):
                            next_generation.add(parent2)
        else:
            next_generation: list[DerivationTree[NT, T, G]] = []
            while len(next_generation) < int(self.population_size - self.elitism):
                parent1, parent2 = self.selection(fit)
                cx = nrandom.choice([True, False], p=[self.crossover_rate, 1 - self.crossover_rate], size=1).item()
                assert isinstance(cx, bool)
                if cx:
                    child = parent1.crossover(parent2, self.search_space)
                    while child is None:
                        parent1, parent2 = self.selection(fit)  # if parents are impotent... TODO: is this possible for crossover?
                        child = parent1.crossover(parent2, self.search_space)
                    mt = nrandom.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate], size=1).item()
                    assert isinstance(mt, bool)
                    if mt:
                        child = child.mutate(self.search_space)
                    next_generation.append(child)
                else:
                    mt = nrandom.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate], size=1).item()
                    assert isinstance(mt, bool)
                    if mt:
                        mutant1 = parent1.mutate(self.search_space)
                        mutant2 = parent2.mutate(self.search_space)
                        next_generation.append(mutant1)
                        if len(next_generation) < int(self.population_size - self.elitism):
                            next_generation.append(mutant2)
                    else:
                        if len(next_generation) < int(self.population_size - self.elitism):
                                next_generation.append(parent1)
                        if len(next_generation) < int(self.population_size - self.elitism):
                                next_generation.append(parent2)
            # elitism: carry over the best individuals
            elite = sorted(fit.items(), key=lambda x: x[1], reverse=self.greater_is_better)[:self.elitism]
            for e in elite:
                next_generation.append(e[0])
        new_fitnesses = {t: self.fitness_function(t) for t in next_generation}
        #result = fit | new_fitnesses
        return new_fitnesses

    def check_termination_criteria(self, old_fitnesses, new_fitnesses):
        old_best = max(old_fitnesses.values())
        new_best = max(new_fitnesses.values())
        if self.greater_is_better:
            if new_best <= old_best:
                self.stale_generations += 1
        else:
            if new_best >= old_best:
                self.stale_generations += 1
        if self.stale_generations >= 5:  # int(self.generation_limit/10 + 1):
            print(f"Termination criteria met: no improvement in fitness for {5} generations.")
            return True
        return False

