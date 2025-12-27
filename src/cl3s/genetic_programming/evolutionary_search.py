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
                 population_size: int=100, crossover_rate: float=0.8, mutation_rate: float= 0.4,
                 generation_limit: int=30, elitism: int=1,
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
        n = 1
        test = max(current_generation, key=self.fitness_function)
        print(self.fitness_function(test))
        while n <= self.generation_limit:
            print(f"Generation {n}")
            #print(len(current_generation))
            next_generation = self.reproduction(current_generation)
            #assert (max(current_generation, key=self.fitness_function) in next_generation) if self.elitism > 0 else True
            if self.check_termination_criteria(current_generation, next_generation):
                break
            current_generation = next_generation
            #print(test)
            #assert test in current_generation
            n += 1
        fitnesses = {t: self.fitness_function(t) for t in current_generation}
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
        #return list(self.search_space.enumerate_trees(self.request, self.population_size))
        return list(self.search_space.sample(self.population_size, self.request))

    def selection(self, generation: list[DerivationTree[NT, T, G]]):
        if self.enforce_diversity:
            selected: set[DerivationTree[NT, T, G]] = set()
            while len(selected) < 2:
                tournament = random.sample(generation, self.tournament_size)
                if self.greater_is_better:
                    winner = max(tournament, key=self.fitness_function)
                else:
                    winner = min(tournament, key=self.fitness_function)
                selected.add(winner)
            parent1 = selected.pop()
            parent2 = selected.pop()
            return parent1, parent2
        else:
            selected: list[DerivationTree[NT, T, G]] = []
            for _ in range(2):
                tournament = random.sample(generation, self.tournament_size)
                if self.greater_is_better:
                    winner = max(tournament, key=self.fitness_function)
                else:
                    winner = min(tournament, key=self.fitness_function)
                selected.append(winner)
            parent1 = selected.pop()
            parent2 = selected.pop()
            return parent1, parent2

    def reproduction(self, current_generation: list[DerivationTree[NT, T, G]]):
        fit = {t: self.fitness_function(t) for t in current_generation}
        elite = sorted(fit.items(), key=lambda x: x[1], reverse=self.greater_is_better)[:self.elitism]
        if self.enforce_diversity:
            next_generation: set[DerivationTree[NT, T, G]] = set()
            # elitism: carry over the best individuals
            for e in elite:
                next_generation.add(e[0])  # invariant diversity: there are no duplicates in population
            while len(next_generation) < int(self.population_size):
                parent1, parent2 = self.selection(current_generation)
                cx = nrandom.choice([True, False], p=[self.crossover_rate, 1 - self.crossover_rate], size=1).item()
                assert isinstance(cx, bool)
                if cx:
                    child = parent1.crossover(parent2, self.search_space)
                    while child is None:
                        #parent1, parent2 = self.selection(current_generation)  # if parents are impotent... TODO: is this possible for crossover?
                        child = parent1.crossover(parent2, self.search_space)
                    mt = nrandom.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate], size=1).item()
                    assert isinstance(mt, bool)
                    if mt:
                        child2 = child.mutate(self.search_space)
                        while child2 is None:
                            child2 = child.mutate(self.search_space)
                        child = child2
                    if child not in current_generation:
                        next_generation.add(child)
                else:
                    mt = nrandom.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate], size=1).item()
                    assert isinstance(mt, bool)
                    if mt:
                        mutant1 = parent1.mutate(self.search_space)
                        while mutant1 is None:
                            mutant1 = parent1.mutate(self.search_space)
                        mutant2 = parent2.mutate(self.search_space)
                        while mutant2 is None:
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
            # elitism: carry over the best individuals
            next_generation: list[DerivationTree[NT, T, G]] = list(map(lambda x: x[0], elite))
            #assert all([e in next_generation for e, _ in elite])
            while len(next_generation) < int(self.population_size):
                parent1, parent2 = self.selection(current_generation)
                cx = nrandom.choice([True, False], p=[self.crossover_rate, 1 - self.crossover_rate], size=1).item()
                assert isinstance(cx, bool)
                if cx:
                    child = parent1.crossover(parent2, self.search_space)
                    while child is None:
                        parent1, parent2 = self.selection(current_generation)  # if parents are impotent... TODO: is this possible for crossover?
                        child = parent1.crossover(parent2, self.search_space)
                    mt = nrandom.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate], size=1).item()
                    assert isinstance(mt, bool)
                    if mt:
                        child2 = child.mutate(self.search_space)
                        while child2 is None:
                            child2 = child.mutate(self.search_space)
                        child = child2
                    next_generation.append(child)
                else:
                    mt = nrandom.choice([True, False], p=[self.mutation_rate, 1 - self.mutation_rate], size=1).item()
                    assert isinstance(mt, bool)
                    if mt:
                        mutant1 = parent1.mutate(self.search_space)
                        while mutant1 is None:
                            mutant1 = parent1.mutate(self.search_space)
                        mutant2 = parent2.mutate(self.search_space)
                        while mutant2 is None:
                            mutant2 = parent2.mutate(self.search_space)
                        next_generation.append(mutant1)
                        if len(next_generation) < int(self.population_size):
                            next_generation.append(mutant2)
                    else:
                        if len(next_generation) < int(self.population_size):
                                next_generation.append(parent1)
                        if len(next_generation) < int(self.population_size):
                                next_generation.append(parent2)
        assert len(next_generation) == self.population_size
        #new_fitnesses = {t: self.fitness_function(t) for t in next_generation}
        #result = fit | new_fitnesses
        assert all([e in next_generation for e, _ in elite])
        return list(next_generation)

    def check_termination_criteria(self, current_generation, next_generation):
        old_best = max(current_generation, key=self.fitness_function)
        old_fitness = self.fitness_function(old_best)
        new_best = max(next_generation, key=self.fitness_function)
        new_fitness = self.fitness_function(new_best)
        if self.greater_is_better:
            if new_fitness <= old_fitness:
                self.stale_generations += 1
            else:
                self.stale_generations = 0
        else:
            if new_fitness >= old_fitness:
                self.stale_generations += 1
            else:
                self.stale_generations = 0
        if self.stale_generations > int(self.generation_limit/10 + 1):
            print(f"Termination criteria met: no improvement in fitness for {self.stale_generations} generations.")
            return True
        return False

