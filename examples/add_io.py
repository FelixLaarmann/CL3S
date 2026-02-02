from src.cl3s import (SpecificationBuilder, Constructor, Literal, Var,
                      SearchSpaceSynthesizer, DerivationTree, DataGroup, Group)
from src.cl3s.evolutionary_search import TournamentSelection

import random
from typing import Any
import timeit
import sys

sys.setrecursionlimit(15000)

class AddIORepository:

    class Nat(Group):
        def __iter__(self):
            pass
        def __contains__(self, item):
            return isinstance(item, int) and item >= 0

    def specification_refined(self):
        return {
            "Z": Constructor("Nat") & Constructor("even") & Constructor("A"),
            "S" : (Constructor("Nat") ** Constructor("Nat"))
                  & (Constructor("even") ** Constructor("odd"))
                  & (Constructor("odd") ** Constructor("even"))
                  & (Constructor("A") ** Constructor("A")),
            "foldNat" : ((Constructor("A") ** Constructor("A")) ** Constructor("A") ** Constructor("Nat") ** Constructor("A"))
                        & ((Constructor("Nat") ** Constructor("Nat")) ** Constructor("Nat") ** Constructor("Nat") ** Constructor("Nat")),
        }

    def specification(self):
        return {
            "Z": Constructor("Nat") & Constructor("A"),
            "S" : (Constructor("Nat") ** Constructor("Nat"))
                  & (Constructor("A") ** Constructor("A")),
            "foldNat" : ((Constructor("A") ** Constructor("A")) ** Constructor("A") ** Constructor("Nat") ** Constructor("A"))
                        & ((Constructor("Nat") ** Constructor("Nat")) ** Constructor("Nat") ** Constructor("Nat") ** Constructor("Nat")),
            "S-combinator" : (Constructor("A") ** Constructor("A") ** Constructor("A")) ** (Constructor("A") ** Constructor("A")) ** Constructor("A") ** Constructor("A")
                             & (Constructor("Nat") ** Constructor("Nat") ** Constructor("Nat")) ** (Constructor("Nat") ** Constructor("Nat")) ** Constructor("Nat") ** Constructor("Nat")
        }

    def foldNat_recursive(self, z, s, n):
        # n must be an int >= 0!
        if n < 0:
            raise ValueError("Natural numbers must be non-negative")
        if n == 0:
            return z
        else:
            return s(self.foldNat_recursive(z, s, n - 1))

    def foldNat_iterative(self, z, s, n):
        # n must be an int >= 0!
        if n < 0:
            raise ValueError("Natural numbers must be non-negative")
        result = z
        for _ in range(n):
            result = s(result)
        return result

    def pretty_term_algebra(self):
        return {
            "Z": "Z",
            "S": lambda n: f"S({n})",
            "foldNat": lambda s, z, n: f"foldNat(lambda x: {s('x')}, {z}, {n})",
            "S-combinator": lambda x, y, z: f"({x} {z} ({y} {z}))",
        }

    def nat_algebra(self):
        return {
            "Z": 0,
            "S": lambda n: n + 1,
            "foldNat": lambda s, z, n: self.foldNat_iterative(z, s, n),
            "S-combinator": lambda x, y, z: x(z, y(z)),
        }


if __name__ == "__main__":
    repo = AddIORepository()
    synthesizer = SearchSpaceSynthesizer(repo.specification())

    target = Constructor("Nat") ** Constructor("Nat") ** Constructor("Nat")
    # this won't work, because foldNat can't be annotated with even/odd
    target_refined = (Constructor("Nat") ** Constructor("Nat") ** Constructor("Nat")
                      & Constructor("even") ** Constructor("even") ** Constructor("even")
                      & Constructor("odd") ** Constructor("odd") ** Constructor("even")
                      & Constructor("even") ** Constructor("odd") ** Constructor("odd")
                      & Constructor("odd") ** Constructor("even") ** Constructor("odd"))

    nat_sample = random.sample(range(0, 100), 10)
    if 0 not in nat_sample:
        nat_sample.append(0)
    if 1 not in nat_sample:
        nat_sample.append(1)
    data = [((x, y), x*y) for x in nat_sample for y in nat_sample]
    y_true = [z for _, z in data]

    def mse(y_true, y_pred):
        return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

    def f_obj(tree: DerivationTree[Any, str, Any], verbose = False) -> float:
        f = tree.interpret(repo.nat_algebra())
        y_pred = [f(x, y) for (x, y), _ in data]
        #y_true = [z for _, z in data]
        if verbose:
            print("Predictions: ", y_pred)
            print("True values: ", y_true)
        return mse(y_true, y_pred)

    print("start synthesis")
    time_start = timeit.default_timer()
    search_space = synthesizer.construct_search_space(target).prune()
    time_end = timeit.default_timer()
    print(f"Search space constructed in {time_end - time_start:.2f} seconds")
    """
    terms = search_space.enumerate_trees(target, 50)

    time_start = timeit.default_timer()
    terms = list(terms)
    time_end = timeit.default_timer()
    print(f"Enumerated {len(terms)} terms in {time_end - time_start:.2f} seconds")

    for t in terms:
        print("add(a,b) = ", t.interpret(repo.pretty_term_algebra())("a", "b"), f"-> MSE: {f_obj(t)}")
    """
    #"""
    evolutionary_search = TournamentSelection(search_space, target, f_obj, population_size=50, tournament_size=3,
                                              crossover_rate=0.9, mutation_rate=0.05, generation_limit=100, elitism=2,
                                              enforce_diversity=False, greater_is_better=False)

    best_tree = evolutionary_search.optimize()

    print("Best program found: ")
    print(best_tree.interpret(repo.pretty_term_algebra()))

    result = f_obj(best_tree, verbose=True)
    print(f"MSE of best program: {result}")
    #"""
