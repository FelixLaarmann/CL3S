# cl3s/tree.py

from __future__ import annotations

from functools import partial
from inspect import Parameter, signature
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, Optional, TypeVar, Union, Generator
import typing
from dataclasses import dataclass, field
import random

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals

from clsp.tree import Tree
from clsp.solution_space import RHSRule

from .core import SearchSpace, DerivationTree, NT, T

@dataclass(slots=True)
class DerivationTree(Tree[T], Generic[NT, T]):
    """
    A tree from clsp annotated with its derivation information.
    """

    children: tuple["DerivationTree[NT, T]", ...] = field(default=())

    derived_from: NT | None = field(default=None)
    rhs_rule: RHSRule[NT, T] | None = field(default=None)
    is_literal: bool = field(default=False)
    literal_group: str = field(default="")
    frozen: bool = field(default=False)


    def to_labeled_adjacency_dict(self, next_index: int = 0) -> tuple[dict[int, list[int]], dict[int, T], int]:
        """
        Convert the tree to a tuple of a dictionary, mapping node indices to their argument indices, a dictionary
        mapping node indices to combinators, and the highest index from the mapping.
        :param next_index: The next index to use for the root node.
        :return: A tuple containing the edges dictionary, labels dictionary, and the next index.
        """
        edges: dict[int, list[int]] = {next_index: []}
        labels: dict[int, T] = {next_index: self.root}
        i = next_index + 1
        for child in self.children:
            edges[next_index].append(i)
            labels[i] = child.root
            child_edges, child_labels, n = child.to_labeled_adjacency_dict(i)
            edges.update(child_edges)
            labels.update(child_labels)
            i = n
        return edges, labels, i

    def subtrees(self, path: list[int]) -> Generator[tuple["DerivationTree[NT, T]", list[int]], ...]:
        """
        Compute all subtrees of the tree and their paths, including the tree itself.
        :param path: The path to the current tree.
        :return: A list of tuples, where each tuple contains a subtree and its path.
        """""
        for i, child in enumerate(self.children):
            # recursively compute the subtrees of the children
            for subtree, child_path in child.subtrees(path + [i]):
                yield subtree, child_path

    def replace(self, path: list[int], subtree: "DerivationTree[NT, T]") -> "DerivationTree[NT, T]":
        """
        Replace a subtree at the given path with another subtree.
        :param path: The path to the subtree to replace.
        :param subtree: The subtree to replace with.
        :return: A new derivation tree with the subtree replaced.
        """
        if not path:
            # if the path is empty, return the subtree
            return subtree
        # create a copy of the current tree
        new_tree = DerivationTree(
            root=self.root,
            children=self.children,
            derived_from=self.derived_from,
            rhs_rule=self.rhs_rule,
            is_literal=self.is_literal,
            literal_group=self.literal_group,
            frozen=self.frozen
        )
        # traverse the path to the subtree to replace
        current = new_tree
        for i in path[:-1]:
            if i < 0 or i >= len(current.children):
                raise ValueError(f"Invalid path.")
            current = current.children[i]
        # replace the subtree at the given path
        current.children = tuple(current.children[:path[-1]] + (subtree,) + current.children[path[-1] + 1:])
        return new_tree

    def is_valid_crossover(self, primary_path: list[int], secondary_tree: "DerivationTree[NT, T]",
                           search_space: SearchSpace[NT, T], max_depth: int | None = None) -> bool:
        # TODO
        raise NotImplementedError("This method still needs to be implemented.")

    def is_consistent_with(self, search_space: SearchSpace[NT, T]) -> bool:
        # TODO
        raise NotImplementedError("This method still needs to be implemented.")

    def crossover(self, secondary_derivation_tree: "DerivationTree[NT, T]",
                  search_space: SearchSpace[NT, T], max_depth: int | None = None,
                  seed: int | None = None) -> Union["DerivationTree[NT, T]", None]:
        """
        Perform a crossover operation with another derivation tree.
        Crossover is closed under the search space, meaning the resulting tree is guaranteed to be a member of the search space.
        :param secondary_derivation_tree: The secondary derivation tree to crossover with.
        :param search_space: The search space crossing is performed in.
        :param max_depth: The maximum depth of the resulting tree.
        :param seed: Optional seed for random number generation to ensure reproducibility.
        :return: A new derivation tree that is a member of the search space, or None if no valid crossover could be performed.
        """
        # compute all subtrees and their paths, excluding the whole primary derivation tree (self)
        primary_subtrees = list(self.subtrees([]))
        primary_subtrees.remove((self, []))
        # compute all subtrees of the secondary derivation tree, including the secondary derivation tree itself
        secondary_subtrees = list(secondary_derivation_tree.subtrees([]))
        # choose a random primary subtree as crossover point, until crossover is successful
        while primary_subtrees:
            if seed is not None:
                random.seed(seed)
            selected_primary, primary_path = random.choice(primary_subtrees)
            # remove the selected primary subtree from the list of primary subtrees
            primary_subtrees.remove((selected_primary, primary_path))
            # if the selected primary subtree is frozen, continue
            if selected_primary.frozen:
                continue
            temp_secondary_subtrees = secondary_subtrees.copy()
            # if the selected primary subtree is a literal, choose the literal group as crossover point
            # choose a random secondary subtree that matches the crossover point
            if selected_primary.is_literal:
                crossover_point = selected_primary.literal_group
                if not crossover_point:
                    raise ValueError("Selected primary subtree has no literal group. The empty string is not a valid literal group.")
                temp_secondary_subtrees = list(filter(lambda x: x[0].literal_group == crossover_point, temp_secondary_subtrees))
            else:
                # else choose its type (non-terminal) as crossover point
                crossover_point = selected_primary.derived_from
                if crossover_point is None:
                    raise ValueError("Selected primary subtree has no derived_from.")
                temp_secondary_subtrees = list(filter(lambda x: x[0].derived_from == crossover_point, temp_secondary_subtrees))
            # choose a random secondary subtree that matches the crossover point, until crossover is successful
            while temp_secondary_subtrees:
                if seed is not None:
                    random.seed(seed)
                selected_secondary, secondary_path = random.choice(temp_secondary_subtrees)
                # remove the selected secondary subtree from the list of secondary subtrees
                temp_secondary_subtrees.remove((selected_secondary, secondary_path))
                # if the selected secondary subtree is frozen, continue
                if selected_secondary.frozen:
                    continue
                # test if the selected secondary subtree can be inserted into the crossover point of the primary subtree
                valid = self.is_valid_crossover(primary_path, selected_secondary, search_space, max_depth)
                if valid:
                    # perform the crossover
                    offspring = self.replace(primary_path, selected_secondary)
                    # check if the new tree is a member of the search space
                    if offspring.is_consistent_with(search_space):
                        return offspring
        # if no offspring was created for any primary subtree, return None
        return None

    def mutate(self, search_space: SearchSpace[NT, T],
               max_depth: int | None = None, seed: int | None = None) -> Union["DerivationTree[NT, T]", None]:
        """
        Mutates the tree by replacing a random subtree with a new one sampled from the search space.
        Mutation is closed under the search space, meaning the resulting tree is guaranteed to be a member of the search space.
        :param search_space: The search space mutation is performed in.
        :param max_depth: The maximum depth of the resulting tree.
        :param seed: Optional seed for random number generation to ensure reproducibility.
        :return: A new derivation tree that is a member of the search space, or None if no valid mutation could be performed.
        """
        # compute all subtrees and their paths
        subtrees = list(self.subtrees([]))
        # choose a random primary subtree as mutation point, until mutation is successful
        while subtrees:
            if seed is not None:
                random.seed(seed)
            selected, mutate_at = random.choice(subtrees)
            # remove the selected primary subtree from the list of primary subtrees
            subtrees.remove((selected, mutate_at))
            # frozen trees, literals, and trees with no children will not be mutated
            if selected.frozen or selected.is_literal or not selected.children:
                continue
            mutation_point = selected.derived_from
            if mutation_point is None:
                raise ValueError("Selected primary subtree has no derived_from.")
            # sample a new tree from the search space
            mutation = search_space.sample_tree(mutation_point, max_depth)
            # replace the selected subtree with the sampled tree
            offspring = self.replace(mutate_at, mutation)
            # check if the offspring is a member of the search space
            if offspring.is_consistent_with(search_space):
                return offspring
        # if no offspring was created for any primary subtree, return None
        return None
