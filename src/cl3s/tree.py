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
import networkx as nx
import grakel
from grakel.utils import graph_from_networkx

from copy import copy, deepcopy

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names

from cosy.tree import Tree
from cosy.solution_space import RHSRule, NonTerminalArgument

if typing.TYPE_CHECKING:
    from .search_space import SearchSpace


class DerivationTree(Tree[T], Generic[NT, T, G]):
    """
    A tree from clsp annotated with its derivation information.
    """

    children: tuple["DerivationTree[NT, T, G]", ...]

    derived_from: NT | None  # the non-terminal this tree is derived from, None if it is a literal
    rhs_rule: RHSRule[NT, T, G]
    is_literal: bool
    literal_group: str | None  # if the tree is a literal, this is the group it belongs to
    frozen: bool

    def __init__(self, root: T, children: tuple["DerivationTree[NT, T, G]", ...],
                 derived_from: NT | None, rhs_rule: RHSRule[NT, T, G],
                 is_literal: bool, literal_group: str | None,
                 frozen: bool = False):
        """
        Initialize a derivation tree.
        :param root: The root of the tree.
        :param children: The children of the tree.
        :param derived_from: The non-terminal this tree is derived from.
        :param rhs_rule: The rule used to derive this tree.
        :param is_literal: Whether this tree is a literal.
        :param literal_group: The group this literal belongs to, if applicable.
        :param frozen: Whether this tree is frozen (i.e., cannot be mutated).
        """
        super().__init__(root, children)
        self.derived_from = derived_from
        self.rhs_rule = rhs_rule
        self.is_literal = is_literal
        self.literal_group = literal_group
        self.frozen = frozen

    def __copy__(self) -> "DerivationTree[NT, T, G]":
        children_copy = tuple(copy(child) for child in self.children)
        return DerivationTree(
            root=self.root,
            children=children_copy,
            derived_from=self.derived_from,
            rhs_rule=self.rhs_rule,
            is_literal=self.is_literal,
            literal_group=self.literal_group,
            frozen=self.frozen
        )

    def to_indexed_nx_digraph(self, start_index: int = 0) -> tuple[nx.DiGraph, dict[int, T]]:
        if self.is_literal:
            G = nx.DiGraph()
            G.add_node(start_index, symbol=str(self.root), literal=True, type=str(self.literal_group))
            return G, {start_index: self.root}
        elif not self.children:
            G = nx.DiGraph()
            G.add_node(start_index, symbol=str(self.root), literal=False, type=str(self.derived_from))
            return G, {start_index: self.root}
        else:
            G = nx.DiGraph()
            G.add_node(start_index, symbol=str(self.root), literal=False, type=str(self.derived_from))
            index_mapping = {start_index: self.root}
            current_index = start_index + 1
            for child in self.children:
                Gc, child_mapping = child.to_indexed_nx_digraph(current_index)
                G = nx.compose(G, Gc)
                G.add_edge(start_index, current_index, argument_type=str(child.derived_from) if child.derived_from is not None else str(child.literal_group))
                index_mapping.update(child_mapping)
                current_index += len(child_mapping)
            return G, index_mapping

    def to_grakel_graph(self) -> grakel.Graph:
        nx_graph, node_labels = self.to_indexed_nx_digraph()
        gk_graph = graph_from_networkx([nx_graph.to_undirected()], node_labels_tag='symbol', edge_labels_tag='argument_type')
        # I hate grakel! Why do they write tuple in their docs when it is actually a str?!
        return gk_graph

    def subtrees(self, path: list[int]) -> list[tuple["DerivationTree[NT, T, G]", list[int]]]:
        """
        Compute all subtrees of the tree and their paths including the tree itself.
        :param path: The path to the current tree.
        :return: A list of tuples, where each tuple contains a subtree and its path.
        """""
        result = [(self, path)]
        for i, child in enumerate(self.children):
            # recursively compute the subtrees of the children
            for subtree, child_path in child.subtrees(path + [i]):
                result.append((subtree, child_path))
        return result

    def non_literal_subtrees(self, path: list[int]) -> list[tuple["DerivationTree[NT, T, G]", list[int]]]:
        """
        Compute all non-literal subtrees of the tree and their paths including the tree itself.
        :param path: The path to the current tree.
        :return: A list of tuples, where each tuple contains a non-literal subtree and its path.
        """""
        result = []
        if not self.is_literal:
            result.append((self, path))
        for i, child in enumerate(self.children):
            # recursively compute the non-literal subtrees of the children
            for subtree, child_path in child.non_literal_subtrees(path + [i]):
                result.append((subtree, child_path))
        return result

    def replace(self, path: list[int], subtree: "DerivationTree[NT, T, G]") -> "DerivationTree[NT, T, G]":
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
        """
        new_tree = DerivationTree(
            root=self.root,
            children=self.children,
            derived_from=self.derived_from,
            rhs_rule=self.rhs_rule,
            is_literal=self.is_literal,
            literal_group=self.literal_group,
            frozen=self.frozen
        )
        """
        new_tree = copy(self)
        # traverse the path to the subtree to replace
        current = new_tree
        for i in path[:-1]:
            if i < 0 or i >= len(current.children):
                raise ValueError(f"Invalid path.")
            current = current.children[i]
        # replace the subtree at the given path
        insert = copy(subtree)
        current.children = tuple(current.children[:path[-1]] + (insert,) + current.children[path[-1] + 1:])
        return current

    def is_valid_crossover(self, primary_tree: "DerivationTree[NT, T, G]", secondary_tree: "DerivationTree[NT, T, G]",
                           search_space: "SearchSpace[NT, T, G]", max_depth: int | None = None) -> bool:
        rule = primary_tree.rhs_rule
        if rule == secondary_tree.rhs_rule:
            # if the rules are the same, we can perform the crossover
            return True
        if len(rule.arguments) == len(secondary_tree.children):
            substitution: dict[str, DerivationTree[NT, T, G]] = {}
            for arg, child in zip(rule.arguments, secondary_tree.children):
                if isinstance(arg, NonTerminalArgument):
                    # if the argument is a non-terminal, check if the derived_from matches
                    if child.derived_from != arg.origin:
                        return False
                    substitution[arg.name] = child
            return all(predicate(substitution | secondary_tree.rhs_rule.literal_substitution)
                       for predicate in rule.predicates)
        else:
            return False

    def is_consistent_with(self, search_space: SearchSpace[NT, T, G]) -> bool:
        return search_space.contains_tree(self.derived_from, self)

    def crossover(self, secondary_derivation_tree: "DerivationTree[NT, T, G]",
                  search_space: "SearchSpace[NT, T, G]", max_depth: int | None = None,
                  seed: int | None = None) -> Union["DerivationTree[NT, T, G]", None]:
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
        primary_subtrees = list(self.non_literal_subtrees([]))
        # TODO: this case should not occur
        if (self, []) in primary_subtrees:
            primary_subtrees.remove((self, []))
        # compute all subtrees of the secondary derivation tree, including the secondary derivation tree itself
        secondary_subtrees = list(secondary_derivation_tree.non_literal_subtrees([]))
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
                #temp_secondary_subtrees = list(filter(lambda x: x[0].literal_group == crossover_point, temp_secondary_subtrees))
                temp_secondary_subtrees = []
            else:
                # else choose its type (non-terminal) as crossover point
                crossover_point = selected_primary.derived_from
                if crossover_point is None:
                    raise ValueError("Selected primary subtree has no derived_from.")
                temp_secondary_subtrees = list(filter(lambda x: x[0].derived_from == crossover_point and x != selected_primary, temp_secondary_subtrees))
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
                valid = self.is_valid_crossover(selected_primary, selected_secondary, search_space, max_depth)
                if valid:
                    # perform the crossover
                    offspring = self.replace(primary_path, selected_secondary)
                    # check if the new tree is a member of the search space
                    if offspring.is_consistent_with(search_space):
                        return offspring
        # if no offspring was created for any primary subtree, return None
        return None

    def mutate(self, search_space: "SearchSpace[NT, T, G]",
               max_depth: int | None = None, seed: int | None = None) -> Union["DerivationTree[NT, T, G]", None]:
        """
        Mutates the tree by replacing a random subtree with a new one sampled from the search space.
        Mutation is closed under the search space, meaning the resulting tree is guaranteed to be a member of the search space.
        :param search_space: The search space mutation is performed in.
        :param max_depth: The maximum depth of the resulting tree.
        :param seed: Optional seed for random number generation to ensure reproducibility.
        :return: A new derivation tree that is a member of the search space, or None if no valid mutation could be performed.
        """
        # compute all subtrees and their paths
        subtrees = list(self.non_literal_subtrees([]))
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
