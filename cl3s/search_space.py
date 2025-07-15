# cl3s/search_space.py

"""Solution space given by a logic program."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Sequence, Mapping, Iterable
from itertools import product
from queue import PriorityQueue

from typing import Any, Generic, TypeVar, Union, Generator, override
import typing
from types import FunctionType

from cosy.solution_space import SolutionSpace, RHSRule, NonTerminalArgument, ConstantArgument

if typing.TYPE_CHECKING:
    from .tree import DerivationTree

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names

class SearchSpace(SolutionSpace[NT, T, G], Generic[NT, T, G]):
    """
    This class extends the SolutionSpace to provide additional methods
    for searching and manipulating the solution space.
    It relies on DerivationTrees instead of Trees.
    """

    @override
    def __init__(self, rules: dict[NT, deque[RHSRule[NT, T, G]]] | None = None) -> None:
        """
        Initialize the SearchSpace with a set of rules.
        The rules should be provided as a dictionary mapping non-terminals to their possible right-hand sides.
        """
        super().__init__(rules)


    # new methods for SearchSpace

    def sample_tree(self, non_terminal: NT, max_depth: int | None = None) -> "DerivationTree[NT, T, G]":
        """
        Sample a random tree from the search space.
        """
        # TODO
        raise NotImplementedError("This method still needs to be implemented.")

    def sample(self, size: int, non_terminal: NT, max_depth: int | None = None) -> list["DerivationTree[NT, T, G]"]:
        """
        Sample a list of random trees from the search space.
        """
        # TODO
        raise NotImplementedError("This method still needs to be implemented.")

    # old methods from SolutionSpace, that are adapted for DerivationTree

    @override
    def _enumerate_tree_vectors(
        self,
        non_terminals: Sequence[NT | None],
        existing_terms: Mapping[NT, set["DerivationTree[NT, T, G]"]],
        nt_term: tuple[NT, "DerivationTree[NT, T, G]"] | None = None,
    ) -> Iterable[tuple[Union["DerivationTree[NT, T, G]", None], ...]]:
        """Enumerate possible term vectors for a given list of non-terminals and existing terms. Use nt_term at least once (if given)."""
        if nt_term is None:
            yield from product(*([n] if n is None else existing_terms[n] for n in non_terminals))
        else:
            nt, term = nt_term
            for i, n in enumerate(non_terminals):
                if n == nt:
                    arg_lists: Iterable[Iterable[Union["DerivationTree[NT, T, G]", None]]] = (
                        [None] if m is None else [term] if i == j else existing_terms[m]
                        for j, m in enumerate(non_terminals)
                    )
                    yield from product(*arg_lists)

    @override
    def _generate_new_trees(
        self,
        lhs: NT,
        rule: RHSRule[NT, T, G],
        existing_terms: Mapping[NT, set["DerivationTree[NT, T, G]"]],
        max_count: int | None = None,
        nt_old_term: tuple[NT, "DerivationTree[NT, T, G]"] | None = None,
    ) -> set["DerivationTree[NT, T, G]"]:
        # Genererate new terms for rule `rule` from existing terms up to `max_count`
        # the term `old_term` should be a subterm of all resulting terms, at a position, that corresponds to `nt`

        output_set: set["DerivationTree[NT, T, G]"] = set()
        if max_count == 0:
            return output_set

        named_non_terminals = [
            a.origin if isinstance(a, NonTerminalArgument) and a.name is not None else None for a in rule.arguments
        ]
        unnamed_non_terminals = [
            a.origin if isinstance(a, NonTerminalArgument) and a.name is None else None for a in rule.arguments
        ]
        literal_arguments = [DerivationTree(a.value, tuple(), None, rule, True, a.origin, False) if isinstance(a, ConstantArgument) else None for a in rule.arguments]

        def interleave(
            parameters: Sequence[Union["DerivationTree[NT, T, G]", None]],
            literal_arguments: Sequence[Union["DerivationTree[NT, T, G]", None]],
            arguments: Sequence[Union["DerivationTree[NT, T, G]", None]],
        ) -> Iterable["DerivationTree[NT, T, G]"]:
            """Interleave parameters, literal arguments and arguments."""
            for parameter, literal_argument, argument in zip(parameters, literal_arguments, arguments, strict=True):
                if parameter is not None:
                    yield parameter
                elif literal_argument is not None:
                    yield literal_argument
                elif argument is not None:
                    yield argument
                else:
                    msg = "All arguments of interleave are None"
                    raise ValueError(msg)

        def construct_tree(
            rule: RHSRule[NT, T, G],
            parameters: Sequence[Union["DerivationTree[NT, T, G]", None]],
            literal_arguments: Sequence[Union["DerivationTree[NT, T, G]", None]],
            arguments: Sequence[Union["DerivationTree[NT, T, G]", None]],
        ) -> "DerivationTree[NT, T, G]":
            """Construct a new tree from the rule and the given specific arguments."""
            return DerivationTree(
                rule.terminal,
                tuple(interleave(parameters, literal_arguments, arguments)),
                lhs,
                rule,
                False,
                None,
                False
            )

        def specific_substitution(parameters):
            return {
                a.name: p
                for p, a in zip(parameters, rule.arguments, strict=True)
                if isinstance(a, NonTerminalArgument) and a.name is not None
            } | rule.literal_substitution

        def valid_parameters(
            nt_term: tuple[NT, "DerivationTree[NT, T, G]"] | None,
        ) -> Iterable[tuple[Union["DerivationTree[NT, T, G]", None], ...]]:
            """Enumerate all valid parameters for the rule."""
            for parameters in self._enumerate_tree_vectors(named_non_terminals, existing_terms, nt_term):
                substitution = specific_substitution(parameters)
                if all(predicate(substitution) for predicate in rule.predicates):
                    yield parameters

        for parameters in valid_parameters(nt_old_term):
            for arguments in self._enumerate_tree_vectors(unnamed_non_terminals, existing_terms):
                output_set.add(construct_tree(rule, parameters, literal_arguments, arguments))
                if max_count is not None and len(output_set) >= max_count:
                    return output_set

        if nt_old_term is not None:
            all_parameters: deque[tuple[Union["DerivationTree[NT, T, G]", None], ...]] | None = None
            for arguments in self._enumerate_tree_vectors(unnamed_non_terminals, existing_terms):
                all_parameters = all_parameters if all_parameters is not None else deque(valid_parameters(None))
                for parameters in all_parameters:
                    output_set.add(construct_tree(rule, parameters, literal_arguments, arguments))
                    if max_count is not None and len(output_set) >= max_count:
                        return output_set
        return output_set

    def enumerate_trees(
        self,
        start: NT,
        max_count: int | None = None,
        max_bucket_size: int | None = None,
    ) -> Iterable["DerivationTree[NT, T, G]"]:
        """
        Enumerate terms as an iterator efficiently - all terms are enumerated, no guaranteed term order.
        """
        if start not in self.nonterminals():
            return

        queues: dict[NT, PriorityQueue["DerivationTree[NT, T, G]"]] = {n: PriorityQueue() for n in self.nonterminals()}
        existing_terms: dict[NT, set["DerivationTree[NT, T, G]"]] = {n: set() for n in self.nonterminals()}
        inverse_grammar: dict[NT, deque[tuple[NT, RHSRule[NT, T, G]]]] = {n: deque() for n in self.nonterminals()}
        all_results: set["DerivationTree[NT, T, G]"] = set()

        for n, exprs in self._rules.items():
            for expr in exprs:
                if all(m in self.nonterminals() for m in expr.non_terminals):
                    for m in expr.non_terminals:
                        inverse_grammar[m].append((n, expr))
                #if expr.non_terminals.issubset(self.nonterminals()):
                #    for m in expr.non_terminals:
                #        if m in self.nonterminals():
                #            inverse_grammar[m].append((n, expr))
                    for new_term in self._generate_new_trees(n, expr, existing_terms):
                        queues[n].put(new_term)
                        if n == start and new_term not in all_results:
                            if max_count is not None and len(all_results) >= max_count:
                                return
                            yield new_term
                            all_results.add(new_term)

        current_bucket_size = 1

        while (max_bucket_size is None or current_bucket_size <= max_bucket_size) and any(
            not queue.empty() for queue in queues.values()
        ):
            non_terminals = {n for n in self.nonterminals() if not queues[n].empty()}

            while non_terminals:
                n = non_terminals.pop()
                results = existing_terms[n]
                while len(results) < current_bucket_size and not queues[n].empty():
                    term = queues[n].get()
                    if term in results:
                        continue
                    results.add(term)
                    for m, expr in inverse_grammar[n]:
                        if len(existing_terms[m]) < current_bucket_size:
                            non_terminals.add(m)
                        if m == start:
                            for new_term in self._generate_new_trees(m, expr, existing_terms, max_count, (n, term)):
                                if new_term not in all_results:
                                    if max_count is not None and len(all_results) >= max_count:
                                        return
                                    yield new_term
                                    all_results.add(new_term)
                                    queues[start].put(new_term)
                        else:
                            for new_term in self._generate_new_trees(m, expr, existing_terms, max_bucket_size, (n, term)):
                                queues[m].put(new_term)
            current_bucket_size += 1
        return

    # TODO: refactor this method to use the new DerivationTree class and work with consistency checks
    def contains_tree(self, start: NT, tree: "DerivationTree[NT, T, G]") -> bool:
        """Check if the solution space contains a given `tree` derivable from `start`."""
        if start not in self.nonterminals():
            return False

        stack: deque[tuple | Callable] = deque([(start, tree)])
        results: deque[bool] = deque()

        def get_inputs(count: int) -> Generator[bool]:
            for _ in range(count):
                yield results.pop()
            return

        while stack:
            task = stack.pop()
            if isinstance(task, tuple):
                nt, tree = task
                relevant_rhss = [
                    rhs
                    for rhs in self._rules[nt]
                    if len(rhs.arguments) == len(tree.children)
                    and rhs.terminal == tree.root
                    and all(
                        argument.value == child.root and len(child.children) == 0
                        for argument, child in zip(rhs.arguments, tree.children, strict=True)
                        if isinstance(argument, ConstantArgument)
                    )
                ]

                # if there is a relevant rule containing only TerminalArgument which are equal to the children of the tree
                if any(
                    all(isinstance(argument, ConstantArgument) for argument in rhs.arguments) for rhs in relevant_rhss
                ):
                    results.append(True)
                    continue

                # disjunction of the results for individual rules
                def or_inputs(count: int = len(relevant_rhss)) -> None:
                    results.append(any(get_inputs(count)))

                stack.append(or_inputs)

                for rhs in relevant_rhss:
                    substitution = {
                        argument.name: child.root if isinstance(argument, ConstantArgument) else child
                        for argument, child in zip(rhs.arguments, tree.children, strict=True)
                        if argument.name is not None
                    }

                    # conjunction of the results for individual arguments in the rule
                    def and_inputs(
                        count: int = sum(1 for argument in rhs.arguments if isinstance(argument, NonTerminalArgument)),
                        substitution: dict[str, Any] = substitution,
                        predicates=rhs.predicates,
                    ) -> None:
                        results.append(
                            all(get_inputs(count)) and all(predicate(substitution) for predicate in predicates)
                        )

                    stack.append(and_inputs)
                    for argument, child in zip(rhs.arguments, tree.children, strict=True):
                        if isinstance(argument, NonTerminalArgument):
                            stack.append((argument.origin, child))
            elif isinstance(task, FunctionType):
                # task is a function to execute
                task()
        return results.pop()
