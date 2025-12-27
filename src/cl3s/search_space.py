# cl3s/search_space.py

"""Solution space given by a logic program."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Callable, Hashable, Sequence, Mapping, Iterable
from itertools import product
from queue import PriorityQueue

from typing import Any, Generic, TypeVar, Union
from types import FunctionType

from cosy.solution_space import SolutionSpace, RHSRule, NonTerminalArgument, ConstantArgument

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

    _annotated_rules: dict[NT, deque[tuple[RHSRule[NT, T, G], int]]] | None = None
    _annotated_symbol_depths: dict[NT, int] | None = None

    #@override
    def __init__(self, rules: dict[NT, deque[RHSRule[NT, T, G]]] | None = None) -> None:
        """
        Initialize the SearchSpace with a set of rules.
        The rules should be provided as a dictionary mapping non-terminals to their possible right-hand sides.
        """
        super().__init__(rules)


    # new methods for SearchSpace
    def annotations(self) -> tuple[dict[NT, deque[tuple[RHSRule[NT, T, G], int]]], dict[NT, int]]:
        """
        Following the grammar based initialization method (GBIM) for context free grammars,
        we annotate terminals, nonterminals and rules with the expected minimum depth of generated terms.
        In contrast to context free grammars, these depths are overapproximations, because we can not include the
        evaluation of predicates in the computation of the expected depth.
        But even this lower bounds of an expected depth should be a good enough approximation to compute an
        initial population of terms with a suitable distribution of term depths.

        The length of a terminal symbol is always 0, therefore we don't need to return annotations for terminals.
        """

        if self._annotated_rules is not None and self._annotated_symbol_depths is not None:
            # if the rules and symbol depths are already annotated, we can return them directly
            return self._annotated_rules, self._annotated_symbol_depths

        # Because annotated and symbol_depths needs to be hashable, I wasn't able to use a dict for each of them...
        # annotated: tuple[tuple[tuple[NT, RHSRule[NT, T]], int],...] = tuple()
        annotated: dict[NT, deque[tuple[RHSRule[NT, T, G], int]]] = dict()

        not_annotated: list[tuple[NT, RHSRule[NT, T, G]]] = [
            (nt, rhs)
            for nt, rules in self._rules.items()
            for rhs in rules
        ]

        symbol_depths: dict[NT, int] = {}

        nts: list[NT] = []

        check = not_annotated.copy()
        # every rule that only derives nonterminals has length 1
        for nt, rhs in check:
            if not list(rhs.non_terminals):
                # rule only derives terminals
                rs: deque[tuple[RHSRule[NT, T, G], int]] | None = annotated.get(nt)
                # add the rule to the annotated rules
                if rs is None:  # this might be ommited, since there are no rules with the same rhs annotated yet
                    rs = deque()
                    rs.append((rhs, 1))
                # the next block can be ommited, since there are no rules with the same rhs annotated yet
                # else:
                #    for r, i in rs:
                #        if r == rhs:
                #            rs.remove((r, i))
                #            rs.append((r, 1))
                #            break
                annotated[nt] = rs
                not_annotated.remove((nt, rhs))
                nts.append(nt)

        assert len(annotated) > 0
        assert all([list(rhs.non_terminals) for _, rhs in not_annotated])

        for nt in nts:
            # if a right hand side has the minmal length 1, the symbol also has this length
            symbol_depths[nt] = 1

        while not_annotated:
            termination_check = not_annotated.copy()
            for nt, rhs in termination_check:
                assert (list(rhs.non_terminals))
                # check if all nonterminals in rhs are already annotated
                if all(s in symbol_depths.keys() for s in rhs.non_terminals):
                    # the length of a rule is the maximum of its nonterminal lenghts + 1
                    ris: deque[tuple[RHSRule[NT, T, G], int]] | None = annotated.get(nt)
                    new_depth = max(symbol_depths[t] for t in rhs.non_terminals) + 1
                    if ris is None:
                        ris = deque()
                    if rhs not in map(lambda x: x[0], ris):
                        ris.append((rhs, new_depth))
                    else:
                        for r, i in ris:
                            if r == rhs:
                                ris.remove((r, i))
                                ris.append((r, new_depth))
                                break
                    annotated[nt] = ris
                    not_annotated.remove((nt, rhs))
                    # The first time we derive a length for a right hand side, we can assume, that it is the minimum length and therefore set the symbol depth
                    sd = symbol_depths.get(nt)
                    if sd is None:
                        symbol_depths[nt] = new_depth
                    # rs == annotated[nt] and therefore corresponds to the annoted rules for nonterminal nt
                    if all(rule in map(lambda x: x[0], ris) for rule in self._rules[nt]):
                        # all rules of this nonterminal are already annotated
                        # the length of a terminal symbol is the minimum of the length of its rules
                        symbol_depths[nt] = min(map(lambda x: x[1], ris))
            if termination_check == not_annotated:
                # no more rules can be annotated
                break

        # if there are still rules that are not annotated, we have a problem with the grammar
        if not_annotated:
            raise ValueError(
                f"Grammar contains problems. The following rules could not be annotated: {not_annotated} \n These rules have been annotated: {annotated} \n These symbols have been annotated: {symbol_depths}"
            )

        self._annotated_rules = annotated
        self._annotated_symbol_depths = symbol_depths
        return annotated, symbol_depths

    def minimum_tree_depth(self, start: NT) -> int:
        """
        Compute a lower bound for the minimum depth of a tree generated by the grammar from the given nonterminal.
        """
        _, nt_length = self.annotations()
        return nt_length[start]

    def build_tree_annotated(self, nt: NT, cs: int, candidate: RHSRule[NT, T, G]) -> DerivationTree[NT, T, G] | None:
        if not list(candidate.non_terminals):
            if cs + 1 > self.max_tree_depth:
                return None
            # rule only derives terminals, therefore arguments has no nonterminals, but to silence the type checker:
            params: list[G] = [lit for lit in candidate.arguments if isinstance(lit, ConstantArgument)]
            cands: tuple[DerivationTree[NT, T, G], ...] = tuple(
                map(lambda p: DerivationTree(p.value, tuple(),
                                             derived_from=None, rhs_rule=candidate, frozen=False,
                                             is_literal=True, literal_group=p.origin), params))
            if all(predicate({}) for predicate in candidate.predicates):
                return DerivationTree(candidate.terminal, cands, derived_from=nt, rhs_rule=candidate,
                                      is_literal=False, literal_group=None, frozen=False)
            else:
                return None
        else:
            # rule derives non-terminals
            children: tuple[DerivationTree[NT, T, G], ...] = ()
            substitution: dict[str, DerivationTree[NT, T, G]] = {}
            interleave: Callable[[Mapping[str, DerivationTree[NT, T, G]]], tuple[DerivationTree[NT, T, G], ...]] = \
                lambda subs: tuple(
                    subs[t] if isinstance(t, str) else t for t in [
                        DerivationTree(p.value, tuple(), derived_from=None, rhs_rule=candidate,
                                       is_literal=True, literal_group=p.origin, frozen=False)
                        if isinstance(p, ConstantArgument)
                        else p.name
                        for p in candidate.arguments
                    ]
            )
            for _ in range(100):  # self.cost): # TODO: handle cost differently!
                for arg in candidate.arguments:
                    if isinstance(arg, NonTerminalArgument):
                        child_depth = self._annotated_symbol_depths.get(arg.origin)
                        if child_depth is None:
                            child_depth = self.max_tree_depth
                        if cs + child_depth <= self.max_tree_depth:
                            new_cs = cs + child_depth
                            child_tree: DerivationTree[NT, T, G] | None = self.sample_random_term_annotated(arg.origin, new_cs)
                            if child_tree is not None and arg.name is not None:
                                children = children + (child_tree,)
                                substitution[arg.name] = child_tree
                            else:
                                return None
                        else:
                            return None
                if all(predicate(substitution | candidate.literal_substitution) for predicate in candidate.predicates):
                    return DerivationTree(
                        candidate.terminal,
                        interleave(substitution),
                        derived_from=nt,
                        rhs_rule=candidate,
                        is_literal=False,
                        literal_group=None,
                        frozen=False
                    )
            return None

    def build_tree(self, nt: NT, candidate: RHSRule[NT, T, G]) -> DerivationTree[NT, T, G] | None:
        if not list(candidate.non_terminals):
            # rule only derives terminals, therefore arguments has no nonterminals, but to silence the type checker:
            params: list[G] = [lit for lit in candidate.arguments if isinstance(lit, ConstantArgument)]
            cands: tuple[DerivationTree[NT, T, G], ...] = tuple(
                map(lambda p: DerivationTree(p.value, tuple(),
                                             derived_from=None, rhs_rule=candidate, frozen=False,
                                             is_literal=True, literal_group=p.origin), params))
            if all(predicate({}) for predicate in candidate.predicates):
                return DerivationTree(candidate.terminal, cands, derived_from=nt, rhs_rule=candidate,
                                      is_literal=False, literal_group=None, frozen=False)
            else:
                return None
        else:
            # rule derives non-terminals
            children: tuple[DerivationTree[NT, T, G], ...] = ()
            substitution: dict[str, DerivationTree[NT, T, G]] = {}

            for a in candidate.arguments:
                if isinstance(a, NonTerminalArgument):
                    child_tree = self.sample_random_term(a.origin)
                    while child_tree is None:
                        child_tree = self.sample_random_term(a.origin)
                    if child_tree is not None:
                        children = children + (child_tree,)
                        if a.name is not None:
                            substitution[a.name] = child_tree
                elif isinstance(a, ConstantArgument):
                    children = children + (DerivationTree(a.value, tuple(), derived_from=None, rhs_rule=candidate,
                                       is_literal=True, literal_group=a.origin, frozen=False),)
                else:
                    raise ValueError("rule argument is neither NonTerminalArgument nor ConstantArgument")

            if all((predicate(substitution | candidate.literal_substitution) for predicate in candidate.predicates)):
                return DerivationTree(
                        candidate.terminal,
                        children,
                        derived_from=nt,
                        rhs_rule=candidate,
                        is_literal=False,
                        literal_group=None,
                        frozen=False
                    )
            else:
                return None

    def sample_random_term_annotated(self, nt: NT, cs: int) -> DerivationTree[NT, T, G] | None:
        applicable: list[tuple[RHSRule[NT, T, G], int]] = []
        #for (lhs, rhs), n in self.rules:
        #    new_cs = cs + n
        #    if lhs == nt and new_cs <= self.max_tree_depth:
        #        applicable.append(rhs)
        for rhs, n in self._annotated_rules[nt]:
            new_cs = cs + n
            if new_cs <= self.max_tree_depth:
                applicable.append((rhs, new_cs))

        while applicable:
            candidate, next_cs = random.choice(applicable)
            applicable.remove((candidate, next_cs))
            tree = self.build_tree_annotated(nt, next_cs, candidate)
            if tree is not None:
                return tree
        return None

    def sample_random_term(self, nt: NT) -> DerivationTree[NT, T, G] | None:
        applicable: list[RHSRule[NT, T, G]] = []
        for lhs, rules in self._rules.items():
            if lhs == nt:
                for rhs in rules:
                    applicable.append(rhs)

        while applicable:
            candidate = random.choice(applicable)
            applicable.remove(candidate)
            tree = self.build_tree(nt, candidate)
            if tree is not None:
                return tree
        return None


    def sample(self, size: int, non_terminal: NT, infinite=False, max_depth: int | None = None) -> set[DerivationTree[NT, T, G]]:
        """
        Sample a list of length size of random trees from the search space.
        """
        sample: set[DerivationTree[NT, T, G]] = set()
        if not infinite:
            while len(sample) < size:
                term: DerivationTree[NT, T, G] | None = self.sample_random_term(non_terminal)
                while term is None:
                    term = self.sample_random_term(non_terminal)
                sample.add(term)
        else:
            self.min_size: int = self.minimum_tree_depth(non_terminal)
            if max_depth is not None:
                self.max_tree_depth = max_depth
            else:
                self.max_tree_depth = self.min_size + 10000
            if self.max_tree_depth < self.min_size:
                raise ValueError(f"max_tree_depth {self.max_tree_depth} is less than minimum tree depth {self.min_size}")
            # for _ in range(size*10):
            while len(sample) < size:
                # this only works for big search spaces and small sample sizes.
                # If the sample size is near the size of the search space, this might loop for a long time
                cs = 0
                term: DerivationTree[NT, T, G] | None = self.sample_random_term_annotated(non_terminal, cs)
                while term is None:
                    term = self.sample_random_term_annotated(non_terminal, cs)
                sample.add(term)
                # if len(sample) >= size:
                #    break
        return sample

    def sample_tree(self, non_terminal: NT, max_depth: int | None = None) -> DerivationTree[NT, T, G]:
        """
        Sample a random tree from the search space.
        """
        tree = self.sample(1, non_terminal, max_depth=max_depth)
        return tree.pop()  # return the only element in the set


    # old methods from SolutionSpace, that are adapted for DerivationTree


    #@override
    def prune(self) -> SearchSpace[NT, T, G]:
        """Keep only productive rules."""

        ground_types: set[NT] = set()
        queue: set[NT] = set()
        inverse_grammar: dict[NT, set[tuple[NT, frozenset[NT]]]] = defaultdict(set)

        for n, exprs in self._rules.items():
            for expr in exprs:
                non_terminals = expr.non_terminals
                for m in non_terminals:
                    inverse_grammar[m].add((n, non_terminals))
                if not non_terminals:
                    queue.add(n)

        while queue:
            n = queue.pop()
            if n not in ground_types:
                ground_types.add(n)
                for m, non_terminals in inverse_grammar[n]:
                    if m not in ground_types and all(t in ground_types for t in non_terminals):
                        queue.add(m)

        return SearchSpace[NT, T, G](
            defaultdict(
                deque,
                {
                    target: deque(
                        possibility
                        for possibility in self._rules[target]
                        if all(t in ground_types for t in possibility.non_terminals)
                    )
                    for target in ground_types
                },
            )
        )

    #@override
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

    #@override
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

    def contains_tree(self, start: NT, tree: DerivationTree[NT, T, G]) -> bool:
        """Check if the solution space contains a given `tree` derivable from `start`."""
        if start not in self.nonterminals():
            return False

        stack: deque[tuple | Callable] = deque([(start, tree)])
        results: deque[bool] = deque()

        def get_inputs(count: int) -> list[bool]:
            return [results.pop() for _ in range(count)]

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
        assert len(results) == 1
        return results.pop()
