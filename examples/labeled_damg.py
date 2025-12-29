from src.cl3s import (SpecificationBuilder, Constructor, Literal, Var,
                      SearchSpaceSynthesizer, DerivationTree, DataGroup, Group)

from src.cl3s.genetic_programming.evolutionary_search import TournamentSelection
from src.cl3s.scikit.graph_kernel import WeisfeilerLehmanKernel

from grakel.utils import graph_from_networkx
from itertools import product


import numpy as np
from src.cl3s.scikit.bayesian_optimization import BayesianOptimization
from src.cl3s.scikit.acquisition_function import SimplifiedExpectedImprovement, ExpectedImprovement
from sklearn.gaussian_process import GaussianProcessRegressor

from typing import Any

import networkx as nx
import matplotlib.pyplot as plt
from itertools import accumulate

import timeit

class Labeled_DAMG_Repository:
    """
            The terms have to be in normal form under the following term rewriting system:

            associativity laws:  (handled with types)

            beside(beside(x,y),z)
            ->
            beside(x, beside(y,z))

            before(before(x,y),z)
            ->
            before(x, before(y,z))

            abiding law:   (handled with types)

            beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p')))
            ->
            before(m+m', n+r, p+p', beside(w(m,n),y(m',r)), beside(x(n,p),z(r,p')))

            neutrality of edge:   (handled with types)

            before(edge(), x)
            ->
            x

            before(x, edge())
            ->
            x

            swap laws:    (they need to be term predicates :-(...)


            before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
            ->
            beside(y(m,q),x(n,p))


            (the three laws below may be handled with types...)
            (one could introduce an "only edges" flag, like ID and non_ID,
            but that would also require a lot of thinking...)
            (nope, that's not possible, because we need at least one term predicate and therefore cannot use
            function types in suffix(), but must abstract with argument() ... this prevents handling these
            laws with types...)

            before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
            ->
            swap(m + n + p, m, n+p)

            before(swap(m+n, m, n), swap(n+m, n, m))
            ->
            copy(m+n, edge())

            before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
            ->
            swap(m + n + p, m+n, p)


            additionally, we should satisfy the following law (which is implicit in Gibbon's paper...)
            (this is handled with types, too)

            beside(swap(n, 0, n), swap(m, 0, m))
            ->
            swap(n+m, 0, n+m)

            """

    def __init__(self, labels, dimensions):
        # additionally to labeled nodes, we have (unlabelled) edges, that needs to be handled additionally
        if "swap" in labels:
            raise ValueError("Label 'swap' is reserved and cannot be used as a node label.")
        self.labels = labels
        self.dimensions = dimensions
        self.id_seed = 0

    class Para(Group):
        name = "Para"

        def __init__(self, labels, dimensions):
            self.labels = list(labels) + [None]
            self.dimensions = list(dimensions) + [None]

        def __iter__(self):
            for l in self.labels:
                if l is not None:
                    for i in self.dimensions:
                        if i is not None:
                            for o in self.dimensions:
                                if o is not None:
                                    yield l, i, o
                                    if i == o:
                                        for n in range (0, i):
                                            m = i - n
                                            assert m > 0
                                            yield ("swap", n, m), i, o
            yield None


        def __contains__(self, value):
            return value is None or ((isinstance(value, tuple) and len(value) == 3 and (value[0] in self.labels or
                                                                      (value[0][0] == "swap" and
                                                                       (value[0][1] in self.dimensions or value[0][1] == 0) and
                                                                       value[0][2] in self.dimensions))
                    and value[1] in self.dimensions and value[2] in self.dimensions))

    class ParaTuples(Group):
        name = "ParaTuples"

        def __init__(self, para, max_length=3):
            self.para = para
            self.max_length = max_length

        def __iter__(self):
            result = set()

            for n in range(0, self.max_length + 1):
                if n == 0:
                    result.add(())
                else:
                    old_result = result.copy()
                    for para in self.para:
                        for suffix in old_result:
                            result.add((para,) + suffix)
            yield from result

        def __contains__(self, value):
            return value is None or (isinstance(value, tuple) and all(True if v is None else v in self.para for v in value))

        def normalform(self, value) -> bool:
            """
            beside(swap(n, 0, n), swap(m, 0, m))
            ->
            swap(n+m, 0, n+m)
            """
            if value is None:
                return True # because synthesis enforces, that every variance for None will be in normal form
            for l, r in zip(value[:-1], value[1:]):
                if l is not None and r is not None:
                    if len(l) == 3 and l[0] is not None:
                        label, i, o = l
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap" and label[1] == 0:
                            if len(r) == 3 and r[0] is not None:
                                right_label, right_i, right_o = r
                                if isinstance(right_label, tuple) and len(right_label) == 3 and right_label[0] == "swap" and right_label[1] == 0:
                                    """
                                    beside(swap(n, 0, n), swap(m, 0, m))
                                    ->
                                    swap(n+m, 0, n+m)
                                    """
                                    return False
            return True

        def normalize(self, value):
            while(not self.normalform(value)):
                new_value = value
                index = 0 # index in new_value
                for l, r in zip(value[:-1], value[1:]):
                    if len(l) == 3:
                        label, i, o = l
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap" and label[1] == 0:
                            n = label[2]
                            if len(r) == 3 and r[0] is not None:
                                right_label, right_i, right_o = r
                                if (isinstance(right_label, tuple) and len(right_label) == 3 and
                                        right_label[0] == "swap" and right_label[1] == 0):
                                    m = right_label[2]
                                    """
                                    beside(swap(n, 0, n), swap(m, 0, m))
                                    ->
                                    swap(n+m, 0, n+m)
                                    """
                                    # i < len(new_value) is an invariant, because len(zip(value[:-1], value[1:])) == len(value) - 1
                                    before_i = new_value[:index]
                                    after_i = new_value[index+2:]
                                    new_value = before_i + ((("swap", 0, n + m), n + m, n + m),) + after_i
                                    break
                    index += 1
                value = new_value
            return value

    class ParaTupleTuples(Group):
        name = "ParaTupleTuples"

        def __init__(self, para_tuples):
            self.para_tuples = para_tuples

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value):
            return value is None or (isinstance(value, tuple) and all(True if v is None else v in self.para_tuples for v in value))

        def normalform(self, value) -> bool:
            """
            The associativity laws are handled by the way we use python tuples.
            The abiding law is an invariance, because otherwise we couldn't use tuples of tuples.

            Therefore, we only need to check:
            - Neutrality of edges
            - Swap laws
            - Unique representation of parallel edges (swaps with n=0) (handled in ParaTuples)
            """
            if value is None:
                return True # because synthesis enforces, that every variance for None will be in normal form
            for l, r in zip(value[:-1], value[1:]):
                if l is not None and r is not None:
                    if not (self.para_tuples.normalform(l) and self.para_tuples.normalform(r)):
                        return False
                    if len(l) == 1 and l[0] is not None:
                        label, i, o = l[0]
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap":
                            m = label[1]
                            n = label[2]
                            if m == 0:
                                """
                                before(edge(), x)
                                ->
                                x
                                """
                                return False
                            if len(r) == 1 and r[0] is not None:
                                """
                                before(swap(m+n, m, n), swap(n+m, n, m))
                                ->
                                copy(m+n, edge())
                                """
                                right_label, right_i, right_o = r[0]
                                if isinstance(right_label, tuple) and len(right_label) == 3 and right_label[0] == "swap":
                                    right_m = right_label[1]
                                    right_n = right_label[2]
                                    if right_m is not None and right_n is not None:
                                        if m == right_n and n == right_m:
                                            return False
                    if len(r) == 1 and r[0] is not None:
                        """
                        before(x, edge())
                        ->
                        x
                        """
                        label, i, o = r[0]
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap":
                            if label[1] == 0:
                                return False
                    if len(l) == 2 and len(r) == 2:
                        left_first, left_second = l
                        right_first, right_second = r
                        if left_first is not None and left_second is not None and right_first is not None and right_second is not None:
                            label_l_1, i_l_1, o_l_1 = left_first
                            label_l_2, i_l_2, o_l_2 = left_second
                            label_r_1, i_r_1, o_r_1 = right_first
                            label_r_2, i_r_2, o_r_2 = right_second
                            if isinstance(label_l_1, tuple) and len(label_l_1) == 3 and label_l_1[0] == "swap":
                                m = label_l_1[1]
                                n = label_l_1[2]
                                if isinstance(label_l_2, tuple) and len(label_l_2) == 3 and label_l_2[0] == "swap" and label_l_2[1] == 0:
                                    p = label_l_2[2]
                                    if isinstance(label_r_1, tuple) and len(label_r_1) == 3 and label_r_1[0] == "swap" and label_r_1[1] == 0:
                                        right_n = label_r_1[2]
                                        if isinstance(label_r_2, tuple) and len(label_r_2) == 3 and label_r_2[0] == "swap":
                                            right_m = label_r_2[1]
                                            right_p = label_r_2[2]
                                            if m is not None and n is not None and p is not None and right_m is not None and right_n is not None and right_p is not None:
                                                if m == right_m and n == right_n and p == right_p:
                                                    """
                                                    before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
                                                    ->
                                                    swap(m + n + p, m, n+p)
                                                    """
                                                    return False
                            if isinstance(label_l_1, tuple) and len(label_l_1) == 3 and label_l_1[0] == "swap" and label_l_1[1] == 0:
                                m = label_l_1[2]
                                if isinstance(label_l_2, tuple) and len(label_l_2) == 3 and label_l_2[0] == "swap":
                                    n = label_l_2[1]
                                    p = label_l_2[2]
                                    if isinstance(label_r_1, tuple) and len(label_r_1) == 3 and label_r_1[0] == "swap":
                                        right_m = label_r_1[1]
                                        right_p = label_r_1[2]
                                        if isinstance(label_r_2, tuple) and len(label_r_2) == 3 and label_r_2[0] == "swap" and label_r_2[1] == 0:
                                            right_n = label_r_2[2]
                                            if m is not None and n is not None and p is not None and right_m is not None and right_n is not None and right_p is not None:
                                                if m == right_m and n == right_n and p == right_p:
                                                    """
                                                    before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
                                                    ->
                                                    swap(m + n + p, m+n, p)
                                                    """
                                                    return False

            for l, m, r in zip(value[:-2], value[1:-1], value[2:]):
                if l is not None and m is not None and r is not None:
                    if len(l) == 1 and l[0] is not None:
                        left_label, left_i, left_o = l[0]
                        if isinstance(left_label, tuple) and len(left_label) == 3 and left_label[0] == "swap":
                            left_m = left_label[1]
                            left_n = left_label[2]
                            if len(m) == 2 and len(r) == 1 and r[0] is not None:
                                mid_first, mid_second = m
                                right_label, right_i, right_o = r[0]
                                if mid_first is not None and mid_second is not None:
                                    mid_first_label, mid_n, mid_p = mid_first
                                    mid_second_label, mid_m, mid_q = mid_second
                                    if isinstance(right_label, tuple) and len(right_label) == 3 and right_label[0] == "swap":
                                        right_p = right_label[1]
                                        right_q = right_label[2]
                                        if left_m is not None and left_n is not None and mid_m is not None and mid_n is not None and mid_p is not None and mid_q is not None and right_p is not None and right_q is not None:
                                            if left_m == mid_m and left_n == mid_n and right_p == mid_p and right_q == mid_q:
                                                """
                                                before(swap(m + n, m, n), before(beside(x(n, p), y(m, q)), swap(p + q, p, q)))
                                                ->
                                                beside(y(m, q), x(n, p))
                                                """
                                                return False
            return True

        def normalize(self, value):
            if (not self.normalform(value)):
                value = tuple(map(self.para_tuples.normalize, value))
            while(not self.normalform(value)):
                new_value = value
                index = 0 # index in new_value
                for l, r in zip(value[:-1], value[1:]):
                    if len(l) == 1:
                        label, i, o = l[0]
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap":
                            m = label[1]
                            n = label[2]
                            if m == 0:
                                """
                                before(edge(), x)
                                ->
                                x
                                """
                                # i < len(new_value) is an invariant, because len(zip(value[:-1], value[1:])) == len(value) - 1
                                before_i = new_value[:index]
                                after_i = new_value[index + 2:]
                                new_value = before_i + (r,) + after_i
                                break
                            if len(r) == 1:
                                """
                                before(swap(m+n, m, n), swap(n+m, n, m))
                                ->
                                copy(m+n, edge())
                                """
                                right_label, right_i, right_o = r[0]
                                if (isinstance(right_label, tuple) and len(right_label) == 3
                                        and right_label[0] == "swap"):
                                    right_m = right_label[1]
                                    right_n = right_label[2]
                                    if m == right_n and n == right_m:
                                        before_i = new_value[:index]
                                        after_i = new_value[index + 2:]
                                        new_value = before_i + (((("swap", 0, n + m), n + m, n + m),),) + after_i
                                        break
                    if len(r) == 1 and r[0] is not None:
                        """
                        before(x, edge())
                        ->
                        x
                        """
                        label, i, o = r[0]
                        if isinstance(label, tuple) and len(label) == 3 and label[0] == "swap":
                            if label[1] == 0:
                                before_i = new_value[:index]
                                after_i = new_value[index + 2:]
                                new_value = before_i + (l,) + after_i
                                break
                    if len(l) == 2 and len(r) == 2:
                        left_first, left_second = l
                        right_first, right_second = r
                        if left_first and left_second and right_first and right_second:
                            label_l_1, i_l_1, o_l_1 = left_first
                            label_l_2, i_l_2, o_l_2 = left_second
                            label_r_1, i_r_1, o_r_1 = right_first
                            label_r_2, i_r_2, o_r_2 = right_second
                            if isinstance(label_l_1, tuple) and len(label_l_1) == 3 and label_l_1[0] == "swap":
                                m = label_l_1[1]
                                n = label_l_1[2]
                                if (isinstance(label_l_2, tuple) and len(label_l_2) == 3 and label_l_2[0] == "swap"
                                        and label_l_2[1] == 0):
                                    p = label_l_2[2]
                                    if (isinstance(label_r_1, tuple) and len(label_r_1) == 3 and label_r_1[0] == "swap"
                                            and label_r_1[1] == 0):
                                        right_n = label_r_1[2]
                                        if (isinstance(label_r_2, tuple) and len(label_r_2) == 3 and
                                                label_r_2[0] == "swap"):
                                            right_m = label_r_2[1]
                                            right_p = label_r_2[2]
                                            if m == right_m and n == right_n and p == right_p:
                                                """
                                                before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
                                                ->
                                                swap(m + n + p, m, n+p)
                                                """
                                                before_i = new_value[:index]
                                                after_i = new_value[index + 2:]
                                                new_value = before_i + (((("swap", m, n + p), m + n + p, m + n + p),),) + after_i
                                                break
                            if (isinstance(label_l_1, tuple) and len(label_l_1) == 3 and label_l_1[0] == "swap" and
                                    label_l_1[1] == 0):
                                m = label_l_1[2]
                                if isinstance(label_l_2, tuple) and len(label_l_2) == 3 and label_l_2[0] == "swap":
                                    n = label_l_2[1]
                                    p = label_l_2[2]
                                    if isinstance(label_r_1, tuple) and len(label_r_1) == 3 and label_r_1[0] == "swap":
                                        right_m = label_r_1[1]
                                        right_p = label_r_1[2]
                                        if (isinstance(label_r_2, tuple) and len(label_r_2) == 3 and
                                                label_r_2[0] == "swap" and label_r_2[1] == 0):
                                            right_n = label_r_2[2]
                                            if m == right_m and n == right_n and p == right_p:
                                                """
                                                before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
                                                ->
                                                swap(m + n + p, m+n, p)
                                                """
                                                before_i = new_value[:index]
                                                after_i = new_value[index + 2:]
                                                new_value = before_i + (((("swap", m + n, p), m + n + p, m + n + p),),) + after_i
                                                break
                for l, m, r in zip(value[:-2], value[1:-1], value[2:]):
                    if len(l) == 1:
                        left_label, left_i, left_o = l[0]
                        if isinstance(left_label, tuple) and len(left_label) == 3 and left_label[0] == "swap":
                            left_m = left_label[1]
                            left_n = left_label[2]
                            if len(m) == 2 and len(r) == 1:
                                mid_first, mid_second = m
                                right_label, right_i, right_o = r[0]
                                mid_first_label, mid_n, mid_p = mid_first
                                mid_second_label, mid_m, mid_q = mid_second
                                if isinstance(right_label, tuple) and len(right_label) == 3 and right_label[0] == "swap":
                                    right_p = right_label[1]
                                    right_q = right_label[2]
                                    if left_m == mid_m and left_n == mid_n and right_p == mid_p and right_q == mid_q:
                                        """
                                        before(swap(m + n, m, n), before(beside(x(n, p), y(m, q)), swap(p + q, p, q)))
                                        ->
                                        beside(y(m, q), x(n, p))
                                        """
                                        before_i = new_value[:index]
                                        after_i = new_value[index + 3:]
                                        new_value = before_i + ((mid_second, mid_first),) + after_i
                                        break
                value = new_value
            return value





    @staticmethod
    def swaplaw1(head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
        ->
        beside(y(m,q),x(n,p))

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """

        left = head.root
        right = tail.root

        if "beside_singleton" in left and "before_cons" in right:
            if len(head.children) != 5 or len(tail.children) != 9:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_head = tail.children[7]
            right_tail = tail.children[8]

            left_term_root = left_term.root
            right_head_root = right_head.root
            right_tail_root = right_tail.root
            if (left_term_root == "swap") and "beside_cons" in right_head_root and "before_singleton" in right_tail_root:
                if len(left_term.children) != 19 or len(right_head.children) != 11 or len(right_tail.children) != 6:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_term = right_tail.children[5]
                right_head_tail_root = right_head_tail.root
                right_tail_term_root = right_tail_term.root
                if "beside_singleton" in right_head_tail_root and "beside_singleton" in right_tail_term_root:
                    if len(right_head_tail.children) != 5 or len(right_tail_term.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    y_m = right_head_tail.children[0]
                    y_q = right_head_tail.children[1]
                    right_swap = right_tail_term.children[4]
                    right_swap_root = right_swap.root
                    if right_swap_root == "swap":
                        if len(right_swap.children) != 19:
                            raise ValueError("Derivation trees have not the expected shape.")
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False
            elif (left == "swap") and "beside_cons" in right_head_root and "before_cons" in right_tail_root:
                if len(left_term.children) != 19 or len(right_head.children) != 11 or len(right_tail.children) != 9:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_head = right_tail.children[7]
                right_head_tail_root = right_head_tail.root
                right_tail_term_root = right_tail_head.root
                if "beside_singleton" in right_head_tail_root and "beside_singleton" in right_tail_term_root:
                    if len(right_head_tail.children) != 5 or len(right_tail_head.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    y_m = right_head_tail.children[0]
                    y_q = right_head_tail.children[1]
                    right_swap = right_tail_head.children[4]
                    right_swap_root = right_swap.root
                    if right_swap_root == "swap":
                        if len(right_swap.children) != 19:
                            raise ValueError("Derivation trees have not the expected shape.")
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False
        return True

    @staticmethod
    def swaplaw2(head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
        ->
        swap(m + n + p, m, n+p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched

        """

        left = head.root
        right = tail.root

        if "beside_cons" in left and "before_singleton" in right:
            if len(head.children) != 11 or len(tail.children) != 6:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[5]

            left_swap = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if (left_swap == "swap") and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 19 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4] # swap(p, 0, p)
                right_head = right_term.children[9] # swap(n, 0, n)
                right_tail = right_term.children[10]

                left_tail_swap = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_swap == "edges" and right_head_root == "edges" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 17 or len(right_head.children) != 17 or len(right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]

                    right_tail_term = right_tail.children[4] # swap(m+p, m, p)
                    right_tail_term_root = right_tail_term.root
                    if (right_tail_term_root == "swap") and n == right_n:
                        if len(right_tail_term.children) != 19:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False
        elif "beside_cons" in left and "before_cons" in right:
            if len(head.children) != 11 or len(tail.children) != 9:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[7]

            left_swap = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if (left_swap == "swap") and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 19 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[9]  # swap(n, 0, n)
                right_tail = right_term.children[10]

                left_tail_swap = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_swap == "edges" and right_head_root == "edges" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 17 or len(right_head.children) != 17 or len(
                            right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]

                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)
                    right_tail_term_root = right_tail_term.root
                    if right_tail_term_root == "swap" and n == right_n:
                        if len(right_tail_term.children) != 19:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False
        return True

    @staticmethod
    def swaplaw3(head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(swap(m+n, m, n), swap(n+m, n, m))
        ->
        copy(m+n, edge())

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        left = head.root
        right = tail.root

        if "beside_singleton" in left and "before_singleton" in right:
            if len(head.children) != 5 or len(tail.children) != 6:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_term = tail.children[5]


            left_term_root = left_term.root
            right_term_root = right_term.root
            if (left_term_root == "swap") and "beside_singleton" in right_term_root:
                if len(left_term.children) != 19 or len(right_term.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]

                right_beside = right_term.children[4]
                right_beside_root = right_beside.root
                if right_beside_root == "swap" or right_beside_root == "swap":
                    if len(right_beside.children) != 19:
                        raise ValueError("Derivation trees have not the expected shape.")
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False
        elif "beside_singleton" in left and "before_cons" in right:
            if len(head.children) != 5 or len(tail.children) != 9:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_term = tail.children[7]


            left_term_root = left_term.root
            right_term_root = right_term.root
            if (left_term_root == "swap") and "beside_singleton" in right_term_root:
                if len(left_term.children) != 19 or len(right_term.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]

                right_beside = right_term.children[4]
                right_beside_root = right_beside.root
                if right_beside_root == "swap":
                    if len(right_beside.children) != 4:
                        raise ValueError("Derivation trees have not the expected shape.")
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False
        return True

    @staticmethod
    def swaplaw4(head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
        ->
        swap(m + n + p, m+n, p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        left = head.root
        right = tail.root

        if "beside_cons" in left and "before_singleton" in right:
            if len(head.children) != 11 or len(tail.children) != 6:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[5]

            left_head_root = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if left_head_root == "edges" and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 17 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[0]

                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                left_tail_term_root = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_term_root == "swap" and right_head_root == "swap" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 19 or len(right_head.children) != 19 or len(right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]

                    right_m = right_head.children[1]
                    right_p = right_head.children[2]

                    right_tail_term = right_tail.children[4]
                    right_tail_term_root = right_tail_term.root

                    if right_tail_term_root == "edges" and m == right_m and p == right_p:
                        if len(right_tail_term.children) != 17:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False

        elif "beside_cons" in left and "before_cons" in right:
            if len(head.children) != 11 or len(tail.children) != 9:
                raise ValueError("Derivation trees have not the expected shape.")
            left_head = head.children[9]
            left_tail = head.children[10]
            right_term = tail.children[7]

            left_head_root = left_head.root
            left_tail_root = left_tail.root
            right_term_root = right_term.root
            if left_head_root == "edges" and "beside_singleton" in left_tail_root and "beside_cons" in right_term_root:
                if len(left_head.children) != 17 or len(left_tail.children) != 5 or len(right_term.children) != 11:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_head.children[0]

                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                left_tail_term_root = left_tail_term.root
                right_head_root = right_head.root
                right_tail_root = right_tail.root
                if left_tail_term_root == "swap" and right_head_root == "swap" and "beside_singleton" in right_tail_root:
                    if len(left_tail_term.children) != 19 or len(right_head.children) != 19 or len(
                            right_tail.children) != 5:
                        raise ValueError("Derivation trees have not the expected shape.")
                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]

                    right_m = right_head.children[1]
                    right_p = right_head.children[2]

                    right_tail_term = right_tail.children[4]
                    right_tail_term_root = right_tail_term.root

                    if right_tail_term_root == "edges" and m == right_m and p == right_p:
                        if len(right_tail_term.children) != 17:
                            raise ValueError("Derivation trees have not the expected shape.")
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False
        return True


    def specification(self):
        labels = DataGroup("para", self.labels)
        para_labels = self.Para(self.labels, self.dimensions)
        paratuples = self.ParaTuples(para_labels, max_length=max(self.dimensions))
        paratupletuples = self.ParaTupleTuples(paratuples)
        dimension = DataGroup("dimension", self.dimensions)

        return {
            # atomic components are nodes and edges
            # but edges are a special case of swaps (swaps, that do not change anything)
            # so we can just define edges as a special case of swaps here
            #"edge": Constructor("DAG", Constructor("input", Literal(1))
            #                    & Constructor("input", Literal(None))
            #                    & Constructor("output", Literal(1))
            #                    & Constructor("output", Literal(None))
            #                    & Constructor("structure", Literal(((("swap", Literal(0), Literal(1)),),)))
            #                    & Constructor("structure", Literal(((None,),)))),
            # (m parallel) edges are swaps with n=0
            "edges": SpecificationBuilder()
            .parameter("io", dimension)
            #.parameter("para", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], v["io"]),
            #                                           (("swap", 0, None), v["io"], v["io"]),
            #                                           (("swap", None, v["io"]), v["io"], v["io"]),
            #                                           (("swap", None, None), v["io"], v["io"]),
            #                                           (("swap", 0, v["io"]), v["io"], None),
            #                                           (("swap", 0, None), v["io"], None),
            #                                           (("swap", None, v["io"]), v["io"], None),
            #                                           (("swap", None, None), v["io"], None),
            #                                           (("swap", 0, v["io"]), None, v["io"]),
            #                                           (("swap", 0, None), None, v["io"]),
            #                                           (("swap", None, v["io"]), None, v["io"]),
            #                                           (("swap", None, None), None, v["io"]),
            #                                           (("swap", 0, v["io"]), None, None),
            #                                           (("swap", 0, None), None, None),
            #                                           (("swap", None, v["io"]), None, None),
            #                                           (("swap", None, None), None, None), None])
                                                        # (None, None, None)])
            .parameter("para1", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], v["io"])])
            .parameter("para2", para_labels, lambda v: [(("swap", 0, None), v["io"], v["io"])])
            .parameter("para3", para_labels, lambda v: [(("swap", None, v["io"]), v["io"], v["io"])])
            .parameter("para4", para_labels, lambda v: [(("swap", None, None), v["io"], v["io"])])
            .parameter("para5", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], None)])
            .parameter("para6", para_labels, lambda v: [(("swap", 0, None), v["io"], None)])
            .parameter("para7", para_labels, lambda v: [(("swap", None, v["io"]), v["io"], None)])
            .parameter("para8", para_labels, lambda v: [(("swap", None, None), v["io"], None)])
            .parameter("para9", para_labels, lambda v: [(("swap", 0, v["io"]), None, v["io"])])
            .parameter("para10", para_labels, lambda v: [(("swap", 0, None), None, v["io"])])
            .parameter("para11", para_labels, lambda v: [(("swap", None, v["io"]), None, v["io"])])
            .parameter("para12", para_labels, lambda v: [(("swap", None, None), None, v["io"])])
            .parameter("para13", para_labels, lambda v: [(("swap", 0, v["io"]), None, None)])
            .parameter("para14", para_labels, lambda v: [(("swap", 0, None), None, None)])
            .parameter("para15", para_labels, lambda v: [(("swap", None, v["io"]), None, None)])
            .parameter("para16", para_labels, lambda v: [(("swap", None, None), None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                & Constructor("structure", Var("para8"))
                                & Constructor("structure", Var("para9"))
                                & Constructor("structure", Var("para10"))
                                & Constructor("structure", Var("para11"))
                                & Constructor("structure", Var("para12"))
                                & Constructor("structure", Var("para13"))
                                & Constructor("structure", Var("para14"))
                                & Constructor("structure", Var("para15"))
                                & Constructor("structure", Var("para16"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("ID")
                    ),

            "swap": SpecificationBuilder()
            .parameter("io", dimension)
            .parameter("n", dimension, lambda v: range(1, v["io"]))
            .parameter("m", dimension, lambda v: [v["io"] - v["n"]]) # m > 0
            #.parameter("para", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], v["io"]),
            #                                           (("swap", v["n"], None), v["io"], v["io"]),
            #                                           (("swap", None, v["m"]), v["io"], v["io"]),
            #                                           (("swap", None, None), v["io"], v["io"]),
            #                                           (("swap", v["n"], v["m"]), v["io"], None),
            #                                           (("swap", v["n"], None), v["io"], None),
            #                                           (("swap", None, v["m"]), v["io"], None),
            #                                           (("swap", None, None), v["io"], None),
            #                                           (("swap", v["n"], v["m"]), None, v["io"]),
            #                                           (("swap", v["n"], None), None, v["io"]),
            #                                           (("swap", None, v["m"]), None, v["io"]),
            #                                           (("swap", None, None), None, v["io"]),
            #                                           (("swap", v["n"], v["m"]), None, None),
            #                                           (("swap", v["n"], None), None, None),
            #                                           (("swap", None, v["m"]), None, None),
            #                                           (("swap", None, None), None, None), None])
                                                       #(None, None, None)])
            .parameter("para1", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], v["io"])])
            .parameter("para2", para_labels, lambda v: [(("swap", v["n"], None), v["io"], v["io"])])
            .parameter("para3", para_labels, lambda v: [(("swap", None, v["m"]), v["io"], v["io"])])
            .parameter("para4", para_labels, lambda v: [(("swap", None, None), v["io"], v["io"])])
            .parameter("para5", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], None)])
            .parameter("para6", para_labels, lambda v: [(("swap", v["n"], None), v["io"], None)])
            .parameter("para7", para_labels, lambda v: [(("swap", None, v["m"]), v["io"], None)])
            .parameter("para8", para_labels, lambda v: [(("swap", None, None), v["io"], None)])
            .parameter("para9", para_labels, lambda v: [(("swap", v["n"], v["m"]), None, v["io"])])
            .parameter("para10", para_labels, lambda v: [(("swap", v["n"], None), None, v["io"])])
            .parameter("para11", para_labels, lambda v: [(("swap", None, v["m"]), None, v["io"])])
            .parameter("para12", para_labels, lambda v: [(("swap", None, None), None, v["io"])])
            .parameter("para13", para_labels, lambda v: [(("swap", v["n"], v["m"]), None, None)])
            .parameter("para14", para_labels, lambda v: [(("swap", v["n"], None), None, None)])
            .parameter("para15", para_labels, lambda v: [(("swap", None, v["m"]), None, None)])
            .parameter("para16", para_labels, lambda v: [(("swap", None, None), None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                & Constructor("structure", Var("para8"))
                                & Constructor("structure", Var("para9"))
                                & Constructor("structure", Var("para10"))
                                & Constructor("structure", Var("para11"))
                                & Constructor("structure", Var("para12"))
                                & Constructor("structure", Var("para13"))
                                & Constructor("structure", Var("para14"))
                                & Constructor("structure", Var("para15"))
                                & Constructor("structure", Var("para16"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),
            "node": SpecificationBuilder()
            .parameter("l", labels)
            .parameter("i", dimension)
            .parameter("o", dimension)
            #.parameter("para", para_labels, lambda v: [(v["l"], v["i"], v["o"]),
            #                                           (v["l"], v["i"], None),
            #                                           (v["l"], None, v["o"]),
            #                                           (None, v["i"], v["o"]),
            #                                           (v["l"], None, None),
            #                                           (None, None, v["o"]),
            #                                           (None, v["i"], None), None])
                                                       #(None, None, None)])
            .parameter("para1", para_labels, lambda v: [(v["l"], v["i"], v["o"])])
            .parameter("para2", para_labels, lambda v: [(v["l"], v["i"], None)])
            .parameter("para3", para_labels, lambda v: [(v["l"], None, v["o"])])
            .parameter("para4", para_labels, lambda v: [(None, v["i"], v["o"])])
            .parameter("para5", para_labels, lambda v: [(v["l"], None, None)])
            .parameter("para6", para_labels, lambda v: [(None, None, v["o"])])
            .parameter("para7", para_labels, lambda v: [(None, v["i"], None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para1"))
                                & Constructor("structure", Var("para2"))
                                & Constructor("structure", Var("para3"))
                                & Constructor("structure", Var("para4"))
                                & Constructor("structure", Var("para5"))
                                & Constructor("structure", Var("para6"))
                                & Constructor("structure", Var("para7"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),
            #'''
            "beside_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) == 1)
            .parameter("para", para_labels, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: (len(v["para"]) == 3 and
                                             (v["para"][1] == v["i"] if v["para"][1] is not None else True) and
                                             (v["para"][2] == v["o"] if v["para"][2] is not None else True)
                                             ) if v["para"] is not None else True)
            .suffix(
                ((Constructor("DAG_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("non_ID")
                  )
                 **
                 (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                  & Constructor("non_ID") & Constructor("last", Constructor("non_ID"))
                  )
                 )
                &
                ((Constructor("DAG_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("ID")
                     )
                 **
                 (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                     & Constructor("ID")
                     )
                 )),

            "beside_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("i1", dimension)
            .parameter("i2", dimension, lambda v: [v["i"] - v["i1"]])
            .parameter("o", dimension)
            .parameter("o1", dimension)
            .parameter("o2", dimension, lambda v: [v["o"] - v["o1"]])
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) > 1)
            .parameter("head", para_labels, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["head"] is None or (len(v["head"]) == 3 and
                                                                  (v["head"][1] == v["i1"] or v["head"][1] is None) and
                                                                  (v["head"][2] == v["o1"] or v["head"][2] is None)))
            .parameter("tail", paratuples, lambda v: [v["ls"][1:]])
            .suffix(
                    ((Constructor("DAG_component",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("head")))
                      & Constructor("ID"))
                    **
                    (Constructor("DAG_parallel",
                                       Constructor("input", Var("i2"))
                                       & Constructor("output", Var("o2"))
                                       & Constructor("structure", Var("tail")))
                     & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     **
                     (Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                      & Constructor("non_ID") & Constructor("last", Constructor("ID")))
                     )
                    &
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("structure", Var("tail")))
                      & Constructor("ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("structure", Var("ls"))
                                  & Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    &
                    ((Constructor("DAG_component",
                                  Constructor("input", Var("i1"))
                                  & Constructor("output", Var("o1"))
                                  & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i2"))
                                  & Constructor("output", Var("o2"))
                                  & Constructor("structure", Var("tail")))
                      & Constructor("non_ID"))
                     **
                     (Constructor("DAG_parallel",
                                  Constructor("input", Var("i"))
                                  & Constructor("input", Literal(None))
                                  & Constructor("output", Var("o"))
                                  & Constructor("output", Literal(None))
                                  & Constructor("structure", Var("ls"))
                                  & Constructor("structure", Literal(None))
                                  )
                      & Constructor("non_ID") & Constructor("last", Constructor("non_ID")))
                     )
                    ),

            "before_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) == 1)
            .parameter("ls1", paratuples, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["ls1"] is None or
                                            (
                                                    (
                                                        v["i"] == sum([t[1] for t in v["ls1"]])
                                                        if None not in [t for t in v["ls1"]]
                                                           and None not in [t[1] for t in v["ls1"]]
                                                        else v["i"] > sum([t[1] for t in v["ls1"]
                                                                           if t is not None
                                                                           and t[1] is not None])
                                                    )
                                                    and
                                                    (
                                                        v["o"] == sum([t[2] for t in v["ls1"]])
                                                        if None not in [t for t in v["ls1"]]
                                                           and None not in [t[2] for t in v["ls1"]]
                                                        else v["o"] > sum([t[2] for t in v["ls1"]
                                                                           if t is not None
                                                                           and t[2] is not None]))
                                            )
                                  )
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("ls1"))) & Constructor("non_ID"))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request"))
                                )),

            "before_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) > 1)
            .parameter("head", paratuples, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["head"] is None or
                                            (
                                                (v["i"] == sum([t[1] for t in v["head"]])
                                                 if None not in [t for t in v["head"]]
                                                    and None not in [t[1] for t in v["head"]]
                                                 else v["i"] > sum([t[1] for t in v["head"]
                                                                    if t is not None and t[1] is not None]))
                                                and (v["j"] == sum([t[2] for t in v["head"]])
                                                     if None not in [t for t in v["head"]]
                                                        and None not in [t[2] for t in v["head"]]
                                                     else v["j"] > sum([t[2] for t in v["head"]
                                                                        if t is not None and t[2] is not None]))
                                            )
                                  )
            .parameter("tail", paratupletuples, lambda v: [v["ls"][1:]])
            .parameter_constraint(lambda v: v["tail"] is None or
                                            (
                                                    (len(v["tail"]) > 0) and
                                                    (
                                                            v["tail"][0] is None or
                                                            (
                                                                v["j"] == sum([t[1] for t in v["tail"][0]])
                                                                if None not in [t for t in v["tail"][0]]
                                                                   and None not in [t[1] for t in v["tail"][0]]
                                                                else v["j"] > sum([t[1] for t in v["tail"][0]
                                                                                   if t is not None
                                                                                   and t[1] is not None])
                                                             )
                                                    ) and
                                                    (
                                                            v["tail"][-1] is None or
                                                            (v["o"] == sum([t[2] for t in v["tail"][-1]])
                                                             if None not in [t for t in v["tail"][-1]]
                                                                and None not in [t[2] for t in v["tail"][-1]]
                                                             else v["o"] > sum([t[2] for t in v["tail"][-1]
                                                                                if t is not None
                                                                                and t[2] is not None]))
                                                    )
                                            )
                                  )
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("j"))
                                       & Constructor("structure", Var("head"))) & Constructor("non_ID"))
            .argument("y", Constructor("DAG",
                                       Constructor("input", Var("j"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("tail"))))
            .constraint(lambda v: self.swaplaw1(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw2(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw3(v["x"], v["y"]))
            .constraint(lambda v: self.swaplaw4(v["x"], v["y"]))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request")))),
        }

    # Interpretations of terms are algebras in my language

    def pretty_term_algebra(self):
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: f"edges({io})"),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: f"swap({io}, {n}, {m})"),

            "node": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"node({l}, {i}, {o})"),

            "beside_singleton": (lambda i, o, ls, para, x: f"{x}"),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: f"{x} || {y}"),

            "before_singleton": (lambda i, o, r, ls, ls1, x: f"{x}"),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: f"{x} ; {y}"),
        }

    def edgelist_algebra(self):
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs, {})),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "node": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: ([(x,l + str(id)) for x in inputs],  [l + str(id) for _ in range(0,o)], {l + str(id) : id})),

            "beside_singleton": (lambda i, o, ls, para, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id, inputs:
                (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
                 x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
                 x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x, i)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (lambda id, inputs:
                                                                      (y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
                                                                       y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
                                                                       y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]),
                                                                      i)),
        }

    def edgelist_algebra_detailed(self):
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs, {})),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "node": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: ([(x,l + str(id)) for x in inputs],  [str((l, i, o)) + str(id) for _ in range(0,o)], {str((l, i, o)) + str(id) : id})),

            "beside_singleton": (lambda i, o, ls, para, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id, inputs:
                (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
                 x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
                 x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x, i)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (lambda id, inputs:
                                                                      (y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
                                                                       y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
                                                                       y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]),
                                                                      i)),
        }

    def structure_algebra(self):
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: (("swap", 0, None), None, None)),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: (("swap", None, None), None, None)),

            "node": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, None, None)),

            "beside_singleton": (lambda i, o, ls, para, x: (x,)),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: (x,) + y),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x,)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (x,) + y),
        }

if __name__ == "__main__":
    repo = Labeled_DAMG_Repository(labels=["Conv1D", "LinearLayer", "Maxpool1D", "Upsample"], dimensions=range(1, 6))

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    target101 = Constructor("DAG",
                            Constructor("input", Literal(1))
                            & Constructor("output", Literal(1))
                            & Constructor("structure", Literal(
                                ((None,), (None, None), (None, None), (None,), (None,))
                                #((None,), (None, None), (None,))
                            )))

    edge = (("swap", 0, 1), 1, 1)

    target102 = Constructor("DAG",
                            Constructor("input", Literal(1))
                            & Constructor("output", Literal(1))
                            & Constructor("structure", Literal(
                                ((("Conv1D", 1, 2),), (edge, ("Maxpool1D", 1, 1),), (edge, ("Upsample", 1, 1),), (("Conv1D", 2, 1),), (("LinearLayer", 1, 1),))
                            )))

    target_u_net_like = Constructor("DAG",
                                    Constructor("input", Literal(1))
                                    & Constructor("output", Literal(1))
                                    & Constructor("structure", Literal(
                                        ((("Conv1D", 1, 2),), (edge, ("Maxpool1D", 1, 1),),
                                         (edge, ("Conv1D", 1, 2),), (edge, edge, ("Maxpool1D", 1, 1),),
                                         (edge, edge, ("Conv1D", 1, 1),), (edge, edge, ("Upsample", 1, 1),),
                                         (edge, ("Conv1D", 2, 1),), (edge, ("Upsample", 1, 1),),
                                         (("Conv1D", 2, 1),),
                                         (("Conv1D", 1, 1),),
                                         (("LinearLayer", 1, 1),),
                                         )
                                    )))

    target_u_net_like_general = Constructor("DAG",
                                    Constructor("input", Literal(1))
                                    & Constructor("output", Literal(1))
                                    & Constructor("structure", Literal(
                                        (None,
                                         (edge, ("Maxpool1D", None, None),),
                                         (edge, None,),
                                         (edge, edge, ("Maxpool1D", None, None),),
                                         (edge, edge, None,),
                                         (edge, edge, ("Upsample", None, None),),
                                         (edge, None,), (edge, ("Upsample", 1, 1),),
                                         None,
                                         None,
                                         (("LinearLayer", 1, 1),),
                                         )
                                    )))

    #search_space0 = synthesizer.construct_search_space(target102).prune()
    #term = list(search_space0.enumerate_trees(target102, 2))[0]

    target = target101
    search_space = synthesizer.construct_search_space(target).prune()
    """"
    test = search_space.as_tuples()
    test = filter(lambda x: "DAG_component" in str(x[0]), test)

    for nt, rule in test:
        if "DAG_component" in str(nt):
            for subrule in rule:
                if "node" in str(subrule.terminal):
                    print(f"{nt} -> {subrule.terminal}{tuple(str(arg.name) for arg in subrule.arguments)}")
    """
    term = list(search_space.enumerate_trees(target, 2))[0]

    terms = list(search_space.enumerate_trees(target, 10))

    def to_grakel_graph(t):
        p = t.interpret(repo.edgelist_algebra())
        if not isinstance(p, tuple):
            print(t.interpret(repo.pretty_term_algebra()))
            raise ValueError("edgelist interpretation did not return a tuple")
        f, inputs = p
        edgelist, to_outputs, pos_A = f((-3.8, -3.8), ["input" for _ in range(0, inputs)])
        edgelist = edgelist + [(o, "output") for o in to_outputs]

        #pos_A = pos_A | {"input": (-5.5, -3.8), "output": (max([x for x, y in pos_A.values()]) + 2.5, -3.8)}

        G = nx.MultiDiGraph()
        G.add_edges_from(edgelist)

        relabel = {n: ("Conv1D" if "Conv1D" in n
                       else "LinearLayer" if "LinearLayer" in n
                       else "Maxpool1D" if "Maxpool1D" in n
                       else "Upsample" if "Upsample" in n
                       else n)
                   for n in G.nodes()}

        for n in G.nodes():
            G.nodes[n]['label'] = relabel[n]

        gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='label')

        return gk_graph

    def to_grakel_graph_detailed(t):
        f, inputs = t.interpret(repo.edgelist_algebra_detailed())
        edgelist, to_outputs, pos_A = f((-3.8, -3.8), ["input" for _ in range(0, inputs)])
        edgelist = edgelist + [(o, "output") for o in to_outputs]

        #pos_A = pos_A | {"input": (-5.5, -3.8), "output": (max([x for x, y in pos_A.values()]) + 2.5, -3.8)}

        G = nx.MultiDiGraph()
        G.add_edges_from(edgelist)

        relabel = {n: (n[:16] if "Conv1D" in n
                       else n[:21] if "LinearLayer" in n
                       else n[:19] if "Maxpool1D" in n
                       else n[:18] if "Upsample" in n
                       else n)
                   for n in G.nodes()}

        for n in G.nodes():
            G.nodes[n]['label'] = relabel[n]

        gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='label')

        return gk_graph

    kernel0 = WeisfeilerLehmanKernel(to_grakel_graph=to_grakel_graph)

    kernel1 = WeisfeilerLehmanKernel(to_grakel_graph=to_grakel_graph_detailed)

    kernel2 = WeisfeilerLehmanKernel()

    def fit0(t):
        return kernel0._f(term, t)

    def fit1(t):
        return kernel1._f(term, t)

    def fit2(t):
        return kernel2._f(term, t)

    """
    for t in terms:
        print(t.interpret(repo.pretty_term_algebra()))
        print(fit0(t))
        print(fit1(t))
        print("-----")
    print(term.interpret(repo.pretty_term_algebra()))


    evo_alg = TournamentSelection(search_space, target, fit0, population_size=100, crossover_rate=0.85,
                                      mutation_rate=0.35, generation_limit=30, tournament_size=10,
                                      greater_is_better=True, enforce_diversity=False, elitism=1)

    print("starting evolutionary search")
    result = evo_alg.optimize()
    print("finished evolutionary search")

    print(term.interpret(repo.pretty_term_algebra()))
    print(result.interpret(repo.pretty_term_algebra()))

    print(kernel0._f(term, result))

    if kernel0._f(term, result) == 1:
        structure = result.interpret(repo.structure_algebra())
        next_target = Constructor("DAG",
                            Constructor("input", Literal(1))
                            & Constructor("output", Literal(1))
                            & Constructor("structure", Literal(
                                structure
                            )))
        print(target)
        print(next_target)
        next_search_space = synthesizer.construct_search_space(next_target).prune()
        next_evo_alg = TournamentSelection(next_search_space, next_target, fit1, population_size=100, crossover_rate=0.85,
                                      mutation_rate=0.4, generation_limit=50, tournament_size=10,
                                      greater_is_better=True, enforce_diversity=False, elitism=1)
        print("starting evolutionary search on detailed kernel")
        next_result = next_evo_alg.optimize()
        print("finished evolutionary search on detailed kernel")
        print(term.interpret(repo.pretty_term_algebra()))
        print(next_result.interpret(repo.pretty_term_algebra()))
        print(kernel1._f(term, next_result))
    """

    #"""
    x_list = []
    y_list = []

    x0 = list(search_space.sample(10, target))

    for tree in x0:
        x_list.append(tree)
        y_list.append(fit1(tree))

    xp = np.array(x_list)
    yp = np.array(y_list)

    alpha = 1e-10

    model = GaussianProcessRegressor(kernel=kernel1,
                                     alpha=alpha,
                                     # n_restarts_optimizer=10,
                                     optimizer=None,  # we currently need this, to prevent derivation of the kernel
                                     normalize_y=False)

    print("start fitting")

    model.fit(xp, yp)

    print("finished fitting")

    acquisition_function = SimplifiedExpectedImprovement(model, True)

    acquisition_function2 = ExpectedImprovement(model, True)

    print(yp)
    print(model.y_train_)

    print("acquisition values on initial data:")
    for x in xp:
        print(x in model.X_train_)
        print(x.interpret(repo.pretty_term_algebra()))
        print(fit1(x))
        print(acquisition_function(x))
        print(acquisition_function2(x))
        print("-----")

    test = list(search_space.sample(10, target))
    test2 = list(search_space.enumerate_trees(target, 10))

    print("acquisition values on test data:")
    for x in test + test2:
        print(x.interpret(repo.pretty_term_algebra()))
        print(fit1(x))
        print(acquisition_function(x))
        print(acquisition_function2(x))
        print("-----")

    print(term.interpret(repo.pretty_term_algebra()))

    evo_alg = TournamentSelection(search_space, target, acquisition_function2, population_size=100, crossover_rate=0.85,
                                  mutation_rate=0.3, generation_limit=40, tournament_size=10, greater_is_better=True,
                                  enforce_diversity=False, elitism=1)

    print("starting evolutionary search")
    result = evo_alg.optimize()
    print("finished evolutionary search")

    print(term.interpret(repo.pretty_term_algebra()))
    print(result.interpret(repo.pretty_term_algebra()))

    print(kernel1._f(term, result))
    print(acquisition_function2(result))
    #"""


    """
    bo = BayesianOptimization(search_space, target, kernel=kernel1, population_size=100, crossover_rate=0.85,
                              mutation_rate=0.35, generation_limit=10, tournament_size=10,
                              enforce_diversity=False, elitism=1)
    print("starting bayesian optimisation")
    tree_bo, X, Y = bo.bayesian_optimisation(3, fit1, greater_is_better=True, n_pre_samples=10)
    print("finished bayesian optimisation")
    print(tree_bo.interpret(repo.pretty_term_algebra()))
    print(kernel1._f(term, tree_bo))
    for x, y in zip(X, Y):
        print(x.interpret(repo.pretty_term_algebra()), y)
    print("-----")
    print(term.interpret(repo.pretty_term_algebra()))
    """

    """
    for t in terms:
        print(t.interpret(repo.pretty_term_algebra()))
        f, inputs = t.interpret(repo.edgelist_algebra())
        edgelist, to_outputs, pos_A = f((-3.8, -3.8), ["input" for _ in range(0, inputs)])
        edgelist = edgelist + [(o, "output") for o in to_outputs]

        pos_A = pos_A | {"input": (-5.5, -3.8), "output": (max([x for x, y in pos_A.values()]) + 2.5, -3.8)}

        G = nx.MultiDiGraph()
        G.add_edges_from(edgelist)

        relabel = {n: ("Conv1D" if "Conv1D" in n
                       else "LinearLayer" if "LinearLayer" in n
                       else "Maxpool1D" if "Maxpool1D" in n
                       else "Upsample" if "Upsample" in n
                       else n)
                   for n in G.nodes()}

        for n in G.nodes():
            G.nodes[n]['symbol'] = relabel[n]

        gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

        for g in gk_graph:
            for x in g:
                print(x)




        connectionstyle = [f"arc3,rad={r}" for r in accumulate([0.3] * 4)]

        plt.figure(figsize=(25, 25))

        pos_G = nx.bfs_layout(G, "input")
        node_size = 3000
        nx.draw_networkx_nodes(G, pos_A, node_size=node_size, node_color='lightblue', alpha=0.5, margins=0.05)
        nx.draw_networkx_labels(G, pos_A, labels=relabel, font_size=6, font_weight="bold")
        nx.draw_networkx_edges(G, pos_A, edge_color="black", connectionstyle=connectionstyle, node_size=node_size,
                               width=2)
        plt.figtext(0.01, 0.02, t.interpret(repo.pretty_term_algebra()), fontsize=14)

        #plt.show()
    """
