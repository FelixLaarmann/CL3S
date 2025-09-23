from typing import Iterable, Any

from cl3s import DSL, Constructor, Literal, Type, Var, SearchSpaceSynthesizer, DerivationTree
from collections.abc import Container


class DAMGRepository:
    """
    A repository for directed acyclic graphs,
    following "An initial algebra approach to directed acyclic graphs" from Jeremy Gibbons.
    The following algebraic laws are defined on DAGs (if we skip arguments,
    the variables are only the term arguments and the literals that are independent of the term arguments):

    beside(x, beside(y,z)) = beside(beside(x,y),z)  (associativity of beside)


    before(x, before(y,z)) = before(before(x,y),z)  (associativity of before)


    beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p')))
    =                                                                               (abiding law)
    before(m+m', n+r, p+p', beside(w(m,n),y(m',r)), beside(x(n,p),z(r,p')))


    swap(n, n, 0) and swap(n, 0, n) are neutral elements and therefore make no difference.

                                                                                                (swap simplification laws)

    before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
    =
    swap(m + n + p, m, n+p)


    before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
    =                                                                                 (swap law)
    beside(y(m,q),x(n,p))


    before(swap(m+n, m, n), swap(n+m, n, m)) = copy(m+n, edge())                  (simplified swap law)


    before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
    =                                                                                            (derived simplification law)
    swap(m + n + p, m+n, p)

    before(edge(), x)
    =                                                                    (left neutrality of edge for before)
    x

    before(x, edge())
    =                                                                     (right neutrality of edge for before)
    x


    These laws will be interpreted as directed equalities, such that they correspond to the following
    term rewriting system:

    beside(beside(x,y),z)
    ->
    beside(x, beside(y,z))

    before(before(x,y),z)
    ->
    before(x, before(y,z))

    beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p')))
    ->
    before(m+m', n+r, p+p', beside(w(m,n),y(m',r)), beside(x(n,p),z(r,p')))

    before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
    ->
    swap(m + n + p, m, n+p)

    before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
    ->
    beside(y(m,q),x(n,p))

    before(swap(m+n, m, n), swap(n+m, n, m))
    ->
    copy(m+n, edge())

    before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
    ->
    swap(m + n + p, m+n, p)

    before(edge(), x)
    ->
    x

    before(x, edge())
    ->
    x

    additionally any sequence of beside applied to the same term will be rewritten to copy this term to the length of this sequence:

    beside(x, beside(x,y))
    -> beside(copy(2, x), y)

    beside(copy(n, x), beside(copy(m, x), y))
    ->
    beside(copy(n+m, x), y)

    beside(x, x)
    ->
    copy(2, x)

    beside(copy(n, x), x)
    ->
    copy(n+1, x)

    beside(x, copy(n, x))
    ->
    copy(n+1, x)


    From our FSCD-paper "Restricting tree grammars with term rewriting systems" we know,
    that it should be sufficient to define a term-predicate that forbids all left-hand sides of the rules,
    to describe the set of combinatory terms, that are normal forms of the term rewriting system.
    """

    def __init__(self, dimension_upper_bound: int, dimension_lower_bound: int = 1):
        self.dimension_upper_bound: int = dimension_upper_bound
        self.dimension_lower_bound: int = dimension_lower_bound
        self.dimension: Iterable[int] = range(self.dimension_lower_bound, self.dimension_upper_bound + 1)

    # to allow infinite literal groups, we need to define a subclass of Contains for that group
    class Nat(Container):
        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    def beside_constraints(self, fst_tree: DerivationTree[Any, str, Any], snd_tree: DerivationTree[Any, str, Any]) -> bool:
        """
        :param fst_tree:
        :param snd_tree:
        :return: True, if beside(fst_tree, snd_tree) is in normal form, False otherwise.

        Checks for the left-hand sides of the term rewriting rules, that have beside as root, namely:
        - associativity of beside: beside(beside(x,y),z)
        - abiding law: beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p')))
        - enforce copy: beside(x, x)
        - enforce copy: beside(x, beside(x,y))
        - enforce copy: beside(copy(n, x), beside(copy(m, x), y))
        TODO:
        - enforce copy: beside(x, copy(n, x))
        - enforce copy: beside(copy(n, x), x)
        """
        left = fst_tree.root
        right = snd_tree.root
        # enforce right associativity of beside
        if left == "beside":
            return False
        # abiding law
        if left == "before" and right == "before":
            return False
        # enforce copy
        if fst_tree == snd_tree:  # beside(x, x)
            return False
        if right == "beside": # TODO: this seems buggy, since graphs with no vertices, mapping inputs directly to outputs aren't in normal form
            snd_tree_children = [t for t in snd_tree.children if not t.is_literal]
            right_left_tree = snd_tree_children[0] if len(snd_tree_children) > 0 else None
            if right_left_tree is None:
                return True
            if fst_tree == right_left_tree:
                return False
            right_left = right_left_tree.root
            if left == "copy" and right_left == "copy":
                fst_tree_children = [t for t in fst_tree.children if not t.is_literal]
                right_left_children = [t for t in right_left_tree.children if not t.is_literal]
                if len(fst_tree_children) != 1 or len(right_left_children) != 1:
                    raise ValueError("copy must have exactly one non-literal argument")
                if fst_tree_children[0] == right_left_children[0]:
                    return False
        return True

    def before_constraints(self, fst_tree: DerivationTree[Any, str, Any], snd_tree: DerivationTree[Any, str, Any]) -> bool:
        """
        :param fst_tree:
        :param snd_tree:
        :return: True, if before(fst_tree, snd_tree) is in normal form, False otherwise.

        Checks for the left-hand sides of the term rewriting rules, that have before as root, namely:
        - associativity of before: before(before(x,y),z)
        - swap simplification law: before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
        - swap law: before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
        - simplified swap law: before(swap(m+n, m, n), swap(n+m, n, m))
        - derived simplification law: before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
        - left neutrality of edge for before: before(edge(), x)
        - right neutrality of edge for before: before(x, edge())
        """
        left = fst_tree.root
        right = snd_tree.root
        # enforce right associativity of before
        if left == "before":
            return False

        # neutrality of edge for before
        if left == "edge" or right == "edge":
            return False

        # simplified swap law
        if left == "swap" and right == "swap":
            if len(fst_tree.children) != 4 or len(snd_tree.children) != 4:
                raise ValueError("swap must have exactly four (literal) arguments")
            if fst_tree.children[1] == snd_tree.children[2] and fst_tree.children[2] == snd_tree.children[1]:
                return False

        # swap law
        if left == "swap" and right == "before":
            right_children = [t for t in snd_tree.children if not t.is_literal]
            if len(right_children) != 2:
                raise ValueError("before must have exactly two (non-literal) arguments")
            right_left = right_children[0]
            right_right = right_children[1]
            if right_left.root == "beside" and right_right.root == "swap":
                if len(fst_tree.children) != 4 or len(snd_tree.children) != 4:
                    raise ValueError("swap must have exactly four (literal) arguments")
                swap_m = fst_tree.children[1]
                swap_n = fst_tree.children[2]
                if len(right_left.children) != 14:
                    raise ValueError("beside must have exactly 14 (overall) arguments")
                beside_n = right_left.children[0]
                beside_p = right_left.children[1]
                beside_m = right_left.children[4]
                beside_q = right_left.children[5]
                if len(right_right.children) != 4:
                    raise ValueError("swap must have exactly four (literal) arguments")
                swap_p = right_right.children[1]
                swap_q = right_right.children[2]
                if swap_m == beside_m and swap_n == beside_n and swap_p == beside_p and swap_q == beside_q:
                    return False

        if left == "beside" and right == "beside":
            fst_tree_children = [t for t in fst_tree.children if not t.is_literal]
            snd_tree_children = [t for t in snd_tree.children if not t.is_literal]
            if len(fst_tree_children) != 2 or len(snd_tree_children) != 2:
                raise ValueError("beside must have exactly two (non-literal) arguments")
            left_left = fst_tree_children[0]
            left_right = fst_tree_children[1]
            right_left = snd_tree_children[0]
            right_right = snd_tree_children[1]
            # swap simplification law
            if left_left.root == "swap" and left_right.root == "copy" and right_left.root == "copy" and right_right.root == "swap":
                if len(left_right.children) != 10 or len(right_left.children) != 10:
                    raise ValueError("copy must have exactly 10 (overall arguments")
                if left_right.children[9].root == "edge" and right_left.children[9].root == "edge":
                    if len(left_left.children) != 4 or len(right_right.children) != 4:
                        raise ValueError("swap must have exactly four (literal) arguments")
                    left_m = left_left.children[1]
                    left_n = left_left.children[2]
                    left_p = left_right.children[2]
                    right_m = right_right.children[1]
                    right_n = right_left.children[2]
                    right_p = right_right.children[2]
                    if left_m == right_m and left_n == right_n and left_p == right_p:
                        return False
            # derived simplification law
            if left_left.root == "copy" and left_right.root == "swap" and right_left.root == "swap" and right_right.root == "copy":
                if len(left_left.children) != 10 or len(right_right.children) != 10:
                    raise ValueError("copy must have exactly 10 (overall) arguments")
                if left_left.children[9].root == "edge" and right_right.children[9].root == "edge":
                    if len(left_right.children) != 4 or len(right_left.children) != 4:
                        raise ValueError("swap must have exactly four (literal) arguments")
                    left_m = left_left.children[2]
                    left_n = left_right.children[1]
                    left_p = left_right.children[2]
                    right_m = right_left.children[1]
                    right_n = right_right.children[2]
                    right_p = right_left.children[2]
                    if left_m == right_m and left_n == right_n and left_p == right_p:
                        return False
        return True

    # Our Delta will contain Booleans as the two elementary set and natural numbers as an infinite set (good for indexing).
    def base_delta(self) -> dict[str, list[Any]]:
        return {"nat": self.Nat(),
                "bool": [True, False]}

    def delta(self) -> dict[str, Any]:
        return self.base_delta() | {
            "dimension": self.dimension,
        }

    def gamma(self):
        return {
            "edge": Constructor("graph",
                            Constructor("input", Literal(1, "dimension")) &
                            Constructor("output", Literal(1, "dimension")) &
                            Constructor("edge_size", Literal(1, "nat")) &
                            Constructor("vertex_size", Literal(0, "nat"))),
            "vertex": DSL()
            .parameter("m", "dimension")
            .parameter("n", "dimension",
                       lambda v: range(max(1, self.dimension_lower_bound), self.dimension_upper_bound + 1)
                       if v["m"] == 0 else range(self.dimension_lower_bound, self.dimension_upper_bound + 1))
            .parameter("es", "nat", lambda v: [v["m"] + v["n"]])
            .suffix(Constructor("graph",
                            Constructor("input", Var("m")) &
                            Constructor("output", Var("n")) &
                            Constructor("edge_size", Var("es")) &
                            Constructor("vertex_size", Literal(1, "nat")))),
            "beside": DSL() # only parallel composition of graphs with in and out > 0 is allowed, otherwise disconnected graphs would be possible TODO: check paper
            .parameter("m", "dimension", lambda v: range(max(1, self.dimension_lower_bound), self.dimension_upper_bound + 1))
            .parameter("n", "dimension", lambda v: range(max(1, self.dimension_lower_bound), self.dimension_upper_bound + 1))
            .parameter("i", "dimension", lambda v: range(max(1, self.dimension_lower_bound), self.dimension_upper_bound + 1))
            .parameter("o", "dimension", lambda v: range(max(1, self.dimension_lower_bound), self.dimension_upper_bound + 1))
            .parameter("p", "dimension", lambda v: [v["i"] - v["m"]])
            .parameter("q", "dimension", lambda v: [v["o"] - v["n"]])
            .parameter("es3", "nat")
            .parameter("es1", "nat", lambda v: range(1, v["es3"] + 1))
            .parameter("es2", "nat", lambda v: [v["es3"] - v["es1"]])
            .parameter("vs3", "nat")
            .parameter("vs1", "nat", lambda v: range(0, v["vs3"] + 1))
            .parameter("vs2", "nat", lambda v: [v["vs3"] - v["vs1"]])
            .argument("x", Constructor("graph",
                                  Constructor("input", Var("m")) &
                                  Constructor("output", Var("n"))&
                                  Constructor("edge_size", Var("es1")) &
                                  Constructor("vertex_size", Var("vs1"))))
            #.constraint(lambda v: v["x"].root != "beside") # (enforce right associativity of beside)
            .argument("y", Constructor("graph",
                                  Constructor("input", Var("p")) &
                                  Constructor("output", Var("q")) &
                                  Constructor("edge_size", Var("es2")) &
                                  Constructor("vertex_size", Var("vs2"))))
            #.constraint(lambda v: v["x"].root != "before" and v["y"].root != "before")  # (abiding law)
            .constraint(lambda v: self.beside_constraints(v["x"], v["y"]))
            .suffix(Constructor("graph",
                            Constructor("input", Var("i")) &
                            Constructor("output", Var("o")) &
                            Constructor("edge_size", Var("es3")) &
                            Constructor("vertex_size", Var("vs3")))),
            "before": DSL()
            .parameter("es3", "nat")
            .parameter("m", "dimension", lambda v: range(self.dimension_lower_bound, min(v["es3"] - 1, self.dimension_upper_bound + 1)))
            .parameter("p", "dimension", lambda v: range(self.dimension_lower_bound, min(v["es3"] - v["m"] + 1, self.dimension_upper_bound + 1)))
            .parameter("n", "dimension",
                       lambda v: range(max(1, self.dimension_lower_bound), v["es3"] - v["m"] - v["p"] + 1))  # n must be at least 1, otherwise its equal to beside?
            .parameter("es1", "nat", lambda v: range(v["m"] + v["n"] - 1, v["es3"] - v["p"] + 1)) # m + n - 1 because of edge_size == 1 of edge-combinator, but sum of i and o is 2
            .parameter("es2", "nat", lambda v: [v["es3"] - v["es1"] + v["n"]])  # TODO: edge size computations are still of, if a graph has edges-constructors in it, since they behave different then vertices!
            .parameter("vs3", "nat")
            .parameter("vs1", "nat", lambda v: range(0, v["vs3"] + 1))
            .parameter("vs2", "nat", lambda v: [v["vs3"] - v["vs1"]])
            .argument("x", Constructor("graph",
                                       Constructor("input", Var("m")) &
                                       Constructor("output", Var("n")) &
                                       Constructor("edge_size", Var("es1")) &
                                       Constructor("vertex_size", Var("vs1"))))
            #.constraint(lambda v: v["x"].root != "before")  # (enforce right associativity of before)
            .argument("y", Constructor("graph",
                                       Constructor("input", Var("n")) &
                                       Constructor("output", Var("p")) &
                                       Constructor("edge_size", Var("es2")) &
                                       Constructor("vertex_size", Var("vs2"))))
            #.constraint(lambda v: v["x"].root != "edge" or v["y"].root != "edge")  # (neutrality of edge for before)
            .constraint(lambda v: self.before_constraints(v["x"], v["y"]))
            .suffix(Constructor("graph",
                            Constructor("input", Var("m")) &
                            Constructor("output", Var("p")) &
                            Constructor("edge_size", Var("es3")) &
                            Constructor("vertex_size", Var("vs3")))),
            "swap": DSL()
            .parameter("io", "dimension")
            .parameter("m", "dimension", lambda v: range(1, v["io"]))  # 0 < m < io, swapping zero connections is neutral
            .parameter("n", "dimension", lambda v: [v["io"] - v["m"]])
            .parameter("s", "nat", lambda v: [v["io"]])  # s is io cast from dimension to nat
            .suffix(Constructor("graph",
                            Constructor("input", Var("io")) &
                            Constructor("output", Var("io")) &
                            Constructor("edge_size", Var("s")) &
                            Constructor("vertex_size", Literal(0, "nat")))),  # Constructor("size", Literal(0, "nat")))),
            "copy": DSL()
            .parameter("i", "dimension")
            .parameter("o", "dimension")
            .parameter("m", "dimension", # copy with m = 1 is just the identity, so we exclude that
                       lambda v: [x for x in range(2, max(v["i"], v["o"]) + 1) if v["i"] % x == 0 and v["o"] % x == 0])  # m must be a common divisor of i and o
            .parameter("p", "dimension", lambda v: [v["i"] // v["m"]])
            .parameter("q", "dimension", lambda v: [v["o"] // v["m"]])
            .parameter("es2", "nat")
            .parameter("es1", "nat", lambda v: [x for x in range(0, v["es2"] + 1) if x * v["m"] == v["es2"]])
            .parameter("vs2", "nat")
            .parameter("vs1", "nat", lambda v: [x for x in range(0, v["vs2"] + 1) if x * v["m"] == v["vs2"]])
            .argument("x", Constructor("graph",
                                  Constructor("input", Var("p")) &
                                  Constructor("output", Var("q")) &
                                  Constructor("edge_size", Var("es1")) &
                                  Constructor("vertex_size", Var("vs1"))))
            .suffix(Constructor("graph",
                            Constructor("input", Var("i")) &
                            Constructor("output", Var("o")) &
                            Constructor("edge_size", Var("es2")) &
                            Constructor("vertex_size", Var("vs2")))),

        }


if __name__ == "__main__":

    repo = DAMGRepository(9, 0)
    target = Constructor("graph",
                    Constructor("input", Literal(4, "dimension")) &
                    Constructor("output", Literal(4, "dimension")) &
                    Constructor("edge_size", Literal(4, "nat")) &
                    Constructor("vertex_size", Literal(0, "nat")))
    synthesizer = SearchSpaceSynthesizer(repo.gamma(), repo.delta(), {})
    search_space = synthesizer.construct_search_space(target).prune()
    trees = search_space.enumerate_trees(target, 100)
    #trees = search_space.sample(10, target)
    for t in trees:
        print(t)
