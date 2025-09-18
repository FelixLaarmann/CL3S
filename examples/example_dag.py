from typing import Iterable, Any

from cl3s import DSL, Constructor, Literal, Type, Var, SearchSpaceSynthesizer, DerivationTree
from collections.abc import Container


class DAGRepository:
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

    additionally any sequence of beside applied to the same term will be rewritten to copy this term to the length of this sequence:

    beside(x, beside(x,y))
    -> beside(copy(2, x), y)

    beside(copy(n, x), beside(copy(m, x), y))
    ->
    beside(copy(n+m, x), y)


    From our FSCD-paper "Restricting tree grammars with term rewriting systems" we know,
    that it should be sufficient to define a term-predicate that forbids all left-hand sides of the rules,
    to describe the set of combinatory terms, that are normal forms of the term rewriting system.
    """

    def __init__(self, dimension_upper_bound: int, dimension_lower_bound: int = 0):
        self.dimension_upper_bound: int = dimension_upper_bound
        self.dimension_lower_bound: int = dimension_lower_bound
        self.dimension: Iterable[int] = range(self.dimension_lower_bound, self.dimension_upper_bound + 1)

    def recognize_beside_associativity(self, term: DerivationTree[Any, str, Any]) -> bool:
        """
        Recognizes the left-hand side of the beside associativity rewriting rule.
        :param term: The term to check.
        :return: True if the term is a left-hand side of a rule, False otherwise.
        """
        # beside(beside(x,y),z)
        children: list[DerivationTree[Any, str, Any]] = list(term.children)
        non_literal_children: list[DerivationTree[Any, Any, Any]] = [c for c in children if not c.is_literal]
        if not term.root == "beside":
            return all([self.recognize_beside_associativity(t) for t in non_literal_children])
        assert (
                len(non_literal_children) == 2
        )
        left_term = non_literal_children[0]
        if left_term.root == "beside":
            return True
        return all([self.recognize_beside_associativity(t) for t in non_literal_children])

    def recognize_before_associativity(self, term: DerivationTree[Any, str, Any]) -> bool:
        """
        Recognizes the left-hand side of the before associativity rewriting rule.
        :param term: The term to check.
        :return: True if the term is a left-hand side of a rule, False otherwise.
        """
        # before(before(x,y),z)
        children: list[DerivationTree[Any, str, Any]] = list(term.children)
        non_literal_children: list[DerivationTree[Any, str, Any]] = [c for c in children if not c.is_literal]
        if not term.root == "before":
            return all([self.recognize_before_associativity(t) for t in non_literal_children])
        assert (
                len(non_literal_children) == 2
        )
        left_term = non_literal_children[0]
        if left_term.root == "before":
            return True
        return all([self.recognize_before_associativity(t) for t in non_literal_children])

    def recognize_abiding(self, term: DerivationTree[Any, str, Any]) -> bool:
        """
        Recognizes the left-hand side of the abiding rewriting rule.
        :param term: The term to check.
        :return: True if the term is a left-hand side of a rule, False otherwise.
        """
        # beside(before(m,n,p, w(m,n), x(n,p)), before(m',r,p', y(m',r), z(r,p')))
        children: list[DerivationTree[Any, str, Any]] = list(term.children)
        non_literal_children: list[DerivationTree[Any, str, Any]] = [c for c in children if not c.is_literal]
        # TODO
        return False

    def recognize_left_hand_sides(self, term: DerivationTree[Any, str, Any]) -> bool:
        """
        Recognizes the left-hand sides of the rules.
        :param term: The term to check.
        :return: True if the term is a left-hand side of a rule, False otherwise.
        """
        # TODO
        return True

    # to allow infinite literal groups, we need to define a subclass of Contains for that group
    class Nat(Container):
        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    # Our Delta will contain Booleans as the two elementary set and natural numbers as an infinite set (good for indexing).
    def base_delta(self) -> dict[str, list[Any]]:
        return {"nat": self.Nat(),
                "bool": [True, False]}

    def delta(self) -> dict[str, Any]:
        return self.base_delta() | {
            "dimension": self.dimension
        }

    def gamma(self):
        return {
            "edge": Constructor("graph",
                            Constructor("input", Literal(1, "dimension")) &
                            Constructor("output", Literal(1, "dimension")) &
                            Constructor("size", Literal(1, "nat"))),  # Constructor("size", Literal(0, "nat")))),
            "vertex": DSL()
            .parameter("m", "dimension")
            .parameter("n", "dimension")
            .suffix(Constructor("graph",
                            Constructor("input", Var("m")) &
                            Constructor("output", Var("n")) &
                            Constructor("size", Literal(1, "nat")))),
            "beside": DSL()
            .parameter("m", "dimension")
            .parameter("n", "dimension")
            .parameter("i", "dimension")
            .parameter("o", "dimension")
            .parameter("p", "dimension", lambda v: [v["i"] - v["m"]])
            .parameter("q", "dimension", lambda v: [v["o"] - v["n"]])
            .parameter("s3", "nat")
            .parameter("s1", "nat", lambda v: range(0, v["s3"] + 1))
            .parameter("s2", "nat", lambda v: [v["s3"] - v["s1"]])
            .argument("x", Constructor("graph",
                                  Constructor("input", Var("m")) &
                                  Constructor("output", Var("n")) &
                                  Constructor("size", Var("s1"))))
            .argument("y", Constructor("graph",
                                  Constructor("input", Var("p")) &
                                  Constructor("output", Var("q")) &
                                  Constructor("size", Var("s2"))))
            .suffix(Constructor("graph",
                            Constructor("input", Var("i")) &
                            Constructor("output", Var("o")) &
                            Constructor("size", Var("s3")))),
            "before": DSL()
            .parameter("m", "dimension")
            .parameter("n", "dimension")
            .parameter("p", "dimension")
            .parameter("s3", "nat")
            .parameter("s1", "nat", lambda v: range(0, v["s3"] + 1))
            .parameter("s2", "nat", lambda v: [v["s3"] - v["s1"]])
            .argument("x", Constructor("graph",
                                  Constructor("input", Var("m")) &
                                  Constructor("output", Var("n")) &
                                  Constructor("size", Var("s1"))))
            .argument("y", Constructor("graph",
                                  Constructor("input", Var("n")) &
                                  Constructor("output", Var("p")) &
                                  Constructor("size", Var("s2"))))
            .suffix(Constructor("graph",
                            Constructor("input", Var("m")) &
                            Constructor("output", Var("p")) &
                            Constructor("size", Var("s3")))),
            "swap": DSL()
            .parameter("io", "dimension")
            .parameter("m", "dimension", lambda v: range(1, v["io"]))  # 0 < m < io, swapping zero connections is neutral
            .parameter("n", "dimension", lambda v: [v["io"] - v["m"]])
            .suffix(Constructor("graph",
                            Constructor("input", Var("io")) &
                            Constructor("output", Var("io")) &
                            Constructor("size", Literal(1, "nat")))),  # Constructor("size", Literal(0, "nat")))),
            "copy": DSL()
            .parameter("m", "dimension", lambda _: range(1, self.dimension_upper_bound + 1))  # m > 0, otherwise // is bad
            .parameter("i", "dimension")
            .parameter("o", "dimension")
            .parameter("p", "dimension", lambda v: [v["i"] // v["m"]]) # dimensionsberechnung so falsch!
            .parameter("q", "dimension", lambda v: [v["o"] // v["m"]])
            .parameter("s2", "nat")
            .parameter("s1", "nat", lambda v: [v["s2"] // v["m"]])
            .argument("x", Constructor("graph",
                                  Constructor("input", Var("p")) &
                                  Constructor("output", Var("q")) &
                                  Constructor("size", Var("s1"))))
            .suffix(Constructor("graph",
                            Constructor("input", Var("i")) &
                            Constructor("output", Var("o")) &
                            Constructor("size", Var("s2")))),

        }


if __name__ == "__main__":

    repo = DAGRepository(5)
    target = Constructor("graph",
                    Constructor("input", Literal(1, "dimension")) &
                    Constructor("output", Literal(1, "dimension")) &
                    Constructor("size", Literal(10, "nat")))
    synthesizer = SearchSpaceSynthesizer(repo.gamma(), repo.delta(), {})
    search_space = synthesizer.construct_search_space(target).prune()
    trees = search_space.enumerate_trees(target, 100)
    for t in trees:
        print(t)