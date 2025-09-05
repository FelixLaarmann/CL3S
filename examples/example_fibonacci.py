##Fibonacci##
"""
Overall description of this example goes here.
"""

from cl3s import CL3S
from cl3s import DSL
from cl3s import Constructor, Literal, Type, Var
from cl3s import SearchSpaceSynthesizer

import networkx as nx

def fib_zero() -> int:
    """
    The Fibonacci number at index 0.

    :return: The Fibonacci number at index 0.
    """
    return 0


def fib_one() -> int:
    """
    The Fibonacci number at index .

    :return: The Fibonacci number at index 1.
    """
    return 1


def fib_next(_z: int, _y: int, _x: int, f1: int, f2: int) -> int:
    """
    Calculate the Fibonacci number at a given index z using the Fibonacci numbers
    at indices x = z - 2 and y = z - 1.

    :param _z: The index for which the Fibonacci number is calculated.
    :param _y: The index z - 1.
    :param _x: The index z - 2.
    :param f1: The Fibonacci number at index (z - 1).
    :param f2: The Fibonacci number at index (z - 2).
    :return: The Fibonacci number at index z.
    """
    return f1 + f2


def main():
    component_specifications = {
        fib_zero: DSL().suffix(Constructor("fib") & Constructor("at", Literal(0, "int"))),
        fib_one: DSL().suffix(Constructor("fib") & Constructor("at", Literal(1, "int"))),
        fib_next: DSL()
        .parameter("z", "int")
        .parameter("y", "int", lambda vs: [vs["z"] - 1])
        .parameter("x", "int", lambda vs: [vs["z"] - 2])
        .argument("f1", Constructor("fib") & Constructor("at", Var("y")))
        .argument("f2", Constructor("fib") & Constructor("at", Var("x")))
        .suffix(Constructor("fib") & Constructor("at", Var("z"))),
    }

    # range of relevant indices for Fibonacci numbers
    parameter_space = {"int": list(range(20))}

    # CoSy instance with the component specifications and parameter space
    cosy = CL3S(component_specifications, parameter_space)

    # query for Fibonacci numbers at relevant indices
    query: Type = Constructor("fib")

    # solve the query and print the solutions
    for solution in cosy.sample(query, 10):
        print(solution)

    for i in range(5):
        # query for Fibonacci numbers at index i
        query = Constructor("fib") & Constructor("at", Literal(i, "int"))

        # solve the query and print the only solution
        print(i, next(iter(cosy.solve(query))))

    sss = SearchSpaceSynthesizer(component_specifications, parameter_space, {})
    search_space = sss.construct_search_space(query).prune()
    trees = search_space.sample(10, query)
    for tree in trees:
        G, d = tree.to_indexed_nx_digraph()
        PG = nx.nx_pydot.to_pydot(G)
        print(PG)
        print(d)
        # nx.draw(G, with_labels=True)


if __name__ == "__main__":
    main()
