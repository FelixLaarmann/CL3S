from cosy import SpecificationBuilder, Constructor, Literal, Var, Synthesizer
from cosy.types import Group

class MinimalExample:
    class ExampleGroup(Group):
        name = "ExampleGroup"

        def __init__(self):
            super().__init__()
            self.values = [0, 1, 2, 3]

        def __iter__(self):
            yield from self.values

        def __contains__(self, item):
            return item is None or item in self.values

    def specification(self):
        group = self.ExampleGroup()

        return {
            "x" : Constructor("a") & Literal(0),
            "y": Constructor("b") & Literal(1),
            "f" : SpecificationBuilder()
            .parameter("n", group)
            .parameter("m", group, lambda v: [v["n"] - 1] if v["n"] is not None else group)
            .parameter_constraint(lambda v: True if print(f' f is {v["n"]}') else True)
            .argument("y", Constructor("a") & Var("m"))
            .suffix(Constructor("a") & Var("n")),
            "g": SpecificationBuilder()
            .parameter("n", group)
            .parameter("m", group, lambda v: [v["n"] - 1] if v["n"] is not None else group)
            .parameter_constraint(lambda v: True if print(f' g is {v["n"]}') else True)
            .suffix(((Constructor("a") & Var("m")) ** (Constructor("a") & Var("n")))),
            "h" : SpecificationBuilder()
            .parameter("n", group)
            .parameter("m", group, lambda v: [v["n"] - 1] if v["n"] is not None else group)
            .parameter_constraint(lambda v: True if print(f' h is {v["n"]}') else True)  #h is never None, because __iter__ is called instead of __contains__
            .suffix(((Constructor("a") & Var("m")) ** (Constructor("a") & Var("n"))) & ((Constructor("b") & Var("m")) ** (Constructor("b") & Var("n")))),
        }


if __name__ == "__main__":
    repo = MinimalExample()
    synthesizer = Synthesizer(repo.specification(), {})

    target0 = Constructor("a") & Literal(None)
    target1 = Constructor("a") & Literal(2)

    target = target0

    solution_space = synthesizer.construct_solution_space(target)
    terms = solution_space.enumerate_trees(target, 10)

    for t in terms:
        print(t)

