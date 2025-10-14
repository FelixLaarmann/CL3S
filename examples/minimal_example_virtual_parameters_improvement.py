from cosy import DSL, Constructor, Literal, Type, Var, Synthesizer
from collections.abc import Container
from typing import Any

class MinimalRepo:
    def __init__(self, parameters: frozenset[Any]):
        self.paras = parameters

        if not None in parameters:
            self.paras = self.paras.union({None})

    class Nat(Container):
        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    class Maybe_Nat(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, int) and value >= 0)

    class Maybe_Para_Tuple(Container):
        def __init__(self, para_choices):
            self.para_choices = para_choices

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.para_choices for v in value)

    def parameters(self) -> dict[str, list[Any]]:
        return {
            "bool": [True, False],
            "nat": self.Nat(),
            "para": self.paras,
            "para_tuple": self.Maybe_Para_Tuple(self.paras),
            "length": self.Maybe_Nat(),
        }

    def specification(self):
        return {
            "Para_func": DSL()
            .parameter("p", "para", lambda v: [x for x in self.paras if x is not None])
            .suffix(Constructor("p_func") & Constructor("para", Var("p"))),
            "Singleton": DSL()
            .parameter("p", "para", lambda v: [x for x in self.paras if x is not None])
            .parameter("ps", "para_tuple", lambda v: [(v["p"],), (None,)])
            .argument("f", Constructor("p_func") & Constructor("para", Var("p")))
            .suffix(Constructor("func_list", Constructor("paras", Var("ps")))
                    & Constructor("func_list", Constructor("length", Literal(1, "length")))
                    & Constructor("func_list", Constructor("length", Literal(None, "length")))
                    ),
            "Cons": DSL()
            .parameter("p", "para", lambda v: [x for x in self.paras if x is not None])
            .parameter("pps", "para_tuple")
            .parameter_constraint(lambda v: len(v["pps"]) > 1 and (v["pps"][0] == v["p"] or v["pps"][0] is None))
            .parameter("ps", "para_tuple", lambda v: [v["pps"][1:]])
            .parameter("l2", "length")
            .parameter("l1", "length", lambda v: [v["l2"] - 1 if v["l2"] is not None else None])
            .parameter_constraint(lambda v: len(v["pps"]) == v["l2"] if v["l2"] is not None else True)
            .argument("f", Constructor("p_func") & Constructor("para", Var("p")))
            .argument("fs", Constructor("func_list", Constructor("paras", Var("ps")))
                      & Constructor("func_list", Constructor("length", Var("l1")))
                      )
            .suffix(Constructor("func_list", Constructor("paras", Var("pps")))
                    & Constructor("func_list", Constructor("length", Var("l2")))
                    ),
        }

    def pretty_term_algebra(self):
        return {
            "Para_func": (lambda p: f"Para_func({p})"),
            "Singleton": (lambda p, ps, f: f"Singleton({f})"),
            "Cons": (lambda p, pps, ps, l2, l1, f, fs: f"Cons({f}, {fs})"),
        }

if __name__ == "__main__":
    repo = MinimalRepo(frozenset({"A", "B", "C"}))

    print(repo.parameters())

    # gewünscht
    target0 = Constructor("func_list", Constructor("length", Literal(3, "length")))
    target1 = Constructor("func_list", Constructor("paras", Literal(("A", "B", "C"), "para_tuple")))

    # funktioniert so, aber nicht gewünscht
    target01 = (Constructor("func_list", Constructor("length", Literal(3, "length")))
                & Constructor("func_list", Constructor("paras", Literal((None, None, None), "para_tuple"))))

    target10 = (Constructor("func_list", Constructor("length", Literal(None, "length")))
                & Constructor("func_list", Constructor("paras", Literal(("A", None, "C"), "para_tuple"))))

    target = target10

    synthesizer = Synthesizer(repo.specification(), repo.parameters(), {})

    search_space = synthesizer.construct_solution_space(target).prune()

    trees = search_space.enumerate_trees(target, 10)

    for t in trees:
        print(t.interpret(repo.pretty_term_algebra()))