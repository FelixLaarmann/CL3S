from src.cl3s import SpecificationBuilder, Constructor, Literal, Var, SearchSpaceSynthesizer, DerivationTree, DataGroup, Group

class Labeled_DAG_Repository:
    def __init__(self, labels, dimensions):
        self.labels = labels
        self.dimensions = dimensions

    class Para(Group):
        name = "Para"

        def __init__(self, labels, dimensions):
            self.labels = labels
            self.dimensions = dimensions

        def __iter__(self):
            for l in self.labels:
                for i in self.dimensions:
                    for o in self.dimensions:
                        yield (l, i, o)

        def __contains__(self, value):
            return (isinstance(value, tuple) and len(value) == 3 and value[0] in self.labels
                    and value[1] in self.dimensions and value[2] in self.dimensions)

        def unfold_none(self, value):
            result = [()]
            for i, v in enumerate(value):
                old_result = result
                result = []
                if v is not None:
                    for r in old_result:
                        result.append(r + (v,))
                else:
                    for r in old_result:
                        if i == 0:
                            for l in self.labels:
                                result.append(r + (l,))
                        else:
                            for d in self.dimensions:
                                result.append(r + (d,))
            return result

    class ParaTuples(Group):
        name = "ParaTuples"

        def __init__(self, labels, max_length=100):
            self.labels = labels
            self.max_length = max_length

        def iter_len(self, length: int, pred=None):
            if length > self.max_length:
                return
            elif length == 0:
                yield ()
            else:
                for label in self.labels:
                    if pred is None:
                        for suffix in self.iter_len(length - 1):
                            yield (label,) + suffix
                    else:
                        for suffix in pred:
                            yield (label,) + suffix

        def __iter__(self):
            pred = None
            for n in range(self.max_length + 1):
                pred = self.iter_len(n, pred)
                yield from pred # TODO: fix __iter__()!

        def __contains__(self, value):
            return (isinstance(value, tuple) or value is None) and all(True if v is None else v in self.labels for v in value)

        def unfold_none(self, value):
            if value is None:
                return self.__iter__()
            else:
                result = [()]
                for v in value:
                    old_result = result
                    result = []
                    if v is not None:
                        for r in old_result:
                            result.append(r + (v,))
                    else:
                        for r in old_result:
                            for label in self.labels:
                                result.append(r + (label,))
                return result

    class ParaTupleTuples(Group):
        name = "ParaTupleTuples"

        def __init__(self, label_tuples):
            self.label_tuples = label_tuples

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value):
            return isinstance(value, tuple) and all(True if v is None else v in self.label_tuples for v in value)

        def unfold_none(self, value):
            result = [()]
            for v in value:
                old_result = result
                result = []
                if v is not None:
                    for r in old_result:
                        result.append(r + (v,))
                else:
                    for r in old_result:
                        for label in self.label_tuples:
                            result.append(r + (label,))
            return result

    def specification(self):
        labels = DataGroup("labels", self.labels)
        io_labels = self.Para(self.labels, self.dimensions)
        labeltuples = self.ParaTuples(io_labels)
        labeltupletuples = self.ParaTupleTuples(labeltuples)
        dimension = DataGroup("dimension", self.dimensions)
        return {
            "edge": Constructor("DAG", Constructor("input", Literal(1)) & Constructor("output", Literal(1))
                                & Constructor("structure", Literal((((),),)))),
            "node": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("l", labels)
            .parameter("ls", labeltupletuples, lambda v: [(((v["l"], v["i"], v["o"]),),)])
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("output", Var("o"))
                                & Constructor("structure", Var("ls")))),

            "beside": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("i1", dimension)
            .parameter("i2", dimension, lambda v: [v["i"] - v["i1"]])
            .parameter("o", dimension)
            .parameter("o1", dimension)
            .parameter("o2", dimension, lambda v: [v["o"] - v["o1"]])
            .parameter("l", io_labels, lambda v: io_labels.unfold_none((None, v["i1"], v["o1"])))
            .parameter("ls1", labeltupletuples, lambda v: [((v["l"],),), (((),),)] if v["i1"] == v["o1"] == 1 else [((v["l"],),)])
            .parameter("ls", labeltupletuples)
            .parameter("ls3", labeltupletuples, lambda v: labeltupletuples.unfold_none(v["ls"]))
            .parameter_constraint(lambda v: len(v["ls3"]) > 1 and v["ls3"][:1] == v["ls1"])
            .parameter("ls2", labeltupletuples, lambda v: [v["ls3"][1:]])
            .argument("x", Constructor("DAG",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("ls1"))))
            .argument("y", Constructor("DAG",
                                       Constructor("input", Var("i2"))
                                       & Constructor("output", Var("o2"))
                                       & Constructor("structure", Var("ls2"))))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("output", Var("o"))
                                & Constructor("structure", Var("ls")))),

            "before": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("ls", labeltupletuples)
            .parameter_constraint(lambda v: len(v["ls"]) > 0)
            .parameter("ls3", labeltupletuples, lambda v: labeltupletuples.unfold_none(v["ls"]))
            .parameter("ls1", labeltuples, lambda v: [v["ls3"][0]])
            .parameter("ls2", labeltupletuples, lambda v: [v["ls3"][1:]])
            .argument("x", Constructor("DAG",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("j"))
                                       & Constructor("structure", Var("ls1"))))
            .argument("y", Constructor("DAG",
                                       Constructor("input", Var("j"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("ls2"))))
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("output", Var("o"))
                                & Constructor("structure", Var("ls"))))
        }

if __name__ == "__main__":
    repo = Labeled_DAG_Repository(labels=['A', 'B', 'C'], dimensions=range(1,10))
    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    io_labels = repo.Para(['A', 'B', 'C'], range(1, 10))
    labeltuples = repo.ParaTuples(io_labels, 1)
    labeltupletuples = repo.ParaTupleTuples(labeltuples)

    #x = labeltuples.iter_len(1, [()])

    #print(list(x))

    for t in labeltuples:
        print(t)

    #for n in labeltupletuples.unfold_none(((("A", 2, 3), ("B", None, 5)), (("C", 2, 3), None), None)):
    #    print(n)


    target0 = Constructor("DAG",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal((((),),))))

    target1 = Constructor("DAG",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(2))
                          & Constructor("structure", Literal(((("A", 1, 2),),))))

    target2 = Constructor("DAG",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(3))
                          & Constructor("structure", Literal(((("A", 1, 2), ("B", 2, 1),),))))

    target3 = Constructor("DAG",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(((("A", 1, 2), ("B", 2, 1)), (("C", 3, 1)),))))

    target = target2

    search_space = synthesizer.construct_search_space(target)

    terms = search_space.enumerate_trees(target, 10)

    for t in terms:
        print(t)
