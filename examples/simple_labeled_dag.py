from src.cl3s import SpecificationBuilder, Constructor, Literal, Var, SearchSpaceSynthesizer, DerivationTree, DataGroup, Group

class Labeled_DAG_Repository:
    def __init__(self, labels, dimensions):
        self.labels = labels
        self.dimensions = dimensions

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
                                    yield (l, i, o)

        def __contains__(self, value):
            return (isinstance(value, tuple) and len(value) == 3 and value[0] in self.labels
                    and value[1] in self.dimensions and value[2] in self.dimensions)

        def unfold_none(self, value):
            result = [()]
            for i, v in enumerate(value):
                old_result = result.copy()
                result = []
                if v is not None:
                    for r in old_result:
                        result.append(r + (v,))
                else:
                    for r in old_result:
                        if i == 0:
                            for l in self.labels:
                                if l is not None:
                                    result.append(r + (l,))
                        else:
                            for d in self.dimensions:
                                if d is not None:
                                    result.append(r + (d,))
            return result

    class ParaTuples(Group):
        name = "ParaTuples"

        def __init__(self, labels, max_length=3):
            self.labels = labels
            self.max_length = max_length

        def __iter__(self):
            result = set()

            for n in range(0, self.max_length + 1):
                if n == 0:
                    result.add(())
                else:
                    old_result = result.copy()
                    for label in self.labels:
                        for suffix in old_result:
                            result.add((label,) + suffix)
            yield from result

        def __contains__(self, value):
            return (isinstance(value, tuple) or value is None) and all(True if v is None else v in self.labels for v in value)

        def unfold_none(self, value):
            if value is None:
                return self.__iter__()
            else:
                result = [()]
                for v in value:
                    old_result = result.copy()
                    result = []
                    if v is not None:
                        for r in old_result:
                            for vv in self.labels.unfold_none(v):
                                result.append(r + (vv,))
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
                old_result = result.copy()
                result = []
                if v is not None:
                    for r in old_result:
                        for vv in self.label_tuples.unfold_none(v):
                            if r == ():
                                result.append(r + (vv,))
                            else:
                                (r_i, r_o) = Labeled_DAG_Repository.compute_dimension_of_tuple_tuple(r)
                                (vv_i, vv_o) = Labeled_DAG_Repository.compute_dimension_of_tuple_tuple((vv,))
                                if r_o == vv_i:
                                    result.append(r + (vv,))
                else:
                    for r in old_result:
                        for vv in self.label_tuples:
                            if r == ():
                                result.append(r + (vv,))
                            else:
                                (r_i, r_o) = Labeled_DAG_Repository.compute_dimension_of_tuple_tuple(r)
                                (vv_i, vv_o) = Labeled_DAG_Repository.compute_dimension_of_tuple_tuple((vv,))
                                if r_o == vv_i:
                                    result.append(r + (vv,))
            return result

    @staticmethod
    def compute_dimension_of_tuple_tuple(para_tuple_tuple):
        input_dim = sum(t[1] for t in para_tuple_tuple[0])
        output_dim = sum(t[2] for t in para_tuple_tuple[-1])
        return input_dim, output_dim



    def specification(self):
        labels = DataGroup("labels", self.labels)
        io_labels = self.Para(self.labels, self.dimensions)
        labeltuples = self.ParaTuples(io_labels)
        labeltupletuples = self.ParaTupleTuples(labeltuples)
        dimension = DataGroup("dimension", self.dimensions)
        return {
            "edge": Constructor("DAG", Constructor("input", Literal(1))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Literal(1))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Literal((((),),)))
                                & Constructor("structure", Literal(((None,),)))),  # TODO: include edges in variance, but respect laws
            "node": SpecificationBuilder()
            .parameter("l", labels)
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", labeltupletuples, lambda v: [(((v["l"], v["i"], v["o"]),),),
                                                          (((v["l"], v["i"], None),),),
                                                          (((v["l"], None, v["o"]),),),
                                                          (((None, v["i"], v["o"]),),),
                                                          (((v["l"], None, None),),),
                                                          (((None, None, v["o"]),),),
                                                          (((None, v["i"], None),),),
                                                          (((None, None, None),),)])
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
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
            .parameter_constraint(lambda v: len(v["ls3"]) > 0 and len(v["ls3"][0]) > 1 and ((v["ls3"][0][0],),) == v["ls1"])
            .parameter("ls2", labeltupletuples, lambda v: [(v["ls3"][0][1:],)])
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
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls")))),

            "before": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("ls", labeltupletuples)
            .parameter_constraint(lambda v: len(v["ls"]) > 1)
            .parameter("ls3", labeltupletuples, lambda v: labeltupletuples.unfold_none(v["ls"]))
            .parameter("ls1", labeltupletuples, lambda v: [v["ls3"][:1]])
            .parameter("ls2", labeltupletuples, lambda v: [v["ls3"][1:]])
            .parameter_constraint(lambda v: self.compute_dimension_of_tuple_tuple(v["ls1"]) == (v["i"], v["j"]) and
                                            self.compute_dimension_of_tuple_tuple(v["ls2"]) == (v["j"], v["o"]))
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
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls")))),

            # TODO: include swap
        }

if __name__ == "__main__":
    repo = Labeled_DAG_Repository(labels=['A', 'B', 'C'], dimensions=range(1,5))
    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})

    io_labels = repo.Para(['A', 'B', 'C'], range(1, 5))
    labeltuples = repo.ParaTuples(io_labels, 3)
    labeltupletuples = repo.ParaTupleTuples(labeltuples)

    #x = labeltuples.iter_len(1, [()])

    #y = labeltuples.iter_len(2, x)

    #print(list(x))

    #print(list(y))

    #for t in io_labels:
    #    print(t)

    for n in labeltupletuples.unfold_none(((("A", 1, 2), ("B", 2, 1),), (("C", 3, 1),), None)):
        print(n)
        print(repo.compute_dimension_of_tuple_tuple(n))


    target0 = Constructor("DAG",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal((((),),))))

    target1 = Constructor("DAG",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(((("C", 3, 1),),))))

    target2 = Constructor("DAG",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(3))
                          & Constructor("structure", Literal(((("A", 1, 2), ("B", 2, 1)),))))

    target3 = Constructor("DAG",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(((("A", 1, 2), ("B", 2, 1),), (("C", 3, 1),),))))

    target4 = Constructor("DAG",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(None))
                          & Constructor("structure", Literal(((("C", None, None),),))))

    target5 = Constructor("DAG",
                          Constructor("input", Literal(None))
                          & Constructor("output", Literal(3))
                          & Constructor("structure", Literal(((("A", None, None), (None, 2, None)),))))

    target6 = Constructor("DAG",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(((("A", 1, 2), ("B", 2, 1),), (("C", 3, 1),), None))))


    target = target6

    print(target)

    search_space = synthesizer.construct_search_space(target)

    terms = search_space.enumerate_trees(target, 10)

    for t in terms:
        print(t)
