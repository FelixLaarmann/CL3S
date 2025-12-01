
from src.cl3s import SpecificationBuilder, Constructor, Literal, Var, SearchSpaceSynthesizer, DerivationTree, DataGroup, Group

from typing import Any

class Labeled_DAG_Repository:
    def __init__(self, labels, dimensions):
        # additionally to labeled nodes, we have (unlabelled) edges, that needs to be handled additionally
        if "swap" in labels:
            raise ValueError("Label 'swap' is reserved and cannot be used as a node label.")
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
                                    yield l, i, o
                                    if i == o:
                                    #if i is not None and o is not None and i == o:
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

    class ParaTupleTuples(Group):
        name = "ParaTupleTuples"

        def __init__(self, para_tuples):
            self.para_tuples = para_tuples

        def __iter__(self):
            return super().__iter__()

        def __contains__(self, value):
            return value is None or (isinstance(value, tuple) and all(True if v is None else v in self.para_tuples for v in value))


    def swaplaw1(self, head: DerivationTree[Any, str, Any], tail: DerivationTree[Any, str, Any]) -> bool:
        """
        before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
        ->
        beside(y(m,q),x(n,p))

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched

        :param t:
        :return:
        """

        left = head.root
        right = tail.root

        if "beside_singleton" in left and "before_cons" in right:
            if len(head.children) != 5 or len(tail.children) != 8:
                raise ValueError("Derivation trees have not the expected shape.")
            left_term = head.children[4]
            right_head = tail.children[6]
            right_tail = tail.children[7]

            left = left_term.root
            right_head_root = right_head.root
            right_tail_root = right_tail.root
            if left == "swap" and "beside_cons" in right_head_root and "before_singleton" in right_tail_root:
                # further checks needed
                if len(left_term.children) != 4 or len(right_head.children) != 11 or len(right_tail.children) != 5:
                    raise ValueError("Derivation trees have not the expected shape.")
                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_term = right_tail.children[4]
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
                        if len(right_swap.children) != 4:
                            raise ValueError("Derivation trees have not the expected shape.")
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False
        return True  # placeholder implementation


    def specification(self):
        labels = DataGroup("para", self.labels)
        para_labels = self.Para(self.labels, self.dimensions)
        paratuples = self.ParaTuples(para_labels, max_length=max(self.dimensions))
        paratupletuples = self.ParaTupleTuples(paratuples)
        dimension = DataGroup("dimension", self.dimensions)
        #dimension_with_zero = DataGroup("dimension_with_zero", list(self.dimensions) + [0])

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
        
        
        (the three laws below may be handled with types, but only if the contains-bug is solved!)
        (one could introduce a "only edges" flag, like ID and non_ID, but that would also require a lot of thinking...)
        
        before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
        ->
        swap(m + n + p, m, n+p)

        before(swap(m+n, m, n), swap(n+m, n, m))
        ->
        copy(m+n, edge())

        before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
        ->
        swap(m + n + p, m+n, p)

        """

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
            .parameter("para", para_labels, lambda v: [(("swap", 0, v["io"]), v["io"], v["io"]),
                                                       (("swap", 0, None), v["io"], v["io"]),
                                                       (("swap", None, v["io"]), v["io"], v["io"]),
                                                       (("swap", None, None), v["io"], v["io"]),
                                                       (("swap", 0, v["io"]), v["io"], None),
                                                       (("swap", 0, None), v["io"], None),
                                                       (("swap", None, v["io"]), v["io"], None),
                                                       (("swap", None, None), v["io"], None),
                                                       (("swap", 0, v["io"]), None, v["io"]),
                                                       (("swap", 0, None), None, v["io"]),
                                                       (("swap", None, v["io"]), None, v["io"]),
                                                       (("swap", None, None), None, v["io"]),
                                                       (("swap", 0, v["io"]), None, None),
                                                       (("swap", 0, None), None, None),
                                                       (("swap", None, v["io"]), None, None),
                                                       (("swap", None, None), None, None), None])
                                                        # (None, None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("ID")
                    ),

            "swap": SpecificationBuilder()
            .parameter("io", dimension)
            .parameter("n", dimension, lambda v: range(1, v["io"]))
            .parameter("m", dimension, lambda v: [v["io"] - v["n"]]) # m > 0
            .parameter("para", para_labels, lambda v: [(("swap", v["n"], v["m"]), v["io"], v["io"]),
                                                       (("swap", v["n"], None), v["io"], v["io"]),
                                                       (("swap", None, v["m"]), v["io"], v["io"]),
                                                       (("swap", None, None), v["io"], v["io"]),
                                                       (("swap", v["n"], v["m"]), v["io"], None),
                                                       (("swap", v["n"], None), v["io"], None),
                                                       (("swap", None, v["m"]), v["io"], None),
                                                       (("swap", None, None), v["io"], None),
                                                       (("swap", v["n"], v["m"]), None, v["io"]),
                                                       (("swap", v["n"], None), None, v["io"]),
                                                       (("swap", None, v["m"]), None, v["io"]),
                                                       (("swap", None, None), None, v["io"]),
                                                       (("swap", v["n"], v["m"]), None, None),
                                                       (("swap", v["n"], None), None, None),
                                                       (("swap", None, v["m"]), None, None),
                                                       (("swap", None, None), None, None), None])
                                                       #(None, None, None)])
            .suffix(Constructor("DAG_component", Constructor("input", Var("io"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("io"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),
            "node": SpecificationBuilder()
            .parameter("l", labels)
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("para", para_labels, lambda v: [(v["l"], v["i"], v["o"]),
                                                       (v["l"], v["i"], None),
                                                       (v["l"], None, v["o"]),
                                                       (None, v["i"], v["o"]),
                                                       (v["l"], None, None),
                                                       (None, None, v["o"]),
                                                       (None, v["i"], None), None])
                                                       #(None, None, None)])
            .suffix(Constructor("DAG_component",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("para"))
                                & Constructor("structure", Literal(None))
                                ) & Constructor("non_ID")
                    ),
            #'''
            "beside_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) == 1)
            .parameter("para", para_labels, lambda v: [v["ls"][0]]) # [l for l in para_labels if l[1] == v["i"] and l[2] == v["o"]] if v["ls"][0] is None else [v["ls"][0]])
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
                  & Constructor("non_ID")
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
                 #''' : Constructor("comment"),
            '''
            "beside_singleton_1": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) == 1)
            .parameter("para", para_labels, lambda v: [v["ls"][
                                                           0]])  # [l for l in para_labels if l[1] == v["i"] and l[2] == v["o"]] if v["ls"][0] is None else [v["ls"][0]])
            .parameter_constraint(lambda v: (len(v["para"]) == 3 and
                                            (v["para"][1] == v["i"] if v["para"][1] is not None else True) and
                                            (v["para"][2] == v["o"] if v["para"][2] is not None else True)
                                             ) if v["para"] is not None else True)
            # .parameter_constraint(lambda v: True if print(v["para"]) else True)
            .argument("x",Constructor("DAG_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("ID"))
            .suffix(Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                  & Constructor("ID")),


            "beside_singleton_2": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratuples)
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) == 1)
            .parameter("para", para_labels, lambda v: [v["ls"][
                                                           0]])  # [l for l in para_labels if l[1] == v["i"] and l[2] == v["o"]] if v["ls"][0] is None else [v["ls"][0]])
            .parameter_constraint(lambda v: (len(v["para"]) == 3 and
                                            (v["para"][1] == v["i"] if v["para"][1] is not None else True) and
                                            (v["para"][2] == v["o"] if v["para"][2] is not None else True)
                                             ) if v["para"] is not None else True)
            # .parameter_constraint(lambda v: True if print(v["para"]) else True)
            .argument("x",Constructor("DAG_component",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("para")))
                     & Constructor("non_ID"))
            .suffix(Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                  & Constructor("non_ID")),
                  ''': Constructor("comment"),

            #'''
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
            .suffix((
                    (Constructor("DAG_component",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("head")))
                     & Constructor("ID"))
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
                      & Constructor("ID"))
                     )
                    &
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
                      & Constructor("non_ID"))
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
                      & Constructor("non_ID"))
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
                      & Constructor("non_ID"))
                     )
                    ),
                    #''': Constructor("comment"),
            '''
            "beside_cons_1": SpecificationBuilder()
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
            .argument("x", Constructor("DAG_component",
                                         Constructor("input", Var("i1"))
                                         & Constructor("output", Var("o1"))
                                         & Constructor("structure", Var("head")))
                             & Constructor("ID"))
            .argument("y", Constructor("DAG_parallel",
                                         Constructor("input", Var("i2"))
                                         & Constructor("output", Var("o2"))
                                         & Constructor("structure", Var("tail")))
                             & Constructor("ID"))
            .suffix(Constructor("DAG_parallel",
                                         Constructor("input", Var("i"))
                                         & Constructor("input", Literal(None))
                                         & Constructor("output", Var("o"))
                                         & Constructor("output", Literal(None))
                                         & Constructor("structure", Var("ls"))
                                         & Constructor("structure", Literal(None))
                                         )
                             & Constructor("ID")),

            "beside_cons_2": SpecificationBuilder()
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
            .argument("x", Constructor("DAG_component",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
            .argument("y", Constructor("DAG_parallel",
                                       Constructor("input", Var("i2"))
                                       & Constructor("output", Var("o2"))
                                       & Constructor("structure", Var("tail")))
                      & Constructor("ID"))
            .suffix(Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                    & Constructor("non_ID")),

            "beside_cons_3": SpecificationBuilder()
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
            .argument("x", Constructor("DAG_component",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("head")))
                      & Constructor("ID"))
            .argument("y", Constructor("DAG_parallel",
                                       Constructor("input", Var("i2"))
                                       & Constructor("output", Var("o2"))
                                       & Constructor("structure", Var("tail")))
                      & Constructor("non_ID"))
            .suffix(Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                    & Constructor("non_ID")),

            "beside_cons_4": SpecificationBuilder()
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
            .argument("x", Constructor("DAG_component",
                                       Constructor("input", Var("i1"))
                                       & Constructor("output", Var("o1"))
                                       & Constructor("structure", Var("head")))
                      & Constructor("non_ID"))
            .argument("y", Constructor("DAG_parallel",
                                       Constructor("input", Var("i2"))
                                       & Constructor("output", Var("o2"))
                                       & Constructor("structure", Var("tail")))
                      & Constructor("non_ID"))
            .suffix(Constructor("DAG_parallel",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls"))
                                & Constructor("structure", Literal(None))
                                )
                    & Constructor("non_ID")),
                    ''': Constructor("comment"),

            "before_singleton": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratupletuples)
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
                                & Constructor("structure", Var("ls"))
                                )),

            "before_cons": SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("ls", paratupletuples)
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) > 1)
            .parameter("head", paratuples, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: (
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
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("ls")))),
        }

if __name__ == "__main__":
    repo = Labeled_DAG_Repository(labels=["A", "B"], dimensions=range(1,4))

    io_labels = repo.Para(["A", "B"], range(1, 4))

    #for l in io_labels:
    #    print(l)

    paratuples = repo.ParaTuples(io_labels, max_length=3)



    """    
    print(None in io_labels)
    print((None,) in paratuples)

    test = (None,)
    print(test in paratuples)
    print(test[0] in io_labels)

    para_labels = io_labels

    v = {"i": 3, "o": 3, "ls": (None,), "para": ("A", 3, None)}

    print([l for l in para_labels if l[1] == v["i"] and l[2] == v["o"]] if v["ls"][0] is None else [v["ls"][0]])

    print(v["para"] in para_labels)

    print(len(v["para"]) == 3 and
          (v["para"][1] == v["i"] if v["para"][1] is not None else True) and
          (v["para"][2] == v["o"] if v["para"][2] is not None else True)
          )

    for l in para_labels:
        print(l)
    """

    synthesizer = SearchSpaceSynthesizer(repo.specification(), {})


    target0 = Constructor("DAG_component",
                          Constructor("input", Literal(1))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(None)))

    target1 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(None)))

    target2 = Constructor("DAG_component",
                          Constructor("input", Literal(None))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(None)))

    target3 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(None))
                          & Constructor("structure", Literal(None)))

    target4 = Constructor("DAG_component",
                          Constructor("input", Literal(None))
                          & Constructor("output", Literal(None))
                          & Constructor("structure", Literal(None)))

    target5 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(("A", 3, 1))))

    target6 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(("A", 3, None))))

    target7 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(3))
                          & Constructor("structure", Literal(("A", None, 3))))

    target8 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal((None, 3, 1))))

    target9 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(("A", None, None))))

    target10 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal((None, None, 1))))

    target11 = Constructor("DAG_component",
                          Constructor("input", Literal(None))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(("A", 3, 1))))

    target12 = Constructor("DAG_component",
                          Constructor("input", Literal(None))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(("A", None, 1))))

    target13 = Constructor("DAG_component",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(1))
                           & Constructor("structure", Literal(("A", None, None))))

    target14 = Constructor("DAG_component",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(1))
                           & Constructor("structure", Literal((None, 3, None))))

    target15 = Constructor("DAG_component",
                          Constructor("input", Literal(None))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal(("A", None, 1))))

    target16 = Constructor("DAG_component",
                          Constructor("input", Literal(None))
                          & Constructor("output", Literal(1))
                          & Constructor("structure", Literal((None, None, 1))))

    target17 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(None))
                          & Constructor("structure", Literal(("A", 3, 1))))

    target18 = Constructor("DAG_component",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal(("A", 3, None))))

    target19 = Constructor("DAG_component",
                          Constructor("input", Literal(None))
                          & Constructor("output", Literal(None))
                          & Constructor("structure", Literal(("A", 3, 1))))

    target20 = Constructor("DAG_component",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal(("A", None, None))))

    target21 = Constructor("DAG_component",
                          Constructor("input", Literal(3))
                          & Constructor("output", Literal(3))
                          & Constructor("structure", Literal((("swap",0, 3), 3, 3))))

    target22 = Constructor("DAG_component",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal((("swap", 2, 1), 3, 3))))

    target23 = Constructor("DAG_component",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal((("swap", 2, None), 3, 3))))

    target24 = Constructor("DAG_component",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal((("swap", None, 1), 3, 3))))

    target25 = Constructor("DAG_component",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal((("swap", None, None), 3, 3))))

    target26 = Constructor("DAG_component",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal((("swap", None, None), 3, 3))))

    target27 = Constructor("DAG_component",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal((("swap", None, None), None, None))))

    target28 = Constructor("DAG_parallel",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal((("A", 3, 3),))))

    target29 = Constructor("DAG_parallel",
                           Constructor("input", Literal(2))
                           & Constructor("output", Literal(2))
                           & Constructor("structure", Literal(((("swap", 0, 1), 1, 1), ("A", 1, 1)))))

    target30 = Constructor("DAG_parallel",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal(((("swap", 0, 1), 1, 1), ("A", 1, 1)))))

    target31 = Constructor("DAG_parallel",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal((("A", None, 3),))))

    target32 = Constructor("DAG_parallel",
                           Constructor("input", Literal(1))
                           & Constructor("output", Literal(1))
                           & Constructor("structure", Literal(None)))

    target33 = Constructor("DAG_parallel",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal((None,))))

    target34 = Constructor("DAG_parallel",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal((None, None))))

    target35 = Constructor("DAG_parallel",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal(((("swap", 0, 1), 1, None), ("A", None, 1)))))

    target36 = Constructor("DAG_parallel",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal(((("swap", 0, 1), 1, None), None))))

    target37 = Constructor("DAG_parallel",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal((None, ("A", None, 1)))))

    target38 = Constructor("DAG",
                           Constructor("input", Literal(2))
                           & Constructor("output", Literal(2))
                           & Constructor("structure", Literal(
                               ((("B", 1, 1), ("A", 1, 1)), ((("swap", 0, 1), 1, 1), ("A", 1, 1)),
                                ((("swap", 1, 1), 2, 2),)))))

    target39 = Constructor("DAG",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal(
                               ((("B", 1, 1), ("A", 1, 1)), ((("swap", 0, 1), 1, 1), ("A", 1, 1)),
                                ((("swap", 1, 1), 2, 2),))
                           )))

    target40 = Constructor("DAG",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal(
                               ((("B", 1, None), ("A", 1, None)), ((("swap", 0, 1), 1, 1), ("A", 2, None)),
                                ((("swap", 1, 1), 2, 2),))
                           )))

    target41 = Constructor("DAG",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal(
                               ((None,),)
                           )))

    target42 = Constructor("DAG",
                           Constructor("input", Literal(3))
                           & Constructor("output", Literal(3))
                           & Constructor("structure", Literal(
                               ((("B", None, 1), ("A", 1, None)), (("A", 1, 2), None), None)
                           )))


    targets = [target0, target1, target2, target3, target4, target5, target6, target7,
               target8, target9, target10, target11, target12, target13, target14,
               target15, target16, target17, target18, target19, target20, target21,
               target22, target23, target24, target25, target26, target27, target28, target29, target30, target31,
               target32, target33, target34, target35, target36, target37, target38,
               target39, target40, target41, target42]


    # TODO: targets to test for normal forms
    # TODO: normalization procedure for structure literals

    """
    test swaplaw1 - predicate:
    terms of structure
    before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
    should be False, everything else True
    """
    def target43(m, n, p, q):
        return Constructor("DAG",
                           Constructor("input", Literal(m + n))
                           & Constructor("output", Literal(p + q))
                           & Constructor("structure", Literal(
                               (((("swap", m, n), m+n, m+n),), (("A", n, p), ("B", m, q)),
                                ((("swap", p, q), p+q, p+q),))
                           )))

    target44 = Constructor("DAG",
                           Constructor("input", Literal(None))
                           & Constructor("output", Literal(None))
                           & Constructor("structure", Literal(
                               (((("swap", None, None), None, None),), (("A", None, None), ("B", None, None)),
                                ((("swap", None, None), None, None),))
                           )))

    #"""
    target = target44

    print(target)

    search_space = synthesizer.construct_search_space(target)

    #for r in search_space.as_tuples():
    #    print(r)

    terms = search_space.enumerate_trees(target, 100)

    for t in terms:
        print(t)
        print(repo.swaplaw1(t.children[6], t.children[7]))
   # """
"""
    rs = []
    for t in targets:
        print("Searching for target:")
        print(t)
        search_space = synthesizer.construct_search_space(t)
        terms = search_space.enumerate_trees(t, 2)
        terms = list(terms)
        n = len(terms)
        print(f"Found {n} terms.")
        rs.append(n > 0)

    print(all(rs))
"""
