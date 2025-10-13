import torch
from torch import nn

from collections.abc import Container

from cl3s import DSL, Constructor, Literal, Type, Var, SearchSpaceSynthesizer

from typing import Any

class UtimeRepository:
    def __init__(self, dimension_choices, normalization_eps_choices,
                 dropout_p_choices, kernel_size_choices, #conv_block_length_choices: list[int],
                 maxpool_size_choices,
                 ):
        self.dimension_choices = dimension_choices + [x for x in map(lambda x: x * 2, dimension_choices) if x not in dimension_choices]
        self.normalization_eps_choices = normalization_eps_choices
        self.dropout_p_choices = dropout_p_choices
        self.kernel_size_choices = kernel_size_choices
        #self.conv_block_length_choices = conv_block_length_choices
        self.maxpool_size_choices = maxpool_size_choices

        if 1 not in self.kernel_size_choices:
            self.kernel_size_choices.append(1)

        # Parameters that are optional in request language need to have None as a choice
        self.normalization_eps_choices.append(None)
        self.dropout_p_choices.append(None)

    convs = ["simple_convolution", "depthwise_separable_convolution", None]

    afs = ["ReLu", "ELU", "Tanh", None]

    norms = ["batch_norm", "conv1d_layer_norm", "channel_wise_norm", None]

    class Maybe_Nat(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, int) and value >= 0)

    class Nat(Container):
        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    class Maybe_Nat_List(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(isinstance(v, int) and v >= 0 for v in value))

    # TODO: refactor inner class to match the factory pattern of Maybe_dropout_list
    class Maybe_Conv_list(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in UtimeRepository.convs for v in value))

    class Maybe_Conv_list_list(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(isinstance(v, list) and all(c in UtimeRepository.convs for c in v) for v in value))

    class Maybe_AF_list(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in UtimeRepository.afs for v in value))

    class Maybe_Norm_list(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in UtimeRepository.norms for v in value))

    class Maybe_dropout_list(Container):
        def __init__(self, dropout_p_choices):
            self.dropout_p_choices = dropout_p_choices
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in self.dropout_p_choices for v in value))

    class Maybe_kernel_size_list(Container):
        def __init__(self, kernel_size_choices):
            self.kernel_size_choices = kernel_size_choices
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in self.kernel_size_choices for v in value))

    class Maybe_maxpool_size_list(Container):
        def __init__(self, maxpool_size_choices):
            self.maxpool_size_choices = maxpool_size_choices
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in self.maxpool_size_choices for v in value))

    class Maybe_dimension_list(Container):
        def __init__(self, dimension_choices):
            self.dimension_choices = dimension_choices
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in self.dimension_choices for v in value))

    def parameters(self) -> dict[str, list[Any]]:
        return {
            "bool": [True, False],
            "dimension": self.dimension_choices,
            "normalization_eps": self.normalization_eps_choices,
            "dropout_p": self.dropout_p_choices,
            "kernel_size": self.kernel_size_choices,
            "maxpool_size": self.maxpool_size_choices,
            #"conv_block_length": self.conv_block_length_choices,
            # Some labels to utilize in the specification
            "activation_function": self.afs,
            "convolution": self.convs,
            "normalization": self.norms,
            "length": self.Nat(), # self.Maybe_Nat(),
            "size_list": self.Maybe_Nat_List(),
            "convolution_list": self.Maybe_Conv_list(),
            "activation_function_list": self.Maybe_AF_list(),
            "normalization_list": self.Maybe_Norm_list(),
            "dropout_list": self.Maybe_dropout_list(self.dropout_p_choices),
            "kernel_size_list": self.Maybe_kernel_size_list(self.kernel_size_choices),
            "maxpool_size_list": self.Maybe_maxpool_size_list(self.maxpool_size_choices),
            "dimension_list": self.Maybe_dimension_list(self.dimension_choices),
        }

    def specification(self):
        return {
            "ReLu": Constructor("activation_function") & Literal("ReLu", "activation_function")
                    & Literal(None, "activation_function"),

            "ELU": Constructor("activation_function") & Literal("ELU", "activation_function")
                   & Literal(None, "activation_function"),

            "Tanh": Constructor("activation_function") & Literal("Tanh", "activation_function")
                   & Literal(None, "activation_function"),

            "BatchNorm1d": DSL()
            .parameter("n", "dimension")
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n")))
                    & Constructor("batch_norm")
                    ),

            "ChannelWiseNorm": DSL()
            .parameter("n", "dimension")
            .parameter("e", "normalization_eps")
            .parameter_constraint(lambda v: v["e"] is not None)
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n")))
                    & Constructor("channel_wise_norm",
                                  Constructor("normalization_epsilon", Var("e"))) &
                    Constructor("channel_wise_norm",
                                Constructor("normalization_epsilon", Literal(None, "normalization_eps")))
                    ),

            "Conv1dLayerNorm": DSL()
            .parameter("n", "dimension")
            .suffix(Constructor("normalization",
                                Constructor("output", Var("n")))
                    & Constructor("conv1d_layer_norm")
                    ),

            "Dropout1d": DSL()
            .parameter("d", "dropout_p")
            .parameter_constraint(lambda v: v["d"] is not None)
            .suffix(Constructor("dropout",
                                Constructor("probability", Var("d")))
                    & Constructor("dropout",
                                  Constructor("probability", Literal(None, "dropout_p")))
                    ),

            "Maxpool1d": DSL()
            .parameter("n", "maxpool_size")
            .suffix(Constructor("maxpool1d",
                                Constructor("maxpool_size", Var("n")))
                    ),

            "Upsample1d": DSL()
            .parameter("n", "maxpool_size")
            .suffix(Constructor("upsample1d",
                                Constructor("scale_factor", Var("n")))
                    ),

            "Conv1d": DSL()
            .parameter("in", "dimension")
            .parameter("out", "dimension")
            .parameter("k", "kernel_size")
            .suffix(Constructor("1d_conv_layer",
                                Constructor("input", Var("in"))
                                & Constructor("output", Var("out"))
                                & Constructor("kernel_size", Var("k")))
                    & Literal("simple_convolution", "convolution")
                    & Literal(None, "convolution")),

            "DepthwiseSeparableConv1d": DSL()
            .parameter("in", "dimension")
            .parameter("out", "dimension")
            .parameter("k", "kernel_size")
            .suffix(Constructor("1d_conv_layer",
                                Constructor("input", Var("in"))
                                & Constructor("output", Var("out"))
                                & Constructor("kernel_size", Var("k")))
                    & Literal("depthwise_separable_convolution", "convolution")
                    & Literal(None, "convolution")),

            "ConvBlock": DSL()
            .parameter("in", "dimension")
            .parameter("out", "dimension")
            .parameter("k", "kernel_size")
            .parameter("d", "dropout_p")
            .parameter("af", "activation_function")
            .parameter("c", "convolution")
            .parameter("e", "normalization_eps")
            .argument("activation", Constructor("activation_function") & Var("af"))
            .argument("dropout", Constructor("dropout", Constructor("probability", Var("d"))))
            .suffix(
                (
                        (
                                Constructor("1d_conv_layer",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k"))
                                            )
                                & Var("c")
                        )
                        ** (
                                Constructor("1d_conv_layer",
                                            Constructor("input", Var("out"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k"))
                                            )
                                & Var("c")
                        )
                        ** (
                                Constructor("normalization",
                                       Constructor("output", Var("out")))
                                & Constructor("batch_norm")
                        )
                        ** (
                                Constructor("conv_block",
                                        Constructor("input", Var("in"))
                                        & Constructor("output", Var("out"))
                                        & Constructor("kernel_size", Var("k")))
                                & Constructor("homogeneous",
                                              Constructor("convolution", Var("c"))
                                              & Constructor("activation", Var("af"))
                                              & Constructor("dropout_p", Var("d"))
                                              & Constructor("normalization", Literal("batch_norm", "normalization"))
                                              & Constructor("normalization", Literal(None, "normalization"))
                                              & Constructor("normalization_epsilon", Literal(None, "normalization_eps")))
                        )
                )
                &
                (
                        (
                                Constructor("1d_conv_layer",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k"))
                                            )
                                & Var("c")
                        )
                        ** (
                                Constructor("1d_conv_layer",
                                            Constructor("input", Var("out"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k"))
                                            )
                                & Var("c")
                        )
                        ** (
                                Constructor("normalization",
                                            Constructor("output", Var("out")))
                                & Constructor("conv1d_layer_norm")
                        )
                        ** (
                                Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("homogeneous",
                                              Constructor("convolution", Var("c"))
                                              & Constructor("activation", Var("af"))
                                              & Constructor("dropout_p", Var("d"))
                                              & Constructor("normalization", Literal("conv1d_layer_norm", "normalization"))
                                              & Constructor("normalization", Literal(None, "normalization"))
                                              & Constructor("normalization_epsilon",
                                                            Literal(None, "normalization_eps")))
                        )
                )
                &
                (
                        (
                                Constructor("1d_conv_layer",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k"))
                                            )
                                & Var("c")
                        )
                        ** (
                                Constructor("1d_conv_layer",
                                            Constructor("input", Var("out"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k"))
                                            )
                                & Var("c")
                        )
                        ** (
                                Constructor("normalization",
                                            Constructor("output", Var("out")))
                                & Constructor("channel_wise_norm",
                                              Constructor("normalization_epsilon", Var("e")))
                        )
                        ** (
                                Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("homogeneous",
                                              Constructor("convolution", Var("c"))
                                              & Constructor("activation", Var("af"))
                                              & Constructor("dropout_p", Var("d"))
                                              & Constructor("normalization",
                                                            Literal("channel_wise_norm", "normalization"))
                                              & Constructor("normalization", Literal(None, "normalization"))
                                              & Constructor("normalization_epsilon", Var("e")))
                        )
                )
            ),

            "Encoder": DSL()
            .parameter("in", "dimension")
            .parameter("out", "dimension")
            .parameter("k", "kernel_size")
            .parameter("d", "dropout_p")
            .parameter("af", "activation_function")
            .parameter("c", "convolution")
            .parameter("e", "normalization_eps")
            .parameter("n", "normalization")
            .parameter("m", "maxpool_size")
            .argument("mp", Constructor("maxpool1d", Constructor("maxpool_size", Var("m"))))
            .argument("cb", Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))))
            .suffix(Constructor("encoder",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("maxpool_size", Var("m")))),

            "Decoder": DSL()
            .parameter("in", "dimension")
            .parameter("out", "dimension")
            .parameter("k", "kernel_size")
            .parameter("d", "dropout_p")
            .parameter("af", "activation_function")
            .parameter("c", "convolution")
            .parameter("e", "normalization_eps")
            .parameter("n", "normalization")
            .parameter("sf", "maxpool_size")
            .argument("up", Constructor("upsample1d", Constructor("scale_factor", Var("sf"))))
            .argument("cb", Constructor("conv_block",
                                        Constructor("input", Var("in"))
                                        & Constructor("output", Var("out"))
                                        & Constructor("kernel_size", Var("k")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))))
            .suffix(Constructor("decoder",
                                Constructor("input", Var("in"))
                                & Constructor("output", Var("out"))
                                & Constructor("kernel_size", Var("k")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("upsample_size", Var("sf")))),

            "Linear": DSL()
            .parameter("in", "dimension")
            .parameter("out", "dimension")
            .suffix(
                Constructor("linear",
                            Constructor("input", Var("in"))
                            & Constructor("output", Var("out"))
                            )
            ),

            "UModel_length": DSL()
            .parameter("in", "dimension")
            .parameter("out_enc", "dimension")
            .parameter("in_dec", "dimension", lambda v: [2 * v["out_enc"]])
            .parameter("k1", "kernel_size")
            .parameter("k2", "kernel_size")
            .parameter("d", "dropout_p")
            .parameter("af", "activation_function")
            .parameter("c", "convolution")
            .parameter("e", "normalization_eps")
            .parameter("n", "normalization")
            .parameter("m", "maxpool_size")
            .argument("enc", Constructor("encoder",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out_enc"))
                                            & Constructor("kernel_size", Var("k1")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("maxpool_size", Var("m"))))
            .argument("dec", Constructor("decoder",
                                Constructor("input", Var("in_dec"))
                                & Constructor("output", Var("in"))
                                & Constructor("kernel_size", Var("k1")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("upsample_size", Var("m"))))
            .argument("cb", Constructor("conv_block",
                                        Constructor("input", Var("out_enc"))
                                        & Constructor("output", Var("out_enc"))
                                        & Constructor("kernel_size", Var("k2")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))))
            .suffix(Constructor("u_model",
                                                Constructor("in_and_out", Var("in"))
                                                & Constructor("length", Literal(1, "length"))
                                                )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("out_enc"))
                                  & Constructor("kernel_size", Var("k2")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e")))
            ),

            "UModel_Cons_length": DSL()
            .parameter("in_u", "dimension")  # in_u == out_enc
            .parameter("in_enc", "dimension")
            .parameter("in_dec", "dimension", lambda v: [2 * v["in_u"]])
            .parameter("bd", "dimension")
            .parameter("k", "kernel_size")
            .parameter("bk", "kernel_size")
            .parameter("d", "dropout_p")
            .parameter("af", "activation_function")
            .parameter("c", "convolution")
            .parameter("e", "normalization_eps")
            .parameter("n", "normalization")
            .parameter("m", "maxpool_size")
            .parameter("l", "length")
            .parameter("l_u", "length", lambda v: [v["l"] - 1])
            .argument("enc", Constructor("encoder",
                                         Constructor("input", Var("in_enc"))
                                         & Constructor("output", Var("in_u"))
                                         & Constructor("kernel_size", Var("k")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    & Constructor("maxpool_size", Var("m"))))
            .argument("dec", Constructor("decoder",
                                         Constructor("input", Var("in_dec"))
                                         & Constructor("output", Var("in_enc"))
                                         & Constructor("kernel_size", Var("k")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    & Constructor("upsample_size", Var("m"))))
            .argument("u_model", Constructor("u_model",
                                                Constructor("in_and_out", Var("in_u"))
                                                & Constructor("length", Var("l_u"))
                                                )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("bd"))
                                  & Constructor("kernel_size", Var("bk")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e"))))
            .suffix(Constructor("u_model",
                                                Constructor("in_and_out", Var("in_enc"))
                                                & Constructor("length", Var("l"))
                                                )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("bd"))
                                  & Constructor("kernel_size", Var("bk")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e")))),
        }

if __name__ == "__main__":
    repo = UtimeRepository(
        dimension_choices=[64, 128, 256],
        normalization_eps_choices=[1e-3],
        dropout_p_choices=[0.1],
        kernel_size_choices=[5],
        maxpool_size_choices=[3])

    target0 = (Constructor("u_model",
                                                Constructor("in_and_out", Literal(128, "dimension"))
                                                & Constructor("length", Literal(3, "length"))
                                                )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Literal(None, "convolution"))
                                  & Constructor("activation", Literal(None, "activation_function"))
                                  & Constructor("dropout_p", Literal(None, "dropout_p"))
                                  & Constructor("normalization", Literal(None, "normalization"))
                                  & Constructor("normalization_epsilon", Literal(None, "normalization_eps"))
                                  )
              )

    target1 = (Constructor("u_model",
                                                Constructor("in_and_out", Literal(128, "dimension"))
                                                & Constructor("length", Literal(3, "length"))
                                                )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Literal(64, "dimension"))
                                  & Constructor("kernel_size", Literal(1, "kernel_size")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Literal(None, "convolution"))
                                  & Constructor("activation", Literal(None, "activation_function"))
                                  & Constructor("dropout_p", Literal(None, "dropout_p"))
                                  & Constructor("normalization", Literal(None, "normalization"))
                                  & Constructor("normalization_epsilon", Literal(None, "normalization_eps"))
                                  )
              )

    target = target0

    print(repo.parameters())

    synthesizer = SearchSpaceSynthesizer(repo.specification(), repo.parameters(), {})

    search_space = synthesizer.construct_search_space(target).prune()

    trees = search_space.enumerate_trees(target, 10)

    for t in trees:
        print(t)
