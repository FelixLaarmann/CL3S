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
        self.dimension_choices = dimension_choices
        self.normalization_eps_choices = normalization_eps_choices
        self.dropout_p_choices = dropout_p_choices
        self.kernel_size_choices = kernel_size_choices if 1 in kernel_size_choices else kernel_size_choices.append(1)
        #self.conv_block_length_choices = conv_block_length_choices
        self.maxpool_size_choices = maxpool_size_choices

        # Parameters that are optional in request language need to have None as a choice
        self.normalization_eps_choices.append(None)
        self.dropout_p_choices.append(None)

    class Maybe_Nat(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, int) and value >= 0)

    class Nat(Container):
        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    class Maybe_Nat_List(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(isinstance(v, int) and v >= 0 for v in value))

    convs = ["simple_convolution", "depthwise_separable_convolution", None]

    afs = ["ReLu", "ELU", None]

    class Maybe_Conv_list(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in UtimeRepository.convs for v in value))

    class Maybe_AF_list(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, list) and all(v in UtimeRepository.afs for v in value))



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
            "length": self.Nat(), # self.Maybe_Nat(),
            "size_list": self.Maybe_Nat_List(),
            "convolution_list": self.Maybe_Conv_list(),
            "activation_function_list": self.Maybe_AF_list(),
        }

    def specification(self):
        return {
            "ReLu": Constructor("activation_function") & Literal("Relu", "activation_function")
                    & Literal(None, "activation_function"),

            "ELU": Constructor("activation_function") & Literal("ELU", "activation_function")
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
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization", Constructor("batch_norm"))
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
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization", Constructor("conv1d_layer_norm"))
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
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization",
                                              Constructor("channel_wise_norm",
                                                          Constructor("normalization_epsilon", Var("e"))
                                                          )
                                              )
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
            .parameter("m", "maxpool_size")
            .argument("mp", Constructor("maxpool1d", Constructor("maxpool_size", Var("m"))))
            .suffix(
                (
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization",
                                              Constructor("channel_wise_norm",
                                                          Constructor("normalization_epsilon", Var("e"))
                                                          )
                                              )
                        )
                        **
                        (
                                Constructor("encoder",
                                            (
                                                    Constructor("conv_block",
                                                                Constructor("input", Var("in"))
                                                                & Constructor("output", Var("out"))
                                                                & Constructor("kernel_size", Var("k")))
                                                    & Constructor("convolution", Var("c"))
                                                    & Constructor("activation", Var("af"))
                                                    & Constructor("dropout_p", Var("d"))
                                                    & Constructor("normalization",
                                                                  Constructor("channel_wise_norm",
                                                                              Constructor("normalization_epsilon",
                                                                                          Var("e"))
                                                                              )
                                                                  )
                                            )
                                            & Constructor("maxpool1d", Constructor("maxpool_size", Var("m")))
                                            )
                        )
                )
                &
                (
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization", Constructor("conv1d_layer_norm"))
                        )
                        **
                        (
                            Constructor("encoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in"))
                                                            & Constructor("output", Var("out"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization", Constructor("conv1d_layer_norm"))
                                        )
                                        & Constructor("maxpool1d", Constructor("maxpool_size", Var("m")))
                                        )
                        )
                )
                &
                (
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization", Constructor("batch_norm"))
                        )
                        **
                        (
                            Constructor("encoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in"))
                                                            & Constructor("output", Var("out"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization", Constructor("batch_norm"))
                                        )
                                        & Constructor("maxpool1d", Constructor("maxpool_size", Var("m")))
                                        )
                        )
                )
            ),

            "Decoder": DSL()
            .parameter("in", "dimension")
            .parameter("out", "dimension")
            .parameter("k", "kernel_size")
            .parameter("d", "dropout_p")
            .parameter("af", "activation_function")
            .parameter("c", "convolution")
            .parameter("e", "normalization_eps")
            .parameter("sf", "maxpool_size")
            .argument("up", Constructor("upsample1d", Constructor("scale_factor", Var("sf"))))
            .suffix(
                (
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization",
                                              Constructor("channel_wise_norm",
                                                          Constructor("normalization_epsilon", Var("e"))
                                                          )
                                              )
                        )
                        **
                        (
                            Constructor("decoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in"))
                                                            & Constructor("output", Var("out"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization",
                                                              Constructor("channel_wise_norm",
                                                                          Constructor("normalization_epsilon",
                                                                                      Var("e"))
                                                                          )
                                                              )
                                        )
                                        & Constructor("upsample1d", Constructor("scale_factor", Var("sf")))
                                        )
                        )
                )
                &
                (
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization", Constructor("conv1d_layer_norm"))
                        )
                        **
                        (
                            Constructor("decoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in"))
                                                            & Constructor("output", Var("out"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization", Constructor("conv1d_layer_norm"))
                                        )
                                        & Constructor("upsample1d", Constructor("scale_factor", Var("sf")))
                                        )
                        )
                )
                &
                (
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("in"))
                                            & Constructor("output", Var("out"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization", Constructor("batch_norm"))
                        )
                        **
                        (
                            Constructor("decoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in"))
                                                            & Constructor("output", Var("out"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization", Constructor("batch_norm"))
                                        )
                                        & Constructor("upsample1d", Constructor("scale_factor", Var("sf")))
                                        )
                        )
                )
            ),

            "Linear": DSL()
            .parameter("in", "dimension")
            .parameter("out", "dimension")
            .suffix(
                Constructor("linear",
                            Constructor("input", Var("in"))
                            & Constructor("output", Var("out"))
                            )
            ),

            "UModel": DSL()
            .parameter("in", "dimension")
            .parameter("out_enc", "dimension")
            .parameter("in_dec", "dimension", lambda v: [2 * v["out_enc"]])
            .parameter("k", "kernel_size")
            .parameter("d", "dropout_p")
            .parameter("af", "activation_function")
            .parameter("c", "convolution")
            .parameter("e", "normalization_eps")
            .parameter("m", "maxpool_size")
            .parameter("cs", "convolution_list", lambda v: [v["c"]])
            .parameter("afs", "activation_function_list", lambda v: [v["af"]])
            .suffix(
                (
                        (
                            Constructor("encoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in"))
                                                            & Constructor("output", Var("out_enc"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization", Constructor("batch_norm"))
                                        )
                                        & Constructor("maxpool1d", Constructor("maxpool_size", Var("m")))
                                        )
                        )
                        **
                        (
                            Constructor("decoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in_dec"))
                                                            & Constructor("output", Var("in"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization", Constructor("batch_norm"))
                                        )
                                        & Constructor("upsample1d", Constructor("scale_factor", Var("m")))
                                        )
                        )
                        **
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("out_enc"))
                                            & Constructor("output", Var("out_enc"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization", Constructor("batch_norm"))
                        )
                        **
                        (
                                    Constructor("u_model",
                                                Constructor("in_and_out", Var("in"))
                                                #& Constructor("output", Var("in"))
                                                & Constructor("length", Literal(0, "length"))
                                                & Constructor("length", Literal(None, "length"))
                                                & Constructor("conv_list", Literal(None, "convolution_list"))
                                                & Constructor("conv_list", Var("cs"))
                                                & Constructor("af_list", Literal(None, "activation_function_list"))
                                                & Constructor("af_list", Var("afs")) # TODO improve this
                                                )
                                    & Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Constructor("batch_norm"))
                        )
                )
                &
                (
                        (
                            Constructor("encoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in"))
                                                            & Constructor("output", Var("out_enc"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization", Constructor("conv1d_layer_norm"))
                                        )
                                        & Constructor("maxpool1d", Constructor("maxpool_size", Var("m")))
                                        )
                        )
                        **
                        (
                            Constructor("decoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in_dec"))
                                                            & Constructor("output", Var("in"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization", Constructor("conv1d_layer_norm"))
                                        )
                                        & Constructor("upsample1d", Constructor("scale_factor", Var("m")))
                                        )
                        )
                        **
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("out_enc"))
                                            & Constructor("output", Var("out_enc"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization", Constructor("conv1d_layer_norm"))
                        )
                        **
                        (
                                        Constructor("u_model",
                                                    Constructor("in_and_out", Var("in"))
                                                    #& Constructor("output", Var("in"))
                                                    & Constructor("length", Literal(0, "length"))
                                                    & Constructor("length", Literal(None, "length"))
                                                    )
                                        & Constructor("convolution", Var("c"))
                                        & Constructor("activation", Var("af"))
                                        & Constructor("dropout_p", Var("d"))
                                        & Constructor("normalization", Constructor("conv1d_layer_norm"))
                        )
                )
                &
                (
                        (
                            Constructor("encoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in"))
                                                            & Constructor("output", Var("out_enc"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization",
                                                              Constructor("channel_wise_norm",
                                                                          Constructor("normalization_epsilon",
                                                                                      Var("e"))
                                                                          )
                                                              )
                                        )
                                        & Constructor("maxpool1d", Constructor("maxpool_size", Var("m")))
                                        )
                        )
                        **
                        (
                            Constructor("decoder",
                                        (
                                                Constructor("conv_block",
                                                            Constructor("input", Var("in_dec"))
                                                            & Constructor("output", Var("in"))
                                                            & Constructor("kernel_size", Var("k")))
                                                & Constructor("convolution", Var("c"))
                                                & Constructor("activation", Var("af"))
                                                & Constructor("dropout_p", Var("d"))
                                                & Constructor("normalization",
                                                              Constructor("channel_wise_norm",
                                                                          Constructor("normalization_epsilon",
                                                                                      Var("e"))
                                                                          )
                                                              )
                                        )
                                        & Constructor("upsample1d", Constructor("scale_factor", Var("m")))
                                        )
                        )
                        **
                        (
                                Constructor("conv_block",
                                            Constructor("input", Var("out_enc"))
                                            & Constructor("output", Var("out_enc"))
                                            & Constructor("kernel_size", Var("k")))
                                & Constructor("convolution", Var("c"))
                                & Constructor("activation", Var("af"))
                                & Constructor("dropout_p", Var("d"))
                                & Constructor("normalization",
                                              Constructor("channel_wise_norm",
                                                          Constructor("normalization_epsilon",
                                                                      Var("e"))
                                                          )
                                              )
                        )
                        **
                        (
                                        Constructor("u_model",
                                                    Constructor("in_and_out", Var("in"))
                                                    #& Constructor("output", Var("in"))
                                                    & Constructor("length", Literal(0, "length"))
                                                    & Constructor("length", Literal(None, "length"))
                                                    )
                                        & Constructor("convolution", Var("c"))
                                        & Constructor("activation", Var("af"))
                                        & Constructor("dropout_p", Var("d"))
                                        & Constructor("normalization",
                                                      Constructor("channel_wise_norm",
                                                                  Constructor("normalization_epsilon",
                                                                              Var("e"))
                                                                  )
                                                      )
                        )
                )
            ),

            "UModel_Cons": DSL()
            .parameter("in_u", "dimension")  # in_u == out_enc
            .parameter("in_enc", "dimension")
            .parameter("in_dec", "dimension", lambda v: [2 * v["in_u"]])
            .parameter("k", "kernel_size")
            .parameter("d", "dropout_p")
            .parameter("af", "activation_function")
            .parameter("c", "convolution")
            .parameter("e", "normalization_eps")
            .parameter("m", "maxpool_size")
            .parameter("n", "length")
            .parameter("n_u", "length", lambda v: [v["n"] - 1])
            .suffix(Constructor("TODO")) # TODO

        }
