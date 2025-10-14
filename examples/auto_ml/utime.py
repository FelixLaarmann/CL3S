import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

from collections.abc import Container

from cl3s import DSL, Constructor, Literal, Type, Var, SearchSpaceSynthesizer

from typing import Any


class ChannelWiseNormalization(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ChannelWiseNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x has shape (batch_size, num_channels, time_steps)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_scaled = self.gamma.view(1, -1, 1) * x_normalized + self.beta.view(1, -1, 1)
        return x_scaled


class Conv1dLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super(Conv1dLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # Permute to [batch_size, length, channels] for LayerNorm
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        # Permute back to [batch_size, channels, length]
        x = x.permute(0, 2, 1)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()

        # Depthwise convolution: one filter per input channel
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )

        # Pointwise (1x1) convolution: mixes channels
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

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
        self.kernel_size_choices.append(None)
        self.maxpool_size_choices.append(None)
        self.dimension_choices.append(None)

        self.convs = ["simple_convolution", "depthwise_separable_convolution", None]

        self.afs = ["ReLu", "ELU", "Tanh", None]

        self.norms = ["batch_norm", "conv1d_layer_norm", "channel_wise_norm", None]

    class Maybe_Nat(Container):
        def __contains__(self, value: object) -> bool:
            return value is None or (isinstance(value, int) and value >= 0)

    class Nat(Container):
        def __contains__(self, value: object) -> bool:
            return isinstance(value, int) and value >= 0

    class Maybe_Nat_Tuple(Container):
        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else isinstance(v, int) and v >= 0 for v in value)

    class Maybe_Conv_Tuple(Container):
        def __init__(self, conv_choices):
            self.conv_choices = conv_choices

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.conv_choices for v in value)

    class Maybe_Conv_Tuple_Tuple(Container):
        def __init__(self, conv_choices):
            self.conv_choices = conv_choices

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and
                    all(isinstance(v, tuple) and
                        all(True if c is None else c in self.conv_choices for c in v) for v in value))

    class Maybe_AF_Tuple(Container):
        def __init__(self, af_choices):
            self.af_choices = af_choices

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.af_choices for v in value)

    class Maybe_Norm_Tuple(Container):
        def __init__(self, norm_choices):
            self.norm_choices = norm_choices

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.norm_choices for v in value)

    class Maybe_Dropout_Tuple(Container):
        def __init__(self, dropout_p_choices):
            self.dropout_p_choices = dropout_p_choices
        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.dropout_p_choices for v in value)

    class Maybe_Kernel_Size_Tuple(Container):
        def __init__(self, kernel_size_choices):
            self.kernel_size_choices = kernel_size_choices

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.kernel_size_choices for v in value)

    class Maybe_Maxpool_Size_Tuple(Container):
        def __init__(self, maxpool_size_choices):
            self.maxpool_size_choices = maxpool_size_choices

        def __contains__(self, value: object) -> bool:
            return isinstance(value, tuple) and all(True if v is None else v in self.maxpool_size_choices for v in value)

    class Maybe_Dimension_Tuple(Container):
        def __init__(self, dimension_choices):
            self.dimension_choices = dimension_choices

        def __contains__(self, value: object) -> bool:
            return (isinstance(value, tuple) and all(True if v is None else v in self.dimension_choices for v in value))

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
            "size_list": self.Maybe_Nat_Tuple(),
            "convolution_list": self.Maybe_Conv_Tuple(self.convs),
            "activation_function_list": self.Maybe_AF_Tuple(self.afs),
            "normalization_list": self.Maybe_Norm_Tuple(self.norms),
            "dropout_list": self.Maybe_Dropout_Tuple(self.dropout_p_choices),
            "kernel_size_list": self.Maybe_Kernel_Size_Tuple(self.kernel_size_choices),
            "maxpool_size_list": self.Maybe_Maxpool_Size_Tuple(self.maxpool_size_choices),
            "dimension_list": self.Maybe_Dimension_Tuple(self.dimension_choices),
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
                                            & Constructor("kernel_size", Var("k"))
                                            #& Constructor("kernel_size", Literal(None, "kernel_size"))
                                            )
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
                                            & Constructor("kernel_size", Var("k"))
                                            #& Constructor("kernel_size", Literal(None, "kernel_size"))
                                            )
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
                                            & Constructor("kernel_size", Var("k"))
                                            #& Constructor("kernel_size", Literal(None, "kernel_size"))
                                            )
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
                                & Constructor("kernel_size", Var("k"))
                                & Constructor("kernel_size", Literal(None, "kernel_size"))
                                & Constructor("maxpool_size", Var("m"))
                                & Constructor("maxpool_size", Literal(None, "maxpool_size"))
                                )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e")))
                    ),

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
                                & Constructor("kernel_size", Var("k"))
                                #& Constructor("kernel_size", Literal(None, "kernel_size"))
                                & Constructor("upsample_size", Var("sf"))
                                #& Constructor("upsample_size", Literal(None, "maxpool_size"))
                                )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization_epsilon", Var("e"))
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
            .parameter("in", "dimension", lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("out_enc", "dimension", lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("in_dec", "dimension", lambda v: [2 * v["out_enc"]])
            .parameter("k1", "kernel_size", lambda v: [x for x in self.kernel_size_choices if x is not None])
            .parameter("k2", "kernel_size", lambda v: [x for x in self.kernel_size_choices if x is not None])
            .parameter("d", "dropout_p", lambda v: [x for x in self.dropout_p_choices if x is not None])
            .parameter("af", "activation_function", lambda v: [x for x in self.afs if x is not None])
            .parameter("c", "convolution", lambda v: [x for x in self.convs if x is not None])
            .parameter("e", "normalization_eps", lambda v: [x for x in self.normalization_eps_choices if x is not None])
            .parameter("n", "normalization", lambda v: [x for x in self.norms if x is not None])
            .parameter("m", "maxpool_size", lambda v: [x for x in self.maxpool_size_choices if x is not None])
            .parameter("ds", "dimension_list", lambda v: [(v["in"],), (None,)])
            .parameter("ks", "kernel_size_list", lambda v: [(v["k1"],), (None,)])
            .parameter("ms", "maxpool_size_list", lambda v: [(v["m"],), (None,)])
            .argument("enc", Constructor("encoder",
                                         Constructor("input", Var("in"))
                                         & Constructor("output", Var("out_enc"))
                                         & Constructor("kernel_size", Var("k1"))
                                         & Constructor("maxpool_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .argument("dec", Constructor("decoder",
                                         Constructor("input", Var("in_dec"))
                                         & Constructor("output", Var("in"))
                                         & Constructor("kernel_size", Var("k1"))
                                         & Constructor("upsample_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
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
                                Constructor("dimensions", Var("ds"))
                                & Constructor("kernel_sizes", Var("ks"))
                                & Constructor("maxpool_sizes", Var("ms"))
                                )
                    #& Constructor("u_model",
                    #              Constructor("in_and_out", Var("in"))
                    #              & Constructor("length", Literal(1, "length")))
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("out_enc"))
                                  & Constructor("in_and_out", Literal(None, "dimension"))
                                  & Constructor("kernel_size", Var("k2"))
                                  & Constructor("kernel_size", Literal(None, "kernel_size"))
                                  )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("convolution", Literal(None, "convolution"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("activation", Literal(None, "activation_function"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("dropout_p", Literal(None, "dropout_p"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization", Literal(None, "normalization"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("normalization_epsilon", Literal(None, "normalization_eps"))
                                  )
                    ),

            "UModel_Cons": DSL()
            .parameter("in_u", "dimension", lambda v: [x for x in self.dimension_choices if x is not None])  # in_u == out_enc
            .parameter("in_enc", "dimension", lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("in_dec", "dimension", lambda v: [2 * v["in_u"]])
            .parameter("bd", "dimension")
            .parameter("k", "kernel_size", lambda v: [x for x in self.kernel_size_choices if x is not None])
            .parameter("bk", "kernel_size")
            .parameter("d", "dropout_p", lambda v: [x for x in self.dropout_p_choices if x is not None])
            .parameter("af", "activation_function", lambda v: [x for x in self.afs if x is not None])
            .parameter("c", "convolution", lambda v: [x for x in self.convs if x is not None])
            .parameter("e", "normalization_eps", lambda v: [x for x in self.normalization_eps_choices if x is not None])
            .parameter("n", "normalization", lambda v: [x for x in self.norms if x is not None])
            .parameter("m", "maxpool_size", lambda v: [x for x in self.maxpool_size_choices if x is not None])
            .parameter("dds", "dimension_list")
            .parameter_constraint(lambda v: len(v["dds"]) > 1 and (v["dds"][0] == v["in_enc"] or v["dds"][0] is None))
            .parameter("ds", "dimension_list", lambda v: [v["dds"][1:]])
            .parameter_constraint(lambda v: len(v["ds"]) > 0 and (v["ds"][0] == v["in_u"] or v["ds"][0] is None))
            .parameter("kks", "kernel_size_list")
            .parameter_constraint(lambda v: len(v["kks"]) > 1 and (v["kks"][0] == v["k"] or v["kks"][0] is None))
            .parameter("ks", "kernel_size_list", lambda v: [v["kks"][1:]])
            .parameter_constraint(lambda v: len(v["ks"]) > 0)
            .parameter("mms", "maxpool_size_list")
            .parameter_constraint(lambda v: len(v["mms"]) > 1 and (v["mms"][0] == v["m"] or v["mms"][0] is None))
            .parameter("ms", "maxpool_size_list", lambda v: [v["mms"][1:]])
            .parameter_constraint(lambda v: len(v["ms"]) > 0)
            #.parameter("l", "length")
            #.parameter("l_u", "length", lambda v: [v["l"] - 1])  # l_u == len(ds), but (- 1) is more efficient then len
            .argument("enc", Constructor("encoder",
                                         Constructor("input", Var("in_enc"))
                                         & Constructor("output", Var("in_u"))
                                         & Constructor("kernel_size", Var("k"))
                                         & Constructor("maxpool_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .argument("dec", Constructor("decoder",
                                         Constructor("input", Var("in_dec"))
                                         & Constructor("output", Var("in_enc"))
                                         & Constructor("kernel_size", Var("k"))
                                         & Constructor("upsample_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .argument("u_model", Constructor("u_model",
                                             Constructor("dimensions", Var("ds"))
                                             & Constructor("kernel_sizes", Var("ks"))
                                             & Constructor("maxpool_sizes", Var("ms"))
                                             )
                      #& Constructor("u_model",
                      #              Constructor("in_and_out", Var("in_u"))
                      #              & Constructor("length", Var("l_u")))
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
                                Constructor("dimensions", Var("dds"))
                                & Constructor("kernel_sizes", Var("kks"))
                                & Constructor("maxpool_sizes", Var("mms"))
                                )
                    #& Constructor("u_model",
                    #              Constructor("in_and_out", Var("in_enc"))
                    #              & Constructor("length", Var("l"))
                    #              )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("bd"))
                                  & Constructor("kernel_size", Var("bk")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("convolution", Literal(None, "convolution"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("activation", Literal(None, "activation_function"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("dropout_p", Literal(None, "dropout_p"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization", Literal(None, "normalization"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("normalization_epsilon", Literal(None, "normalization_eps"))
                                  )
                    ),

            "UModel_length": DSL()
            .parameter("in", "dimension", lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("out_enc", "dimension", lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("in_dec", "dimension", lambda v: [2 * v["out_enc"]])
            .parameter("k1", "kernel_size", lambda v: [x for x in self.kernel_size_choices if x is not None])
            .parameter("k2", "kernel_size", lambda v: [x for x in self.kernel_size_choices if x is not None])
            .parameter("d", "dropout_p", lambda v: [x for x in self.dropout_p_choices if x is not None])
            .parameter("af", "activation_function", lambda v: [x for x in self.afs if x is not None])
            .parameter("c", "convolution", lambda v: [x for x in self.convs if x is not None])
            .parameter("e", "normalization_eps", lambda v: [x for x in self.normalization_eps_choices if x is not None])
            .parameter("n", "normalization", lambda v: [x for x in self.norms if x is not None])
            .parameter("m", "maxpool_size", lambda v: [x for x in self.maxpool_size_choices if x is not None])
            #.parameter("ds", "dimension_list", lambda v: [(v["in"],), (None,)])
            #.parameter("ks", "kernel_size_list", lambda v: [(v["k1"],), (None,)])
            #.parameter("ms", "maxpool_size_list", lambda v: [(v["m"],), (None,)])
            .argument("enc", Constructor("encoder",
                                         Constructor("input", Var("in"))
                                         & Constructor("output", Var("out_enc"))
                                         & Constructor("kernel_size", Var("k1"))
                                         & Constructor("maxpool_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .argument("dec", Constructor("decoder",
                                         Constructor("input", Var("in_dec"))
                                         & Constructor("output", Var("in"))
                                         & Constructor("kernel_size", Var("k1"))
                                         & Constructor("upsample_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
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
            .suffix( # Constructor("u_model",
                     #            Constructor("dimensions", Var("ds"))
                     #            & Constructor("kernel_sizes", Var("ks"))
                     #            & Constructor("maxpool_sizes", Var("ms"))
                     #            )
                    Constructor("u_model",
                                  Constructor("in_and_out", Var("in"))
                                  & Constructor("length", Literal(1, "length")))
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("out_enc"))
                                  & Constructor("in_and_out", Literal(None, "dimension"))
                                  & Constructor("kernel_size", Var("k2"))
                                  & Constructor("kernel_size", Literal(None, "kernel_size"))
                                  )
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("convolution", Literal(None, "convolution"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("activation", Literal(None, "activation_function"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("dropout_p", Literal(None, "dropout_p"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization", Literal(None, "normalization"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("normalization_epsilon", Literal(None, "normalization_eps"))
                                  )
                    ),

            "UModel_Cons_length": DSL()
            .parameter("in_u", "dimension",
                       lambda v: [x for x in self.dimension_choices if x is not None])  # in_u == out_enc
            .parameter("in_enc", "dimension", lambda v: [x for x in self.dimension_choices if x is not None])
            .parameter("in_dec", "dimension", lambda v: [2 * v["in_u"]])
            .parameter("bd", "dimension")
            .parameter("k", "kernel_size", lambda v: [x for x in self.kernel_size_choices if x is not None])
            .parameter("bk", "kernel_size")
            .parameter("d", "dropout_p", lambda v: [x for x in self.dropout_p_choices if x is not None])
            .parameter("af", "activation_function", lambda v: [x for x in self.afs if x is not None])
            .parameter("c", "convolution", lambda v: [x for x in self.convs if x is not None])
            .parameter("e", "normalization_eps", lambda v: [x for x in self.normalization_eps_choices if x is not None])
            .parameter("n", "normalization", lambda v: [x for x in self.norms if x is not None])
            .parameter("m", "maxpool_size", lambda v: [x for x in self.maxpool_size_choices if x is not None])
            #.parameter("dds", "dimension_list")
            #.parameter_constraint(lambda v: len(v["dds"]) > 1 and (v["dds"][0] == v["in_enc"] or v["dds"][0] is None))
            #.parameter("ds", "dimension_list", lambda v: [v["dds"][1:]])
            #.parameter_constraint(lambda v: len(v["ds"]) > 0 and (v["ds"][0] == v["in_u"] or v["ds"][0] is None))
            #.parameter("kks", "kernel_size_list")
            #.parameter_constraint(lambda v: len(v["kks"]) > 1 and (v["kks"][0] == v["k"] or v["kks"][0] is None))
            #.parameter("ks", "kernel_size_list", lambda v: [v["kks"][1:]])
            #.parameter_constraint(lambda v: len(v["ks"]) > 0)
            #.parameter("mms", "maxpool_size_list")
            #.parameter_constraint(lambda v: len(v["mms"]) > 1 and (v["mms"][0] == v["m"] or v["mms"][0] is None))
            #.parameter("ms", "maxpool_size_list", lambda v: [v["mms"][1:]])
            #.parameter_constraint(lambda v: len(v["ms"]) > 0)
            .parameter("l", "length")
            .parameter("l_u", "length", lambda v: [v["l"] - 1])  # l_u == len(ds), but (- 1) is more efficient then len
            .argument("enc", Constructor("encoder",
                                         Constructor("input", Var("in_enc"))
                                         & Constructor("output", Var("in_u"))
                                         & Constructor("kernel_size", Var("k"))
                                         & Constructor("maxpool_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .argument("dec", Constructor("decoder",
                                         Constructor("input", Var("in_dec"))
                                         & Constructor("output", Var("in_enc"))
                                         & Constructor("kernel_size", Var("k"))
                                         & Constructor("upsample_size", Var("m"))
                                         )
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))
                                    )
                      )
            .argument("u_model", #Constructor("u_model",
                                       #      Constructor("dimensions", Var("ds"))
                                       #      & Constructor("kernel_sizes", Var("ks"))
                                        #     & Constructor("maxpool_sizes", Var("ms"))
                                        #     )
                      Constructor("u_model",
                                    Constructor("in_and_out", Var("in_u"))
                                    & Constructor("length", Var("l_u")))
                      & Constructor("bottleneck",
                                    Constructor("in_and_out", Var("bd"))
                                    & Constructor("kernel_size", Var("bk")))
                      & Constructor("homogeneous",
                                    Constructor("convolution", Var("c"))
                                    & Constructor("activation", Var("af"))
                                    & Constructor("dropout_p", Var("d"))
                                    & Constructor("normalization", Var("n"))
                                    & Constructor("normalization_epsilon", Var("e"))))
            .suffix(#Constructor("u_model",
                    #            Constructor("dimensions", Var("dds"))
                    #            & Constructor("kernel_sizes", Var("kks"))
                    #            & Constructor("maxpool_sizes", Var("mms"))
                    #            )
                    Constructor("u_model",
                                  Constructor("in_and_out", Var("in_enc"))
                                  & Constructor("length", Var("l"))
                                  )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Var("bd"))
                                  & Constructor("kernel_size", Var("bk")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Var("c"))
                                  & Constructor("convolution", Literal(None, "convolution"))
                                  & Constructor("activation", Var("af"))
                                  & Constructor("activation", Literal(None, "activation_function"))
                                  & Constructor("dropout_p", Var("d"))
                                  & Constructor("dropout_p", Literal(None, "dropout_p"))
                                  & Constructor("normalization", Var("n"))
                                  & Constructor("normalization", Literal(None, "normalization"))
                                  & Constructor("normalization_epsilon", Var("e"))
                                  & Constructor("normalization_epsilon", Literal(None, "normalization_eps"))
                                  )
                    ),

        }

    @staticmethod
    def _conv_block(activation, dropout, c1, c2, norm, x):
        x = c1(x)
        x = norm(x)
        x = activation(x)
        x = c2(x)
        x = norm(x)
        x = activation(x)
        x = dropout(x)
        return x

    @staticmethod
    def _encoder(cb, mp, x):
        y = cb(x)
        x = mp(y)
        if x.shape[-1] < 1:
            raise ValueError("Encoder output is empty after pooling. Reduce the pooling size or number of layers.")
        return x, y

    @staticmethod
    def _decoder(cb, up, x, skip):
        x = up(x)
        output_size = skip.size(2)

        if x.size(2) != output_size:
            diff = output_size - x.size(2)
            x = F.pad(x, (0, diff))  # Apply zero padding to the end of the dimension

        x = torch.cat([x, skip], dim=1)
        x = cb(x)
        return x

    @staticmethod
    def _umodel_length(enc, dec, cb, x, return_intermediate=False):
        x, y = enc(x)
        z = cb(x)
        x = dec(z, y)
        if return_intermediate:
            return x, z.flatten(start_dim=1)
        else:
            return x

    @staticmethod
    def _umodel_cons_length(enc, dec, u_model, x, return_intermediate=False):
        x, y = enc(x)
        if return_intermediate:
            z, u_intermediate = u_model(x, return_intermediate=return_intermediate)
            x = dec(z, y)
            return x, u_intermediate
        else:
            z = u_model(x, return_intermediate=return_intermediate)
            x = dec(z, y)
            return x


    def pretty_term_algebra(self):
        return {
            "ReLu": "ReLu()",
            "ELU": "ELU()",
            "Tanh": "Tanh()",
            "BatchNorm1d": (lambda n: f"BatchNorm1d({n})"),
            "ChannelWiseNorm": (lambda n, e: f"ChannelWiseNormalization({n}, {e})"),
            "Conv1dLayerNorm": (lambda n: f"Conv1dLayerNorm({n})"),
            "Dropout1d": (lambda d: f"Dropout1d({d})"),
            "Maxpool1d": (lambda n: f"MaxPool1d({n})"),
            "Upsample1d": (lambda n: f"Upsample({n})"),
            "Conv1d": (lambda i, o, k: f"Conv1d({i}, {o}, {k})"),
            "DepthwiseSeparableConv1d": (lambda i, o, k: f"DepthwiseSeparableConv1d({i}, {o}, {k})"),
            "ConvBlock": (lambda i, o, k, d, af, c, e, activation, dropout, c1, c2, norm:
                          f"Conv_Block({activation}, {dropout}, {c1}, {c2}, {norm})"),
            "Encoder": (lambda i, o, k, d, af, c, e, n, m, mp, cb: f"Encoder({cb}, {mp})"),
            "Decoder": (lambda i, o, k, d, af, c, e, n, m, mp, cb: f"Decoder({cb}, {mp})"),
            "Linear": (lambda i, o: f"Linear({i}, {o})"),
            "UModel": (lambda i, out_enc, in_dec, k1, k2, d, af, c, e, n, m, ds, ks, ms, enc, dec, cb:
                       f"U_Model({enc}, {dec}, {cb})"),

            "UModel_Cons": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, e, n, m,
                                            dds, ds, kks, ks, mms, ms, enc, dec, u_model:
                            f"U_Model_Cons({enc}, {dec}, {u_model})"),
            "UModel_length": (lambda i, out_enc, in_dec, k1, k2, d, af, c, e, n, m, enc, dec, cb:
                       f"U_Model({enc}, {dec}, {cb})"),

            "UModel_Cons_length": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, e, n, m, l, l_u, enc, dec, u_model:
                            f"U_Model_Cons({enc}, {dec}, {u_model})"),
        }

    def torch_algebra(self):
        return {
            "ReLu": nn.ReLU(),
            "ELU": nn.ELU(),
            "Tanh": nn.Tanh(),
            "BatchNorm1d": (lambda n: nn.BatchNorm1d(n)),
            "ChannelWiseNorm": (lambda n, e: ChannelWiseNormalization(n, e)),
            "Conv1dLayerNorm": (lambda n: Conv1dLayerNorm(n)),
            "Dropout1d": (lambda d: nn.Dropout1d(p=d)),
            "Maxpool1d": (lambda n: nn.MaxPool1d(n)),
            "Upsample1d": (lambda n: nn.Upsample(scale_factor=n, mode="nearest")),
            "Conv1d": (lambda i, o, k: nn.Conv1d(i, o, kernel_size=k, padding=k // 2)),
            "DepthwiseSeparableConv1d": (lambda i, o, k: DepthwiseSeparableConv1d(i, o, kernel_size=k, padding=k // 2)),
            "ConvBlock": (lambda i, o, k, d, af, c, e, activation, dropout, c1, c2, norm, x:
                          self._conv_block(activation, dropout, c1, c2, norm, x)),
            "Encoder": (lambda i, o, k, d, af, c, e, n, m, mp, cb, x: self._encoder(cb, mp, x)),
            "Decoder": (lambda i, o, k, d, af, c, e, n, m, mp, cb, x, y: self._decoder(cb, mp, x, y)),
            "Linear": (lambda i, o: nn.Linear(i, o)),
            "UModel": (lambda i, out_enc, in_dec, k1, k2, d, af, c, e, n, m, ds, ks, ms, enc, dec, cb, x:
                       self._umodel_length(enc, dec, cb, x)),

            "UModel_Cons": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, e, n, m,
                                   dds, ds, kks, ks, mms, ms, enc, dec, u_model, x:
                            self._umodel_cons_length(enc, dec, u_model, x)),
            "UModel_length": (lambda i, out_enc, in_dec, k1, k2, d, af, c, e, n, m, enc, dec, cb, x:
                              self._umodel_length(enc, dec, cb, x)),
            "UModel_Cons_length": (lambda in_u, in_enc, in_dec, bd, k, bk, d, af, c, e, n, m, l, l_u, enc, dec, u_model, x:
                                   self._umodel_cons_length(enc, dec, u_model, x)),
        }

if __name__ == "__main__":
    repo = UtimeRepository(
        dimension_choices=[64, 128, 256],
        normalization_eps_choices=[1e-3],
        dropout_p_choices=[0.1],
        kernel_size_choices=[5, 3, 2],
        maxpool_size_choices=[3, 5])

    target0 = (Constructor("u_model",
                                                Constructor("in_and_out", Literal(128, "dimension"))
                                                & Constructor("length", Literal(3, "length"))
                                                )
                    & Constructor("bottleneck",
                                  Constructor("in_and_out", Literal(64, "dimension"))
                                  & Constructor("kernel_size", Literal(1, "kernel_size")))
                    & Constructor("homogeneous",
                                  Constructor("convolution", Literal("simple_convolution", "convolution"))
                                  & Constructor("activation", Literal("ReLu", "activation_function"))
                                  & Constructor("dropout_p", Literal(0.1, "dropout_p"))
                                  & Constructor("normalization", Literal("channel_wise_norm", "normalization"))
                                  & Constructor("normalization_epsilon", Literal(1e-3, "normalization_eps"))
                                  )
              )

    target1 = (Constructor("u_model",
                           Constructor("in_and_out", Literal(256, "dimension"))
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

    target2 = (
            Constructor("u_model",
                        Constructor("dimensions", Literal((256, 128, 128), "dimension_list"))
                        & Constructor("kernel_sizes", Literal((2, 3, 5), "kernel_size_list"))
                        & Constructor("maxpool_sizes", Literal((5, 5, 3), "maxpool_size_list"))
                        )
            & Constructor("bottleneck",
                          Constructor("in_and_out", Literal(64, "dimension"))
                          & Constructor("kernel_size", Literal(1, "kernel_size")))
            & Constructor("homogeneous",
                          Constructor("convolution", Literal("simple_convolution", "convolution"))
                          & Constructor("activation", Literal("ReLu", "activation_function"))
                          & Constructor("dropout_p", Literal(0.1, "dropout_p"))
                          & Constructor("normalization", Literal("channel_wise_norm", "normalization"))
                          & Constructor("normalization_epsilon", Literal(1e-3, "normalization_eps"))
                          )
              )

    target3 = (
            Constructor("u_model",
                        Constructor("dimensions",
                                    Literal((256, None, 128), "dimension_list"))
                        & Constructor("kernel_sizes", Literal((None, None, 5), "kernel_size_list"))
                        & Constructor("maxpool_sizes", Literal((None, 5, None), "maxpool_size_list"))
                        )
            & Constructor("bottleneck",
                          Constructor("in_and_out", Literal(None, "dimension"))
                          & Constructor("kernel_size", Literal(None, "kernel_size")))
            & Constructor("homogeneous",
                          Constructor("convolution", Literal("simple_convolution", "convolution"))
                          & Constructor("activation", Literal(None, "activation_function"))
                          & Constructor("dropout_p", Literal(0.1, "dropout_p"))
                          & Constructor("normalization", Literal("channel_wise_norm", "normalization"))
                          & Constructor("normalization_epsilon", Literal(1e-3, "normalization_eps"))
                          )
    )

    target = target3

    print(repo.parameters())

    synthesizer = SearchSpaceSynthesizer(repo.specification(), repo.parameters(), {})

    search_space = synthesizer.construct_search_space(target).prune()

    trees = search_space.enumerate_trees(target, 10)

    #trees = search_space.sample(2, target)

    for t in trees:
        print(t.interpret(repo.pretty_term_algebra()))
