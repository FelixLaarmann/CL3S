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
