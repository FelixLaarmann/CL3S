import torch
import gpytorch
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch import Module

from copy import deepcopy
from collections.abc import Callable, Hashable, Sequence

from grakel.kernels import (
    RandomWalk,
    WeisfeilerLehman,
    VertexHistogram
)

from typing import Any, Generic, Optional, TypeVar, Union, Generator
import typing

from ..tree import DerivationTree

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names

class GraphKernel(Module):
    """
    A base class supporting external graph kernels.
    The external kernel must have a method `fit_transform`, which, when
    evaluated on an `Inputs` instance `X`, returns a scaled kernel matrix
    v * k(X, X).

    As gradients are not propagated through to the external kernel, outputs are
    cached to avoid repeated computation.
    """

    def __init__(
            self,
            dtype=torch.float,
    ) -> None:
        super().__init__()
        self.node_label = None
        self.edge_label = None
        self._scale_variance = torch.nn.Parameter(
            torch.tensor([0.1], dtype=dtype)
        )

    def scale(self, S: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self._scale_variance) * S

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.scale(self.kernel(X))

    def kernel(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")

class WeisfeilerLehmanKernel(GraphKernel):
    """
    A GraKel wrapper for the Weisfeiler-Lehman kernel.
    This kernel needs node labels to be specified and
    can optionally use edge labels for the base kernel.

    See https://ysig.github.io/GraKeL/0.1a8/kernels/weisfeiler_lehman.html
    for more details.
    """

    def __init__(self, n_iter=5, base_graph_kernel=VertexHistogram, normalize=True, dtype=torch.float):
        super().__init__(dtype=dtype)
        self.n_iter = n_iter
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize,

    # @lru_cache(maxsize=5)
    def kernel(self, X: list[DerivationTree[NT, T, G]]) -> torch.Tensor:
        wl_kernel = WeisfeilerLehman(
            n_iter=self.n_iter,
            base_graph_kernel=self.base_graph_kernel,
            normalize=self.normalize,
        )
        gs = [x.to_grakel_graph() for x in X]
        return torch.tensor(
            wl_kernel.fit_transform(gs)
        ).float()

