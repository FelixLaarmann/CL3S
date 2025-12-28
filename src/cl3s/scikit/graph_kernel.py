import numpy as np

from sklearn.base import clone
from sklearn.gaussian_process.kernels import GenericKernelMixin, NormalizedKernelMixin, Hyperparameter, Kernel
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx

from collections.abc import Callable, Hashable, Sequence

from typing import Any, Generic, Optional, TypeVar, Union, Generator
import typing

from ..tree import DerivationTree

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names

class WeisfeilerLehmanKernel(GenericKernelMixin, NormalizedKernelMixin, Kernel, Generic[NT, T, G]):
    """ Weisfeiler Lehman Kernel for derivation trees. """

    def __init__(self, n_iter=5, base_graph_kernel=VertexHistogram, normalize=True, to_grakel_graph=None):
        self.n_iter = int(n_iter)
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.to_grakel_graph = to_grakel_graph

    """
    @property
    def hyperparameter_n_iter(self):
        return Hyperparameter(
            "n_iter", "numeric", (1, self.n_iter * 2)
        )
    """

    def _f(self, t1: DerivationTree[NT, T, G], t2: DerivationTree[NT, T, G]) -> float:
        """
        kernel value between two derivation trees
        """
        if self.to_grakel_graph is None:
            g1 = t1.to_grakel_graph()
            g2 = t2.to_grakel_graph()
        else:
            g1 = self.to_grakel_graph(t1)
            g2 = self.to_grakel_graph(t2)
        wl_kernel = WeisfeilerLehman(
            n_iter=self.n_iter,
            base_graph_kernel=self.base_graph_kernel,
            normalize=self.normalize,
        )

        """ For some Grakel kernels there is a method pairwise_operation() that can be used to compute the kernel value
        between two graphs. However, the Weisfeiler Lehman kernel does not implement this method..."""

        """We employ the Weifeiler Lehman kernel and we first compute the kernel value between the graph representation 
        of the first derivation tree and itself."""
        wl_kernel.fit_transform(g1)

        """Then, we compute the kernel value between the graph representation of the second derivation tree 
        and the first one and return the result."""

        return wl_kernel.transform(g2)[0][0]


    def _g(self, s1, s2):
        """
        This should be the kernel derivative between a pair of derivation trees.
        I have no clou, whether this derivative exists.
        """
        raise NotImplementedError("How do you derive a Weisfeiler Lehman kernel?")

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        if eval_gradient:
            return (
                np.array([[self._f(x, y) for y in Y] for x in X]),
                np.array([[[self._g(x, y)] for y in Y] for x in X]),
            )
        else:
            #wl_kernel = WeisfeilerLehman(
            #    n_iter= self.n_iter,
            #    base_graph_kernel=self.base_graph_kernel,
            #    normalize=self.normalize,
            #)
            # gs = [x.to_grakel_graph() for x in X]
            # return wl_kernel.fit_transform(gs)
            return np.array([[self._f(x, y) for y in Y] for x in X])
            #nxs = [tree.to_indexed_nx_digraph()[0].to_undirected() for tree in X]
            #GS = graph_from_networkx(nxs, node_labels_tag='symbol', edge_labels_tag='argument_type')
            #return wl_kernel.fit_transform(GS)

    def is_stationary(self):
        return False

    def clone_with_theta(self, theta):
        cloned = clone(self)
        cloned.theta = theta
        return cloned
