import numpy as np
from scipy.linalg import cholesky

from sklearn.base import clone
from sklearn.gaussian_process.kernels import GenericKernelMixin, NormalizedKernelMixin, Hyperparameter, Kernel
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx

from collections.abc import Callable, Hashable, Sequence

from typing import Any, Generic, Optional, TypeVar, Union, Generator
import typing

from .tree import DerivationTree

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names

class WeisfeilerLehmanKernel(GenericKernelMixin, NormalizedKernelMixin, Kernel, Generic[NT, T, G]):
    """ Weisfeiler Lehman Kernel for derivation trees. """

    def __init__(self, n_iter=1, base_graph_kernel=VertexHistogram, normalize=True, to_grakel_graph=None, n_jobs=None):
        self.n_iter = 1 if n_iter < 1 else int(n_iter) if n_iter <= 3 else 3
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.to_grakel_graph = to_grakel_graph
        self.n_jobs = n_jobs


    @property
    def hyperparameter_n_iter(self):
        return Hyperparameter(
            "n_iter", "numeric", (1, 3)
        )

    def optimize_hyperparameter(self, obj_fun, initial_theta, bounds):
        theta_opt = initial_theta
        func_min = obj_fun(theta_opt, eval_gradient=False)
        for _ in range(10):
            for b in bounds:
                theta = np.array([np.random.uniform(b[0], b[1])])
                f_val = obj_fun(theta, eval_gradient=False)
                if f_val < func_min:
                    func_min = f_val
                    theta_opt = theta
        return theta_opt, func_min


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
            n_iter=int(self.n_iter) if self.n_iter >= 1 else int(1),
            base_graph_kernel=self.base_graph_kernel,
            normalize=self.normalize,
            n_jobs=self.n_jobs,
        )

        """ For some Grakel kernels there is a method pairwise_operation() that can be used to compute the kernel value
        between two graphs. However, the Weisfeiler Lehman kernel does not implement this method..."""

        """We employ the Weisfeiler Lehman kernel and we first compute the kernel value between the graph representation 
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

        K = np.array([[self._f(x, y) for y in Y] for x in X])

        if eval_gradient:
            # try to compute the inverse of K using cholesky decomposition for numerical stability
            try:
                K_i = cholesky(K + 1e-10 * np.eye(K.shape[0]), lower=True)
                K_inv = np.linalg.solve(K_i.T, np.linalg.solve(K_i, np.eye(K.shape[0])))
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
            expected_shape = K_inv.shape + (1,)
            return (
                K,
                K_inv.reshape(expected_shape),
            )
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def is_stationary(self):
        return True

    def clone_with_theta(self, theta):
        cloned = clone(self)
        cloned.theta = theta
        return cloned

class HierarchicalWeisfeilerLehmanKernel(GenericKernelMixin, NormalizedKernelMixin, Kernel, Generic[NT, T, G]):
    """ Weisfeiler Lehman Kernel for derivation trees. """

    def __init__(self, to_graph_list, weights, n_iters, base_graph_kernel=VertexHistogram, normalize=True, n_jobs=None):
        if len(to_graph_list) != len(weights) or len(to_graph_list) != len(n_iters):
            raise ValueError("The number of to_graph functions must be equal to the number of weights.")
        self.to_graphs = to_graph_list
        self.weights = weights
        self.n_iters = n_iters
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.n_jobs = n_jobs

    def _f(self, t1: DerivationTree[NT, T, G], t2: DerivationTree[NT, T, G]) -> float:
        """
        kernel value between two derivation trees
        """
        weighted_values = []
        for to_graph, w, i in zip(self.to_graphs, self.weights, self.n_iters):
            if to_graph is None:
                g1 = t1.to_grakel_graph()
                g2 = t2.to_grakel_graph()
            else:
                g1 = to_graph(t1)
                g2 = to_graph(t2)
            wl_kernel = WeisfeilerLehman(
                n_iter=i,
                base_graph_kernel=self.base_graph_kernel,
                normalize=self.normalize,
                n_jobs=self.n_jobs,
            )

            """ For some Grakel kernels there is a method pairwise_operation() that can be used to compute the kernel value
            between two graphs. However, the Weisfeiler Lehman kernel does not implement this method..."""

            """We employ the Weisfeiler Lehman kernel and we first compute the kernel value between the graph representation 
            of the first derivation tree and itself."""
            wl_kernel.fit_transform(g1)
            """Then, we compute the kernel value between the graph representation of the second derivation tree 
                and the first one and return the result."""
            v = wl_kernel.transform(g2)[0][0]
            weighted_v = w * v
            weighted_values.append(weighted_v)

        return sum(weighted_values)



    def _g(self, s1, s2):
        """
        This should be the kernel derivative between a pair of derivation trees.
        I have no clou, whether this derivative exists.
        """
        raise NotImplementedError("How do you derive a Graph kernel?")

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        if eval_gradient:
            return (
                np.array([[self._f(x, y) for y in Y] for x in X]),
                np.array([[[self._g(x, y)] for y in Y] for x in X]),
            )
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def is_stationary(self):
        return True

    def clone_with_theta(self, theta):
        cloned = clone(self)
        cloned.theta = theta
        return cloned


class OptimizableHierarchicalWeisfeilerLehmanKernel(GenericKernelMixin, NormalizedKernelMixin, Kernel, Generic[NT, T, G]):
    """ Weisfeiler Lehman Kernel for derivation trees. """

    def __init__(self, to_grakel_graph1, to_grakel_graph2, to_grakel_graph3,
                 weight1, weight2, weight3, n_iter1, n_iter2, n_iter3,
                 base_graph_kernel=VertexHistogram, normalize=True, n_jobs=None):
        self.to_grakel_graph1 = to_grakel_graph1
        self.to_grakel_graph2 = to_grakel_graph2
        self.to_grakel_graph3 = to_grakel_graph3
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.n_iter1 = n_iter1
        self.n_iter2 = n_iter2
        self.n_iter3 = n_iter3
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.n_jobs = n_jobs

    @property
    def hyperparameter_n_iter1(self):
        return Hyperparameter(
            "n_iter1", "numeric", (1, 3)
        )

    @property
    def hyperparameter_n_iter2(self):
        return Hyperparameter(
            "n_iter2", "numeric", (1, 3)
        )

    @property
    def hyperparameter_n_iter3(self):
        return Hyperparameter(
            "n_iter3", "numeric", (1, 3)
        )

    @property
    def hyperparameter_weight1(self):
        return Hyperparameter(
            "weight1", "numeric", (1, 100)
        )

    @property
    def hyperparameter_weight2(self):
        return Hyperparameter(
            "weight2", "numeric", (1, 100)
        )

    @property
    def hyperparameter_weight3(self):
        return Hyperparameter(
            "weight3", "numeric", (1, 100)
        )

    def optimize_hyperparameter(self, obj_fun, initial_theta, bounds):
        theta_opt = initial_theta
        func_min = obj_fun(theta_opt, eval_gradient=False)
        n_iter_bounds = bounds[:3]
        n_i = n_iter_bounds[0]
        weight_bounds = bounds[3:]
        n_w = weight_bounds[0]
        for _ in range(10):
            theta_i = np.random.uniform(n_i[0], n_i[1], 3)
            theta_w = np.random.uniform(n_w[0], n_w[1], 3)
            theta_w = np.array([x / sum(theta_w) * n_w[1] for x in theta_w])
            theta = np.concatenate((theta_i, theta_w), axis=None)
            f_val = obj_fun(theta, eval_gradient=False)
            if f_val < func_min:
                func_min = f_val
                theta_opt = theta
        return theta_opt, func_min

    def _f(self, t1: DerivationTree[NT, T, G], t2: DerivationTree[NT, T, G]) -> float:
        """
        kernel value between two derivation trees
        """
        weighted_values = []
        for to_graph, w, i in zip([self.to_grakel_graph1, self.to_grakel_graph2, self.to_grakel_graph3],
                                  [self.weight1, self.weight2, self.weight3],
                                  [self.n_iter1, self.n_iter2, self.n_iter3]):
            if to_graph is None:
                g1 = t1.to_grakel_graph()
                g2 = t2.to_grakel_graph()
            else:
                g1 = to_graph(t1)
                g2 = to_graph(t2)
            wl_kernel = WeisfeilerLehman(
                n_iter=int(i) if i >= 1 else int(1),
                base_graph_kernel=self.base_graph_kernel,
                normalize=self.normalize,
                n_jobs=self.n_jobs,
            )

            """ For some Grakel kernels there is a method pairwise_operation() that can be used to compute the kernel value
            between two graphs. However, the Weisfeiler Lehman kernel does not implement this method..."""

            """We employ the Weisfeiler Lehman kernel and we first compute the kernel value between the graph representation 
            of the first derivation tree and itself."""
            wl_kernel.fit_transform(g1)
            """Then, we compute the kernel value between the graph representation of the second derivation tree 
                and the first one and return the result."""
            v = wl_kernel.transform(g2)[0][0]
            weighted_v = (w / 100) * v
            weighted_values.append(weighted_v)

        return sum(weighted_values)



    def _g(self, s1, s2):
        """
        This should be the kernel derivative between a pair of derivation trees.
        I have no clou, whether this derivative exists.
        """
        raise NotImplementedError("How do you derive a Graph kernel?")

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        K = np.array([[self._f(x, y) for y in Y] for x in X])

        if eval_gradient:
            # try to compute the inverse of K using cholesky decomposition for numerical stability
            try:
                K_i = cholesky(K + 1e-10 * np.eye(K.shape[0]), lower=True)
                K_inv = np.linalg.solve(K_i.T, np.linalg.solve(K_i, np.eye(K.shape[0])))
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
            expected_shape = K_inv.shape + (1,)
            return (
                K,
                K_inv.reshape(expected_shape),
            )
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def is_stationary(self):
        return True

    def clone_with_theta(self, theta):
        cloned = clone(self)
        cloned.theta = theta
        return cloned
