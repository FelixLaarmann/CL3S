import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import GenericKernelMixin

from collections.abc import Callable, Hashable, Sequence

from typing import Any, Generic, Optional, TypeVar, Union, Generator
import typing

from cl3s.scikit.graph_kernel import WeisfeilerLehmanKernel

from acquisition_function import AcquisitionFunction, AcquisitionFunctionOptimization, ExpectedImprovement

from ..search_space import SearchSpace
from ..tree import DerivationTree

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names


class BayesianOptimization(Generic[NT, T, G]):

    def __init__(self, grammar: SearchSpace[NT, T, G], request: NT,
                 kernel: GenericKernelMixin = WeisfeilerLehmanKernel()):
        self.search_space = grammar
        self.request = request
        self.kernel = WeisfeilerLehmanKernel()

    def bayesian_optimisation(self, n_iters, obj_fun: Callable[DerivationTree[NT, T, G], float], x0=None, n_pre_samples=5,
                              gp_params=None, alpha=1e-5, epsilon=1e-7, greater_is_better: bool = False):
        """ bayesian_optimisation

        Uses Gaussian Processes to optimise the loss function `sample_loss`.

        Arguments:
        ----------
            n_iters: integer.
                Number of iterations to run the search algorithm.
            sample_loss: function.
                Function to be optimised.
            x0: array-like, shape = [n_pre_samples, n_params].
                Array of initial points to sample the loss function for. If None, randomly
                samples from the loss function.
            n_pre_samples: integer.
                If x0 is None, samples `n_pre_samples` initial points from the loss function.
            gp_params: dictionary.
                Dictionary of parameters to pass on to the underlying Gaussian Process.
            alpha: double.
                Variance of the error term of the GP.
            epsilon: double.
                Precision tolerance for floats.
        """

        x_list = []
        y_list = []

        if x0 is None:
            # Sample n_pre_samples initial points from grammar
            raise NotImplementedError()
        else:
            for tree in x0:
                x_list.append(tree)
                y_list.append(obj_fun(tree))

        xp = np.array(x_list)
        yp = np.array(y_list)

        # Create the GP
        if gp_params is not None:
            model = GaussianProcessRegressor(**gp_params)
        else:
            kernel = self.kernel
            model = GaussianProcessRegressor(kernel=kernel,
                                             alpha=alpha,
                                             # n_restarts_optimizer=10,
                                             normalize_y=True)

        for n in range(n_iters):

            model.fit(xp, yp)

            # Acquisition function for current GP
            acquisition_function = ExpectedImprovement(model, greater_is_better)

            # Sample next tree
            next_sample = AcquisitionFunctionOptimization.evolutionary_acquisition_function_optimization()

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            while next_sample in x_list:
                # print("Duplicate detected. Sampling randomly.")
                next_sample = self.search_space.sample_tree(self.request)

            # objective function evaluation for new derivation tree
            cv_score = obj_fun(next_sample)

            # Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)

            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)

        return xp, yp

