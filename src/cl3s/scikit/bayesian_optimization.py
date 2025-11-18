import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import GenericKernelMixin

from collections.abc import Hashable

from typing import Generic, TypeVar

from src.cl3s.scikit.graph_kernel import WeisfeilerLehmanKernel

from .acquisition_function import ExpectedImprovement, EvolutionaryAcquisitionFunctionOptimization

from ..search_space import SearchSpace

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names


class BayesianOptimization(Generic[NT, T, G]):

    def __init__(self, search_space: SearchSpace[NT, T, G], request: NT,
                 kernel: GenericKernelMixin = WeisfeilerLehmanKernel()):
        self.search_space = search_space
        self.request = request
        self.kernel = kernel

    def bayesian_optimisation(self, n_iters, obj_fun, x0=None,
                              n_pre_samples=10,
                              gp_params=None, alpha=1e-10, greater_is_better: bool = False):
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
            x0 = list(self.search_space.sample(n_pre_samples, self.request))
        for tree in x0:
            x_list.append(tree)
            y_list.append(obj_fun(tree))

        xp = np.array(x_list)
        yp = np.array(y_list)

        # Create the GP
        if gp_params is not None:
            model = GaussianProcessRegressor(**gp_params)
        else:
            model = GaussianProcessRegressor(kernel=self.kernel,
                                             alpha=alpha,
                                             # n_restarts_optimizer=10,
                                             optimizer=None,  # we currently need this, to prevent derivation of the kernel
                                             normalize_y=True)

        for n in range(n_iters):

            print("iteration:", n)

            model.fit(xp, yp)

            # Acquisition function for current GP
            acquisition_function = ExpectedImprovement(model, greater_is_better)

            # Sample next tree
            optimizer = EvolutionaryAcquisitionFunctionOptimization(self.search_space,
                                                                    self.request,
                                                                    acquisition_function,
                                                                    greater_is_better=greater_is_better,
                                                                    population_size=50,
                                                                    reproduction_rate=0.1,
                                                                    generation_limit=10)

            next_sample = optimizer()
            print(next_sample in x_list)

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            while next_sample in x_list:
                # print("Duplicate detected. Sampling randomly.")
                next_sample = self.search_space.sample_tree(self.request)

            # objective function evaluation for new derivation tree
            cv_score = obj_fun(next_sample)
            print(cv_score)

            # Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)

            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)

        if greater_is_better:
            best_tree = xp[np.argmax(yp)]
        else:
            best_tree = xp[np.argmin(yp)]
        return best_tree, xp, yp

