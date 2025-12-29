import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import GenericKernelMixin

from collections.abc import Hashable

from typing import Generic, TypeVar

from .graph_kernel import WeisfeilerLehmanKernel

from .acquisition_function import ExpectedImprovement, EvolutionaryAcquisitionFunctionOptimization

from .search_space import SearchSpace

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names


class BayesianOptimization(Generic[NT, T, G]):

    def __init__(self, search_space: SearchSpace[NT, T, G], request: NT,
                 kernel: GenericKernelMixin = WeisfeilerLehmanKernel(),
                 population_size: int = 200, tournament_size: int = 10, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.3,
                 generation_limit: int = 50, elitism: int = 1,
                 enforce_diversity: bool = False
                 ):
        self.search_space = search_space
        self.request = request
        self.kernel = kernel
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation_limit = generation_limit
        self.elitism = elitism
        self.enforce_diversity = enforce_diversity

    def bayesian_optimisation(self, n_iters, obj_fun, x0=None,
                              n_pre_samples=10,
                              gp_params=None, alpha=1e-10, greater_is_better: bool = False):
        """ bayesian_optimisation

        from here: https://github.com/thuijskens/bayesian-optimization
        ???
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
                                             normalize_y=False)

        for n in range(n_iters):

            print("iteration:", n)

            model.fit(xp, yp)

            # Acquisition function for current GP
            acquisition_function = ExpectedImprovement(model, greater_is_better)

            # Sample next tree
            optimizer = EvolutionaryAcquisitionFunctionOptimization(self.search_space,
                                                                    self.request,
                                                                    acquisition_function,
                                                                    population_size=self.population_size,
                                                                    crossover_rate=self.crossover_rate,
                                                                    mutation_rate=self.mutation_rate,
                                                                    generation_limit=self.generation_limit,
                                                                    tournament_size=self.tournament_size,
                                                                    greater_is_better=greater_is_better,
                                                                    enforce_diversity=self.enforce_diversity,
                                                                    elitism=self.elitism)

            next_sample = optimizer()
            print(f"next_sample in x_list: {next_sample in x_list}")

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            while next_sample in x_list:
                # print("Duplicate detected. Sampling randomly.")
                next_sample = self.search_space.sample_tree(self.request)

            # objective function evaluation for new derivation tree
            cv_score = obj_fun(next_sample)
            print(f"acquisition: {acquisition_function(next_sample)}")
            print(f"cv_score: {cv_score}")

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

