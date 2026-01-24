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
                 kernel_optimizer = None,
                 n_restarts_optimizer: int = 0,
                 population_size: int = 200, tournament_size: int = 10, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.3,
                 generation_limit: int = 50, elitism: int = 1,
                 enforce_diversity: bool = False
                 ):
        self.search_space = search_space
        self.request = request
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.kernel_optimizer = kernel_optimizer
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation_limit = generation_limit
        self.elitism = elitism
        self.enforce_diversity = enforce_diversity

    def bayesian_optimisation(self, n_iters, obj_fun, x0=None, y0=None,
                              n_pre_samples=10,
                              gp_params=None, alpha=1e-10, greater_is_better: bool = False, ei_xi=0.01):
        """ bayesian_optimisation

        returns a dictionary with keys:
        'best_tree': the best derivation tree found
        'x': all sampled derivation trees
        'y': all objective function values
        'gp_model': the final GP model
        """

        x_list = []
        y_list = []

        if x0 is None:
            # Sample n_pre_samples initial points from grammar, without duplicates under the kernel
            next = self.search_space.sample_tree(self.request)
            x0 = []
            while len(x0) < n_pre_samples:
                is_duplicate = False
                for tree in x0:
                    k = self.kernel._f(next, tree)
                    if k > 0.99:  # almost identical
                        is_duplicate = True
                        break
                if not is_duplicate:
                    x0.append(next)
                next = self.search_space.sample_tree(self.request)

            #x0 = list(self.search_space.sample(n_pre_samples, self.request))
            for tree in x0:
                x_list.append(tree)
                y_list.append(obj_fun(tree))
        x_list = list(x0)
        y_list = list(y0) if y0 is not None else y_list
        x_size = len(x_list)
        if x_size != len(y_list):
            raise ValueError("The length of x0 and y0 must be the same.")
        """
        Make sure we have at least n_pre_samples samples before starting the BO loop    
        """
        while x_size < n_pre_samples:
            next = self.search_space.sample_tree(self.request)
            x1 = []
            while len(x1) < (n_pre_samples - x_size):
                is_duplicate = False
                for tree in x1:
                    k = self.kernel._f(next, tree)
                    if k > 0.99:  # almost identical
                        is_duplicate = True
                        break
                if not is_duplicate:
                    x1.append(next)
                next = self.search_space.sample_tree(self.request)
            #x1 = list(self.search_space.sample(n_pre_samples - x_size, self.request))
            for tree in x1:
                x_list.append(tree)
                y_list.append(obj_fun(tree))
            x_size = len(x_list)

        xp = np.array(x_list)
        yp = np.array(y_list)

        # Create the GP
        if gp_params is not None:
            model = GaussianProcessRegressor(**gp_params)
        else:
            model = GaussianProcessRegressor(kernel=self.kernel,
                                             alpha=alpha,
                                             n_restarts_optimizer=self.n_restarts_optimizer,
                                             optimizer=self.kernel_optimizer,  # we currently need this, to prevent derivation of the kernel
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
                                                                    greater_is_better=True,  # always maximize EI
                                                                    enforce_diversity=self.enforce_diversity,
                                                                    elitism=self.elitism)

            next_sample = optimizer()
            print(f"next_sample in x_list: {next_sample in x_list}")

            def identical(t1, t2):
                k = self.kernel._f(t1, t2)
                return k > 0.99  # almost identical

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            while any(identical(next_sample, t) for t in x_list):
                # print("Duplicate detected. Sampling randomly.")
                next_sample = self.search_space.sample_tree(self.request)

            # objective function evaluation for new derivation tree
            cv_score = obj_fun(next_sample)
            print(f"EI(next_sample): {acquisition_function(next_sample)}")
            print(f"f_obj(next_sample): {cv_score}")

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

        result = {"best_tree": best_tree, "x": xp, "y": yp, "gp_model": model}
        return result

    def verbose_bayesian_optimisation(self, n_iters, verbose_obj_fun, x0=None,
                              n_pre_samples=10,
                              gp_params=None, alpha=1e-10, greater_is_better: bool = False, ei_xi=0.01):
        """ bayesian_optimisation
        for verbose objective functions, meaning that the objective function returns a dictionary with at least
        the key 'score' for the actual objective value

        The verbose bayesian optimization can't allow a warm start, because it needs to gather the verbose information
        of the obj_function calls.
        But you are allowed to "seed" the initial sample with x0.
        """

        x_list = []
        y_list = []
        verbose_list = []

        if x0 is None:
            # Sample n_pre_samples initial points from grammar
            x0 = list(self.search_space.sample(n_pre_samples, self.request))
        for tree in x0:
            x_list.append(tree)
            y_list.append(verbose_obj_fun(tree)["score"])
            verbose_list.append(verbose_obj_fun(tree))

        x_size = len(x_list)
        if x_size != len(y_list):
            raise ValueError("The length of x0 and y0 must be the same.")
        """
        Make sure we have at least n_pre_samples samples before starting the BO loop    
        """
        while x_size < n_pre_samples:
            x1 = list(self.search_space.sample(n_pre_samples - x_size, self.request))
            for tree in x1:
                x_list.append(tree)
                y_list.append(verbose_obj_fun(tree)["score"])
                verbose_list.append(verbose_obj_fun(tree))
            x_size = len(x_list)

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
                                                                    greater_is_better=True,  # always maximize EI
                                                                    enforce_diversity=self.enforce_diversity,
                                                                    elitism=self.elitism)

            next_sample = optimizer()
            print(f"next_sample in x_list: {next_sample in x_list}")

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            while next_sample in x_list:
                # print("Duplicate detected. Sampling randomly.")
                next_sample = self.search_space.sample_tree(self.request)

            # objective function evaluation for new derivation tree
            cv_score = verbose_obj_fun(next_sample)["score"]
            print(f"EI(next_sample): {acquisition_function(next_sample)}")
            print(f"verbose_f_obj(next_sample): {cv_score}")

            # Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)
            verbose_list.append(verbose_obj_fun(next_sample))

            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)

        if greater_is_better:
            best_tree = xp[np.argmax(yp)]
        else:
            best_tree = xp[np.argmin(yp)]

        result = {"best_tree": best_tree, "x": xp, "y": yp, "gp_model": model, "verbose_calls": verbose_list}
        return result


