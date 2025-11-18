import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.stats import norm

from collections.abc import Hashable

from typing import Generic, TypeVar

from src.cl3s.genetic_programming.evolutionary_search import TournamentSelection
from src.cl3s.search_space import SearchSpace

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names

from ..tree import DerivationTree


class AcquisitionFunction(Generic[NT, T, G]):
    def __init__(self,  gp: GaussianProcessRegressor, greater_is_better: bool = False):
        """
        Arguments:
        ----------
            t: DerivationTree.
                The derivation tree for which the expected improvement needs to be computed.
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated DerivationTrees.
            greater_is_better: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
        """

        self.gp = gp
        self.greater_is_better = greater_is_better

    def __call__(self, t: DerivationTree[NT, T, G]):
        raise NotImplementedError("Subclasses must implement this method.")


class ExpectedImprovement(AcquisitionFunction[NT, T, G]):
    def __call__(self, t: DerivationTree[NT, T, G]):
        mu, sigma = self.gp.predict([t], return_std=True)
        y = self.gp.y_train_

        if self.greater_is_better:
            loss_optimum = np.max(y)
        else:
            loss_optimum = np.min(y)

        scaling_factor = (-1) ** (not self.greater_is_better)

        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            ei = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] == 0.0

        return -1 * ei


class AcquisitionFunctionOptimization(Generic[NT, T, G]):

    def __init__(self, search_space: SearchSpace[NT, T, G], request: NT, acquisition_function: AcquisitionFunction[NT, T, G],
                 greater_is_better: bool = False):
        self.search_space = search_space
        self.acquisition_function = acquisition_function
        self.greater_is_better = greater_is_better
        self.request = request

    def __call__(self):
        raise NotImplementedError("Subclasses must implement this method.")


class EvolutionaryAcquisitionFunctionOptimization(AcquisitionFunctionOptimization[NT, T, G]):

    def __init__(self, search_space: SearchSpace[NT, T, G], request: NT, acquisition_function: AcquisitionFunction[NT, T, G],
                 population_size=10, reproduction_rate=0.2, generation_limit=5, greater_is_better: bool = False):
        super().__init__(search_space, request, acquisition_function, greater_is_better)
        self.evolutionary_search = TournamentSelection(search_space, request, acquisition_function, population_size,
                                                       reproduction_rate, generation_limit)

    def __call__(self):
        return self.evolutionary_search.optimize()
