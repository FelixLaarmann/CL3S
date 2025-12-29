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
            gaussian_process: GaussianProcessRegressor object.
                Gaussian process trained on previously evaluated DerivationTrees.
            greater_is_better: Boolean.
                Boolean flag that indicates whether the loss function is to be maximised or minimised.
        """

        self.gp = gp
        self.greater_is_better = greater_is_better

    def __call__(self, t: DerivationTree[NT, T, G]):
        raise NotImplementedError("Subclasses must implement this method.")


class SimplifiedExpectedImprovement(AcquisitionFunction[NT, T, G]):
    def __call__(self, t: DerivationTree[NT, T, G]):
        mu, sigma = self.gp.predict([t], return_std=True)
        y = self.gp.y_train_

        if self.greater_is_better:
            loss_optimum = np.max(y)
        else:
            loss_optimum = np.min(y)

        scaling_factor = (-1) ** (not self.greater_is_better)

        ei = scaling_factor * (mu + 1.96 * sigma)

        # In case sigma equals zero
        """
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (mu - loss_optimum) / sigma
            ei = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        """

        return ei.item()

class ExpectedImprovement(AcquisitionFunction[NT, T, G]):
    def __call__(self, t: DerivationTree[NT, T, G], xi=0.01):
        mu, sigma = self.gp.predict([t], return_std=True)
        y = self.gp.y_train_

        if self.greater_is_better:
            loss_optimum = np.max(y)
        else:
            loss_optimum = np.min(y)

        if t in self.gp.X_train_:
            return 0.0

        values = np.zeros_like(mu)
        mask = sigma > 0 #and sigma > 1.001e-5 # I don't know why, but sigma is not 0.0 for all trees in X_train_ ...
        improve = loss_optimum - xi - mu[mask]
        scaled = improve / sigma[mask]
        cdf = norm.cdf(scaled)
        pdf = norm.pdf(scaled)
        exploit = improve * cdf
        explore = sigma[mask] * pdf
        values[mask] = exploit + explore

        return values.item()


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
                 population_size: int = 200, tournament_size: int = 10, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.3,
                 generation_limit: int = 50, elitism: int = 1,
                 greater_is_better: bool = True, enforce_diversity: bool = False):
        super().__init__(search_space, request, acquisition_function, greater_is_better)
        self.evolutionary_search = TournamentSelection(search_space, request, acquisition_function,
                                                       population_size=population_size, crossover_rate=crossover_rate,
                                                       mutation_rate=mutation_rate, generation_limit=generation_limit,
                                                       tournament_size=tournament_size,
                                                       greater_is_better=greater_is_better,
                                                       enforce_diversity=enforce_diversity, elitism=elitism)


    def __call__(self):
        return self.evolutionary_search.optimize()
