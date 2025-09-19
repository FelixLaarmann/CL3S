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

from typing import Any, Generic, Optional, TypeVar, Union, Generator
import typing

from ..tree import DerivationTree

from graph_kernel import GraphKernel, WeisfeilerLehmanKernel
from gaussian_process import TreeGP

NT = TypeVar("NT", bound=Hashable) # type of non-terminals
T = TypeVar("T", bound=Hashable) # type of terminals
G = TypeVar("G", bound=Hashable)  # type of constants/literal group names