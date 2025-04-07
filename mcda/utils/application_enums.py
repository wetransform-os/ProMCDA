"""
This script contains all the Enumeration classes corresponding to different types of
configuration object. So that it is easier to access keys across the application.
"""

from enum import Enum


class RobustnessAnalysis(Enum):
    """
    Enum for what kind of robustness analysis is performed
    """
    NONE = "none"
    INDICATORS = "indicators"
    WEIGHTS = "weights"


class RobustnessWeightLevels(Enum):
    """
    Enum for where uncertainties are introduced
    """
    NONE = "none"
    ALL = "all"
    SINGLE = "single"


class SensitivityAnalysis(Enum):
    """
    Enum for if sensitivity analysis is performed
    """
    YES = "yes"
    NO = "no"


class SensitivityNormalization(Enum):
    """
    Enum describing what functions are used for normalizing the indicator values
    """
    MINMAX = "minmax"
    STANDARDIZED = "standardized"
    TARGET = "target"
    RANK = "RANK"


class SensitivityAggregation(Enum):
    """
    Enum to describe aggregating the indicator values
    """
    WEIGHTED_SUM = "weighted_sum"
    GEOMETRIC = "geometric"
    HARMONIC = "harmonic"
    MINIMUM = "minimum"


class MonteCarloMarginalDistributions(Enum):
    """
    Enum for Marginal distribution describing the indicators
    """
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    EXACT = "exact"
    POISSON = "poisson"
    UNIFORM = "uniform"
