from enum import Enum

"""
This module defines enumerations for use throughout the package to enhance maintainability.

Enumerations (enums) provide a way to define a set of named values, which can be used to represent options or
categories in a more manageable manner, avoiding string literals or hard-coded values.
"""

class NormalizationFunctions(Enum):
    """
    Implemented normalization functions
    """
    MINMAX = 'minmax'
    STANDARDIZED = 'standardized'
    TARGET = 'target'
    RANK = 'rank'


class AggregationFunctions(Enum):
    """
    Implemented aggregation functions
    """
    WEIGHTED_SUM = 'weighted_sum'
    GEOMETRIC = 'geometric'
    HARMONIC = 'harmonic'
    MINIMUM = 'minimum'


class NormalizationNames4Sensitivity(Enum):
    """
    Names of normalization functions in case of sensitivity analysis
    """
    MINMAX_WITHOUT_ZERO = 'minmax_without_zero'
    MINMAX_01 = 'minmax_01'
    TARGET_WITHOUT_ZERO = 'target_without_zero'
    TARGET_01 = 'target_01'
    STANDARDIZED_ANY = 'standardized_any'
    STANDARDIZED_WITHOUT_ZERO = 'standardized_without_zero'
    RANK = 'rank'


class OutputColumnNames4Sensitivity(Enum):
    """
    Names of output columns in case of sensitivity analysis
    aggregation name + normalization name
    """
    WS_MINMAX_01 = 'ws-minmax_01'
    WS_TARGET_01 = 'ws-target_01'
    WS_STANDARDIZED_ANY = 'ws-standardized_any'
    WS_RANK = 'ws-rank'
    GEOM_MINMAX_WITHOUT_ZERO = 'geom-minmax_without_zero'
    GEOM_TARGET_WITHOUT_ZERO = 'geom-target_without_zero'
    GEOM_STANDARDIZED_WITHOUT_ZERO = 'geom-standardized_without_zero'
    GEOM_RANK = 'geom-rank'
    HARM_MINMAX_WITHOUT_ZERO = 'harm-minmax_without_zero'
    HARM_TARGET_WITHOUT_ZERO = 'harm-target_without_zero'
    HARM_STANDARDIZED_WITHOUT_ZERO = 'harm-standardized_without_zero'
    HARM_RANK = 'harm-rank'
    MIN_STANDARDIZED_ANY = 'min-standardized_any'


class PDFType(Enum):
    """
    Names of probability density functions, which describe the indicators in case of robustness analysis
    """
    EXACT = "exact"
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    POISSON = "poisson"


class RobustnessAnalysisType(Enum):
    NONE = "none"
    ALL_WEIGHTS = "all_weights"
    SINGLE_WEIGHTS = "single_weights"
    INDICATORS = "indicators"




