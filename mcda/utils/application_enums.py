from enum import Enum


class RobustnessAnalysis(Enum):
    NONE = "none"
    INDICATORS = "indicators"
    WEIGHTS = "weights"


class RobustnessWightLevels(Enum):
    NONE = "none"
    ALL = "all"
    SINGLE = "single"


class SensitivityAnalysis(Enum):
    YES = "yes"
    NO = "no"


class SensitivityNormalization(Enum):
    MINMAX = "minmax"
    STANDARDIZED = "standardized"
    TARGET = "target"
    rank = "RANK"


class SensitivityAggregation(Enum):
    WEIGHTED_SUM = "weighted_sum"
    GEOMETRIC = "geometric"
    HARMONIC = "harmonic"
    MINIMUM = "minimum"


class MonteCarloMarginalDistributions(Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    EXACT = "exact"
    POISSON = "poisson"
    UNIFORM = "uniform"