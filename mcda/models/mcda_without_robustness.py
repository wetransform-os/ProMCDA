import sys
import copy
import logging
import pandas as pd

from mcda.configuration.enums import NormalizationFunctions, OutputColumNames4Sensitivity, \
    NormalizationNames4Sensitivity, AggregationFunctions
from mcda.mcda_functions.normalization import Normalization
from mcda.mcda_functions.aggregation import Aggregation

log = logging.getLogger(__name__)

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA aggregation")


class MCDAWithoutRobustness:
    """
    Class MCDA without indicators' uncertainty

    This class allows one to run MCDA without considering the uncertainties related to the indicators.
    All indicators are described by the exact marginal distribution.
    However, it's possible to have randomly sampled weights.
    """

    def __init__(self, config: dict, input_matrix: pd.DataFrame):
        self.normalized_indicators = None
        self.weights = None
        self._config = copy.deepcopy(config)
        self._input_matrix = copy.deepcopy(input_matrix)

    def normalize_indicators(self, method=None) -> dict:
        """
        Normalize the input matrix using the specified normalization method.

        Parameters:
        - method (optional): the normalization method to use. If None, all available methods will be applied.
          Supported methods: 'minmax', 'target', 'standardized', 'rank'.

        Returns:
        - a dictionary containing the normalized values of each indicator per normalization method.

        Notes:
        Some aggregation methods do not work with indicator values equal or smaller than zero. For that reason:
        - for the 'minmax' method, two sets of normalized indicators are returned: one with the range (0, 1) and
          another with the range (0.1, 1).
        - for the 'target' method, two sets of normalized indicators are returned: one with the range (0, 1) and
          another with the range (0.1, 1).
        - for the 'standardized' method, two sets of normalized indicators are returned: one with the range (-inf, +inf)
          and another with the range (0.1, +inf).
        """
        norm = Normalization(self._input_matrix,
                             self._config["polarity"])

        normalized_indicators = {}

        if isinstance(method, NormalizationFunctions):
            method = method.value
        if method is None or method == NormalizationFunctions.MINMAX.value:
            indicators_scaled_minmax_01 = norm.minmax(feature_range=(0, 1))
            # for aggregation "geometric" and "harmonic" that do not accept 0
            indicators_scaled_minmax_without_zero = norm.minmax(feature_range=(0.1, 1))
            normalized_indicators[NormalizationNames4Sensitivity.MINMAX_WITHOUT_ZERO.value] = \
                indicators_scaled_minmax_without_zero
            normalized_indicators[NormalizationNames4Sensitivity.MINMAX_01.value] = indicators_scaled_minmax_01
        if method is None or method == NormalizationFunctions.TARGET.value:
            indicators_scaled_target_01 = norm.target(feature_range=(0, 1))
            indicators_scaled_target_without_zero = norm.target(
                feature_range=(0.1, 1))  # for aggregation "geometric" and "harmonic" that do not accept 0
            normalized_indicators[NormalizationNames4Sensitivity.TARGET_WITHOUT_ZERO.value] = \
                indicators_scaled_target_without_zero
            normalized_indicators[NormalizationNames4Sensitivity.TARGET_01.value] = indicators_scaled_target_01
        if method is None or method == NormalizationFunctions.STANDARDIZED.value:
            indicators_scaled_standardized_any = norm.standardized(
                feature_range=('-inf', '+inf'))
            indicators_scaled_standardized_without_zero = norm.standardized(
                feature_range=(0.1, '+inf'))
            normalized_indicators[NormalizationNames4Sensitivity.STANDARDIZED_ANY.value] = indicators_scaled_standardized_any
            normalized_indicators[NormalizationNames4Sensitivity.STANDARDIZED_WITHOUT_ZERO.value] = indicators_scaled_standardized_without_zero
        if method is None or method == NormalizationFunctions.RANK.value:
            indicators_scaled_rank = norm.rank()
            normalized_indicators[NormalizationNames4Sensitivity.RANK.value] = indicators_scaled_rank
        if isinstance(method, NormalizationFunctions):
            method = method.value
        if method is not None and method not in [method.value for method in NormalizationFunctions]:
            logger.error('Error Message', stack_info=True)
            raise ValueError(
                'The selected normalization method is not supported')

        return normalized_indicators

    def aggregate_indicators(self, normalized_indicators: dict, weights: list, method=None) -> pd.DataFrame:
        """
        Aggregate the normalized indicators using the specified aggregation method.

        Parameters:
        - normalized_indicators: a dictionary containing the normalized values of each indicator per normalization
          method.
        - weights: the weights to be applied during aggregation.
        - method (optional): The aggregation method to use. If None, all available methods will be applied.
        Supported methods: 'weighted_sum', 'geometric', 'harmonic', 'minimum'.

        Returns:
        - a DataFrame containing the aggregated scores per each alternative, and per each normalization method.

        :param normalized_indicators: dict
        :param weights: list
        :param method: str
        :return scores: pd.DataFrame
        """
        self.normalized_indicators = normalized_indicators
        self.weights = weights

        agg = Aggregation(self.weights)

        scores_weighted_sum = {}
        scores_geometric = {}
        scores_harmonic = {}
        scores_minimum = {}

        scores = pd.DataFrame()
        col_names_method = []
        col_names = [method.value for method in OutputColumNames4Sensitivity]
        # column names has the same order as in the following loop

        for key, values in self.normalized_indicators.items():
            if isinstance(method, AggregationFunctions):
                method = method.value
            if method is None or method == AggregationFunctions.WEIGHTED_SUM.value:
                if key in [NormalizationNames4Sensitivity.STANDARDIZED_ANY.value,
                           NormalizationNames4Sensitivity.MINMAX_01.value,
                           NormalizationNames4Sensitivity.TARGET_01.value, NormalizationNames4Sensitivity.RANK.value]:
                           # ws goes only with some specific normalizations
                    scores_weighted_sum[key] = agg.weighted_sum(values)
                    col_names_method.append("ws-" + key)
            if method is None or method == AggregationFunctions.GEOMETRIC.value:
                if key in [NormalizationNames4Sensitivity.STANDARDIZED_WITHOUT_ZERO.value,
                           NormalizationNames4Sensitivity.MINMAX_WITHOUT_ZERO.value,
                           NormalizationNames4Sensitivity.TARGET_WITHOUT_ZERO.value,
                           NormalizationNames4Sensitivity.RANK.value]:  # geom goes only with some specific normalizations
                    scores_geometric[key] = pd.Series(agg.geometric(values))
                    col_names_method.append("geom-" + key)
            if method is None or method == AggregationFunctions.HARMONIC.value:
                if key in [NormalizationNames4Sensitivity.STANDARDIZED_WITHOUT_ZERO.value,
                           NormalizationNames4Sensitivity.MINMAX_WITHOUT_ZERO.value,
                           NormalizationNames4Sensitivity.TARGET_WITHOUT_ZERO.value,
                           NormalizationNames4Sensitivity.RANK.value]:  # harm goes only with some specific normalizations
                    scores_harmonic[key] = pd.Series(agg.harmonic(values))
                    col_names_method.append("harm-" + key)
            if method is None or method == AggregationFunctions.MINIMUM.value:
                if key == NormalizationNames4Sensitivity.STANDARDIZED_ANY.value:
                    scores_minimum[key] = pd.Series(agg.minimum(
                        self.normalized_indicators[NormalizationNames4Sensitivity.STANDARDIZED_ANY.value]))
                    col_names_method.append("min-" + key)

        dict_list = [scores_weighted_sum, scores_geometric,
                     scores_harmonic, scores_minimum]

        for d in dict_list:
            if d:
                scores = pd.concat([scores, pd.DataFrame.from_dict(d)], axis=1)

        if method is None:
            scores.columns = col_names
        else:
            scores.columns = col_names_method

        return scores

