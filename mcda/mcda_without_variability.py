import logging
import copy
import numpy as np
import pandas as pd

from mcda.configuration.config import Config
from mcda.utility_functions.normalization import Normalization
from mcda.utility_functions.aggregation import Aggregation

class MCDAWithoutVar():
    """
    Class MCDA without variability

    This class allows one to run MCDA without considering the uncertainties related to the indicators.
    All indicators are associated to the exact type of marginal distribution.

    :input:  configuration dictionary
             input_matrix with no alternatives column
    :output:

    """

    def __init__(self, config: dict, input_matrix: pd.DataFrame()):

        self._config = copy.deepcopy(config)
        self._input_matrix = copy.deepcopy(input_matrix)

    def normalize_indicators(self) -> dict:
        """
        Get the input matrix.
        :return: a df that concatenates the normalized values of each indicator per normalization method.
        Normalization functions implemented: minmax; target; standardized; rank
        """
        config = Config(self._config)
        norm = Normalization(self._input_matrix)

        indicators_scaled_minmax_01 = norm.minmax(config.polarity_for_each_indicator, feature_range=(0, 1))
        indicators_scaled_minmax_no0 = norm.minmax(config.polarity_for_each_indicator, feature_range=(0.1, 1)) # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_target_01 = norm.target(config.polarity_for_each_indicator, feature_range=(0, 1))
        indicators_scaled_target_no0 = norm.target(config.polarity_for_each_indicator, feature_range=(0.1, 1)) # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_standardized = norm.standardized(config.polarity_for_each_indicator)
        indicators_scaled_rank = norm.rank(config.polarity_for_each_indicator)

        normalized_indicators = {"standardized": indicators_scaled_standardized,
                                 "minmax_01":  indicators_scaled_minmax_01,
                                 "minmax_no0": indicators_scaled_minmax_no0,
                                 "target_01":  indicators_scaled_target_01,
                                 "target_no0": indicators_scaled_target_no0,
                                 "rank":  indicators_scaled_rank
                                 }

        return normalized_indicators

    def aggregate_indicators(self, normalized_indicators, weights) -> pd.DataFrame():
        """
        Get the normalized indicators per normalization methods in a dictionary.
        :return: aggregated scores per each alternative, and per each normalization method.
        Aggregation functions implemented: weighted-sum;
        """

        self.normalized_indicators = normalized_indicators
        self.weights = weights

        agg = Aggregation(self.weights)
        scores_weighted_sum_minmax = agg.weighted_sum(self.normalized_indicators["minmax"])

        # vector of scores (lenght = no. alternatives) estimated by the weighted-sum method, per each different normalization functions

        #return scores