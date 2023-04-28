import logging
import numpy as np
import pandas as pd

from mcda.utility_functions.normalization import Normalization

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

        config = Config(self._config)

    def normalize_indicators(self) -> pd.DataFrame():
        """
        Get the input matrix.
        :return: normalized values of each indicator.
        """
        norm = Normalization(self._input_matrix)

        indicators_scaled_minmax_01 = norm.minmax(config.polarity_for_each_indicator, feature_range=(0, 1))
        indicators_scaled_minmax_no0 = norm.minmax(config.polarity_for_each_indicator, feature_range=(0.1, 1)) # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_target = norm.target(config.polarity_for_each_indicator)

        return indicators_scaled_minmax