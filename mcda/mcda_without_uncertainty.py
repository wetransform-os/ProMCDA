import sys
import copy
import logging
import pandas as pd

from mcda.configuration.config import Config
from mcda.utility_functions.normalization import Normalization
from mcda.utility_functions.aggregation import Aggregation

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA aggregation")


class MCDAWithoutUncertainty():
    """
    Class MCDA without indicators' uncertainty

    This class allows one to run MCDA without considering the uncertainties related to the indicators.
    All indicators are associated to the exact type of marginal distribution.
    However, it's possible to have randomly sampled weights.

    :input:  configuration dictionary
             input_matrix with no alternatives column
    :output: write csv files with the scores, normalized scores and ranks; log file

    """

    def __init__(self, config: Config, input_matrix: pd.DataFrame()):
        self._config = copy.deepcopy(config)
        self._input_matrix = copy.deepcopy(input_matrix)

    def normalize_indicators(self, method=None) -> dict:
        """
        Get the input matrix.
        :param method: The normalization method to use (None for all methods).
        :return: a dictionary that contains the normalized values of each indicator per normalization method.
        Normalization functions implemented: minmax; target; standardized; rank
        """
        norm = Normalization(self._input_matrix, self._config.polarity_for_each_indicator)

        normalized_indicators = {}

        if method is None or method == 'minmax':
            indicators_scaled_minmax_01 = norm.minmax(feature_range=(0, 1))
            indicators_scaled_minmax_no0 = norm.minmax(
                feature_range=(0.1, 1))  # for aggregation "geometric" and "harmonic" that accept no 0
            normalized_indicators["minmax_no0"] = indicators_scaled_minmax_no0
            normalized_indicators["minmax_01"] = indicators_scaled_minmax_01
        if method is None or method == 'target':
            indicators_scaled_target_01 = norm.target(feature_range=(0, 1))
            indicators_scaled_target_no0 = norm.target(
                feature_range=(0.1, 1))  # for aggregation "geometric" and "harmonic" that accept no 0
            normalized_indicators["target_no0"] = indicators_scaled_target_no0
            normalized_indicators["target_01"] = indicators_scaled_target_01
        if method is None or method == 'standardized':
            indicators_scaled_standardized_any = norm.standardized(feature_range=('-inf', '+inf'))
            indicators_scaled_standardized_no0 = norm.standardized(feature_range=(0.1, '+inf'))
            normalized_indicators["standardized_any"] = indicators_scaled_standardized_any
            normalized_indicators["standardized_no0"] = indicators_scaled_standardized_no0
        if method is None or method == 'rank':
            indicators_scaled_rank = norm.rank()
            normalized_indicators["rank"] = indicators_scaled_rank

        return normalized_indicators

    def aggregate_indicators(self, normalized_indicators: dict, weights: list, method=None) -> pd.DataFrame():
        #     """
        #     Get the normalized indicators per normalization methods in a dictionary.
        #     :param: method The normalization method to use (None for all methods).
        #     :return: aggregated scores per each alternative, and per each normalization method.
        #     Aggregation functions implemented: weighted-sum; geometric; harmonic; minimum
        #     """

        self.normalized_indicators = normalized_indicators
        self.weights = weights

        agg = Aggregation(self.weights)

        scores_weighted_sum = {}
        scores_geometric = {}
        scores_harmonic = {}
        scores_minimum = {}

        scores = pd.DataFrame()
        col_names_method_none = []
        col_names = ['ws-minmax_01', 'ws-target_01', 'ws-standardized_any', 'ws-rank',
                     'geom-minmax_no0', 'geom-target_no0', 'geom-standardized_no0', 'geom-rank',
                     'harm-minmax_no0', 'harm-target_no0', 'harm-standardized_no0', 'harm-rank'
                     'min-standardized_any'] # same order as in the following loop

        for key, values in self.normalized_indicators.items():
            if method is None or method == 'weighted_sum':
                if key in ["standardized_any", "minmax_01", "target_01",
                           "rank"]:  # ws goes only with some specific normalizations
                    scores_weighted_sum[key] = agg.weighted_sum(values)
                    col_names_method_none.append("ws-" + key)
            if method is None or method == 'geometric':
                if key in ["standardized_no0", "minmax_no0", "target_no0",
                           "rank"]:  # geom goes only with some specific normalizations
                    scores_geometric[key] = pd.Series(agg.geometric(values))
                    col_names_method_none.append("geom-" + key)
            if method is None or method == 'harmonic':
                if key in ["standardized_no0", "minmax_no0", "target_no0",
                           "rank"]:  # harm goes only with some specific normalizations
                    scores_harmonic[key] = pd.Series(agg.harmonic(values))
                    col_names_method_none.append("harm-" + key)
            if method is None or method == 'minimum':
                if key == "standardized_any":
                    scores_minimum[key] = pd.Series(agg.minimum(self.normalized_indicators["standardized_any"]))
                    col_names_method_none.append("min-" + key)
                else:
                    logger.info('The aggregation function minimum can be paired with standardized normalization only '
                                '(here missing)', stack_info=True)

        dict_list = [scores_weighted_sum, scores_geometric, scores_harmonic, scores_minimum]

        for d in dict_list:
            if d:
                scores = pd.concat([scores, pd.DataFrame.from_dict(d)], axis=1)

        if method is None:
            scores.columns = col_names
        else:
            scores.columns = col_names_method_none

        return scores
