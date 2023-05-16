import copy
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
    :output: write csv files with the scores, normalized scores and ranks; log file

    """

    def __init__(self, config: object, input_matrix: pd.DataFrame()):

        self._config = copy.deepcopy(config)
        self._input_matrix = copy.deepcopy(input_matrix)

    def normalize_indicators(self) -> dict:
        """
        Get the input matrix.
        :return: a dictionary that contains the normalized values of each indicator per normalization method.
        Normalization functions implemented: minmax; target; standardized; rank
        """
        norm = Normalization(self._input_matrix, self._config.polarity_for_each_indicator)

        indicators_scaled_minmax_01 = norm.minmax(feature_range=(0, 1))
        indicators_scaled_minmax_no0 = norm.minmax(feature_range=(0.1, 1)) # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_target_01 = norm.target(feature_range=(0, 1))
        indicators_scaled_target_no0 = norm.target(feature_range=(0.1, 1)) # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_standardized = norm.standardized()
        indicators_scaled_rank = norm.rank()

        normalized_indicators = {"standardized": indicators_scaled_standardized,
                                 "minmax_01":  indicators_scaled_minmax_01,
                                 "minmax_no0": indicators_scaled_minmax_no0,
                                 "target_01":  indicators_scaled_target_01,
                                 "target_no0": indicators_scaled_target_no0,
                                 "rank":  indicators_scaled_rank
                                 }

        return normalized_indicators

    def aggregate_indicators(self, normalized_indicators: dict, weights: list) -> pd.DataFrame():
        """
        Get the normalized indicators per normalization methods in a dictionary.
        :return: aggregated scores per each alternative, and per each normalization method.
        Aggregation functions implemented: weighted-sum; geometric; harmonic; minimum
        """

        self.normalized_indicators = normalized_indicators
        self.weights = weights

        agg = Aggregation(self.weights)

        scores_weighted_sum_standardized = agg.weighted_sum(self.normalized_indicators["standardized"])
        scores_weighted_sum_minmax = agg.weighted_sum(self.normalized_indicators["minmax_01"])
        scores_weighted_sum_target = agg.weighted_sum(self.normalized_indicators["target_01"])
        scores_weighted_sum_rank = agg.weighted_sum(self.normalized_indicators["rank"])

        scores_geometric_standardized = agg.geometric(self.normalized_indicators["standardized"])
        scores_geometric_minmax = agg.geometric(self.normalized_indicators["minmax_no0"])
        scores_geometric_target = agg.geometric(self.normalized_indicators["target_no0"])
        scores_geometric_rank = agg.geometric(self.normalized_indicators["rank"])

        scores_harmonic_standardized = agg.harmonic(self.normalized_indicators["standardized"])
        scores_harmonic_minmax = agg.harmonic(self.normalized_indicators["minmax_no0"])
        scores_harmonic_target = agg.harmonic(self.normalized_indicators["target_no0"])
        scores_harmonic_rank = agg.harmonic(self.normalized_indicators["rank"])

        scores_minimum_standardized = agg.minimum(self.normalized_indicators["standardized"])
        scores_minimum_minmax = agg.minimum(self.normalized_indicators["minmax_01"])
        scores_minimum_target = agg.minimum(self.normalized_indicators["target_01"])
        scores_minimum_rank = agg.minimum(self.normalized_indicators["rank"])

        scores = pd.concat([scores_weighted_sum_standardized,scores_weighted_sum_minmax,scores_weighted_sum_target,scores_weighted_sum_rank,
                            scores_geometric_standardized,scores_geometric_minmax,scores_geometric_target,scores_geometric_rank,
                            scores_harmonic_standardized,scores_harmonic_minmax,scores_harmonic_target,scores_harmonic_rank,
                            scores_minimum_standardized,scores_minimum_minmax,scores_minimum_target,scores_minimum_rank], axis=1)
        col_names = ['ws-stand', 'ws-minmax', 'ws-target', 'ws-rank',
                     'geom-stand', 'geom-minmax', 'geom-target', 'geom-rank',
                     'harm-stand', 'harm-minmax', 'harm-target', 'harm-rank',
                     'min-stand', 'min-minmax', 'min-target', 'min-rank']

        scores.columns=col_names

        return scores