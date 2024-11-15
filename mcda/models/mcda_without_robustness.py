import sys
import copy
import logging

import numpy as np
import pandas as pd

from mcda.configuration.enums import NormalizationFunctions, OutputColumnNames4Sensitivity, \
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

    import pandas as pd

    def normalize_indicators(self, method=None) -> pd.DataFrame:
        """
        Normalize the input matrix using the specified normalization method.

        Parameters:
        - method (optional): the normalization method to use. If None, all available methods will be applied.
          Supported methods: 'minmax', 'target', 'standardized', 'rank'.

        Returns:
        - A DataFrame containing the normalized values of each indicator per normalization method.
          Columns are named according to the normalization method applied.

        Notes:
        Some aggregation methods do not work with indicator values equal or smaller than zero. For that reason:
        - for the 'minmax' method, two sets of normalized indicators are returned: one with the range (0, 1) and
          another with the range (0.1, 1).
        - for the 'target' method, two sets of normalized indicators are returned: one with the range (0, 1) and
          another with the range (0.1, 1).
        - for the 'standardized' method, two sets of normalized indicators are returned: one with the range (-inf, +inf)
          and another with the range (0.1, +inf).
        """
        norm = Normalization(self._input_matrix, self._config["polarity"])

        normalized_dataframes = []

        def add_normalized_df(df, method_name):
            df.columns = [f"{col}_{method_name}" for col in self._input_matrix.columns.tolist()]
            normalized_dataframes.append(df)

        if isinstance(method, NormalizationFunctions):
            method = method.value

        if method is None or method == NormalizationFunctions.MINMAX.value:
            indicators_minmax_01 = norm.minmax(feature_range=(0, 1))
            indicators_minmax_without_zero = norm.minmax(feature_range=(0.1, 1))
            add_normalized_df(indicators_minmax_01, "minmax_01")
            add_normalized_df(indicators_minmax_without_zero, "minmax_without_zero")

        if method is None or method == NormalizationFunctions.TARGET.value:
            indicators_target_01 = norm.target(feature_range=(0, 1))
            indicators_target_without_zero = norm.target(feature_range=(0.1, 1))
            add_normalized_df(indicators_target_01, "target_01")
            add_normalized_df(indicators_target_without_zero, "target_without_zero")

        if method is None or method == NormalizationFunctions.STANDARDIZED.value:
            indicators_standardized_any = norm.standardized(feature_range=('-inf', '+inf'))
            indicators_standardized_without_zero = norm.standardized(feature_range=(0.1, '+inf'))
            add_normalized_df(indicators_standardized_any, "standardized_any")
            add_normalized_df(indicators_standardized_without_zero, "standardized_without_zero")

        if method is None or method == NormalizationFunctions.RANK.value:
            indicators_rank = norm.rank()
            add_normalized_df(indicators_rank, "rank")

        if method is not None and method not in [method.value for method in NormalizationFunctions]:
            logger.error('Error Message', stack_info=True)
            raise ValueError('The selected normalization method is not supported')

        # Concatenate all normalized DataFrames along columns
        normalized_df = pd.concat(normalized_dataframes, axis=1)

        return normalized_df

    def aggregate_indicators(self, normalized_indicators: pd.DataFrame, weights: list, agg_method=None) -> pd.DataFrame:
        """
        Aggregate the normalized indicators using the specified aggregation method.

        Parameters:
        - normalized_indicators: a DataFrame containing the normalized values of each indicator per normalization
          method.
        - weights: the weights to be applied during aggregation.
        - method (optional): The aggregation method to use. If None, all available methods will be applied.
          Supported methods: 'weighted_sum', 'geometric', 'harmonic', 'minimum'.

        Returns:
        - A DataFrame containing the aggregated scores for each alternative and normalization method.
        """
        if isinstance(agg_method, AggregationFunctions):
            method = agg_method.value

        self.normalized_indicators = normalized_indicators
        self.weights = weights

        agg= Aggregation(weights)

        score_list = []

        def _apply_aggregation(norm_method, agg_method, df_subset):
            """
            Apply the aggregation method to a subset of the DataFrame and store results in the appropriate DataFrame.
            """
            agg_functions = {
                AggregationFunctions.WEIGHTED_SUM.value: agg.weighted_sum,
                AggregationFunctions.GEOMETRIC.value: agg.geometric,
                AggregationFunctions.HARMONIC.value: agg.harmonic,
                AggregationFunctions.MINIMUM.value: agg.minimum,
            }

            agg_methods = list(agg_functions.keys()) if agg_method is None else [agg_method]

            for method in agg_methods:
                agg_function = agg_functions[method]
                aggregated_scores = agg_function(df_subset)

                if isinstance(aggregated_scores, np.ndarray):
                    aggregated_scores = pd.DataFrame(aggregated_scores, index=df_subset.index)
                elif isinstance(aggregated_scores, pd.Series):
                    aggregated_scores = aggregated_scores.to_frame()

                aggregated_scores.columns = [f"{norm_method}_{method}"]
                score_list.append(aggregated_scores)

        for norm_method in self.normalized_indicators.columns.str.split("_", n=0).str[1].unique():

            norm_method_columns = self.normalized_indicators.filter(regex=rf"{norm_method}")

            without_zero_columns = norm_method_columns.filter(regex="without_zero$")
            with_zero_columns = norm_method_columns[norm_method_columns.columns.difference(without_zero_columns.columns)]

            # Apply WEIGHTED_SUM only to columns with zero in the suffix
            if agg_method is None or agg_method == AggregationFunctions.WEIGHTED_SUM.value:
                _apply_aggregation(norm_method, AggregationFunctions.WEIGHTED_SUM.value,
                                   with_zero_columns)

            # Apply GEOMETRIC and HARMONIC only to columns without zero in the suffix
            if agg_method is None or agg_method == AggregationFunctions.GEOMETRIC.value:
                _apply_aggregation(norm_method, AggregationFunctions.GEOMETRIC.value,
                                   without_zero_columns)
            elif agg_method is None or agg_method == AggregationFunctions.HARMONIC.value:
                _apply_aggregation(norm_method, AggregationFunctions.HARMONIC.value,
                                   without_zero_columns)

            # Apply MINIMUM only to columns with zero in the suffix
            if agg_method is None or agg_method == AggregationFunctions.MINIMUM.value:
                _apply_aggregation(norm_method, AggregationFunctions.MINIMUM.value,
                                   with_zero_columns)

        # Concatenate all score DataFrames into a single DataFrame
        scores = pd.concat(score_list, axis=1)

        return scores
