import sys
import copy
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from promcda.configuration.output_column_mapping import output_column_mapping
from promcda.enums import NormalizationFunctions, AggregationFunctions, OutputColumnNames4Sensitivity
from promcda.mcda_functions.normalization import Normalization
from promcda.mcda_functions.aggregation import Aggregation

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

    def __init__(self, polarity: Tuple[str, ...], input_matrix: pd.DataFrame):
        self.normalized_indicators = None
        self.weights = None
        self.polarity = polarity
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
        norm = Normalization(self._input_matrix, self.polarity)

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
        :rtype: object
        """
        #if isinstance(agg_method, AggregationFunctions):
        #    method = agg_method.value

        self.normalized_indicators = normalized_indicators
        self.weights = weights

        agg= Aggregation(self.weights)

        final_scores = pd.DataFrame()

        def _apply_aggregation(norm_function, method, df_subset):
            """
            Apply the aggregation method to a subset of the DataFrame and store results in the appropriate DataFrame.
            """
            agg_methods = [e.value for e in AggregationFunctions] if method is None else [method.value]

            for method in agg_methods:
                aggregation_function = getattr(agg, method)
                aggregated_scores = aggregation_function(df_subset)

                if isinstance(aggregated_scores, np.ndarray):
                    aggregated_scores = pd.DataFrame(aggregated_scores, index=df_subset.index)
                elif isinstance(aggregated_scores, pd.Series):
                    aggregated_scores = aggregated_scores.to_frame()

                column_name = output_column_mapping.get((method, norm_function))

                if column_name in [e.value for e in OutputColumnNames4Sensitivity]:
                    aggregated_scores.columns = [column_name]
                else:
                    raise ValueError(f"Column name '{column_name}' not found in OutputColumnNames4Sensitivity")
                score_list.append(aggregated_scores)

        for norm_method in self.normalized_indicators.columns.str.split("_", n=0).str[1].unique():
            score_list = []

            norm_method_columns = self.normalized_indicators.filter(regex=rf"{norm_method}")

            without_zero_columns = norm_method_columns.filter(regex="without_zero$")
            with_zero_columns = norm_method_columns[norm_method_columns.columns.difference(without_zero_columns.columns)]
            rank_columns = norm_method_columns.filter(regex="rank$")
            without_zero_columns_rank = pd.concat([without_zero_columns, rank_columns], axis=1)

            # Apply WEIGHTED_SUM only to columns with zero in the suffix
            if agg_method is None or agg_method == AggregationFunctions.WEIGHTED_SUM:
                # Apply WEIGHTED_SUM to columns with zero in the suffix and only some normalization methods
                if norm_method in [NormalizationFunctions.STANDARDIZED.value, NormalizationFunctions.MINMAX.value,
                                   NormalizationFunctions.TARGET.value, NormalizationFunctions.RANK.value]:
                    _apply_aggregation(norm_method, AggregationFunctions.WEIGHTED_SUM,
                                   with_zero_columns)
            # Apply GEOMETRIC and HARMONIC only to columns without zero in the suffix and only some normalization methods
            if agg_method is None or agg_method == AggregationFunctions.GEOMETRIC:
                if norm_method in [NormalizationFunctions.STANDARDIZED.value, NormalizationFunctions.MINMAX.value,
                                   NormalizationFunctions.TARGET.value, NormalizationFunctions.RANK.value]:
                    _apply_aggregation(norm_method, AggregationFunctions.GEOMETRIC,
                                   without_zero_columns_rank)
            elif agg_method is None or agg_method == AggregationFunctions.HARMONIC:
                if norm_method in [NormalizationFunctions.STANDARDIZED.value, NormalizationFunctions.MINMAX.value,
                                   NormalizationFunctions.TARGET.value, NormalizationFunctions.RANK.value]:
                    _apply_aggregation(norm_method, AggregationFunctions.HARMONIC,
                                   without_zero_columns_rank)
            # Apply MINIMUM to columns with zero in the suffix and only some normalization methods
            if agg_method is None or agg_method == AggregationFunctions.MINIMUM:
                if norm_method in [NormalizationFunctions.STANDARDIZED.value]:
                    _apply_aggregation(norm_method, AggregationFunctions.MINIMUM,
                                   with_zero_columns)

            # Concatenate all score DataFrames into a single DataFrame if there are any
            if score_list:
                scores: DataFrame = pd.concat(score_list, axis=1)
                final_scores = pd.concat([final_scores, scores], axis=1)

        return final_scores
