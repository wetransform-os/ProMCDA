import sys
import copy
import logging
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
            df.columns = [f"{col}_{method_name}" for col in df.columns]
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

    def aggregate_indicators(self, normalized_indicators: pd.DataFrame, weights: list, method=None) -> pd.DataFrame:
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
        # Convert `method` to string if it’s an enum instance
        if isinstance(method, AggregationFunctions):
            method = method.value

        self.normalized_indicators = normalized_indicators
        self.weights = weights

        agg = Aggregation(self.weights)

        # Dictionary to map aggregation methods to their corresponding score DataFrames
        score_dfs = {
            AggregationFunctions.WEIGHTED_SUM.value: pd.DataFrame(),
            AggregationFunctions.GEOMETRIC.value: pd.DataFrame(),
            AggregationFunctions.HARMONIC.value: pd.DataFrame(),
            AggregationFunctions.MINIMUM.value: pd.DataFrame(),
        }

        def _apply_aggregation(agg_method, df_subset, suffix):
            """
            Apply the aggregation method to a subset of the DataFrame and store results in the appropriate DataFrame.
            """
            agg_function = {
                AggregationFunctions.WEIGHTED_SUM.value: agg.weighted_sum,
                AggregationFunctions.GEOMETRIC.value: agg.geometric,
                AggregationFunctions.HARMONIC.value: agg.harmonic,
                AggregationFunctions.MINIMUM.value: agg.minimum,
            }.get(agg_method)

            if agg_function:
                aggregated_scores = agg_function(df_subset)

                if isinstance(aggregated_scores, pd.Series):
                    aggregated_scores = aggregated_scores.to_frame()

                aggregated_scores.columns = [f"{col}_{agg_method}_{suffix}" for col in
                                             df_subset.columns.unique(level=0)]

        for base_col_name in self.normalized_indicators.columns.str.split("_").str[0].unique():
            relevant_columns = self.normalized_indicators.filter(regex=f"^{base_col_name}_")

            for suffix in relevant_columns.columns.str.split("_", n=1).str[1].unique():
                # Define the correct columns based on whether "without_zero" is in the suffix or not
                if method is None or method == AggregationFunctions.WEIGHTED_SUM.value:
                    if "without_zero" not in suffix:
                        # Only select columns ending with the exact suffix that doesn't contain "without_zero"
                        selected_columns = relevant_columns.filter(regex=f"_{suffix}$")
                        _apply_aggregation(AggregationFunctions.WEIGHTED_SUM.value, selected_columns, suffix)

                elif method in [AggregationFunctions.GEOMETRIC.value, AggregationFunctions.HARMONIC.value]:
                    if "without_zero" in suffix:
                        selected_columns = relevant_columns.filter(regex=f"_{suffix}$")
                        if method == AggregationFunctions.GEOMETRIC.value:
                            _apply_aggregation(AggregationFunctions.GEOMETRIC.value, selected_columns, suffix)
                        elif method == AggregationFunctions.HARMONIC.value:
                            _apply_aggregation(AggregationFunctions.HARMONIC.value, selected_columns, suffix)

                elif method == AggregationFunctions.MINIMUM.value:
                    if "without_zero" not in suffix:
                        selected_columns = relevant_columns.filter(regex=f"_{suffix}$")
                        _apply_aggregation(AggregationFunctions.MINIMUM.value, selected_columns, suffix)

        # Loop through all columns to detect normalization methods
        # for normalization_col_name in self.normalized_indicators.columns.str.split("_").str[1].unique():
        #     suffix = normalized_indicators.columns.str.split("_", n=1).str[1]
        #     relevant_columns = self.normalized_indicators.filter(regex=f"_{normalization_col_name}(_|$)")
        #
        #      # weighted_sum
        #     if method is None or method == AggregationFunctions.WEIGHTED_SUM.value:
        #         if "without_zero" not in suffix:
        #             _apply_aggregation(AggregationFunctions.WEIGHTED_SUM.value, relevant_columns, suffix)
        #
        #     # geometric or harmonic
        #     if method in [AggregationFunctions.GEOMETRIC.value,
        #                   AggregationFunctions.HARMONIC.value] and "without_zero" in suffix:
        #     # minimum
        #         if method == AggregationFunctions.GEOMETRIC.value:
        #             _apply_aggregation(AggregationFunctions.GEOMETRIC.value, relevant_columns,
        #                                    f"_geom_{suffix}")
        #         elif method == AggregationFunctions.HARMONIC.value:
        #             _apply_aggregation(AggregationFunctions.HARMONIC.value, relevant_columns, f"_harm_{suffix}")
        #     if method == AggregationFunctions.MINIMUM.value:
        #         if "without_zero" not in suffix:
        #             _apply_aggregation(AggregationFunctions.MINIMUM.value, selected_columns, f"_min_{suffix}")

        # Concatenate all score DataFrames into a single DataFrame
        scores = pd.concat([df for df in score_dfs.values() if not df.empty], axis=1)

        return scores
