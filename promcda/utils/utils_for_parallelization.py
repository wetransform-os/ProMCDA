import sys
import logging
import pandas as pd
import multiprocessing
from functools import partial
from typing import List, Tuple, Optional

from promcda.mcda_functions.aggregation import Aggregation
from promcda.mcda_functions.normalization import Normalization
from promcda.enums import NormalizationFunctions, AggregationFunctions, NormalizationNames4Sensitivity
from promcda.configuration.output_column_mapping import output_column_mapping

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA utils for parallelization")


def initialize_and_call_aggregation(args: Tuple[list, dict], method: Optional[AggregationFunctions] = None) \
        -> pd.DataFrame:
    """
    Initialize an Aggregation object with given weights and call the aggregation method to calculate scores.

    Parameters:
    - args: a tuple containing a list of weights and a dictionary of data.
    - method (optional): the aggregation method to use. Defaults to None.

    Returns:
    - pd.DataFrame: DataFrame containing scores calculated using the aggregation method.

    Example:
    ```python
    weights = [0.2, 0.3, 0.5]
    data = {'indicator1': [10, 20, 30], 'indicator2': [40, 50, 60], 'indicator3': [70, 80, 90]}

    scores = initialize_and_call_aggregation((weights, data), method='weighted_sum')

    scores
    0   32.0
    1   50.0
    2   77.0

    ```

    This example initializes an Aggregation object with the given weights and calls the specified aggregation method
    to calculate scores using the provided data. The resulting scores are returned as a DataFrame.

    :param args: Tuple[list, dict]
    :param method: str
    :return scores_one_run: pd.DataFrame
    """
    weights, data = args
    agg = Aggregation(weights)

    scores_one_run = aggregate_indicators_in_parallel(agg, data, method)

    return scores_one_run


def initialize_and_call_normalization(args: Tuple[pd.DataFrame, Tuple[str, ...], NormalizationFunctions]) -> dict:
    """
    Initialize a Normalization object with given matrix and polarities, and call the normalization method to
    calculate normalized indicators.

    Parameters:
    - args: a tuple containing a DataFrame of indicators, a tuple of polarities,
      and a string specifying the normalization method.

    Returns:
    - dict: a dictionary containing normalized indicators.

    Example:
    ```python
    matrix = pd.DataFrame({'indicator1': [10, 20, 30], 'indicator2': [40, 50, 60]})
    polarities = ['+', '-', '+']
    method = 'minmax'

    normalized_indicators = initialize_and_call_normalization((matrix, polarities, method))

    normalized_indicators:
    {
    'indicator1': [0.0, 0.5, 1.0],
    'indicator2': [0.0, 0.5, 1.0]
     }

    ```

    This example initializes a Normalization object with the given matrix and polarities, and calls the specified
    normalization method to calculate normalized indicators. The resulting normalized indicators are returned
    as a dictionary.

    :param args: Tuple[pd.DataFrame, List[str], str]
    :return dict_normalized_matrix: dict
    """

    matrix, polarities, method = args
    norm = Normalization(matrix, polarities)

    dict_normalized_matrix = normalize_indicators_in_parallel(norm, method)

    return dict_normalized_matrix


def normalize_indicators_in_parallel(norm: object, method: Optional[NormalizationFunctions] = None) -> dict:
    """
        Normalize indicators in parallel using different normalization methods.

        Parameters:
        - norm: an object representing the normalization process.
        - method: the normalization method to use. Options are 'minmax', 'target', 'standardized', or 'rank'.

        Returns:
        - A dictionary containing the normalized indicators for each method.

        Example:
        ```python
        norm = Normalization(matrix, polarities)
        normalized_indicators = normalize_indicators_in_parallel(norm, method='minmax')

        ```

        This example normalizes indicators in parallel using the min-max normalization method.
        Supported methods include 'minmax', 'target', 'standardized', and 'rank'.
        The resulting normalized indicators are returned as a dictionary, where keys represent the normalization method
        used and values represent the corresponding normalized indicators.

        :param norm: object
        :param method: str
        :return normalized_indicators: dict
        """
    indicators_scaled_standardized_any = None
    indicators_scaled_standardized_without_zero = None
    indicators_scaled_minmax_01 = None
    indicators_scaled_minmax_without_zero = None
    indicators_scaled_target_01 = None
    indicators_scaled_target_without_zero = None
    indicators_scaled_rank = None

    # Avoid passing complex objects like Enums to multiprocessing workers.
    # On macOS and Windows, the default 'spawn' start method is used for multiprocessing,
    # which requires all objects to be pickleable. Enums can cause pickling issues,
    # especially if they are not defined in an importable module or are passed directly.
    # To ensure compatibility, consider passing only simple types (e.g., strings or integers)
    # derived from Enum values.

    if method is None or method == NormalizationFunctions.MINMAX:
        indicators_scaled_minmax_01 = norm.minmax(feature_range=(0, 1))
        # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_minmax_without_zero = norm.minmax(feature_range=(0.1, 1))

    if method is None or method == NormalizationFunctions.TARGET:
        indicators_scaled_target_01 = norm.target(feature_range=(0, 1))
        # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_target_without_zero = norm.target(feature_range=(0.1, 1))

    if method is None or method == NormalizationFunctions.STANDARDIZED:
        indicators_scaled_standardized_any = norm.standardized(
            feature_range=('-inf', '+inf'))
        indicators_scaled_standardized_without_zero = norm.standardized(
            feature_range=(0.1, '+inf'))

    if method is None or method == NormalizationFunctions.RANK:
        indicators_scaled_rank = norm.rank()

    elif method is not None and (method != NormalizationFunctions.MINMAX and method != NormalizationFunctions.TARGET
                                 and method != NormalizationFunctions.STANDARDIZED
                                 and method != NormalizationFunctions.RANK):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The selected normalization method is not supported')


    normalized_indicators = {
        NormalizationNames4Sensitivity.STANDARDIZED_ANY.value: indicators_scaled_standardized_any,
        NormalizationNames4Sensitivity.STANDARDIZED_WITHOUT_ZERO.value: indicators_scaled_standardized_without_zero,
        NormalizationNames4Sensitivity.MINMAX_01.value: indicators_scaled_minmax_01,
        NormalizationNames4Sensitivity.MINMAX_WITHOUT_ZERO.value: indicators_scaled_minmax_without_zero,
        NormalizationNames4Sensitivity.TARGET_01.value: indicators_scaled_target_01,
        NormalizationNames4Sensitivity.TARGET_WITHOUT_ZERO.value: indicators_scaled_target_without_zero,
        NormalizationNames4Sensitivity.RANK.value: indicators_scaled_rank}

    # Filter out None values - when some norm functions are not used
    normalized_indicators = {key: value for key, value in normalized_indicators.items() if value is not None}

    return normalized_indicators


def aggregate_indicators_in_parallel(agg: object, normalized_indicators: dict,
                                     method: Optional[AggregationFunctions] = None) -> pd.DataFrame:
    """
    Aggregate normalized indicators in parallel using different aggregation methods.

    Parameters:
    - agg: an object representing the aggregation process.
    - normalized_indicators: a dictionary containing the normalized indicators for each method.
    - method: the aggregation method to use. Options are 'weighted_sum', 'geometric', 'harmonic', 'minimum', or None
      (for all methods).

    Returns:
    - a DataFrame containing the aggregated scores for each method.

    Example:
    ```python
    agg = Aggregation(weights)
    aggregated_scores = aggregate_indicators_in_parallel(agg, normalized_indicators, method='weighted_sum')

    ```

    This example aggregates the normalized indicators in parallel using the weighted_sum method. The resulting
    aggregated scores are returned as a DataFrame, where columns represent the aggregation method used and rows
    represent the corresponding aggregated scores for each indicator.

    :param agg: object
    :param normalized_indicators: dict
    :param method: str
    :return scores: pd.DataFrame
    """
    scores_weighted_sum = {}
    scores_geometric = {}
    scores_harmonic = {}
    scores_minimum = {}

    scores = pd.DataFrame()
    col_names_method = []
    initial_method_is_none = None

    if isinstance(normalized_indicators, dict): # robustness on indicators
        for key, values in normalized_indicators.items():
            if method is None or method == AggregationFunctions.WEIGHTED_SUM:
                initial_method_is_none = method is None
                if method is None:
                    method = AggregationFunctions.WEIGHTED_SUM
                # ws goes only with some specific normalizations
                allowed_suffixes = {"standardized_any", "minmax_01", "target_01", "rank"}
                valid_suffixes = [
                    member.value for member in NormalizationNames4Sensitivity if member.value in allowed_suffixes
                ]
                if not valid_suffixes:
                    raise ValueError(
                        "No matching normalization names were found in the allowed_suffixes. "
                        "Please check weather the column names in normalization are conform to the configuration."
                    )
                matching_suffixes = [suffix.split('_')[0] for suffix in valid_suffixes if suffix in key]
                col_names = [
                    output_column_mapping.get(((method.value if method else AggregationFunctions.WEIGHTED_SUM), suffix))
                    for suffix in matching_suffixes
                ]
                if any(substring in key for substring in valid_suffixes):
                    scores_weighted_sum[key] = agg.weighted_sum(values)
                    col_names_method.append(col_names)
                if initial_method_is_none:
                    method = None
            if method is None or method == AggregationFunctions.GEOMETRIC:
                initial_method_is_none = method is None
                if method is None:
                    method = AggregationFunctions.GEOMETRIC
                # geom goes only with some specific normalizations
                allowed_suffixes = {"standardized_without_zero", "minmax_without_zero", "target_without_zero", "rank"}
                valid_suffixes = [
                    member.value for member in NormalizationNames4Sensitivity if member.value in allowed_suffixes
                ]
                if not valid_suffixes:
                    raise ValueError(
                        "No matching normalization names were found in the allowed_suffixes. "
                        "Please check weather the column names in normalization are conform to the configuration."
                    )
                matching_suffixes = [suffix.split('_')[0] for suffix in valid_suffixes if suffix in key]
                col_names = [
                    output_column_mapping.get(((method.value if method else AggregationFunctions.GEOMETRIC), suffix))
                    for suffix in matching_suffixes
                ]
                if any(substring in key for substring in valid_suffixes):
                    scores_geometric[key] = pd.Series(agg.geometric(values))
                    col_names_method.append(col_names)
                if initial_method_is_none:
                    method = None
            if method is None or method == AggregationFunctions.HARMONIC:
                initial_method_is_none = method is None
                if method is None:
                    method = AggregationFunctions.HARMONIC
                # harm goes only with some specific normalizations
                allowed_suffixes = {"standardized_without_zero", "minmax_without_zero", "target_without_zero", "rank"}
                valid_suffixes = [
                    member.value for member in NormalizationNames4Sensitivity if member.value in allowed_suffixes
                ]
                if not valid_suffixes:
                    raise ValueError(
                        "No matching normalization names were found in the allowed_suffixes. "
                        "Please check weather the column names in normalization are conform to the configuration."
                    )
                matching_suffixes = [suffix.split('_')[0] for suffix in valid_suffixes if suffix in key]
                col_names = [
                    output_column_mapping.get(((method.value if method else AggregationFunctions.HARMONIC), suffix))
                    for suffix in matching_suffixes
                ]
                if any(substring in key for substring in valid_suffixes):
                    scores_harmonic[key] = pd.Series(agg.harmonic(values))
                    col_names_method.append(col_names)
                if initial_method_is_none:
                    method = None
            if method is None or method == AggregationFunctions.MINIMUM:
                if method is None:
                    method = AggregationFunctions.MINIMUM
                allowed_suffixes = {"standardized_any"}
                valid_suffixes = [
                    member.value for member in NormalizationNames4Sensitivity if member.value in allowed_suffixes
                ]
                if not valid_suffixes:
                    raise ValueError(
                        "No matching normalization names were found in the allowed_suffixes. "
                        "Please check weather the column names in normalization are conform to the configuration."
                    )
                matching_suffixes = [suffix.split('_')[0] for suffix in valid_suffixes if suffix in key]
                col_names = [
                    output_column_mapping.get(((method.value if method else AggregationFunctions.MINIMUM), suffix))
                    for suffix in matching_suffixes
                ]
                if any(substring in key for substring in valid_suffixes):
                    scores_minimum[key] = pd.Series(agg.minimum(values))
                    col_names_method.append(col_names)
                if initial_method_is_none:
                    method = None

        col_names_method = [item for sublist in col_names_method for item in sublist]

    elif isinstance(normalized_indicators, pd.DataFrame): # robustness on weights
        if method is None or method == AggregationFunctions.WEIGHTED_SUM:
            # ws goes only with some specific normalizations
            allowed_suffixes = {"standardized_any", "minmax_01", "target_01", "rank"}
            valid_suffixes = [
                member.value for member in NormalizationNames4Sensitivity if member.value in allowed_suffixes
            ]
            if not valid_suffixes:
                raise ValueError(
                    "No matching normalization names were found in the allowed_suffixes. "
                    "Please check weather the column names in normalization are conform to the configuration."
                )
            scores_weighted_sum = pd.DataFrame()
            for suffix in valid_suffixes:
                selected_columns = [
                    column for column in normalized_indicators.columns
                    if column.endswith("_" + suffix)
                ]
                if selected_columns:
                    partial_scores = agg.weighted_sum(normalized_indicators[selected_columns])
                    scores_weighted_sum = pd.concat([scores_weighted_sum, partial_scores], axis=1)
                    col_names_method.extend([f"ws-{suffix}"])
        if method is None or method == AggregationFunctions.GEOMETRIC:
            # geom goes only with some specific normalizations
            allowed_suffixes = {"standardized_without_zero", "minmax_without_zero", "target_without_zero", "rank"}
            valid_suffixes = [
                member.value for member in NormalizationNames4Sensitivity if member.value in allowed_suffixes
            ]
            if not valid_suffixes:
                raise ValueError(
                    "No matching normalization names were found in the allowed_suffixes. "
                    "Please check weather the column names in normalization are conform to the configuration."
                )
            scores_geometric = pd.DataFrame()
            for suffix in valid_suffixes:
                selected_columns = [
                    column for column in normalized_indicators.columns
                    if column.endswith("_" + suffix)
                ]
                if selected_columns:
                    partial_scores = agg.geometric(normalized_indicators[selected_columns])
                    scores_geometric = pd.concat([scores_geometric, partial_scores], axis=1)
                    col_names_method.extend([f"geom-{suffix}"])
        if method is None or method == AggregationFunctions.HARMONIC:
            # harm goes only with some specific normalizations
            allowed_suffixes = {"standardized_without_zero", "minmax_without_zero", "target_without_zero", "rank"}
            valid_suffixes = [
                member.value for member in NormalizationNames4Sensitivity if member.value in allowed_suffixes
            ]
            if not valid_suffixes:
                raise ValueError(
                    "No matching normalization names were found in the allowed_suffixes. "
                    "Please check weather the column names in normalization are conform to the configuration."
                )
            scores_harmonic = pd.DataFrame()
            for suffix in valid_suffixes:
                selected_columns = [
                    column for column in normalized_indicators.columns
                    if column.endswith("_" + suffix)
                ]
                if selected_columns:
                    partial_scores = agg.harmonic(normalized_indicators[selected_columns])
                    scores_harmonic = pd.concat([scores_harmonic, partial_scores], axis=1)
                    col_names_method.extend([f"harm-{suffix}"])
        if method is None or method == AggregationFunctions.MINIMUM:
            # min goes only with specific normalizations
            allowed_suffixes = {"standardized_any"}
            valid_suffixes = [
                member.value for member in NormalizationNames4Sensitivity if member.value in allowed_suffixes
            ]
            if not valid_suffixes:
                raise ValueError(
                    "No matching normalization names were found in the allowed_suffixes. "
                    "Please check weather the column names in normalization are conform to the configuration."
                )
            scores_minimum = pd.DataFrame()
            for suffix in valid_suffixes:
                selected_columns = [
                    column for column in normalized_indicators.columns
                    if column.endswith("_" + suffix)
                ]
                if selected_columns:
                    partial_scores = agg.minimum(normalized_indicators[selected_columns])
                    scores_minimum = pd.concat([scores_minimum, partial_scores], axis=1)
                    col_names_method.extend([f"min-{suffix}"])

    dict_list = [scores_weighted_sum, scores_geometric,
                 scores_harmonic, scores_minimum]

    for d in dict_list:
        if isinstance(d, pd.DataFrame): # Robustness weights
            scores = pd.concat([scores, d], axis=1)

        elif isinstance(d, dict): # Robustness indicators
            scores = pd.concat([scores, pd.DataFrame.from_dict(d)], axis=1)

    scores.columns = col_names_method

    return scores


def parallelize_aggregation(args: List[tuple], aggregation_method=None) -> List[pd.DataFrame]:
    partial_func = partial(initialize_and_call_aggregation, method=aggregation_method)
    # create a synchronous multiprocessing pool with the desired number of processes
    with multiprocessing.Pool() as pool:
        res = pool.map(partial_func, args)

    return res


def parallelize_normalization(input_matrices: List[pd.DataFrame], polar: Tuple[str, ...], method=None) -> List[dict]:
    """
    Parallelize the normalization process for multiple input matrices using multiprocessing.

    Parameters:
    - input_matrices: a list of input matrices to be normalized.
    - polar: a list containing the polarities for each input matrix.
    - method: the normalization method to use. Options are 'minmax', 'target', 'standardized', 'rank', or None
      (for all methods).

    Returns:
    - a list of dictionaries containing the normalized indicators for each input matrix.

    Example:
    ```python
    input_matrices = [matrix1, matrix2, matrix3]
    polar = ['+', '-', '+']

    normalized_results = parallelize_normalization(input_matrices, polar, method='minmax')

    ```
    This example parallelizes the normalization process for multiple input matrices using multiprocessing.
    It creates a synchronous multiprocessing pool with the desired number of processes and maps the initialization
    and normalization calls across the input matrices. The function returns a list of dictionaries containing the
    normalized indicators for each input matrix.

    :param input_matrices: List[pd.DataFrame]
    :param polar: List[str]
    :param method: str
    :return res: List[dict]
    """
    args_for_parallel_norm = [(df, polar, method) for df in input_matrices]
    with multiprocessing.Pool() as pool:
        res = pool.map(initialize_and_call_normalization, args_for_parallel_norm)

    return res


def estimate_runs_mean_std(res: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Estimate the mean and standard deviation of scores from multiple runs.

    Parameters:
    - res: a list of DataFrames containing scores from multiple runs.

    Returns:
    - a list containing two DataFrames: one for the mean scores and one for the standard deviations.

    :param res: List[pd.DataFrame]
    :return all_scores_mean_std: List[pd.DataFrame]
    """
    all_runs = pd.concat(res, axis=0)
    by_index = all_runs.groupby(all_runs.index)
    df_means = by_index.mean()
    df_stds = by_index.std()
    all_scores_mean_std = [df_means, df_stds]

    return all_scores_mean_std
