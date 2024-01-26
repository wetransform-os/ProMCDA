from mcda.mcda_functions.aggregation import Aggregation
from mcda.mcda_functions.normalization import Normalization
import sys
import logging
import pandas as pd
import multiprocessing
from functools import partial
from typing import List, Tuple

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA utils for parallelization")


def initialize_and_call_aggregation(args: Tuple[list, dict], method=None) -> pd.DataFrame:
    weights, data = args
    agg = Aggregation(weights)

    scores_one_run = aggregate_indicators_in_parallel(agg, data, method)

    return scores_one_run


def initialize_and_call_normalization(args: Tuple[pd.DataFrame, list, str]) -> List[dict]:
    matrix, polarities, method = args
    norm = Normalization(matrix, polarities)

    dict_normalized_matrix = normalize_indicators_in_parallel(norm, method)

    return dict_normalized_matrix


def normalize_indicators_in_parallel(norm: object, method=None) -> dict:
    indicators_scaled_standardized_any = None
    indicators_scaled_standardized_no0 = None
    indicators_scaled_minmax_01 = None
    indicators_scaled_minmax_no0 = None
    indicators_scaled_target_01 = None
    indicators_scaled_target_no0 = None
    indicators_scaled_rank = None

    if method is None or method == 'minmax':
        indicators_scaled_minmax_01 = norm.minmax(feature_range=(0, 1))
        # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_minmax_no0 = norm.minmax(feature_range=(0.1, 1))
    if method is None or method == 'target':
        indicators_scaled_target_01 = norm.target(feature_range=(0, 1))
        # for aggregation "geometric" and "harmonic" that accept no 0
        indicators_scaled_target_no0 = norm.target(feature_range=(0.1, 1))
    if method is None or method == 'standardized':
        indicators_scaled_standardized_any = norm.standardized(
            feature_range=('-inf', '+inf'))
        indicators_scaled_standardized_no0 = norm.standardized(
            feature_range=(0.1, '+inf'))
    if method is None or method == 'rank':
        indicators_scaled_rank = norm.rank()
    if method is not None and method not in ['minmax', 'target', 'standardized', 'rank']:
        logger.error('Error Message', stack_info=True)
        raise ValueError('The selected normalization method is not supported')

    normalized_indicators = {"standardized_any": indicators_scaled_standardized_any,
                             "standardized_no0": indicators_scaled_standardized_no0,
                             "minmax_01": indicators_scaled_minmax_01,
                             "minmax_no0": indicators_scaled_minmax_no0,
                             "target_01": indicators_scaled_target_01,
                             "target_no0": indicators_scaled_target_no0,
                             "rank": indicators_scaled_rank
                             }

    normalized_indicators = {
        k: v for k, v in normalized_indicators.items() if v is not None}

    return normalized_indicators


def aggregate_indicators_in_parallel(agg: object, normalized_indicators: dict, method=None) -> pd.DataFrame():
    scores_weighted_sum = {}
    scores_geometric = {}
    scores_harmonic = {}
    scores_minimum = {}

    scores = pd.DataFrame()
    col_names_method = []
    col_names = ['ws-minmax_01', 'ws-target_01', 'ws-standardized_any', 'ws-rank',
                 'geom-minmax_no0', 'geom-target_no0', 'geom-standardized_no0', 'geom-rank',
                 'harm-minmax_no0', 'harm-target_no0', 'harm-standardized_no0', 'harm-rank',
                 'min-standardized_any']  # same order as in the following loop
    for key, values in normalized_indicators.items():
        if method is None or method == 'weighted_sum':
            # ws goes only with some specific normalizations
            if key in ["standardized_any", "minmax_01", "target_01", "rank"]:
                scores_weighted_sum[key] = agg.weighted_sum(values)
                col_names_method.append("ws-" + key)
        if method is None or method == 'geometric':
            # geom goes only with some specific normalizations
            if key in ["standardized_no0", "minmax_no0", "target_no0", "rank"]:
                scores_geometric[key] = pd.Series(agg.geometric(values))
                col_names_method.append("geom-" + key)
        if method is None or method == 'harmonic':
            # harm goes only with some specific normalizations
            if key in ["standardized_no0", "minmax_no0", "target_no0", "rank"]:
                scores_harmonic[key] = pd.Series(agg.harmonic(values))
                col_names_method.append("harm-" + key)
        if method is None or method == 'minimum':
            if key == "standardized_any":
                scores_minimum[key] = pd.Series(agg.minimum(
                    normalized_indicators["standardized_any"]))
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


def parallelize_aggregation(args: List[tuple], method=None) -> List[pd.DataFrame]:
    partial_func = partial(initialize_and_call_aggregation, method=method)
    # create a synchronous multiprocessing pool with the desired number of processes
    pool = multiprocessing.Pool()
    res = pool.map(partial_func, args)
    pool.close()
    pool.join()

    return res


def parallelize_normalization(input_matrices: List[pd.DataFrame], polar: list, method=None) -> List[dict]:
    # create a synchronous multiprocessing pool with the desired number of processes
    pool = multiprocessing.Pool()
    args_for_parallel_norm = [(df, polar, method) for df in input_matrices]
    res = pool.map(initialize_and_call_normalization, args_for_parallel_norm)
    pool.close()
    pool.join()

    return res


def estimate_runs_mean_std(res: List[pd.DataFrame]) -> List[pd.DataFrame]:
    all_runs = pd.concat(res, axis=0)
    by_index = all_runs.groupby(all_runs.index)
    df_means = by_index.mean()
    df_stds = by_index.std()
    all_scores_mean_std = [df_means, df_stds]

    return all_scores_mean_std
