import sys
import logging

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional

from pandas.core import series

from mcda.configuration.enums import PDFType
from mcda.utils.utils_for_main import pop_indexed_elements, check_norm_sum_weights, randomly_sample_all_weights, \
     randomly_sample_ix_weight, check_input_matrix

log = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)
FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")

from typing import Dict, List, Any


def validate_configuration(
    input_matrix: pd.DataFrame,
    polarity: tuple,
    weights: list,
    marginal_distributions: tuple,
    num_runs: int,
    num_cores: int,
    random_seed: int,
    robustness_weights: bool,
    robustness_single_weights: bool,
    robustness_indicators: bool
):
    # Required parameters
    if input_matrix is None:
        raise ValueError("The parameter 'input_matrix' is required but was not provided.")
    if not isinstance(input_matrix, pd.DataFrame):
        raise TypeError(f"Expected 'input_matrix' to be a DataFrame, got {type(input_matrix).__name__}")
    if polarity is None:
        raise ValueError("The parameter 'polarity' is required but was not provided.")
    if not isinstance(polarity, tuple):
        raise TypeError(f"Expected 'polarity' to be a Tuple, got {type(polarity).__name__}")
    if not all(isinstance(item, str) for item in polarity):
        raise ValueError("All elements in 'polarity' must be strings.")

    # Optional parameters
    # Check the input matrix
    if robustness_indicators is False:
        num_indicators = input_matrix.shape[1]
    else:
        num_non_exact_and_non_poisson = (len(marginal_distributions) -
                                         marginal_distributions.count(PDFType.EXACT.value) -
                                         marginal_distributions.count(PDFType.POISSON.value))
        num_indicators = (input_matrix.shape[1] - num_non_exact_and_non_poisson)

    for param_name, param_value in {
        'robustness_weights': robustness_weights,
        'robustness_single_weights': robustness_single_weights,
        'robustness_indicators': robustness_indicators
    }.items():
        if not isinstance(param_value, bool):
            raise TypeError(f"Expected '{param_name}' to be a bool, got {type(param_value).__name__}")

    if robustness_weights and robustness_single_weights:
        raise ValueError("'robustness_weights' and 'robustness_single_weights' cannot both be True.")

    if (robustness_weights or robustness_single_weights) and robustness_indicators:
        raise ValueError(
            "If 'robustness_weights' or 'robustness_single_weights' is True, 'robustness_indicators' must be False.")

    if robustness_indicators and (robustness_weights or robustness_single_weights):
        raise ValueError(
            "If 'robustness_indicators' is True, 'robustness_weights' and 'robustness_single_weights' must be False.")

    if weights is not None and len(weights) != num_indicators:
        raise ValueError("'weights' must have the same number of elements as 'num_indicators'.")
    if weights is not None and not all(isinstance(w, (float, int)) for w in weights):
        raise TypeError(f"Expected 'weights' to be a float, int, or None, got {type(weights).__name__}")
    if weights is not None and any(w < 0 for w in weights): # TODO: check if this is correct
        raise ValueError("'weights' must be a non-negative value.")

    if not isinstance(num_runs, int):
        raise TypeError(f"Expected 'num_runs' to be an int, got {type(num_runs).__name__}")
    if num_runs <= 0:
        raise ValueError("'num_runs' must be a positive integer.")

    if not isinstance(num_cores, int):
        raise TypeError(f"Expected 'num_cores' to be an int, got {type(num_cores).__name__}")
    if num_cores < 1:
        raise ValueError("'num_cores' must be greater than or equal to 1.")

    if not isinstance(random_seed, int):
        raise TypeError(f"Expected 'random_seed' to be an int, got {type(random_seed).__name__}")
    if random_seed < 0:
        raise ValueError("'random_seed' must be a non-negative integer.")

    if marginal_distributions is not None and len(marginal_distributions) != num_indicators:
        raise ValueError("'marginal_distributions' must have the same number of elements as 'num_indicators'.")
    if not isinstance(marginal_distributions, (tuple, None)):
        raise TypeError(
            f"Expected 'marginal_distributions' to be a Tuple or None, got {type(marginal_distributions).__name__}")
    if marginal_distributions is not None and not all(isinstance(pdf, PDFType) for pdf in marginal_distributions):
        raise TypeError(f"Each element in 'marginal_distributions' should be of type 'PDFType'.")


def process_indicators_and_weights(input_matrix: pd.DataFrame,
                                   robustness_indicators: bool,
                                   robustness_weights: bool,
                                   robustness_single_weights: bool,
                                   polarity: Tuple[str, ...],
                                   mc_runs: int,
                                   num_indicators: int,
                                   weights: List[str]) \
        -> Tuple[List[str], Union[list, List[list], dict]]:
    """
    Process indicators and weights based on input parameters in the configuration.

    Parameters:
    - input_matrix: the input matrix without alternatives.
    - robustness_indicators: a flag indicating whether the matrix should include indicator uncertainties
      (True or False).
    - robustness_weights: a flag indicating whether robustness analysis is considered for the weights (True or False).
    - robustness_single_weights: a flag indicating whether robustness analysis is considered for a single weight
      at time (True or False).
    - polarity: a tuple containing the original polarity associated to each indicator.
    - mc_runs: number of Monte Carlo runs for robustness analysis.
    - num_indicators: the number of indicators in the input matrix.
    - weights: a list containing the assigned weights.

    Raises:
    - ValueError: If there are duplicated rows in the input matrix or if there is an issue with the configuration.

    Returns:
    - a shorter list of polarities if one has been dropped together with the relative indicator,
      which brings no information. Otherwise, the same list.
    - the normalised weights (either fixed or random sampled weights, depending on the settings)

    Notes:
    - For robustness_indicators == False:
        - Identifies and removes columns with constant values.
        - Logs the number of alternatives and indicators.

    - For robustness_indicators == True:
        - Handles uncertainty in indicators.
        - Logs the number of alternatives and indicators.

    - For robustness_weights == False:
        - Processes fixed weights if given.
        - Logs weights and normalised weights.

    - For robustness_weights == True:
        - Performs robustness analysis on weights.
        - Logs randomly sampled weights.

    :param input_matrix: pd.DataFrame
    :param robustness_weights: bool
    :param robustness_single_weights:
    :param robustness_indicators: bool
    :param polarity: List[str]
    :param mc_runs: int
    :param num_indicators: int
    :param weights: List[str]
    :rtype: Tuple[List[str], Union[List[list], dict]]
    """
    num_unique = input_matrix.nunique()
    cols_to_drop = num_unique[num_unique == 1].index
    col_to_drop_indexes = input_matrix.columns.get_indexer(cols_to_drop)

    if robustness_indicators is False:
        _handle_no_robustness_indicators(input_matrix)
    else:  # matrix with uncertainty on indicators
        logger.info("Number of alternatives: {}".format(input_matrix.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
        # TODO: eliminate indicators with constant values (i.e. same mean and 0 std) - optional

    polarities_and_weights = _handle_polarities_and_weights(robustness_indicators, robustness_weights,
                                                            robustness_single_weights, num_unique,
                                                            col_to_drop_indexes, polarity, mc_runs, num_indicators,
                                                            weights)

    polar, norm_weights = tuple(item for item in polarities_and_weights if item is not None)

    return polar, norm_weights


def _handle_polarities_and_weights(robustness_indicators: bool,
                                   robustness_weights: bool,
                                   robustness_single_weights: bool,
                                   num_unique: series,
                                   col_to_drop_indexes: np.ndarray,
                                   polarity: Tuple[str, ...],
                                   mc_runs: int,
                                   num_indicators: int,
                                   weights: List[str]) \
        -> Union[Tuple[List[str], list, None, None], Tuple[List[str], None, List[List], None],
        Tuple[List[str], None, None, dict]]:
    """
    Manage polarities and weights based on the specified robustness settings, ensuring that the appropriate adjustments
    and normalizations are applied before returning the necessary data structures.
    """
    norm_random_weights = []
    rand_weight_per_indicator = {}

    # Managing polarities
    if robustness_indicators is False:
        if any(value == 1 for value in num_unique):
            polarity = pop_indexed_elements(col_to_drop_indexes, polarity)
    logger.info("Polarities: {}".format(polarity))

    # Managing weights
    if robustness_weights is False and robustness_single_weights is False:
        fixed_weights = weights
        if any(value == 1 for value in num_unique):
            fixed_weights = pop_indexed_elements(col_to_drop_indexes, fixed_weights)
        norm_fixed_weights = check_norm_sum_weights(fixed_weights)
        logger.info("Weights: {}".format(fixed_weights))
        logger.info("Normalized weights: {}".format(norm_fixed_weights))
        return polarity, norm_fixed_weights, None, None
        #  Return None for norm_random_weights and rand_weight_per_indicator
    else:
        output_weights = _handle_robustness_weights(mc_runs, num_indicators, robustness_weights,
                                                    robustness_single_weights)
        if output_weights is not None:
            norm_random_weights, rand_weight_per_indicator = output_weights
        if norm_random_weights:
            return polarity, None, norm_random_weights, None
        else:
            return polarity, None, None, rand_weight_per_indicator
        #  Return None for norm_fixed_weights and one of the other two cases of randomness


def _handle_robustness_weights(mc_runs: int, num_indicators: int, robustness_weights: bool,
                               robustness_single_weight: bool) -> Tuple[Union[List[list], None], Union[dict, None]]:
    """
    Handle the generation and normalization of random weights based on the specified settings
    when a robustness analysis is requested on all the weights.
    """
    norm_random_weights = []
    rand_weight_per_indicator = {}

    if mc_runs == 0:
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of MC runs should be larger than 0 for robustness analysis')

    if robustness_single_weight is False and robustness_weights is True:
        random_weights = randomly_sample_all_weights(num_indicators, mc_runs)
        for weights in random_weights:
            weights = check_norm_sum_weights(weights)
            norm_random_weights.append(weights)
        return norm_random_weights, None  # Return norm_random_weights, and None for rand_weight_per_indicator
    elif robustness_single_weight is True and robustness_weights is False:
        i = 0
        while i < num_indicators:
            random_weights = randomly_sample_ix_weight(num_indicators, i, mc_runs)
            norm_random_weight = []
            for weights in random_weights:
                weights = check_norm_sum_weights(weights)
                norm_random_weight.append(weights)
            rand_weight_per_indicator["indicator_{}".format(i + 1)] = norm_random_weight
            i += 1
        return None, rand_weight_per_indicator  # Return None for norm_random_weights, and rand_weight_per_indicator


def _handle_no_robustness_indicators(input_matrix: pd.DataFrame):
    """
    Handle the indicators in case of no robustness analysis required.
    (The input matrix is without the alternative column)
    """
    num_unique = input_matrix.nunique()
    cols_to_drop = num_unique[num_unique == 1].index

    if any(value == 1 for value in num_unique):
        logger.info("Indicators {} have been dropped because they carry no information".format(cols_to_drop))
        input_matrix = input_matrix.drop(cols_to_drop, axis=1)

    num_indicators = input_matrix.shape[1]
    logger.info("Number of alternatives: {}".format(input_matrix.shape[0]))
    logger.info("Number of indicators: {}".format(num_indicators))


def check_indicator_weights_polarities(num_indicators: int, polar: List[str], robustness_weights: bool,
                                       weights: List[int]):
    """
    Check the consistency of indicators, polarities, and fixed weights in a configuration.

    Parameters:
    - num_indicators: the number of indicators in the input matrix.
    - polar: a list containing the polarity associated to each indicator.
    - config: the configuration dictionary.

    This function raises a ValueError if the following conditions are not met:
    1. The number of indicators does not match the number of polarities.
    2. "robustness_on_all_weights" is set to "no," and the number of fixed weights
        does not correspond to the number of indicators.

    Raises:
    - ValueError: if the conditions for indicator-polarity and fixed weights consistency are not met.

    :param weights: List[int]
    :param robustness_weights: bool
    :param num_indicators: int
    :param polar: List[str]
    :param config: dict
    :return: None
    """
    if num_indicators != len(polar):
        raise ValueError('The number of polarities does not correspond to the no. of indicators')

    # Check the number of fixed weights if "robustness_on_all_weights" is set to "no"
    if (robustness_weights is False) and (num_indicators != len(weights)):
        raise ValueError('The no. of fixed weights does not correspond to the no. of indicators')
