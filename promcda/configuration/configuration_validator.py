import sys
import logging

import numpy as np
import pandas as pd
from typing import Tuple, Union

from pandas.core import series

from promcda.enums import PDFType, RobustnessAnalysisType
from promcda.utils.utils_for_main import pop_indexed_elements, check_norm_sum_weights, randomly_sample_all_weights, \
     randomly_sample_ix_weight

log = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)
FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")

from typing import List


def validate_configuration(
    input_matrix: pd.DataFrame,
    polarity: tuple,
    weights: list,
    marginal_distributions: tuple,
    num_runs: int,
    num_cores: int,
    random_seed: int,
    robustness: RobustnessAnalysisType
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
    if robustness != RobustnessAnalysisType.INDICATORS:
        num_indicators = input_matrix.shape[1]
    elif robustness == RobustnessAnalysisType.INDICATORS:
        num_non_exact_and_non_poisson = (len(marginal_distributions) -
                                         marginal_distributions.count(PDFType.EXACT) -
                                         marginal_distributions.count(PDFType.POISSON))
        num_indicators = (input_matrix.shape[1] - num_non_exact_and_non_poisson)

    if not isinstance(robustness, RobustnessAnalysisType):
        raise TypeError(f"'robustness' must be of type RobustnessAnalysisType, got {type(robustness).__name__}")

    if weights is not None and len(weights) != num_indicators:
        raise ValueError("'weights' must have the same number of elements as 'num_indicators'.")
    if weights is not None and not all(isinstance(w, (float, int)) for w in weights):
        raise TypeError(f"Expected 'weights' to be a float, int, or None, got {type(weights).__name__}")
    if weights is not None and any(w < 0 for w in weights):
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
    if not (marginal_distributions is None or isinstance(marginal_distributions, tuple)):
        raise TypeError(
            f"Expected 'marginal_distributions' to be a Tuple or None, got {type(marginal_distributions).__name__}")
    if marginal_distributions is not None and not all(isinstance(pdf, PDFType) for pdf in marginal_distributions):
        raise TypeError(f"Each element in 'marginal_distributions' should be of type 'PDFType'.")


def process_indicators_and_weights(input_matrix: pd.DataFrame,
                                   robustness: RobustnessAnalysisType,
                                   polarity: Tuple[str, ...],
                                   mc_runs: int,
                                   weights: List[str],
                                   marginal_distributions: Tuple[str, ...]) \
        -> Tuple[pd.DataFrame, int, Tuple[str, ...], Union[list, List[list], dict]]:
    """
    Process indicators and weights based on input parameters in the setup configuration.

    Parameters:
    - input_matrix: the input matrix without alternatives.
    - robustness: the type of robustness analysis to be performed.
    - polarity: a tuple containing the original polarity associated to each indicator.
    - mc_runs: number of Monte Carlo runs for robustness analysis.
    - weights: a list containing the assigned weights.

    Raises:
    - ValueError: If there are duplicated rows in the input matrix or if there is an issue with the configuration.

    Returns:
    - a shorter Tuple of polarities if one has been dropped together with the relative indicator,
      which brings no information. Otherwise, the same Tuple.
    - the normalised weights (either fixed or random sampled weights, depending on the settings)

    Notes:
    - For robustness == RobustnessAnalysisType.NONE:
        - Identifies and removes columns with constant values.
        - Logs the number of alternatives and indicators.

    - For robustness == RobustnessAnalysisType.INDICATORS:
        - Handles uncertainty in indicators.
        - Logs the number of alternatives and indicators.

    - For robustness != RobustnessAnalysisType.WEIGHTS:
        - Processes fixed weights if given.
        - Logs weights and normalised weights.

    - For robustness == RobustnessAnalysisType.WEIGHTS:
        - Performs robustness analysis on weights.
        - Logs randomly sampled weights.

    :param marginal_distributions:
    :param robustness:
    :param input_matrix: pd.DataFrame
    :param robustness: RobustnessAnalysisType
    :param polarity: List[str]
    :param mc_runs: int
    :param weights: List[str]
    :rtype: Tuple[List[str], Union[List[list], dict]]
    """
    num_unique = input_matrix.nunique()
    cols_to_drop = num_unique[num_unique == 1].index
    col_to_drop_indexes = input_matrix.columns.get_indexer(cols_to_drop)

    num_indicators = input_matrix.shape[1]

    if robustness != RobustnessAnalysisType.INDICATORS:
        if any(value == 1 for value in num_unique):
            logger.info("Indicators {} have been dropped because they carry no information".format(cols_to_drop))
            input_matrix = input_matrix.drop(cols_to_drop, axis=1)
        logger.info("Number of indicators: {}".format(input_matrix.shape[1]))
        if weights is None: weights = [0.5] * num_indicators
    elif robustness == RobustnessAnalysisType.INDICATORS:  # matrix with uncertainty on indicators
        num_non_exact_and_non_poisson = (len(marginal_distributions) - marginal_distributions.count(PDFType.EXACT) -
                                         marginal_distributions.count(PDFType.POISSON))
        num_indicators = (input_matrix.shape[1] - num_non_exact_and_non_poisson)
        if weights is None: weights = [0.5] * num_indicators

        logger.info("Number of alternatives: {}".format(input_matrix.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
        # TODO: eliminate indicators with constant values (i.e. same mean and 0 std) - optional

    polarities_and_weights = _handle_polarities_and_weights(robustness, num_unique,
                                                            col_to_drop_indexes, polarity, mc_runs, num_indicators,
                                                            weights)

    polar, norm_weights = tuple(item for item in polarities_and_weights if item is not None)

    return input_matrix, num_indicators, polar, norm_weights


def _handle_polarities_and_weights(robustness: RobustnessAnalysisType,
                                   num_unique: series,
                                   col_to_drop_indexes: np.ndarray,
                                   polarity: Tuple[str, ...],
                                   mc_runs: int,
                                   num_indicators: int,
                                   weights: List[str]) \
        -> Union[Tuple[List[str], list, None, None], Tuple[List[str], None, List[List], None],
        Tuple[Tuple[str, ...], None, None, dict]]:
    """
    Manage polarities and weights based on the specified robustness settings, ensuring that the appropriate adjustments
    and normalizations are applied before returning the necessary data structures.
    """
    norm_random_weights = []
    rand_weight_per_indicator = {}

    # Managing polarity
    polarity = pop_indexed_elements(col_to_drop_indexes, polarity)
    logger.info("Polarities are checked: {}".format(polarity))

    # Managing weights for no robustness indicators
    if robustness != RobustnessAnalysisType.INDICATORS:
        if any(value == 1 for value in num_unique):
           weights = pop_indexed_elements(col_to_drop_indexes, weights)
        # Managing weights for no robustness weights
        if (weights is not None
                and robustness != RobustnessAnalysisType.ALL_WEIGHTS
                and robustness != RobustnessAnalysisType.SINGLE_WEIGHTS):
            fixed_weights = weights
            norm_fixed_weights = check_norm_sum_weights(fixed_weights)
            logger.info("Weights: {}".format(fixed_weights))
            logger.info("Normalized weights: {}".format(norm_fixed_weights))
            return polarity, norm_fixed_weights, None, None
            #  Return None for norm_random_weights and rand_weight_per_indicator
        else:
            output_weights = handle_robustness_weights(mc_runs, num_indicators, robustness)
        if output_weights is not None:
            norm_random_weights, rand_weight_per_indicator = output_weights
        if norm_random_weights:
            return polarity, None, norm_random_weights, None
        else:
            return polarity, None, None, rand_weight_per_indicator
        #  Return None for norm_fixed_weights and one of the other two cases of randomness
    else: # robustness_indicators is True - in this case no columns are dropped!
        fixed_weights = weights
        norm_fixed_weights = check_norm_sum_weights(fixed_weights)
        return polarity, norm_fixed_weights, None, None


def handle_robustness_weights(mc_runs: int, num_indicators: int, robustness: RobustnessAnalysisType) \
        -> Union[None, tuple[None, dict[str, list[list]]], tuple[list[list], None]]:
    """
    Handle the generation and normalization of random weights based on the specified settings
    when a robustness analysis is requested.

    Depending on the robustness types, this function performs one of the following:
    1. If robustness on all weights is enabled and single-weight analysis is not,
       it generates `mc_runs` random weight vectors of length `num_indicators`, and normalizes them to sum to 1.
    2. If robustness on single weights is enabled and robustness on all weights is not,
       it generates `mc_runs` vectors per indicator, each time varying only one indicator's weight while keeping others constant,
       and normalizes each generated vector.

    Parameters:
    - mc_runs: Number of Monte Carlo simulation runs to perform for the robustness analysis.
    - num_indicators: The number of indicators (criteria) used in the decision-making process.
    - robustness: Indicates what kind of robustness should be performed.

    Returns:
    - norm_random_weights: A list of normalized weight vectors if robustness on all weights is selected, otherwise None.
    - rand_weight_per_indicator: A dictionary where each key corresponds to an indicator and the value is a list of normalized vectors,
        each varying that specific indicator's weight, if single-weight robustness is selected. Otherwise, None.

    Raises:
    - ValueError: if `mc_runs` is set to 0 or negative, as no robustness analysis can be performed in that case.

    :param mc_runs: int
    :param num_indicators: int
    :param robustness: RobustnessAnalysisType
    :return: norm_random_weights: List[list] or None
    :return: rand_weight_per_indicator: dict or None
    """
    norm_random_weights = []
    rand_weight_per_indicator = {}

    if mc_runs == 0:
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of MC runs should be larger than 0 for robustness analysis')

    if robustness == RobustnessAnalysisType.ALL_WEIGHTS:
        random_weights = randomly_sample_all_weights(num_indicators, mc_runs)
        for weights in random_weights:
            weights = check_norm_sum_weights(weights)
            norm_random_weights.append(weights)
        return norm_random_weights, None  # Return norm_random_weights, and None for rand_weight_per_indicator
    elif robustness == RobustnessAnalysisType.SINGLE_WEIGHTS :
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


def check_indicator_weights_polarities(num_indicators: int, polar: Tuple[str, ...], robustness: RobustnessAnalysisType,
                                       weights: List[int]):
    """
    Check the consistency of indicators, polarities, and fixed weights in a configuration.

    Parameters:
    - num_indicators: The number of indicators in the input matrix.
    - polar: A list containing the polarity associated to each indicator.
    - robustness: It specifies the type of robustness performed.
    - weights: A list containing the fixed weights assigned to each indicator.

    This function raises a ValueError if the following conditions are not met:
    1. The number of indicators does not match the number of polarities.
    2. "robustness_on_all_weights" is set to "no," and the number of fixed weights
        does not correspond to the number of indicators.

    Raises:
    - ValueError: if the conditions for indicator-polarity and fixed weights consistency are not met.

    :param num_indicators: int
    :param polar: List[str]
    :param robustness: RobustnessAnalysisType
    :param weights: List[int]
    :return: None
    """
    if num_indicators != len(polar):
        raise ValueError('The number of polarities does not correspond to the no. of indicators')

    # Check the number of fixed weights if no robustness on all weights is performed
    if (robustness != RobustnessAnalysisType.ALL_WEIGHTS) and (num_indicators != len(weights)):
        raise ValueError('The no. of fixed weights does not correspond to the no. of indicators')
