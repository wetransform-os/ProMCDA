import sys
import logging

import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Dict, Any

from mcda.utils.utils_for_main import pop_indexed_elements, check_norm_sum_weights, randomly_sample_all_weights, \
    randomly_sample_ix_weight, check_input_matrix

log = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)
FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")

from typing import Dict, List, Any


def check_configuration_keys(sensitivity: dict, robustness: dict, monte_carlo: dict) -> bool:
    """
    Checks for required keys in sensitivity, robustness, and monte_carlo dictionaries.
    TODO: revisit this logic when substitute classes to handle configuration settings.

    :param sensitivity : dict
    :param robustness : dict
    :param: monte_carlo : dict
    :rtype: bool
    """

    keys_of_dict_values = {
        'sensitivity': ['sensitivity_on', 'normalization', 'aggregation'],
        'robustness': ['robustness_on', 'on_single_weights', 'on_all_weights', 'given_weights', 'on_indicators'],
        'monte_carlo': ['monte_carlo_runs', 'num_cores', 'random_seed', 'marginal_distribution_for_each_indicator']
    }

    _check_dict_keys(sensitivity, keys_of_dict_values['sensitivity'])
    _check_dict_keys(robustness, keys_of_dict_values['robustness'])
    _check_dict_keys(monte_carlo, keys_of_dict_values['monte_carlo'])

    return True


def _check_dict_keys(dic: Dict[str, Any], expected_keys: List[str]) -> None:
    """
    Helper function to check if the dictionary contains the required keys.
    """
    for key in expected_keys:
        if key not in dic:
            raise KeyError(f"The key '{key}' is missing in the provided dictionary")


def extract_configuration_values(input_matrix: pd.DataFrame, polarity: Tuple[str], sensitivity: dict, robustness: dict,
                                 monte_carlo: dict, output_path: str) -> dict:
    """
    Extracts relevant configuration values required for running the ProMCDA process.

    This function takes input parameters related to the decision matrix, polarity, sensitivity analysis,
    robustness analysis, and Monte Carlo simulations, and returns a dictionary containing the necessary
    configuration values for further processing.

    Parameters:
    -----------
    The decision matrix containing the alternatives and indicators.

    A tuple indicating the polarity (positive/negative) of each indicator.

    A dictionary specifying the sensitivity analysis configuration, including whether sensitivity is enabled.

    A dictionary specifying the robustness analysis configuration, including which robustness options are
    enabled (e.g., on single weights, on all weights, and on indicators).

    A dictionary containing Monte Carlo simulation parameters such as the number of runs and the random seed.

    :param input_matrix : pd.DataFrame
    :param polarity : Tuple[str]
    :param sensitivity : dict
    :param robustness : dict
    :param: monte_carlo : dict
    :param: output_path: str
    :return: extracted_values: dict
    """

    extracted_values = {
        "input_matrix": input_matrix,
        "polarity": polarity,
        # Sensitivity settings
        "sensitivity_on": sensitivity["sensitivity_on"],
        "normalization": None if sensitivity["sensitivity_on"] == 'yes' else sensitivity["normalization"],
        "aggregation": None if sensitivity["sensitivity_on"] == 'yes' else sensitivity["aggregation"],
        # Robustness settings
        "robustness_on": robustness["robustness_on"],
        "robustness_on_single_weights": robustness["on_single_weights"],
        "robustness_on_all_weights": robustness["on_all_weights"],
        "given_weights": robustness["given_weights"],
        "robustness_on_indicators": robustness["on_indicators"],
        # Monte Carlo settings
        "monte_carlo_runs": monte_carlo["monte_carlo_runs"],
        "num_cores": monte_carlo["num_cores"],
        "random_seed": monte_carlo["random_seed"],
        "marginal_distribution_for_each_indicator": monte_carlo["marginal_distribution_for_each_indicator"],
        "output_path": output_path
    }

    return extracted_values


def check_configuration_values(extracted_values: dict) -> Tuple[int, int, List[str], Union[list, List[list], dict]]:
    """
    Validates the configuration settings for the ProMCDA process based on the input parameters.

    This function checks the validity of the input parameters related to sensitivity analysis, robustness analysis,
    and Monte Carlo simulations. It ensures that the configuration is coherent and alerts the user to any inconsistencies.

    Parameters:
    -----------
    A dictionary containing configuration values extracted from the input parameters. It includes:
    - input_matrix (pd.DataFrame): The decision matrix for alternatives and indicators.
    - polarity (Tuple[str]): A tuple indicating the polarity of each indicator.
    - sensitivity_on (str): Indicates whether sensitivity analysis is enabled ("yes" or "no").
    - normalization (str): The normalization method to be used if sensitivity analysis is enabled.
    - aggregation (str): The aggregation method to be used if sensitivity analysis is enabled.
    - robustness_on (str): Indicates whether robustness analysis is enabled ("yes" or "no").
    - robustness_on_single_weights (str): Indicates if robustness is applied on individual weights.
    - robustness_on_all_weights (str): Indicates if robustness is applied on all weights.
    - robustness_on_indicators (str): Indicates if robustness is applied on indicators.
    - monte_carlo_runs (int): The number of Monte Carlo simulation runs.
    - random_seed (int): The seed for random number generation.
    - marginal_distribution_for_each_indicator (List[str]): The distribution types for each indicator.

    Raises:
    -------
    ValueError
    If any configuration settings are found to be inconsistent or contradictory.

    Returns:
    --------
    int
    A flag indicating whether robustness analysis will be performed on indicators (1) or not (0).

    :param: extracted_values : dict
    :return: is_robustness_indicators: int
    """

    is_robustness_indicators = 0
    is_robustness_weights = 0

    # Access the values from the dictionary
    input_matrix = extracted_values["input_matrix"]
    polarity = extracted_values["polarity"]
    sensitivity_on = extracted_values["sensitivity_on"]
    normalization = extracted_values["normalization"]
    aggregation = extracted_values["aggregation"]
    robustness_on = extracted_values["robustness_on"]
    robustness_on_single_weights = extracted_values["robustness_on_single_weights"]
    robustness_on_all_weights = extracted_values["robustness_on_all_weights"]
    robustness_on_indicators = extracted_values["robustness_on_indicators"]
    monte_carlo_runs = extracted_values["monte_carlo_runs"]
    random_seed = extracted_values["random_seed"]
    marginal_distribution = extracted_values["marginal_distribution_for_each_indicator"]

    # Check for sensitivity-related configuration errors
    if sensitivity_on == "no":
        check_config_error(normalization not in ['minmax', 'target', 'standardized', 'rank'],
                           'The available normalization functions are: minmax, target, standardized, rank.')
        check_config_error(aggregation not in ['weighted_sum', 'geometric', 'harmonic', 'minimum'],
                           'The available aggregation functions are: weighted_sum, geometric, harmonic, minimum.'
                           '\nWatch the correct spelling in the configuration.')
        logger.info("ProMCDA will only use one pair of norm/agg functions: " + normalization + '/' + aggregation)
    else:
        logger.info("ProMCDA will use a set of different pairs of norm/agg functions")

    # Check for robustness-related configuration errors
    if robustness_on == "no":
        logger.info("ProMCDA will run without uncertainty on the indicators or weights")
    else:
        check_config_error((robustness_on_single_weights == "no" and
                            robustness_on_all_weights == "no" and
                            robustness_on_indicators == "no"),
                           'Robustness analysis has been requested, but it’s unclear whether it should be applied to '
                           'weights or indicators. Please clarify it.')

        check_config_error((robustness_on_single_weights == "yes" and
                            robustness_on_all_weights == "yes" and
                            robustness_on_indicators == "no"),
                           'Robustness analysis has been requested for the weights, but it’s unclear whether it should '
                           'be applied to all weights or just one at a time? Please clarify.')

        check_config_error(((robustness_on_single_weights == "yes" and
                             robustness_on_all_weights == "yes" and
                             robustness_on_indicators == "yes") or
                            (robustness_on_single_weights == "yes" and
                             robustness_on_all_weights == "no" and
                             robustness_on_indicators == "yes") or
                            (robustness_on_single_weights == "no" and
                             robustness_on_all_weights == "yes" and
                             robustness_on_indicators == "yes")),
                           'Robustness analysis has been requested, but it’s unclear whether it should be applied to '
                           'weights or indicators. Please clarify.')

        # Check seetings for robustness analysis on weights or indicators
        condition_robustness_on_weights = (
                (robustness_on_single_weights == 'yes' and
                 robustness_on_all_weights == 'no' and
                 robustness_on_indicators == 'no') or
                (robustness_on_single_weights == 'no' and
                 robustness_on_all_weights == 'yes' and
                 robustness_on_indicators == 'no'))

        condition_robustness_on_indicators = (
            (robustness_on_single_weights == 'no' and
             robustness_on_all_weights == 'no' and
             robustness_on_indicators == 'yes'))

        is_robustness_weights, is_robustness_indicators = check_config_setting(condition_robustness_on_weights,
                                                                               condition_robustness_on_indicators,
                                                                               monte_carlo_runs, random_seed)

    # Check the input matrix for duplicated rows in the alternatives,
    # rescale negative indicator values and drop the column containing the alternatives
    input_matrix_no_alternatives = check_input_matrix(input_matrix)

    if is_robustness_indicators == 0:
        num_indicators = input_matrix_no_alternatives.shape[1]
    else:
        num_non_exact_and_non_poisson = \
            len(marginal_distribution) - marginal_distribution.count('exact') - marginal_distribution.count('poisson')
        num_indicators = (input_matrix_no_alternatives.shape[1] - num_non_exact_and_non_poisson)

    # Process indicators and weights based on input parameters in the configuration
    polar, weights = process_indicators_and_weights(extracted_values, input_matrix_no_alternatives,
                                                    is_robustness_indicators,
                                                    is_robustness_weights, polarity, monte_carlo_runs, num_indicators)

    # Check the number of indicators, weights, and polarities
    try:
        check_indicator_weights_polarities(num_indicators, polar, extracted_values)
    except ValueError as e:
        logging.error(str(e), stack_info=True)
        raise

    return is_robustness_indicators, is_robustness_weights, polar, weights


def check_config_error(condition: bool, error_message: str):
    """
    Check a condition and raise a ValueError with a specified error message if the condition is True.

    Parameters:
    - condition (bool): The condition to check.
    - error_message (str): The error message to raise if the condition is True.

    Raises:
    - ValueError: If the condition is True, with the specified error message.

    :param error_message: str
    :param condition: bool
    :return: None
    """

    if condition:
        logger.error('Error Message', stack_info=True)
        raise ValueError(error_message)


def check_config_setting(condition_robustness_on_weights: bool, condition_robustness_on_indicators: bool, mc_runs: int,
                         random_seed: int) -> (int, int):
    """
    Checks configuration settings and logs information messages.

    Returns:
    - is_robustness_weights, is_robustness_indicators, booleans indicating if robustness is considered
    on weights or indicators.

    Example:
    ```python
    is_robustness_weights, is_robustness_indicators = check_config_setting(True, False, 1000, 42)
    ```

    :param condition_robustness_on_weights: bool
    :param condition_robustness_on_indicators: bool
    :param mc_runs: int
    :param random_seed: int
    :return: (is_robustness_weights, is_robustness_indicators)
    :rtype: Tuple[int, int]
    """
    is_robustness_weights = 0
    is_robustness_indicators = 0

    if condition_robustness_on_weights:
        logger.info("ProMCDA will consider uncertainty on the weights.")
        logger.info("Number of Monte Carlo runs: {}".format(mc_runs))
        logger.info("The random seed used is: {}".format(random_seed))
        is_robustness_weights = 1

    elif condition_robustness_on_indicators:
        logger.info("ProMCDA will consider uncertainty on the indicators.")
        logger.info("Number of Monte Carlo runs: {}".format(mc_runs))
        logger.info("The random seed used is: {}".format(random_seed))
        is_robustness_indicators = 1

    return is_robustness_weights, is_robustness_indicators


def process_indicators_and_weights(config: dict, input_matrix: pd.DataFrame,
                                   is_robustness_indicators: int, is_robustness_weights: int, polar: List[str],
                                   mc_runs: int, num_indicators: int) \
        -> Tuple[List[str], Union[list, List[list], dict]]:
    """
    Process indicators and weights based on input parameters in the configuration.

    Parameters:
    - config: the configuration dictionary.
    - input_matrix: the input matrix without alternatives.
    - is_robustness_indicators: a flag indicating whether the matrix should include indicator uncertainties
      (0 or 1).
    - is_robustness_weights: a flag indicating whether robustness analysis is considered for the weights (0 or 1).
    - marginal_pdf: a list of marginal probability density functions for indicators.
    - mc_runs: number of Monte Carlo runs for robustness analysis.
    - num_indicators: the number of indicators in the input matrix.

    Raises:
    - ValueError: If there are duplicated rows in the input matrix or if there is an issue with the configuration.

    Returns:
    - a shorter list of polarities if one has been dropped together with the relative indicator,
      which brings no information. Otherwise, the same list.
    - the normalised weights (either fixed or random sampled weights, depending on the settings)

    Notes:
    - For is_robustness_indicators == 0:
    - Identifies and removes columns with constant values.
    - Logs the number of alternatives and indicators.

    - For is_robustness_indicators == 1:
    - Handles uncertainty in indicators.
    - Logs the number of alternatives and indicators.

    - For is_robustness_weights == 0:
    - Processes fixed weights if given.
    - Logs weights and normalised weights.

    - For is_robustness_weights == 1:
    - Performs robustness analysis on weights.
    - Logs randomly sampled weights.

    :param mc_runs: int
    :param polar: List[str]
    :param is_robustness_weights: int
    :param is_robustness_indicators: int
    :param input_matrix: pd.DataFrame
    :param config: dict
    :param num_indicators: int
    :return: polar, norm_weights
    :rtype: Tuple[List[str], Union[List[list], dict]]
    """
    num_unique = input_matrix.nunique()
    cols_to_drop = num_unique[num_unique == 1].index
    col_to_drop_indexes = input_matrix.columns.get_indexer(cols_to_drop)

    if is_robustness_indicators == 0:
        _handle_no_robustness_indicators(input_matrix)
    else:  # matrix with uncertainty on indicators
        logger.info("Number of alternatives: {}".format(input_matrix.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
        # TODO: eliminate indicators with constant values (i.e. same mean and 0 std) - optional

    polarities_and_weights = _handle_polarities_and_weights(is_robustness_indicators, is_robustness_weights, num_unique,
                                                            col_to_drop_indexes, polar, config, mc_runs, num_indicators)

    polar, norm_weights = tuple(item for item in polarities_and_weights if item is not None)

    return polar, norm_weights


def _handle_polarities_and_weights(is_robustness_indicators: int, is_robustness_weights: int, num_unique,
                                   col_to_drop_indexes: np.ndarray, polar: List[str], config: dict, mc_runs: int,
                                   num_indicators: int) \
        -> Union[Tuple[List[str], list, None, None], Tuple[List[str], None, List[List], None],
        Tuple[List[str], None, None, dict]]:
    """
    Manage polarities and weights based on the specified robustness settings, ensuring that the appropriate adjustments
    and normalizations are applied before returning the necessary data structures.
    """
    norm_random_weights = []
    rand_weight_per_indicator = {}

    # Managing polarities
    if is_robustness_indicators == 0:
        if any(value == 1 for value in num_unique):
            polar = pop_indexed_elements(col_to_drop_indexes, polar)
    logger.info("Polarities: {}".format(polar))

    # Managing weights
    if is_robustness_weights == 0:
        fixed_weights = config["given_weights"]
        if any(value == 1 for value in num_unique):
            fixed_weights = pop_indexed_elements(col_to_drop_indexes, fixed_weights)
        norm_fixed_weights = check_norm_sum_weights(fixed_weights)
        logger.info("Weights: {}".format(fixed_weights))
        logger.info("Normalized weights: {}".format(norm_fixed_weights))
        return polar, norm_fixed_weights, None, None
        #  Return None for norm_random_weights and rand_weight_per_indicator
    else:
        output_weights = _handle_robustness_weights(config, mc_runs, num_indicators)
        if output_weights is not None:
            norm_random_weights, rand_weight_per_indicator = output_weights
        if norm_random_weights:
            return polar, None, norm_random_weights, None
        else:
            return polar, None, None, rand_weight_per_indicator
        #  Return None for norm_fixed_weights and one of the other two cases of randomness


def _handle_robustness_weights(config: dict, mc_runs: int, num_indicators: int) \
        -> Tuple[Union[List[list], None], Union[dict, None]]:
    """
    Handle the generation and normalization of random weights based on the specified settings
    when a robustness analysis is requested on all the weights.
    """
    norm_random_weights = []
    rand_weight_per_indicator = {}

    if mc_runs == 0:
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')

    if config["robustness_on_single_weights"] == "no" and config["robustness_on_all_weights"] == "yes":
        random_weights = randomly_sample_all_weights(num_indicators, mc_runs)
        for weights in random_weights:
            weights = check_norm_sum_weights(weights)
            norm_random_weights.append(weights)
        return norm_random_weights, None  # Return norm_random_weights, and None for rand_weight_per_indicator
    elif config["robustness_on_single_weights"] == "yes" and config["robustness_on_all_weights"] == "no":
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


def check_indicator_weights_polarities(num_indicators: int, polar: List[str], config: dict):
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

    :param num_indicators: int
    :param polar: List[str]
    :param config: dict
    :return: None
    """
    if num_indicators != len(polar):
        raise ValueError('The number of polarities does not correspond to the no. of indicators')

    # Check the number of fixed weights if "robustness_on_all_weights" is set to "no"
    if (config["robustness_on_all_weights"] == "no") and (
            num_indicators != len(config["given_weights"])):
        raise ValueError('The no. of fixed weights does not correspond to the no. of indicators')
