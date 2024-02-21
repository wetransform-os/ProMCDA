#! /usr/bin/env python3

"""
This script serves as the main entry point for running all pieces of functionality in a consequential way by
following the settings given in the configuration file 'configuration.json'.

Usage (from root directory):
    $ python3 -m mcda.mcda_run -c configuration.json
"""

import time
import logging

from mcda.configuration.config import Config
from mcda.utils.utils_for_main import *
from mcda.utils.utils_for_plotting import *
from mcda.utils.utils_for_parallelization import *

log = logging.getLogger(__name__)

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")

# noinspection PyTypeChecker
def main(input_config: dict):
    """
        Execute the ProMCDA (Probabilistic Multi-Criteria Decision Analysis) process.

        Parameters:
        - input_config (dictionary): Configuration parameters for the ProMCDA process.

        Raises:
        - ValueError: If there are issues with the input matrix, weights, or indicators.

        This function performs the ProMCDA process based on the provided configuration.
        It handles various aspects such as the sensitivity analysis and the robustness analysis.
        The results are saved in output files, and plots are generated to visualize the scores and rankings.

        Note: Ensure that the input matrix, weights, polarities and indicators (with or without uncertainty)
        are correctly specified in the input configuration.
        """

    is_sensitivity = None
    is_robustness = None
    is_robustness_indicators = 0
    is_robustness_weights = 0
    f_norm = None
    f_agg = None
    marginal_pdf = []
    num_unique = []

    t = time.time()

    # Extracting relevant configuration values
    config = Config(input_config)
    input_matrix = read_matrix(config.input_matrix_path)
    index_column_name = input_matrix.index.name
    index_column_values = input_matrix.index.tolist()
    polar = config.polarity_for_each_indicator
    is_sensitivity = config.sensitivity['sensitivity_on']
    is_robustness = config.robustness['robustness_on']
    mc_runs = config.monte_carlo_sampling["monte_carlo_runs"]

    # Check for sensitivity-related configuration errors
    if is_sensitivity == "no":
        f_norm = config.sensitivity['normalization']
        f_agg = config.sensitivity['aggregation']
        check_config_error(f_norm not in ['minmax', 'target', 'standardized', 'rank'],
                           'The available normalization functions are: minmax, target, standardized, rank.')
        check_config_error(f_agg not in ['weighted_sum', 'geometric', 'harmonic', 'minimum'],
                           'The available aggregation functions are: weighted_sum, geometric, harmonic, minimum.'
                           '\nWatch the correct spelling in the configuration file.')
        logger.info("ProMCDA will only use one pair of norm/agg functions: " + f_norm + '/' + f_agg)
    else:
        logger.info("ProMCDA will use a set of different pairs of norm/agg functions")

    # Check for robustness-related configuration errors
    if is_robustness == "no":
        logger.info("ProMCDA will run without uncertainty on the indicators or weights")
        logger.info("Read input matrix without uncertainties at {}".format(config.input_matrix_path))
    else:
        check_config_error((config.robustness["on_single_weights"] == "no" and
                            config.robustness["on_all_weights"] == "no" and
                            config.robustness["on_indicators"] == "no"),
                           'Robustness analysis is requested but where is not defined: weights or indicators? Please clarify.')

        check_config_error((config.robustness["on_single_weights"] == "yes" and
                            config.robustness["on_all_weights"] == "yes" and
                            config.robustness["on_indicators"] == "no"),
                           'Robustness analysis is requested on the weights: but on all or one at a time? Please clarify.')

        check_config_error(((config.robustness["on_single_weights"] == "yes" and
                             config.robustness["on_all_weights"] == "yes" and
                             config.robustness["on_indicators"] == "yes") or
                            (config.robustness["on_single_weights"] == "yes" and
                             config.robustness["on_all_weights"] == "no" and
                             config.robustness["on_indicators"] == "yes") or
                            (config.robustness["on_single_weights"] == "no" and
                             config.robustness["on_all_weights"] == "yes" and
                             config.robustness["on_indicators"] == "yes")),
                           'Robustness analysis is requested: but on weights or indicators? Please clarify.')

        # Check seetings for robustness analysis on weights or indicators
        condition_robustness_on_weights = ((config.robustness["on_single_weights"] == "yes" and
                                            config.robustness["on_all_weights"] == "no" and
                                            config.robustness["on_indicators"] == "no") or
                                           (config.robustness["on_single_weights"] == "no" and
                                            config.robustness["on_all_weights"] == "yes" and
                                            config.robustness["on_indicators"] == "no"))
        condition_robustness_on_indicators = (config.robustness["on_single_weights"] == "no" and
                                              config.robustness["on_all_weights"] == "no" and
                                              config.robustness["on_indicators"] == "yes")

        condition = condition_robustness_on_weights if condition_robustness_on_weights \
                                                    else condition_robustness_on_indicators

        is_robustness_weights, is_robustness_indicators = \
            check_config_setting(condition,
                                 'ProMCDA will consider uncertainty on the weights',
                                  str(condition_robustness_on_weights),str(condition_robustness_on_indicators),
                                  mc_runs)

        is_robustness_weights, is_robustness_indicators = \
            check_config_setting(condition,
                                 'ProMCDA will consider uncertainty on the indicators',
                                 str(condition_robustness_on_weights),str(condition_robustness_on_indicators),
                                 mc_runs)

        marginal_pdf = config.monte_carlo_sampling["marginal_distribution_for_each_indicator"]
        logger.info("Read input matrix with uncertainty of the indicators at {}".format(
            config.input_matrix_path))

    # Check the input matrix for duplicated rows in the alternatives, rescale negative indicator values and
    # drop the column containing the alternatives
    input_matrix_no_alternatives = check_input_matrix(input_matrix)
    if is_robustness_indicators == 0:
        num_indicators = input_matrix_no_alternatives.shape[1]
    else:
        num_non_exact_and_non_poisson = len(marginal_pdf) - marginal_pdf.count('exact') - marginal_pdf.count('poisson')
        num_indicators = (input_matrix_no_alternatives.shape[1] - num_non_exact_and_non_poisson)

    # Process indicators and weights based on input parameters in the configuration
    polar, weights = process_indicators_and_weights(config, input_matrix_no_alternatives, is_robustness_indicators,
                                                    is_robustness_weights, polar, mc_runs, num_indicators)

    # Check the number of indicators, weights, and polarities
    try:
        check_indicator_weights_polarities(num_indicators, polar, config)
    except ValueError as e:
        logging.error(str(e), stack_info=True)
        raise

    # If there is no uncertainty of the indicators:
    if is_robustness_indicators == 0:
        run_mcda_without_indicator_uncertainty(input_config, index_column_name, index_column_values,
                                               input_matrix_no_alternatives, weights, f_norm, f_agg)
    # else (i.e. there is uncertainty):
    else:
        run_mcda_with_indicator_uncertainty(input_config, input_matrix, index_column_name, index_column_values,
                                            mc_runs, is_sensitivity, f_agg, f_norm, weights, polar, marginal_pdf)

    logger.info(
            "ProMCDA finished calculations: check the output files")
    elapsed = time.time() - t
    logger.info(
            "All calculations finished in seconds {}".format(elapsed))

if __name__ == '__main__':
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config=input_config)
