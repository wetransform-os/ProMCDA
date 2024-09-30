#! /usr/bin/env python3

"""
This script serves as the main entry point for running all pieces of functionality in a consequential way by
following the settings given in the configuration file 'configuration.json'.

Usage (from root directory):
    $ python3 -m ProMCDA.mcda.mcda_ranking_run -c ProMCDA/input_files/toy_example/car-data-input-sample-from-ranking-service.json
"""

import time
from ProMCDA.mcda.utils.utils_for_main import *
from ProMCDA.mcda.utils.utils_for_plotting import *
from ProMCDA.mcda.utils.utils_for_parallelization import *

log = logging.getLogger(__name__)

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")

# randomly assign seed if not specified as environment variable
RANDOM_SEED = int(os.environ.get('RANDOM_SEED')) if os.environ.get('RANDOM_SEED') else 67
NUM_CORES = int(os.environ.get('NUM_CORES')) if os.environ.get('NUM_CORES') else 1


# noinspection PyTypeChecker
def main(input_config: dict) -> dict:
    """
    Execute the ProMCDA (Probabilistic Multi-Criteria Decision Analysis) process.

    Parameters:
    - input_config : Configuration parameters for the ProMCDA process.

    Raises:
    - ValueError: If there are issues with the input matrix, weights, or indicators.

    This function performs the ProMCDA process based on the provided configuration.
    It handles various aspects such as the sensitivity analysis and the robustness analysis.
    The results are saved in output files, and plots are generated to visualize the scores and rankings.

    Note: Ensure that the input matrix, weights, polarities and indicators (with or without uncertainty)
    are correctly specified in the input configuration.

    :param input_config: dict
    :return: None
    """
    is_robustness_indicators = 0
    is_robustness_weights = 0
    f_norm = None
    f_agg = None

    # Extracting relevant configuration values
    config = Configuration.from_dict(input_config)
    input_matrix = pd.DataFrame(config.input_matrix)
    index_column_name = input_matrix.index.name
    index_column_values = input_matrix.index.tolist()
    polar = config.polarity
    robustness = config.robustness.robustness
    mc_runs = config.monte_carlo_sampling.monte_carlo_runs
    marginal_pdf = config.monte_carlo_sampling.marginal_distributions

    f_agg, f_norm, is_robustness_indicators, is_robustness_weights = verify_input(config, f_agg, f_norm,
                                                                                  is_robustness_indicators,
                                                                                  is_robustness_weights,
                                                                                  mc_runs,
                                                                                  robustness,
                                                                                  RANDOM_SEED)

    # Check the input matrix for duplicated rows in the alternatives, rescale negative indicator values and
    # drop the column containing the alternatives
    input_matrix_no_alternatives = check_input_matrix(input_matrix)

    if is_robustness_indicators == 0:
        num_indicators = input_matrix_no_alternatives.shape[1]
        # Process indicators and weights based on input parameters in the configuration
        polar, weights = get_polar_and_weights(config, input_matrix_no_alternatives, is_robustness_indicators,
                                               is_robustness_weights, mc_runs, num_indicators, polar)
        return run_mcda_without_indicator_uncertainty(config, index_column_name, index_column_values,
                                                      input_matrix_no_alternatives, weights, f_norm, f_agg,
                                                      is_robustness_weights)
    else:
        num_non_exact_and_non_poisson = len(marginal_pdf) - marginal_pdf.count('exact') - marginal_pdf.count('poisson')
        num_indicators = (input_matrix_no_alternatives.shape[1] - num_non_exact_and_non_poisson)
        polar, weights = get_polar_and_weights(config, input_matrix_no_alternatives, is_robustness_indicators,
                                               is_robustness_weights, mc_runs, num_indicators, polar)
        return run_mcda_with_indicator_uncertainty(config, input_matrix_no_alternatives, index_column_name,
                                                   index_column_values, mc_runs, RANDOM_SEED,
                                                   config.sensitivity.sensitivity_on, f_agg, f_norm,
                                                   weights, polar, marginal_pdf)


def get_polar_and_weights(config, input_matrix_no_alternatives, is_robustness_indicators, is_robustness_weights,
                          mc_runs, num_indicators, polar):
    # Process indicators and weights based on input parameters in the configuration
    polar, weights = process_indicators_and_weights(config, input_matrix_no_alternatives, is_robustness_indicators,
                                                    is_robustness_weights, polar, mc_runs, num_indicators)
    try:
        check_indicator_weights_polarities(num_indicators, polar, config)
    except ValueError as e:
        logging.error(str(e), stack_info=True)
        raise
    return polar, weights


def read_matrix_from_file(column_names_list: list[str], file_from_stream) -> {}:
    result_dict = utils_for_main.read_matrix_from_file(file_from_stream).to_dict()
    if len(column_names_list) != len(result_dict.keys()):
        return {"error": "Number of provided column names does not match the CSV columns"}, 400
    return {col: result_dict[col] for col in column_names_list if col in result_dict}


if __name__ == '__main__':
    '''
    Entry point to run code with ranking-service input configuration
    command to run 
    > python3 -m ProMCDA.mcda.mcda_ranking_run -c ProMCDA/input_files/toy_example/car-data-input-sample-from-ranking-service.json
    '''
    t = time.time()
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config)
    elapsed = time.time() - t
    logger.info("All calculations finished in seconds {}".format(elapsed))
