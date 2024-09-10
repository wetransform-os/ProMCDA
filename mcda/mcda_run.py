#! /usr/bin/env python3

"""
This script serves as the main entry point for running all pieces of functionality in a consequential way by
following the settings given in the configuration file 'configuration.json'.

Usage (from root directory):
    $ python3 -m mcda.mcda_run -c configuration.json
"""
import json
import time

from ProMCDA.mcda import mcda_ranking_run
from ProMCDA.mcda.configuration.config import Config
from ProMCDA.mcda.utils.utils_for_main import *
from ProMCDA.mcda.utils.utils_for_parallelization import *

log = logging.getLogger(__name__)

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")


# noinspection PyTypeChecker
def main(input_config: dict):
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
    ranking_input_config = config_dict_to_configuration_model(input_config)
    mcda_ranking_run.main_using_model(ranking_input_config)


def config_dict_to_configuration_model(input_config):
    # Extracting relevant configuration values
    config = Config(input_config)
    input_matrix = read_matrix(config.input_matrix_path)
    # input_matrix = input_matrix.dropna()
    robustness = "none"
    on_weights_level = "none"
    if config.robustness["robustness_on"] == "yes" and config.robustness["on_single_weights"] == "yes":
        robustness = "weights"
        on_weights_level = "single"
    elif config.robustness["robustness_on"] == "yes" and config.robustness["on_all_weights"] == "yes":
        robustness = "weights"
        on_weights_level = "all"
    elif config.robustness["robustness_on"] == "yes" and config.robustness["on_indicators"] == "yes":
        robustness = "indicators"
    input_json = input_matrix.to_json()
    os.environ['NUM_CORES'] = str(input_config["monte_carlo_sampling"]["num_cores"])
    os.environ['RANDOM_SEED'] = str(input_config["monte_carlo_sampling"]["random_seed"])
    ranking_input_config = {
        "inputMatrix": input_matrix,
        "weights": input_config["robustness"]["given_weights"],
        "polarity": config.polarity_for_each_indicator,
        "sensitivity": {
            "sensitivityOn": input_config["sensitivity"]["sensitivity_on"],
            "normalization": input_config["sensitivity"]["normalization"],
            "aggregation": input_config["sensitivity"]["aggregation"]
        },
        "robustness": {
            "robustness": robustness,
            "onWeightsLevel": on_weights_level,
            "givenWeights": config.robustness["given_weights"]
        },
        "monteCarloSampling": {
            "monteCarloRuns": config.monte_carlo_sampling["monte_carlo_runs"],
            "marginalDistributions": config.monte_carlo_sampling["marginal_distribution_for_each_indicator"]
        }
    }
    return ranking_input_config


if __name__ == '__main__':
    t = time.time()
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config)
    elapsed = time.time() - t
    logger.info("All calculations finished in seconds {}".format(elapsed))
