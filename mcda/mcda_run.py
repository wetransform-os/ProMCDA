#! /usr/bin/env python3

"""
This script serves as the main entry point for running all pieces of functionality in a consequential way by
following the settings given in the configuration file 'configuration.json'.

Usage (from root directory of ranking service):
    $ python3 -m ProMCDA.mcda.mcda_run -c ProMCDA/configuration.json
"""
import time

from ProMCDA.mcda import mcda_ranking_run
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
    mcda_ranking_run.main(ranking_input_config)


if __name__ == '__main__':
    '''
    Entry class if you want to run ProMCDA from cli.
    The command to run the code must be run from outside of root directory. 
    > python3 -m ProMCDA.mcda.mcda_run -c ProMCDA/configuration.json
    '''
    t = time.time()
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config)
    elapsed = time.time() - t
    logger.info("All calculations finished in seconds {}".format(elapsed))
