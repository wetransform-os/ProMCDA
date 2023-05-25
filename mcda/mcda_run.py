#! /usr/bin/env python3

import logging
import time
import sys
import os

from mcda.configuration.config import Config
from mcda.utils import *
from mcda.mcda_without_variability import MCDAWithoutVar

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("MCDTool")

def main(input_config: dict):
    logger.info("Loading the configuration file")
    config = Config(input_config)

    logger.info("Read input matrix at {}".format(config.input_matrix_path))
    input_matrix = read_matrix(config.input_matrix_path)
    logger.info("Alternatives are {}".format(input_matrix.iloc[:,0].tolist()))
    input_matrix_no_alternatives = input_matrix.drop(input_matrix.columns[0],axis=1) # drop first column with alternatives

    logger.info("Marginal distributions: {}".format(config.marginal_distribution_for_each_indicator))
    marginal_pdf = config.marginal_distribution_for_each_indicator

    if all(element == 'exact' for element in marginal_pdf):
       # every column of the input matrix represents an indicator
        no_indicators = input_matrix_no_alternatives.shape[1]
        logger.info("Number of alternatives: {}".format(input_matrix_no_alternatives.shape[0]))
        logger.info("Number of indicators: {}".format(no_indicators))
    else:
        # non-exact indicators in the input matrix are associated to a column representing its mean
        # and a second column representing its std
        no_non_exact = len(marginal_pdf) - marginal_pdf.count('exact')
        no_indicators = input_matrix_no_alternatives.shape[1]-no_non_exact
        logger.info("Number of alternatives: {}".format(input_matrix_no_alternatives.shape[0]))
        logger.info("Number of indicators: {}".format(no_indicators))

    logger.info("Number of Monte Carlo runs: {}".format(config.monte_carlo_runs))
    mc_runs = config.monte_carlo_runs

    logger.info("Polarities: {}".format(config.polarity_for_each_indicator))
    polar = config.polarity_for_each_indicator

    logger.info("Weights: {}".format(config.weight_for_each_indicator))
    weights = config.weight_for_each_indicator

    if sum(weights) != 1:
        weights = [val/sum(weights) for val in weights] # the normalization is perfomed again in Aggregation, here is only for logging purposes
        weights_rounded = [round(elem, 2) for elem in weights]
        logger.info("The sum of the weights of the indicators is not equal to 1, their values have been normalized: {}".format(weights_rounded))

    cores = config.no_cores

    # checks on the number of indicators, weights, and polarities
    if no_indicators != len(polar):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The no. of polarities does not correspond to the no. of indicators')
    if no_indicators != len(weights):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The no. of weights does not correspond to the no. of indicators')

    # checks on the settings for non-variability/variability
    if (len(set(config.marginal_distribution_for_each_indicator))==1):
        if (config.monte_carlo_runs > 0):
            logger.error('Error Message', stack_info=True)
            raise ValueError('If the number of Monte-Carlo runs is larger than 0, at least some of the marginal distributions are expected to be non-exact')
        else:
            logger.info("Start MCDA without variability")
            # estimate the scores
            mcda_no_var = MCDAWithoutVar(config, input_matrix_no_alternatives)
            normalized_indicators = mcda_no_var.normalize_indicators()
            scores = mcda_no_var.aggregate_indicators(normalized_indicators, config.weight_for_each_indicator)
            # normalize the scores
            normalized_scores = rescale_minmax(scores)
            normalized_scores.insert(0, 'Alternatives', input_matrix.iloc[:,0])
            # estimate the ranks
            ranks = scores.rank(pct=True)
            ranks.insert(0, 'Alternatives', input_matrix.iloc[:,0])
            # save output files
            logger.info("Saving results in {}".format(config.output_file_path))
            check_path_exists(config.output_file_path)
            scores.insert(0, 'Alternatives', input_matrix.iloc[:,0])
            save_df(scores, config.output_file_path, 'scores.csv')
            save_df(normalized_scores, config.output_file_path, 'normalized_scores.csv')
            save_df(ranks, config.output_file_path, 'ranks.csv')
            save_config(input_config, config.output_file_path, 'configuration.json')
            # plots
            plot_norm_scores = plot_norm_scores_without_uncert(normalized_scores)
            save_figure(plot_norm_scores, config.output_file_path, "MCDA_no_var.png")
            logger.info("Finished MCDA without variability: check the output files")
    else:
        if (config.monte_carlo_runs > 0):
            if (config.monte_carlo_runs < 1000):
                logger.info("The number of Monte-Carlo runs is only {}".format(config.monte_carlo_runs))
                logger.info("A meaningful number of Monte-Carlo runs is equal or larger than 1000")
                time.sleep(5)
                # variability
                logger.info("Start MCDA with variability")
            else:
                # variability
                logger.info("Start MCDA with variability")
        else:
            logger.error('Error Message', stack_info=True)
            raise ValueError('If the number of Monte-Carlo runs is 0, all marginal distributions are expected to be exact')

    # logger.info("Saving results in {}".format(config.output_path))
    # save_result(res, config.output_path)

if __name__ == '__main__':
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config=input_config)
