#! /usr/bin/env python3

import logging
import time
import sys
import os

from mcda.configuration.config import Config
from mcda.utils import *
from mcda.utils_for_parallelization import *
from mcda.mcda_without_variability import MCDAWithoutVar

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("MCDTool")

def main(input_config: dict):
    logger.info("Loading the configuration file")
    config = Config(input_config)

    logger.info("Read input matrix at {}".format(config.input_matrix_path))
    input_matrix = read_matrix(config.input_matrix_path)
    if input_matrix.duplicated().any():
        logger.error('Error Message', stack_info=True)
        raise ValueError('There are duplicated rows in the input matrix')
    elif input_matrix.iloc[:,0].duplicated().any():
        logger.error('Error Message', stack_info=True)
        raise ValueError('There are duplicated rows in the alternatives column')
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
    if config.weight_for_each_indicator["random_weights"] == "no":
        fixed_weights = config.weight_for_each_indicator["given_weights"]
        norm_fixed_weights = check_norm_sum_weights(fixed_weights)
    else:
        no_runs = config.weight_for_each_indicator["no_samples"]
        is_random_w_iterative = config.weight_for_each_indicator["iterative"]
        if is_random_w_iterative == "no":
            random_weights = randomly_sample_all_weights(no_indicators, no_runs)
            norm_random_weights = []
            for weights in random_weights:
                weights = check_norm_sum_weights(weights)
                norm_random_weights.append(weights)
        else:
            i=0
            rand_weight_per_indicator = {}
            while i< no_indicators:
                random_weights = randomly_sample_ix_weight(no_indicators, i, no_runs)
                norm_random_weight = []
                for weights in random_weights:
                    weights = check_norm_sum_weights(weights)
                    norm_random_weight.append(weights)
                rand_weight_per_indicator["indicator_{}".format(i+1)] = norm_random_weight
                i=i+1

    cores = config.no_cores

    # checks on the number of indicators, weights, and polarities
    if no_indicators != len(polar):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The no. of polarities does not correspond to the no. of indicators')
    if no_indicators != len(config.weight_for_each_indicator["given_weights"]):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The no. of fixed weights does not correspond to the no. of indicators')

    # non-variability/variability
    if (len(set(config.marginal_distribution_for_each_indicator))==1):
        if (config.monte_carlo_runs > 0):
            logger.error('Error Message', stack_info=True)
            raise ValueError('If the number of Monte-Carlo runs is larger than 0, at least some of the marginal distributions are expected to be non-exact')
        # NO VARIABILITY OF INDICATORS
        else: # MC runs = 0
            scores = pd.DataFrame()
            all_weights_means = pd.DataFrame()
            logger.info("Start MCDA without variability for the indicators")
            t = time.time()
            mcda_no_var = MCDAWithoutVar(config, input_matrix_no_alternatives)
            # normalize the indicators
            normalized_indicators = mcda_no_var.normalize_indicators()
            # estimate the scores through aggregation
            if config.weight_for_each_indicator["random_weights"] == "no": # FIXED WEIGHTS
                scores = mcda_no_var.aggregate_indicators(normalized_indicators, norm_fixed_weights)
            elif config.weight_for_each_indicator["random_weights"] == "yes":
                if is_random_w_iterative == "no": # ALL RANDOMLY SAMPLED WEIGHTS (MCDA runs no_samples times)
                    logger.info("All weights are randomly sampled from a uniform distribution")
                    args_for_parallel_agg = [(lst, normalized_indicators) for lst in norm_random_weights]
                    all_weights = parallelize_aggregation(args_for_parallel_agg)
                    all_weights_means, all_weights_stds = estimate_runs_mean_std(all_weights)
                elif is_random_w_iterative == "yes": # ONE RANDOMLY SAMPLED WEIGHT A TIME (MCDA runs (no_samples * no_indicators) times)
                    logger.info("One weight at time is randomly sampled from a uniform distribution")
                    iterative_random_w_means = {}
                    iterative_random_w_stds = {}
                    for index in range(no_indicators):
                        norm_one_random_weight = rand_weight_per_indicator["indicator_{}".format(index+1)]
                        args_for_parallel_agg = [(lst, normalized_indicators) for lst in norm_one_random_weight]
                        one_random_weight = parallelize_aggregation(args_for_parallel_agg)
                        one_random_weight_means, one_random_weight_stds = estimate_runs_mean_std(one_random_weight)
                        iterative_random_w_means["indicator_{}".format(index+1)] = one_random_weight_means
                        iterative_random_w_stds["indicator_{}".format(index+1)] = one_random_weight_stds
            # normalize the output scores (no randomness)
            if not scores.empty: # no randomly sampled weights
                normalized_scores = rescale_minmax(scores)
                normalized_scores.insert(0, 'Alternatives', input_matrix.iloc[:,0])
            elif not all_weights_means.empty: # all randomly sampled weights
                all_weights_means.insert(0, 'Alternatives', input_matrix.iloc[:,0])
                all_weights_stds.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            elif not bool(iterative_random_w_means) == 'False': # one randomly sampled weight at time
                for index in range(no_indicators):
                    one_random_weight_means = iterative_random_w_means["indicator_{}".format(index+1)]
                    one_random_weight_stds = iterative_random_w_stds["indicator_{}".format(index+1)]
                    one_random_weight_means.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                    one_random_weight_stds.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                    iterative_random_w_means["indicator_{}".format(index + 1)] = one_random_weight_means
                    iterative_random_w_stds["indicator_{}".format(index + 1)] = one_random_weight_stds
            # estimate the ranks
            if not scores.empty:
                ranks = scores.rank(pct=True)
                ranks.insert(0, 'Alternatives', input_matrix.iloc[:,0])
            elif not all_weights_means.empty:
                ranks = all_weights_means.rank(pct=True)
                ranks.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            elif not bool(iterative_random_w_means) == 'False':
                pass
            # save output files
            logger.info("Saving results in {}".format(config.output_file_path))
            check_path_exists(config.output_file_path)
            if not scores.empty:
                scores.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                save_df(scores, config.output_file_path, 'scores.csv')
                save_df(normalized_scores, config.output_file_path, 'normalized_scores.csv')
                save_df(ranks, config.output_file_path, 'ranks.csv')
            elif not all_weights_means.empty:
                save_df(all_weights_means, config.output_file_path, 'score_means.csv')
                save_df(all_weights_stds, config.output_file_path, 'score_stds.csv')
                save_df(ranks, config.output_file_path, 'ranks.csv')
            elif not bool(iterative_random_w_means) == 'False':
                save_dict(iterative_random_w_means,config.output_file_path, 'score_means.pkl')
                save_dict(iterative_random_w_stds, config.output_file_path, 'score_stds.pkl')
            save_config(input_config, config.output_file_path, 'configuration.json')
            # plots
            if not scores.empty:
                plot_norm_scores = plot_norm_scores_without_uncert(normalized_scores)
                save_figure(plot_norm_scores, config.output_file_path, "MCDA_norm_scores_no_var.png")
                plot_no_norm_scores = plot_non_norm_scores_without_uncert(scores)
                save_figure(plot_no_norm_scores, config.output_file_path, "MCDA_rough_scores_no_var.png")
            elif not all_weights_means.empty:
                plot_weight_mean_scores = plot_mean_scores(all_weights_means, all_weights_stds)
                save_figure(plot_weight_mean_scores, config.output_file_path, "MCDA_weights_var.png")
            elif not bool(iterative_random_w_means) == 'False':
                images = []
                for index in range(no_indicators):
                    one_random_weight_means = iterative_random_w_means["indicator_{}".format(index + 1)]
                    one_random_weight_stds = iterative_random_w_stds["indicator_{}".format(index + 1)]
                    plot_weight_mean_scores = plot_mean_scores_iterative(one_random_weight_means, one_random_weight_stds, input_matrix_no_alternatives.columns, index)
                    print(type(plot_weight_mean_scores))
                    images.append(plot_weight_mean_scores)
                combine_images(images, config.output_file_path, "MCDA_one_weight_randomness.png")
            logger.info("Finished MCDA without variability: check the output files")
            elapsed = time.time() - t
            logger.info("All calculations finished in seconds {}".format(elapsed))
    # VARIABILITY OF INDICATORS
    else: # if some marginal distributions are not exact
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


if __name__ == '__main__':
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config=input_config)
