#! /usr/bin/env python3

import logging
import time
import sys
import os

from mcda.configuration.config import Config
from mcda.utils import *
from mcda.utils_for_parallelization import *
from mcda.mcda_without_variability import MCDAWithoutVar
from mcda.mcda_with_variability import MCDAWithVar

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("MCDTool")

def main(input_config: dict):
    logger.info("Loading the configuration file")
    config = Config(input_config)
    marginal_pdf = config.marginal_distribution_for_each_indicator
    if all(element == 'exact' for element in marginal_pdf):
        logger.info("MCDA will be run without uncertainty on the indicators")
        is_uncertainty = 0
        logger.info("Read input matrix without uncertainty at {}".format(config.input_matrix_path))
    else:
        logger.info("MCDA will be run by considering uncertainty on the indicators")
        is_uncertainty = 1
        logger.info("Read input matrix with uncertainty at {}".format(config.input_matrix_path))
    # read input matrix with or without uncertainty, if needed preprocess it
    input_matrix = read_matrix(config.input_matrix_path)
    if (len(input_matrix.columns) - 1) % 2 != 0: # Alternatives column is still in, therefore -1
        raise ValueError("Number of columns for non exact indicators in the input matrix must be even.")
    if input_matrix.duplicated().any():
        logger.error('Error Message', stack_info=True)
        raise ValueError('There are duplicated rows in the input matrix')
    elif input_matrix.iloc[:,0].duplicated().any():
        logger.error('Error Message', stack_info=True)
        raise ValueError('There are duplicated rows in the alternatives column')
    logger.info("Alternatives are {}".format(input_matrix.iloc[:, 0].tolist()))
    input_matrix_no_alternatives = input_matrix.drop(input_matrix.columns[0],axis=1)  # drop first column with alternatives
    if is_uncertainty == 0:
        num_unique = input_matrix_no_alternatives.nunique() # search for column with constant values
        cols_to_drop = num_unique[num_unique == 1].index
        col_to_drop_indexes = input_matrix_no_alternatives.columns.get_indexer(cols_to_drop)
        if (num_unique.any() == 1): logger.info("Indicators {} have been dropped because they carry no information".format(cols_to_drop))
        input_matrix_no_alternatives = input_matrix_no_alternatives.drop(cols_to_drop, axis=1)
        if (num_unique.any() == 1): marginal_pdf = pop_indexed_elements(col_to_drop_indexes, marginal_pdf)
        logger.info("Marginal distributions: {}".format(marginal_pdf))
        # every column of the input matrix represents an indicator
        num_indicators = input_matrix_no_alternatives.shape[1]
        logger.info("Number of alternatives: {}".format(input_matrix_no_alternatives.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
    # matrix with uncertainty
    else:
        # non-exact indicators in the input matrix are associated to a column representing its mean
        # and a second column representing its std
        num_non_exact = len(marginal_pdf) - marginal_pdf.count('exact')
        num_indicators = input_matrix_no_alternatives.shape[1]-input_matrix_no_alternatives.shape[1]/2
        logger.info("Number of alternatives: {}".format(input_matrix_no_alternatives.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
        # TODO: eliminate indicators with constant values (i.e. same mean and 0 std) - optional

    logger.info("Number of Monte Carlo runs: {}".format(config.monte_carlo_runs))
    mc_runs = config.monte_carlo_runs

    polar = config.polarity_for_each_indicator
    if (is_uncertainty == 0):
        if (num_unique.any() == 1): polar = pop_indexed_elements(col_to_drop_indexes, polar)
    logger.info("Polarities: {}".format(polar))

    if config.weight_for_each_indicator["random_weights"] == "no":
        fixed_weights = config.weight_for_each_indicator["given_weights"]
        if (is_uncertainty == 0):
            if (num_unique.any() == 1): fixed_weights = pop_indexed_elements(col_to_drop_indexes,fixed_weights)
        norm_fixed_weights = check_norm_sum_weights(fixed_weights)
        logger.info("Weights: {}".format(fixed_weights))
        logger.info("Normalized weights: {}".format(norm_fixed_weights))
    else:
        num_runs = config.weight_for_each_indicator["num_samples"]
        is_random_w_iterative = config.weight_for_each_indicator["iterative"]
        if is_random_w_iterative == "no":
            random_weights = randomly_sample_all_weights(num_indicators, num_runs)
            norm_random_weights = []
            for weights in random_weights:
                weights = check_norm_sum_weights(weights)
                norm_random_weights.append(weights)
        else:
            i=0
            rand_weight_per_indicator = {}
            while i< num_indicators:
                random_weights = randomly_sample_ix_weight(num_indicators, i, num_runs)
                norm_random_weight = []
                for weights in random_weights:
                    weights = check_norm_sum_weights(weights)
                    norm_random_weight.append(weights)
                rand_weight_per_indicator["indicator_{}".format(i+1)] = norm_random_weight
                i=i+1

    cores = config.num_cores

    # checks on the number of indicators, weights, and polarities
    if num_indicators != len(polar):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of polarities does not correspond to the no. of indicators')
    if ((config.weight_for_each_indicator["random_weights"] == "no") & (num_indicators != len(config.weight_for_each_indicator["given_weights"]))):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The no. of fixed weights does not correspond to the no. of indicators')
    # ----------------------------
    # NO VARIABILITY OF INDICATORS
    # ----------------------------
    if is_uncertainty == 0:
        if (config.monte_carlo_runs > 0):
            logger.error('Error Message', stack_info=True)
            raise ValueError('If the number of Monte-Carlo runs is larger than 0, at least some of the marginal distributions are expected to be non-exact')
        else: # MC runs = 0
            logger.info("Start MCDA without variability for the indicators")
            t = time.time()
            scores = pd.DataFrame()
            all_weights_means = pd.DataFrame()
            mcda_no_var = MCDAWithoutVar(config, input_matrix_no_alternatives)
            # normalize the indicators
            normalized_indicators = mcda_no_var.normalize_indicators()
            # estimate the scores through aggregation
            if config.weight_for_each_indicator["random_weights"] == "no": # FIXED WEIGHTS
                scores = mcda_no_var.aggregate_indicators(normalized_indicators, norm_fixed_weights)
                normalized_scores = rescale_minmax(scores) # normalized scores
                normalized_scores.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            elif config.weight_for_each_indicator["random_weights"] == "yes":
                if is_random_w_iterative == "no": # ALL RANDOMLY SAMPLED WEIGHTS (MCDA runs num_samples times)
                    logger.info("All weights are randomly sampled from a uniform distribution")
                    all_weights_normalized = []
                    args_for_parallel_agg = [(lst, normalized_indicators) for lst in norm_random_weights]
                    all_weights = parallelize_aggregation(args_for_parallel_agg) # rough scores coming from all runs
                    for matrix in all_weights: # rescale the scores coming from all runs
                        normalized_matrix = rescale_minmax(matrix) # all score normalization
                        all_weights_normalized.append(normalized_matrix)
                    all_weights_means, all_weights_stds = estimate_runs_mean_std(all_weights) # mean and std of rough scores
                    all_weights_means_normalized, all_weights_stds_normalized = estimate_runs_mean_std(all_weights_normalized) # mean and std of norm. scores
                    all_weights_stds.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                    all_weights_means_normalized.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                    all_weights_stds_normalized.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                elif is_random_w_iterative == "yes": # ONE RANDOMLY SAMPLED WEIGHT A TIME (MCDA runs (num_samples * num_indicators) times)
                    logger.info("One weight at time is randomly sampled from a uniform distribution")
                    scores_one_random_weight_normalized = []
                    iterative_random_w_means = {}
                    iterative_random_w_stds = {}
                    iterative_random_w_means_normalized = {}
                    iterative_random_w_stds_normalized = {}
                    for index in range(num_indicators):
                        norm_one_random_weight = rand_weight_per_indicator["indicator_{}".format(index+1)] # 'norm' refers to all weights, which are normalized
                        args_for_parallel_agg = [(lst, normalized_indicators) for lst in norm_one_random_weight]
                        scores_one_random_weight = parallelize_aggregation(args_for_parallel_agg)
                        for matrix in scores_one_random_weight:
                            matrix_normalized = rescale_minmax(matrix) # normalize scores
                            scores_one_random_weight_normalized.append(matrix_normalized)
                        one_random_weight_means, one_random_weight_stds = estimate_runs_mean_std(scores_one_random_weight)
                        one_random_weight_means_normalized, one_random_weight_stds_normalized = estimate_runs_mean_std(scores_one_random_weight_normalized) # normalized mean and std
                        one_random_weight_means.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                        one_random_weight_stds.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                        one_random_weight_means_normalized.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                        one_random_weight_stds_normalized.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                        iterative_random_w_means["indicator_{}".format(index+1)] = one_random_weight_means # create output dictionaries
                        iterative_random_w_stds["indicator_{}".format(index+1)] = one_random_weight_stds
                        iterative_random_w_means_normalized["indicator_{}".format(index + 1)] = one_random_weight_means_normalized
                        iterative_random_w_stds_normalized["indicator_{}".format(index + 1)] = one_random_weight_stds_normalized
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
                all_weights_means.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
                save_df(all_weights_means, config.output_file_path, 'score_means.csv')
                save_df(all_weights_stds, config.output_file_path, 'score_stds.csv')
                save_df(all_weights_means_normalized, config.output_file_path, 'score_means_normalized.csv')
                #save_df(all_weights_stds_normalized, config.output_file_path, 'score_stds_normalized.csv')
                # the std on rescaled values is not statistically informative
                save_df(ranks, config.output_file_path, 'ranks.csv')
            elif not bool(iterative_random_w_means) == 'False':
                save_dict(iterative_random_w_means,config.output_file_path, 'score_means.pkl')
                save_dict(iterative_random_w_stds, config.output_file_path, 'score_stds.pkl')
                save_dict(iterative_random_w_means_normalized, config.output_file_path, 'score_means_normalized.pkl')
                #save_dict(iterative_random_w_stds_normalized, config.output_file_path, 'score_stds_normalized.pkl')
                # the std on rescaled values is not statistically informative
            save_config(input_config, config.output_file_path, 'configuration.json')
            # plots
            if not scores.empty:
                plot_norm_scores = plot_norm_scores_without_uncert(normalized_scores)
                save_figure(plot_norm_scores, config.output_file_path, "MCDA_norm_scores_no_var.png")
                plot_no_norm_scores = plot_non_norm_scores_without_uncert(scores)
                save_figure(plot_no_norm_scores, config.output_file_path, "MCDA_rough_scores_no_var.png")
            elif not all_weights_means.empty:
                plot_weight_mean_scores = plot_mean_scores(all_weights_means, all_weights_stds, "plot_std")
                plot_weight_mean_scores_norm = plot_mean_scores(all_weights_means_normalized, all_weights_stds_normalized, "not_plot_std")
                save_figure(plot_weight_mean_scores, config.output_file_path, "MCDA_rand_weights_rough_scores.png")
                save_figure(plot_weight_mean_scores_norm, config.output_file_path, "MCDA_rand_weights_norm_scores.png")
            elif not bool(iterative_random_w_means) == 'False':
                images = []
                images_norm = []
                for index in range(num_indicators):
                    one_random_weight_means = iterative_random_w_means["indicator_{}".format(index + 1)]
                    one_random_weight_stds = iterative_random_w_stds["indicator_{}".format(index + 1)]
                    one_random_weight_means_normalized = iterative_random_w_means_normalized["indicator_{}".format(index + 1)]
                    one_random_weight_stds_normalized = iterative_random_w_stds_normalized["indicator_{}".format(index + 1)]
                    plot_weight_mean_scores = plot_mean_scores_iterative(one_random_weight_means,
                                                                         one_random_weight_stds,
                                                                         input_matrix_no_alternatives.columns, index, "plot_std")
                    plot_weight_mean_scores_norm = plot_mean_scores_iterative(one_random_weight_means_normalized,
                                                                         one_random_weight_stds_normalized,
                                                                         input_matrix_no_alternatives.columns, index, "not_plot_std")
                    images.append(plot_weight_mean_scores)
                    images_norm.append(plot_weight_mean_scores_norm)
                combine_images(images, config.output_file_path, "MCDA_one_weight_randomness_rough_scores.png")
                combine_images(images_norm, config.output_file_path, "MCDA_one_weight_randomness_norm_scores.png")
            logger.info("Finished MCDA without variability: check the output files")
            elapsed = time.time() - t
            logger.info("All calculations finished in seconds {}".format(elapsed))
    # -------------------------
    # VARIABILITY OF INDICATORS
    # -------------------------
    else: # if is_variability == 1
        all_indicators_normalized = []
        if (config.monte_carlo_runs > 0):
            if (config.monte_carlo_runs < 1000):
                logger.info("The number of Monte-Carlo runs is only {}".format(config.monte_carlo_runs))
                logger.info("A meaningful number of Monte-Carlo runs is equal or larger than 1000")
                time.sleep(5)
            logger.info("Start MCDA with variability on the indicators")
            mcda_with_var = MCDAWithVar(config, input_matrix_no_alternatives)
            n_random_input_matrices = mcda_with_var.create_n_randomly_sampled_matrices() # N random matrices
            n_normalized_input_matrices = parallelize_normalization(n_random_input_matrices, polar) # parallel normalization
            args_for_parallel_agg = [(norm_fixed_weights, normalized_indicators) for normalized_indicators in n_normalized_input_matrices] # weights are fixed
            all_indicators = parallelize_aggregation(args_for_parallel_agg)  # rough scores coming from all runs with random indicator values
            for matrix in all_indicators:  # rescale the scores coming from all runs
                normalized_matrix = rescale_minmax(matrix)  # all score normalization
                all_indicators_normalized.append(normalized_matrix)
            all_indicators_means, all_indicators_stds = estimate_runs_mean_std(all_indicators)  # mean and std of rough scores
            all_indicators_means_normalized, all_indicators_stds_normalized = estimate_runs_mean_std(all_indicators_normalized)  # mean and std of norm. scores
            # estimate the ranks
            ranks = all_indicators_means.rank(pct=True)
            # re-insert the Alternatives column
            all_indicators_means.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            all_indicators_stds.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            all_indicators_means_normalized.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            all_indicators_stds_normalized.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            ranks.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            # save output files
            logger.info("Saving results in {}".format(config.output_file_path))
            check_path_exists(config.output_file_path)
            save_df(all_indicators_means, config.output_file_path, 'score_means.csv')
            save_df(all_indicators_stds, config.output_file_path, 'score_stds.csv')
            save_df(all_indicators_means_normalized, config.output_file_path, 'score_means_normalized.csv')
            save_df(ranks, config.output_file_path, 'ranks.csv')
            save_config(input_config, config.output_file_path, 'configuration.json')
            # plots
            plot_indicators_mean_scores = plot_mean_scores(all_indicators_means, all_indicators_stds, "plot_std")
            plot_indicators_mean_scores_norm = plot_mean_scores(all_indicators_means_normalized,
                                                                all_indicators_stds_normalized, "not_plot_std")
            save_figure(plot_weight_mean_scores, config.output_file_path, "MCDA_rand_indicators_rough_scores.png")
            save_figure(plot_weight_mean_scores_norm, config.output_file_path, "MCDA_rand_indicators_norm_scores.png")
            logger.info("Finished MCDA with variability of the indicators: check the output files")
            elapsed = time.time() - t
            logger.info("All calculations finished in seconds {}".format(elapsed))
        else:
            logger.error('Error Message', stack_info=True)
            raise ValueError('If the number of Monte-Carlo runs is 0, all marginal distributions are expected to be exact')


if __name__ == '__main__':
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config=input_config)
