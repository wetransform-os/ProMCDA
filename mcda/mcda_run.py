#! /usr/bin/env python3

import logging
import time
import sys
import os

from mcda.configuration.config import Config
from mcda.utils import *
from mcda.utils_for_parallelization import *
from mcda.mcda_without_robustness import MCDAWithoutRobustness
from mcda.mcda_with_robustness import MCDAWithRobustness

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA")

def main(input_config: dict):

    is_sensitivity = None
    is_robustness = None
    is_robustness_indicators = 0
    is_robustness_weights = 0
    norm_random_weights = []
    marginal_pdf = []
    num_unique = []
    iterative_random_w_means = {}
    iterative_random_w_stds = {}
    iterative_random_w_means_normalized = {}
    iterative_random_w_stds_normalized = {}


    config = Config(input_config)
    input_matrix = read_matrix(config.input_matrix_path)
    print(input_matrix)
    polar = config.polarity_for_each_indicator
    is_sensitivity = config.sensitivity['sensitivity_on']
    is_robustness = config.robustness['robustness_on']
    mc_runs = config.monte_carlo_sampling["monte_carlo_runs"]
    if is_sensitivity == "no":
        f_norm = config.sensitivity['normalization']
        f_agg = config.sensitivity['aggregation']
        logger.info("ProMCDA will only use one pair of norm/agg functions: " + f_norm + '/' + f_agg)
    else:
        logger.info("ProMCDA will use a set of different pairs of norm/agg functions")
    if is_robustness == "no":
        logger.info("ProMCDA will run without uncertainty on the indicators or weights")
        logger.info("Read input matrix without uncertainties at {}".format(config.input_matrix_path))
    else: # robustness yes
        if (config.robustness["on_single_weights"] == "no"
                and config.robustness["on_all_weights"] == "no"
                and config.robustness["on_indicators"] == "no"):
            logger.error('Error Message', stack_info=True)
            raise ValueError('Robustness analysis is requested but where is not defined: weights or indicators? Please clarify.')
        if (config.robustness["on_single_weights"] == "yes"
                and config.robustness["on_all_weights"] == "yes"
                and config.robustness["on_indicators"] == "no"):
            logger.error('Error Message', stack_info=True)
            raise ValueError('Robustness analysis is requested on the weights: but on all or one at time? Please clarify.')
        if ((config.robustness["on_single_weights"] == "yes"
                and config.robustness["on_all_weights"] == "yes"
                and config.robustness["on_indicators"] == "yes") or
            (config.robustness["on_single_weights"] == "yes"
               and config.robustness["on_all_weights"] == "no"
               and config.robustness["on_indicators"] == "yes") or
            (config.robustness["on_single_weights"] == "no"
               and config.robustness["on_all_weights"] == "yes"
               and config.robustness["on_indicators"] == "yes")):
            logger.error('Error Message', stack_info=True)
            raise ValueError('Robustness analysis is requested: but on weights or indicators? Please clarify.')
        if ((config.robustness["on_single_weights"] == "yes"
                 and config.robustness["on_all_weights"] == "no"
                 and config.robustness["on_indicators"] == "no") or
                (config.robustness["on_single_weights"] == "no"
                 and config.robustness["on_all_weights"] == "yes"
                 and config.robustness["on_indicators"] == "no")):
            logger.info("ProMCDA will consider uncertainty on the weights")
            is_robustness_weights = 1
            logger.info("Number of Monte Carlo runs: {}".format(mc_runs))
        elif (config.robustness["on_single_weights"] == "no"
              and config.robustness["on_all_weights"] == "no"
              and config.robustness["on_indicators"] == "yes"):
            logger.info("ProMCDA will consider uncertainty on the indicators")
            is_robustness_indicators = 1
            marginal_pdf = config.monte_carlo_sampling["marginal_distribution_for_each_indicator"]
            logger.info("Number of Monte Carlo runs: {}".format(mc_runs))
            logger.info("Read input matrix with robustness at {}".format(config.input_matrix_path))
    num_exact = marginal_pdf.count('exact')
    if (is_robustness_indicators==1 and (len(input_matrix.columns) - 1) % 2 != 0): # Alternatives column is still in, therefore -1
        raise ValueError("Number of columns for non exact indicators in the input matrix must be even.")
    if input_matrix.duplicated().any():
        logger.error('Error Message', stack_info=True)
        raise ValueError('There are duplicated rows in the input matrix.')
    elif input_matrix.iloc[:,0].duplicated().any():
        logger.info('There are duplicated rows in the alternatives column.')
        while True:
            user_input = input("Do you want to continue (C) or stop (S)? ").strip().lower()
            if user_input == 'c':
                break
            elif user_input == 's':
                raise SystemExit
            else:
                print("Invalid input. Please enter 'C' to continue or 'S' to stop.")
    logger.info("Alternatives are {}".format(input_matrix.iloc[:, 0].tolist()))
    input_matrix_no_alternatives = input_matrix.drop(input_matrix.columns[0],axis=1)  # drop first column with alternatives
    if is_robustness_indicators == 0:
        num_unique = input_matrix_no_alternatives.nunique() # search for column with constant values
        cols_to_drop = num_unique[num_unique == 1].index
        col_to_drop_indexes = input_matrix_no_alternatives.columns.get_indexer(cols_to_drop)
        if any(value == 1 for value in num_unique):
            logger.info("Indicators {} have been dropped because they carry no information".format(cols_to_drop))
            input_matrix_no_alternatives = input_matrix_no_alternatives.drop(cols_to_drop, axis=1)
        # every column of the input matrix represents an indicator
        num_indicators = input_matrix_no_alternatives.shape[1]
        logger.info("Number of alternatives: {}".format(input_matrix_no_alternatives.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
    else: # matrix with robustness on indicators
        # non-exact indicators in the input matrix are associated to a column representing its mean
        # and a second column representing its std
        num_non_exact = len(marginal_pdf) - marginal_pdf.count('exact')
        num_indicators = int(input_matrix_no_alternatives.shape[1]-input_matrix_no_alternatives.shape[1]/2)
        logger.info("Number of alternatives: {}".format(input_matrix_no_alternatives.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
        # TODO: eliminate indicators with constant values (i.e. same mean and 0 std) - optional

    if (is_robustness_indicators == 0):
        if any(value == 1 for value in num_unique): polar = pop_indexed_elements(col_to_drop_indexes, polar)
    logger.info("Polarities: {}".format(polar))

    if (is_robustness_weights == 0):
        fixed_weights = config.robustness["given_weights"]
        if any(value == 1 for value in num_unique): fixed_weights = pop_indexed_elements(col_to_drop_indexes,fixed_weights)
        norm_fixed_weights = check_norm_sum_weights(fixed_weights)
        logger.info("Weights: {}".format(fixed_weights))
        logger.info("Normalized weights: {}".format(norm_fixed_weights))
    else:
        if mc_runs == 0:
            logger.error('Error Message', stack_info=True)
            raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')
        if config.robustness["on_single_weights"] == "no" and config.robustness["on_all_weights"] == "yes":
            random_weights = randomly_sample_all_weights(num_indicators, mc_runs)
            for weights in random_weights:
                weights = check_norm_sum_weights(weights)
                norm_random_weights.append(weights)
        elif config.robustness["on_single_weights"] == "yes" and config.robustness["on_all_weights"] == "no":
            i=0
            rand_weight_per_indicator = {}
            while i< num_indicators:
                random_weights = randomly_sample_ix_weight(num_indicators, i, mc_runs)
                norm_random_weight = []
                for weights in random_weights:
                    weights = check_norm_sum_weights(weights)
                    norm_random_weight.append(weights)
                rand_weight_per_indicator["indicator_{}".format(i+1)] = norm_random_weight
                i=i+1

    # checks on the number of indicators, weights, and polarities
    if num_indicators != len(polar):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of polarities does not correspond to the no. of indicators')
    if ((config.robustness["on_all_weights"] == "no") and (num_indicators != len(config.robustness["given_weights"]))):
        logger.error('Error Message', stack_info=True)
        raise ValueError('The no. of fixed weights does not correspond to the no. of indicators')
    # ----------------------------
    # NO robustness OF INDICATORS
    # ----------------------------
    if is_robustness_indicators == 0:
        logger.info("Start ProMCDA without robustness of the indicators")
        t = time.time()
        scores = pd.DataFrame()
        all_weights_means = pd.DataFrame()
        mcda_no_uncert = MCDAWithoutRobustness(config, input_matrix_no_alternatives)
        # normalize the indicators
        if is_sensitivity == "yes":
            normalized_indicators = mcda_no_uncert.normalize_indicators()
        else:
            normalized_indicators = mcda_no_uncert.normalize_indicators(f_norm)
        # estimate the scores through aggregation
        if (config.robustness["on_single_weights"] == "no" and config.robustness["on_all_weights"] == "no"): # FIXED WEIGHTS
            if is_sensitivity == "yes":
                scores = mcda_no_uncert.aggregate_indicators(normalized_indicators, norm_fixed_weights)
            else:
                scores = mcda_no_uncert.aggregate_indicators(normalized_indicators, norm_fixed_weights, f_agg)
            normalized_scores = rescale_minmax(scores) # normalized scores
            normalized_scores.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
        elif (config.robustness["on_all_weights"]  == "yes" and config.robustness["robustness_on"] == "yes"): # ALL RANDOMLY SAMPLED WEIGHTS (MCDA runs num_samples times)
            logger.info("All weights are randomly sampled from a uniform distribution")
            all_weights_normalized = []
            args_for_parallel_agg = [(lst, normalized_indicators) for lst in norm_random_weights]
            if is_sensitivity == "yes":
                all_weights = parallelize_aggregation(args_for_parallel_agg) # rough scores coming from all runs
            else:
                all_weights = parallelize_aggregation(args_for_parallel_agg, f_agg)
            for matrix in all_weights: # rescale the scores coming from all runs
                normalized_matrix = rescale_minmax(matrix) # all score normalization
                all_weights_normalized.append(normalized_matrix)
            all_weights_means, all_weights_stds = estimate_runs_mean_std(all_weights) # mean and std of rough scores
            all_weights_means_normalized, all_weights_stds_normalized = estimate_runs_mean_std(all_weights_normalized) # mean and std of norm. scores
            all_weights_stds.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            all_weights_means_normalized.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            all_weights_stds_normalized.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
        elif (config.robustness["on_single_weights"]  == "yes") and (config.robustness["robustness_on"] == "yes"): # ONE RANDOMLY SAMPLED WEIGHT A TIME (MCDA runs (num_samples * num_indicators) times)
            logger.info("One weight at time is randomly sampled from a uniform distribution")
            scores_one_random_weight_normalized = []
            for index in range(num_indicators):
                norm_one_random_weight = rand_weight_per_indicator["indicator_{}".format(index+1)] # 'norm' refers to all weights, which are normalized
                args_for_parallel_agg = [(lst, normalized_indicators) for lst in norm_one_random_weight]
                if is_robustness == "yes":
                    scores_one_random_weight = parallelize_aggregation(args_for_parallel_agg)
                else:
                    scores_one_random_weight = parallelize_aggregation(args_for_parallel_agg, f_agg)
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
            plot_weight_mean_scores = plot_mean_scores(all_weights_means, all_weights_stds, "plot_std", "weights")
            plot_weight_mean_scores_norm = plot_mean_scores(all_weights_means_normalized, all_weights_stds_normalized, "not_plot_std", "weights")
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
        logger.info("ProMCDA finished calculations: check the output files")
        elapsed = time.time() - t
        logger.info("All calculations finished in seconds {}".format(elapsed))
    # -------------------------
    # robustness OF INDICATORS
    # -------------------------
    else: # if is_robustness_indicators == 1
        all_indicators_normalized = []
        # cores = config.monte_carlo_sampling["num_cores"] # never used yet
        if (mc_runs > 0):
            if (mc_runs < 1000):
                logger.info("The number of Monte-Carlo runs is only {}".format(config.monte_carlo_runs))
                logger.info("A meaningful number of Monte-Carlo runs is equal or larger than 1000")
                while True:
                    user_input = input("Do you want to continue (C) or stop (S)? ").strip().lower()
                    if user_input == 'c':
                        break
                    elif user_input == 's':
                        raise SystemExit
                    else:
                        print("Invalid input. Please enter 'C' to continue or 'S' to stop.")
            logger.info("Start ProMCDA with robustness on the indicators")
            t = time.time()
            mcda_with_uncert = MCDAWithRobustness(config, input_matrix_no_alternatives)
            n_random_input_matrices = mcda_with_uncert.create_n_randomly_sampled_matrices() # N random matrices
            if is_sensitivity == "yes":
                n_normalized_input_matrices = parallelize_normalization(n_random_input_matrices, polar) # parallel normalization
            else:
                n_normalized_input_matrices = parallelize_normalization(n_random_input_matrices, polar, f_norm)
            args_for_parallel_agg = [(norm_fixed_weights, normalized_indicators) for normalized_indicators in n_normalized_input_matrices] # weights are fixed
            if is_sensitivity == "yes":
                all_indicators = parallelize_aggregation(args_for_parallel_agg)  # rough scores coming from all runs with random indicator values
            else:
                all_indicators = parallelize_aggregation(args_for_parallel_agg, f_agg)
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
            plot_indicators_mean_scores = plot_mean_scores(all_indicators_means, all_indicators_stds, "plot_std", "indicators")
            plot_indicators_mean_scores_norm = plot_mean_scores(all_indicators_means_normalized,
                                                                all_indicators_stds_normalized, "not_plot_std", "indicators")
            save_figure(plot_indicators_mean_scores, config.output_file_path, "MCDA_rand_indicators_rough_scores.png")
            save_figure(plot_indicators_mean_scores_norm, config.output_file_path, "MCDA_rand_indicators_norm_scores.png")
            logger.info("ProMCDA finished calculations: check the output files")
            elapsed = time.time() - t
            logger.info("All calculations finished in seconds {}".format(elapsed))
        else:
            logger.error('Error Message', stack_info=True)
            raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')

if __name__ == '__main__':
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config=input_config)
