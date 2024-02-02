#! /usr/bin/env python3

"""
This script serves as the main entry point for running all pieces of functionality in a consequential way
following the settings given in the configuration file configuration.json.

Usage (from root directory):
    $ python3 -m mcda.mcda_run -c configuration.json
"""

import time
import logging

from mcda.configuration.config import Config
from mcda.mcda_with_robustness import MCDAWithRobustness
from mcda.mcda_without_robustness import MCDAWithoutRobustness
from mcda.utils.utils_for_main import *
from mcda.utils.utils_for_plotting import *
from mcda.utils.utils_for_parallelization import *

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")


class UserStoppedInfo(Exception):
    """A class representing information about user stopping conditions"""
    pass


# noinspection PyTypeChecker
def main(input_config: dict, user_input_callback=input):
    """
        Execute the ProMCDA (Probabilistic Multi-Criteria Decision Analysis) process.

        Parameters:
        - input_config (dictionary): Configuration parameters for the ProMCDA process.
        - user_input_callback (function, optional): Callback function for user input.

        Raises:
        - ValueError: If there are issues with the input matrix, weights, or indicators.
        - UserStoppedInfo: If the user chooses to stop the process during execution.

        This function performs the ProMCDA process based on the provided configuration.
        It handles various aspects such as sensitivity analysis, robustness analysis,
        and uncertainty in indicators. The results are saved in output files, and plots
        are generated to visualize the scores and rankings.

        Note: Ensure that the input matrix, weights, and indicators are correctly specified
        in the input configuration.
        """

    is_sensitivity = None
    is_robustness = None
    is_robustness_indicators = 0
    is_robustness_weights = 0
    marginal_pdf = []
    num_unique = []
    iterative_random_w_means = {}
    iterative_random_w_stds = {}
    iterative_random_w_means_normalized = {}
    iterative_random_w_stds_normalized = {}

    # Extracting relevant configuration values
    config = Config(input_config)
    input_matrix = read_matrix(config.input_matrix_path)
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

        check_config_setting(((config.robustness["on_single_weights"] == "yes" and
                             config.robustness["on_all_weights"] == "no" and
                             config.robustness["on_indicators"] == "no") or
                            (config.robustness["on_single_weights"] == "no" and
                             config.robustness["on_all_weights"] == "yes" and
                             config.robustness["on_indicators"] == "no")),
                           'ProMCDA will consider uncertainty on the weights')
        is_robustness_weights = 1
        logger.info("Number of Monte Carlo runs: {}".format(mc_runs))

        check_config_setting((config.robustness["on_single_weights"] == "no" and
                            config.robustness["on_all_weights"] == "no" and
                            config.robustness["on_indicators"] == "yes"),
                           'ProMCDA will consider uncertainty on the indicators')
        is_robustness_indicators = 1
        marginal_pdf = config.monte_carlo_sampling["marginal_distribution_for_each_indicator"]
        logger.info("Number of Monte Carlo runs: {}".format(mc_runs))
        logger.info("Read input matrix with uncertainty of the indicators at {}".format(
            config.input_matrix_path))

    # Check the input matrix for duplicated rows in the alternatives and rescale negative indicator values
    input_matrix_no_alternatives = check_input_matrix(input_matrix)
    if is_robustness_indicators == 0:
        num_indicators = input_matrix_no_alternatives.shape[1]
    else:
        num_non_exact_and_non_poisson = len(marginal_pdf) - marginal_pdf.count('exact') - marginal_pdf.count('poisson')
        num_indicators = (input_matrix_no_alternatives.shape[1] - num_non_exact_and_non_poisson)

    # Process indicators and weights based on input parameters in the configuration
    process_indicators_and_weights(config, input_matrix_no_alternatives, is_robustness_indicators, is_robustness_weights,
                                   polar, mc_runs, num_indicators)

    # Check the number of indicators, weights, and polarities
    try:
        check_indicator_weights_polarities(num_indicators, polar, config)
    except ValueError as e:
        logging.error(str(e), stack_info=True)
        raise

    # ----------------------------
    # NO UNCERTAINTY OF INDICATORS   QUI, refactor!
    # ----------------------------
    if is_robustness_indicators == 0:
        logger.info("Start ProMCDA without robustness of the indicators")
        t = time.time()
        scores = pd.DataFrame()
        all_weights_means = pd.DataFrame()
        mcda_no_uncert = MCDAWithoutRobustness(
            config, input_matrix_no_alternatives)
        # normalize the indicators
        if is_sensitivity == "yes":
            normalized_indicators = mcda_no_uncert.normalize_indicators()
        else:
            normalized_indicators = mcda_no_uncert.normalize_indicators(f_norm)
        # estimate the scores through aggregation
        if config.robustness["robustness_on"] == "no":  # FIXED INDICATORS & WEIGHTS
            if is_sensitivity == "yes":
                scores = mcda_no_uncert.aggregate_indicators(
                    normalized_indicators, norm_fixed_weights)
            else:
                scores = mcda_no_uncert.aggregate_indicators(
                    normalized_indicators, norm_fixed_weights, f_agg)
            normalized_scores = rescale_minmax(scores)  # normalized scores
            normalized_scores.insert(
                0, 'Alternatives', input_matrix.iloc[:, 0])
        # ALL RANDOMLY SAMPLED WEIGHTS (MCDA runs num_samples times)
        elif config.robustness["on_all_weights"] == "yes" and config.robustness["robustness_on"] == "yes":
            logger.info(
                "All weights are randomly sampled from a uniform distribution")
            all_weights_normalized = []
            args_for_parallel_agg = [(lst, normalized_indicators)
                                     for lst in norm_random_weights]
            if is_sensitivity == "yes":
                all_weights = parallelize_aggregation(
                    args_for_parallel_agg)  # rough scores coming from all runs
            else:
                all_weights = parallelize_aggregation(
                    args_for_parallel_agg, f_agg)
            for matrix in all_weights:  # rescale the scores coming from all runs
                normalized_matrix = rescale_minmax(
                    matrix)  # all score normalization
                all_weights_normalized.append(normalized_matrix)
            all_weights_means, all_weights_stds = estimate_runs_mean_std(
                all_weights)  # mean and std of rough scores
            all_weights_means_normalized, all_weights_stds_normalized = estimate_runs_mean_std(
                all_weights_normalized)  # mean and std of norm. scores
            all_weights_stds.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            all_weights_means_normalized.insert(
                0, 'Alternatives', input_matrix.iloc[:, 0])
            all_weights_stds_normalized.insert(
                0, 'Alternatives', input_matrix.iloc[:, 0])
        # ONE RANDOMLY SAMPLED WEIGHT A TIME (MCDA runs (num_samples * num_indicators) times)
        elif (config.robustness["on_single_weights"] == "yes") and (config.robustness["robustness_on"] == "yes"):
            logger.info(
                "One weight at time is randomly sampled from a uniform distribution")
            scores_one_random_weight_normalized = []
            for index in range(num_indicators):
                norm_one_random_weight = rand_weight_per_indicator["indicator_{}".format(
                    index + 1)]  # 'norm' refers to all weights, which are normalized
                args_for_parallel_agg = [(lst, normalized_indicators)
                                         for lst in norm_one_random_weight]
                if is_sensitivity == "yes":
                    scores_one_random_weight = parallelize_aggregation(
                        args_for_parallel_agg)
                else:
                    scores_one_random_weight = parallelize_aggregation(
                        args_for_parallel_agg, f_agg)
                for matrix in scores_one_random_weight:
                    matrix_normalized = rescale_minmax(
                        matrix)  # normalize scores
                    scores_one_random_weight_normalized.append(
                        matrix_normalized)
                one_random_weight_means, one_random_weight_stds = estimate_runs_mean_std(
                    scores_one_random_weight)
                one_random_weight_means_normalized, one_random_weight_stds_normalized = estimate_runs_mean_std(
                    scores_one_random_weight_normalized)  # normalized mean and std
                one_random_weight_means.insert(
                    0, 'Alternatives', input_matrix.iloc[:, 0])
                one_random_weight_stds.insert(
                    0, 'Alternatives', input_matrix.iloc[:, 0])
                one_random_weight_means_normalized.insert(
                    0, 'Alternatives', input_matrix.iloc[:, 0])
                one_random_weight_stds_normalized.insert(
                    0, 'Alternatives', input_matrix.iloc[:, 0])
                iterative_random_w_means["indicator_{}".format(
                    index + 1)] = one_random_weight_means  # create output dictionaries
                iterative_random_w_stds["indicator_{}".format(
                    index + 1)] = one_random_weight_stds
                iterative_random_w_means_normalized["indicator_{}".format(
                    index + 1)] = one_random_weight_means_normalized
                iterative_random_w_stds_normalized["indicator_{}".format(
                    index + 1)] = one_random_weight_stds_normalized
        # estimate the ranks
        if not scores.empty:
            ranks = scores.rank(pct=True)
            ranks.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
        elif not all_weights_means.empty:
            ranks = all_weights_means.rank(pct=True)
            ranks.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
        elif not bool(iterative_random_w_means) == False:
            pass
        # save output files
        logger.info("Saving results in {}".format(config.output_file_path))
        check_path_exists(config.output_file_path)
        if not scores.empty:
            scores.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            save_df(scores, config.output_file_path, 'scores.csv')
            save_df(normalized_scores, config.output_file_path,
                    'normalized_scores.csv')
            save_df(ranks, config.output_file_path, 'ranks.csv')
        elif not all_weights_means.empty:
            all_weights_means.insert(
                0, 'Alternatives', input_matrix.iloc[:, 0])
            save_df(all_weights_means, config.output_file_path, 'score_means.csv')
            save_df(all_weights_stds, config.output_file_path, 'score_stds.csv')
            save_df(all_weights_means_normalized,
                    config.output_file_path, 'score_means_normalized.csv')
            # save_df(all_weights_stds_normalized, config.output_file_path, 'score_stds_normalized.csv')
            # the std on rescaled values is not statistically informative
            save_df(ranks, config.output_file_path, 'ranks.csv')
        elif not bool(iterative_random_w_means) == False:
            save_dict(iterative_random_w_means,
                      config.output_file_path, 'score_means.pkl')
            save_dict(iterative_random_w_stds,
                      config.output_file_path, 'score_stds.pkl')
            save_dict(iterative_random_w_means_normalized,
                      config.output_file_path, 'score_means_normalized.pkl')
            # save_dict(iterative_random_w_stds_normalized, config.output_file_path, 'score_stds_normalized.pkl')
            # the std on rescaled values is not statistically informative
        save_config(input_config, config.output_file_path,
                    'configuration.json')
        # plots
        if not scores.empty:
            plot_norm_scores = plot_norm_scores_without_uncert(
                normalized_scores)
            save_figure(plot_norm_scores, config.output_file_path,
                        "MCDA_norm_scores.png")
            plot_no_norm_scores = plot_non_norm_scores_without_uncert(scores)
            save_figure(plot_no_norm_scores,
                        config.output_file_path, "MCDA_rough_scores.png")
        elif not all_weights_means.empty:
            plot_weight_mean_scores = plot_mean_scores(
                all_weights_means, all_weights_stds, "plot_std", "weights")
            plot_weight_mean_scores_norm = plot_mean_scores(
                all_weights_means_normalized, all_weights_stds_normalized, "not_plot_std", "weights")
            save_figure(plot_weight_mean_scores,
                        config.output_file_path, "MCDA_rough_scores.png")
            save_figure(plot_weight_mean_scores_norm,
                        config.output_file_path, "MCDA_norm_scores.png")
        elif not bool(iterative_random_w_means) == False:
            images = []
            images_norm = []
            for index in range(num_indicators):
                one_random_weight_means = iterative_random_w_means["indicator_{}".format(
                    index + 1)]
                one_random_weight_stds = iterative_random_w_stds["indicator_{}".format(
                    index + 1)]
                one_random_weight_means_normalized = iterative_random_w_means_normalized["indicator_{}".format(
                    index + 1)]
                one_random_weight_stds_normalized = iterative_random_w_stds_normalized["indicator_{}".format(
                    index + 1)]
                plot_weight_mean_scores = plot_mean_scores_iterative(one_random_weight_means,
                                                                     one_random_weight_stds,
                                                                     input_matrix_no_alternatives.columns, index,
                                                                     "plot_std")
                plot_weight_mean_scores_norm = plot_mean_scores_iterative(one_random_weight_means_normalized,
                                                                          one_random_weight_stds_normalized,
                                                                          input_matrix_no_alternatives.columns, index,
                                                                          "not_plot_std")
                images.append(plot_weight_mean_scores)
                images_norm.append(plot_weight_mean_scores_norm)
            combine_images(images, config.output_file_path,
                           "MCDA_one_weight_randomness_rough_scores.png")
            combine_images(images_norm, config.output_file_path,
                           "MCDA_one_weight_randomness_norm_scores.png")
        logger.info("ProMCDA finished calculations: check the output files")
        elapsed = time.time() - t
        logger.info("All calculations finished in seconds {}".format(elapsed))
    # -------------------------
    # UNCERTAINTY OF INDICATORS
    # -------------------------
    else:  # if is_robustness_indicators == 1
        all_indicators_normalized = []
        # cores = config.monte_carlo_sampling["num_cores"] # never used yet
        if mc_runs > 0:
            if mc_runs < 1000:
                logger.info(
                    "The number of Monte-Carlo runs is only {}".format(mc_runs))
                logger.info(
                    "A meaningful number of Monte-Carlo runs is equal or larger than 1000")
                while True:
                    user_input = user_input_callback(
                        "Do you want to continue (C) or stop (S)? ").strip().lower()
                    if user_input == 'c':
                        break
                    elif user_input == 's':
                        raise UserStoppedInfo()
                    else:
                        print(
                            "Invalid input. Please enter 'C' to continue or 'S' to stop.")
            logger.info("Start ProMCDA with uncertainty on the indicators")
            are_parameters_correct = check_parameters_pdf(
                input_matrix_no_alternatives, config)
            if any(not value for value in are_parameters_correct):
                logger.info(
                    'There is a problem with the parameters given in the input matrix with uncertainties. Check your data!')
                logger.info(
                    'Either standard deviation values of normal/lognormal distributed indicators are larger than their means,')
                logger.info(
                    'or max. values of uniform distributed indicators are smaller than their min. values.')
                logger.info(
                    'If you continue, the negative values will be rescaled internally to a positive range.')
                while True:
                    user_input = user_input_callback(
                        "Do you want to continue (C) or stop (S)? ").strip().lower()
                    if user_input == 'c':
                        break
                    elif user_input == 's':
                        raise UserStoppedInfo()
                    else:
                        print(
                            "Invalid input. Please enter 'C' to continue or 'S' to stop.")
            t = time.time()
            mcda_with_uncert = MCDAWithRobustness(
                config, input_matrix_no_alternatives)
            # N random matrices
            n_random_input_matrices = mcda_with_uncert.create_n_randomly_sampled_matrices()
            if is_sensitivity == "yes":
                n_normalized_input_matrices = parallelize_normalization(
                    n_random_input_matrices, polar)  # parallel normalization
            else:
                n_normalized_input_matrices = parallelize_normalization(
                    n_random_input_matrices, polar, f_norm)
            args_for_parallel_agg = [(norm_fixed_weights, normalized_indicators)
                                     for normalized_indicators in n_normalized_input_matrices]  # weights are fixed
            if is_sensitivity == "yes":
                # rough scores coming from all runs with random indicator values
                all_indicators = parallelize_aggregation(args_for_parallel_agg)
            else:
                all_indicators = parallelize_aggregation(
                    args_for_parallel_agg, f_agg)
            for matrix in all_indicators:  # rescale the scores coming from all runs
                normalized_matrix = rescale_minmax(
                    matrix)  # all score normalization
                all_indicators_normalized.append(normalized_matrix)
            all_indicators_means, all_indicators_stds = estimate_runs_mean_std(
                all_indicators)  # mean and std of rough scores
            all_indicators_means_normalized, all_indicators_stds_normalized = estimate_runs_mean_std(
                all_indicators_normalized)  # mean and std of norm. scores
            # estimate the ranks
            ranks = all_indicators_means.rank(pct=True)
            # re-insert the Alternatives column
            all_indicators_means.insert(
                0, 'Alternatives', input_matrix.iloc[:, 0])
            all_indicators_stds.insert(
                0, 'Alternatives', input_matrix.iloc[:, 0])
            all_indicators_means_normalized.insert(
                0, 'Alternatives', input_matrix.iloc[:, 0])
            all_indicators_stds_normalized.insert(
                0, 'Alternatives', input_matrix.iloc[:, 0])
            ranks.insert(0, 'Alternatives', input_matrix.iloc[:, 0])
            # save output files
            logger.info("Saving results in {}".format(config.output_file_path))
            check_path_exists(config.output_file_path)
            save_df(all_indicators_means,
                    config.output_file_path, 'score_means.csv')
            save_df(all_indicators_stds,
                    config.output_file_path, 'score_stds.csv')
            save_df(all_indicators_means_normalized,
                    config.output_file_path, 'score_means_normalized.csv')
            save_df(ranks, config.output_file_path, 'ranks.csv')
            save_config(input_config, config.output_file_path,
                        'configuration.json')
            # plots
            plot_indicators_mean_scores = plot_mean_scores(
                all_indicators_means, all_indicators_stds, "plot_std", "indicators")
            plot_indicators_mean_scores_norm = plot_mean_scores(all_indicators_means_normalized,
                                                                all_indicators_stds_normalized, "not_plot_std",
                                                                "indicators")
            save_figure(plot_indicators_mean_scores,
                        config.output_file_path, "MCDA_rough_scores.png")
            save_figure(plot_indicators_mean_scores_norm,
                        config.output_file_path, "MCDA_norm_scores.png")
            logger.info(
                "ProMCDA finished calculations: check the output files")
            elapsed = time.time() - t
            logger.info(
                "All calculations finished in seconds {}".format(elapsed))
        else:
            logger.error('Error Message', stack_info=True)
            raise ValueError(
                'The number of MC runs should be larger than 0 for a robustness analysis')


if __name__ == '__main__':
    config_path = parse_args()
    input_config = get_config(config_path)
    main(input_config=input_config)
