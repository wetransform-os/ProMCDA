import logging
import sys
import pandas as pd
from typing import Tuple, List, Union, Optional

from promcda.configuration import process_indicators_and_weights
from promcda.configuration.configuration_validator import handle_robustness_weights
from promcda.enums import PDFType, NormalizationFunctions, AggregationFunctions
from promcda.utils import check_parameters_pdf, check_if_pdf_is_exact, check_if_pdf_is_poisson, rescale_minmax, \
    compute_scores_for_single_random_weight, compute_scores_for_all_random_weights

log = logging.getLogger(__name__)
formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA")


class ProMCDA:
    def __init__(self, input_matrix: pd.DataFrame, polarity: Tuple[str, ...],
                 weights: Optional[list] = None,
                 robustness_weights: Optional[bool] = False,
                 robustness_single_weights: Optional[bool] = False, robustness_indicators: Optional[bool] = False,
                 marginal_distributions: Optional[Tuple[PDFType, ...]] = None,
                 num_runs: Optional[int] = 10000, num_cores: Optional[int] = 1, random_seed: Optional[int] = 43):

        from promcda.configuration.configuration_validator import validate_configuration
        from promcda.utils.utils_for_main import check_input_matrix
        """
        Initialize the ProMCDA class with configuration parameters.

        # Required parameters
        :param input_matrix: DataFrame containing the alternatives and criteria.
        :param polarity: Tuple of polarities for each indicator ("+" or "-").

        # Optional parameters
        :param robustness_weights: Boolean flag indicating whether to perform robustness analysis on weights
                                   (True or False).
        :param robustness_single_weights: Boolean flag indicating whether to perform robustness analysis on one single
                                   weight at time (True or False).
        :param robustness_indicators: Boolean flag indicating whether to perform robustness analysis on indicators
                                      (True or False).
        :param marginal_distributions: Tuple of marginal distributions, which describe the indicators
                                       (the distribution types are defined in the enums class).
        :param num_runs: Number of Monte Carlo sampling runs (default: 10000).
        :param num_cores: Number of cores used for the calculations (default: 1).
        :param random_seed: The random seed used for the sampling (default: 43).

        Notes:
        - The input_matrix should contain the alternatives as rows and the criteria as columns.
        - If weights are not provided, they are set to 0.5 for each criterion.
        - If robustness_weights is enabled, the robustness_single_weights should be disabled, and viceversa.
        - If robustness_indicators is enabled, the robustness on weights should be disabled.

        # Example of instantiating the class and using its methods:
        from promcda import ProMCDA

        data = {'Criterion1': [3, 4, 5], 'Criterion2': [7, 2, 8], 'Criterion3': [1, 6, 4]}
        input_matrix = pd.DataFrame(data, index=['Alternative1', 'Alternative2', 'Alternative3'])

        # Define polarities for each criterion
        polarity = ("+", "-", "+")

        # Optional robustness and distributions
        robustness_weights = True
        robustness_indicators = False
        marginal_distributions = (PDFType.NORMAL, PDFType.UNIFORM, PDFType.NORMAL)

        promcda = ProMCDA(input_matrix=input_matrix,
                          polarity=polarity,
                          robustness_weights=robustness_weights,
                          robustness_indicators=robustness_indicators,
                          marginal_distributions=marginal_distributions,
                          num_runs=5000,
                          num_cores=2,
                          random_seed=123)

        # Run normalization, aggregation, and MCDA methods
        df_normalized = promcda.normalize()
        df_aggregated = promcda.aggregate()
        promcda.run_mcda()
        """

        self.input_matrix = input_matrix
        self.polarity = polarity
        self.weights = weights.copy() if weights is not None else None
        self.robustness_weights = robustness_weights
        self.robustness_single_weights = robustness_single_weights
        self.robustness_indicators = robustness_indicators
        self.num_runs = num_runs
        self.marginal_distributions = marginal_distributions
        self.num_cores = num_cores
        self.random_seed = random_seed
        self.normalized_values_without_robustness = None
        self.normalized_values_with_robustness = None
        self.aggregated_scores = None
        self.all_indicators_scores_means = None
        self.all_indicators_scores_stds = None
        self.all_indicators_means_scores_normalized = None
        self.all_indicators_scores_stds_normalized = None
        self.all_weights_score_means = None
        self.all_weights_score_stds = None,
        self.all_weights_score_means_normalized = None,
        self.all_weights_score_stds_normalized = None,
        self.iterative_random_w_score_means = None,
        self.iterative_random_w_score_stds = None,
        self.iterative_random_w_score_means_normalized = None

        self.input_matrix_no_alternatives = check_input_matrix(self.input_matrix)

        if self.weights is None and robustness_indicators is False:
            self.weights = [0.5] * self.input_matrix_no_alternatives.shape[1]
        elif self.weights is None and robustness_indicators is True:
            self.input_matrix, num_indicators, polarity, weights = process_indicators_and_weights( # an input matrix without alternatives & unuseful columns
                self.input_matrix_no_alternatives,
                self.robustness_indicators,
                self.robustness_weights,
                self.robustness_single_weights,
                self.polarity, self.num_runs,
                weights, self.marginal_distributions)
            self.weights = weights.copy()

        validate_configuration(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            weights=self.weights,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed,
            robustness_weights=self.robustness_weights,
            robustness_single_weights=self.robustness_single_weights,
            robustness_indicators=self.robustness_indicators)

    # noinspection PyArgumentList
    def normalize(self, normalization_method: Optional[NormalizationFunctions] = None) -> Union[pd.DataFrame, str]:
        """
        Normalize the input data using the specified method.

        Notes:
        The normalizations methods are defined in the NormalizationFunctions enum class.

        Parameters:
        - method (optional): The normalization method to use. If None, all available methods will be applied for a
                             Sensitivity Analysis.

        Returns:
        - A pd.DataFrame containing the normalized values of each indicator per normalization method,
          if no robustness on indicators is performed.

        :param normalization_method: NormalizationFunctions
        :return normalized_df: pd.DataFrame or string
        """
        from promcda.models import MCDAWithRobustness
        from promcda.models.mcda_without_robustness import MCDAWithoutRobustness
        from promcda.utils import utils_for_parallelization
        from promcda.configuration import process_indicators_and_weights

        # Configuration parameters' validation
        if normalization_method is not None:
            if normalization_method not in vars(NormalizationFunctions).values():
                raise ValueError(
                    f"Invalid 'normalization_method'. Expected one of {list(vars(NormalizationFunctions).values())}, "
                    f"got '{normalization_method}'.")

        # Process indicators and weights based on input parameters in the configuration
        input_matrix_no_alternatives, num_indicators, polarity, norm_weights = process_indicators_and_weights(self.input_matrix_no_alternatives,
                                                                        self.robustness_indicators,
                                                                        self.robustness_weights,
                                                                        self.robustness_single_weights,
                                                                        self.polarity, self.num_runs,
                                                                        self.weights, self.marginal_distributions)

        if not self.robustness_indicators:
            mcda_without_robustness = MCDAWithoutRobustness(polarity, input_matrix_no_alternatives)
            self.normalized_values_without_robustness = mcda_without_robustness.normalize_indicators(
                normalization_method)

            return self.normalized_values_without_robustness

        elif self.robustness_indicators and not self.robustness_weights:
            check_parameters_pdf(input_matrix_no_alternatives, self.marginal_distributions, for_testing=False)
            is_exact_pdf_mask = check_if_pdf_is_exact(self.marginal_distributions)
            is_poisson_pdf_mask = check_if_pdf_is_poisson(self.marginal_distributions)

            mcda_with_robustness = MCDAWithRobustness(input_matrix_no_alternatives, self.marginal_distributions,
                                                      self.num_runs, is_exact_pdf_mask, is_poisson_pdf_mask,
                                                      self.random_seed)
            n_random_input_matrices = mcda_with_robustness.create_n_randomly_sampled_matrices()

            if not normalization_method:
                n_normalized_input_matrices = utils_for_parallelization.parallelize_normalization(
                    n_random_input_matrices, polarity)
            else:
                n_normalized_input_matrices = utils_for_parallelization.parallelize_normalization(
                    n_random_input_matrices, polarity, normalization_method)

            self.normalized_values_with_robustness = n_normalized_input_matrices

            return f"{self.num_runs} randomly sampled matrices have been normalized."

        if self.robustness_weights and self.robustness_indicators:
            raise ValueError(
                "Inconsistent configuration: 'robustness_weights' and 'robustness_indicators' are both enabled.")

    def get_normalized_values_with_robustness(self) -> Optional[pd.DataFrame]:
        """
        Getter method to access normalized values when robustness on indicators is performed.

        Returns:
            A dictionary containing normalized values if robustness is enabled; otherwise None.
        """
        return getattr(self, 'normalized_values_with_robustness', None)

    def aggregate(self, aggregation_method: Optional[AggregationFunctions] = None) \
            -> Union[pd.DataFrame, str]:
        """
        Aggregate normalized indicators using the specified agg_method.

        Notes:
        The aggregation methods are defined in the AggregationFunctions enum class.
        This agg_method should follow the normalization. It acquires the normalized
        values from the normalization step.
        The weights are used for aggregation. If None in the intialization of ProMCDA, they are set all the same.
        Or, if robustness on weights is enabled, then the weights are sampled from the Monte Carlo simulation.

        Parameters (optional):
        - aggregation_method: The aggregation agg_method to use. If None, all available methods will be applied.

        Returns:
        - A pd.DataFrame containing the aggregated scores per normalization and aggregation methods,
          if robustness on indicators is not performed.

        :param aggregation_method: AggregationFunctions
        :return scores_df: pd.DataFrame or string
        """
        from promcda.utils import utils_for_parallelization
        from promcda.models.mcda_without_robustness import MCDAWithoutRobustness
        from promcda.configuration.configuration_validator import (check_indicator_weights_polarities,
                                                                   process_indicators_and_weights)

        # Configuration parameters' validation
        if aggregation_method is not None:
            if aggregation_method not in vars(AggregationFunctions).values():
                raise ValueError(
                    f"Invalid 'aggregation_method'. Expected one of {list(vars(AggregationFunctions).values())}, "
                    f"got '{aggregation_method}'.")

        index_column_name = self.input_matrix.index.name
        index_column_values = self.input_matrix.index.tolist()

        # Process indicators and weights based on input parameters in the configuration
        input_matrix_no_alternatives, num_indicators, polarity, norm_weights = process_indicators_and_weights(self.input_matrix_no_alternatives,
                                                        self.robustness_indicators,
                                                        self.robustness_weights, self.robustness_single_weights,
                                                        self.polarity, self.num_runs, self.weights, self.marginal_distributions)

        # Check the number of indicators, weights, and polarities, assign random weights if uncertainty is enabled
        try:
            check_indicator_weights_polarities(num_indicators, polarity, robustness_weights=self.robustness_weights,
                                               weights=norm_weights)
        except ValueError as e:
            logging.error(str(e), stack_info=True)
            raise

        # Assign values to weights when they are None
        if norm_weights is None and self.robustness_weights is False:
            if self.robustness_indicators:
                num_non_indicators = (
                        len(self.marginal_distributions) - self.marginal_distributions.count('exact')
                        - self.marginal_distributions.count('poisson'))
                num_indicators = (input_matrix_no_alternatives.shape[1] - num_non_indicators)
                weights = [0.5] * num_indicators
            else:
                weights = [0.5] * num_indicators

        # Apply aggregation in the different configuration settings
        # NO UNCERTAINTY ON INDICATORS AND WEIGHTS
        if not self.robustness_indicators and not self.robustness_weights and not self.robustness_single_weights:
            mcda_without_robustness = MCDAWithoutRobustness(self.polarity, input_matrix_no_alternatives)
            normalized_indicators = self.normalized_values_without_robustness
            if normalized_indicators is None:
                raise ValueError("Normalization must be performed before aggregation.")
            if aggregation_method is None:
                aggregated_scores = pd.DataFrame()
                for agg_method in AggregationFunctions:
                    result = mcda_without_robustness.aggregate_indicators(
                        normalized_indicators=normalized_indicators,
                        weights=norm_weights,
                        agg_method=agg_method
                    )
                    aggregated_scores = pd.concat([aggregated_scores, result], axis=1)
            else:
                aggregated_scores = mcda_without_robustness.aggregate_indicators(
                    normalized_indicators=normalized_indicators,
                    weights=norm_weights,
                    agg_method=aggregation_method
                )
            self.aggregated_scores = aggregated_scores
            return self.aggregated_scores

        # NO UNCERTAINTY ON INDICATORS, ALL RANDOMLY SAMPLED WEIGHTS (MCDA runs num_samples times)
        elif self.robustness_weights and not self.robustness_single_weights and not self.robustness_indicators:
            logger.info("Start ProMCDA with uncertainty on the weights")
            all_weights_score_means, all_weights_score_stds, \
                all_weights_score_means_normalized, all_weights_score_stds_normalized = \
                compute_scores_for_all_random_weights(self.normalized_values_without_robustness, norm_weights,
                                                      aggregation_method)
            self.all_weights_score_means = all_weights_score_means
            self.all_weights_score_stds = all_weights_score_stds
            self.all_weights_score_means_normalized = all_weights_score_means_normalized
            self.all_weights_score_stds_normalized = all_weights_score_stds_normalized
            return "Aggregation considered uncertainty on all weights, results are not explicitly shown."

        # NO UNCERTAINTY ON INDICATORS, ONE SINGLE RANDOM WEIGHT AT TIME
        elif self.robustness_single_weights and not self.robustness_weights and not self.robustness_indicators:
            logger.info("Start ProMCDA with uncertainty on one weight at time")
            iterative_random_weights_statistics: dict = compute_scores_for_single_random_weight(
                self.normalized_values_without_robustness, norm_weights, index_column_name, index_column_values,
                self.input_matrix, aggregation_method)
            iterative_random_w_score_means = iterative_random_weights_statistics['score_means']
            iterative_random_w_score_stds = iterative_random_weights_statistics['score_stds']
            iterative_random_w_score_means_normalized = (
                iterative_random_weights_statistics)['score_means_normalized']
            self.iterative_random_w_score_means = iterative_random_w_score_means
            self.iterative_random_w_score_stds = iterative_random_w_score_stds
            self.iterative_random_w_score_means_normalized = iterative_random_w_score_means_normalized
            return "Aggregation considered uncertainty on one weight at time, results are not explicitly shown."

        # UNCERTAINTY ON INDICATORS, NO UNCERTAINTY ON WEIGHTS
        elif self.robustness_indicators and not self.robustness_weights and not self.robustness_single_weights:
            all_indicators_scores_normalized = []
            logger.info("Start ProMCDA with uncertainty on the indicators")
            n_normalized_input_matrices = self.normalized_values_with_robustness
            if self.num_runs <= 0:
                logger.error('Error Message', stack_info=True)
                raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')
            if self.num_runs < 1000:
                logger.info("The number of Monte-Carlo runs is only {}".format(self.num_runs))
                logger.info("A meaningful number of Monte-Carlo runs is equal or larger than 1000")
            args_for_parallel_agg = [(norm_weights, normalized_indicators)
                                     for normalized_indicators in n_normalized_input_matrices]
            if aggregation_method is None:
                all_indicators_scores = utils_for_parallelization.parallelize_aggregation(args_for_parallel_agg)
            else:
                all_indicators_scores = utils_for_parallelization.parallelize_aggregation(args_for_parallel_agg,
                                                                                          aggregation_method)
            for matrix in all_indicators_scores:
                normalized_matrix = rescale_minmax(matrix)
                all_indicators_scores_normalized.append(normalized_matrix)

            all_indicators_scores_means, all_indicators_scores_stds = \
                utils_for_parallelization.estimate_runs_mean_std(all_indicators_scores)
            all_indicators_means_scores_normalized, all_indicators_scores_stds_normalized = \
                utils_for_parallelization.estimate_runs_mean_std(all_indicators_scores_normalized)

            self.aggregated_scores = all_indicators_scores_normalized
            self.all_indicators_scores_means = all_indicators_scores_means
            self.all_indicators_scores_stds = all_indicators_scores_stds
            self.all_indicators_means_scores_normalized = all_indicators_means_scores_normalized
            self.all_indicators_scores_stds_normalized = all_indicators_scores_stds_normalized
            return "Aggregation considered uncertainty on indicators, results are not explicitly shown."
        else:
            logger.error('Error Message', stack_info=True)
            raise ValueError(
                'Inconsistent configuration: robustness_weights and robustness_indicators are both enabled.')

    def get_aggregated_values_with_robustness_indicators(self) \
            -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Getter method to access aggregated scores when robustness on indicators is performed.

        Returns:
        A tuple containing two DataFrames:
        - The mean scores of the aggregated indicators.
        - The standard deviations of the aggregated indicators.
        If robustness is not enabled, returns None.
        """

        means = getattr(self, 'all_indicators_scores_means', None)
        normalized_means = getattr(self, 'all_indicators_means_scores_normalized', None)
        stds = getattr(self, 'all_indicators_scores_stds', None)

        if means is not None and normalized_means is not None and stds is not None:
            return means, normalized_means, stds
        return None

    def get_aggregated_values_with_robustness_weights(self) \
            -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Getter method to access aggregated scores when robustness on weights is performed.

        Returns:
        A tuple containing two DataFrames:
        - The mean scores of the aggregated indicators.
        - The standard deviations of the aggregated indicators.
        If robustness is not enabled, returns None.
        """

        means = getattr(self, 'all_weights_score_means', None)
        normalized_means = getattr(self, 'all_weights_score_means_normalized', None)
        stds = getattr(self, 'all_weights_score_stds', None)

        if means is not None and normalized_means is not None and stds is not None:
            return means, normalized_means, stds
        return None

    def get_aggregated_values_with_robustness_one_weight(self) \
            -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Getter method to access aggregated scores when robustness on one weight at time is performed.

        Returns:
        A tuple containing two DataFrames:
        - The mean scores of the aggregated indicators.
        - The standard deviations of the aggregated indicators.
        If robustness is not enabled, returns None.
        """

        means = getattr(self, 'iterative_random_w_score_means', None)
        normalized_means = getattr(self, 'iterative_random_w_score_means_normalized', None)
        stds = getattr(self, 'iterative_random_w_score_stds', None)

        if means is not None and normalized_means is not None and stds is not None:
            return means, normalized_means, stds
        return None



    def run_mcda(self, is_robustness_indicators: int, is_robustness_weights: int,
                 weights: Union[list, List[list], dict]):
        """
        Execute the full ProMCDA process, either with or without uncertainties on the indicators.
        """

        # Normalize
        # self.normalize()

        # Aggregate
        # self.aggregate()

        # Run
        # no uncertainty
        # TODO
        # uncertainty
        # TODO


    def plot_results(self, data: Union[pd.DataFrame, dict], plot_type: str, **kwargs):
        """
        Plot the results based on the specified type of plot.

        Parameters:
        - data (pd.DataFrame or dictionary): The data to be plotted.
        - plot_type (str): The type of plot to generate. Must be one of:
            - "normalized_scores"
            - "non_normalized_scores"
            - "average_scores"
            - "iterative_average_scores"
        - **kwargs: Additional keyword arguments for customization, such as labels, titles, or styles.

        Returns:
        None
        """
        from promcda.utils.utils_for_plotting import plot_norm_scores_without_uncert, plot_non_norm_scores_with_altair

        #TODO: what if plot_results is used within a Python environment from the Terminal?

        if plot_type == "normalized_scores":
           plot_norm_scores_without_uncert(self.aggregated_scores)
        elif plot_type == "non_normalized_scores":
           #return(plot_non_norm_scores_without_uncert(self.aggregated_scores))
           return plot_non_norm_scores_with_altair(self.aggregated_scores)
        #     # Implement the logic for non-normalized scores without uncertainty
        #     plt.plot(data, linestyle='--', **kwargs)
        #     plt.title("Non-Normalized Scores Without Uncertainty")
        # elif plot_type == "mean_scores":
        #     # Implement the logic for mean scores
        #     data.mean().plot(kind='bar', **kwargs)
        #     plt.title("Mean Scores")
        # elif plot_type == "mean_scores_iterative":
        #     # Implement the logic for iterative mean scores
        #     data.cumsum().mean().plot(kind='line', **kwargs)
        #     plt.title("Mean Scores (Iterative)")
        # else:
        #     raise ValueError(f"Invalid plot_type: {plot_type}. Must be one of: "
        #                      f"'norm_scores_without_uncert', 'non_norm_scores_without_uncert', 'mean_scores', 'mean_scores_iterative'")
        #
        # plt.xlabel(kwargs.get("xlabel", "X-axis"))
        # plt.ylabel(kwargs.get("ylabel", "Y-axis"))
        # plt.grid(kwargs.get("grid", True))
        # plt.show()

    # def get_results(self):
    #     """
    #     Return the final results as a DataFrame or other relevant structure.
    #     """
    #     # Return the aggregated results (or any other relevant results)
    #     return self.aggregated_matrix
