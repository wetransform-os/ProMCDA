import logging
import sys
import time
import pandas as pd
from typing import Tuple, List, Union, Optional

from build.lib.mcda.mcda_with_robustness import MCDAWithRobustness
from mcda.configuration.configuration_validator import extract_configuration_values, check_configuration_values, \
    check_configuration_keys, check_indicator_weights_polarities, process_indicators_and_weights
from mcda.configuration.enums import PDFType, NormalizationFunctions, AggregationFunctions
from mcda.models import mcda_with_robustness
from mcda.models.mcda_without_robustness import MCDAWithoutRobustness
from mcda.utils import utils_for_parallelization
from mcda.utils.utils_for_main import run_mcda_without_indicator_uncertainty, run_mcda_with_indicator_uncertainty, \
    check_input_matrix, check_if_pdf_is_exact, check_if_pdf_is_poisson, check_parameters_pdf, rescale_minmax

log = logging.getLogger(__name__)
formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA")

class ProMCDA:
    def __init__(self, input_matrix: pd.DataFrame, polarity: Tuple[str, ...], robustness_weights: Optional[bool] = None,
                 robustness_single_weights: Optional[bool] = None, robustness_indicators: Optional[bool] = None,
                 marginal_distributions: Optional[Tuple[PDFType, ...]] = None,
                 num_runs: Optional[int] = 10000, num_cores: Optional[int] = 1, random_seed: Optional[int] = 43):
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

        self.input_matrix_no_alternatives = check_input_matrix(self.input_matrix)

    # def validate_inputs(self) -> Tuple[int, int, list, Union[list, List[list], dict], dict]:
    #     """
    #     Extract and validate input configuration parameters to ensure they are correct.
    #     Return a flag indicating whether robustness analysis will be performed on indicators (1) or not (0).
    #     """
    #
    #     configuration_values = extract_configuration_values(self.input_matrix, self.polarity,
    #                                                         self.robustness, self.monte_carlo)
    #     is_robustness_indicators, is_robustness_weights, polar, weights = check_configuration_values(
    #         configuration_values)
    #
    #     return is_robustness_indicators, is_robustness_weights, polar, weights, configuration_values

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

        if not self.robustness_indicators:
            mcda_without_robustness = MCDAWithoutRobustness(self.polarity, self.input_matrix_no_alternatives)
            self.normalized_values_without_robustness = mcda_without_robustness.normalize_indicators(normalization_method)

            return self.normalized_values_without_robustness

        elif self.robustness_indicators and not self.robustness_weights:
            check_parameters_pdf(self.input_matrix_no_alternatives, self.marginal_distributions, for_testing=False)
            is_exact_pdf_mask = check_if_pdf_is_exact(self.marginal_distributions)
            is_poisson_pdf_mask = check_if_pdf_is_poisson(self.marginal_distributions)

            mcda_with_robustness = MCDAWithRobustness(self.input_matrix_no_alternatives, self.marginal_distributions,
                                                      self.num_runs, is_exact_pdf_mask, is_poisson_pdf_mask,
                                                      self.random_seed)
            n_random_input_matrices = mcda_with_robustness.create_n_randomly_sampled_matrices()

            if not normalization_method:
                n_normalized_input_matrices = utils_for_parallelization.parallelize_normalization(
                    n_random_input_matrices, self.polarity)
            else:
                n_normalized_input_matrices = utils_for_parallelization.parallelize_normalization(
                    n_random_input_matrices, self.polarity, normalization_method)

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

    def aggregate(self, aggregation_method: Optional[AggregationFunctions] = None, weights: Optional[List[str]] = None) \
            -> Union[pd.DataFrame, str]:
        """
        Aggregate normalized indicators using the specified agg_method.

        Notes:
        The aggregation methods are defined in the AggregationFunctions enum class.
        This agg_method should follow the normalization. It acquires the normalized
        values from the normalization step.

        Parameters (optional):
        - aggregation_method: The aggregation agg_method to use. If None, all available methods will be applied.
        - weights: The weights to be used for aggregation. If None, they are set all the same. Or, if robustness on
          weights is enabled, then the weights are sampled from the Monte Carlo simulation.

        Returns:
        - A pd.DataFrame containing the aggregated scores per normalization and aggregation methods,
          if robustness on indicators is not performed.

        :param aggregation_method: AggregationFunctions
        :param weights : list or None
        :return scores_df: pd.DataFrame or string
        """
        num_indicators = self.input_matrix_no_alternatives.shape[1]
        if weights is None:
            if self.robustness_indicators:
                num_non_indicators = (
                        len(self.marginal_distributions) - self.marginal_distributions.count('exact')
                        - self.marginal_distributions.count('poisson'))
                num_indicators = (self.input_matrix_no_alternatives.shape[1] - num_non_indicators)
                weights =  [0.5] * num_indicators
            else:
                weights = [0.5] * num_indicators

        # Process indicators and weights based on input parameters in the configuration
        polar, weights = process_indicators_and_weights(self.input_matrix_no_alternatives,
                                                        self.robustness_indicators,
                                                        self.robustness_weights, self.robustness_single_weights,
                                                        self.polarity, self.num_runs, num_indicators, weights)

        # Check the number of indicators, weights, and polarities, assign random weights if uncertainty is enabled
        try:
            check_indicator_weights_polarities(num_indicators, polar, robustness_weights=self.robustness_weights,
                                               weights=weights)
        except ValueError as e:
            logging.error(str(e), stack_info=True)
            raise

        if not self.robustness_indicators:
            mcda_without_robustness = MCDAWithoutRobustness(self.polarity, self.input_matrix_no_alternatives)
            normalized_indicators = self.normalized_values_without_robustness
            if normalized_indicators is None:
                raise ValueError("Normalization must be performed before aggregation.")

            if aggregation_method is None:
                aggregated_scores = pd.DataFrame()
                for agg_method in AggregationFunctions:
                    result = mcda_without_robustness.aggregate_indicators(
                        normalized_indicators=normalized_indicators,
                        weights=weights,
                        agg_method=agg_method
                    )
                    aggregated_scores = pd.concat([aggregated_scores, result], axis=1)
            else:
                aggregated_scores = mcda_without_robustness.aggregate_indicators(
                normalized_indicators=normalized_indicators,
                weights=weights,
                agg_method=aggregation_method
            )

            self.aggregated_scores = aggregated_scores
            return self.aggregated_scores

        elif self.robustness_indicators and not self.robustness_weights:
            all_indicators_scores_normalized = []
            logger.info("Start ProMCDA with uncertainty on the indicators")
            n_normalized_input_matrices = self.normalized_values_with_robustness

            if self.num_runs <= 0:
                logger.error('Error Message', stack_info=True)
                raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')

            if self.num_runs < 1000:
                logger.info("The number of Monte-Carlo runs is only {}".format(self.num_runs))
                logger.info("A meaningful number of Monte-Carlo runs is equal or larger than 1000")

            args_for_parallel_agg = [(weights, normalized_indicators)
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

        else:
            # write message error

            return ("Aggregation considered uncertainty on indicators, resulsts are not explicitly shown.")


    def run_mcda(self, is_robustness_indicators: int, is_robustness_weights: int,
                 weights: Union[list, List[list], dict]):
        """
        Execute the full ProMCDA process, either with or without uncertainties on the indicators.
        """
        start_time = time.time()

        # Normalize
        # self.normalize()

        # Aggregate
        # self.aggregate()

        # Run
        # no uncertainty
        if is_robustness_indicators == 0:
            run_mcda_without_indicator_uncertainty(self.configuration_settings, is_robustness_weights, weights)
        # uncertainty
        else:
            run_mcda_with_indicator_uncertainty(self.configuration_settings)

        elapsed_time = time.time() - start_time

    # def get_results(self):
    #     """
    #     Return the final results as a DataFrame or other relevant structure.
    #     """
    #     # Return the aggregated results (or any other relevant results)
    #     return self.aggregated_matrix
