import logging
import sys
import pandas as pd
from typing import Tuple, List, Union, Optional

from promcda.configuration import process_indicators_and_weights
from promcda.enums import PDFType, NormalizationFunctions, AggregationFunctions, RobustnessAnalysisType
from promcda.utils import check_parameters_pdf, check_if_pdf_is_exact, check_if_pdf_is_poisson, rescale_minmax, \
    compute_scores_for_single_random_weight, compute_scores_for_all_random_weights, check_if_pdf_is_uniform

log = logging.getLogger(__name__)
formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA")


class ProMCDA:
    def __init__(self, input_matrix: pd.DataFrame, polarity: Tuple[str, ...],
                 weights: Optional[list] = None,
                 robustness: RobustnessAnalysisType = RobustnessAnalysisType.NONE,
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
        : weights: List of weights for each criterion (default: None, which sets all weights to 0.5).
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
        - The input_matrix should contain the alternatives as rows and the criteria as columns, 
          including row and column names.
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
        self.robustness = robustness
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

        if not isinstance(robustness, RobustnessAnalysisType):
            raise TypeError(f"'robustness' must be of type RobustnessAnalysisType, got {type(robustness).__name__}")

        if self.weights is None and RobustnessAnalysisType.INDICATORS.value != "indicators":
            self.weights = [0.5] * self.input_matrix_no_alternatives.shape[1]
        elif self.weights is None and self.robustness == RobustnessAnalysisType.INDICATORS:
            self.input_matrix, num_indicators, polarity, weights = process_indicators_and_weights( # an input matrix without alternatives & unuseful columns
                self.input_matrix_no_alternatives,
                self.robustness,
                self.polarity,
                self.num_runs,
                weights,
                self.marginal_distributions)
            self.weights = weights.copy()

        validate_configuration(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            weights=self.weights,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed,
            robustness=self.robustness)

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
        if normalization_method is not None and normalization_method not in [member for member in
                                                                             NormalizationFunctions]:
            raise ValueError(
                f"Invalid 'normalization_method'. Expected one of {[member.value for member in NormalizationFunctions]}, "
                f"got '{normalization_method}'.")

        # Process indicators and weights based on input parameters in the configuration
        input_matrix_no_alternatives, num_indicators, polarity, norm_weights = process_indicators_and_weights(self.input_matrix_no_alternatives,
                                                                        self.robustness,
                                                                        self.polarity, self.num_runs,
                                                                        self.weights, self.marginal_distributions)

        if not self.robustness == RobustnessAnalysisType.INDICATORS:
            mcda_without_robustness = MCDAWithoutRobustness(polarity, input_matrix_no_alternatives)
            self.normalized_values_without_robustness = mcda_without_robustness.normalize_indicators(
                normalization_method)

            return self.normalized_values_without_robustness

        elif self.robustness == RobustnessAnalysisType.INDICATORS and not self.robustness == RobustnessAnalysisType.ALL_WEIGHTS:
            is_exact_pdf_mask = check_if_pdf_is_exact(self.marginal_distributions)
            is_poisson_pdf_mask = check_if_pdf_is_poisson(self.marginal_distributions)
            is_uniform_pdf_mask = check_if_pdf_is_uniform(self.marginal_distributions)
            check_parameters_pdf(input_matrix_no_alternatives,
                                 is_uniform_pdf_mask, is_exact_pdf_mask, is_poisson_pdf_mask,
                                 for_testing=False)

            mcda_with_robustness = MCDAWithRobustness(input_matrix_no_alternatives, self.marginal_distributions,
                                                      self.num_runs, is_exact_pdf_mask, is_poisson_pdf_mask,
                                                      self.random_seed)
            n_random_input_matrices = mcda_with_robustness.create_n_randomly_sampled_matrices()

            if not normalization_method:
                n_normalized_input_matrices = utils_for_parallelization.parallelize_normalization(
                    n_random_input_matrices, polarity)
            else:
                normalization_method = normalization_method.value
                n_normalized_input_matrices = utils_for_parallelization.parallelize_normalization(
                    n_random_input_matrices, polarity, normalization_method)

            self.normalized_values_with_robustness = n_normalized_input_matrices

            return f"{self.num_runs} randomly sampled matrices have been normalized."

        if self.robustness == RobustnessAnalysisType.ALL_WEIGHTS and self.robustness == RobustnessAnalysisType.INDICATORS:
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
                                                        self.robustness,
                                                        self.polarity, self.num_runs, self.weights,
                                                        self.marginal_distributions)

        # Check the number of indicators, weights, and polarities, assign random weights if uncertainty is enabled
        check_indicator_weights_polarities(num_indicators, polarity, self.robustness,
                                               weights=norm_weights)

        # Apply aggregation in the different configuration settings
        # NO UNCERTAINTY ON INDICATORS AND WEIGHTS
        if (not self.robustness == RobustnessAnalysisType.INDICATORS
                and not self.robustness == RobustnessAnalysisType.ALL_WEIGHTS
                and not self.robustness == RobustnessAnalysisType.SINGLE_WEIGHTS):
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
        elif (self.robustness == RobustnessAnalysisType.ALL_WEIGHTS
              and self.robustness != RobustnessAnalysisType.SINGLE_WEIGHTS
              and self.robustness != RobustnessAnalysisType.INDICATORS):
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
        elif (self.robustness == RobustnessAnalysisType.SINGLE_WEIGHTS
              and self.robustness != RobustnessAnalysisType.ALL_WEIGHTS
              and self.robustness != RobustnessAnalysisType.INDICATORS):
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
        elif (self.robustness == RobustnessAnalysisType.INDICATORS
              and self.robustness != RobustnessAnalysisType.ALL_WEIGHTS
              and self.robustness != RobustnessAnalysisType.SINGLE_WEIGHTS):
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
        - A tuple containing two DataFrames:
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
        - A tuple containing two DataFrames:
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
        - A tuple containing two DataFrames:
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

    @staticmethod
    def evaluate_ranks(scores) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute percentile ranks from scores.

        Parameters
        - scores : array-like, pandas Series, or DataFrame
            A 1D array or Series, or a 2D DataFrame with multiple columns of scores.

        Returns
        - ranks : pandas.Series or pandas.DataFrame
            Percentile ranks. If input is a Series or 1D array, returns a Series.
            If input is a DataFrame, returns a DataFrame with ranks computed per column.

        Examples
        evaluate_ranks([0.8, 0.6, 0.8])
        0    0.833333
        1    0.333333
        2    0.833333

        evaluate_ranks(pd.DataFrame({"A": [0.8, 0.6], "B": [0.4, 0.9]}))
                  A         B
        0  1.000000  0.000000
        1  0.000000  1.000000
        """

        if isinstance(scores, pd.DataFrame):
            return scores.rank(pct=True)

        if not isinstance(scores, pd.Series):
            scores = pd.Series(scores)

        return scores.rank(pct=True)


    def run(self, normalization_method: Optional[NormalizationFunctions] = None,
                  aggregation_method: Optional[AggregationFunctions] = None):
        """
        Execute the full ProMCDA process, either with or without uncertainties on the indicators.

        Parameters
        - normalization method (optional): The normalization method to use. If None, all available methods will be
            applied for a Sensitivity Analysis.
        - aggregation method (optional): The aggregation method to use. If None, all available methods will be
            applied for a Sensitivity Analysis.

        Returns
        - A dictionary containing the:
          - scores: a pd.DataFrame containing the aggregated scores per normalization and aggregation methods,
                if robustness on indicators is not performed.
          - means scores and standard deviations: two pd.DataFrames containing:
                - the mean scores of the aggregated indicators;
                - the standard deviations of the aggregated indicators,
                if robustness on indicators or weights is performed.
          - ranks : pandas.Series or pandas.DataFrame
                Percentile ranks. If input is a Series or 1D array, returns a Series.
                If input is a DataFrame, returns a DataFrame with ranks computed per column.

          Notes
          - In case of randomness on a single weight, the ranks are not computed.
        """
        results = {}
        print("ProMCDA starts...")

        self.normalize(normalization_method)
        print(f"Indicators are normalized with: {normalization_method}")

        scores = self.aggregate(aggregation_method)
        print(f"MCDA scores are estimated through: {aggregation_method}")

        ranks = self.evaluate_ranks(scores)
        print(f"MCDA ranks are retrieved")

        if not any([self.robustness == RobustnessAnalysisType.ALL_WEIGHTS,
                    self.robustness == RobustnessAnalysisType.INDICATORS,
                    self.robustness == RobustnessAnalysisType.SINGLE_WEIGHTS]):
            results = {
                 "scores": scores,
                 "ranks": ranks
            }
        elif self.robustness == RobustnessAnalysisType.ALL_WEIGHTS:
            avg, normalized_avg, std = self.get_aggregated_values_with_robustness_weights()
            results = {
                 "normalized_scores": normalized_avg,
                 "average_scores": avg,
                 "standard deviations": std,
                 "ranks": self.evaluate_ranks(avg)
            }
        elif self.robustness == RobustnessAnalysisType.SINGLE_WEIGHTS:
            avg, normalized_avg, std = self.get_aggregated_values_with_robustness_one_weight()
            results = {
                 "normalized_scores": normalized_avg,
                 "average_scores": avg,
                 "standard deviations": std
            }
        elif self.robustness == RobustnessAnalysisType.INDICATORS:
            avg, normalized_avg, std = self.get_aggregated_values_with_robustness_indicators()
            results = {
                 "normalized_scores": normalized_avg,
                 "average_scores": avg,
                 "standard deviations": std,
                 "ranks": self.evaluate_ranks(avg)
            }

        print("ProMCDA completed evaluation.")
        return results