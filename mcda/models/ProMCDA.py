import time
import pandas as pd
from typing import Tuple, List, Union, Optional

from mcda.configuration.configuration_validator import extract_configuration_values, check_configuration_values, \
    check_configuration_keys
from mcda.models.mcda_without_robustness import MCDAWithoutRobustness
from mcda.utils.utils_for_main import run_mcda_without_indicator_uncertainty, run_mcda_with_indicator_uncertainty, \
    check_input_matrix


class ProMCDA:
    def __init__(self, input_matrix: pd.DataFrame, polarity: Tuple[str, ...], robustness_weights: Optional[bool] = None,
                 robustness_indicators: Optional[bool] = None, marginal_distributions: Optional[bool] = None,
                 num_runs: Optional[int] = 10000, num_cores: Optional[int] = 1, random_seed: Optional[int] = 43):
        """
        Initialize the ProMCDA class with configuration parameters.

        # Required parameters
        :param input_matrix: DataFrame containing the alternatives and criteria.
        :param polarity: Tuple of polarities for each indicator ("+" or "-").

        # Optional parameters
        :param robustness_weights: Boolean flag indicating whether to perform robustness analysis on weights
                                   (True or False).
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
        self.robustness_indicators = robustness_indicators
        self.num_runs = num_runs
        self.marginal_distributions = marginal_distributions
        self.num_cores = num_cores
        self.random_seed = random_seed
        self.normalized_matrix = None
        self.scores = None


    def validate_inputs(self) -> Tuple[int, int, list, Union[list, List[list], dict], dict]:
        """
        Extract and validate input configuration parameters to ensure they are correct.
        Return a flag indicating whether robustness analysis will be performed on indicators (1) or not (0).
        """

        configuration_values = extract_configuration_values(self.input_matrix, self.polarity,
                                                            self.robustness, self.monte_carlo)
        is_robustness_indicators, is_robustness_weights, polar, weights = check_configuration_values(
            configuration_values)

        return is_robustness_indicators, is_robustness_weights, polar, weights, configuration_values

    def normalize(self, method=None) -> pd.DataFrame:
        """
        Normalize the input data using the specified method.

        # TODO: for now normalize works only with indicators without uncertanties. Review this logic if needed.
        Notes:
        The normalizations methods are defined in the NormalizationFunctions enum class.
        This method expects the input matrix to not have uncertainties on the indicators.

        Parameters:
        - method (optional): The normalization method to use. If None, all available methods will be applied for a
                             Sensitivity Analysis.

        Returns:
        - A pd.DataFrame containing the normalized values of each indicator per normalization method.

        :param method: str
        :return normalized_df: pd.DataFrame
        """
        input_matrix_no_alternatives = check_input_matrix(self.input_matrix)
        mcda_without_robustness = MCDAWithoutRobustness(self.configuration_settings, input_matrix_no_alternatives)
        normalized_values = mcda_without_robustness.normalize_indicators(method)

        return normalized_values

    def aggregate(self, normalization_method=None, aggregation_method=None, weights=None) -> pd.DataFrame:
        """
        Aggregate normalized indicators using the specified method.

        # TODO: for now aggregate works only with indicators without uncertanties. Review this logic if needed.
        Notes:
        The aggregation methods are defined in the AggregationFunctions enum class.
        This method expects the input matrix to not have uncertainties on the indicators.

        Parameters (optional):
        - normalization_method: The normalization method to use. If None, all available methods will be applied for a
                                Sensitivity Analysis.
        - aggregation_method: The aggregation method to use. If None, all available methods will be applied.
        - weights: The weights to be used for aggregation. If None, they are set all the same.

        Returns:
        - A DataFrame containing the aggregated scores per normalization and aggregation methods.
        """

        input_matrix_no_alternatives = check_input_matrix(self.input_matrix)
        mcda_without_robustness = MCDAWithoutRobustness(self.configuration_settings, input_matrix_no_alternatives)
        normalized_indicators = self.normalize(normalization_method)

        aggregated_scores = mcda_without_robustness.aggregate_indicators(
            normalized_indicators=normalized_indicators,
            weights=weights,
            agg_method=aggregation_method
        )

        return aggregated_scores


    # def aggregate_with_robustness(self, normalization_method=None, aggregation_method=None, weights=None,
    #                polarity: list, robustness: dict, monte_carlo: dict) -> pd.DataFrame:
    #     """
    #     Estimate scores of alternatives using the specified normalization and aggregation methods.
    #
    #
    #     Notes:
    #     The aggregation methods are defined in the AggregationFunctions enum class.
    #
    #     Parameters (optional):
    #     - normalization_method: The normalization method to use. If None, all available methods will be applied.
    #     - aggregation_method: The aggregation method to use. If None, all available methods will be applied.
    #     - weights: The weights to be used for aggregation. If None, they are set all the same.
    #     - polarity: List of polarity for each indicator (+ or -).
    #     - robustness: Robustness analysis configuration.
    #     - monte_carlo: Monte Carlo sampling configuration.
    #
    #     Returns:
    #     - A DataFrame containing the aggregated scores per normalization and aggregation methods.
    #     """
    #
    #     input_matrix_no_alternatives = check_input_matrix(self.input_matrix)
    #     mcda_without_robustness = MCDAWithoutRobustness(self.configuration_settings, input_matrix_no_alternatives)
    #     normalized_indicators = self.normalize(normalization_method)
    #
    #     aggregated_scores = mcda_without_robustness.aggregate_indicators(
    #         normalized_indicators=normalized_indicators,
    #         weights=weights,
    #         agg_method=aggregation_method
    #     )
    #
    #     return aggregated_scores


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
