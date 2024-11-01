import sys
import time
import logging
import pandas as pd
from typing import Tuple, List, Union


from mcda.configuration.configuration_validator import extract_configuration_values, check_configuration_values, \
    check_configuration_keys
from mcda.models.mcda_without_robustness import MCDAWithoutRobustness
from mcda.utils.utils_for_main import run_mcda_without_indicator_uncertainty, run_mcda_with_indicator_uncertainty, \
    check_input_matrix

log = logging.getLogger(__name__)

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")


class ProMCDA:
    def __init__(self, input_matrix: pd.DataFrame, polarity: Tuple[str, ...], sensitivity: dict, robustness: dict,
                 monte_carlo: dict, output_path: str):
        """
        Initialize the ProMCDA class with configuration parameters.

        :param input_matrix: DataFrame containing the alternatives and criteria.
        :param polarity: List of polarity for each indicator (+ or -).
        :param sensitivity: Sensitivity analysis configuration.
        :param robustness: Robustness analysis configuration.
        :param monte_carlo: Monte Carlo sampling configuration.
        :param output_path: path for saving output files.

        # Example of instantiating the class and using it
            promcda = ProMCDA(input_matrix, polarity, sensitivity, robustness, monte_carlo)
            promcda.run_mcda()
            df_normalized = promcda.normalize()
            df_aggregated = promcda.aggregate()
        """
        self.logger = logging.getLogger("ProMCDA")
        self.input_matrix = input_matrix
        self.polarity = polarity
        self.sensitivity = sensitivity
        self.robustness = robustness
        self.monte_carlo = monte_carlo
        self.output_path = output_path

        # Check configuration dictionary keys and handle potential issues
        # TODO: revisit this logic when substitute classes to handle configuration settings
        try:
            check_configuration_keys(self.sensitivity, self.robustness, self.monte_carlo)
        except KeyError as e:
            print(f"Configuration Error: {e}")
            raise  # Optionally re-raise the error after logging it

        self.configuration_settings = extract_configuration_values(self.input_matrix, self.polarity, self.sensitivity,
                                                                   self.robustness, self.monte_carlo, self.output_path)
        is_robustness_indicators, is_robustness_weights, polar, weights, configuration_settings = self.validate_inputs()
        self.run_mcda(is_robustness_indicators, is_robustness_weights, weights)

        self.normalized_matrix = None
        self.aggregated_matrix = None
        self.ranked_matrix = None

    def validate_inputs(self) -> Tuple[int, int, list, Union[list, List[list], dict], dict]:
        """
        Extract and validate input configuration parameters to ensure they are correct.
        Return a flag indicating whether robustness analysis will be performed on indicators (1) or not (0).
        """

        configuration_values = extract_configuration_values(self.input_matrix, self.polarity, self.sensitivity,
                                                            self.robustness, self.monte_carlo, self.output_path)
        is_robustness_indicators, is_robustness_weights, polar, weights = check_configuration_values(
            configuration_values)

        return is_robustness_indicators, is_robustness_weights, polar, weights, configuration_values

    def normalize(self, method=None) -> dict:
        """
        Normalize the input data using the specified method.

        Parameters:
        - method (optional): The normalization method to use. If None, all available methods will be applied.

        Returns:
        - A dictionary containing the normalized values of each indicator per normalization method.
        """
        input_matrix_no_alternatives = check_input_matrix(self.input_matrix)
        mcda_without_robustness = MCDAWithoutRobustness(self.configuration_settings, input_matrix_no_alternatives)
        normalized_values = mcda_without_robustness.normalize_indicators(method)

        return normalized_values

    def aggregate(self, normalization_method=None, aggregation_method=None, weights=None):
        """
        Aggregate normalized indicators using the specified method.

        Parameters (optional):
        - normalization_method: The normalization method to use. If None, all available methods will be applied.
        - aggregation_method: The aggregation method to use. If None, all available methods will be applied.
        - weights: The weights to be used for aggregation. If None, they are set all the same.

        Returns:
        - A DataFrame containing the aggregated scores.
        """

        input_matrix_no_alternatives = check_input_matrix(self.input_matrix)
        mcda_without_robustness = MCDAWithoutRobustness(self.configuration_settings, input_matrix_no_alternatives)
        normalized_indicators = self.normalize(normalization_method)

        aggregated_scores = mcda_without_robustness.aggregate_indicators(
            normalized_indicators=normalized_indicators,
            weights=weights,
            method=aggregation_method
        )

        return aggregated_scores


        # def normalize(self, feature_range=(0, 1)) -> Union[pd.DataFrame, dict]:
        #     """
        #     Normalize the decision matrix based on the configuration `f_norm`.
        #
        #     If `f_norm` is a string representing a single normalization method,
        #     it applies that method to the decision matrix.
        #
        #     If `f_norm` is a list of functions, each normalization function will be
        #     applied to the input matrix sequentially, and the results will be stored
        #     in a dictionary where the keys are function names.
        #
        #     Args:
        #         feature_range (tuple): Range for normalization methods that require it, like MinMax normalization.
        #                                The range (0.1, 1) is not needed when no aggregation will follow.
        #
        #     Returns:
        #         A single normalized DataFrame or a dictionary of DataFrames if multiple
        #         normalization methods are applied.
        #     """
        # normalization = Normalization(self.input_matrix, self.polarity)
        #
        # sensitivity_on = self.sensitivity['sensitivity_on']
        # f_norm = self.sensitivity['normalization']
        # if isinstance(f_norm, NormalizationFunctions):
        #     f_norm = f_norm.value
        #
        # if sensitivity_on == "yes":
        #     self.normalized_matrix = {}
        #     for norm_function in f_norm:
        #         self.logger.info("Applying normalization method: %s", norm_function)
        #         norm_method = getattr(normalization, norm_function, None)
        #         if norm_function in {NormalizationFunctions.MINMAX.value, NormalizationFunctions.STANDARDIZED.value,
        #                              NormalizationFunctions.TARGET.value}:
        #             result = norm_method(feature_range)
        #             if result is None:
        #                 raise ValueError(f"{norm_function} method returned None")
        #             self.normalized_matrix[norm_function] = result
        #         else:
        #             result = normalization.rank()
        #             if result is None:
        #                 raise ValueError(f"{norm_function} method returned None")
        #             self.normalized_matrix[norm_function] = result
        # else:
        #     self.logger.info("Normalizing matrix with method(s): %s", f_norm)
        #     norm_method = getattr(normalization, f_norm, None)
        #     if f_norm in {NormalizationFunctions.MINMAX.value, NormalizationFunctions.STANDARDIZED.value,
        #                   NormalizationFunctions.TARGET.value}:
        #         result = norm_method(feature_range)
        #         if result is None:
        #             raise ValueError(f"{f_norm} method returned None")
        #         self.normalized_matrix = result
        #     else:
        #         result = norm_method()
        #         if result is None:
        #             raise ValueError(f"{f_norm} method returned None")
        #         self.normalized_matrix = result
        #
        # return self.normalized_matrix


    # def aggregate(self, normalized_matrix=None) -> Union[pd.DataFrame, dict]:
    #     """
    #     Aggregate the normalized indicators based on the configuration.
    #
    # Parameters:
    #     normalized_matrix (pd.DataFrame, optional): The matrix to aggregate.
    #                                                 Defaults to self.normalized_matrix.
    # Raises:
    #     ValueError: If no normalized matrix is provided or normalization was not performed.
    #
    # Returns:
    #     The aggregated matrix or dictionary of aggregated results.
    #     """
    #     normalized_matrix = normalized_matrix if normalized_matrix is not None else self.normalized_matrix
    #
    #     if normalized_matrix is None or (
    #             isinstance(self.normalized_matrix, pd.DataFrame) and normalized_matrix.empty) or \
    #             (isinstance(self.normalized_matrix, dict) and all(df.empty for df in self.normalized_matrix.values())):
    #         raise ValueError("Normalization must be performed before aggregation.")
    #
    #     configuration_values = extract_configuration_values(self.input_matrix, self.polarity, self.sensitivity,
    #                                                         self.robustness, self.monte_carlo, self.output_path)
    #
    #     weights = configuration_values["given_weights"]
    #     sensitivity_on = self.sensitivity['sensitivity_on']
    #     aggregation = Aggregation(weights)
    #     f_agg = self.sensitivity['aggregation']
    #     if isinstance(f_agg, AggregationFunctions):
    #         f_agg = f_agg.value
    #     self.logger.info("Aggregating with method: %s", f_agg)
    #
    #     if sensitivity_on == "yes":
    #         self.aggregated_matrix = {}
    #         for agg_function in f_agg:
    #             self.logger.info("Applying aggregation methods: %s", agg_function)
    #
    #             if isinstance(normalized_matrix, dict):
    #                 self.aggregated_matrix = {}
    #                 for norm_method, norm_df in normalized_matrix.items():
    #                     agg_method = getattr(aggregation, agg_function, None)
    #                     result = agg_method(norm_df)
    #                     if result is None:
    #                         raise ValueError(f"{f_agg} method returned None for {norm_method}")
    #                     result_key = f"{f_agg}_{norm_method}"
    #                     self.aggregated_matrix[result_key] = result
    #
    #             if result is None:
    #                 raise ValueError(f"{agg_function} method returned None")
    #                 self.aggregated_matrix[agg_function] = result
    #     else:
    #         self.logger.info("Applying aggregation method: %s", f_agg)
    #         agg_method = getattr(aggregation, f_agg, None)
    #         result = agg_method(normalized_matrix)
    #         if result is None:
    #             raise ValueError(f"{f_agg} method returned None")
    #         self.aggregated_matrix = result
    #
    #     return self.aggregated_matrix


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
        self.logger.info("ProMCDA finished calculations in %s seconds", elapsed_time)


    # def get_results(self):
    #     """
    #     Return the final results as a DataFrame or other relevant structure.
    #     """
    #     # Return the aggregated results (or any other relevant results)
    #     return self.aggregated_matrix
