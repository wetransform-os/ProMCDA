import time
import pandas as pd
from typing import Tuple, List, Union

from mcda.configuration.configuration_validator import extract_configuration_values, check_configuration_values, \
    check_configuration_keys
from mcda.models.mcda_without_robustness import MCDAWithoutRobustness
from mcda.utils.utils_for_main import run_mcda_without_indicator_uncertainty, run_mcda_with_indicator_uncertainty, \
    check_input_matrix


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

    def normalize(self, method=None) -> pd.DataFrame:
        """
        Normalize the input data using the specified method.

        Parameters:
        - method (optional): The normalization method to use. If None, all available methods will be applied.

        Returns:
        - A pd.DataFrame containing the normalized values of each indicator per normalization method.

        :param method: str
        :return normalized_df: pd.DataFrame
        """
        input_matrix_no_alternatives = check_input_matrix(self.input_matrix)
        mcda_without_robustness = MCDAWithoutRobustness(self.configuration_settings, input_matrix_no_alternatives)
        normalized_values = mcda_without_robustness.normalize_indicators(method)

        if method is None:
            normalized_df = pd.concat(normalized_values, axis=1)
            normalized_df.columns = [f"{col}_{method}" for method, cols in normalized_values.items() for col in
                                     input_matrix_no_alternatives.columns]
        else:
            normalized_df = normalized_values

        return normalized_df

    def aggregate(self, normalization_method=None, aggregation_method=None, weights=None) -> pd.DataFrame:
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
