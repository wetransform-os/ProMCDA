import sys
import time
import logging
import pandas as pd
from typing import Tuple, List, Union

from mcda.configuration.configuration_validator import extract_configuration_values, check_configuration_values, \
    check_configuration_keys
from mcda.mcda_functions.normalization import Normalization
from mcda.utils.utils_for_main import run_mcda_without_indicator_uncertainty, run_mcda_with_indicator_uncertainty

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

        is_robustness_indicators, is_robustness_weights, polar, weights, configuration_settings = self.validate_inputs()
        self.run_mcda(is_robustness_indicators, is_robustness_weights, weights, configuration_settings)

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

    def normalize(self, feature_range=(0, 1)) -> Union[pd.DataFrame, dict]:
        """
        Normalize the decision matrix based on the configuration `f_norm`.

        If `f_norm` is a string representing a single normalization method,
        it applies that method to the decision matrix.

        If `f_norm` is a list of functions, each normalization function will be
        applied to the input matrix sequentially, and the results will be stored
        in a dictionary where the keys are function names.

        Args:
            feature_range (tuple): Range for normalization methods that require it, like MinMax normalization.
                                   The range (0.1, 1) is not needed when no aggregation will follow.

        Returns:
            A single normalized DataFrame or a dictionary of DataFrames if multiple
            normalization methods are applied.
        """
        normalization = Normalization(self.input_matrix, self.polarity)

        sensitivity_on = self.sensitivity['sensitivity_on']
        f_norm = self.sensitivity['normalization']
        f_norm_list = ['minmax', 'target', 'standardized', 'rank']

        if sensitivity_on == "yes":
            self.normalized_matrix = {}
            for norm_function in f_norm_list:
                self.logger.info("Applying normalization method: %s", norm_function)
                norm_method = getattr(normalization, norm_function, None)
                if norm_function in ['minmax', 'target', 'standardized']:
                    result = norm_method(feature_range)
                    if result is None:
                        raise ValueError(f"{norm_function} method returned None")
                    self.normalized_matrix[norm_function] = result
                else:
                    result = normalization.rank()
                    if result is None:
                        raise ValueError(f"{norm_function} method returned None")
                    self.normalized_matrix[norm_function] = result
        else:
            self.logger.info("Normalizing matrix with method(s): %s", f_norm)
            norm_method = getattr(normalization, f_norm, None)
            if f_norm in ['minmax', 'target', 'standardized']:
                result = norm_method(feature_range)
                if result is None:
                    raise ValueError(f"{f_norm} method returned None")
                self.normalized_matrix = result
            else:
                result = norm_method()
                if result is None:
                    raise ValueError(f"{f_norm} method returned None")
                self.normalized_matrix = result

            return self.normalized_matrix

    # def aggregate(self):
    #     """
    #     Aggregate the decision matrix based on the configuration.
    #     """
    #     f_agg = self.sensitivity['aggregation']
    #     self.logger.info("Aggregating matrix with method: %s", f_agg)
    #
    #     # Perform aggregation (replace this with actual logic)
    #     self.aggregated_matrix = aggregate_matrix(self.normalized_matrix, f_agg)
    #
    #     return self.aggregated_matrix

    def run_mcda(self, is_robustness_indicators: int, is_robustness_weights: int,
                 weights: Union[list, List[list], dict],
                 configuration_settings: dict):
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
            run_mcda_without_indicator_uncertainty(configuration_settings, is_robustness_weights, weights)
        # uncertainty
        else:
            run_mcda_with_indicator_uncertainty(configuration_settings)

        elapsed_time = time.time() - start_time
        self.logger.info("ProMCDA finished calculations in %s seconds", elapsed_time)

    # def get_results(self):
    #     """
    #     Return the final results as a DataFrame or other relevant structure.
    #     """
    #     # Return the aggregated results (or any other relevant results)
    #     return self.aggregated_matrix
