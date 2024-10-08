import sys
import time
import logging
import pandas as pd
from typing import Tuple

from mcda.configuration.configuration_validator import extract_configuration_values, check_configuration_values

log = logging.getLogger(__name__)

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")


class ProMCDA:
    def __init__(self, input_matrix:pd.DataFrame, polarity:Tuple[str], sensitivity:dict, robustness:dict,
                 monte_carlo:dict):
        """
        Initialize the ProMCDA class with configuration parameters.

        :param input_matrix: DataFrame containing the alternatives and criteria.
        :param polarity: List of polarity for each indicator (+ or -).
        :param sensitivity: Sensitivity analysis configuration.
        :param robustness: Robustness analysis configuration.
        :param monte_carlo: Monte Carlo sampling configuration.
        """
        self.input_matrix = input_matrix
        self.polarity = polarity
        self.sensitivity = sensitivity
        self.robustness = robustness
        self.monte_carlo = monte_carlo

        #self.validate_input_parameters_keys # TODO: still need a formal check as made in old config class,
                                             # maybe use some of following functions validate_
        is_robustness_indicators = self.validate_inputs()
        self.run_mcda(is_robustness_indicators)

        self.normalized_matrix = None
        self.aggregated_matrix = None
        self.ranked_matrix = None

        #self.logger = logging.getLogger("ProMCDA")


    def validate_inputs(self) -> int:
        """
        Extract and validate input configuration parameters to ensure they are correct.
        Return a flag indicating whether robustness analysis will be performed on indicators (1) or not (0).
        """
        configuration_values = extract_configuration_values(self.input_matrix, self.polarity, self.sensitivity,
                                                            self.robustness, self.monte_carlo)
        is_robustness_indicators = check_configuration_values(configuration_values)

        # Validate input TODO: move into a different function validate_input_parameters_keys
        #self.validate_normalization(self.sensitivity['normalization'])
        #self.validate_aggregation(self.sensitivity['aggregation'])
        #self.validate_robustness(self.robustness)

        return is_robustness_indicators

    def validate_normalization(self, f_norm):
        """
        Validate the normalization method.
        """
        valid_norm_methods = ['minmax', 'target', 'standardized', 'rank']
        if f_norm not in valid_norm_methods:
            raise ValueError(f"Invalid normalization method: {f_norm}. Available methods: {valid_norm_methods}")

    def validate_aggregation(self, f_agg):
        """
        Validate the aggregation method.
        """
        valid_agg_methods = ['weighted_sum', 'geometric', 'harmonic', 'minimum']
        if f_agg not in valid_agg_methods:
            raise ValueError(f"Invalid aggregation method: {f_agg}. Available methods: {valid_agg_methods}")

    def validate_robustness(self, robustness):
        """
        Validate robustness analysis settings.
        """
        if not isinstance(robustness, dict):
            raise ValueError("Robustness settings must be a dictionary.")

        # Add more specific checks based on robustness config structure
        if robustness['on_single_weights'] == 'yes' and robustness['on_all_weights'] == 'yes':
            raise ValueError("Conflicting settings for robustness analysis on weights.")

    def normalize(self):
        """
        Normalize the decision matrix based on the configuration.
        """
        f_norm = self.sensitivity['normalization']
        self.logger.info("Normalizing matrix with method: %s", f_norm)

        # Perform normalization (replace this with actual logic)
        self.normalized_matrix = normalize_matrix(self.input_matrix, f_norm)

        return self.normalized_matrix

    def aggregate(self):
        """
        Aggregate the decision matrix based on the configuration.
        """
        f_agg = self.sensitivity['aggregation']
        self.logger.info("Aggregating matrix with method: %s", f_agg)

        # Perform aggregation (replace this with actual logic)
        self.aggregated_matrix = aggregate_matrix(self.normalized_matrix, f_agg)

        return self.aggregated_matrix

    def run_mcda(self, is_robustness_indicators: int):
        """
        Execute the full ProMCDA process.
        """
        start_time = time.time()

        # Normalize
        self.normalize()

        # Aggregate
        self.aggregate()

        # Run
        # no uncertainty
        if is_robustness_indicators == 0:
            run_mcda_without_indicator_uncertainty(input_config, index_column_name, index_column_values,
                                                   input_matrix_no_alternatives, weights, f_norm, f_agg,
                                                   is_robustness_weights)
        # uncertainty
        else:
            run_mcda_with_indicator_uncertainty(input_config, input_matrix_no_alternatives, index_column_name,
                                                index_column_values, mc_runs, random_seed, is_sensitivity, f_agg,
                                                f_norm,
                                                weights, polar, marginal_distribution)

        elapsed_time = time.time() - start_time
        self.logger.info("ProMCDA finished calculations in %s seconds", elapsed_time)

    def get_results(self):
        """
        Return the final results as a DataFrame or other relevant structure.
        """
        # Return the aggregated results (or any other relevant results)
        return self.aggregated_matrix


# Example of instantiating the class and using it
promcda_object = ProMCDA(input_matrix, polarity, sensitivity, robustness, monte_carlo)
promcda_object.run_mcda()
df_normalized = promcda_object.normalized_matrix
df_aggregated = promcda_object.get_results()