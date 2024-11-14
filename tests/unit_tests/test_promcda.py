import os
import shutil
import unittest
import warnings

import pandas as pd

from mcda.models.ProMCDA import ProMCDA
from mcda.configuration.enums import NormalizationFunctions, AggregationFunctions, OutputColumnNames4Sensitivity, \
    NormalizationNames4Sensitivity


class TestProMCDA(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("error", category=ResourceWarning)
        # Mock input data for testing
        self.input_matrix = pd.DataFrame({
            'Alternatives': ['A', 'B', 'C'],
            'Criteria 1': [0.5, 0.2, 0.8],
            'Criteria 2': [0.3, 0.6, 0.1]
        })
        self.input_matrix.set_index('Alternatives', inplace=True)
        self.polarity = ('+', '-',)

        self.sensitivity = {
            'sensitivity_on': 'no',
            'normalization': NormalizationFunctions.MINMAX,
            'aggregation': AggregationFunctions.WEIGHTED_SUM
        }

        self.robustness = {
            'robustness_on': 'no',
            'on_single_weights': 'yes',
            'on_all_weights': 'no',
            'on_indicators': 'no',
            'given_weights': [0.6, 0.7]
        }

        self.monte_carlo = {
            'monte_carlo_runs': 1000,
            'num_cores': 2,
            'random_seed': 42,
            'marginal_distribution_for_each_indicator': 'normal'
        }

        self.output_path = 'mock_output/'

    def test_init(self):
        """
        Test if ProMCDA initializes correctly.
        """
        # Given
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        # Then
        self.assertEqual(promcda.input_matrix.shape, (3, 2))
        self.assertEqual(promcda.polarity, self.polarity)
        self.assertEqual(promcda.sensitivity, self.sensitivity)
        self.assertEqual(promcda.robustness, self.robustness)
        self.assertEqual(promcda.monte_carlo['monte_carlo_runs'], 1000)

    def test_validate_inputs(self):
        """
        Test if input validation works and returns the expected values.
        """
        # Given
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        # When
        (is_robustness_indicators, is_robustness_weights, polar, weights, config) = promcda.validate_inputs()

        # Then
        self.assertIsInstance(is_robustness_indicators, int)
        self.assertIsInstance(is_robustness_weights, int)
        self.assertIsInstance(polar, tuple)
        self.assertIsInstance(weights, list)
        self.assertIsInstance(config, dict)
        self.assertEqual(is_robustness_indicators, 0)
        self.assertEqual(is_robustness_weights, 0)

    def test_normalize_all_methods(self):
        # Given
        self.sensitivity['sensitivity_on'] = 'yes'
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)


        # TODO: delete if return is not a dic
        expected_suffixes = [method.value for method in NormalizationNames4Sensitivity]

        # When
        normalized_values = promcda.normalize()

        # Then
        # TODO: delete if return is not a dic
        actual_suffixes = {col.split('_',1)[1] for col in normalized_values.columns}
        self.assertCountEqual(actual_suffixes, expected_suffixes,
                              "Not all methods were applied or extra suffixes found in column names.")

    def test_normalize_specific_method(self):
        # Given
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        method = 'minmax'

        # When
        normalized_values = promcda.normalize(method=method)
        expected_keys = ['0_minmax_01', '1_minmax_01', '0_minmax_without_zero', '1_minmax_without_zero']

        # Then
        self.assertCountEqual(expected_keys, list(normalized_values.keys()))
        self.assertEqual(list(normalized_values), expected_keys)

    def test_aggregate_all_methods(self):
        # Given
        self.sensitivity['sensitivity_on'] = 'yes'
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        aggregated_scores = promcda.aggregate(normalization_method=None,
                                              aggregation_method=None,
                                              weights=self.robustness['given_weights'])

        # When
        expected_columns = [
            'minmax_weighted_sum', 'target_weighted_sum', 'standardized_weighted_sum', 'rank_weighted_sum',
            'minmax_geometric', 'minmax_minimum', 'target_geometric', 'target_minimum', 'standardized_geometric',
            'standardized_minimum', 'rank_geometric', 'rank_minimum']

        # Then
        self.assertCountEqual(aggregated_scores.columns, expected_columns,
                              "Not all methods were applied or extra columns found.")
        self.assertEqual(len(aggregated_scores), len(self.input_matrix),
                         "Number of alternatives does not match input matrix rows.")

    def test_aggregate_with_specific_normalization_and_aggregation_methods(self):
        # Given
        normalization_method = 'minmax'
        aggregation_method = 'weighted_sum'
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        aggregated_scores = promcda.aggregate(normalization_method=normalization_method,
                                              aggregation_method=aggregation_method,
                                              weights=self.robustness['given_weights'])

        # When
        expected_columns = ['minmax_weighted_sum']

        # Then
        self.assertCountEqual(aggregated_scores.columns, expected_columns, "Only specified methods should be applied.")
        self.assertTrue(
            (aggregated_scores['minmax_weighted_sum'] >= 0).all() and (aggregated_scores['minmax_weighted_sum'] <= 1).all(),
            "Values should be in the range [0, 1] for minmax normalization with weighted sum.")



    def tearDown(self):
        """
        Clean up temporary directories and files after each test.
        """
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

if __name__ == '__main__':
    unittest.main()


