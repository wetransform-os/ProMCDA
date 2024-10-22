import os
import shutil
import unittest
import warnings

import pandas as pd
from mcda.models.ProMCDA import ProMCDA


class TestProMCDA(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("error", category=ResourceWarning)
        # Mock input data for testing
        self.input_matrix = pd.DataFrame({
            'Criteria 1': [0.5, 0.2, 0.8],
            'Criteria 2': [0.3, 0.6, 0.1]
        }, index=['A', 'B', 'C'])
        self.polarity = ('+', '-',)

        self.sensitivity = {
            'sensitivity_on': 'no',
            'normalization': 'minmax',
            'aggregation': 'weighted_sum'
        }

        self.robustness = {
            'robustness_on': 'no',
            'on_single_weights': 'yes',
            'on_all_weights': 'no',
            'on_indicators': 'no',
            'given_weights': [0.6, 0.4]
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

    def test_normalize_single_method(self):
        """
        Test normalization with a single methods.
        Test the correctness of the output values happens in unit_tests/test_normalization.py
        """
        # Given
        self.sensitivity['sensitivity_on'] = 'no'

        # When
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        normalized_matrix = promcda.normalize()

        # Then
        self.assertIsInstance(normalized_matrix, pd.DataFrame)

    def test_normalize_multiple_methods(self):
        """
        Test normalization with multiple methods.
        Test the correctness of the output values happens in unit_tests/test_normalization.py
        """
        self.sensitivity['sensitivity_on'] = 'yes'
        self.sensitivity['normalization'] = ['minmax', 'standardized', 'rank', 'target']

        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                              self.output_path)
        normalized_matrices = promcda.normalize()

        self.assertIsInstance(normalized_matrices, dict)
        self.assertIn('minmax', normalized_matrices)
        self.assertIn('standardized', normalized_matrices)
        self.assertIn('rank', normalized_matrices)
        self.assertIn('target', normalized_matrices)

        self.assertIsInstance(normalized_matrices['minmax'], pd.DataFrame)
        self.assertIsInstance(normalized_matrices['standardized'], pd.DataFrame)
        self.assertIsInstance(normalized_matrices['rank'], pd.DataFrame)
        self.assertIsInstance(normalized_matrices['target'], pd.DataFrame)


    def tearDown(self):
        """
        Clean up temporary directories and files after each test.
        """
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

if __name__ == '__main__':
    unittest.main()


