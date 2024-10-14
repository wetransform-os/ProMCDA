import unittest
import pandas as pd
from mcda.models.ProMCDA import ProMCDA


class TestProMCDA(unittest.TestCase):

    def setUp(self):
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
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        self.assertEqual(promcda.input_matrix.shape, (3, 3))
        self.assertEqual(promcda.polarity, self.polarity)
        self.assertEqual(promcda.sensitivity, self.sensitivity)
        self.assertEqual(promcda.robustness, self.robustness)
        self.assertEqual(promcda.monte_carlo['monte_carlo_runs'], 1000)

    def test_validate_inputs(self):
        """
        Test if input validation works and returns the expected values.
        """
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        is_robustness_indicators, polar, weights, config = promcda.validate_inputs()

        # Validate the result
        self.assertIsInstance(is_robustness_indicators, int)
        self.assertIsInstance(polar, tuple)
        self.assertIsInstance(weights, list)
        self.assertEqual(is_robustness_indicators, 0)

    # You can write additional tests for normalization, aggregation, etc.