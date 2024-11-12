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

    def test_normalize_all_methods(self):
        # Given
        self.sensitivity['sensitivity_on'] = 'yes'
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)

        # expected_keys = [method.value for method in NormalizationNames4Sensitivity]
        # TODO: delete if return is not a dic
        expected_suffixes = [method.value for method in NormalizationNames4Sensitivity]

        # When
        normalized_values = promcda.normalize()

        # Then
        #self.assertCountEqual(list(set(normalized_values.keys())), expected_keys,
        #                      "Not all methods were applied or extra keys found.")
        #self.assertEqual(list(normalized_values), (expected_keys))
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

        # Then
        expected_keys = ['Criteria 1_minmax_without_zero', 'Criteria 2_minmax_without_zero', 'Criteria 1_minmax_01',
         'Criteria 2_minmax_01']
        self.assertCountEqual(expected_keys, list(normalized_values.keys()))
        self.assertEqual(list(normalized_values), expected_keys)

    def test_aggregate_all_methods(self):
        # Given
        promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
                          self.output_path)
        aggregated_scores = promcda.aggregate(weights=self.robustness['given_weights'])

        # When
        expected_columns = [
            'ws-minmax_01', 'ws-target_01', 'ws-standardized_any', 'ws-rank',
            'geom-minmax_without_zero', 'geom-target_without_zero', 'geom-standardized_without_zero', 'geom-rank',
            'harm-minmax_without_zero', 'harm-target_without_zero', 'harm-standardized_without_zero', 'harm-rank',
            'min-standardized_any'
        ]

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
        expected_columns = ['ws-minmax_01']

        # Then
        self.assertCountEqual(aggregated_scores.columns, expected_columns, "Only specified methods should be applied.")
        self.assertTrue(
            (aggregated_scores['ws-minmax_01'] >= 0).all() and (aggregated_scores['ws-minmax_01'] <= 1).all(),
            "Values should be in the range [0, 1] for minmax normalization with weighted sum.")



    # def test_normalize_single_method(self):
    #     """
    #     Test normalization with a single methods.
    #     Test the correctness of the output values happens in unit_tests/test_normalization.py
    #     """
    #     # Given
    #     self.sensitivity['sensitivity_on'] = 'no'
    #
    #     # When
    #     promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
    #                       self.output_path)
    #     normalized_matrix = promcda.normalize()
    #
    #     # Then
    #     self.assertIsInstance(normalized_matrix, pd.DataFrame)
    #
    # def test_normalize_multiple_methods(self):
    #     """
    #     Test normalization with multiple methods.
    #     Test the correctness of the output values happens in unit_tests/test_normalization.py
    #     """
    #     # Given
    #     self.sensitivity['sensitivity_on'] = 'yes'
    #     self.sensitivity['normalization'] = [method.value for method in NormalizationFunctions]
    #     promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
    #                       self.output_path)
    #     # When
    #     normalized_matrices = promcda.normalize()
    #
    #     # Then
    #     self.assertIsInstance(normalized_matrices, dict)
    #     self.assertIn(NormalizationFunctions.MINMAX.value, normalized_matrices)
    #     self.assertIn(NormalizationFunctions.STANDARDIZED.value, normalized_matrices)
    #     self.assertIn(NormalizationFunctions.RANK.value, normalized_matrices)
    #     self.assertIn(NormalizationFunctions.TARGET.value, normalized_matrices)
    #
    #     self.assertIsInstance(normalized_matrices[NormalizationFunctions.MINMAX.value], pd.DataFrame)
    #     self.assertIsInstance(normalized_matrices[NormalizationFunctions.STANDARDIZED.value], pd.DataFrame)
    #     self.assertIsInstance(normalized_matrices[NormalizationFunctions.RANK.value], pd.DataFrame)
    #     self.assertIsInstance(normalized_matrices[NormalizationFunctions.TARGET.value], pd.DataFrame)
    #
    # def test_aggregate_with_sensitivity_on(self):
    #     """
    #     Test aggregation when sensitivity_on is 'yes'
    #     """
    #
    #     # Given
    #     self.sensitivity['sensitivity_on'] = 'yes'
    #     self.sensitivity['normalization'] = [method.value for method in NormalizationFunctions]
    #     self.sensitivity['aggregation'] = [method.value for method in AggregationFunctions]
    #
    #     promcda = ProMCDA(self.input_matrix, self.polarity, self.sensitivity, self.robustness, self.monte_carlo,
    #                       self.output_path)
    #
    #     normalized_matrix = { # Mock normalized matrices in a dictionary for sensitivity-on
    #          "minmax": pd.DataFrame({
    #              "indicator_1": [0.1, 0.2, 0.3],
    #              "indicator_2": [0.4, 0.5, 0.6]
    #         }),
    #         "standardized": pd.DataFrame({
    #             "indicator_1": [1, 0.1, 0.7],
    #             "indicator_2": [0.5, 0.1, 0.8]
    #         }),
    #         "target": pd.DataFrame({
    #             "indicator_1": [0.15, 0.25, 0.35],
    #             "indicator_2": [0.45, 0.55, 0.65]
    #         }),
    #         "rank": pd.DataFrame({
    #             "indicator_1": [1, 2, 3],
    #             "indicator_2": [1, 2, 3]
    #         })
    #     }
    #
    #     # When
    #     result = promcda.aggregate(normalized_matrix)
    #     expected_results = {
    #         OutputColumnNames4Sensitivity.WS_MINMAX_01.value: pd.Series([0.25, 0.35, 0.45], name="ws-minmax_01"),
    #         OutputColumnNames4Sensitivity.WS_TARGET_01.value: pd.Series([0.3, 0.4, 0.5], name="ws-target_01"),
    #         OutputColumnNames4Sensitivity.WS_STANDARDIZED_ANY.value: pd.Series([0.8, 0, 0.74], name="ws-standardized_any"),
    #         OutputColumnNames4Sensitivity.WS_RANK.value: pd.Series([1.5, 2.5, 3.5], name="ws-rank"),
    #         OutputColumnNames4Sensitivity.GEOM_MINMAX_WITHOUT_ZERO.value: pd.Series([0.2, 0.3, 0.4], name="geom-minmax_without_zero"),
    #         OutputColumnNames4Sensitivity.GEOM_TARGET_WITHOUT_ZERO.value: pd.Series([0.25, 0.35, 0.45], name="geom-target_without_zero"),
    #         OutputColumnNames4Sensitivity.GEOM_STANDARDIZED_WITHOUT_ZERO.value: pd.Series([0.1, 0, 0.3], name="geom-standardized_without_zero"),
    #         OutputColumnNames4Sensitivity.GEOM_RANK.value: pd.Series([1, 2, 3], name="geom-rank"),
    #         OutputColumnNames4Sensitivity.HARM_MINMAX_WITHOUT_ZERO.value: pd.Series([0.18, 0.27, 0.36], name="harm-minmax_without_zero"),
    #         OutputColumnNames4Sensitivity.HARM_TARGET_WITHOUT_ZERO.value: pd.Series([0.22, 0.33, 0.44], name="harm-target_without_zero"),
    #         OutputColumnNames4Sensitivity.HARM_STANDARDIZED_WITHOUT_ZERO.value: pd.Series([0.05, None, 0.15], name="harm-standardized_without_zero"),
    #         OutputColumnNames4Sensitivity.HARM_RANK.value: pd.Series([0.9, 1.8, 2.7], name="harm-rank"),
    #         OutputColumnNames4Sensitivity.MIN_STANDARDIZED_ANY.value: pd.Series([-1, 0, 1], name="min-standardized_any"),
    #     }
    #     expected_df = pd.DataFrame(expected_results, index=['A', 'B', 'C'])
    #
    #     # Then
    #     #self.assertEqual(result, expected_df)
    #     self.assertIsInstance(result, pd.DataFrame)
    #
    #     def test_aggregate_with_sensitivity_off(self):
    #         """
    #         Test aggregation when sensitivity_on is 'no'
    #         """
    #         # Given
    #         self.promcda.sensitivity['sensitivity_on'] = 'no'
    #         self.promcda.sensitivity['aggregation'] = 'weighted_sum'
    #
    #         # When
    #         expected_result = {
    #             'Min-Max + Weighted Sum': [0.5667, 0.4, 0.6],
    #             'Standardized + Weighted Sum': [-0.1069, -0.3667, 0.3667],
    #             'Rank + Weighted Sum': [2.0, 1.8, 2.2],
    #             'Min-Max + Geometric': [0.5695, 0.0, 0.0],
    #             'Rank + Geometric': [2.0, 1.933, 2.067],
    #             'Min-Max + Harmonic': [0.5455, 0.0, 0.0],
    #             'Rank + Harmonic': [2.0, 1.36, 1.85],
    #             'Min-Max + Minimum': [0.5, 0.0, 0.0],
    #             'Rank + Minimum': [2, 1, 1]
    #         }
    #         df = pd.DataFrame(expected_result, index=['A', 'B', 'C'])
    #
    #         # Then


    def tearDown(self):
        """
        Clean up temporary directories and files after each test.
        """
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

if __name__ == '__main__':
    unittest.main()


