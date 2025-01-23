import unittest
import warnings

import pandas as pd

from mcda.models.ProMCDA import ProMCDA
from mcda.configuration.enums import NormalizationFunctions, AggregationFunctions, OutputColumnNames4Sensitivity, \
    NormalizationNames4Sensitivity, PDFType


class TestProMCDA(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings("error", category=ResourceWarning)
        # Mock input data for testing
        self.input_matrix = pd.DataFrame({
            'Alternatives': ['A', 'B', 'C'],
            'Criterion1': [0.5, 0.2, 0.8],
            'Criterion2': [0.3, 0.6, 0.1]
        })
        self.input_matrix.set_index('Alternatives', inplace=True)

        self.input_matrix_with_uncertainty = pd.DataFrame({
            'Alternatives': ['A', 'B', 'C'],
            'Criterion1_mean': [0.5, 0.2, 0.8],
            'Criterion1_std': [0.1, 0.02, 0.07],
            'Criterion2_mean': [0.3, 0.6, 0.1],
            'Criterion2_std': [0.03, 0.06, 0.01]
        })
        self.input_matrix_with_uncertainty.set_index('Alternatives', inplace=True)

        self.polarity = ('+', '-',)

        # Define optional parameters
        self.robustness_weights = False
        self.robustness_indicators = False
        self.marginal_distributions = (PDFType.NORMAL, PDFType.NORMAL)
        self.num_runs = 5
        self.num_cores = 2
        self.random_seed = 123

    def test_init(self):
        """
        Test if ProMCDA initializes correctly.
        """
        # Given
        promcda = ProMCDA(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            robustness_weights=self.robustness_weights,
            robustness_indicators=self.robustness_indicators,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )

        # Then
        self.assertEqual(promcda.input_matrix.shape, (3, 2))
        self.assertEqual(promcda.polarity, self.polarity)
        self.assertFalse(promcda.robustness_weights)
        self.assertFalse(promcda.robustness_indicators)
        self.assertEqual(promcda.marginal_distributions, self.marginal_distributions)
        self.assertEqual(promcda.num_runs, self.num_runs)
        self.assertEqual(promcda.num_cores, self.num_cores)
        self.assertEqual(promcda.random_seed, self.random_seed)
        self.assertIsNone(promcda.normalized_values_without_robustness)
        self.assertIsNone(promcda.normalized_values_with_robustness)
        self.assertIsNone(promcda.aggregated_scores)
        self.assertIsNone(promcda.all_indicators_scores_means)
        self.assertIsNone(promcda.all_indicators_scores_stds)
        self.assertIsNone(promcda.all_indicators_means_scores_normalized)
        self.assertIsNone(promcda.all_indicators_scores_stds_normalized)
        self.assertIsNone(promcda.all_weights_score_means)
        self.assertEqual(promcda.all_weights_score_stds, (None,))
        self.assertEqual(promcda.all_weights_score_means_normalized, (None,))
        self.assertEqual(promcda.all_weights_score_stds_normalized, (None,))
        self.assertEqual(promcda.iterative_random_w_score_means, (None,))
        self.assertEqual(promcda.iterative_random_w_score_stds, (None,))
        self.assertIsNone(promcda.iterative_random_w_score_means_normalized)
        #TODO: self.assertIsNone(promcda.scores)

    # def test_validate_inputs(self):
    #     """
    #     Test if input validation works and returns the expected values.
    #     """
    #     # Given
    #     promcda = ProMCDA(self.input_matrix, self.polarity, self.robustness, self.monte_carlo)
    #     # When
    #     (is_robustness_indicators, is_robustness_weights, polar, weights, config) = promcda.validate_inputs()
    #
    #     # Then
    #     self.assertIsInstance(is_robustness_indicators, int)
    #     self.assertIsInstance(is_robustness_weights, int)
    #     self.assertIsInstance(polar, tuple)
    #     self.assertIsInstance(weights, list)
    #     self.assertIsInstance(config, dict)
    #     self.assertEqual(is_robustness_indicators, 0)
    #     self.assertEqual(is_robustness_weights, 0)

    def test_normalize_all_methods(self):
        """
        Test normalization under sensitivity analysis (all methods are used).
        """
        # Given
        normalization_method = None
        promcda = ProMCDA(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            robustness_weights=self.robustness_weights,
            robustness_indicators=self.robustness_indicators,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )

        # When
        expected_suffixes = [method.value for method in NormalizationNames4Sensitivity]
        normalized_values = promcda.normalize(normalization_method)
        actual_suffixes = {"_".join(col.split("_", 2)[1:]) for col in normalized_values.columns}

        # Then
        self.assertCountEqual(actual_suffixes, expected_suffixes,
                              "Not all methods were applied or extra suffixes found in column names.")

    def test_normalize_specific_method(self):
        """
        Test normalization when a specific method is selected (MINMAX).
        """
        # Given
        promcda = ProMCDA(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            robustness_weights=self.robustness_weights,
            robustness_indicators=self.robustness_indicators,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )

        # When
        normalized_values = promcda.normalize(normalization_method=NormalizationFunctions.MINMAX)
        expected_keys = ['Criterion1_minmax_01', 'Criterion2_minmax_01', 'Criterion1_minmax_without_zero',
                         'Criterion2_minmax_without_zero']

        # Then
        self.assertCountEqual(expected_keys, list(normalized_values.keys()))
        self.assertEqual(list(normalized_values), expected_keys)

    def test_normalization_with_robustness(self):
        """
        Test normalization when a specific method is selected (MINMAX) and
        under robustness analysis (criteria with uncertainty).
        """
        # Given
        robustness_indicators = True
        promcda = ProMCDA(
            input_matrix=self.input_matrix_with_uncertainty,
            polarity=self.polarity,
            robustness_weights=self.robustness_weights,
            robustness_indicators=robustness_indicators,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )

        # When
        promcda.normalize(normalization_method=NormalizationFunctions.MINMAX)

        # Then
        normalized_values = promcda.get_normalized_values_with_robustness()
        self.assertIsNotNone(normalized_values)
        self.assertEqual(len(normalized_values), self.num_runs)

    def test_aggregate_all_methods(self):
        """
        Test aggregation under sensitivity analysis (all methods are used).
        """
        # Given
        promcda = ProMCDA(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            robustness_weights=self.robustness_weights,
            robustness_indicators=self.robustness_indicators,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )
        promcda.normalize()

        # When
        aggregated_scores = promcda.aggregate()
        expected_columns = [e.value for e in OutputColumnNames4Sensitivity]

        # Then
        self.assertCountEqual(aggregated_scores.columns, expected_columns,
                              "Not all methods were applied or extra columns found.")
        self.assertEqual(len(aggregated_scores), len(self.input_matrix),
                         "Number of alternatives does not match input matrix rows.")

    def test_aggregate_with_specific_aggregation_method(self):
        """
        Test aggregation when a specific method is selected (MINMAX/WEIGHTED_SUM).
        """
        # Given
        normalization_method = NormalizationFunctions.MINMAX
        aggregation_method = AggregationFunctions.WEIGHTED_SUM

        # When
        promcda = ProMCDA(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            robustness_weights=self.robustness_weights,
            robustness_indicators=self.robustness_indicators,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )
        promcda.normalize(normalization_method)
        aggregated_scores = promcda.aggregate(aggregation_method=aggregation_method)
        expected_columns = [OutputColumnNames4Sensitivity.WS_MINMAX_01.value]

        # Then
        self.assertCountEqual(aggregated_scores.columns, expected_columns, "Only specified methods should be applied.")
        self.assertTrue(
            (aggregated_scores['ws-minmax_01'] >= 0).all() and (
                        aggregated_scores['ws-minmax_01'] <= 1).all(),
            "Values should be in the range [0, 1] for minmax normalization with weighted sum.")

    def test_aggregate_with_robustness_indicators(self):
        """
        Test aggregation when a specific method is selected (MINMAX/WEIGHTED_SUM) and
        under robustness analysis (criteria with uncertainty).
        """
        # Given
        normalization_method = NormalizationFunctions.MINMAX
        aggregation_method = AggregationFunctions.WEIGHTED_SUM

        # When
        promcda = ProMCDA(
            input_matrix=self.input_matrix_with_uncertainty,
            polarity=self.polarity,
            robustness_weights=self.robustness_weights,
            robustness_indicators=True,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )
        promcda.normalize(normalization_method)
        promcda.aggregate(aggregation_method=aggregation_method)
        aggregated_scores, aggregated_scores_normalized, aggregated_stds = promcda.get_aggregated_values_with_robustness_indicators()
        expected_columns = [OutputColumnNames4Sensitivity.WS_MINMAX_01.value]

        # Then
        self.assertCountEqual(aggregated_scores.columns, expected_columns,
                              "Only specified methods should be applied.")
        self.assertTrue(
            (aggregated_scores['ws-minmax_01'] >= 0).all() and (
                    aggregated_scores['ws-minmax_01'] <= 1).all(),
            "Values should be in the range [0, 1] for minmax normalization with weighted sum.")

    def test_aggregate_with_robustness_weights_no_sensitivity(self):
        """
        Test aggregation when a specific method is selected (MINMAX/WEIGHTED_SUM) and
        under robustness analysis on the weights.
        """
        # Given
        normalization_method = NormalizationFunctions.MINMAX
        aggregation_method = AggregationFunctions.WEIGHTED_SUM

        # When
        promcda = ProMCDA(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            robustness_weights=True,
            robustness_indicators=self.robustness_indicators,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )
        promcda.normalize(normalization_method)
        promcda.aggregate(aggregation_method=aggregation_method)
        aggregated_scores, aggregated_scores_normalized, aggregated_stds = promcda.get_aggregated_values_with_robustness_weights()
        expected_columns = [OutputColumnNames4Sensitivity.WS_MINMAX_01.value]

        # Then
        self.assertCountEqual(aggregated_scores.columns, expected_columns,
                              "Only specified methods should be applied.")
        self.assertTrue(
            (aggregated_scores['ws-minmax_01'] >= 0).all() and (
                    aggregated_scores['ws-minmax_01'] <= 1).all(),
            "Values should be in the range [0, 1] for minmax normalization with weighted sum.")

    def test_aggregate_with_robustness_weights_with_sensitivity(self):
        """
        Test aggregation under sensitivity and robustness analysis on the weights.
        """

        # When
        promcda = ProMCDA(
            input_matrix=self.input_matrix,
            polarity=self.polarity,
            robustness_weights=True,
            robustness_indicators=self.robustness_indicators,
            marginal_distributions=self.marginal_distributions,
            num_runs=self.num_runs,
            num_cores=self.num_cores,
            random_seed=self.random_seed
        )
        promcda.normalize()
        promcda.aggregate()
        aggregated_scores, aggregated_scores_normalized, aggregated_stds = promcda.get_aggregated_values_with_robustness_weights()
        expected_columns = [e.value for e in OutputColumnNames4Sensitivity]

        # Then
        self.assertCountEqual(aggregated_scores.columns, expected_columns,
                              "Only specified methods should be applied.")
        self.assertTrue(
            (aggregated_scores_normalized['ws-minmax_01'] >= 0).all() and (
                    aggregated_scores_normalized['ws-minmax_01'] <= 1).all(),
            "Values should be in the range [0, 1] for minmax normalization with weighted sum.")

        # TODO: clarify why "aggregated_scores" do not range within 0 and 1 but show values < 1 for 'ws-minmax_01'


#    def tearDown(self):
#        """
#        Clean up temporary directories and files after each test.
#        """
#        if os.path.exists(self.output_path):
#            shutil.rmtree(self.output_path)


if __name__ == '__main__':
    unittest.main()
