import os
import unittest
from distutils.core import setup

import pandas as pd
from unittest import TestCase

from promcda.enums import PDFType, NormalizationFunctions, AggregationFunctions, OutputColumnNames4Sensitivity
from promcda.models import ProMCDA

current_directory = os.path.dirname(os.path.abspath(__file__))
resources_directory = os.path.join(current_directory, '..', 'resources')

class TestMCDA_without_robustness(unittest.TestCase):

    @staticmethod
    def setup():
        polarity = ("-", "-", "+", "+", "+", "+")
        weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        robustness_weights = False
        robustness_single_weights = False
        robustness_indicators = False

        # Return the setup parameters as a dictionary
        setup = {
            'input_matrix': None,  # will be given later
            'polarity': polarity,
            'weights': weights,
            'robustness_weights': robustness_weights,
            'robustness_single_weights': robustness_single_weights,
            'robustness_indicators': robustness_indicators,
            'num_cores': 1,
            'random_seed': 42,
            'num_runs': 10,
            'marginal_distributions': tuple([PDFType.EXACT, PDFType.UNIFORM, PDFType.NORMAL, PDFType.POISSON])
        }

        return setup


    @staticmethod
    def setup_robustness_weights():

        polarity = ("-", "-", "+", "+", "+", "+")
        weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        robustness_weights = True
        robustness_single_weights = False
        robustness_indicators = False

        # Return the setup parameters as a dictionary
        setup_robustness_weights = {
            'input_matrix': None,  # will be given later
            'polarity': polarity,
            'weights': weights,
            'robustness_weights': robustness_weights,
            'robustness_single_weights': robustness_single_weights,
            'robustness_indicators': robustness_indicators,
            'num_cores': 1,
            'random_seed': 42,
            'num_runs': 10,
            'marginal_distributions': tuple([PDFType.EXACT, PDFType.UNIFORM, PDFType.NORMAL, PDFType.POISSON])
        }

        return setup_robustness_weights

    @staticmethod
    def get_input_matrix():
        import promcda.utils.utils_for_main as utils_for_main
        input_matrix_file_path = os.path.join(resources_directory, 'input_matrix_without_uncert.csv')
        input_matrix = utils_for_main.read_matrix(input_matrix_file_path)

        return input_matrix

    @staticmethod
    def get_list_of_df():
        list_df = [TestMCDA_without_robustness.get_input_matrix(), TestMCDA_without_robustness.get_input_matrix()]

        return list_df

    def test_normalize_indicators(self):
        # Given
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        setup = TestMCDA_without_robustness.setup()
        polarity = setup['polarity']

        # When
        promcda_sensitivity = ProMCDA(input_matrix, polarity)
        res_sensitivity = promcda_sensitivity.normalize()

        promcda_simple = ProMCDA(input_matrix, polarity)
        res_simple = promcda_simple.normalize(NormalizationFunctions.MINMAX)

        # Then
        assert isinstance(res_sensitivity, pd.DataFrame)
        expected_columns = ['standardized_any', 'standardized_without_zero', 'minmax_01',
                'minmax_without_zero', 'target_01', 'target_without_zero', 'rank']
        for member in expected_columns:
                self.assertTrue(any(member in col for col in res_sensitivity.columns),
                                msg=f"Column containing '{member}' not found in DataFrame")

        assert isinstance(res_simple, pd.DataFrame)
        present_columns = ["minmax_01", "minmax_without_zero"]
        absent_columns = ["standardized_any", "standardized_without_zero", "target_01", "target_without_zero", "rank"]
        for col in present_columns:
            self.assertTrue(any(col in column for column in res_simple.columns))
        for col in absent_columns:
            self.assertFalse(any(col in column for column in res_simple.columns))
        for col in res_simple.columns:
            self.assertEqual(res_simple.shape[0], input_matrix.shape[0])
            self.assertEqual(res_simple.shape[1], input_matrix.shape[1]*2)

    def test_aggregate_indicators_simple(self):
        # Given
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        setup = TestMCDA_without_robustness.setup()
        polarity = setup['polarity']
        weights = setup["weights"]

        # When
        promcda = ProMCDA(input_matrix, polarity, weights)
        promcda.normalize(NormalizationFunctions.MINMAX)
        res_simple_promcda = promcda.aggregate(AggregationFunctions.WEIGHTED_SUM)
        simple_mcda_col_names = ['ws-minmax_01']

        # Then
        assert isinstance(res_simple_promcda, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res_simple_promcda.columns.tolist(), list2=simple_mcda_col_names)
        assert res_simple_promcda.shape[0] == input_matrix.shape[0]
        assert res_simple_promcda.shape[1] == len(simple_mcda_col_names)

    def test_aggregate_indicators_weights_none(self):
        # Given
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        setup = TestMCDA_without_robustness.setup()
        polarity = setup['polarity']

        # When
        promcda = ProMCDA(input_matrix, polarity)
        promcda.normalize(NormalizationFunctions.MINMAX)
        res_simple_promcda = promcda.aggregate(AggregationFunctions.WEIGHTED_SUM)
        simple_mcda_col_names = ['ws-minmax_01']

        # Then
        assert isinstance(res_simple_promcda, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res_simple_promcda.columns.tolist(), list2=simple_mcda_col_names)
        assert res_simple_promcda.shape[0] == input_matrix.shape[0]
        assert res_simple_promcda.shape[1] == len(simple_mcda_col_names)

    def test_aggregate_indicators_sensitivity(self):
        # Given
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        setup = TestMCDA_without_robustness.setup()
        polarity = setup['polarity']
        weights = setup["weights"]

        # When
        promcda = ProMCDA(input_matrix, polarity, weights)
        promcda.normalize()
        res_sensitivity_promcda = promcda.aggregate()

        col_names = [e.value for e in OutputColumnNames4Sensitivity]

        # Then
        assert isinstance(res_sensitivity_promcda, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res_sensitivity_promcda.columns.tolist(), list2=col_names)
        assert res_sensitivity_promcda.shape[0] == input_matrix.shape[0]
        assert res_sensitivity_promcda.shape[1] == len(col_names)


    def test_aggregate_indicators_in_parallel_simple(self):
        # Given
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        setup = TestMCDA_without_robustness.setup_robustness_weights()
        polarity = setup['polarity']
        weights = setup['weights']

        # When
        promcda = ProMCDA(input_matrix, polarity, weights)
        promcda.normalize(NormalizationFunctions.MINMAX)
        res = promcda.aggregate(AggregationFunctions.WEIGHTED_SUM)
        col_names = ['ws-minmax_01']

        # Then
        assert isinstance(res, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res.columns.tolist(), list2=col_names)
        assert res.shape[0] == input_matrix.shape[0]
        assert res.shape[1] == len(col_names)

    def test_aggregate_indicators_in_parallel_sensitivity(self):
        # Given
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        setup = TestMCDA_without_robustness.setup_robustness_weights()
        polarity = setup['polarity']
        weights = setup['weights']

        # When
        promcda = ProMCDA(input_matrix, polarity, weights)
        promcda.normalize()
        res = promcda.aggregate()
        col_names = [e.value for e in OutputColumnNames4Sensitivity]

        # Then
        assert isinstance(res, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res.columns.tolist(), list2=col_names)
        assert res.shape[0] == input_matrix.shape[0]
        assert res.shape[1] == len(col_names)

    def test_estimate_runs_mean_std(self):
        import promcda.utils.utils_for_parallelization as utils_for_parallelization

        # Given
        list_of_df = TestMCDA_without_robustness.get_list_of_df()

        # When
        res = utils_for_parallelization.estimate_runs_mean_std(list_of_df)
        std = {'col1': [0, 0, 0, 0, 0, 0], 'col2': [0, 0, 0, 0, 0, 0], 'col3': [0, 0, 0, 0, 0, 0],
               'col4': [0, 0, 0, 0, 0, 0]}
        df_std = pd.DataFrame(data=std)

        # Then
        assert len(res) == 2
        assert isinstance(res, list)
        assert isinstance(res[0], pd.DataFrame)
        assert res[0].to_numpy().all() == TestMCDA_without_robustness.get_input_matrix().to_numpy().all()
        assert res[1].to_numpy().all() == df_std.to_numpy().all()

if __name__ == '__main__':
    unittest.main()
