import os
import tempfile
import unittest
import pandas as pd
from unittest import TestCase

from ProMCDA.mcda import mcda_run
from ProMCDA.mcda.mcda_without_robustness import MCDAWithoutRobustness
from ProMCDA.mcda.configuration.config import Config
from ProMCDA.mcda.mcda_functions.aggregation import Aggregation
import ProMCDA.mcda.utils.utils_for_main as utils_for_main
import ProMCDA.mcda.utils.utils_for_parallelization as utils_for_parallelization
from ProMCDA.mcda.models.configuration import Configuration

current_directory = os.path.dirname(os.path.abspath(__file__))
resources_directory = os.path.join(current_directory, '..', 'resources')


class TestMCDA_without_robustness(unittest.TestCase):

    @staticmethod
    def get_test_config():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ['-', '-', '+', '+', '+', '+'],
            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "no",
                "on_single_weights": "no",
                "on_all_weights": "yes",
                "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "on_indicators": "no"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 10,
                "num_cores": 4,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact']},
            "output_directory_path": "/path/to/output"
        }

    @staticmethod
    def get_test_config_simple_mcda():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ['-', '-', '+', '+', '+', '+'],
            "sensitivity": {
                "sensitivity_on": "no",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "no",
                "on_single_weights": "no",
                "on_all_weights": "yes",
                "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "on_indicators": "no"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 10,
                "num_cores": 4,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact']},
            "output_directory_path": "/path/to/output"
        }

    @staticmethod
    def get_test_config_randomness():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ['-', '-', '+', '+', '+', '+'],
            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "yes",
                "on_single_weights": "no",
                "on_all_weights": "yes",
                "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "on_indicators": "no"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 10,
                "num_cores": 4,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact']},
            "output_directory_path": "/path/to/output"
        }

    @staticmethod
    def get_test_config_randomness_simple_mcda():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ['-', '-', '+', '+', '+', '+'],
            "sensitivity": {
                "sensitivity_on": "no",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "yes",
                "on_single_weights": "no",
                "on_all_weights": "yes",
                "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "on_indicators": "no"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 10,
                "num_cores": 4,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact']},
            "output_directory_path": "/path/to/output"
        }

    @staticmethod
    def get_input_matrix():
        input_matrix_file_path = os.path.join(resources_directory, 'input_matrix_without_uncert.csv')
        input_matrix = utils_for_main.read_matrix(input_matrix_file_path)

        return input_matrix

    @staticmethod
    def get_list_of_df():
        list_df = [TestMCDA_without_robustness.get_input_matrix(), TestMCDA_without_robustness.get_input_matrix()]

        return list_df

    def test_normalize_indicators(self):
        # Given
        config_general = TestMCDA_without_robustness.get_test_config()
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        temp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            temp_path = tmp_file.name

            # Step 2: Store the DataFrame to the temporary file
            input_matrix.to_csv(temp_path, index=True, columns=input_matrix.columns)
        config_general["input_matrix_path"] = temp_path
        config_general = Configuration.from_dict(mcda_run.config_dict_to_configuration_model(config_general))

        config_simple_mcda = TestMCDA_without_robustness.get_test_config_simple_mcda()
        config_simple_mcda["input_matrix_path"] = temp_path
        config_simple_mcda = Configuration.from_dict(mcda_run.config_dict_to_configuration_model(config_simple_mcda))

        # When
        mcda_no_uncert_general = MCDAWithoutRobustness(config_general, input_matrix)
        res_general = mcda_no_uncert_general.normalize_indicators()

        mcda_no_uncert_simple_mcda = MCDAWithoutRobustness(config_simple_mcda, input_matrix)
        res_simple_mcda = mcda_no_uncert_simple_mcda.normalize_indicators('minmax')

        # Then
        assert isinstance(res_general, dict)
        TestCase.assertIn(self, member='standardized_any', container=res_general.keys())
        TestCase.assertIn(self, member='standardized_without_zero', container=res_general.keys())
        TestCase.assertIn(self, member='minmax_01', container=res_general.keys())
        TestCase.assertIn(self, member='minmax_without_zero', container=res_general.keys())
        TestCase.assertIn(self, member='target_01', container=res_general.keys())
        TestCase.assertIn(self, member='target_without_zero', container=res_general.keys())
        TestCase.assertIn(self, member='rank', container=res_general.keys())
        for key in res_general.keys():
            assert (res_general[key].shape == input_matrix.shape)

        assert isinstance(res_simple_mcda, dict)
        TestCase.assertIn(self, member='minmax_01', container=res_simple_mcda.keys())
        TestCase.assertIn(self, member='minmax_without_zero', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='standardized_any', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='standardized_without_zero', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='target_01', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='target_without_zero', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='rank', container=res_simple_mcda.keys())
        for key in res_simple_mcda.keys():
            assert (res_simple_mcda[key].shape == input_matrix.shape)

    def test_aggregate_indicators(self):
        # Given
        config = TestMCDA_without_robustness.get_test_config()
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        temp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            temp_path = tmp_file.name

            # Step 2: Store the DataFrame to the temporary file
            input_matrix.to_csv(temp_path, index=True, columns=input_matrix.columns)
        config["input_matrix_path"] = temp_path
        config = Configuration.from_dict(mcda_run.config_dict_to_configuration_model(config))
        config_simple_mcda = TestMCDA_without_robustness.get_test_config_simple_mcda()
        config_simple_mcda["input_matrix_path"] = temp_path
        config_simple_mcda = Configuration.from_dict(mcda_run.config_dict_to_configuration_model(config_simple_mcda))

        # When
        weights = config.robustness.given_weights

        mcda_no_uncert = MCDAWithoutRobustness(config, input_matrix)
        normalized_indicators = mcda_no_uncert.normalize_indicators()
        mcda_no_uncert_simple_mcdaa = MCDAWithoutRobustness(config_simple_mcda, input_matrix)
        normalized_indicators_simple_mcda = mcda_no_uncert_simple_mcdaa.normalize_indicators(
            config_simple_mcda.sensitivity.normalization)

        res = mcda_no_uncert.aggregate_indicators(normalized_indicators, weights)
        res_simple_mcda = mcda_no_uncert_simple_mcdaa.aggregate_indicators(normalized_indicators_simple_mcda, weights,
                                                                           config_simple_mcda.sensitivity.aggregation)

        col_names = ['ws-minmax_01', 'ws-target_01', 'ws-standardized_any', 'ws-rank',
                     'geom-minmax_without_zero', 'geom-target_without_zero', 'geom-standardized_without_zero',
                     'geom-rank', 'harm-minmax_without_zero', 'harm-target_without_zero',
                     'harm-standardized_without_zero', 'harm-rank', 'min-standardized_any']

        simple_mcda_col_names = ['ws-minmax_01']

        # Then
        assert isinstance(res, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res.columns.tolist(), list2=col_names)
        assert res.shape[0] == input_matrix.shape[0]
        assert res.shape[1] == len(col_names)

        assert isinstance(res_simple_mcda, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res_simple_mcda.columns.tolist(), list2=simple_mcda_col_names)
        assert res_simple_mcda.shape[0] == input_matrix.shape[0]
        assert res_simple_mcda.shape[1] == len(simple_mcda_col_names)

    def test_aggregate_indicators_in_parallel(self):
        # Given
        config = TestMCDA_without_robustness.get_test_config_randomness()
        input_matrix = TestMCDA_without_robustness.get_input_matrix()
        temp_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            temp_path = tmp_file.name

            # Step 2: Store the DataFrame to the temporary file
            input_matrix.to_csv(temp_path, index=True, columns=input_matrix.columns)

        config["input_matrix_path"] = temp_path
        config_based_on_model = Configuration.from_dict(mcda_run.config_dict_to_configuration_model(config))
        weights = config_based_on_model.robustness.given_weights
        agg = Aggregation(weights)

        config_randomness_simple_mcda = TestMCDA_without_robustness.get_test_config_randomness_simple_mcda()
        config_randomness_simple_mcda["input_matrix_path"] = temp_path
        config_randomness_simple_mcda = (
            Configuration.from_dict(mcda_run.config_dict_to_configuration_model(config_randomness_simple_mcda)))

        # When
        mcda_no_uncert = MCDAWithoutRobustness(config_based_on_model, input_matrix)
        normalized_indicators = mcda_no_uncert.normalize_indicators()
        res = utils_for_parallelization.aggregate_indicators_in_parallel(agg, normalized_indicators)

        mcda_no_uncert_simple_mcda = MCDAWithoutRobustness(config_based_on_model, input_matrix)
        normalized_indicators = mcda_no_uncert_simple_mcda.normalize_indicators(
            config_randomness_simple_mcda.sensitivity.normalization)
        res_simple_mcda = utils_for_parallelization.aggregate_indicators_in_parallel(agg, normalized_indicators,
            config_randomness_simple_mcda.sensitivity.aggregation)

        col_names = ['ws-minmax_01', 'ws-target_01', 'ws-standardized_any', 'ws-rank',
                     'geom-minmax_without_zero', 'geom-target_without_zero', 'geom-standardized_without_zero',
                     'geom-rank', 'harm-minmax_without_zero', 'harm-target_without_zero',
                     'harm-standardized_without_zero', 'harm-rank', 'min-standardized_any']

        simple_mcda_col_names = ['ws-minmax_01']

        # Then
        assert isinstance(res, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res.columns.tolist(), list2=col_names)
        assert res.shape[0] == input_matrix.shape[0]
        assert res.shape[1] == len(col_names)

        assert isinstance(res_simple_mcda, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res_simple_mcda.columns.tolist(), list2=simple_mcda_col_names)
        assert res_simple_mcda.shape[0] == input_matrix.shape[0]
        assert res_simple_mcda.shape[1] == len(simple_mcda_col_names)

    def test_estimate_runs_mean_std(self):
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
