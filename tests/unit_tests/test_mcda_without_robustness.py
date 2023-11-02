import unittest
from pathlib import Path
from unittest import TestCase

from mcda.mcda_without_robustness import MCDAWithoutRobustness
from mcda.configuration.config import Config
from mcda.utils import *
from mcda.utils_for_parallelization import *
from mcda.utility_functions.aggregation import Aggregation

class TestMCDA_without_robustness(unittest.TestCase):

    @staticmethod
    def get_test_config():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ['-','-','+','+','+','+'],
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
                "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact']},
            "output_path": "/path/to/output"
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
                "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact']},
            "output_path": "/path/to/output"
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
                "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact']},
            "output_path": "/path/to/output"
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
                "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact']},
            "output_path": "/path/to/output"
        }

    @staticmethod
    def get_input_matrix():
        test_data_directory = Path(__file__).resolve().parent.parent / "resources"
        file_path = test_data_directory / "input_matrix_without_uncert.csv"

        input_matrix = read_matrix(file_path)
        input_matrix_no_alternatives = input_matrix.drop(input_matrix.columns[0], axis=1)

        return input_matrix_no_alternatives

    @staticmethod
    def get_list_of_df():
        list_df = [TestMCDA_without_robustness.get_input_matrix(), TestMCDA_without_robustness.get_input_matrix()]

        return list_df

    def test_normalize_indicators(self):
        # Given
        config_general = TestMCDA_without_robustness.get_test_config()
        config_general = Config(config_general)
        input_matrix = TestMCDA_without_robustness.get_input_matrix()

        config_simple_mcda = TestMCDA_without_robustness.get_test_config_simple_mcda()
        config_simple_mcda = Config(config_simple_mcda)

        # When
        MCDA_no_uncert_general = MCDAWithoutRobustness(config_general, input_matrix)
        res_general = MCDA_no_uncert_general.normalize_indicators()

        MCDA_no_uncert_simple_mcda = MCDAWithoutRobustness(config_simple_mcda, input_matrix)
        res_simple_mcda = MCDA_no_uncert_simple_mcda.normalize_indicators('minmax')

        # Then
        assert isinstance(res_general, dict)
        TestCase.assertIn(self, member='standardized_any', container=res_general.keys())
        TestCase.assertIn(self, member='standardized_no0', container=res_general.keys())
        TestCase.assertIn(self, member='minmax_01', container=res_general.keys())
        TestCase.assertIn(self, member='minmax_no0', container=res_general.keys())
        TestCase.assertIn(self, member='target_01', container=res_general.keys())
        TestCase.assertIn(self, member='target_no0', container=res_general.keys())
        TestCase.assertIn(self, member='rank', container=res_general.keys())
        for key in res_general.keys(): assert (res_general[key].shape == input_matrix.shape)

        assert isinstance(res_simple_mcda, dict)
        TestCase.assertIn(self, member='minmax_01', container=res_simple_mcda.keys())
        TestCase.assertIn(self, member='minmax_no0', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='standardized_any', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='standardized_no0', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='target_01', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='target_no0', container=res_simple_mcda.keys())
        TestCase.assertNotIn(self, member='rank', container=res_simple_mcda.keys())
        for key in res_simple_mcda.keys(): assert (res_simple_mcda[key].shape == input_matrix.shape)

    def test_aggregate_indicators(self):
        # Given
        config = TestMCDA_without_robustness.get_test_config()
        config = Config(config)
        input_matrix = TestMCDA_without_robustness.get_input_matrix()

        config_simple_mcda = TestMCDA_without_robustness.get_test_config_simple_mcda()
        config_simple_mcda = Config(config_simple_mcda)

        # When
        weights = config.robustness["given_weights"]

        MCDA_no_uncert = MCDAWithoutRobustness(config, input_matrix)
        normalized_indicators = MCDA_no_uncert.normalize_indicators()
        MCDA_no_uncert_simple_mcda = MCDAWithoutRobustness(config_simple_mcda, input_matrix)
        normalized_indicators_simple_mcda = MCDA_no_uncert_simple_mcda.normalize_indicators(config_simple_mcda.sensitivity['normalization'])

        res = MCDA_no_uncert.aggregate_indicators(normalized_indicators, weights)
        res_simple_mcda = MCDA_no_uncert_simple_mcda.aggregate_indicators(normalized_indicators_simple_mcda, weights, config_simple_mcda.sensitivity['aggregation'])

        col_names = ['ws-minmax_01', 'ws-target_01', 'ws-standardized_any', 'ws-rank',
                    'geom-minmax_no0', 'geom-target_no0', 'geom-standardized_no0', 'geom-rank',
                    'harm-minmax_no0', 'harm-target_no0', 'harm-standardized_no0', 'harm-rank',
                    'min-standardized_any']

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
            config = Config(config)
            input_matrix = TestMCDA_without_robustness.get_input_matrix()
            weights = config.robustness["given_weights"]
            agg =  Aggregation(weights)

            config_randomness_simple_mcda = TestMCDA_without_robustness.get_test_config_randomness_simple_mcda()
            config_randomness_simple_mcda = Config(config_randomness_simple_mcda)

            # When
            MCDA_no_uncert = MCDAWithoutRobustness(config, input_matrix)
            normalized_indicators = MCDA_no_uncert.normalize_indicators()
            res = aggregate_indicators_in_parallel(agg, normalized_indicators)

            MCDA_no_uncert_simple_mcda = MCDAWithoutRobustness(config_randomness_simple_mcda, input_matrix)
            normalized_indicators = MCDA_no_uncert_simple_mcda.normalize_indicators(config_randomness_simple_mcda.sensitivity['normalization'])
            res_simple_mcda = aggregate_indicators_in_parallel(agg, normalized_indicators, config_randomness_simple_mcda.sensitivity['aggregation'])

            col_names = ['ws-minmax_01', 'ws-target_01', 'ws-standardized_any', 'ws-rank',
                         'geom-minmax_no0', 'geom-target_no0', 'geom-standardized_no0', 'geom-rank',
                         'harm-minmax_no0', 'harm-target_no0', 'harm-standardized_no0', 'harm-rank',
                         'min-standardized_any']

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
        res = estimate_runs_mean_std(list_of_df)
        std = {'col1': [0,0,0,0,0,0], 'col2': [0,0,0,0,0,0], 'col3': [0,0,0,0,0,0], 'col4': [0,0,0,0,0,0]}
        df_std = pd.DataFrame(data=std)

        # Then
        assert len(res) ==2
        assert isinstance(res, list)
        assert isinstance(res[0], pd.DataFrame)
        assert res[0].to_numpy().all() == TestMCDA_without_robustness.get_input_matrix().to_numpy().all()
        assert res[1].to_numpy().all() == df_std.to_numpy().all()


if __name__ == '__main__':
    unittest.main()
