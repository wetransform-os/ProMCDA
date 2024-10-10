import unittest
from statistics import mean, stdev

from pandas.testing import assert_frame_equal

from mcda.models.mcda_without_robustness import *
from mcda.utils.utils_for_parallelization import *


class TestUtilsForParallelization(unittest.TestCase):

    @staticmethod
    def get_test_config():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ["+", "+", "+", "+", "+"],
            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "yes",
                "on_single_weights": "no",
                "on_all_weights": "no",
                "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5],
                "on_indicators": "yes"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 10000,
                "num_cores": 1,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['exact', 'uniform', 'normal', 'exact', 'uniform']},
            "output_directory_path": "/path/to/output"
        }

    @staticmethod
    def get_list_of_dfs():
        data1 = [10, 2, 30]
        data2 = [20, 2, 3]
        data3 = [30, 2, 0.3]

        df1 = pd.DataFrame(data1, columns=['numbers'])
        df2 = pd.DataFrame(data2, columns=['numbers'])
        df3 = pd.DataFrame(data3, columns=['numbers'])

        list_of_dfs = [df1, df2, df3]

        return list_of_dfs

    @staticmethod
    def get_list_of_output_dfs():
        average1 = mean([10, 20, 30])
        stand_dev1 = stdev([10, 20, 30])
        average2 = mean([2, 2, 2])
        stand_dev2 = stdev([2, 2, 2])
        average3 = mean([30, 3, 0.3])
        stand_dev3 = stdev([30, 3, 0.3])
        average = pd.DataFrame([average1, average2, average3], columns=['numbers'])
        stand_dev = pd.DataFrame([stand_dev1, stand_dev2, stand_dev3], columns=['numbers'])

        list_of_dfs = [average, stand_dev]

        return list_of_dfs

    @staticmethod
    def get_input_list() -> list[pd.DataFrame]:
        data = {'0': [1, 1, 2, 3], '1': [4, 5, 6, 7], '2': [8, 9, 10, 11], '3': [12, 13, 14, 15], '4': [16, 17, 18, 19]}
        df = pd.DataFrame(data=data)
        out_list = [df, df, df]

        return out_list

    @staticmethod
    def get_output_dict() -> list[dict]:
        data = {'0': [1, 1, 2, 3], '1': [4, 5, 6, 7], '2': [8, 9, 10, 11], '3': [12, 13, 14, 15], '4': [16, 17, 18, 19]}
        df = pd.DataFrame(data=data)
        config = TestUtilsForParallelization.get_test_config()
        config = Config(config)
        mcda_no_var = MCDAWithoutRobustness(config, df)
        df_norm = mcda_no_var.normalize_indicators()

        out_list = [df_norm, df_norm, df_norm]

        return out_list

    def test_estimate_runs_mean_std(self):
        # Given
        input = TestUtilsForParallelization.get_list_of_dfs()
        output = TestUtilsForParallelization.get_list_of_output_dfs()

        # When
        res = estimate_runs_mean_std(input)

        # Then
        isinstance(res, list)
        for i in range(2): isinstance(res[i], pd.DataFrame)
        assert_frame_equal(res[0], output[0])
        assert_frame_equal(res[1], output[1])

    def test_parallelize_normalization(self):
        # Given
        list_input_matrices = TestUtilsForParallelization.get_input_list()
        list_output_dictionaries = TestUtilsForParallelization.get_output_dict()
        polarities = ["+", "+", "+", "+", "+"]

        # When
        res = parallelize_normalization(list_input_matrices, polarities)

        # Then
        isinstance(res, list)
        for i in range(3):
            assert isinstance(res[i], dict)
            assert set(res[i].keys()) == set(list_output_dictionaries[i].keys())
            for key in res[i].keys():
                df1 = res[i][key]
                df2 = list_output_dictionaries[i][key]
                assert_frame_equal(df1, df2, check_dtype=False)
