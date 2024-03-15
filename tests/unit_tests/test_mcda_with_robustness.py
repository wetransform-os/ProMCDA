import pandas as pd
import numpy as np
import unittest

from mcda.mcda_with_robustness import MCDAWithRobustness
from mcda.configuration.config import Config


class TestMCDA_with_robustness(unittest.TestCase):

    @staticmethod
    def get_test_config():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ["-", "-", "+", "+"],
            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "yes",
                "on_single_weights": "no",
                "on_all_weights": "no",
                "given_weights": [0.5, 0.5, 0.5, 0.5],
                "on_indicators": "yes"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 100,
                "num_cores": 1,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['exact', 'uniform', 'normal', 'poisson']},
            "output_path": "/path/to/output"
        }

    @staticmethod
    def get_input_matrix() -> pd.DataFrame:
        data = {'col1': [0, 1, 2, 3], 'col2': [-4, -5, -6, -7], 'col3': [4, 5, 6, 7],
                'col4': [8, 9, 10, 11], 'col5': [0.1, 0.1, 0.1, 0.1], 'col6': [8, 9, 10, 11]}
        df = pd.DataFrame(data=data)

        return df

    @staticmethod
    def get_input_matrix_rescale() -> pd.DataFrame:
        data = {'col1': [0, 1, 2, 3], 'col2': [-4, -5, -6, -7], 'col3': [4, 5, 6, 7],
                'col4': [8, 9, 10, 11], 'col5': [10, 0.1, 0.1, 0.1], 'col6': [8, 9, 10, 11]}
        df = pd.DataFrame(data=data)

        return df

    @staticmethod
    def get_input_list() -> list[pd.DataFrame]:
        data = {'0': [0, 1, 2, 3], '1': [4, 5, 6, 7], '2': [8, 9, 10, 11], '3': [12, 13, 14, 15], '4': [16, 17, 18, 19]}
        df = pd.DataFrame(data=data)
        out_list = [df, df, df]

        return out_list

    @staticmethod
    def get_expected_out_list() -> list[pd.DataFrame]:
        data1 = {'0': [0, 1, 2, 3], '1': [0, 1, 2, 3], '2': [0, 1, 2, 3]}
        data2 = {'0': [4, 5, 6, 7], '1': [4, 5, 6, 7], '2': [4, 5, 6, 7]}
        data3 = {'0': [8, 9, 10, 11], '1': [8, 9, 10, 11], '2': [8, 9, 10, 11]}
        data4 = {'0': [12, 13, 14, 15], '1': [12, 13, 14, 15], '2': [12, 13, 14, 15]}
        data5 = {'0': [16, 17, 18, 19], '1': [16, 17, 18, 19], '2': [16, 17, 18, 19]}
        data = [data1, data2, data3, data4, data5]
        out_list = []
        for i in range(0, 5):
            df_name = pd.DataFrame(data=data[i])
            out_list.append(df_name)

        return out_list

    @staticmethod
    def get_list_random_input_matrices() -> list[pd.DataFrame]:
        np.random.seed(42)
        is_exact_pdf_mask = (1, 0, 0, 0)
        is_poisson_pdf_mask = (0, 0, 0, 1)
        data1 = pd.DataFrame(
            data={'0': [0, 1, 2, 3], '1': [0, 1, 2, 3], '2': [0, 1, 2, 3], '3': [0, 1, 2, 3], '4': [0, 1, 2, 3],
                  '5': [0, 1, 2, 3], '6': [0, 1, 2, 3], '7': [0, 1, 2, 3], '8': [0, 1, 2, 3], '9': [0, 1, 2, 3]})
        low_values = np.array([-4, -5, -6, -7])
        high_values = np.array([4, 5, 6, 7])
        low_values_2d = low_values[:, np.newaxis]  # broadcasting the 1D arrays to a shape of (4, 10)
        high_values_2d = high_values[:, np.newaxis]
        data2 = np.random.uniform(low=low_values_2d, high=high_values_2d, size=(4, 10))
        mean_values = np.array([8, 9, 10, 11])
        std_values = np.array([0.1, 0.1, 0.1, 0.1])
        mean_values_2d = mean_values[:, np.newaxis]
        std_values_2d = std_values[:, np.newaxis]
        data2 = pd.DataFrame(data2)
        data3 = np.random.normal(loc=mean_values_2d, scale=std_values_2d, size=(4, 10))
        data3 = pd.DataFrame(data3)
        lambda_values = np.array([8, 9, 10, 11])
        lambda_values_2d = lambda_values[:, np.newaxis]
        data4 = np.random.poisson(lam=lambda_values_2d, size=(4, 10))
        data4 = pd.DataFrame(data4)
        out_list = [data1, data2, data3, data4]

        input_matrix = TestMCDA_with_robustness.get_input_matrix()
        config = TestMCDA_with_robustness.get_test_config()
        config = Config(config)
        mcda_with_robustness = MCDAWithRobustness(config, input_matrix, is_exact_pdf_mask, is_poisson_pdf_mask)
        output_list = mcda_with_robustness.convert_list(out_list)

        return output_list

    def test_repeat_series_to_create_df(self):
        # Given
        input_matrix = self.get_input_matrix()
        input_series = input_matrix.iloc[:, 0]
        config = TestMCDA_with_robustness.get_test_config()
        config = Config(config)
        num_runs = 10
        exp_matrix = pd.DataFrame(
            data={'0': [0, 1, 2, 3], '1': [0, 1, 2, 3], '2': [0, 1, 2, 3], '3': [0, 1, 2, 3], '4': [0, 1, 2, 3],
                  '5': [0, 1, 2, 3], '6': [0, 1, 2, 3], '7': [0, 1, 2, 3], '8': [0, 1, 2, 3], '9': [0, 1, 2, 3]})

        # When
        mcda_with_robustness = MCDAWithRobustness(config, input_matrix)
        output_matrix = mcda_with_robustness.repeat_series_to_create_df(input_series, num_runs)

        # Then
        assert isinstance(output_matrix, pd.DataFrame)
        assert exp_matrix.values.all() == output_matrix.values.all()
        assert exp_matrix.values.shape == output_matrix.shape

    def test_convert_list(self):
        # Given
        input_matrix = self.get_input_matrix()
        config = TestMCDA_with_robustness.get_test_config()
        input_list = TestMCDA_with_robustness.get_input_list()
        expected_output_list = TestMCDA_with_robustness.get_expected_out_list()
        is_exact_pdf_mask = (1, 0, 0, 0)
        is_poisson_pdf_mask = (0, 0, 0, 1)

        # When
        config = Config(config)
        mcda_with_robustness = MCDAWithRobustness(config, input_matrix, is_exact_pdf_mask, is_poisson_pdf_mask)
        output_list = mcda_with_robustness.convert_list(input_list)

        # Then
        assert isinstance(output_list, list)
        assert len(output_list) == 5
        for df1, df2 in zip(output_list, expected_output_list):
            assert df1.shape == (4, 3)
            assert df1.values.all() == df2.values.all()

    def test_create_n_randomly_sampled_matrices(self):
        # Given
        input_matrix = self.get_input_matrix()
        input_matrix_rescale = self.get_input_matrix_rescale()
        config = TestMCDA_with_robustness.get_test_config()
        is_exact_pdf_mask = (1, 0, 0, 0)
        is_poisson_pdf_mask = (0, 0, 0, 1)

        # When
        config = Config(config)
        mcda_with_robustness = MCDAWithRobustness(config, input_matrix,
                                                  is_exact_pdf_mask, is_poisson_pdf_mask)
        n_random_matrices = mcda_with_robustness.create_n_randomly_sampled_matrices()
        mcda_with_robustness_rescale = MCDAWithRobustness(config, input_matrix_rescale,
                                                          is_exact_pdf_mask, is_poisson_pdf_mask)
        n_random_matrices_rescale = mcda_with_robustness_rescale.create_n_randomly_sampled_matrices()

        exp_n_random_matrices = TestMCDA_with_robustness.get_list_random_input_matrices()

        # Then
        assert isinstance(n_random_matrices, list)
        assert len(n_random_matrices) == config.monte_carlo_sampling["monte_carlo_runs"]
        for df1, df2 in zip(n_random_matrices, exp_n_random_matrices):
            assert df1.shape == (4, 4)
            assert df1.values.all() == df2.values.all()
        for df in n_random_matrices_rescale:
            assert (df >= 0).all().all()
