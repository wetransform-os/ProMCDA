import unittest
from unittest import TestCase
from unittest.mock import patch

from promcda.utils.utils_for_main import *
from promcda.utils.utils_for_main import _check_and_rescale_negative_indicators


class TestUtils(unittest.TestCase):

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
                "monte_carlo_runs": 10000,
                "num_cores": 1,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['exact', 'uniform', 'normal', 'poisson']},
            "output_directory_path": "/path/to/output"
        }

    @staticmethod
    def get_input_matrix_1() -> pd.DataFrame:
        data = {'ind1': [1, 2, 3], 'ind2_min': [1, 2, 3], 'ind2_max': [5, 6, 7],
                'ind3': [9, 10, 11], 'std3': [0.1, 0.1, 12], 'ind4_rate': [9, 10, 11]}
        df = pd.DataFrame(data=data)

        return df

    @staticmethod
    def get_input_matrix_2() -> pd.DataFrame:
        data = {'ind1': [1, 2, 3], 'ind2_min': [-5, -6, -7], 'ind2_max': [5, 6, 7],
                'ind3': [9, 10, 11], 'std3': [0.1, 0.1, 11], 'ind4_rate': [9, 10, 11]}
        df = pd.DataFrame(data=data)

        return df

    @staticmethod
    def get_input_matrix_3() -> pd.DataFrame:
        data = {'ind1': [1, 2, 3], 'ind2_min': [6, -6, -7], 'ind2_max': [5, 6, 7],
                'ind3': [9, 10, 11], 'std3': [0.1, 0.1, 11], 'ind4_rate': [9, 10, 11]}
        df = pd.DataFrame(data=data)

        return df

    @staticmethod
    def get_input_matrix_alternatives() -> pd.DataFrame:
        data = {'Anyname': [1,2,3],'ind1': [1, 2, 3], 'ind2_min': [6, -6, -7], 'ind2_max': [5, 6, 7],
                'ind3': [9, 10, 11], 'std3': [0.1, 0.1, 11], 'ind4_rate': [9, 10, 11]}
        df = pd.DataFrame(data=data)

        return df

    @staticmethod
    def get_list_pdf() -> list:
        list_pdf = ['exact', 'lnorm', 'norm', 'poisson', 'exact']
        return list_pdf

    @staticmethod
    def get_input_matrix_negative() -> pd.DataFrame:
        data = {'ind1': [1, -2, 3], 'ind2': [-5, -6, -7], 'ind3': [9, 10, -11], 'ind4': [9, 10, -11]}
        df = pd.DataFrame(data=data)

        return df

    def test_randomly_sample_ix_weight(self):
        # Given
        num_weights = 4
        index = 1
        num_runs = 10

        # When
        out = randomly_sample_ix_weight(num_weights, index, num_runs)

        # Then
        isinstance(out, list)
        assert len(out) == num_runs
        assert [len(out[i]) == num_weights for i in range(num_weights)]
        for j in range(num_runs):
            for i in range(num_weights):
                if i != index:
                    assert out[j][i] == 1
                else:
                    assert out[j][i] != 1

    def test_randomly_sample_all_weights(self):
        # Given
        num_weights = 4
        num_runs = 10

        # When
        out = randomly_sample_all_weights(num_weights, num_runs)

        # Then
        isinstance(out, list)
        assert len(out) == num_runs
        assert [len(out[i]) == num_weights for i in range(num_weights)]
        for j in range(num_runs):
            assert (elem != out[0] for elem in out)  # check all items are different
            for i in range(num_weights):
                assert out[j][i] <= 1
                assert out[j][i] >= 0

    def test_check_norm_sum_weights(self):
        # Given
        weights = [1, 2, 3, 4, 5]

        # When
        out = check_norm_sum_weights(weights)

        # Then
        assert len(out) == len(weights)
        assert sum(out) == 1.0

    def test_pop_indexed_elements(self):
        # Given
        indexes = [0, 2, 4]
        in_list = [1, 2, 3, 4, 5, 6]

        # When
        out_list = pop_indexed_elements(indexes, in_list)
        expected_list = [2, 4, 6]

        # Then
        isinstance(out_list, list)
        TestCase.assertListEqual(self, out_list, expected_list)

    def test_check_parameters_pdf(self):
        # Given
        input_matrix_1 = TestUtils.get_input_matrix_1()
        input_matrix_2 = TestUtils.get_input_matrix_2()
        input_matrix_3 = TestUtils.get_input_matrix_3()
        config = TestUtils.get_test_config()

        # When
        are_parameters_correct_1 = check_parameters_pdf(input_matrix_1, config, True)
        are_parameters_correct_2 = check_parameters_pdf(input_matrix_2, config, True)
        are_parameters_correct_3 = check_parameters_pdf(input_matrix_3, config, True)

        # Then
        isinstance(are_parameters_correct_1, list)
        self.assertFalse(all(are_parameters_correct_1))
        self.assertEqual(are_parameters_correct_1[2], False)

        isinstance(are_parameters_correct_2, list)
        self.assertTrue(all(are_parameters_correct_2))

        isinstance(are_parameters_correct_3, list)
        self.assertFalse(all(are_parameters_correct_3))
        self.assertEqual(are_parameters_correct_3[1], False)

    def test_check_and_rescale_negative_indicators(self):
        # Given
        input_matrix_negative = TestUtils.get_input_matrix_negative()
        input_matrix_positive = TestUtils.get_input_matrix_1()

        # When
        rescaled_matrix = _check_and_rescale_negative_indicators(input_matrix_negative)
        non_rescaled_matrix = _check_and_rescale_negative_indicators(input_matrix_positive)

        # Then
        isinstance(rescaled_matrix, pd.DataFrame)
        assert (rescaled_matrix >= 0).all().all()
        isinstance(non_rescaled_matrix, pd.DataFrame)
        assert non_rescaled_matrix.equals(input_matrix_positive)

    def test_check_if_pdf_is_exact(self):
        # Given
        list_pdf = TestUtils.get_list_pdf()

        # When
        output_mask = check_if_pdf_is_exact(list_pdf)
        expected_mask = [1, 0, 0, 0, 1]

        # Then
        isinstance(output_mask, list)
        assert (len(output_mask) == len(expected_mask))
        self.assertListEqual(output_mask, expected_mask)


    def test_read_matrix(self):
        # Given
        input_matrix = TestUtils.get_input_matrix_alternatives()

        # When
        with patch('builtins.open') as mocked_open, \
                patch('pandas.read_csv', return_value=input_matrix) as mocked_read_csv:
            output = read_matrix('dummy_path')
            index_column_name = output.index.name
            index_column_values = output.index.tolist()

        # Then
        self.assertIsInstance(output, pd.DataFrame)
        assert index_column_name == "Anyname"
        isinstance(index_column_name, str)
        assert index_column_values == [1, 2, 3]
        isinstance(index_column_values, list)
