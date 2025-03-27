import unittest

from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal

from promcda.mcda_functions.normalization import Normalization
from promcda.utils.utils_for_main import *

current_directory = os.path.dirname(os.path.abspath(__file__))
resources_directory = os.path.join(current_directory, '..', 'resources')


class TestNormalization(unittest.TestCase):

    @staticmethod
    def get_input_matrix():
        input_matrix_file_path = os.path.join(resources_directory, 'input_matrix_without_uncert.csv')
        input_matrix = read_matrix(input_matrix_file_path)

        return input_matrix

    @staticmethod
    def get_input_polarities():
        polarities = ("-", "-", "+", "+", "+", "+")  # same as in ProMCDA/configuration_without_robustness.json

        return polarities

    def test_minmax(self):
        # Given
        polarities = TestNormalization.get_input_polarities()
        input_matrix = TestNormalization.get_input_matrix()

        # When
        expected_res01_file_path = os.path.join(resources_directory, 'normalization/res_minmax_01.csv')
        expected_res_without_zero_file_path = os.path.join(resources_directory, "normalization/res_minmax_without_zero.csv")
        expected_res_01 = read_matrix(expected_res01_file_path)
        expected_res_without_zero = read_matrix(expected_res_without_zero_file_path)
        norm = Normalization(input_matrix, polarities)
        res_01 = norm.minmax(feature_range=(0, 1))
        res_01.columns = res_01.columns.astype('str')  # to match the type of columns in the two dfs
        res_without_zero = norm.minmax(feature_range=(0.1, 1))
        res_without_zero.columns = res_without_zero.columns.astype('str')  # to match the type of columns in the two dfs

        # Assure that the indexes are the same
        res_01 = reset_index_if_needed(res_01)
        res_without_zero = reset_index_if_needed(res_without_zero)
        expected_res_01 = reset_index_if_needed(expected_res_01)
        expected_res_without_zero = reset_index_if_needed(expected_res_without_zero)

        # Then
        assert isinstance(res_01, pd.DataFrame)
        assert isinstance(res_without_zero, pd.DataFrame)
        assert_frame_equal(res_01, expected_res_01, check_like=True, check_dtype=False)
        assert_frame_equal(res_without_zero, expected_res_without_zero, check_like=True, check_dtype=False)
        assert_almost_equal(res_01.to_numpy().min(), 0)
        assert_almost_equal(res_01.to_numpy().max(), 1)
        assert_almost_equal(res_without_zero.to_numpy().min(), 0.1)
        assert_almost_equal(res_without_zero.to_numpy().max(), 1)

    def test_target(self):
        # Given
        polarities = TestNormalization.get_input_polarities()
        input_matrix_no_alternatives = TestNormalization.get_input_matrix()

        # When
        res_target_01 = os.path.join(resources_directory, 'normalization/res_target_01.csv')
        expected_res_01 = read_matrix(res_target_01)
        res_target_without_zero = os.path.join(resources_directory, 'normalization/res_target_without_zero.csv')
        expected_res_without_zero = read_matrix(res_target_without_zero)
        norm = Normalization(input_matrix_no_alternatives, polarities)
        res_01 = norm.target(feature_range=(0, 1))
        res_01.columns = res_01.columns.astype('str')  # to match the type of columns in the two dfs
        res_without_zero = norm.target(feature_range=(0.1, 1))
        res_without_zero.columns = res_without_zero.columns.astype('str')  # to match the type of columns in the two dfs

        # Assure that the indexes are the same
        res_01 = reset_index_if_needed(res_01)
        res_without_zero = reset_index_if_needed(res_without_zero)
        expected_res_01 = reset_index_if_needed(expected_res_01)
        expected_res_without_zero = reset_index_if_needed(expected_res_without_zero)

        # Then
        assert isinstance(res_01, pd.DataFrame)
        assert isinstance(res_without_zero, pd.DataFrame)
        assert_frame_equal(res_01, expected_res_01, check_like=True, check_dtype=False)
        assert_frame_equal(res_without_zero, expected_res_without_zero, check_like=True, check_dtype=False)
        assert_almost_equal(res_01.to_numpy().min(), 0)
        assert_almost_equal(res_01.to_numpy().max(), 1)
        assert_almost_equal(res_without_zero.to_numpy().min(), 0.1)
        assert_almost_equal(res_without_zero.to_numpy().max(), 1)

    def test_standardized(self):
        # Given
        polarities = TestNormalization.get_input_polarities()
        input_matrix_no_alternatives = TestNormalization.get_input_matrix()

        # When
        res_standardized_any = os.path.join(resources_directory, 'normalization/res_standardized_any.csv')
        expected_res_any = read_matrix(res_standardized_any)
        res_standardized_without_zero = os.path.join(resources_directory, 'normalization/res_standardized_without_zero.csv')
        expected_res_without_zero = read_matrix(res_standardized_without_zero)
        norm = Normalization(input_matrix_no_alternatives, polarities)
        res_any = norm.standardized(feature_range=('-inf', '+inf'))
        res_without_zero = norm.standardized(feature_range=(0.1, '+inf'))
        res_any.columns = res_any.columns.astype('str')  # to match the type of columns in the two dfs
        res_without_zero.columns = res_without_zero.columns.astype('str')

        # Assure that the indexes are the same
        res_any = reset_index_if_needed(res_any)
        res_without_zero = reset_index_if_needed(res_without_zero)
        expected_res_any = reset_index_if_needed(expected_res_any)
        expected_res_without_zero = reset_index_if_needed(expected_res_without_zero)

        # Then
        assert isinstance(res_any, pd.DataFrame)
        assert isinstance(res_without_zero, pd.DataFrame)
        assert_frame_equal(res_any, expected_res_any, check_like=True, check_dtype=False)
        assert_frame_equal(res_without_zero, expected_res_without_zero, check_like=True, check_dtype=False)
        assert_almost_equal(res_without_zero.to_numpy().min(), 0.1)
        assert_almost_equal(res_any.to_numpy().mean(), 0, decimal=1)
        assert_almost_equal(res_any.to_numpy().std(), 1, decimal=1)

    def test_rank(self):
        # Given
        polarities = TestNormalization.get_input_polarities()
        input_matrix_no_alternatives = TestNormalization.get_input_matrix()
        no_alternatives = input_matrix_no_alternatives.shape[0]

        # When
        res_rank = os.path.join(resources_directory, 'normalization/res_rank.csv')
        expected_res = read_matrix(res_rank)
        norm = Normalization(input_matrix_no_alternatives, polarities)
        res = norm.rank()
        res.columns = res.columns.astype('str')  # to match the type of columns in the two dfs

        # Assure that the indexes are the same
        res = reset_index_if_needed(res)
        expected_res = reset_index_if_needed(expected_res)

        # Then
        assert isinstance(res, pd.DataFrame)
        assert_frame_equal(res, expected_res, check_like=True, check_dtype=False)
        assert_almost_equal(res.to_numpy().min(), 1)
        assert_almost_equal(res.to_numpy().max(), no_alternatives)


if __name__ == '__main__':
    unittest.main()
