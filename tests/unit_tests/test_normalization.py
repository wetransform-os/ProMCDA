import unittest

from pandas.testing import assert_frame_equal

from mcda.utils import *
from mcda.utility_functions.normalization import Normalization

class TestNormalization(unittest.TestCase):

    @staticmethod
    def get_input_matrix():
        input_matrix = read_matrix("tests/resources/input_matrix_without_uncert.csv")
        input_matrix_no_alternatives = input_matrix.drop(input_matrix.columns[0],axis=1) # drop first column with alternatives
                                                                                         # in the code this is happening in mcda_run
        return input_matrix_no_alternatives

    @staticmethod
    def get_input_polarities():
        polarities = ("-","-","+","+","+","+") # same as in MCDTool/configuration_without_uncertainty.json

        return polarities

    def test_minmax(self):
        # Given
        polarities = TestNormalization.get_input_polarities()
        input_matrix_no_alternatives = TestNormalization.get_input_matrix()

        # When
        expected_res_01 = read_matrix('tests/resources/normalization/res_minmax_01.csv')
        expected_res_no0 = read_matrix('tests/resources/normalization/res_minmax_no0.csv')
        norm = Normalization(input_matrix_no_alternatives, polarities)
        res_01 = norm.minmax(feature_range=(0, 1))
        res_01.columns = res_01.columns.astype('str') # to match the type of columns in the two dfs
        res_no0 = norm.minmax(feature_range=(0.1, 1))
        res_no0.columns = res_no0.columns.astype('str') # to match the type of columns in the two dfs

        # Then
        assert isinstance(res_01, pd.DataFrame)
        assert isinstance(res_no0, pd.DataFrame)
        assert_frame_equal(res_01, expected_res_01, check_like=True)
        assert_frame_equal(res_no0, expected_res_no0, check_like=True)


    def test_target(self):
        # Given
        polarities = TestNormalization.get_input_polarities()
        input_matrix_no_alternatives = TestNormalization.get_input_matrix()

        # When
        expected_res_01 = read_matrix('tests/resources/normalization/res_target_01.csv')
        expected_res_no0 = read_matrix('tests/resources/normalization/res_target_no0.csv')
        norm = Normalization(input_matrix_no_alternatives,polarities)
        res_01 = norm.target(feature_range=(0, 1))
        res_01.columns = res_01.columns.astype('str') # to match the type of columns in the two dfs
        res_no0 = norm.target(feature_range=(0.1, 1))
        res_no0.columns = res_no0.columns.astype('str') # to match the type of columns in the two dfs

        # Then
        assert isinstance(res_01, pd.DataFrame)
        assert isinstance(res_no0, pd.DataFrame)
        assert_frame_equal(res_01, expected_res_01, check_like=True)
        assert_frame_equal(res_no0, expected_res_no0, check_like=True)


    def test_standardized(self):
        # Given
        polarities = TestNormalization.get_input_polarities()
        input_matrix_no_alternatives = TestNormalization.get_input_matrix()

        # When
        expected_res = read_matrix('tests/resources/normalization/res_standardized.csv')
        norm = Normalization(input_matrix_no_alternatives,polarities)
        res = norm.standardized()
        res.columns = res.columns.astype('str') # to match the type of columns in the two dfs

        # Then
        assert isinstance(res, pd.DataFrame)
        assert_frame_equal(res, expected_res, check_like=True)


    def test_rank(self):
        # Given
        polarities = TestNormalization.get_input_polarities()
        input_matrix_no_alternatives = TestNormalization.get_input_matrix()

        # When
        expected_res = read_matrix('tests/resources/normalization/res_rank.csv')
        norm = Normalization(input_matrix_no_alternatives,polarities)
        res = norm.rank()
        res.columns = res.columns.astype('str') # to match the type of columns in the two dfs

        # Then
        assert isinstance(res, pd.DataFrame)
        assert_frame_equal(res, expected_res, check_like=True)

if __name__ == '__main__':
    unittest.main()