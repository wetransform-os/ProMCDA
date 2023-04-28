import pytest
import unittest
from pandas.testing import assert_frame_equal

from mcda.mcda_run import main
from mcda.utils import *
from mcda.utility_functions.normalization import Normalization

class TestNormalization(unittest.TestCase):

    def test_minmax(self):
        # Given
        polarities = ("-","-","+", "+","+","+") # same as in MCDTool/configuration_without_uncertainty.json
        input_matrix = read_matrix("tests/resources/input_matrix_without_uncert.csv")
        input_matrix_no_alternatives = input_matrix.drop(input_matrix.columns[0],axis=1) # drop first column with alternatives
                                                                                         # in the code this is happening in mcda_run
        # When
        expected_res_01 = read_matrix('tests/resources/normalization/res_minmax_pol_01.csv')
        #expected_res_no0 = read_matrix('tests/resources/normalization/res_minmax_no0.csv')

        norm = Normalization(input_matrix_no_alternatives)
        res_01 = norm.minmax(polarities, feature_range=(0, 1))
        res_01.columns = res_01.columns.astype('str') # to match the type of columns in the two dfs
        res_no0 = norm.minmax(polarities, feature_range=(0.1, 1))
        res_no0.columns = res_no0.columns.astype('str') # to match the type of columns in the two dfs


        # Then
        print(res_01)
        print(expected_res_01)
        assert isinstance(res_01, pd.DataFrame)
        assert isinstance(res_no0, pd.DataFrame)
        assert_frame_equal(res_01, expected_res_01, check_like=True)
        #assert_frame_equal(res_no0, expected_res_no0, check_like=True)

        if __name__ == '__main__':
            unittest.main()