import unittest

from mcda.utils_for_parallelization import *
from pandas.testing import assert_frame_equal
from statistics import mean, stdev



class TestUtilsForParallelization(unittest.TestCase):

    @staticmethod
    def get_list_of_dfs():
        data1 = [10, 2, 30]
        data2 = [20, 2, 3]
        data3 = [30, 2, 0.3]

        df1 = pd.DataFrame(data1, columns=['numbers'])
        df2 = pd.DataFrame(data2, columns=['numbers'])
        df3 = pd.DataFrame(data3, columns=['numbers'])

        list_of_dfs = [df1,df2,df3]

        return list_of_dfs

    @staticmethod
    def get_list_of_output_dfs():
        average1 = mean([10,20,30])
        stand_dev1 = stdev([10,20,30])
        average2 = mean([2, 2, 2])
        stand_dev2 = stdev([2, 2, 2])
        average3 = mean([30, 3, 0.3])
        stand_dev3 = stdev([30, 3, 0.3])
        average = pd.DataFrame([average1,average2,average3],columns=['numbers'])
        stand_dev = pd.DataFrame([stand_dev1,stand_dev2,stand_dev3],columns=['numbers'])

        list_of_dfs = [average, stand_dev]

        return list_of_dfs

    def test_estimate_runs_mean_std(self):
        # Given
        input = TestUtilsForParallelization.get_list_of_dfs()
        output = TestUtilsForParallelization.get_list_of_output_dfs()

        # When
        res = estimate_runs_mean_std(input)

        # Then
        isinstance(res, list)
        for i in range(2): isinstance(res[i],pd.DataFrame)
        assert_frame_equal(res[0],output[0])
        assert_frame_equal(res[1],output[1])
