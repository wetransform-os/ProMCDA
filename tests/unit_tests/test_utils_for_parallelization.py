import unittest
from statistics import mean, stdev
from unittest.mock import MagicMock

from pandas.testing import assert_frame_equal

from promcda.models import ProMCDA
from promcda.models.mcda_without_robustness import *
from promcda.utils.utils_for_parallelization import *


class TestUtilsForParallelization(unittest.TestCase):

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
        out_list = [df]

        return out_list

    def setUp(self):
        self.norm_mock = MagicMock(spec=Normalization)
        self.norm_mock.minmax.return_value = 'minmax_output'
        self.norm_mock.target.return_value = 'target_output'
        self.norm_mock.standardized.return_value = 'standardized_output'
        self.norm_mock.rank.return_value = 'rank_output'

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


    def test_normalize_with_minmax(self):
        result = normalize_indicators_in_parallel(self.norm_mock, method='minmax')

        self.assertIn(NormalizationNames4Sensitivity.MINMAX_01.value, result)
        self.assertIn(NormalizationNames4Sensitivity.MINMAX_WITHOUT_ZERO.value, result)

        self.norm_mock.minmax.assert_called_with(feature_range=(0.1, 1))
        self.norm_mock.target.assert_not_called()
        self.norm_mock.standardized.assert_not_called()
        self.norm_mock.rank.assert_not_called()

    def test_normalize_with_target(self):
        result = normalize_indicators_in_parallel(self.norm_mock, method='target')

        self.assertIn(NormalizationNames4Sensitivity.TARGET_01.value, result)
        self.assertIn(NormalizationNames4Sensitivity.TARGET_WITHOUT_ZERO.value, result)

        self.norm_mock.target.assert_called_with(feature_range=(0.1, 1))
        self.norm_mock.minmax.assert_not_called()
        self.norm_mock.standardized.assert_not_called()
        self.norm_mock.rank.assert_not_called()

    def test_normalize_with_standardized(self):
        result = normalize_indicators_in_parallel(self.norm_mock, method='standardized')

        self.assertIn(NormalizationNames4Sensitivity.STANDARDIZED_ANY.value, result)
        self.assertIn(NormalizationNames4Sensitivity.STANDARDIZED_WITHOUT_ZERO.value, result)

        self.norm_mock.standardized.assert_called_with(feature_range=(0.1, '+inf'))
        self.norm_mock.minmax.assert_not_called()
        self.norm_mock.target.assert_not_called()
        self.norm_mock.rank.assert_not_called()

    def test_normalize_with_rank(self):
        result = normalize_indicators_in_parallel(self.norm_mock, method='rank')

        self.assertIn(NormalizationNames4Sensitivity.RANK.value, result)

        self.norm_mock.rank.assert_called_once()
        self.norm_mock.minmax.assert_not_called()
        self.norm_mock.target.assert_not_called()
        self.norm_mock.standardized.assert_not_called()

    def test_normalize_with_invalid_method(self):
        with self.assertRaises(ValueError):
            normalize_indicators_in_parallel(self.norm_mock, method='invalid_method')

    def test_normalize_with_no_method(self):
        result = normalize_indicators_in_parallel(self.norm_mock)

        self.assertIn(NormalizationNames4Sensitivity.MINMAX_01.value, result)
        self.assertIn(NormalizationNames4Sensitivity.MINMAX_WITHOUT_ZERO.value, result)
        self.assertIn(NormalizationNames4Sensitivity.TARGET_01.value, result)
        self.assertIn(NormalizationNames4Sensitivity.TARGET_WITHOUT_ZERO.value, result)
        self.assertIn(NormalizationNames4Sensitivity.STANDARDIZED_ANY.value, result)
        self.assertIn(NormalizationNames4Sensitivity.STANDARDIZED_WITHOUT_ZERO.value, result)
        self.assertIn(NormalizationNames4Sensitivity.RANK.value, result)

        self.norm_mock.minmax.assert_called_with(feature_range=(0.1, 1))
        self.norm_mock.target.assert_called_with(feature_range=(0.1, 1))
        self.norm_mock.standardized.assert_called_with(feature_range=(0.1, '+inf'))
        self.norm_mock.rank.assert_called_once()
