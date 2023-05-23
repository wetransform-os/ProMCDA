import unittest
from unittest import TestCase

from mcda.mcda_without_variability import MCDAWithoutVar
from mcda.configuration.config import Config
from mcda.utils import *

class TestMCDA_without_variability(unittest.TestCase):

    @staticmethod
    def get_test_config():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact'],
            "polarity_for_each_indicator": ["-","-","+","+","+","+"],
            "monte_carlo_runs": 0,
            "no_cores": 1,
            "weight_for_each_indicator": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "output_path": "/path/to/output"
        }


    @staticmethod
    def get_input_matrix():
        input_matrix = read_matrix("tests/resources/input_matrix_without_uncert.csv")
        input_matrix_no_alternatives = input_matrix.drop(input_matrix.columns[0], axis=1)

        return input_matrix_no_alternatives


    def test_normalize_indicators(self):
        # Given
        config = TestMCDA_without_variability.get_test_config()
        config = Config(config)
        input_matrix = TestMCDA_without_variability.get_input_matrix()

        # When
        MCDA_no_var = MCDAWithoutVar(config, input_matrix)
        res = MCDA_no_var.normalize_indicators()

        # Then
        assert isinstance(res, dict)
        TestCase.assertIn(self, member='standardized', container=res.keys())
        TestCase.assertIn(self, member='minmax_01', container=res.keys())
        TestCase.assertIn(self, member='minmax_no0', container=res.keys())
        TestCase.assertIn(self, member='target_01', container=res.keys())
        TestCase.assertIn(self, member='target_no0', container=res.keys())
        TestCase.assertIn(self, member='rank', container=res.keys())
        for key in res.keys(): assert (res[key].shape == input_matrix.shape)

    def test_aggregate_indicators(self):
        # Given
        config = TestMCDA_without_variability.get_test_config()
        config = Config(config)
        input_matrix = TestMCDA_without_variability.get_input_matrix()

        # When
        weights = config.weight_for_each_indicator
        MCDA_no_var = MCDAWithoutVar(config, input_matrix)
        normalized_indicators = MCDA_no_var.normalize_indicators()
        res = MCDA_no_var.aggregate_indicators(normalized_indicators, weights)

        col_names = ['ws-stand', 'ws-minmax', 'ws-target', 'ws-rank',
                     'geom-stand', 'geom-minmax', 'geom-target', 'geom-rank',
                     'harm-stand', 'harm-minmax', 'harm-target', 'harm-rank',
                     'min-stand']

        # Then
        assert isinstance(res, pd.DataFrame)
        TestCase.assertListEqual(self, list1=res.columns.tolist(), list2=col_names)
        assert res.shape[0] == input_matrix.shape[0]
        assert res.shape[1] == len(col_names)

if __name__ == '__main__':
    unittest.main()