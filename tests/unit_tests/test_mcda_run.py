import unittest
import pytest

from mcda.mcda_run import main

class TestMCDA(unittest.TestCase):

    @staticmethod
    def get_uncorrect_config_1():
        return {
            "input_matrix_path": "tests/resources/input_matrix_without_uncert.csv",
            "marginal_distribution_for_each_indicator": ['exact','exact','exact','exact','exact','exact'],
            "polarity_for_each_indicator": ['+','+','-','+','+','-'],
            "monte_carlo_runs": 10,
            "no_cores": 17,
            "weight_for_each_indicator" : {
                "random_weights": "no",
                "iterative": "no",
                "no_samples": 10000,
                "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                                           },
            "output_path": "/path/to/output"
        }

    @staticmethod
    def get_uncorrect_config_2():
        return {
            "input_matrix_path": "tests/resources/input_matrix_with_uncert.csv",
            "marginal_distribution_for_each_indicator": ["norm", "exact", "beta", "norm", "exact", "beta"],
            "polarity_for_each_indicator": ['+','+','-','+','+','-'],
            "monte_carlo_runs": 0,
            "no_cores": 17,
            "weight_for_each_indicator" : {
                "random_weights": "no",
                "iterative": "no",
                "no_samples": 10000,
                "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                                           },
            "output_path": "/path/to/output"
        }

    @staticmethod
    def get_uncorrect_config_3():
        return {
            "input_matrix_path": "tests/resources/input_matrix_without_uncert_duplicates.csv",
            "marginal_distribution_for_each_indicator": ['exact', 'exact', 'exact', 'exact', 'exact', 'exact'],
            "polarity_for_each_indicator": ['+', '+', '-', '+', '+', '-'],
            "monte_carlo_runs": 0,
            "no_cores": 1,
            "weight_for_each_indicator": {
                "random_weights": "no",
                "iterative": "no",
                "no_samples": 10000,
                "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            },
            "output_path": "/path/to/output"
        }

    def test_main(self):
        # Given
        input_config_1 = TestMCDA.get_uncorrect_config_1()
        input_config_2 = TestMCDA.get_uncorrect_config_2()
        input_config_3 = TestMCDA.get_uncorrect_config_3()

        # When, Then
        with pytest.raises(ValueError):
            main(input_config_1)
        with pytest.raises(ValueError):
            main(input_config_2)
        with pytest.raises(ValueError):
            main(input_config_3)

if __name__ == '__main__':
    unittest.main()