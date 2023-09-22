import unittest
import pytest

from mcda.mcda_run import main

class TestMCDA(unittest.TestCase):

    @staticmethod
    def get_incorrect_config_1():
        return {
             "input_matrix_path": "tests/resources/input_matrix_with_uncert.csv",
             "polarity_for_each_indicator": ["+","+","+","+","+","-"],

             "variability": {
                  "variability_on": "yes",
                  "normalization": "minmax",
                  "aggregation": "weighted_sum"},

             "sensitivity": {
                  "sensitivity_on": "yes",
                  "on_single_weights": "no",
                  "on_all_weights": "yes",
                  "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  "on_indicators": "no"},

             "monte_carlo_sampling": {
                 "monte_carlo_runs": 10,
                 "num_cores": 4,
                 "marginal_distribution_for_each_indicator": ["exact", "exact", "exact", "exact", "exact", "exact"]},

             "output_path": "/path/to/output"

}

    @staticmethod
    def get_incorrect_config_2():
        return {
            "input_matrix_path": "tests/resources/input_matrix_with_uncert.csv",
             "polarity_for_each_indicator": ["+","+","+","+","+","-"],

             "variability": {
                  "variability_on": "yes",
                  "normalization": "minmax",
                  "aggregation": "weighted_sum"},

             "sensitivity": {
                  "sensitivity_on": "yes",
                  "on_single_weights": "no",
                  "on_all_weights": "no",
                  "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  "on_indicators": "yes"},

             "monte_carlo_sampling": {
                 "monte_carlo_runs": 0,
                 "num_cores": 4,
                 "marginal_distribution_for_each_indicator": ["norm", "exact", "lnorm", "exact", "poisson", "exact"]},

             "output_path": "/path/to/output"
        }

    @staticmethod
    def get_incorrect_config_3():
        return {
            "input_matrix_path": "tests/resources/input_matrix_with_uncert.csv",
             "polarity_for_each_indicator": ["+","+","+","+","+","-"],

             "variability": {
                  "variability_on": "yes",
                  "normalization": "minmax",
                  "aggregation": "weighted_sum"},

             "sensitivity": {
                  "sensitivity_on": "no",
                  "on_single_weights": "no",
                  "on_all_weights": "no",
                  "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                  "on_indicators": "yes"},

             "monte_carlo_sampling": {
                 "monte_carlo_runs": 10000,
                 "num_cores": 4,
                 "marginal_distribution_for_each_indicator": ["exact", "exact", "exact", "exact", "exact", "exact"]},

             "output_path": "/path/to/output"
        }

    def test_main(self):
        # Given
        input_config_1 = TestMCDA.get_incorrect_config_1()
        input_config_2 = TestMCDA.get_incorrect_config_2()
        input_config_3 = TestMCDA.get_incorrect_config_3()

        # When, Then
        with pytest.raises(ValueError):
            main(input_config_1)
        with pytest.raises(ValueError):
            main(input_config_2)
        with pytest.raises(ValueError):
             main(input_config_3)

if __name__ == '__main__':
    unittest.main()