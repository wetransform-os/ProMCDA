import unittest
import logging
import pytest
from unittest.mock import patch
from mcda.mcda_run import main, UserStoppedError

log = logging.getLogger(__name__)


class TestMCDA(unittest.TestCase):

    @staticmethod
    def get_incorrect_config_1():
        return {
            "input_matrix_path": "tests/resources/input_matrix_with_uncert.csv",
            "polarity_for_each_indicator": ["+", "+", "+", "+", "+", "-"],

            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},

            "robustness": {
                "robustness_on": "yes",
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
            "polarity_for_each_indicator": ["+", "+", "+", "+", "+", "-"],

            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},

            "robustness": {
                "robustness_on": "yes",
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
            "polarity_for_each_indicator": ["+", "+", "+", "+", "+", "-"],

            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},

            "robustness": {
                "robustness_on": "no",
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

    @staticmethod
    def get_correct_config():
         return {
             "input_matrix_path": "tests/resources/input_matrix_with_uncert.csv",
             "polarity_for_each_indicator": ["-", "-", "+", "+", "-", "+"],
             "sensitivity": {
                 "sensitivity_on": "no",
                 "normalization": "minmax",
                 "aggregation": "weighted_sum"},
             "robustness": {
                 "robustness_on": "yes",
                 "on_single_weights": "no",
                 "on_all_weights": "no",
                 "given_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                 "on_indicators": "yes"},
             "monte_carlo_sampling": {
                 "monte_carlo_runs": 1000,
                 "num_cores": 1,
                 "marginal_distribution_for_each_indicator": ['uniform', 'exact', 'normal', 'normal', 'exact',
                                                              'lnorm']},
             "output_path": "/path/to/output"
         }

    @patch('builtins.input', side_effect=['S'])
    def test_main_continue(self, mock_input):
        # Given
        input_config_1 = TestMCDA.get_incorrect_config_1()
        input_config_2 = TestMCDA.get_incorrect_config_2()
        input_config_3 = TestMCDA.get_incorrect_config_3()
        input_config_correct = TestMCDA.get_correct_config()

        # When, Then
        with pytest.raises(ValueError):
            main(input_config_1, user_input_callback=mock_input)
        with pytest.raises(ValueError):
            main(input_config_2)
        with pytest.raises(ValueError):
            main(input_config_3, user_input_callback=mock_input)
        with self.assertLogs(level="INFO") as cm:
            with pytest.raises(UserStoppedError):
                main(input_config_correct, user_input_callback=mock_input)
                self.assertIn('There is a problem with the parameters given in the input matrix with uncertainties. '
                              'Check your data!', cm.output)


if __name__ == '__main__':
    unittest.main()
