import unittest
import logging
import pytest

from unittest.mock import patch

from mcda.mcda_run import main


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

    def setUp(self):
        # Set up logger in the test
        self.logger = logging.getLogger(__name__)

    @patch('mcda.utils.utils_for_main.check_path_exists', side_effect=None)
    @patch('mcda.utils.utils_for_main.save_config', side_effect=None)
    @patch('mcda.utils.utils_for_main._plot_and_save_charts', side_effect=None)
    @patch('mcda.utils.utils_for_main._save_output_files', side_effect=None)
    @patch('mcda.utils.utils_for_main.save_df', side_effect=None)
    @patch('mcda.utils.utils_for_plotting.save_figure', side_effect=None)
    def test_main_continue(self, mock_check_path_exists,
                           mock_save_config,
                           mock_plot_and_save_charts,
                           mock_save_output_files,
                           mock_save_df,
                           mock_save_figure):
        # Given
        input_config_1 = TestMCDA.get_incorrect_config_1()
        input_config_2 = TestMCDA.get_incorrect_config_2()
        input_config_3 = TestMCDA.get_incorrect_config_3()
        input_config_correct = TestMCDA.get_correct_config()

        mock_check_path_exists.return_value = True
        mock_save_config.return_value = None
        mock_plot_and_save_charts.return_value = None
        mock_save_output_files.return_value = None
        mock_save_df.return_value = None
        mock_save_figure.return_value = None

        # When, Then
        with pytest.raises(ValueError):
           main(input_config_1)
        with pytest.raises(ValueError):
          main(input_config_2)
        with pytest.raises(ValueError):
           main(input_config_3)
        with self.assertLogs(level="INFO") as cm:
            main(input_config_correct)
            expected_message = 'INFO:ProMCDA:There is a problem with the parameters given in the input matrix with ' \
                               'uncertainties. Check your data!'
            self.assertIn(expected_message, cm.output)


if __name__ == '__main__':
    unittest.main()
