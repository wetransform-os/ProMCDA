import unittest
import pytest

from mcda.configuration.config import Config


class TestConfig(unittest.TestCase):

    @staticmethod
    def get_correct_config():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ['+', '+', '-'],
            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "yes",
                "on_single_weights": "no",
                "on_all_weights": "yes",
                "given_weights": [0.5, 0.5, 0.5],
                "on_indicators": "no"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 10,
                "num_cores": 4,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['norm', 'lnorm', 'beta']},
            "output_path": "/path/to/output"
        }

    @staticmethod
    def get_uncorrect_config_1():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ['+', '+', '-'],
            "error_key": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "yes",
                "on_single_weights": "no",
                "on_all_weights": "yes",
                "given_weights": [0.5, 0.5, 0.5],
                "on_indicators": "no"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 10,
                "num_cores": 4,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['norm', 'lnorm', 'beta']},
            "output_path": "/path/to/output"
        }

    @staticmethod
    def get_uncorrect_config_2():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "polarity_for_each_indicator": ['+', '+', '-'],
            "sensitivity": {
                "sensitivity_on": "yes",
                "normalization": "minmax",
                "aggregation": "weighted_sum"},
            "robustness": {
                "robustness_on": "yes",
                "on_single_weights": "no",
                "error_key": "yes",
                "given_weights": [0.5, 0.5, 0.5],
                "on_indicators": "no"},
            "monte_carlo_sampling": {
                "monte_carlo_runs": 10,
                "num_cores": 4,
                "random_seed": 42,
                "marginal_distribution_for_each_indicator": ['norm', 'lnorm', 'beta']},
            "output_path": "/path/to/output"
        }

    def test_init(self):
        # Given
        input_config = TestConfig.get_correct_config()

        # When
        config = Config(input_config=input_config)

        # Then
        assert config is not input_config  # but it's a deep copy
        assert isinstance(input_config, dict)
        assert input_config == config._config

    def test_init_should_raise_exception(self):
        # Given
        input_config_1 = TestConfig.get_uncorrect_config_1()
        input_config_2 = TestConfig.get_uncorrect_config_2()

        # When, Then
        with pytest.raises(KeyError):
            Config(input_config=input_config_1)
        with pytest.raises(KeyError):
            Config(input_config=input_config_2)

    def test_get_property(self):
        # Given
        input_config = TestConfig.get_correct_config()

        # When
        config = Config(input_config=input_config)

        # Then
        for key in config._valid_keys:
            assert config.get_property(key) == input_config[key]

    def test_property(self):
        # Given
        input_config = TestConfig.get_correct_config()

        # When
        config = Config(input_config=input_config)

        # Then
        assert config.input_matrix_path == input_config['input_matrix_path']
        assert config.polarity_for_each_indicator == input_config['polarity_for_each_indicator']
        assert config.sensitivity == input_config['sensitivity']
        assert config.robustness == input_config['robustness']
        assert config.monte_carlo_sampling == input_config['monte_carlo_sampling']


if __name__ == '__main__':
    unittest.main()
