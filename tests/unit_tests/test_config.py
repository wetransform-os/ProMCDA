import unittest
import pytest

from mcda.configuration.config import Config

class TestConfig(unittest.TestCase):

    @staticmethod
    def get_correct_config():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "marginal_distribution_for_each_indicator": ['norm', 'lnorm', 'beta'],
            "polarity_for_each_indicator": ['+','+','-'],
            "monte_carlo_runs": 10,
            "no_cores": 17,
            "weight_for_each_indicator": [0.5, 0.5, 0.5],
            "output_file_path": "/path/to/output.csv"
        }

    @staticmethod
    def get_uncorrect_config():
        return {
            "input_matrix_path": "/path/to/input_matrix.csv",
            "error_key": ['norm','lnorm','beta'],
            "polarity_for_each_indicator": ['+','+','-'],
            "monte_carlo_runs": 10,
            "no_cores": 17,
            "weight_for_each_indicator": [0.5, 0.5, 0.5],
            "output_file_path": "/path/to/output.csv"
        }

    def test_init(self):
        # Given
        input_config = TestConfig.get_correct_config()

        # When
        config = Config(input_config=input_config)

        # Then
        assert config is not input_config # but it's a deep copy

    def test_init_should_raise_exception(self):
        # Given
        input_config = TestConfig.get_uncorrect_config()

        # When, Then
        with pytest.raises(KeyError):
            Config(input_config=input_config)

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
        assert config.marginal_distribution_for_each_indicator == input_config['marginal_distribution_for_each_indicator']
        assert config.polarity_for_each_indicator == input_config['polarity_for_each_indicator']
        assert config.monte_carlo_runs == input_config['monte_carlo_runs']
        assert config.no_cores == input_config['no_cores']
        assert config.weight_for_each_indicator == input_config['weight_for_each_indicator']
        assert config.output_file_path == input_config['output_file_path']

        if __name__ == '__main__':
            unittest.main()