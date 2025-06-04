import unittest
import pandas as pd
from promcda.configuration.configuration_validator import validate_configuration, PDFType

class TestConfigurationValidator(unittest.TestCase):

    def setUp(self):
        self.input_matrix = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        self.polarity = ('-', '+')
        self.weights = [1.0, 0.5]
        self.marginal_distributions = (PDFType.EXACT, PDFType.POISSON)
        self.num_runs = 10
        self.num_cores = 2
        self.random_seed = 42
        self.robustness_weights = False
        self.robustness_single_weights = False
        self.robustness_indicators = False

    def test_valid_configuration(self):
        try:
            validate_configuration(
                self.input_matrix,
                self.polarity,
                self.weights,
                self.marginal_distributions,
                self.num_runs,
                self.num_cores,
                self.random_seed,
                self.robustness_weights,
                self.robustness_single_weights,
                self.robustness_indicators
            )
        except Exception as e:
            self.fail(f"validate_configuration raised an exception {type(e)} unexpectedly!")

    def test_missing_input_matrix(self):
        with self.assertRaises(ValueError):
            validate_configuration(
                None,
                self.polarity,
                self.weights,
                self.marginal_distributions,
                self.num_runs,
                self.num_cores,
                self.random_seed,
                self.robustness_weights,
                self.robustness_single_weights,
                self.robustness_indicators
            )

    def test_invalid_polarity_type(self):
        with self.assertRaises(TypeError):
            validate_configuration(
                self.input_matrix,
                "invalid_polarity",
                self.weights,
                self.marginal_distributions,
                self.num_runs,
                self.num_cores,
                self.random_seed,
                self.robustness_weights,
                self.robustness_single_weights,
                self.robustness_indicators
            )

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            validate_configuration(
                self.input_matrix,
                self.polarity,
                [-1.0, 0.5],
                self.marginal_distributions,
                self.num_runs,
                self.num_cores,
                self.random_seed,
                self.robustness_weights,
                self.robustness_single_weights,
                self.robustness_indicators
            )

    def test_conflicting_robustness_flags(self):
        with self.assertRaises(ValueError):
            validate_configuration(
                self.input_matrix,
                self.polarity,
                self.weights,
                self.marginal_distributions,
                self.num_runs,
                self.num_cores,
                self.random_seed,
                True,
                False,
                True
            )

    def test_weights_length_mismatch(self):
        with self.assertRaises(ValueError):
            validate_configuration(
                self.input_matrix,
                self.polarity,
                [1.0],
                self.marginal_distributions,
                self.num_runs,
                self.num_cores,
                self.random_seed,
                self.robustness_weights,
                self.robustness_single_weights,
                self.robustness_indicators
            )


if __name__ == "__main__":
    unittest.main()