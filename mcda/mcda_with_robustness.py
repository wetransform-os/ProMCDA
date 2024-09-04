import copy
import logging
import sys
from typing import List

import pandas as pd
import numpy as np

from ProMCDA.mcda.utils.application_enums import MonteCarloMarginalDistributions
from ProMCDA.models.configuration import Configuration

log = logging.getLogger(__name__)

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("MCDA with sensitivity")


class MCDAWithRobustness:
    """
    Class MCDA with indicators' uncertainty

    This class allows one to run MCDA by considering the uncertainties related to the indicators.
    All indicator values are randomly sampled by different distributions.
    It's possible to have randomly sampled weights too.

    is_exact_pdf_mask and is_poisson_pdf_mask are part of the class interface to avoid circular import issues.
    They are None as default because not needed for certain methods.

    """


    def __init__(self, config: Configuration, input_matrix: pd.DataFrame(), is_exact_pdf_mask=None, is_poisson_pdf_mask=None,
                 random_seed=None):
        self.is_exact_pdf_mask = is_exact_pdf_mask
        self.is_poisson_pdf_mask = is_poisson_pdf_mask
        self.random_seed = random_seed
        self._config = copy.deepcopy(config)
        self._input_matrix = copy.deepcopy(input_matrix)

    @staticmethod
    def repeat_series_to_create_df(initial_series: pd.Series, num_runs: int) -> pd.DataFrame:
        """
        This is a helper function to create a (AxN) df by concatenating
        a series of values of length A for N times. This reproduces a fake random
        sampling in case an indicator has an exact marginal distribution.
        """

        data_list = [initial_series[:] for _ in range(num_runs)]
        out_df = pd.DataFrame(data_list, index=range(num_runs)).T

        return out_df

    @staticmethod
    def convert_list(data_list):
        """
        This is a helper function to convert a list of length I of (AxN) dfs
        into a list of length N of (AxI) dfs.
        """
        alternatives, num_runs = data_list[0].shape

        transposed_list = [pd.DataFrame(index=range(
            alternatives)) for _ in range(num_runs)]

        for i, df in enumerate(data_list):
            for n in range(num_runs):
                transposed_list[n][i] = df.iloc[:, n]

        return transposed_list

    def create_n_randomly_sampled_matrices(self) -> List[pd.DataFrame]:
        """
        Generate N random samples for each indicator in the input matrix based on their marginal distributions.

        This function takes an input matrix of dimensions (A x nI), where:
        - A represents the number of all alternatives,
        - nI is the total number of indicator parameters (including first and second parameters),
        - the columns of the input matrix represent respectively only parameter 1 for exact and Poisson distributions,
          or parameter 1 and parameter 2 for other distributions.

        The function first produces a list of length I, where each element is a matrix of dimensions (A x N).
        Each matrix represents N random samples for every alternative (A) for a specific indicator (I).
        If any of the random samples are negative, they are rescaled to fall within the [0, 1] range.

        Then, the function converts this list into a list of length N, where each element is a matrix of dimensions (A x I).
        The output is a list containing N randomly sampled input matrices, with the PDFs from which random values are sampled
        determined by the indicator marginal distributions.

        Parameters:
        - None

        Returns:
        - list: A list containing N randomly sampled input matrices.

        Notes:
        - 'A' represents the number of all alternatives.
        - 'I' represents the number of all indicators.
        - 'nI' represents the total number of indicator parameters.
        - 'N' represents the number of random samples.

        :return list_random_matrix: List[pd.DataFrame]
        """
        marginal_pdf = self._config.monte_carlo_sampling.marginal_distributions
        num_runs = self._config.monte_carlo_sampling.monte_carlo_runs
        input_matrix = self._input_matrix  # (AxnI)
        is_exact_pdf_mask = self.is_exact_pdf_mask
        is_poisson_pdf_mask = self.is_poisson_pdf_mask
        random_seed = self.random_seed

        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            # TODO: the default_random_seed cannot be used while reading the settings from a JSON file, where None
            #  cannot be given as an option; it will be implemented when the congiguration settings will be passed as a
            #  stream or handle.
            default_random_seed = 42
            np.random.seed(default_random_seed)

        sampled_matrices = []  # list long I

        j = 0
        for i, pdf_type in enumerate(zip(is_exact_pdf_mask, is_poisson_pdf_mask)):
            pdf_exact, pdf_poisson = pdf_type
            par1_position = j
            if pdf_exact == 0 and pdf_poisson == 0:  # non-exact PDF except Poisson
                par2_position = par1_position + 1  # parameter 2's column follows parameter 1's
                parameter1_col = input_matrix.columns[par1_position]
                parameter2_col = input_matrix.columns[par2_position]
                parameter1 = input_matrix[parameter1_col]
                parameter2 = input_matrix[parameter2_col]
                j += 2

            elif pdf_exact == 1 or pdf_poisson == 1:  # exact PDF or Poisson
                parameter1_col = input_matrix.columns[par1_position]
                parameter1 = input_matrix[parameter1_col]
                j += 1

            distribution_type = marginal_pdf[i]

            if distribution_type == MonteCarloMarginalDistributions.EXACT.value:
                samples = self.repeat_series_to_create_df(
                    parameter1, num_runs).T
            elif distribution_type == MonteCarloMarginalDistributions.NORMAL.value:
                samples = np.random.normal(
                    loc=parameter1, scale=parameter2, size=(num_runs, len(parameter1)))
            elif distribution_type == MonteCarloMarginalDistributions.UNIFORM.value:
                samples = np.random.uniform(
                    low=parameter1, high=parameter2, size=(num_runs, len(parameter1)))
            elif distribution_type == MonteCarloMarginalDistributions.LOGNORMAL.value:
                samples = np.random.lognormal(
                    mean=parameter1, sigma=parameter2, size=(num_runs, len(parameter1)))
            elif distribution_type == MonteCarloMarginalDistributions.POISSON.value:
                samples = np.random.poisson(
                    lam=parameter1, size=(num_runs, len(parameter1)))
            else:
                raise ValueError(
                    f"Invalid marginal distribution type: {distribution_type}")

            # check if any sample is negative and rescale btw 0 and 1
            if (samples < 0).any().any():
                samples -= samples.min()
                samples /= samples.max()

            sampled_df = pd.DataFrame(samples.transpose())  # (AxN)
            sampled_matrices.append(sampled_df)

        list_random_matrix = self.convert_list(sampled_matrices)

        return list_random_matrix
