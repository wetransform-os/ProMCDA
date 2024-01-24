import copy
import sys
from typing import List

import numpy as np
import pandas as pd

from mcda.configuration.config import Config
from mcda.utils import *

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("MCDA with sensitivity")


class MCDAWithRobustness():
    """

    Class MCDA with indicators' uncertainty

    This class allows one to run MCDA by considering the uncertainties related to the indicators.
    All indicator values are randomly sampled by different distributions.
    It's possible to have randomly sampled weights too.

    """

    def __init__(self, config: Config, input_matrix: pd.DataFrame()):
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
        This function receives an input matrix of dimensions (AxnI).
        nI = (num. indicators associated with an exact or Poisson PDF) + (2 x num. indicators associated with all others PDF)
        The columns of the input matrix represent parameter 1 for exact and Poisson; or parameter1 and parameter 2 for the rest.
        In a first step, the function produces a list of length I of matrices of dimension (AxN).
        Every matrix represents the N random samples of every alternative (A), per indicator (I).
        If there are negative random samples, they are rescaled into [0-1].
        In a second step, a utility function converts this list into a list of length N of matrices of dimension (AxI).
        The output is therefore a list containing N randomly sampled input matrices. The PDFs from where random values
        are sampled depends on the indicator marginal distributions.

        A: all alternatives
        I: all indicators
        nI: all indicators first and second parameters, if the second is needed
        N: number of random samples
        """
        marginal_pdf = self._config.monte_carlo_sampling["marginal_distribution_for_each_indicator"]
        is_exact_pdf_mask = check_if_pdf_is_exact(marginal_pdf)
        is_poisson_pdf_mask = check_if_pdf_is_poisson(marginal_pdf)

        num_runs = self._config.monte_carlo_sampling["monte_carlo_runs"]  # N
        input_matrix = self._input_matrix  # (AxnI)

        np.random.seed(42)

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

            if distribution_type == 'exact':
                samples = self.repeat_series_to_create_df(
                    parameter1, num_runs).T
            elif distribution_type == 'normal':
                samples = np.random.normal(
                    loc=parameter1, scale=parameter2, size=(num_runs, len(parameter1)))
            elif distribution_type == 'uniform':
                samples = np.random.uniform(
                    low=parameter1, high=parameter2, size=(num_runs, len(parameter1)))
            elif distribution_type == 'lnorm':
                samples = np.random.lognormal(
                    mean=parameter1, sigma=parameter2, size=(num_runs, len(parameter1)))
            elif distribution_type == 'poisson':
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
