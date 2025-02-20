import sys
import logging
import pandas as pd
import numpy as np
from scipy import stats

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("ProMCDA")


class Aggregation(object):
    """
    This class aggregates the normalized values of each indicator by mean of different aggregation functions.
    The input_matrix contains normalized values of the indicators.

    Type of aggregation functions
    Compensatory: weighted-sum (i.e. additive).
    Partially compensatory: geometric; harmonic.
    Non-compensatory: minimum.
    """
    def __init__(self, weights: list):
        self.weights = weights
        if isinstance(self.weights, list) and all(isinstance(i, list) for i in self.weights):
            for i in range(len(self.weights)):
                if sum(self.weights[i]) != 1:
                    self.weights[i] = [val / sum(self.weights[i]) for val in self.weights[i]]
        elif sum(self.weights) != 1:
            self.weights = [val / sum(self.weights) for val in self.weights]


    def weighted_sum(self, norm_indicators: pd.DataFrame) -> pd.Series(dtype='object'):
        """
        Weighted-sum or additive aggregation function gets as input the normalized values of the indicators in a matrix
        and estimates the scores over the indicators, per alternative. The norm_indicators has
        shape = (num. alternatives x num. indicators) and the returned scores has length = num. of alternatives.

        :param norm_indicators: pd.DataFrame
        :returns scores: pd.Series
        """
        scores = (norm_indicators * self.weights).sum(axis=1)

        return scores

    def geometric(self, norm_indicators: pd.DataFrame) -> pd.Series(dtype='object'):
        """
        The weighted geometric mean works only with strictly positive normalized indicator values
        (i.e. not with minmax and target with feature range (0,1); and not with standardized with feature range
        ('-inf','+inf')). It gets as input positive normalized values of the indicators in a matrix and estimates the
        scores over the indicators, per alternative. The norm_indicators has shape = (no.alternatives x num. indicators)
        and the returned scores has length = num. of alternatives.

        :param norm_indicators: pd.DataFrame
        :returns scores: pd.Series
        """
        if (norm_indicators <= 0).any().any():
            logger.error('Error Message', stack_info=True)
            raise ValueError(
                'Weighted geometric mean cannot work with non-positive values in normalized indicators')
        else:
            scores_array = stats.mstats.gmean(norm_indicators.astype(
                float), axis=1, weights=self.weights)
            scores = pd.Series(scores_array, index=norm_indicators.index)

        return scores

    def harmonic(self, norm_indicators: pd.DataFrame) -> pd.Series(dtype='object'):
        """
        The weighted harmonic mean works only with strictly positive normalized indicator values
        (i.e. not with minmax, and target with feature range (0,1); and not with standardized with feature range
        ('-inf','+inf')). It gets as input positive normalized values of the indicators in a matrix and estimates the
        scores over the indicators, per alternative. The norm_indicators has shape = (no.alternatives x num. indicators)
        and the returned scores has length = num. of alternatives.

        :param norm_indicators: pd.DataFrame
        :returns scores: pd.Series
        """

        if (norm_indicators == 0).any().any():
            logger.error('Error Message', stack_info=True)
            raise ValueError(
                'With 0 values normalized indicators, the weighted harmonic mean will output 0s')
        else:
            scores_array = stats.hmean(norm_indicators, axis=1, weights=self.weights)
            scores = pd.Series(scores_array, index=norm_indicators.index)

        return scores

    def minimum(self, norm_indicators: pd.DataFrame) -> pd.Series(dtype='object'):
        """
        Minimum aggregation function. It does not consider the weights.
        It gets as input the normalized values of the indicators in a matrix and estimates the scores over the
        indicators, per alternative. The norm_indicators has shape = (no.alternatives x num. indicators) and the
        returned scores has length = num. of alternatives.

        :param norm_indicators: pd.DataFrame
        :returns scores: pd.Series
        """
        scores = norm_indicators.min(axis=1)

        return scores
