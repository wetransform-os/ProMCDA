import sys
import logging
import pandas as pd

formatter = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=formatter)
logger = logging.getLogger("MCDTool")

class Aggregation(object):
    """
    Class Aggregation

    This class aggregates the normalized values of each indicator
    by mean of different aggregation functions. The input_matrix contains
    normalized values of the indicators

    Type of aggregation functions
    Compensatory: weighted-sum (i.e. additive)
    Partially compensatory: geometric; harmonic
    Non-compensatory: minimum

    """

    def __init__(self, weights: list):

        self.weights = weights

    def weighted_sum(self, norm_indicators: pd.DataFrame()) -> pd.Series(dtype='object'):
        """
        Weighted-sum or additive aggregation function.
        Gets as input the normalized values of the indicators in a matrix
        and estimates the scores over the indicators, per alternative.

        :gets: pd.DataFrame() of shape (no.alternatives x no.indicators)
        :returns: pd.Series of length = no. of alternatives
        """

        scores = (norm_indicators * self.weights).sum(axis=1)

        return scores

    def geometric(self, norm_indicators: pd.DataFrame()) -> pd.Series(dtype='object'):
        """
        Geometric aggregation function works only with strictly positive
        normalized indicator values (i.e. not with minmax and target with feature range (0,1);
        not with standardized with feature range ('-inf','+inf')).
        Gets as input the normalized values of the indicators in a matrix
        and estimates the scores over the indicators, per alternative.

        :gets: pd.DataFrame() of shape (no.alternatives x no.indicators)
        :returns: pd.Series of length = no. of alternatives
        """

        if (norm_indicators == 0).any().any():
            logger.error('Error Message', stack_info=True)
            raise ValueError('Geometric aggregation cannot work with 0 values normalized indicators')
        else:
            scores = (norm_indicators ** self.weights).product(axis=1)

        return scores

    def harmonic(self, norm_indicators: pd.DataFrame()) -> pd.Series(dtype='object'):
        """
        Harmonic aggregation function works only with strictly positive
        normalized indicator values (i.e. not with minmax, and target with feature range (0,1);
        not with standardized with feature range ('-inf','+inf')).
        Gets as input the normalized values of the indicators in a matrix
        and estimates the scores over the indicators, per alternative.

        :gets: pd.DataFrame() of shape (no.alternatives x no.indicators)
        :returns: pd.Series of length = no. of alternatives
        """
        no_indicators = norm_indicators.shape[1]

        if (norm_indicators == 0).any().any():
            logger.error('Error Message', stack_info=True)
            raise ValueError('Harmonic aggregation cannot work with 0 values normalized indicators')
        else:
            scores = no_indicators/((self.weights/norm_indicators).sum(axis=1))

        return scores

    def minimum(self, norm_indicators: pd.DataFrame()) -> pd.Series(dtype='object'):
        """
        Minimum aggregation function. It does not consider the weights.
        Gets as input the normalized values of the indicators
        in a matrix and estimates the scores over the indicators, per alternative.

        :gets: pd.DataFrame() of shape (no.alternatives x no.indicators)
        :returns: pd.Series of length = no. of alternatives
        """

        scores = norm_indicators.min(axis=1)

        return scores


