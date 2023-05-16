import copy
import numpy as np
import pandas as pd
from typing import List, Tuple, Any
from sklearn import preprocessing

class Normalization(object):
    """
    Class Normalization

    This class normalizes the values of each indicator in the input_matrix
    by mean of different normalization functions.
    Each function rescales the values by considering their initial polarity.
    If the polarity is negative (i.e. the scale is reversed), then the
    smallest the value the better.

    Type of normalization functions
    Ordinal: rank
    Interval: standardized; min-max
    Ratio: target

    """

    def __init__(self, input_matrix: pd.DataFrame(), polarities:list):

        self._input_matrix = copy.deepcopy(input_matrix)
        self.polarities = polarities


    def _cast_polarities(self) -> tuple[list[int], list[int], pd.DataFrame, pd.DataFrame]:
        """
        Identifies indicators with positive
        or negative polarity (by indexes) and
        cast them into two separate dfs

        :data: input matrix
        :returns: list of 2 lists for pos and neg indexes, and 2 dfs
        """

        ind_plus = [i for i, e in enumerate(self.polarities) if e == "+"]
        ind_minus = [i for i, e in enumerate(self.polarities) if e == "-"]
        indicators_plus = self._input_matrix.iloc[:, ind_plus]
        indicators_minus = self._input_matrix.iloc[:, ind_minus]

        return (ind_plus, ind_minus, indicators_plus, indicators_minus)

    @staticmethod
    def reversed_minmax_scaler(data, feature_range: tuple):
        """
        Rescales the indicators in a reversed scale
        where the smallest the value the better,
        by using the scaling method min-max.

        :param: range defines if the feature range is (0,1) or (0.1,1)
        :returns: numpy array
        """

        data = np.array(data)

        max_val = np.max(data, axis=0)
        min_val = np.min(data, axis=0)

        if (feature_range == (0,1)):
            scaled_data = (max_val - data) / (max_val - min_val)
        else:
            scaled_data = (max_val - data) / (max_val - min_val)*(1-0.1) + 0.1

        return scaled_data

    def minmax(self, feature_range:tuple) -> pd.DataFrame():
        """
        Normalizes the indicators by using the scaling method min-max.
        Different feature ranges are possible.
        :returns: pd.DataFrame() of same shape as the input
        """

        original_shape = self._input_matrix.shape

        pol = Normalization._cast_polarities(self)
        ind_plus = pol[0]
        ind_minus = pol[1]
        indicators_plus = pol[2]
        indicators_minus = pol[3]

        # for + polarity
        x = indicators_plus.to_numpy() # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range, copy=False)
        x_scaled = min_max_scaler.fit_transform(x)
        indicators_scaled_minmax_plus = pd.DataFrame(x_scaled)

        # for - polarity
        y = indicators_minus.to_numpy()
        y_scaled = Normalization.reversed_minmax_scaler(y,feature_range)
        indicators_scaled_minmax_minus = pd.DataFrame(y_scaled)

        # merge back scaled values for positive and negative polarities
        indicators_scaled_minmax = pd.DataFrame(index=range(original_shape[0]), columns=range(original_shape[1]))
        for i,index_p in enumerate(ind_plus): indicators_scaled_minmax.iloc[:, index_p] = indicators_scaled_minmax_plus.iloc[:, i]
        for j, index_n in enumerate(ind_minus): indicators_scaled_minmax.iloc[:, index_n]=indicators_scaled_minmax_minus.iloc[:,j]

        return indicators_scaled_minmax


    def target(self, feature_range:tuple) -> pd.DataFrame():
        """
        Normalizes indicators using the scaling method target.
        :return: pd.DataFrame() of same shape as the input
        """

        original_shape = self._input_matrix.shape

        pol = Normalization._cast_polarities(self)
        ind_plus = pol[0]
        ind_minus = pol[1]
        indicators_plus = pol[2]
        indicators_minus = pol[3]

        if self._input_matrix.values.all() > 0:
            if (feature_range == (0,1)):
                indicators_scaled_target_plus = indicators_plus/indicators_plus.max(axis=0) # for + polarity
                indicators_scaled_target_minus = 1-indicators_minus/indicators_minus.max(axis=0) # for - polarity
            else:
                indicators_scaled_target_plus = indicators_plus/indicators_plus.max(axis=0)*(1-0.1) + 0.1 # for + polarity
                indicators_scaled_target_minus = (1-indicators_minus/indicators_minus.max(axis=0))*(1-0.1) + 0.1 # for - polarity
        else:
            raise ValueError('Indicators in the input matrix should have all positive values.')

        # merge back scaled values for positive and negative polarities
        indicators_scaled_target = pd.DataFrame(index=range(original_shape[0]), columns=range(original_shape[1]))
        for i,index_p in enumerate(ind_plus): indicators_scaled_target.iloc[:, index_p] = indicators_scaled_target_plus.iloc[:, i]
        for j, index_n in enumerate(ind_minus): indicators_scaled_target.iloc[:, index_n] = indicators_scaled_target_minus.iloc[:,j]

        return indicators_scaled_target


    def standardized(self) -> pd.DataFrame():
        """
        Normalizes indicators using the scaling method standardized (i.e. Z-score).
        :return: pd.DataFrame() of same shape as the input
        """

        original_shape = self._input_matrix.shape

        pol = Normalization._cast_polarities(self)
        ind_plus = pol[0]
        ind_minus = pol[1]
        indicators_plus = pol[2]
        indicators_minus = pol[3]

        indicators_scaled_stand_plus = (indicators_plus - indicators_plus.mean(axis=0))/indicators_plus.std(axis=0) # for + polarity
        indicators_scaled_stand_minus = (indicators_minus.mean(axis=0) - indicators_minus)/indicators_minus.std(axis=0) # for - polarity

        # merge back scaled values for positive and negative polarities
        indicators_scaled_standardized = pd.DataFrame(index=range(original_shape[0]), columns=range(original_shape[1]))
        for i,index_p in enumerate(ind_plus): indicators_scaled_standardized.iloc[:, index_p] = indicators_scaled_stand_plus.iloc[:, i]
        for j, index_n in enumerate(ind_minus): indicators_scaled_standardized.iloc[:, index_n] = indicators_scaled_stand_minus.iloc[:,j]

        return indicators_scaled_standardized


    def rank(self) -> pd.DataFrame():
        """
        Normalizes indicators using the scaling method rank.
        :return: pd.DataFrame() of same shape as the input
        """
        original_shape = self._input_matrix.shape

        pol = Normalization._cast_polarities(self)
        ind_plus = pol[0]
        ind_minus = pol[1]
        indicators_plus = pol[2]
        indicators_minus = pol[3]

        indicators_scaled_rank_plus = indicators_plus.rank(axis=0) # for + polarity
        indicators_scaled_rank_minus = (-1*indicators_minus).rank(axis=0) # for - polarity

        # merge back scaled values for positive and negative polarities
        indicators_scaled_rank = pd.DataFrame(index=range(original_shape[0]), columns=range(original_shape[1]))
        for i,index_p in enumerate(ind_plus): indicators_scaled_rank.iloc[:, index_p] = indicators_scaled_rank_plus.iloc[:, i]
        for j, index_n in enumerate(ind_minus): indicators_scaled_rank.iloc[:, index_n] = indicators_scaled_rank_minus.iloc[:,j]

        return indicators_scaled_rank