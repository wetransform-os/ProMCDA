import copy
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Normalization(object):
    """
    Class Normalization

    This class normalizes the values of each indicator in the input_matrix
    by mean of different normalization functions.

    """

    def __init__(self, input_matrix: pd.DataFrame()):

        self._input_matrix = copy.deepcopy(input_matrix)

    @staticmethod
    def reversed_minmax_scaler(data, feature_range):
        """
        Rescales the given data in a reversed scale
        where the smallest the value the better,
        using the scaling method min-max.

        :param: range defines if the feature range is (0,1) or (0.1,1)
        :returns: numpy array
        """

        data = np.array(data)

        max_val = np.min(data)
        min_val = np.max(data)

        if (feature_range == (0,1)):
            scaled_data = (data - min_val) / (max_val - min_val)
        else:
            scaled_data = (data - min_val) / (max_val - min_val)*(1-0.1) + 0.1

        return scaled_data

    def minmax(self, polarities, feature_range) -> pd.DataFrame():
        """
        Normalizes the given data by using the scaling method min-max.
        It rescales the values by considering also their initial polarity.
        If the polarity is negative (i.e. the scale is reversed), then the
        smallest the value the better.
        Different feature ranges are possible.

        :returns: pd.DataFrame()
        """
        original_shape = self._input_matrix.shape

        # identify indicators with positive or negative polarity and cast into different dfs
        ind_plus = [i for i, e in enumerate(polarities) if e == "+"]
        ind_minus = [i for i, e in enumerate(polarities) if e == "-"]
        indicators_plus = self._input_matrix.iloc[:, ind_plus]
        indicators_minus = self._input_matrix.iloc[:, ind_minus]

        # for + polarity
        x = indicators_plus.to_numpy() # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
        x_scaled = min_max_scaler.fit_transform(x)
        indicators_scaled_minmax_plus = pd.DataFrame(x_scaled)

        # for - polarity
        y = indicators_minus.to_numpy()
        y_scaled = Normalization.reversed_minmax_scaler(y,feature_range)
        indicators_scaled_minmax_minus = pd.DataFrame(y_scaled)

        # merge back scaled values for positive and negative polarities
        indicators_scaled_minmax = pd.DataFrame(columns=range(original_shape[1]))
        for i,index_p in enumerate(ind_plus): indicators_scaled_minmax.iloc[:, index_p] = indicators_scaled_minmax_plus.iloc[:, i]
        for j, index_n in enumerate(ind_minus): indicators_scaled_minmax.iloc[:, index_n]=indicators_scaled_minmax_minus.loc[:,j]

        return indicators_scaled_minmax


    def target(self, polarities) -> pd.DataFrame():
        """
        Normalize indicators using the scaling method target.
        :return: pd.DataFrame()
        """

        x = self._input_matrix

        if x.max().all() > 0:
            indicators_scaled_target = x/x.max()
        else:
            raise ValueError('Indicators in the input matrix should have values larger than 0.')

        return indicators_scaled_target
