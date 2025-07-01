import copy
import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing


class Normalization(object):
    """
    This class normalizes the values of each indicator in the input_matrix by mean of different normalization functions.
    Each function rescales the values by considering their initial polarity. If the polarity is negative (i.e. the scale
    is reversed), then the smallest the value the better.

    Type of normalization functions
    Ordinal: rank.
    Interval: standardized; min-max.
    Ratio: target.
    """

    def __init__(self, input_matrix: pd.DataFrame, polarities: tuple):

        self._input_matrix = copy.deepcopy(input_matrix)
        self.polarities = polarities

    def _cast_polarities(self) -> tuple[list[int], list[int], pd.DataFrame, pd.DataFrame]:
        """
        Identifies indicators with positive or negative polarity (by indexes) and cast them into two separate dfs.
        """
        ind_plus = [i for i, e in enumerate(self.polarities) if e == "+"]
        ind_minus = [i for i, e in enumerate(self.polarities) if e == "-"]
        indicators_plus = self._input_matrix.iloc[:, ind_plus]
        indicators_minus = self._input_matrix.iloc[:, ind_minus]

        return ind_plus, ind_minus, indicators_plus, indicators_minus

    @staticmethod
    def reversed_minmax_scaler(data, feature_range: tuple):
        """
        Rescales the indicators in a reversed scale where the smallest the value the better, by using the scaling method
        min-max. The feature_range defines if the feature range is (0,1) or (0.1,1).

        Example:
        ```python
        data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        feature_range = (0, 1)
        scaled_data = reversed_minmax_scaler(data, feature_range)
        ```
        This will rescale the input data using the reversed min-max scaling method with the specified feature range:

        ```
        scaled_data:
        [[1.  0.  0. ]
        [0.5 0.5 0.5]
        [0.  1.  1. ]]
        ```

        :param data: pd.DataFrame
        :param feature_range: tuple
        :returns scaled_data: np.array
        """
        data = np.array(data)

        max_val = np.max(data, axis=0)
        min_val = np.min(data, axis=0)

        if feature_range == (0, 1):
            scaled_data = (max_val - data) / (max_val - min_val)
        else:
            scaled_data = (max_val - data) / \
                          (max_val - min_val) * (1 - 0.1) + 0.1

        return scaled_data

    def minmax(self, feature_range: tuple) -> pd.DataFrame:
        """
        Normalizes the indicators by using the scaling method min-max. Different feature ranges are possible.
        The returned indicators_scaled_minmax is of same shape as the input data.

        Example:
        ```python
        input_data = pd.DataFrame([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        normalization_instance = Normalization(input_data)
        feature_range = (0, 1)
        normalized_data = normalization_instance.minmax(feature_range)
        ```
        This will normalize the input data using the min-max scaling method with the specified feature range:

        ```
        min-max normalized_data:
           0    1    2
        0  0.0  0.0  0.0
        1  0.5  0.5  0.5
        2  1.0  1.0  1.0
        ```

        :return indicators_scaled_minmax: pd.DataFrame
        """
        original_shape = self._input_matrix.shape

        pol = Normalization._cast_polarities(self)
        ind_plus = pol[0]
        ind_minus = pol[1]
        indicators_plus = pol[2]
        indicators_minus = pol[3]

        # for + polarity
        x = indicators_plus.to_numpy()  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler(
            feature_range=feature_range, copy=False)
        x_scaled = min_max_scaler.fit_transform(x)
        indicators_scaled_minmax_plus = pd.DataFrame(x_scaled)

        # for - polarity
        y = indicators_minus.to_numpy()
        y_scaled = Normalization.reversed_minmax_scaler(y, feature_range)
        indicators_scaled_minmax_minus = pd.DataFrame(y_scaled)

        # merge back scaled values for positive and negative polarities
        indicators_scaled_minmax = pd.DataFrame(index=range(
            original_shape[0]), columns=range(original_shape[1]))
        for i, index_p in enumerate(ind_plus):
            indicators_scaled_minmax.iloc[:,
                                          index_p] = indicators_scaled_minmax_plus.iloc[:, i]
        for j, index_n in enumerate(ind_minus):
            indicators_scaled_minmax.iloc[:,
                                          index_n] = indicators_scaled_minmax_minus.iloc[:, j]

        return indicators_scaled_minmax

    def target(self, feature_range: tuple) -> pd.DataFrame:
        """
        Normalizes the indicators using the scaling method target.
        The returned indicators_scaled_target is of same shape as the input data.

        Example:
        ```python
        input_data = pd.DataFrame([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        normalization_instance = Normalization(input_data)
        feature_range = (0, 1)
        normalized_data = normalization_instance.target(feature_range)
        ```
        This will normalize the input data using the target scaling method with the specified feature range:

        ```
        target normalized_data:
           0     1     2
        0  0.14  0.25  0.33
        1  0.57  0.52  0.66
        2  1.0   1.0    1.0
        ```

        :return indicators_scaled_target: pd.DataFrame
        """
        original_shape = self._input_matrix.shape

        pol = Normalization._cast_polarities(self)
        ind_plus = pol[0]
        ind_minus = pol[1]
        indicators_plus = pol[2]
        indicators_minus = pol[3]

        if feature_range == (0, 1):
            indicators_scaled_target_plus = indicators_plus / \
                indicators_plus.max(axis=0)  # for + polarity
            indicators_scaled_target_minus = 1 - indicators_minus / \
                indicators_minus.max(axis=0)  # for - polarity
        else:
            indicators_scaled_target_plus = indicators_plus / indicators_plus.max(axis=0) * (
                1 - 0.1) + 0.1  # for + polarity
            indicators_scaled_target_minus = (1 - indicators_minus / indicators_minus.max(axis=0)) * (
                1 - 0.1) + 0.1  # for - polarity

        # merge back scaled values for positive and negative polarities
        indicators_scaled_target = pd.DataFrame(index=range(
            original_shape[0]), columns=range(original_shape[1]))
        for i, index_p in enumerate(ind_plus):
            indicators_scaled_target.iloc[:,
                                          index_p] = indicators_scaled_target_plus.iloc[:, i]
        for j, index_n in enumerate(ind_minus):
            indicators_scaled_target.iloc[:,
                                          index_n] = indicators_scaled_target_minus.iloc[:, j]

        return indicators_scaled_target

    def standardized(self, feature_range: tuple) -> pd.DataFrame:
        """
        Normalizes the indicators using the scaling method standardized (i.e. Z-score).
        The returned indicators_scaled_standardized is of same shape as the input data.

        Note:
        If std == 0 for a column (i.e. no variation), all values are set to 0.0
        to avoid division by zero and preserve neutrality in MCDA.

        Example:
        ```python
        input_data = pd.DataFrame([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        normalization_instance = Normalization(input_data)
        feature_range = (0, 1)
        normalized_data = normalization_instance.standardized(feature_range)
        ```
        This will normalize the input data using the standardized scaling method with the specified feature range:

        ```
        Z-score normalized_data:
           0     1     2
        0 -1.22 -1.22 -1.22
        1  0.00  0.00  0.00
        2  1.22  1.22  1.22
        ```

        :return indicators_scaled_standardized: pd.DataFrame
        """
        if self._input_matrix.isnull().values.any():
            raise ValueError("Input matrix contains NaN values, cannot proceed with standardization.")

        original_shape = self._input_matrix.shape
        pol = Normalization._cast_polarities(self)

        ind_plus = pol[0]
        ind_minus = pol[1]
        indicators_plus = pol[2]
        indicators_minus = pol[3]

        all_indices = set(range(original_shape[1]))
        assigned_indices = set(ind_plus + ind_minus)
        unassigned_indices = all_indices - assigned_indices
        if unassigned_indices:
            raise ValueError(
                f"Missing polarity assignment for indicator columns: {sorted(unassigned_indices)}. "
                "Each indicator must have a defined polarity (positive or negative)."
            )

        indicators_scaled_standardized = pd.DataFrame(index=range(original_shape[0]), columns=range(original_shape[1]))

        # Standardize positively oriented indicators
        if ind_plus:
            mean_plus = indicators_plus.mean(axis=0)
            std_plus = indicators_plus.std(axis=0)

            zero_std_cols_plus = std_plus[std_plus == 0].index
            if not zero_std_cols_plus.empty:
                print(f"[WARNING] Indicators with zero std deviation in columns: {list(zero_std_cols_plus)}")
                std_plus.loc[zero_std_cols_plus] = 1.0  # elude 0 division

            indicators_scaled_stand_plus = (indicators_plus - mean_plus) / std_plus
            # std=0 columns set to 0 (i.e., equal to the mean)
            for col in zero_std_cols_plus:
                col_idx = indicators_plus.columns.get_loc(col)
                indicators_scaled_stand_plus.iloc[:, col_idx] = 0.0

            indicators_scaled_standardized.iloc[:, ind_plus] = indicators_scaled_stand_plus.values
        else:
            warnings.warn("No positively oriented indicators found — skipping positive standardization.")

        # Standardize negatively oriented indicators
        if ind_minus:
            mean_minus = indicators_minus.mean(axis=0)
            std_minus = indicators_minus.std(axis=0)

            zero_std_cols_minus = std_minus[std_minus == 0].index
            if not zero_std_cols_minus.empty:
                print(f"[WARNING] Indicators with zero std deviation in columns: {list(zero_std_cols_minus)}")
                std_minus.loc[zero_std_cols_minus] = 1.0

            indicators_scaled_stand_minus = (mean_minus - indicators_minus) / std_minus
            for j, index_n in enumerate(ind_minus):
                if j in zero_std_cols_minus:
                    col_idx = indicators_minus.columns.get_loc(col)
                    indicators_scaled_stand_minus.iloc[:, col_idx] = 0.0 #std=0 -> z=0
                indicators_scaled_standardized.iloc[:, index_n] = indicators_scaled_stand_minus.iloc[:, j]
        else:
            warnings.warn("No negatively oriented indicators found — skipping negative standardization.")

        if feature_range == ('-inf', '+inf'):
            pass
        else:
            if indicators_scaled_standardized.isnull().any().any():
                cols_with_nan = indicators_scaled_standardized.columns[
                    indicators_scaled_standardized.isnull().any()].tolist()
                raise ValueError(f"Empty (NaN) columns after standardization: {cols_with_nan}")

            if indicators_scaled_standardized.isnull().values.any():
                raise ValueError(
                    "Standardization produced NaN values. Check if all indicators were processed correctly.")

            indicators_scaled_standardized = indicators_scaled_standardized + abs(
                indicators_scaled_standardized.min()) + 0.1

        return indicators_scaled_standardized

    def rank(self) -> pd.DataFrame:
        """
        Normalizes indicators using the scaling method rank.
        The returned indicators_scaled_rank is of same shape as the input data.

        Example:
        ```python
        input_data = pd.DataFrame([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        normalization_instance = Normalization(input_data)
        feature_range = (0, 1)
        normalized_data = normalization_instance.rank(feature_range)
        ```
        This will normalize the input data using the rank scaling method with the specified feature range:

        ```
        rank normalized_data:
            0  1  2
        0  -1  2  3
        1   4  5  6
        2   7  8  9
        ```

        :return indicators_scaled_rank: pd.DataFrame
        """
        original_shape = self._input_matrix.shape

        pol = Normalization._cast_polarities(self)
        ind_plus = pol[0]
        ind_minus = pol[1]
        indicators_plus = pol[2]
        indicators_minus = pol[3]

        indicators_scaled_rank_plus = indicators_plus.rank(
            axis=0)  # for + polarity
        # for - polarity
        indicators_scaled_rank_minus = (-1 * indicators_minus).rank(axis=0)

        # merge back scaled values for positive and negative polarities
        indicators_scaled_rank = pd.DataFrame(index=range(
            original_shape[0]), columns=range(original_shape[1]))
        for i, index_p in enumerate(ind_plus):
            indicators_scaled_rank.iloc[:,
                                        index_p] = indicators_scaled_rank_plus.iloc[:, i]
        for j, index_n in enumerate(ind_minus):
            indicators_scaled_rank.iloc[:,
                                        index_n] = indicators_scaled_rank_minus.iloc[:, j]

        return indicators_scaled_rank
