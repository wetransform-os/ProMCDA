"""
This module serves as a configuration object for ProMCDA.
It is designed to store and manage configuration settings in a structured way.
"""

import copy
from typing import List, Dict, Any


# noinspection PyMethodMayBeStatic
class Config:
    """
    Class representing configuration settings.

    This class encapsulates the configuration settings.
    It expects the following keys in the input dictionary:
    - input_matrix_path: path to the input matrix file.
    - polarity_for_each_indicator: list of polarities, one for each indicator.
    - sensitivity: sensitivity configuration.
    - robustness: robustness configuration.
    - monte_carlo_sampling: Monte Carlo sampling configuration.
    - output_directory_path: path to the output file.

    Attributes:
    _valid_keys (List[str]): list of valid keys expected in the input dictionary.
    _list_values (List[str]): list of keys corresponding to list values.
    _str_values (List[str]): list of keys corresponding to string values.
    _int_values (List[str]): list of keys corresponding to integer values.
    _dict_values (List[str]): list of keys corresponding to dictionary values.
    _keys_of_dict_values (Dict[str, List[str]]): dictionary containing keys and their corresponding sub-keys.

    Methods:
    __init__(input_config: dict): instantiate a configuration object.
    _validate(input_config, valid_keys, str_values, int_values, list_values, dict_values): validate the input
    configuration.
    get_property(property_name: str): retrieve a property from the configuration.
    check_dict_keys(dic: Dict[str, Any], keys: List[str]): check if a specific key is in a dictionary.
    check_key(dic: dict, key: str): check if a key is in a dictionary.
    """

    _valid_keys: List[str] = ['input_matrix_path',
                              'polarity_for_each_indicator',
                              'sensitivity',
                              'robustness',
                              'monte_carlo_sampling',
                              'output_directory_path']

    _list_values: List[str] = [
        'marginal_distribution_for_each_indicator', 'polarity_for_each_indicator']

    _str_values: List[str] = ['input_matrix_path', 'output_directory_path', 'sensitivity_on', 'normalization', 'aggregation',
                              'robustness_on', 'on_single_weights', 'on_all_weights', 'given_weights', 'on_indicators']

    _int_values: List[str] = ['monte_carlo_runs', 'num_cores', 'random_seed']

    _dict_values: List[str] = ['sensitivity', 'robustness', 'monte_carlo_sampling']

    _keys_of_dict_values = {'sensitivity': ['sensitivity_on', 'normalization', 'aggregation'],
                            'robustness': ['robustness_on', 'on_single_weights', 'on_all_weights',
                                           'given_weights', 'on_indicators'],
                            'monte_carlo_sampling': ['monte_carlo_runs', 'num_cores', 'random_seed',
                                                     'marginal_distribution_for_each_indicator']}

    def __init__(self, input_config: dict):

        valid_keys = self._valid_keys
        str_values = self._str_values
        int_values = self._int_values
        list_values = self._list_values
        dict_values = self._dict_values
        # keys_of_dict_values = self._keys_of_dict_values

        self._validate(input_config, valid_keys, str_values,
                           int_values, list_values, dict_values)
        self._config = copy.deepcopy(input_config)

    def _validate(self, input_config, valid_keys, str_values, int_values, list_values, dict_values):
        if not isinstance(input_config, dict):
            raise TypeError("input configuration file is not a dictionary")

        for key in valid_keys:
            if key not in input_config:
                raise KeyError("key {} is not in the input config".format(key))

            if key in str_values:
                if not isinstance(input_config[key], str):
                    raise TypeError(
                        "value of {} in the input config is not a string".format(key))

            if key in int_values:
                if not isinstance(input_config[key], int):
                    raise TypeError(
                        "value of {} in the input config is not an integer".format(key))

            if key in list_values:
                if not isinstance(input_config[key], list):
                    raise TypeError(
                        "value of {} in the input config is not a list".format(key))

            if key in dict_values:
                if not isinstance(input_config[key], dict):
                    raise TypeError(
                        "value of {} in the input config is not a dictionary".format(key))
                Config.check_dict_keys(
                    input_config[key], Config._keys_of_dict_values[key])

    def get_property(self, property_name: str):
        return self._config[property_name]

    @property
    def input_matrix_path(self):
        return self.get_property('input_matrix_path')

    @property
    def polarity_for_each_indicator(self):
        return self.get_property('polarity_for_each_indicator')

    @property
    def sensitivity(self):
        return self.get_property('sensitivity')

    @property
    def robustness(self):
        return self.get_property('robustness')

    @property
    def monte_carlo_sampling(self):
        return self.get_property('monte_carlo_sampling')

    @property
    def output_file_path(self):
        return self.get_property('output_directory_path')

    @staticmethod
    def check_dict_keys(dic: Dict[str, Any], keys: List[str]):
        for key in keys:
            Config.check_key(dic, key)

    @staticmethod
    def check_key(dic: dict, key: str):
        if key not in dic.keys():
            raise KeyError(
                "The key = {} is not present in dictionary: {}".format(key, dic))
