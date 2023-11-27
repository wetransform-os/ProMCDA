import copy
from typing import List, Dict, Any


class Config(object):
    """
    Class Configuration

    keys expected in the input dictionary are
    input_matrix_path: path to the input matrix
    marginal_distribution_for_each indicator: list of marginal distributions, one for each indicator
    polarity_for_each_indicator: list of polarities, one for each indicator
    monte_carlo_runs: number of MC runs
    num_cores: number of cores used in the parallelization
    monte_carlo_runs: list of weights, one for each indicator
    output_path: path to the output file

    """

    _valid_keys = ['input_matrix_path',
                   'polarity_for_each_indicator',
                   'sensitivity',
                   'robustness',
                   'monte_carlo_sampling',
                   'output_path']

    _list_values: list[str] = ['marginal_distribution_for_each_indicator', 'polarity_for_each_indicator']

    _str_values = ['input_matrix_path', 'output_path', 'sensitivity_on', 'normalization', 'aggregation',
                   'robustness_on', 'on_single_weights', 'on_all_weights','given_weights', 'on_indicators']

    _int_values = ['monte_carlo_runs', 'num_cores']

    _dict_values = ['sensitivity', 'robustness', 'monte_carlo_sampling']

    _keys_of_dict_values = {'sensitivity': ['sensitivity_on', 'normalization', 'aggregation'],
                            'robustness': ['robustness_on', 'on_single_weights', 'on_all_weights',
                                            'given_weights', 'on_indicators'],
                            'monte_carlo_sampling': ['monte_carlo_runs', 'num_cores',
                                                     'marginal_distribution_for_each_indicator']}

    def __init__(self, input_config: dict):
        """
        Instantiate a configuration object
        :param input_config: dict
        """

        valid_keys = self._valid_keys
        str_values = self._str_values
        int_values = self._int_values
        list_values = self._list_values
        dict_values = self._dict_values
        #keys_of_dict_values = self._keys_of_dict_values

        self._validate(input_config, valid_keys, str_values, int_values, list_values, dict_values)
        self._config = copy.deepcopy(input_config)

    def _validate(self, input_config, valid_keys, str_values, int_values, list_values, dict_values):
        if not isinstance(input_config, dict):
            raise TypeError("input configuration file is not a dictionary")

        for key in valid_keys:
            if key not in input_config:
                raise KeyError("key {} is not in the input config".format(key))

            if key in str_values:
                if not isinstance(input_config[key], str):
                    raise TypeError("value of {} in the input config is not a string".format(key))

            if key in int_values:
                if not isinstance(input_config[key], int):
                    raise TypeError("value of {} in the input config is not an integer".format(key))

            if key in list_values:
                if not isinstance(input_config[key], list):
                    raise TypeError("value of {} in the input config is not a list".format(key))

            if key in dict_values:
                if not isinstance(input_config[key], dict):
                    raise TypeError("value of {} in the input config is not a dictionary".format(key))
                Config.check_dict_keys(input_config[key], Config._keys_of_dict_values[key])


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
        return self.get_property('output_path')

    @staticmethod
    def check_dict_keys(dic: Dict[str, Any], keys: List[str]):
        """Check if a specific key is in a dict"""
        for key in keys:
            Config.check_key(dic, key)

    @staticmethod
    def check_key(dic: dict, key: str):
        """Check if a key is in a dict"""
        if key not in dic.keys():
            raise KeyError("The key = {} is not present in dictionary: {}".format(key, dic))
