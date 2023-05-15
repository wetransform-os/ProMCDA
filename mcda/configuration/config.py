import copy

class Config(object):
	"""
	Class Configuration

	keys expected in the input dictionary are
	input_matrix_path: path to the input matrix
	marginal_distribution_for_each indicator: list of marginal distributions, one for each indicator
	polarity_for_each_indicator: list of polarities, one for each indicator
	monte_carlo_runs: number of MC runs
	no_cores: number of cores used in the parallelization
	monte_carlo_runs: list of weights, one for each indicator
	output_path: path to the output file

	"""

	_valid_keys = ['input_matrix_path','marginal_distribution_for_each_indicator',
				   'polarity_for_each_indicator','monte_carlo_runs',
				   'no_cores','weight_for_each_indicator','output_path']

	_list_values: list[str] = ['marginal_distribution_for_each_indicator','polarity_for_each_indicator',
					'weight_for_each_indicator']

	_str_values = ['input_matrix_path','output_path']

	_int_values = ['monte_carlo_runs','no_cores']

	#_dict_values = []
	#_keys_of_dict_values = {}
	#_float_values = []


	def __init__(self, input_config: dict):
		"""
		Instantiate a configuration object
		:param input_config: dict
		"""

		valid_keys = self._valid_keys
		str_values = self._str_values
		int_values = self._int_values
		list_values = self._list_values

		self._validate(input_config, valid_keys, str_values, int_values, list_values)
		self._config = copy.deepcopy(input_config)


	def _validate(self, input_config, valid_keys, str_values, int_values, list_values):
		if not isinstance(input_config, dict):
			raise TypeError("input configuration file is not a dictionary")

		for key in valid_keys:
			if key not in input_config:
				raise KeyError("key {} is not in the input config".format(key))

			if key in str_values:
				if not isinstance (input_config[key], str):
					raise TypeError("value of {} in the input config is not a string".format(key))

			if key in int_values:
				if not isinstance(input_config[key], int):
					raise TypeError("value of {} in the input config is not an integer".format(key))

			if key in list_values:
				if not isinstance(input_config[key], list):
					raise TypeError("value of {} in the input config is not a list".format(key))


	def get_property(self, property_name: str):
		return self._config[property_name]


	@property
	def input_matrix_path(self):
		return self.get_property('input_matrix_path')

	@property
	def marginal_distribution_for_each_indicator(self):
		return self.get_property('marginal_distribution_for_each_indicator')

	@property
	def polarity_for_each_indicator(self):
		return self.get_property('polarity_for_each_indicator')

	@property
	def monte_carlo_runs(self):
		return self.get_property('monte_carlo_runs')

	@property
	def no_cores(self):
		return self.get_property('no_cores')

	@property
	def weight_for_each_indicator(self):
		return self.get_property('weight_for_each_indicator')

	@property
	def output_file_path(self):
		return self.get_property('output_path')