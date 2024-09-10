import copy
import io
import os
import argparse
import json
import pickle
import random
import logging
import sys
from pprint import pprint
from typing import Union, Any, List, Tuple
from typing import Optional

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import ProMCDA.mcda.utils.utils_for_parallelization as utils_for_parallelization
import ProMCDA.mcda.utils.utils_for_plotting as utils_for_plotting
from ProMCDA.mcda.mcda_without_robustness import MCDAWithoutRobustness
from ProMCDA.mcda.mcda_with_robustness import MCDAWithRobustness
from ProMCDA.mcda.models.configuration import Configuration
from ProMCDA.mcda.utils.application_enums import RobustnessAnalysis, RobustnessWightLevels, SensitivityAnalysis

DEFAULT_INPUT_DIRECTORY_PATH = './input_files'  # present in the root directory of ProMCDA
DEFAULT_OUTPUT_DIRECTORY_PATH = './output_files'  # present in the root directory of ProMCDA

input_directory_path = os.environ.get('PROMCDA_INPUT_DIRECTORY_PATH') if os.environ.get(
    'PROMCDA_INPUT_DIRECTORY_PATH') else DEFAULT_INPUT_DIRECTORY_PATH
output_directory_path = os.environ.get('PROMCDA_OUTPUT_DIRECTORY_PATH') if os.environ.get(
    'PROMCDA_OUTPUT_DIRECTORY_PATH') else DEFAULT_OUTPUT_DIRECTORY_PATH

log = logging.getLogger(__name__)

logging.getLogger('PIL').setLevel(logging.WARNING)

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")


def check_valid_values(value, enum_class, error_message):
    try:
        enum_class(value)
    except:
        logger.error('Error Message', stack_info=True)
        raise ValueError(error_message)


def check_config_error(condition: bool, error_message: str):
    """
    Check a condition and raise a ValueError with a specified error message if the condition is True.

    Parameters:
    - condition (bool): The condition to check.
    - error_message (str): The error message to raise if the condition is True.

    Raises:
    - ValueError: If the condition is True, with the specified error message.

    :param error_message: str
    :param condition: bool
    :return: None
    """

    if condition:
        logger.error('Error Message', stack_info=True)
        raise ValueError(error_message)


def process_indicators_and_weights(config: Configuration, input_matrix: pd.DataFrame,
                                   is_robustness_indicators: int, is_robustness_weights: int, polar: List[str],
                                   mc_runs: int, num_indicators: int) \
        -> Tuple[List[str], Union[list, List[list], dict]]:
    """
    Process indicators and weights based on input parameters in the configuration.

    Parameters:
    - config: the configuration dictionary.
    - input_matrix: the input matrix without alternatives.
    - is_robustness_indicators: a flag indicating whether the matrix should include indicator uncertainties
      (0 or 1).
    - is_robustness_weights: a flag indicating whether robustness analysis is considered for the weights (0 or 1).
    - marginal_pdf: a list of marginal probability density functions for indicators.
    - mc_runs: number of Monte Carlo runs for robustness analysis.
    - num_indicators: the number of indicators in the input matrix.

    Raises:
    - ValueError: If there are duplicated rows in the input matrix or if there is an issue with the configuration.

    Returns:
    - a shorter list of polarities if one has been dropped together with the relative indicator,
      which brings no information. Otherwise, the same list.
    - the normalised weights (either fixed or random sampled weights, depending on the settings)

    Notes:
    - For is_robustness_indicators == 0:
    - Identifies and removes columns with constant values.
    - Logs the number of alternatives and indicators.

    - For is_robustness_indicators == 1:
    - Handles uncertainty in indicators.
    - Logs the number of alternatives and indicators.

    - For is_robustness_weights == 0:
    - Processes fixed weights if given.
    - Logs weights and normalised weights.

    - For is_robustness_weights == 1:
    - Performs robustness analysis on weights.
    - Logs randomly sampled weights.

    :param mc_runs: int
    :param polar: List[str]
    :param is_robustness_weights: int
    :param is_robustness_indicators: int
    :param input_matrix: pd.DataFrame
    :param config: dict
    :param num_indicators: int
    :return: polar, norm_weights
    :rtype: Tuple[List[str], Union[List[list], dict]]
    """
    num_unique = input_matrix.nunique()
    cols_to_drop = num_unique[num_unique == 1].index
    col_to_drop_indexes = input_matrix.columns.get_indexer(cols_to_drop)

    if is_robustness_indicators == 0:
        _handle_no_robustness_indicators(input_matrix)
    else:  # matrix with uncertainty on indicators
        logger.info("Number of alternatives: {}".format(input_matrix.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
        # TODO: eliminate indicators with constant values (i.e. same mean and 0 std) - optional

    polarities_and_weights = _handle_polarities_and_weights(is_robustness_indicators, is_robustness_weights, num_unique,
                                                            col_to_drop_indexes, polar, config, mc_runs, num_indicators)

    polar, norm_weights = tuple(item for item in polarities_and_weights if item is not None)

    return polar, norm_weights


def _handle_polarities_and_weights(is_robustness_indicators: int, is_robustness_weights: int, num_unique,
                                   col_to_drop_indexes: np.ndarray, polar: List[str], config: Configuration,
                                   mc_runs: int,
                                   num_indicators: int) \
        -> Union[Tuple[List[str], list, None, None], Tuple[List[str], None, List[List], None],
        Tuple[List[str], None, None, dict]]:
    """
    Manage polarities and weights based on the specified robustness settings, ensuring that the appropriate adjustments
    and normalizations are applied before returning the necessary data structures.
    """
    norm_random_weights = []
    rand_weight_per_indicator = {}

    # Managing polarities
    if is_robustness_indicators == 0:
        if any(value == 1 for value in num_unique):
            polar = pop_indexed_elements(col_to_drop_indexes, polar)
    logger.info("Polarities: {}".format(polar))

    # Managing weights
    if is_robustness_weights == 0:
        fixed_weights = config.robustness.given_weights
        if any(value == 1 for value in num_unique):
            fixed_weights = pop_indexed_elements(col_to_drop_indexes, fixed_weights)
        norm_fixed_weights = check_norm_sum_weights(fixed_weights)
        logger.info("Weights: {}".format(fixed_weights))
        logger.info("Normalized weights: {}".format(norm_fixed_weights))
        return polar, norm_fixed_weights, None, None
        #  Return None for norm_random_weights and rand_weight_per_indicator
    else:
        output_weights = _handle_robustness_weights(config, mc_runs, num_indicators)
        if output_weights is not None:
            norm_random_weights, rand_weight_per_indicator = output_weights
        if norm_random_weights:
            return polar, None, norm_random_weights, None
        else:
            return polar, None, None, rand_weight_per_indicator
        #  Return None for norm_fixed_weights and one of the other two cases of randomness


def _handle_robustness_weights(config: Configuration, mc_runs: int, num_indicators: int) \
        -> Tuple[Union[List[list], None], Union[dict, None]]:
    """
    Handle the generation and normalization of random weights based on the specified settings
    when a robustness analysis is requested on all the weights.
    """
    norm_random_weights = []
    rand_weight_per_indicator = {}

    if mc_runs == 0:
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')

    if (config.robustness.robustness == RobustnessAnalysis.WEIGHTS.value
            and config.robustness.on_weights_level == RobustnessWightLevels.ALL.value):
        random_weights = randomly_sample_all_weights(num_indicators, mc_runs)
        norm_random_weights = [check_norm_sum_weights(weights) for weights in random_weights]
        return norm_random_weights, None  # Return norm_random_weights, and None for rand_weight_per_indicator
    elif (config.robustness.robustness == RobustnessAnalysis.WEIGHTS.value
          and config.robustness.on_weights_level == RobustnessWightLevels.SINGLE.value):
        i = 0
        while i < num_indicators:
            random_weights = randomly_sample_ix_weight(num_indicators, i, mc_runs)
            norm_random_weight = [check_norm_sum_weights(weights) for weights in random_weights]
            rand_weight_per_indicator["indicator_{}".format(i + 1)] = norm_random_weight
            i += 1
        return None, rand_weight_per_indicator  # Return None for norm_random_weights, and rand_weight_per_indicator


def _handle_no_robustness_indicators(input_matrix: pd.DataFrame):
    """
    Handle the indicators in case of no robustness analysis required.
    (The input matrix is without the alternative column)
    """
    num_unique = input_matrix.nunique()
    cols_to_drop = num_unique[num_unique == 1].index

    if any(value == 1 for value in num_unique):
        logger.info("Indicators {} have been dropped because they carry no information".format(cols_to_drop))
        input_matrix = input_matrix.drop(cols_to_drop, axis=1)

    num_indicators = input_matrix.shape[1]
    logger.info("Number of alternatives: {}".format(input_matrix.shape[0]))
    logger.info("Number of indicators: {}".format(num_indicators))


def check_indicator_weights_polarities(num_indicators: int, polar: List[str], config: Configuration):
    """
    Check the consistency of indicators, polarities, and fixed weights in a configuration.

    Parameters:
    - num_indicators: the number of indicators in the input matrix.
    - polar: a list containing the polarity associated to each indicator.
    - config: the configuration dictionary.

    This function raises a ValueError if the following conditions are not met:
    1. The number of indicators does not match the number of polarities.
    2. "on_all_weights" is set to "no," and the number of fixed weights
        does not correspond to the number of indicators.

    Raises:
    - ValueError: if the conditions for indicator-polarity and fixed weights consistency are not met.

    :param num_indicators: int
    :param polar: List[str]
    :param config: dict
    :return: None
    """
    if num_indicators != len(polar):
        raise ValueError('The number of polarities does not correspond to the no. of indicators')

    # Check the number of fixed weights if "on_all_weights" is set to "no"
    if (config.robustness.on_weights_level != RobustnessWightLevels.ALL.value and
            num_indicators != len(config.robustness.given_weights)):
        raise ValueError('The no. of fixed weights does not correspond to the no. of indicators')


def check_input_matrix(input_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Check the input matrix for duplicated rows in the alternatives column, rescale negative indicator values
    and drop the index column of alternatives.

    Parameters:
    - input_matrix: The input matrix containing the alternatives and indicators.

    Raises:
    - ValueError: If duplicated rows are found in the alternative column.
    - UserStoppedInfo: If the user chooses to stop when duplicates are found.

     :param input_matrix: pd.DataFrame
     :rtype: pd.DataFrame
     :return: input_matrix
    """
    if input_matrix.duplicated().any():
        raise ValueError('Error: Duplicated rows in the alternatives column.')
    elif input_matrix.iloc[:, 0].duplicated().any():
        logger.info('Duplicated rows in the alternatives column.')

    index_column_values = input_matrix.index.tolist()
    logger.info("Alternatives are {}".format(index_column_values))
    input_matrix_no_alternatives = input_matrix.reset_index(drop=True)  # drop the alternative

    input_matrix_no_alternatives = _check_and_rescale_negative_indicators(
        input_matrix_no_alternatives)

    return input_matrix_no_alternatives


def ensure_directory_exists(path):
    """
    Ensure that the directory specified by the given path exists.
    If the directory does not exist, create it and any intermediate directories as needed.

    Parameters:
        path (str): The path of the file in the directory to ensure exists.

    Example:
    ```python
    ensure_directory_exists(/path/to/directory/file.csv)
    ```

    :param path: str
    :return: None
    """
    try:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        logging.error(f"An error occurred while ensuring directory exists for path '{path}': {e}")
        raise  # Re-raise the exception to propagate it to the caller


def read_matrix_from_file(file_from_stream) -> pd.DataFrame:
    """
    Read an input file from a stream and return it as a DataFrame.
    Set the 'Alternatives' column as index column.

    Note: a default path is assigned to the input_matrix_path in mcda_run.py, and it is used
    unless a custom path is set in an environmental variable in the environment.

    Parameters:
    - input_matrix_path (str): path to the CSV file containing the input matrix.

    Raises:
    - Exception: If an error occurs during the file reading or DataFrame creation.

    :param file_from_stream: str
    :rtype: pd.DataFrame
    """
    try:
        matrix = pd.read_csv(io.StringIO(file_from_stream.stream.read().decode("utf-8")), sep="[,;:]", decimal='.', engine='python')
        data_types = {col: 'float64' for col in matrix.columns[1:]}
        matrix = matrix.astype(data_types)
        alternatives_column_name = matrix.columns[0]
        matrix = matrix.set_index(alternatives_column_name)
        return matrix
    except Exception as e:
        print(e)


def read_matrix(input_matrix_path: str) -> pd.DataFrame:
    """
    Read an input matrix from a CSV file and return it as a DataFrame.
    Set the 'Alternatives' column as index column.

    Note: a default path is assigned to the input_matrix_path in mcda_run.py, and it is used
    unless a custom path is set in an environmental variable in the environment.

    Parameters:
    - input_matrix_path (str): path to the CSV file containing the input matrix.

    Raises:
    - Exception: If an error occurs during the file reading or DataFrame creation.

    :param input_matrix_path: str
    :rtype: pd.DataFrame
    """
    try:
        full_file_path = os.path.join(input_directory_path, input_matrix_path)
        with open(full_file_path, 'r') as fp:
            logger.info("Reading the input matrix in {}".format(full_file_path))
            matrix = pd.read_csv(fp, sep="[,;:]", decimal='.', engine='python')
            data_types = {col: 'float64' for col in matrix.columns[1:]}
            matrix = matrix.astype(data_types)
            alternatives_column_name = matrix.columns[0]
            matrix = matrix.set_index(alternatives_column_name)
            return matrix
    except Exception as e:
        print(e)


def reset_index_if_needed(series):
    """
    Reset the index of a pandas Series if it's not a RangeIndex.

    Parameters:
    - series (pd.Series): The input pandas Series.

    Returns:
    - pd.Series: The series with the index reset if needed.

    :param series: pd.Series
    :return series: pd.Series
    """
    if not isinstance(series.index, pd.RangeIndex):
        series = series.reset_index(drop=True)
    return series


def _check_and_rescale_negative_indicators(input_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale indicators of the input matrix if negative into [0-1].
    """

    if (input_matrix < 0).any().any():
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(input_matrix)
        scaled_matrix = pd.DataFrame(
            scaled_data, columns=input_matrix.columns, index=input_matrix.index)
        return scaled_matrix
    else:
        return input_matrix


def parse_args():
    """
    Parse command line arguments for configuration path.

    Raises:
    - argparse.ArgumentError: If the required argument is not provided.

    :rtype: str
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config path', required=True)
    args = parser.parse_args()

    return args.config


def get_config(config_path: str) -> dict:
    """
    Load and return a configuration dictionary from a JSON file.

    Raises:
    - Exception: If an error occurs during file reading or JSON decoding.

    :param config_path: str
    :rtype: dict
    """
    try:
        with open(config_path, 'r') as fp:
            return json.load(fp)
    except Exception as e:
        print(e)


def save_df(df: pd.DataFrame, folder_path: str, filename: str) -> {}:
    """
    Save a DataFrame to a CSV file with a timestamped filename.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be saved.
    - folder_path (str): The path to the folder where the file will be saved.
    - filename (str): The original filename for the CSV file.

    Notes:
    - The saved file will have a timestamp added to its filename.
    - A default output path is assigned, and it is used unless
      a custom path is set in an environmental variable in the environment.

    Example:
    ```python
    save_df(my_dataframe, '/path/to/folder', 'data.csv')
    ```

    :param df: pd.DataFrame
    :param folder_path: str
    :param filename: str
    :return: None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    if not os.path.isdir(folder_path):
        logging.error(f"The provided folder path '{folder_path}' is not a valid directory.")
        return

    full_output_path = os.path.join(output_directory_path, folder_path, new_filename)

    try:
        ensure_directory_exists(os.path.dirname(full_output_path))
    except Exception as e:
        logging.error(f"Error while saving data frame: {e}")
        return

    try:
        df.to_csv(path_or_buf=full_output_path, index=False)
        df_modified = df.set_index(df.columns[0])
        return df_modified[df.columns[1]].to_dict()
    except IOError as e:
        logging.error(f"Error while writing data frame into a CSV file: {e}")


def save_dict(dictionary: dict, folder_path: str, filename: str) -> {}:
    """
    Save a dictionary to a binary file using pickle with a timestamped filename.

    Note: a default output path is assigned, and it is used unless
    a custom path is set in an environmental variable in the environment.

    Parameters:
    - dictionary (dict): The dictionary to be saved.
    - folder_path (str): The path to the folder where the file will be saved.
    - filename (str): The original filename for the binary file.

    Example:
    ```python
    save_dict(my_dict, '/path/to/folder', 'data.pkl')
    ```

    :param dictionary: dict
    :param folder_path: str
    :param filename: str
    :return: None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    if not os.path.isdir(folder_path):
        logging.error(f"The provided folder path '{folder_path}' is not a valid directory.")
        return

    full_output_path = os.path.join(output_directory_path, folder_path, new_filename)
    try:
        ensure_directory_exists(os.path.dirname(full_output_path))
    except Exception as e:
        logging.error(f"Error while saving dictionary: {e}")
        return
    try:
        with open(full_output_path, 'wb') as fp:
            pickle.dump(dictionary, fp)
    except IOError as e:
        logging.error(f"Error while dumping the dictionary into a pickle file: {e}")


def save_config(config: Configuration, folder_path: str, filename: str):
    """
    Save a configuration dictionary to a JSON file with a timestamped filename.

    Note: a default output path is assigned, and it is used unless
    a custom path is set in an environmental variable in the environment.

    Parameters:
    - config (dict): The configuration dictionary to be saved.
    - folder_path (str): The path to the folder where the file will be saved.
    - filename (str): The original filename for the JSON file.

    Example:
    ```python
    save_config(my_config, '/path/to/folder', 'config.json')
    ```

    :param config: dict
    :param folder_path: str
    :param filename: str
    :return: None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    if not os.path.isdir(folder_path):
        logging.error(f"The provided folder path '{folder_path}' is not a valid directory.")
        return

    full_output_path = os.path.join(output_directory_path, folder_path, new_filename)
    try:
        ensure_directory_exists(os.path.dirname(full_output_path))
    except Exception as e:
        logging.error(f"Error while saving configuration: {e}")
        return

    try:
        with open(full_output_path, 'w') as fp:
            json.dump(config.to_dict(), fp)
    except IOError as e:
        logging.error(f"Error while dumping the configuration into a JSON file: {e}")


def check_path_exists(path: str):
    """
    Check if a directory path exists, and create it if it doesn't.

    Example:
    ```python
    check_path_exists('/path/to/directory')
    ```

    :param path: str
    :return: None
    """
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)


def rescale_minmax(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale the values in a DataFrame to a [0,1] range using Min-Max scaling.

    Parameters:
    - scores (pd.DataFrame): The DataFrame containing numerical values to be rescaled.

    Example:
    ```python
    my_dataframe = pd.DataFrame({
    'A': [10, 40, 70],
    'B': [20, 50, 80],
    'C': [30, 60, 90]
     })
    rescaled_df = rescale_minmax(my_dataframe)

    rescaled_df:
         A    B    C
    0   0.0  0.0  0.0
    1   0.5  0.5  0.5
    2   1.0  1.0  1.0
    ```
    :param scores: pd.DataFrame
    :rtype: pd.DataFrame
    """
    x = scores.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalized_scores = pd.DataFrame(x_scaled)
    normalized_scores.columns = scores.columns

    return normalized_scores


def randomly_sample_all_weights(num_weights: int, num_runs: int) -> List[list]:
    """
    Generate multiple lists of random weights for simulations.

    Parameters:
    - num_weights: the number of weights in each list.
    - num_runs: the number of lists to be generated.

    Returns:
    - List[list]: A list containing 'num_runs' lists, each with 'num_weights' randomly sampled elements.

    Example:
    ``python
    sampled_weights = randomly_sample_all_weights(5, 10)
    ```
    This will generate 10 lists, each containing 5 randomly sampled weights.

    :param num_weights: int
    :param num_runs: int
    :return list_of_weights: List[list]
    """
    list_of_weights = []
    for _ in range(num_runs):
        lst = [np.random.uniform(0, 1) for _ in range(num_weights)]
        list_of_weights.append(lst)

    return list_of_weights


def randomly_sample_ix_weight(num_weights: int, index: int, num_runs: int) -> List[list]:
    """
    Generate multiple lists of weights with one randomly sampled element at a specified index.

    Parameters:
    - num_weights: the total number of weights in each list.
    - index: the index at which a weight will be randomly sampled in each list.
    - num_runs: the number of lists to be generated.

    Returns:
    - a list containing 'num_runs' lists, each with 'num_weights' elements
      and one randomly sampled at the specified index.

    Example:
    ```python
    sampled_weights = randomly_sample_ix_weight(5, 2, 10)
    ```
    This will generate 10 lists, each containing 5 weights, with one randomly sampled at index 2.

    :param num_weights: int
    :param index: int
    :param num_runs: int
    :return list_of_weights: List[list]
    """
    list_of_weights = []
    for _ in range(num_runs):
        lst = [1] * num_weights
        lst[index] = random.uniform(0, 1)
        list_of_weights.append(lst)

    return list_of_weights


def check_norm_sum_weights(weights: list) -> list:
    """
    Check if the sum of weights is equal to 1, and normalize the weights if needed.

    Returns:
    - list: the original weights if the sum is already 1, or normalized weights if the sum is not 1.

    :param weights: list
    :return weights: list
    """
    if sum(weights) != 1:
        norm_weights = [val / sum(weights) for val in weights]
        return norm_weights
    else:
        return weights


def pop_indexed_elements(indexes: np.ndarray, original_list: list) -> list:
    """
    Eliminate elements from a list at specified indexes.

    Parameters:
    - indexes: an array of indexes indicating elements to be removed.
    - original_list: the original list from which elements will be removed.

    Example:
    ```python
    new_list = pop_indexed_elements(np.array([1, 3]), [10, 20, 30, 40, 50])
    ```
    This will remove elements at indexes 1 and 3 from the original list and return the new list [10, 30, 50].

    :param indexes: np.ndarray
    :param original_list: list
    :return new_list: list
    """
    for i in range(len(indexes)):
        index = indexes[i]
        if i == 0:
            original_list.pop(index)
        else:
            original_list.pop(index - i)
    new_list = original_list

    return new_list


def check_parameters_pdf(input_matrix: pd.DataFrame, config: Configuration, for_testing=False) -> Union[
    List[bool], None]:
    """
    Check conditions on parameters based on the type of probability distribution function (PDF) for each indicator and
    raise logging information in case of any problem.

    The function checks if the values of parameter1 are larger or equal to the ones of parameter2
    (or vice-versa for the uniform PDF) in case of an input matrix with uncertainties.
    The check runs only in the case of indicators with a PDF of the type normal/lognormal, and uniform.
    In the first case, the parameters represent the (log)mean and (log)std;
    in the second case, they represent the min and max values.

    Parameters:
    - input_matrix: the input matrix containing uncertainties for indicators, no alternatives.
    - config: configuration dictionary containing the Monte Carlo sampling information.
    - for_testing: true only for unit testing

    Returns:
    - List: a list indicating whether the conditions are satisfied for each indicator (only for unit tests).
    - None: default

    :param input_matrix: pd.DataFrame
    :param config: dict
    :param for_testing: bool
    :return: Union[list, None]
    """
    satisfies_condition = False
    problem_logged = False

    marginal_pdf = config.monte_carlo_sampling.marginal_distributions
    is_exact_pdf_mask = check_if_pdf_is_exact(marginal_pdf)
    is_poisson_pdf_mask = check_if_pdf_is_poisson(marginal_pdf)
    is_uniform_pdf_mask = check_if_pdf_is_uniform(marginal_pdf)

    j = 0
    list_of_satisfied_conditions = []
    for i, pdf_type in enumerate(zip(is_exact_pdf_mask, is_poisson_pdf_mask, is_uniform_pdf_mask)):
        pdf_exact, pdf_poisson, pdf_uniform = pdf_type
        param1_position = j
        if pdf_exact == 0 and pdf_poisson == 0:  # non-exact PDF, except Poisson
            param2_position = param1_position + 1  # param2 column follows param1
            param1_col = input_matrix.columns[param1_position]
            param2_col = input_matrix.columns[param2_position]
            param1 = input_matrix[param1_col]
            param2 = input_matrix[param2_col]
            j += 2

            if pdf_uniform == 0:  # normal/lognormal
                satisfies_condition = all(x >= y for x, y in zip(
                    param1, param2))  # check param1 > param2 (mean > std=
            else:  # uniform
                satisfies_condition = all(x <= y for x, y in zip(
                    param1, param2))  # check param2 > param1 (max > min)

        elif pdf_exact == 1 or pdf_poisson == 1:  # exact PDF or Poisson distribution
            satisfies_condition = True
            j += 1  # no check, read one column only (rate)

        list_of_satisfied_conditions.append(satisfies_condition)

        if any(not value for value in list_of_satisfied_conditions) and not problem_logged:
            logger.info(
                'There is a problem with the parameters given in the input matrix with uncertainties. Check your data!')
            logger.info(
                'Either standard deviation values of normal/lognormal indicators are larger than their means')
            logger.info('or max. values of uniform distributed indicators are smaller than their min. values.')
            logger.info('If you continue, the negative values will be rescaled internally to a positive range.')
            problem_logged = True  # Set a flag to True after logging the problem message the first time

    if for_testing:
        return list_of_satisfied_conditions


def check_if_pdf_is_exact(marginal_pdf: list) -> list:
    """
    Check if each indicator's probability distribution function (PDF) is of type 'exact'.

    Parameters:
    - marginal_pdf: a list containing the type of PDF for each indicator.

    Returns:
    - list: a binary mask indicating whether the PDF for each indicator is 'exact' (1) or not (0).

    Example:
    ```python
    is_exact_pdf = check_if_pdf_is_exact(['exact', 'normal', 'exact', 'uniform'])
    ```
    This will return a list [1, 0, 1, 0], indicating that the first and third indicators have 'exact' PDFs.

    :param marginal_pdf: list
    :return exact_pdf_mask: List[int]
    """
    exact_pdf_mask = [1 if pdf == 'exact' else 0 for pdf in marginal_pdf]

    return exact_pdf_mask


def check_if_pdf_is_poisson(marginal_pdf: list) -> list:
    """
    Check if each indicator's probability distribution function (PDF) is of type 'poisson'.

    Parameters:
    - marginal_pdf: a list containing the type of PDF for each indicator.

    Returns:
    - list: a binary mask indicating whether the PDF for each indicator is 'poisson' (1) or not (0).

    Example:
    ```python
    is_poisson_pdf = check_if_pdf_is_poisson(['poisson', 'normal', 'exact', 'uniform'])
    ```
    This will return a list [1, 0, 0, 0], indicating that the first indicator have a 'poisson' PDF.

    :param marginal_pdf: list
    :return poisson_pdf_mask: List[int]
    """
    poisson_pdf_mask = [1 if pdf == 'poisson' else 0 for pdf in marginal_pdf]

    return poisson_pdf_mask


def check_if_pdf_is_uniform(marginal_pdf: list) -> list:
    """
    Check if each indicator's probability distribution function (PDF) is of type 'uniform'.

    Parameters:
    - marginal_pdf (list): a list containing the type of PDF for each indicator.

    Returns:
    - list: a binary mask indicating whether the PDF for each indicator is 'uniform' (1) or not (0).

    Example:
    ```python
    is_exact_pdf = check_if_pdf_is_exact(['exact', 'normal', 'exact', 'uniform'])
    ```
    This will return a list [0, 0, 0, 1], indicating that the fourth indicator have a 'uniform' PDF.

    :param marginal_pdf: list
    :return uniform_pdf_mask: List[int]
    """
    uniform_pdf_mask = [1 if pdf == 'uniform' else 0 for pdf in marginal_pdf]

    return uniform_pdf_mask


def run_mcda_without_indicator_uncertainty(config: Configuration, index_column_name: str, index_column_values: list,
                                           input_matrix: pd.DataFrame,
                                           weights: Union[list, List[list], dict],
                                           f_norm: str, f_agg: str, is_robustness_weights: int) -> dict:
    """
    Runs ProMCDA without uncertainty on the indicators, i.e. without performing
    a robustness analysis.

    This function executes the MCDA process without considering uncertainty
    in the indicators. It computes scores and weights, saves the results,
    and logs the completion time.

    Parameters:
    - input_matrix: the input_matrix without the alternatives.
    - index_column_name: the name of the index column of the original input matrix.
    - index_column_values: the values of the index column of the original input matrix.
    - weights: the normalised weights (either fixed or random sampled weights, depending on the settings).

    :param input_config: dict
    :param index_column_name: str
    :param index_column_values: list
    :param input_matrix: pd:DataFrame
    :param weights: Union[List[str], List[pd.DataFrame], dict, None]
    :param f_norm: str
    :param f_agg: str
    :param is_robustness_weights: int
    :return: None
    """
    scores = pd.DataFrame
    normalized_scores = pd.DataFrame
    all_weights_score_means = pd.DataFrame
    all_weights_score_stds = pd.DataFrame
    all_weights_score_means_normalized = pd.DataFrame
    iterative_random_w_score_means_normalized = {}
    iterative_random_w_score_means = {}
    iterative_random_w_score_stds = {}

    logger.info("Start ProMCDA without robustness of the indicators")
    is_sensitivity = config.sensitivity.sensitivity_on == SensitivityAnalysis.YES.value
    is_robustness = config.robustness.robustness != RobustnessAnalysis.NONE.value
    mcda_no_uncert = MCDAWithoutRobustness(config, input_matrix)

    normalized_indicators = mcda_no_uncert.normalize_indicators() if is_sensitivity == "yes" \
        else mcda_no_uncert.normalize_indicators(f_norm)

    if not is_robustness:
        scores = mcda_no_uncert.aggregate_indicators(normalized_indicators, weights) \
            if is_sensitivity \
            else mcda_no_uncert.aggregate_indicators(normalized_indicators, weights, f_agg)
        normalized_scores = rescale_minmax(scores)
    elif (config.robustness.robustness == RobustnessAnalysis.WEIGHTS.value
          and config.robustness.on_weights_level == RobustnessWightLevels.ALL.value):
        # ALL RANDOMLY SAMPLED WEIGHTS (MCDA runs num_samples times)
        all_weights_score_means, all_weights_score_stds, \
            all_weights_score_means_normalized, all_weights_score_stds_normalized = \
            _compute_scores_for_all_random_weights(normalized_indicators, config.sensitivity.sensitivity_on, weights,
                                                   f_agg)
    elif (config.robustness.robustness == RobustnessAnalysis.WEIGHTS.value
          and config.robustness.on_weights_level == RobustnessWightLevels.SINGLE.value):
        # ONE RANDOMLY SAMPLED WEIGHT A TIME (MCDA runs (num_samples * num_indicators) times)
        iterative_random_weights_statistics = _compute_scores_for_single_random_weight(
            normalized_indicators, weights, config.sensitivity.sensitivity_on, index_column_name, index_column_values,
            f_agg, input_matrix)
        iterative_random_w_score_means = iterative_random_weights_statistics['score_means']
        iterative_random_w_score_stds = iterative_random_weights_statistics['score_stds']
        iterative_random_w_score_means_normalized = iterative_random_weights_statistics['score_means_normalized']

    ranks = _compute_ranks(scores, index_column_name, index_column_values,
                           all_weights_score_means, iterative_random_w_score_means)

    response = _save_output_files(scores=scores, normalized_scores=normalized_scores, ranks=ranks,
                                  score_means=all_weights_score_means, score_stds=all_weights_score_stds,
                                  score_means_normalized=all_weights_score_means_normalized,
                                  iterative_random_w_score_means=iterative_random_w_score_means,
                                  iterative_random_w_score_means_normalized=iterative_random_w_score_means_normalized,
                                  iterative_random_w_score_stds=iterative_random_w_score_stds,
                                  index_column_name=index_column_name, index_column_values=index_column_values,
                                  config=config)

    _plot_and_save_charts(scores=scores, normalized_scores=normalized_scores,
                          score_means=all_weights_score_means, score_stds=all_weights_score_stds,
                          score_means_normalized=all_weights_score_means_normalized,
                          iterative_random_w_score_means=iterative_random_w_score_means,
                          iterative_random_w_score_stds=iterative_random_w_score_stds,
                          iterative_random_w_score_means_normalized=iterative_random_w_score_means_normalized,
                          input_matrix=input_matrix, config=config,
                          is_robustness_weights=is_robustness_weights)
    return response


def run_mcda_with_indicator_uncertainty(config: Configuration, input_matrix: pd.DataFrame, index_column_name: str,
                                        index_column_values: list, mc_runs: int, random_seed: int, is_sensitivity: str,
                                        f_agg: str, f_norm: str, weights: Union[List[list], List[pd.DataFrame], dict],
                                        polar: List[str], marginal_pdf: List[str]) -> dict:
    """
    Runs ProMCDA with uncertainty on the indicators, i.e. with a robustness analysis.

    This function executes the MCDA process with uncertainty in the indicators.
    It computes scores, saves the results, and logs the completion time.

    Parameters:
    - input_matrix: the input_matrix without the alternatives.
    - index_column_name: the name of the index column of the original input matrix.
    - index_column_values: the values of the index column of the original input matrix.
    - weights: the normalised weights (either fixed or random sampled weights, depending on the settings).
      In the context of the robustness analysis, only fixed normalised weights are used, i.e. weights[0].

    :param input_config: dict
    :param index_column_name: str
    :param index_column_values: list
    :param input_matrix: pd:DataFrame
    :param mc_runs: int
    :param is_sensitivity: str
    :param weights: Union[List[str], List[pd.DataFrame], dict, None]
    :param f_norm: str
    :param f_agg: str
    :param polar: List[str]
    :param marginal_pdf: List[str]
    :return: None
    """
    logger.info("Start ProMCDA with uncertainty on the indicators")
    is_robustness_indicators = True
    all_indicators_scores_normalized = []

    if mc_runs <= 0:
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')

    if mc_runs < 1000:
        logger.info("The number of Monte-Carlo runs is only {}".format(mc_runs))
        logger.info("A meaningful number of Monte-Carlo runs is equal or larger than 1000")

    check_parameters_pdf(input_matrix, config)
    is_exact_pdf_mask = check_if_pdf_is_exact(marginal_pdf)
    is_poisson_pdf_mask = check_if_pdf_is_poisson(marginal_pdf)

    mcda_with_uncert = MCDAWithRobustness(config, input_matrix, is_exact_pdf_mask, is_poisson_pdf_mask, random_seed)
    n_random_input_matrices = mcda_with_uncert.create_n_randomly_sampled_matrices()

    if is_sensitivity == "yes":
        n_normalized_input_matrices = utils_for_parallelization.parallelize_normalization(n_random_input_matrices,
                                                                                          polar)
    else:
        n_normalized_input_matrices = utils_for_parallelization.parallelize_normalization(n_random_input_matrices,
                                                                                          polar, f_norm)

    args_for_parallel_agg = [(weights, normalized_indicators)
                             for normalized_indicators in n_normalized_input_matrices]

    if is_sensitivity == "yes":
        all_indicators_scores = utils_for_parallelization.parallelize_aggregation(args_for_parallel_agg)
    else:
        all_indicators_scores = utils_for_parallelization.parallelize_aggregation(args_for_parallel_agg, f_agg)

    for matrix in all_indicators_scores:
        normalized_matrix = rescale_minmax(matrix)
        all_indicators_scores_normalized.append(normalized_matrix)

    all_indicators_scores_means, all_indicators_scores_stds = \
        utils_for_parallelization.estimate_runs_mean_std(all_indicators_scores)
    all_indicators_means_scores_normalized, all_indicators_scores_stds_normalized = \
        utils_for_parallelization.estimate_runs_mean_std(all_indicators_scores_normalized)

    ranks = all_indicators_scores_means.rank(pct=True)

    response = _save_output_files(scores=None, normalized_scores=None,
                                  ranks=ranks,
                                  score_means=all_indicators_scores_means,
                                  score_stds=all_indicators_scores_stds,
                                  score_means_normalized=all_indicators_means_scores_normalized,
                                  iterative_random_w_score_means=None,
                                  iterative_random_w_score_means_normalized=None,
                                  iterative_random_w_score_stds=None,
                                  config=config,
                                  index_column_name=index_column_name, index_column_values=index_column_values)

    _plot_and_save_charts(scores=None, normalized_scores=None,
                          score_means=all_indicators_scores_means, score_stds=all_indicators_scores_stds,
                          score_means_normalized=all_indicators_means_scores_normalized,
                          iterative_random_w_score_means=None,
                          iterative_random_w_score_stds=None,
                          iterative_random_w_score_means_normalized=None,
                          input_matrix=input_matrix, config=config,
                          is_robustness_indicators=is_robustness_indicators)

    return response


def _compute_scores_for_all_random_weights(indicators: dict, is_sensitivity: str,
                                           weights: Union[List[str], List[pd.DataFrame], dict, None],
                                           f_agg: str) -> tuple[Any, Any, Any, Any]:
    """
    Computes the normalized mean scores and std of the alternatives in the case of randomly sampled weights.
    """
    logger.info("All weights are randomly sampled from a uniform distribution.")
    all_weights_scores_normalized = []

    random_weights = None
    try:
        random_weights = weights
    except(TypeError, IndexError):
        logger.error('Error Message', stack_info=True)
        raise ValueError('Error accessing weights. Setting random_weights to None.')

    args_for_parallel_agg = [(lst, indicators) for lst in random_weights]

    if is_sensitivity == "yes":
        all_weights_scores = utils_for_parallelization.parallelize_aggregation(args_for_parallel_agg)
    else:
        all_weights_scores = utils_for_parallelization.parallelize_aggregation(args_for_parallel_agg, f_agg)

    for matrix in all_weights_scores:
        normalized_matrix = rescale_minmax(matrix)  # all score normalization
        all_weights_scores_normalized.append(normalized_matrix)
    all_weights_score_means, all_weights_score_stds = utils_for_parallelization.estimate_runs_mean_std(
        all_weights_scores)  # mean and std of rough scores
    all_weights_score_means_normalized, all_weights_score_stds_normalized = \
        utils_for_parallelization.estimate_runs_mean_std(all_weights_scores_normalized)  # mean and std of norm. scores

    return all_weights_score_means, all_weights_score_stds, \
        all_weights_score_means_normalized, all_weights_score_stds_normalized


def _compute_scores_for_single_random_weight(indicators: dict,
                                             weights: Union[List[str], List[pd.DataFrame], dict, None],
                                             is_sensitivity: str, index_column_name: str, index_column_values: list,
                                             f_agg: str, input_matrix: pd.DataFrame) -> dict:
    """
    Computes the mean scores and std of the alternatives in the case of one randomly sampled weight at time.
    """
    iterative_random_w_score_means = {}
    iterative_random_w_score_stds = {}
    iterative_random_w_score_means_normalized = {}
    iterative_random_w_score_stds_normalized = {}

    logger.info("One weight at time is randomly sampled from a uniform distribution.")
    scores_one_random_weight_normalized = {}
    num_indicators = input_matrix.shape[1]

    rand_weight_per_indicator = None
    try:
        rand_weight_per_indicator = weights
    except(TypeError, IndexError):
        logger.error('Error Message', stack_info=True)
        raise ValueError('Error accessing weights. Setting rand_weight_per_indicator to None.')

    for index in range(num_indicators):
        norm_one_random_weight = rand_weight_per_indicator.get("indicator_{}".format(index + 1), [])
        args_for_parallel_agg = [(lst, indicators) for lst in norm_one_random_weight]
        if is_sensitivity == "yes":
            scores_one_random_weight = utils_for_parallelization.parallelize_aggregation(args_for_parallel_agg)
        else:
            scores_one_random_weight = utils_for_parallelization.parallelize_aggregation(args_for_parallel_agg, f_agg)

        scores_one_random_weight_normalized["indicator_{}".format(index + 1)] = []
        for matrix in scores_one_random_weight:
            matrix_normalized = rescale_minmax(matrix)  # normalize scores
            scores_one_random_weight_normalized["indicator_{}".format(index + 1)].append(matrix_normalized)

        one_random_weight_score_means, one_random_weight_score_stds = \
            utils_for_parallelization.estimate_runs_mean_std(scores_one_random_weight)
        one_random_weight_score_means_normalized, one_random_weight_score_stds_normalized = \
            utils_for_parallelization.estimate_runs_mean_std(
                scores_one_random_weight_normalized["indicator_{}".format(index + 1)])

        one_random_weight_score_means.insert(0, index_column_name, index_column_values)
        one_random_weight_score_stds.insert(0, index_column_name, index_column_values)
        one_random_weight_score_means_normalized.insert(0, index_column_name, index_column_values)
        one_random_weight_score_stds_normalized.insert(0, index_column_name, index_column_values)

        iterative_random_w_score_means["indicator_{}".format(index + 1)] = one_random_weight_score_means
        iterative_random_w_score_stds["indicator_{}".format(index + 1)] = one_random_weight_score_stds
        iterative_random_w_score_means_normalized[
            "indicator_{}".format(index + 1)] = one_random_weight_score_means_normalized
        iterative_random_w_score_stds_normalized[
            "indicator_{}".format(index + 1)] = one_random_weight_score_stds_normalized

    return {
        "score_means": iterative_random_w_score_means,
        "score_stds": iterative_random_w_score_stds,
        "score_means_normalized": iterative_random_w_score_means_normalized,
        "score_stds_normalized": iterative_random_w_score_stds_normalized
    }


def _compute_ranks(scores: Optional[pd.DataFrame], index_column_name: str, index_column_values: list,
                   all_weights_means: Optional[pd.DataFrame], iterative_random_w_means: Optional[dict]) -> pd.DataFrame:
    """
    Compute ranks based on the computed scores, mean scores with random weights, and mean scores for each random weight.
    """
    ranks = pd.DataFrame

    if not scores.empty:
        ranks = scores.rank(pct=True)
    elif not all_weights_means.empty:
        ranks = all_weights_means.rank(pct=True)
    elif not bool(iterative_random_w_means) is False:
        pass

    return ranks


def _save_output_files(scores: Optional[pd.DataFrame],
                       normalized_scores: Optional[pd.DataFrame],
                       ranks: Optional[pd.DataFrame],
                       score_means: Optional[pd.DataFrame],
                       score_stds: Optional[pd.DataFrame],
                       score_means_normalized: Optional[pd.DataFrame],
                       iterative_random_w_score_means: Optional[dict],
                       iterative_random_w_score_stds: Optional[pd.DataFrame],
                       iterative_random_w_score_means_normalized: Optional[dict],
                       config: Configuration,
                       index_column_name: str,
                       index_column_values: list) -> dict:
    """
    Save output files based of the computed scores, ranks, and configuration data.
    """
    output_filepath = "toy_example"
    # full_output_path = os.path.join(output_directory_path, output_filepath )
    logger.info("Saving results in {}".format(output_filepath))
    # check_path_exists(output_filepath)
    response = {}
    if scores is not None and not scores.empty:
        scores.insert(0, index_column_name, index_column_values)
        normalized_scores.insert(0, index_column_name, index_column_values)
        ranks.insert(0, index_column_name, index_column_values)
        scores_response = save_df(scores, output_filepath, 'scores.csv')
        normalized_response = save_df(normalized_scores, output_filepath, 'normalized_scores.csv')
        ranks_response = save_df(ranks, output_filepath, 'ranks.csv')
        response = {
            "rawScores": scores_response,
            "normalizedScores": normalized_response,
            "ranks": ranks_response
        }

    elif score_means is not None and not score_means.empty:
        score_means.insert(0, index_column_name, index_column_values)
        score_stds.insert(0, index_column_name, index_column_values)
        score_means_normalized.insert(0, index_column_name, index_column_values)

        raw_scores_averages = save_df(score_means, output_filepath, 'score_means.csv')
        raw_scores_standard_deviations = save_df(score_stds, output_filepath, 'score_stds.csv')
        normalized_scores_averages = save_df(score_means_normalized, output_filepath, 'score_means_normalized.csv')
        response = {
            "rawScoresAverages": raw_scores_averages,
            "rawScoresStandardDeviations": raw_scores_standard_deviations,
            "normalizedScoresAverages": normalized_scores_averages
        }
    elif iterative_random_w_score_means is not None:
        raw_scores_averages = save_dict(iterative_random_w_score_means, output_filepath, 'score_means.pkl')
        raw_scores_standard_deviations = save_dict(iterative_random_w_score_stds.to_dict(), output_filepath,
                                                   'score_stds.pkl')
        normalized_scores_averages = save_dict(iterative_random_w_score_means_normalized, output_filepath,
                                               'score_means_normalized.pkl')
        response = {
            "rawScoresAverages": raw_scores_averages,
            "rawScoresStandardDeviations": raw_scores_standard_deviations,
            "normalizedScoresAverages": normalized_scores_averages
        }
    save_config(config, output_filepath, 'configuration.json')

    return response


def _plot_and_save_charts(scores: Optional[pd.DataFrame],
                          normalized_scores: Optional[pd.DataFrame],
                          score_means: Optional[pd.DataFrame],
                          score_stds: Optional[pd.DataFrame],
                          score_means_normalized: Optional[pd.DataFrame],
                          iterative_random_w_score_means: Optional[dict],
                          iterative_random_w_score_stds: Optional[dict],
                          iterative_random_w_score_means_normalized: Optional[dict],
                          input_matrix: pd.DataFrame,
                          config: Configuration,
                          is_robustness_weights=None,
                          is_robustness_indicators=None) -> None:
    """
    Generate plots based on the computed scores and save them.
    """
    num_indicators = input_matrix.shape[1]
    output_file_path = "toy_example"
    if scores is not None and not scores.empty:
        plot_no_norm_scores = utils_for_plotting.plot_non_norm_scores_without_uncert(scores)
        utils_for_plotting.save_figure(plot_no_norm_scores, output_file_path, "MCDA_rough_scores.png")

        plot_norm_scores = utils_for_plotting.plot_norm_scores_without_uncert(normalized_scores)
        utils_for_plotting.save_figure(plot_norm_scores, output_file_path, "MCDA_norm_scores.png")

    elif score_means is not None and not score_means.empty:
        if is_robustness_weights is not None and is_robustness_weights == 1:
            chart_mean_scores = utils_for_plotting.plot_mean_scores(score_means, "plot_std", "weights", score_stds)
            chart_mean_scores_norm = utils_for_plotting.plot_mean_scores(score_means_normalized, "not_plot_std",
                                                                         "weights", score_stds)
        elif is_robustness_indicators is not None and is_robustness_indicators == 1:
            chart_mean_scores = utils_for_plotting.plot_mean_scores(score_means, "plot_std", "indicators", score_stds)
            chart_mean_scores_norm = utils_for_plotting.plot_mean_scores(score_means_normalized, "not_plot_std",
                                                                         "indicators", score_stds)

        utils_for_plotting.save_figure(chart_mean_scores, output_file_path, "MCDA_rough_scores.png")
        utils_for_plotting.save_figure(chart_mean_scores_norm, output_file_path, "MCDA_norm_scores.png")

    elif iterative_random_w_score_means is not None:
        images = []
        images_norm = []

        for index in range(num_indicators):
            one_random_weight_means = iterative_random_w_score_means["indicator_{}".format(index + 1)]
            one_random_weight_stds = iterative_random_w_score_stds["indicator_{}".format(index + 1)]
            one_random_weight_means_normalized = iterative_random_w_score_means_normalized[
                "indicator_{}".format(index + 1)]

            plot_weight_mean_scores = utils_for_plotting.plot_mean_scores_iterative(one_random_weight_means,
                                                                                    input_matrix.columns, index,
                                                                                    "plot_std", one_random_weight_stds)
            plot_weight_mean_scores_norm = utils_for_plotting.plot_mean_scores_iterative(
                one_random_weight_means_normalized, input_matrix.columns, index, "not_plot_std")

            images.append(plot_weight_mean_scores)
            images_norm.append(plot_weight_mean_scores_norm)

        utils_for_plotting.combine_images(images, output_file_path, "MCDA_one_weight_randomness_rough_scores.png")
        utils_for_plotting.combine_images(images_norm, output_file_path, "MCDA_one_weight_randomness_norm_scores.png")
