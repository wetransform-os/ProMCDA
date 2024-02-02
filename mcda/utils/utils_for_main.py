import argparse
import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

logging.getLogger('PIL').setLevel(logging.WARNING)

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")


def check_config_error(condition: bool, error_message: str):
    """
    Check a condition and raise a ValueError with a specified error message if the condition is True.

    Parameters:
    - condition (bool): The condition to check.
    - error_message (str): The error message to raise if the condition is True.

    Raises:
    - ValueError: If the condition is True, with the specified error message.

    :return: None
    :param condition: bool
    """

    if condition:
        logger.error('Error Message', stack_info=True)
        raise ValueError(error_message)


def check_config_setting(condition, information_message):
    if condition:
        logger.info(information_message)


def process_indicators_and_weights(config: dict, input_matrix_no_alternatives: pd.DataFrame,
                                   is_robustness_indicators: int, is_robustness_weights: int, polar: List[str],
                                   mc_runs: int, num_indicators: int):
    """
    Process indicators and weights based on input parameters in the configuration.

    Parameters:
    - input_matrix_no_alternatives (DataFrame): The input matrix.
    - is_robustness_indicators (int): Flag indicating whether the matrix should include indicator uncertainties
      (0 or 1).
    - is_robustness_weights (int): Flag indicating whether robustness analysis is considered for the weights
      (0 or 1).
    - marginal_pdf (list): List of marginal probability density functions for indicators.
    - mc_runs (int): Number of Monte Carlo runs for robustness analysis.
    - num_indicators: the number of indicators in the input matrix.
    - config: the configuration dictionary.

    Raises:
    - ValueError: If there are duplicated rows in the input matrix or if there is an issue with the configuration.

    Notes:
    - For is_robustness_indicators == 0:
    - Identifies and removes columns with constant values.
    - Logs the number of alternatives and indicators.

    - For is_robustness_indicators == 1:
    - Handles uncertainty in indicators.
    - Logs the number of alternatives and indicators.

    - For is_robustness_weights == 0:
    - Processes fixed weights if given.
    - Logs weights and normalized weights.

    - For is_robustness_weights == 1:
    - Performs robustness analysis on weights.
    - Logs randomly sampled weights.

    :return: None
    :param mc_runs: int
    :param polar: List[str]
    :param is_robustness_weights: int
    :param is_robustness_indicators: int
    :param input_matrix_no_alternatives: pd.DataFrame
    :param config: dict
    :param num_indicators: int
    """

    num_unique = input_matrix_no_alternatives.nunique()
    cols_to_drop = num_unique[num_unique == 1].index
    col_to_drop_indexes = input_matrix_no_alternatives.columns.get_indexer(cols_to_drop)

    if is_robustness_indicators == 0:
        _handle_no_robustness_indicators(input_matrix_no_alternatives)
    else:  # matrix with uncertainty on indicators
        logger.info("Number of alternatives: {}".format(input_matrix_no_alternatives.shape[0]))
        logger.info("Number of indicators: {}".format(num_indicators))
        # TODO: eliminate indicators with constant values (i.e. same mean and 0 std) - optional

    _handle_polarities_and_weights(is_robustness_indicators, is_robustness_weights, num_unique, col_to_drop_indexes,
                                   polar, config, mc_runs, num_indicators)


def _handle_polarities_and_weights(is_robustness_indicators: int, is_robustness_weights: int, num_unique,
                                   col_to_drop_indexes: np.ndarray, polar: List[str], config: dict, mc_runs: int,
                                   num_indicators: int):
    if is_robustness_indicators == 0:
        if any(value == 1 for value in num_unique):
            polar = pop_indexed_elements(col_to_drop_indexes, polar)
    logger.info("Polarities: {}".format(polar))

    if is_robustness_weights == 0:
        fixed_weights = config.robustness["given_weights"]
        if any(value == 1 for value in num_unique):
            fixed_weights = pop_indexed_elements(col_to_drop_indexes, fixed_weights)
        norm_fixed_weights = check_norm_sum_weights(fixed_weights)
        logger.info("Weights: {}".format(fixed_weights))
        logger.info("Normalized weights: {}".format(norm_fixed_weights))
    else:
        _handle_robustness_weights(config, mc_runs, num_indicators)


def _handle_robustness_weights(config: dict, mc_runs: int, num_indicators: int):
    norm_random_weights = []
    if mc_runs == 0:
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')

    if config.robustness["on_single_weights"] == "no" and config.robustness["on_all_weights"] == "yes":
        random_weights = randomly_sample_all_weights(num_indicators, mc_runs)
        for weights in random_weights:
            weights = check_norm_sum_weights(weights)
            norm_random_weights.append(weights)
    elif config.robustness["on_single_weights"] == "yes" and config.robustness["on_all_weights"] == "no":
        i = 0
        rand_weight_per_indicator = {}
        while i < num_indicators:
            random_weights = randomly_sample_ix_weight(num_indicators, i, mc_runs)
            norm_random_weight = []
            for weights in random_weights:
                weights = check_norm_sum_weights(weights)
                norm_random_weight.append(weights)
            rand_weight_per_indicator["indicator_{}".format(i + 1)] = norm_random_weight
            i += 1


def _handle_no_robustness_indicators(input_matrix_no_alternatives: pd.DataFrame):
    num_unique = input_matrix_no_alternatives.nunique()
    cols_to_drop = num_unique[num_unique == 1].index

    if any(value == 1 for value in num_unique):
        logger.info("Indicators {} have been dropped because they carry no information".format(cols_to_drop))
        input_matrix_no_alternatives = input_matrix_no_alternatives.drop(cols_to_drop, axis=1)

    num_indicators = input_matrix_no_alternatives.shape[1]
    logger.info("Number of alternatives: {}".format(input_matrix_no_alternatives.shape[0]))
    logger.info("Number of indicators: {}".format(num_indicators))


def check_indicator_weights_polarities(num_indicators: int, polar: List[str], config: dict):
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
    - ValueError: If the conditions for indicator-polarity and fixed weights consistency are not met.
        
    :return: None
    :param num_indicators: int
    :param polar: List[str]
    :param config: dict
    """

    if num_indicators != len(polar):
        raise ValueError('The number of polarities does not correspond to the no. of indicators')

    # Check the number of fixed weights if "on_all_weights" is set to "no"
    if (config.robustness["on_all_weights"] == "no") and (
            num_indicators != len(config.robustness["given_weights"])):
        raise ValueError('The no. of fixed weights does not correspond to the no. of indicators')


def check_input_matrix(input_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Check the input matrix for duplicated rows in the alternatives column
    and rescale negative indicator values.

    Parameters:
    - input_matrix: The input matrix containing alternatives and indicators.

    Raises:
    - ValueError: If duplicated rows are found in the alternative column.
    - UserStoppedInfo: If the user chooses to stop when duplicates are found.

     :rtype: pd.DataFrame
     :param input_matrix: pd.DataFrame
    """

    if input_matrix.duplicated().any():
        raise ValueError('Error: Duplicated rows in the alternatives column.')
    elif input_matrix.iloc[:, 0].duplicated().any():
        logger.info('Duplicated rows in the alternatives column.')
    logger.info("Alternatives are {}".format(input_matrix.iloc[:, 0].tolist()))
    input_matrix_no_alternatives = input_matrix.drop(
        input_matrix.columns[0], axis=1)  # drop the first column with alternatives
    input_matrix_no_alternatives = _check_and_rescale_negative_indicators(
        input_matrix_no_alternatives)

    return input_matrix_no_alternatives


def read_matrix(input_matrix_path: str) -> pd.DataFrame():
    """
    Read an input matrix from a CSV file and return it as a DataFrame.

    Parameters:
    - input_matrix_path (str): Path to the CSV file containing the input matrix.

    Raises:
    - Exception: If an error occurs during the file reading or DataFrame creation.

    :rtype: pd.DataFrame
    :param input_matrix_path: str
    """

    try:
        filename = input_matrix_path
        with open(filename, 'r') as fp:
            matrix = pd.read_csv(fp, sep="[,;:]", decimal='.', engine='python')
            data_types = {col: 'float64' for col in matrix.columns[1:]}
            matrix = matrix.astype(data_types)
            return matrix
    except Exception as e:
        print(e)


def _check_and_rescale_negative_indicators(input_matrix: pd.DataFrame) -> pd.DataFrame():
    """If some indicators in the input matrix are negative, they are rescaled into [0-1]"""

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

    :rtype: dict
    :param config_path: str
    """

    try:
        with open(config_path, 'r') as fp:
            return json.load(fp)
    except Exception as e:
        print(e)


def save_df(df: pd.DataFrame, folder_path: str, filename: str):
    """
    Save a DataFrame to a CSV file with a timestamped filename.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be saved.
    - folder_path (str): The path to the folder where the file will be saved.
    - filename (str): The original filename for the CSV file.

    Notes:
    - The saved file will have a timestamp added to its filename.

    Example:
    ```python
    save_df(my_dataframe, '/path/to/folder', 'data.csv')
    ```

    :return: None
    :param df: pd.DataFrame
    :param folder_path: str
    :param filename: str
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    result_path = os.path.join(folder_path, new_filename)
    df.to_csv(path_or_buf=result_path, index=False)


def save_dict(dictionary: dict, folder_path: str, filename: str):
    """
    Save a dictionary to a binary file using pickle with a timestamped filename.

    Parameters:
    - dictionary (dict): The dictionary to be saved.
    - folder_path (str): The path to the folder where the file will be saved.
    - filename (str): The original filename for the binary file.

    Example:
    ```python
    save_dict(my_dict, '/path/to/folder', 'data.pkl')
    ```

    :return: None
    :param dictionary: dict
    :param folder_path: str
    :param filename: str
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"
    result_path = os.path.join(folder_path, new_filename)

    with open(result_path, 'wb') as fp:
        pickle.dump(dictionary, fp)


def save_config(config: dict, folder_path: str, filename: str):
    """
    Save a configuration dictionary to a JSON file with a timestamped filename.

    Parameters:
    - config (dict): The configuration dictionary to be saved.
    - folder_path (str): The path to the folder where the file will be saved.
    - filename (str): The original filename for the JSON file.

    Example:
    ```python
    save_config(my_config, '/path/to/folder', 'config.json')
    ```
    :return: None
    :param config: dict
    :param folder_path: str
    :param filename: str
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    result_path = os.path.join(folder_path, new_filename)
    with open(result_path, 'w') as fp:
        json.dump(config, fp)


def check_path_exists(path: str):
    """
    Check if a directory path exists, and create it if it doesn't.

    Example:
    ```python
    check_path_exists('/path/to/directory')
    ```
    :return: None
    :param path: str
    """
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
        print("The new output directory is created: {}".format(path))


def rescale_minmax(scores: pd.DataFrame) -> pd.DataFrame():
    """
    Rescale the values in a DataFrame to a [0, 1] range using Min-Max scaling.

    Parameters:
    - scores (pd.DataFrame): The DataFrame containing numerical values to be rescaled.

    Example:
    ```python
    rescaled_df = rescale_minmax(my_dataframe)
    ```
    This will rescale the values in the input DataFrame to a [0, 1] range.

    :rtype: pd.DataFrame()
    :param path: str
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
    - num_weights (int): The number of weights in each list.
    - num_runs (int): The number of lists to be generated.

    Returns:
    - List[list]: A list containing 'num_runs' lists, each with 'num_weights' randomly sampled elements.

    Example:
    ``python
    sampled_weights = randomly_sample_all_weights(5, 10)
    ```
    This will generate 10 lists, each containing 5 randomly sampled weights.

    :rtype: List[list]
    :param num_weights: int
    """

    list_of_weights = []
    for _ in range(num_runs):
        lst = [random.uniform(0, 1) for _ in range(num_weights)]
        list_of_weights.append(lst)

    return list_of_weights


def randomly_sample_ix_weight(num_weights: int, index: int, num_runs: int) -> List[list]:
    """
    Generate multiple lists of weights with one randomly sampled element at a specified index.

    Parameters:
    - num_weights (int): The total number of weights in each list.
    - index (int): The index at which a weight will be randomly sampled in each list.
    - num_runs (int): The number of lists to be generated.

    Returns:
    - List[list]: A list containing 'num_runs' lists, each with 'num_weights' elements
      and one randomly sampled at the specified index.

    Example:
    ```python
    sampled_weights = randomly_sample_ix_weight(5, 2, 10)
    ```
    This will generate 10 lists, each containing 5 weights, with one randomly sampled at index 2.

    :rtype: List[list]
    :param num_weights: int
    :param index: int
    :param num_runs: int
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
    - list: The original weights if the sum is already 1, or normalized weights if the sum is not 1.

    Example:
    ```python
    normalized_weights = check_norm_sum_weights([0.2, 0.3, 0.5])
    ```

    :rtype: list
    :param weights: list
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
    - indexes (np.ndarray): An array of indexes indicating elements to be removed.
    - original_list (list): The original list from which elements will be removed.

    Example:
    ```python
    new_list = pop_indexed_elements(np.array([1, 3]), [10, 20, 30, 40, 50])
    ```
    This will remove elements at indexes 1 and 3 from the original list and return the new list [10, 30, 50].

    :rtype: list
    :param indexes: np.ndarray
    :param original_list: list
    """
    for i in range(len(indexes)):
        index = indexes[i]
        if i == 0:
            original_list.pop(index)
        else:
            original_list.pop(index - i)
    new_list = original_list

    return new_list


def check_parameters_pdf(input_matrix: pd.DataFrame, config: dict) -> List[bool]:
    """
    Check conditions on parameters based on the type of probability distribution function (PDF) for each indicator.

    The function checks if the values of parameter1 are larger or equal to the ones of parameter2
    (or vice-versa for the uniform PDF) in case of an input matrix with uncertainties.
    The check runs only in the case of indicators with a PDF of the type normal/lognormal, and uniform.
    In the first case, the parameters represent the (log)mean and (log)std;
    in the second case, they represent min and max values.

    Parameters:
    - input_matrix (pd.DataFrame): The input matrix containing uncertainties for indicators.
    - config (dict): Configuration dictionary containing the Monte Carlo sampling information.

    Returns:
    - List[bool]: A list indicating whether the conditions are satisfied for each indicator.

    Example:
    ```python
    conditions_satisfied = check_parameters_pdf(my_input_matrix, my_config)
    ```

    This will check conditions on parameters based on the PDF type
    for each indicator in the input matrix as described above.
    The result is a list of boolean values indicating whether the conditions are satisfied for each indicator.

    :rtype: list
    :param input_matrix: pd.DataFrame
    :param config: dict
    """

    satisfies_condition = False

    marginal_pdf = config.monte_carlo_sampling["marginal_distribution_for_each_indicator"]
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

    return list_of_satisfied_conditions


def check_if_pdf_is_exact(marginal_pdf: list) -> list:
    """
    Check if each indicator's probability distribution function (PDF) is of type 'exact'.

    Parameters:
    - marginal_pdf (list): A list containing the type of PDF for each indicator.

    Returns:
    - list: A binary mask indicating whether the PDF for each indicator is 'exact' (1) or not (0).

    Example:
    ```python
    is_exact_pdf = check_if_pdf_is_exact(['exact', 'normal', 'exact', 'uniform'])
    ```
    This will return a list [1, 0, 1, 0], indicating that the first and third indicators have 'exact' PDFs.

    :rtype: list
    :param marginal_pdf: list
    """
    exact_pdf_mask = [1 if pdf == 'exact' else 0 for pdf in marginal_pdf]

    return exact_pdf_mask


def check_if_pdf_is_poisson(marginal_pdf: list) -> list:
    """
    Check if each indicator's probability distribution function (PDF) is of type 'poisson'.

    Parameters:
    - marginal_pdf (list): A list containing the type of PDF for each indicator.

    Returns:
    - list: A binary mask indicating whether the PDF for each indicator is 'poisson' (1) or not (0).

    Example:
    ```python
    is_poisson_pdf = check_if_pdf_is_poisson(['poisson', 'normal', 'exact', 'uniform'])
    ```
    This will return a list [1, 0, 0, 0], indicating that the first indicator have a 'poisson' PDF.

    :rtype: list
    :param marginal_pdf: list
    """
    poisson_pdf_mask = [1 if pdf == 'poisson' else 0 for pdf in marginal_pdf]

    return poisson_pdf_mask


def check_if_pdf_is_uniform(marginal_pdf: list) -> list:
    """
    Check if each indicator's probability distribution function (PDF) is of type 'uniform'.

    Parameters:
    - marginal_pdf (list): A list containing the type of PDF for each indicator.

    Returns:
    - list: A binary mask indicating whether the PDF for each indicator is 'uniform' (1) or not (0).

    Example:
    ```python
    is_exact_pdf = check_if_pdf_is_exact(['exact', 'normal', 'exact', 'uniform'])
    ```
    This will return a list [0, 0, 0, 1], indicating that the fourth indicator have a 'uniform' PDF.

    :rtype: list
    :param marginal_pdf: list
    """
    uniform_pdf_mask = [1 if pdf == 'uniform' else 0 for pdf in marginal_pdf]

    return uniform_pdf_mask
