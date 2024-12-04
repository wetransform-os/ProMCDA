import os
import argparse
import json
import pickle
import random
import logging
import sys
from enum import Enum
from typing import Union, Any, List, Tuple
from typing import Optional

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import mcda.utils.utils_for_parallelization as utils_for_parallelization
import mcda.utils.utils_for_plotting as utils_for_plotting
from mcda.configuration.enums import PDFType
from mcda.models.mcda_without_robustness import MCDAWithoutRobustness
from mcda.models.mcda_with_robustness import MCDAWithRobustness

DEFAULT_INPUT_DIRECTORY_PATH = './input_files'  # present in the root directory of ProMCDA
DEFAULT_OUTPUT_DIRECTORY_PATH = './output_files'  # present in the root directory of ProMCDA

CUSTOM_INPUT_PATH = os.environ.get('PROMCDA_INPUT_DIRECTORY_PATH')  # check if an environmental variable is set
CUSTOM_OUTPUT_PATH = os.environ.get('PROMCDA_OUTPUT_DIRECTORY_PATH')  # check if an environmental variable is set

input_directory_path = CUSTOM_INPUT_PATH if CUSTOM_INPUT_PATH else DEFAULT_INPUT_DIRECTORY_PATH
output_directory_path = CUSTOM_OUTPUT_PATH if CUSTOM_OUTPUT_PATH else DEFAULT_OUTPUT_DIRECTORY_PATH

log = logging.getLogger(__name__)

logging.getLogger('PIL').setLevel(logging.WARNING)

FORMATTER: str = '%(levelname)s: %(asctime)s - %(name)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMATTER)
logger = logging.getLogger("ProMCDA")


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


# TODO: maybe give the option of giving either a pd.DataFrame or a path as input parameter in ProMCDA
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


def save_df(df: pd.DataFrame, folder_path: str, filename: str):
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
    except IOError as e:
        logging.error(f"Error while writing data frame into a CSV file: {e}")


def save_dict(dictionary: dict, folder_path: str, filename: str):
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


def preprocess_enums(data) -> Union[Union[dict, list[str]], Any]:
    """
    Preprocess data to convert enums to strings

    Parameters:
    - data: to be processed

    Example:
    ```python
    preprocess_enums(data)
    ```
    :param data: enums
    :return: string
    """
    if isinstance(data, dict):
        return {k: preprocess_enums(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [preprocess_enums(v) for v in data]
    elif isinstance(data, Enum):
        return data.value
    return data


def save_config(config: dict, folder_path: str, filename: str):
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
            processed_config = preprocess_enums(config)
            serializable_config = _prepare_config_for_json(processed_config)
            json.dump(serializable_config, fp)
    except IOError as e:
        logging.error(f"Error while dumping the configuration into a JSON file: {e}")


def _convert_dataframe_to_serializable(df):
    """
    Convert a pandas DataFrame into a serializable dictionary format.
    """
    return {
        'data': df.values.tolist(),  # Convert data to list of lists
        'columns': df.columns.tolist(),  # Convert column names to list
        'index': df.index.tolist()  # Convert index labels to list
    }


def _prepare_config_for_json(config):
    """
    Prepare the config dictionary by converting non-serializable objects into serializable ones.
    """
    config_copy = config.copy()  # Create a copy to avoid modifying the original config
    if isinstance(config_copy['input_matrix'], pd.DataFrame):
        # Convert DataFrame to serializable format
        config_copy['input_matrix'] = _convert_dataframe_to_serializable(config_copy['input_matrix'])
    return config_copy


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


def check_parameters_pdf(input_matrix: pd.DataFrame, marginal_distributions: Tuple[PDFType, ...], for_testing=False) \
        -> Union[List[bool], None]:
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
    - marginal_distributions: the PDFs associated to each indicator.
    - for_testing: true only for unit testing

    Returns:
    - List: a list indicating whether the conditions are satisfied for each indicator (only for unit tests).
    - None: default

    :param input_matrix: pd.DataFrame
    :param marginal_distributions: PDFType
    :param for_testing: bool
    :return: Union[list, None]
    """

    satisfies_condition = False
    problem_logged = False

    marginal_pdf = marginal_distributions
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


def check_if_pdf_is_exact(marginal_pdf: tuple[PDFType, ...]) -> list:
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


def check_if_pdf_is_poisson(marginal_pdf: tuple[PDFType, ...]) -> list:
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


def run_mcda_without_indicator_uncertainty(extracted_values: dict, is_robustness_weights: int,
                                           weights: Union[List[str], List[pd.DataFrame], dict, None]):
    """
    Runs ProMCDA without uncertainty on the indicators, i.e. without performing
    a robustness analysis.

    This function executes the MCDA process without considering uncertainty
    in the indicators. It computes scores and weights, saves the results,
    and logs the completion time.

    Parameters:
    - extracted_values: a dictionary containing configuration values extracted from the input parameters.
    - is_robustness_weights: a flag indicating whether robustness analysis will be performed on indicators or not.
    - weights: the normalised weights (either fixed or random sampled weights, depending on the settings).

    :param weights: Union[List[str], List[pd.DataFrame], dict, None]
    :param extracted_values: dict
    :param is_robustness_weights: int
    :return: None
    """
    scores = pd.DataFrame
    normalized_scores = pd.DataFrame
    all_weights_score_means = pd.DataFrame
    all_weights_score_stds = pd.DataFrame
    all_weights_score_means_normalized = pd.DataFrame
    iterative_random_w_score_means_normalized = {}
    iterative_random_weights_statistics = {}
    iterative_random_w_score_means = {}
    iterative_random_w_score_stds = {}

    # Extract relevant values
    input_matrix = extracted_values["input_matrix"]
    index_column_name = input_matrix.index.name
    index_column_values = input_matrix.index.tolist()
    input_matrix_no_alternatives = check_input_matrix(input_matrix)
    is_sensitivity = extracted_values['sensitivity_on']
    is_robustness = extracted_values['robustness_on']
    f_norm = extracted_values["normalization"]
    f_agg = extracted_values["aggregation"]

    mcda_no_uncert \
        = MCDAWithoutRobustness(extracted_values, input_matrix_no_alternatives)
    logger.info("Start ProMCDA without robustness of the indicators")

    normalized_indicators = mcda_no_uncert.normalize_indicators() if is_sensitivity == "yes" \
        else mcda_no_uncert.normalize_indicators(f_norm)

    if is_robustness == "no":
        scores = mcda_no_uncert.aggregate_indicators(normalized_indicators, weights) \
            if is_sensitivity == "yes" \
            else mcda_no_uncert.aggregate_indicators(normalized_indicators, weights, f_agg)
        normalized_scores = rescale_minmax(scores)
    elif extracted_values["on_all_weights"] == "yes" and extracted_values["robustness_on"] == "yes":
        # ALL RANDOMLY SAMPLED WEIGHTS (MCDA runs num_samples times)
        all_weights_score_means, all_weights_score_stds, \
            all_weights_score_means_normalized, all_weights_score_stds_normalized = \
            _compute_scores_for_all_random_weights(normalized_indicators, is_sensitivity, weights, f_agg)
    elif (extracted_values["on_single_weights"] == "yes") and (extracted_values["robustness_on"] == "yes"):
        # ONE RANDOMLY SAMPLED WEIGHT A TIME (MCDA runs (num_samples * num_indicators) times)
        iterative_random_weights_statistics: dict = _compute_scores_for_single_random_weight(
            normalized_indicators, weights, is_sensitivity, index_column_name, index_column_values, f_agg,
            input_matrix_no_alternatives)
        iterative_random_w_score_means = iterative_random_weights_statistics['score_means']
        iterative_random_w_score_stds = iterative_random_weights_statistics['score_stds']
        iterative_random_w_score_means_normalized = iterative_random_weights_statistics['score_means_normalized']

    ranks = _compute_ranks(scores, index_column_name, index_column_values,
                           all_weights_score_means, iterative_random_w_score_means)

    _save_output_files(scores=scores, normalized_scores=normalized_scores, ranks=ranks,
                       score_means=all_weights_score_means, score_stds=all_weights_score_stds,
                       score_means_normalized=all_weights_score_means_normalized,
                       iterative_random_w_score_means=iterative_random_w_score_means,
                       iterative_random_w_score_means_normalized=iterative_random_w_score_means_normalized,
                       iterative_random_w_score_stds=iterative_random_w_score_stds,
                       index_column_name=index_column_name, index_column_values=index_column_values,
                       input_config=extracted_values)

    _plot_and_save_charts(scores=scores, normalized_scores=normalized_scores,
                          score_means=all_weights_score_means, score_stds=all_weights_score_stds,
                          score_means_normalized=all_weights_score_means_normalized,
                          iterative_random_w_score_means=iterative_random_w_score_means,
                          iterative_random_w_score_stds=iterative_random_w_score_stds,
                          iterative_random_w_score_means_normalized=iterative_random_w_score_means_normalized,
                          input_matrix=input_matrix_no_alternatives, config=extracted_values,
                          is_robustness_weights=is_robustness_weights)


def run_mcda_with_indicator_uncertainty(extracted_values: dict, weights: Union[List[str], List[pd.DataFrame],
dict, None]) -> None:
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

    :param extracted_values: dict
    :param weights: Union[List[str], List[pd.DataFrame], dict, None]
    :return: None
    """
    logger.info("Start ProMCDA with uncertainty on the indicators")
    is_robustness_indicators = True
    all_indicators_scores_normalized = []

    # Extract relevant values
    input_matrix = extracted_values["input_matrix"]
    alternatives_column_name = input_matrix.columns[0]
    input_matrix = input_matrix.set_index(alternatives_column_name)
    index_column_name = input_matrix.index.name
    index_column_values = input_matrix.index.tolist()
    input_matrix_no_alternatives = check_input_matrix(input_matrix)
    mc_runs = extracted_values["monte_carlo_runs"]
    marginal_pdf = extracted_values["marginal_distribution_for_each_indicator"]
    random_seed = extracted_values["random_seed"]
    is_sensitivity = extracted_values['sensitivity_on']
    f_norm = extracted_values["normalization"]
    f_agg = extracted_values["aggregation"]
    polar = extracted_values["polarity_for_each_indicator"]

    if mc_runs <= 0:
        logger.error('Error Message', stack_info=True)
        raise ValueError('The number of MC runs should be larger than 0 for a robustness analysis')

    if mc_runs < 1000:
        logger.info("The number of Monte-Carlo runs is only {}".format(mc_runs))
        logger.info("A meaningful number of Monte-Carlo runs is equal or larger than 1000")

    check_parameters_pdf(input_matrix, extracted_values)
    is_exact_pdf_mask = check_if_pdf_is_exact(marginal_pdf)
    is_poisson_pdf_mask = check_if_pdf_is_poisson(marginal_pdf)

    mcda_with_uncert = MCDAWithRobustness(extracted_values, input_matrix_no_alternatives, is_exact_pdf_mask,
                                          is_poisson_pdf_mask, random_seed)
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

    _save_output_files(scores=None, normalized_scores=None,
                       ranks=ranks,
                       score_means=all_indicators_scores_means,
                       score_stds=all_indicators_scores_stds,
                       score_means_normalized=all_indicators_means_scores_normalized,
                       iterative_random_w_score_means=None,
                       iterative_random_w_score_means_normalized=None,
                       iterative_random_w_score_stds=None,
                       input_config=extracted_values,
                       index_column_name=index_column_name, index_column_values=index_column_values)

    _plot_and_save_charts(scores=None, normalized_scores=None,
                          score_means=all_indicators_scores_means, score_stds=all_indicators_scores_stds,
                          score_means_normalized=all_indicators_means_scores_normalized,
                          iterative_random_w_score_means=None,
                          iterative_random_w_score_stds=None,
                          iterative_random_w_score_means_normalized=None,
                          input_matrix=input_matrix, config=extracted_values,
                          is_robustness_indicators=is_robustness_indicators)


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


def _compute_scores_for_all_random_weights(indicators: pd.DataFrame, is_sensitivity: str,
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


def _compute_scores_for_single_random_weight(indicators: pd.DataFrame,
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
                       input_config: dict,
                       index_column_name: str,
                       index_column_values: list) -> None:
    """
    Save output files based of the computed scores, ranks, and configuration data.
    """
    output_path = input_config["output_path"]
    full_output_path = os.path.join(output_directory_path, output_path)
    logger.info("Saving results in {}".format(full_output_path))
    check_path_exists(output_path)

    if scores is not None and not scores.empty:
        scores.insert(0, index_column_name, index_column_values)
        normalized_scores.insert(0, index_column_name, index_column_values)
        ranks.insert(0, index_column_name, index_column_values)

        save_df(scores, output_path, 'scores.csv')
        save_df(normalized_scores, output_path, 'normalized_scores.csv')
        save_df(ranks, output_path, 'ranks.csv')
    elif score_means is not None and not score_means.empty:
        score_means.insert(0, index_column_name, index_column_values)
        score_stds.insert(0, index_column_name, index_column_values)
        score_means_normalized.insert(0, index_column_name, index_column_values)

        save_df(score_means, output_path, 'score_means.csv')
        save_df(score_stds, output_path, 'score_stds.csv')
        save_df(score_means_normalized, output_path, 'score_means_normalized.csv')
    elif iterative_random_w_score_means is not None:
        save_dict(iterative_random_w_score_means, output_path, 'score_means.pkl')
        save_dict(iterative_random_w_score_stds, output_path, 'score_stds.pkl')
        save_dict(iterative_random_w_score_means_normalized, output_path, 'score_means_normalized.pkl')

    save_config(input_config, output_path, 'configuration.json')


def _plot_and_save_charts(scores: Optional[pd.DataFrame],
                          normalized_scores: Optional[pd.DataFrame],
                          score_means: Optional[pd.DataFrame],
                          score_stds: Optional[pd.DataFrame],
                          score_means_normalized: Optional[pd.DataFrame],
                          iterative_random_w_score_means: Optional[dict],
                          iterative_random_w_score_stds: Optional[dict],
                          iterative_random_w_score_means_normalized: Optional[dict],
                          input_matrix: pd.DataFrame,
                          config: dict,
                          is_robustness_weights=None,
                          is_robustness_indicators=None) -> None:
    """
    Generate plots based on the computed scores and save them.
    """
    output_path = config["output_path"]
    num_indicators = input_matrix.shape[1]

    if scores is not None and not scores.empty:
        plot_no_norm_scores = utils_for_plotting.plot_non_norm_scores_without_uncert(scores)
        utils_for_plotting.save_figure(plot_no_norm_scores, output_path, "MCDA_rough_scores.png")

        plot_norm_scores = utils_for_plotting.plot_norm_scores_without_uncert(normalized_scores)
        utils_for_plotting.save_figure(plot_norm_scores, output_path, "MCDA_norm_scores.png")

    elif score_means is not None and not score_means.empty:
        if is_robustness_weights is not None and is_robustness_weights == 1:
            chart_mean_scores = utils_for_plotting.plot_mean_scores(score_means, "plot_std", "weights", score_stds)
            chart_mean_scores_norm = utils_for_plotting.plot_mean_scores(score_means_normalized, "not_plot_std",
                                                                         "weights", score_stds)
        elif is_robustness_indicators is not None and is_robustness_indicators == 1:
            chart_mean_scores = utils_for_plotting.plot_mean_scores(score_means, "plot_std", "indicators", score_stds)
            chart_mean_scores_norm = utils_for_plotting.plot_mean_scores(score_means_normalized, "not_plot_std",
                                                                         "indicators", score_stds)

        utils_for_plotting.save_figure(chart_mean_scores, output_path, "MCDA_rough_scores.png")
        utils_for_plotting.save_figure(chart_mean_scores_norm, output_path, "MCDA_norm_scores.png")

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

        utils_for_plotting.combine_images(images, output_path,
                                          "MCDA_one_weight_randomness_rough_scores.png")
        utils_for_plotting.combine_images(images_norm, output_path,
                                          "MCDA_one_weight_randomness_norm_scores.png")
