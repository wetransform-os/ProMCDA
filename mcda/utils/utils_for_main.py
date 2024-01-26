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


def check_config_error(condition, error_message):
    if condition:
        logger.error('Error Message', stack_info=True)
        raise ValueError(error_message)


def check_config_setting(condition, information_message):
    if condition:
        logger.info(information_message)


def read_matrix(input_matrix_path: str) -> pd.DataFrame():
    try:
        filename = input_matrix_path
        with open(filename, 'r') as fp:
            matrix = pd.read_csv(fp, sep="[,;:]", decimal='.', engine='python')
            data_types = {col: 'float64' for col in matrix.columns[1:]}
            matrix = matrix.astype(data_types)
            return matrix
    except Exception as e:
        print(e)


def check_and_rescale_negative_indicators(input_matrix: pd.DataFrame) -> pd.DataFrame():
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
    """parse input args"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config path', required=True)
    args = parser.parse_args()

    return args.config


def get_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as fp:
            return json.load(fp)
    except Exception as e:
        print(e)


def save_df(df: pd.DataFrame, folder_path: str, filename: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    result_path = os.path.join(folder_path, new_filename)
    df.to_csv(path_or_buf=result_path, index=False)


def save_dict(dictionary: dict, folder_path: str, filename: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"
    result_path = os.path.join(folder_path, new_filename)

    with open(result_path, 'wb') as fp:
        pickle.dump(dictionary, fp)


def save_config(config: dict, folder_path: str, filename: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    result_path = os.path.join(folder_path, new_filename)
    with open(result_path, 'w') as fp:
        json.dump(config, fp)


def check_path_exists(path: str):
    """check if path exists, if not create a new dir"""
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
        print("The new output directory is created: {}".format(path))


def rescale_minmax(scores: pd.DataFrame) -> pd.DataFrame():
    x = scores.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalized_scores = pd.DataFrame(x_scaled)
    normalized_scores.columns = scores.columns

    return normalized_scores


def randomly_sample_all_weights(num_weights: int, num_runs: int) -> List[list]:
    """ The function generates 'num_runs' lists of weights;
        Each list has 'num_weights' random sampled elements."""
    list_of_weights = []
    for _ in range(num_runs):
        lst = [random.uniform(0, 1) for _ in range(num_weights)]
        list_of_weights.append(lst)

    return list_of_weights


def randomly_sample_ix_weight(num_weights: int, index: int, num_runs: int) -> List[list]:
    """ The function generates 'num_runs' lists of weights;
        Each list has 'num_weights' elements,
        only one randomly sampled, at position index."""
    list_of_weights = []
    for _ in range(num_runs):
        lst = [1] * num_weights
        lst[index] = random.uniform(0, 1)
        list_of_weights.append(lst)

    return list_of_weights


def check_norm_sum_weights(weights: list) -> list:
    if sum(weights) != 1:
        norm_weights = [val / sum(weights) for val in weights]
        return norm_weights
    else:
        return weights


def pop_indexed_elements(indexes: np.ndarray, original_list: list) -> list:
    """ The function eliminates the values in a list corresponding to the given indexes"""
    for i in range(len(indexes)):
        index = indexes[i]
        if i == 0:
            original_list.pop(index)
        else:
            original_list.pop(index - i)
    new_list = original_list

    return new_list


def check_parameters_pdf(input_matrix: pd.DataFrame, config: dict) -> List[bool]:
    """ The function checks if the values of list parameter1 are larger or equal than the ones of list parameter2
        (or vice-versa for the uniform PDF) in case of an input matrix with uncertainties.
        The check runs only in the case of indicators with a PDF of the type normal/lognormal, and uniform.
        In the first case the parameters represent the (log)mean and (log)std;
        in the second case, they represent min and max values.
    """

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
    exact_pdf_mask = [1 if pdf == 'exact' else 0 for pdf in marginal_pdf]

    return exact_pdf_mask


def check_if_pdf_is_poisson(marginal_pdf: list) -> list:
    poisson_pdf_mask = [1 if pdf == 'poisson' else 0 for pdf in marginal_pdf]

    return poisson_pdf_mask


def check_if_pdf_is_uniform(marginal_pdf: list) -> list:
    uniform_pdf_mask = [1 if pdf == 'uniform' else 0 for pdf in marginal_pdf]

    return uniform_pdf_mask
