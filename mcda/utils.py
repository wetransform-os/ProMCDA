from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import plotly.graph_objects as go
from datetime import datetime
from typing import List
from typing import Tuple
import plotly.io as pio
from PIL import Image
import pandas as pd
import numpy as np
import logging
import argparse
import pickle
import random
import json
import os
import io

# logging.basicConfig(level=logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)  # suppress the debug messages produced by PIL internal logging


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
        scaled_matrix = pd.DataFrame(scaled_data, columns=input_matrix.columns, index=input_matrix.index)
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


def save_dict(dict: dict, folder_path: str, filename: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"
    result_path = os.path.join(folder_path, new_filename)

    with open(result_path, 'wb') as fp:
        pickle.dump(dict, fp)


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
                satisfies_condition = all(x >= y for x, y in zip(param1, param2))  # check param1 > param2 (mean > std=
            else:  # uniform
                satisfies_condition = all(x <= y for x, y in zip(param1, param2))  # check param2 > param1 (max > min)

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


def plot_norm_scores_without_uncert(scores: pd.DataFrame) -> object:
    num_of_combinations = scores.shape[1] - 1
    fig = go.Figure(layout_yaxis_range=[scores.iloc[:, 1:].values.min() - 0.5, scores.iloc[:, 1:].values.max() + 0.5],
                    layout_yaxis_title="MCDA normalized score")
    i = 0
    while i <= num_of_combinations - 1:
        fig.add_trace(go.Bar(
            name=scores.columns[i + 1],
            x=scores['Alternatives'].values.tolist(),
            y=scores.iloc[:, i + 1],
            showlegend=True
        ))
        i = i + 1
    fig.update_traces(showlegend=True)
    fig.update_layout(barmode='group', height=600, width=1000,
                      title='<b>MCDA analysis<b>',
                      title_font_size=22,

                      xaxis=dict(
                          tickmode="array",
                          tickvals=np.arange(0, len(scores['Alternatives'][:])),
                          ticktext=scores['Alternatives'][:],
                          tickangle=45),
                      yaxis=dict(
                          range=[scores.iloc[:, 1:].values.min() - 0.5, scores.iloc[:, 1:].values.max() + 0.5])
                      )
    fig.show()
    return fig


def plot_non_norm_scores_without_uncert(scores: pd.DataFrame) -> object:
    num_of_combinations = scores.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA rough score")
    i = 0
    while i <= num_of_combinations - 1:
        fig.add_trace(go.Bar(
            name=scores.columns[i + 1],
            x=scores['Alternatives'].values.tolist(),
            y=scores.iloc[:, i + 1],
            showlegend=True
        ))
        i = i + 1
    fig.update_traces(showlegend=True)
    fig.update_layout(barmode='group', height=600, width=1000,
                      title='<b>MCDA analysis<b>',
                      title_font_size=22,

                      xaxis=dict(
                          tickmode="array",
                          tickvals=np.arange(0, len(scores['Alternatives'][:])),
                          ticktext=scores['Alternatives'][:],
                          tickangle=45)
                      )
    fig.show()
    return fig


def plot_mean_scores(all_means: pd.DataFrame, all_stds: pd.DataFrame, plot_std: str, rand_on: str) -> object:
    num_of_combinations = all_means.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA average scores and std")
    i = 0
    while i <= num_of_combinations - 1:
        if plot_std == "plot_std":
            fig.add_trace(go.Bar(
                name=all_means.columns[i + 1],
                x=all_means['Alternatives'].values.tolist(),
                y=all_means.iloc[:, i + 1],
                error_y=dict(type='data', array=all_stds.iloc[:, i + 1])
            ))
            fig.update_layout(title=f'<b>MCDA analysis with added randomness on the {rand_on}<b> (rough scores)',
                              title_font_size=22)
        else:
            fig.add_trace(go.Bar(
                name=all_means.columns[i + 1],
                x=all_means['Alternatives'].values.tolist(),
                y=all_means.iloc[:, i + 1],
                showlegend=True
            ))
            fig.update_layout(title=f'<b>MCDA analysis with added randomness on the {rand_on}<b> (normalized scores)',
                              title_font_size=22)
        i = i + 1
    fig.update_traces(showlegend=True)
    fig.update_layout(barmode='group', height=600, width=1000,
                      # title=f'<b>MCDA analysis with added randomness on the {rand_on}<b>',
                      # title_font_size=22,

                      xaxis=dict(
                          tickmode="array",
                          tickvals=np.arange(0, len(all_means['Alternatives'][:])),
                          ticktext=all_means['Alternatives'][:],
                          tickangle=45)
                      )
    fig.show()
    return fig


def plot_mean_scores_iterative(all_weights_means: pd.DataFrame, all_weights_stds: pd.DataFrame, indicators: list,
                               index: int, plot_std: str) -> object:
    num_of_combinations = all_weights_means.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA average scores and std")
    i = 0
    while i <= num_of_combinations - 1:
        if plot_std == "plot_std":
            fig.add_trace(go.Bar(
                name=all_weights_means.columns[i + 1],
                x=all_weights_means['Alternatives'][:].values.tolist(),
                y=all_weights_means.iloc[:, i + 1],
                error_y=dict(type='data', array=all_weights_stds.iloc[:, i + 1])
            ))
        else:
            fig.add_trace(go.Bar(
                name=all_weights_means.columns[i + 1],
                x=all_weights_means['Alternatives'][:].values.tolist(),
                y=all_weights_means.iloc[:, i + 1],
                showlegend=True
            ))
        i = i + 1
    fig.update_traces(showlegend=True)
    fig.update_layout(barmode='group', height=600, width=1000,
                      title="MCDA analysis with random sampled weight for the indicator '{}'".format(indicators[index]),
                      title_font_size=22,

                      xaxis=dict(
                          tickmode="array",
                          tickvals=np.arange(0, len(all_weights_means['Alternatives'][:])),
                          ticktext=all_weights_means['Alternatives'][:],
                          tickangle=45)
                      )
    return fig


def save_figure(figure: object, folder_path: str, filename: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    result_path = os.path.join(folder_path, new_filename)
    figure.write_image(result_path)


def combine_images(figures, folder_path: str, filename: str):
    images = []
    result_path = os.path.join(folder_path, filename)
    for fig in figures:
        image_bytes = pio.to_image(fig, format="png")
        image = Image.open(io.BytesIO(image_bytes))
        images.append(image)
    # determine the size of the final image based on the first image
    width, height = images[0].size
    combined_image = Image.new("RGB", (width, height * len(images)))
    for i, image in enumerate(images):
        box = (0, i * height)
        combined_image.paste(image, box)
    combined_image.save(result_path)
