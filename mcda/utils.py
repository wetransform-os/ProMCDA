import plotly.graph_objects as go
from sklearn import preprocessing
from os.path import abspath
import plotly.graph_objects as go
from typing import List
import plotly.io as pio
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import pickle
import random
import json
import os
import io


def read_matrix(input_matrix_path: str) -> pd.DataFrame():
    try:
        filename = abspath(input_matrix_path)
        with open(filename, 'r') as fp:
            test = pd.read_csv(fp, sep="[,;:]", decimal='.')
            return test
    except Exception as e:
        print(e)


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
    result_path = os.path.join(folder_path, filename)
    df.to_csv(path_or_buf=result_path, index=False)


def save_dict(dict: dict, folder_path: str, filename: str):
    result_path = os.path.join(folder_path, filename)
    with open(result_path, 'wb') as fp:
        pickle.dump(dict, fp)


def save_config(config: dict, folder_path: str, filename: str):
    result_path = os.path.join(folder_path, filename)
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


def randomly_sample_all_weights(no_weights: int, no_runs: int) -> List[list]:
    """ The function generates 'no_runs' lists of weights;
        Each list has 'no_weights' random sampled elements."""
    list_of_weights = []
    for _ in range(no_runs):
        lst = [random.uniform(0, 1) for _ in range(no_weights)]
        list_of_weights.append(lst)

    return list_of_weights


def randomly_sample_ix_weight(no_weights: int, index: int, no_runs: int) -> List[list]:
    """ The function generates 'no_runs' lists of weights;
        Each list has 'no_weights' elements,
        only one randomly sampled, at position index."""
    list_of_weights = []
    for _ in range(no_runs):
        lst = [1] * no_weights
        lst[index] = random.uniform(0, 1)
        list_of_weights.append(lst)

    return list_of_weights


def check_norm_sum_weights(weights: list) -> list:
    if sum(weights) != 1:
        norm_weights = [val / sum(weights) for val in weights]
        return norm_weights
    else:
        return weights


def plot_norm_scores_without_uncert(scores: pd.DataFrame) -> object:
    no_of_combinations = scores.shape[1] - 1
    fig = go.Figure(layout_yaxis_range=[scores.iloc[: , 1:].values.min()-0.5, scores.iloc[: , 1:].values.max()+0.5],layout_yaxis_title="MCDA normalized score")
    i = 0
    while i <= no_of_combinations - 1:
        fig.add_trace(go.Bar(
            name=scores.columns[i + 1],
            x=scores['Alternatives'].values.tolist(),
            y=scores.iloc[:, i + 1],
        ))
        i = i + 1
    fig.update_layout(barmode='group', height=600, width=1000,
                      title='<b>MCDA analysis<b>',
                      title_font_size=22,

                      xaxis=dict(
                          tickmode="array",
                          tickvals=np.arange(0, len(scores['Alternatives'][:])),
                          ticktext=scores['Alternatives'][:],
                          tickangle=45),
                      yaxis=dict(
                          range=[scores.iloc[: , 1:].values.min()-0.5, scores.iloc[: , 1:].values.max()+0.5])
                      )
    fig.show()
    return fig


def plot_non_norm_scores_without_uncert(scores: pd.DataFrame) -> object:
    no_of_combinations = scores.shape[1]-1
    fig = go.Figure(layout_yaxis_title="MCDA rough score")
    i = 0
    while i <= no_of_combinations - 1:
        fig.add_trace(go.Bar(
            name=scores.columns[i+1],
            x=scores['Alternatives'].values.tolist(),
            y=scores.iloc[:, i + 1],
        ))
        i = i + 1
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

def plot_mean_scores(all_weights_means:pd.DataFrame, all_weights_stds:pd.DataFrame)-> object:
    no_of_combinations = all_weights_means.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA average scores and std")
    i = 0
    while i <= no_of_combinations - 1:
        fig.add_trace(go.Bar(
            name=all_weights_means.columns[i + 1],
            x=all_weights_means['Alternatives'].values.tolist(),
            y=all_weights_means.iloc[:, i + 1],
            error_y=dict(type='data', array=all_weights_stds.iloc[:, i + 1])
        ))
        i = i + 1
    fig.update_layout(barmode='group', height=600, width=1000,
                      title='<b>MCDA analysis with added randomness on the weights<b>',
                      title_font_size=22,

                      xaxis=dict(
                          tickmode="array",
                          tickvals=np.arange(0, len(all_weights_means['Alternatives'][:])),
                          ticktext=all_weights_means['Alternatives'][:],
                          tickangle=45)
                      )
    fig.show()
    return fig


def plot_mean_scores_iterative(all_weights_means:pd.DataFrame, all_weights_stds:pd.DataFrame, indicators: list, index: int)-> object:
    no_of_combinations = all_weights_means.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA average scores and std")
    i = 0
    while i <= no_of_combinations - 1:
        fig.add_trace(go.Bar(
            name=all_weights_means.columns[i + 1],
            x=all_weights_means['Alternatives'][:].values.tolist(),
            y=all_weights_means.iloc[:, i + 1],
            error_y=dict(type='data', array=all_weights_stds.iloc[:, i + 1])
        ))
        i = i + 1
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
    result_path = os.path.join(folder_path, filename)
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