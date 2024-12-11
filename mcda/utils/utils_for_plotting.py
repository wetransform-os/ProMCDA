import io
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

import mcda.utils.utils_for_main as utils_for_main

DEFAULT_OUTPUT_DIRECTORY_PATH = './output_files'  # root directory of ProMCDA


def plot_norm_scores_without_uncert(scores: pd.DataFrame) -> object:
    """
    Plot the normalized scores without uncertainty as a histogram.

    Parameters:
    - scores: DataFrame containing the normalized scores.

    Returns:
    - Plotly figure object.

    :param scores: pd.DataFrame
    :return fig: object
    """
    alternatives_column_name = scores.columns[0]
    num_of_combinations = scores.shape[1] - 1
    fig = go.Figure(layout_yaxis_range=[scores.iloc[:, 1:].values.min() - 0.5, scores.iloc[:, 1:].values.max() + 0.5],
                    layout_yaxis_title="MCDA normalized score")
    i = 0
    while i <= num_of_combinations - 1:
        fig.add_trace(go.Bar(
            name=scores.columns[i + 1],
            x=scores[alternatives_column_name].values.tolist(),
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
                          tickvals=np.arange(
                              0, len(scores[alternatives_column_name][:])),
                          ticktext=scores[alternatives_column_name][:],
                          tickangle=45),
                      yaxis=dict(
                          range=[scores.iloc[:, 1:].values.min() - 0.5, scores.iloc[:, 1:].values.max() + 0.5])
                      )
    # fig.show()
    return fig


def plot_non_norm_scores_without_uncert(scores: pd.DataFrame) -> object:
    """
        Plot the non-normalized scores without uncertainty as a histogram.

        Parameters:
        - scores: DataFrame containing the non-normalized scores.

        Returns:
        - Plotly figure object.

        :param scores: pd.DataFrame
        :return fig: object
        """
    alternatives_column_name = scores.columns[0]
    num_of_combinations = scores.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA rough score")
    i = 0
    while i <= num_of_combinations - 1:
        fig.add_trace(go.Bar(
            name=scores.columns[i + 1],
            x=scores[alternatives_column_name].values.tolist(),
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
                          tickvals=np.arange(
                              0, len(scores[alternatives_column_name][:])),
                          ticktext=scores[alternatives_column_name][:],
                          tickangle=45)
                      )
    # fig.show() it triggers an open socket warning when on
    return fig


def plot_mean_scores(all_means: pd.DataFrame, plot_std: str, rand_on: str, all_stds=None) -> object:
    """
    Plot the mean scores with optional standard deviation as a histogram.

    Parameters:
    - all_means: DataFrame containing the mean scores.
    - plot_std: string indicating whether to plot standard deviation or not.
    - rand_on: string specifying the randomness added on which parameter for the title.
    - all_stds (optional): DataFrame containing the standard deviations.

    Returns:
    - object: Plotly figure object.

    :param all_means: pd.DataFrame
    :parameter plot_std: str
    :parameter rand_on: str
    :param all_stds: pd.DataFrame
    :return fig: object
    """
    alternatives_column_name = all_means.columns[0]
    num_of_combinations = all_means.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA average scores and std")
    i = 0
    while i <= num_of_combinations - 1:
        if plot_std == "plot_std":
            fig.add_trace(go.Bar(
                name=all_means.columns[i + 1],
                x=all_means[alternatives_column_name].values.tolist(),
                y=all_means.iloc[:, i + 1],
                error_y=dict(type='data', array=all_stds.iloc[:, (i+1)])
            ))
            fig.update_layout(title=f'<b>MCDA analysis with added randomness on the {rand_on}<b> (rough scores)',
                              title_font_size=22)
        else:
            fig.add_trace(go.Bar(
                name=all_means.columns[i + 1],
                x=all_means[alternatives_column_name].values.tolist(),
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
                          tickvals=np.arange(
                              0, len(all_means[alternatives_column_name][:])),
                          ticktext=all_means[alternatives_column_name][:],
                          tickangle=45)
                      )
    # fig.show()
    return fig


def plot_mean_scores_iterative(all_weights_means: pd.DataFrame, indicators: list,
                               index: int, plot_std: str, all_weights_stds=None) -> object:
    """
    Plot the mean scores with optional standard deviation for iterative weight sampling as a histogram.

    Parameters:
    - all_weights_means: DataFrame containing the mean scores.
    - indicators: list of indicator names.
    - index : index of the indicator for which weights are sampled.
    - plot_std: string indicating whether to plot standard deviation or not.
    - all_weights_stds (optional): DataFrame containing the standard deviations.

    Returns:
    - object: Plotly figure object.

    :param all_weights_means: pd.DataFrame
    :parameter indicators: list
    :parameter index: int
    :parameter plot_std: str
    :param all_weights_stds: pd.DataFrame
    :return fig: object
    """
    alternatives_column_name = all_weights_means.columns[0]
    num_of_combinations = all_weights_means.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA average scores and std")
    i = 0
    while i <= num_of_combinations - 1:
        if plot_std == "plot_std":
            fig.add_trace(go.Bar(
                name=all_weights_means.columns[i + 1],
                x=all_weights_means[alternatives_column_name][:].values.tolist(),
                y=all_weights_means.iloc[:, i + 1],
                error_y=dict(
                    type='data', array=all_weights_stds.iloc[:, i + 1])
            ))
        else:
            fig.add_trace(go.Bar(
                name=all_weights_means.columns[i + 1],
                x=all_weights_means[alternatives_column_name][:].values.tolist(),
                y=all_weights_means.iloc[:, i + 1],
                showlegend=True
            ))
        i = i + 1
    fig.update_traces(showlegend=True)
    fig.update_layout(barmode='group', height=600, width=1000,
                      title="MCDA analysis with random sampled weight for the indicator '{}'".format(
                          indicators[index]),
                      title_font_size=22,

                      xaxis=dict(
                          tickmode="array",
                          tickvals=np.arange(
                              0, len(all_weights_means[alternatives_column_name][:])),
                          ticktext=all_weights_means[alternatives_column_name][:],
                          tickangle=45)
                      )
    return fig


def save_figure(figure: object, folder_path: str, filename: str):
    """
    Save a Plotly figure as an image.

    Notes:
    - The saved file will have a timestamp added to its filename.
    - A default output path is assigned, and it is used unless
      a custom path is set in an environmental variable in the environment.

    Parameters:
    - figure: Plotly figure object to be saved.
    - folder_path: path to the folder where the image will be saved.
    - filename: name of the image file.

    :param figure: object
    :param folder_path: str
    :param filename: str
    :return: None
    """
    custom_output_path = os.environ.get('PROMCDA_OUTPUT_DIRECTORY_PATH')  # check if an environmental variable is set
    output_directory_path = custom_output_path if custom_output_path else DEFAULT_OUTPUT_DIRECTORY_PATH

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}_{filename}"

    full_output_path = os.path.join(output_directory_path, folder_path, new_filename)
    try:
        utils_for_main.ensure_directory_exists(full_output_path)
    except IOError:
        logging.error(f"Saving a figure failed.")

    figure.write_image(full_output_path)


def combine_images(figures: list, folder_path: str, filename: str):
    """
    Combine multiple Plotly figures into a single image.

    Parameters:
    - figures: list of Plotly figure objects to be combined.
    - folder_path: path to the folder where the combined image will be saved.
    - filename: name of the combined image file.

    :param figures: list
    :param folder_path: str
    :param filename: str
    :return: None
    """
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
