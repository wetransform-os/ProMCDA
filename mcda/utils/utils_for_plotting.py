import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image


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
                          tickvals=np.arange(
                              0, len(scores['Alternatives'][:])),
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
                          tickvals=np.arange(
                              0, len(scores['Alternatives'][:])),
                          ticktext=scores['Alternatives'][:],
                          tickangle=45)
                      )
    fig.show()
    return fig


def plot_mean_scores(all_means: pd.DataFrame, plot_std: str, rand_on: str, all_stds=None) -> object:
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
                          tickvals=np.arange(
                              0, len(all_means['Alternatives'][:])),
                          ticktext=all_means['Alternatives'][:],
                          tickangle=45)
                      )
    fig.show()
    return fig


def plot_mean_scores_iterative(all_weights_means: pd.DataFrame, indicators: list,
                               index: int, plot_std: str, all_weights_stds=None) -> object:
    num_of_combinations = all_weights_means.shape[1] - 1
    fig = go.Figure(layout_yaxis_title="MCDA average scores and std")
    i = 0
    while i <= num_of_combinations - 1:
        if plot_std == "plot_std":
            fig.add_trace(go.Bar(
                name=all_weights_means.columns[i + 1],
                x=all_weights_means['Alternatives'][:].values.tolist(),
                y=all_weights_means.iloc[:, i + 1],
                error_y=dict(
                    type='data', array=all_weights_stds.iloc[:, i + 1])
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
                      title="MCDA analysis with random sampled weight for the indicator '{}'".format(
                          indicators[index]),
                      title_font_size=22,

                      xaxis=dict(
                          tickmode="array",
                          tickvals=np.arange(
                              0, len(all_weights_means['Alternatives'][:])),
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
