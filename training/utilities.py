import re
import logging
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
from plotly.subplots import make_subplots
from promcda.enums import PDFType, RobustnessAnalysisType


### setups
def setup_no_robustness(input_matrix, focus_on):

    polarity = ('-', '-', '-', '+', '-', '+')

    if focus_on == "cost":
        weights = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    elif focus_on == "travel time":
        weights = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
    elif focus_on == "carbon":
        weights = [0.1, 0.1, 0.1, 0.1, 0.9, 0.1]
    elif focus_on == "preference":
        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    elif focus_on == None:
        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    else:
        raise ValueError("Invalid focus_on value.")
    
    robustness = RobustnessAnalysisType.NONE

    # Return the setup parameters as a dictionary
    setup_no_robustness_indicators = {
        'input_matrix': input_matrix,
        'polarity': polarity,
        'weights': weights,
        'robustness': robustness,
    }

    return(setup_no_robustness_indicators)
    

def setup_robustness_weights(input_matrix):

    polarity = ('-', '-', '-', '+', '-', '+')
    
    robustness = RobustnessAnalysisType.ALL_WEIGHTS

    # Return the setup parameters as a dictionary
    setup_robustness_weights = {
        'input_matrix': input_matrix,
        'polarity': polarity,
        'robustness': robustness
    }

    return(setup_robustness_weights)

def setup_robustness(input_matrix, focus_on):

    polarity = ('-', '-', '-', '+', '-', '+')
    marginal_distributions = (PDFType.NORMAL, PDFType.LOGNORMAL, PDFType.UNIFORM,  PDFType.POISSON,  PDFType.LOGNORMAL,  PDFType.EXACT)

    if focus_on == "cost":
        weights = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    elif focus_on == "travel time":
        weights = [0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
    elif focus_on == "carbon":
        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
    else:
        raise ValueError("Invalid focus_on value.")
    
    robustness = RobustnessAnalysisType.INDICATORS

    # Return the setup parameters as a dictionary
    setup_robustness = {
        'input_matrix': input_matrix,
        'polarity': polarity,
        'marginal_distributions': marginal_distributions,
        'weights': weights,
        'robustness': robustness
    }

    return(setup_robustness)    

### plots
def plot_data_and_normalized(data, normalized, norm_name="minmax"):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    filtered_columns = [col for col in normalized.columns 
                        if col.endswith("_minmax_01") and "without_zero" not in col]

    clean_labels = [col.replace("_minmax_01", "") for col in filtered_columns]

    # Crea figura con height_ratios tramite GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Plot 1
    sns.boxplot(data=data, ax=ax1)
    ax1.set_title("Indicator values", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Value", fontsize=14)
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_xticks([])
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2
    sns.boxplot(data=normalized[filtered_columns], ax=ax2)
    means = normalized[filtered_columns].mean()
    for i, mean in enumerate(means):
        ax2.hlines(mean, i - 0.4, i + 0.4, colors='black', linestyles=':', linewidth=3)

    ax2.set_title(f"Normalized indicators ({norm_name})", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Normalized value", fontsize=14)
    ax2.set_xticks(range(len(clean_labels)))
    ax2.set_xticklabels(clean_labels, rotation=45, fontsize=14)
    ax2.set_xlabel("Indicators", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_normalization_comparison(normalized_df):
    data_long = []

    for col in normalized_df.columns:
        match = re.match(r"(.+?)_(minmax|target|standardized|rank)(?:_(01|without_zero|any))?$", col)
        if match:
            indicator = match.group(1).strip()
            method = match.group(2)
            suffix = match.group(3)  

            if method == "standardized" and suffix != "any":
                continue
            if method in ["minmax", "target"] and suffix != "01":
                continue
            if method == "rank" and suffix not in [None, ""]:
                continue

            values = normalized_df[col].values
            for val in values:
                data_long.append({
                    "Indicator": indicator,
                    "Method": method,
                    "Value": val
                })

    df_long = pd.DataFrame(data_long)

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_long, x="Indicator", y="Value", hue="Method")

    plt.title("Comparison of normalization methods", fontsize=16, fontweight='bold')
    plt.ylabel("Normalized Value", fontsize=14)
    plt.xlabel("Indicator", fontsize=14)
    plt.xticks(rotation=45, fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend(title="Normalization Method")
    plt.tight_layout()
    plt.show()


def plot_scores_comparison(scores):
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(data=scores)
    plt.xticks(rotation=90)

    for i, col in enumerate(scores):
        max_value = scores[col].max()
        max_index = scores[col].idxmax()
        alternative = f'Alt. {max_index + 1}'
        plt.text(i, max_value, f'{alternative}\n{max_value:.2f}', ha='center', va='bottom', fontsize=10, color='red')

    # Personalizza le etichette degli assi e il titolo
    plt.xlabel("Pair norm/aggr", fontsize=14, fontweight='bold')
    plt.ylabel("Scores", fontsize=14, fontweight='bold')
    plt.title("The best alternative's score across different norm/aggr pairs", fontsize=16, fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_non_norm_scores_without_uncert(scores: pd.DataFrame, alternatives: list) -> object:
    # Verify that the number of alternatives matches the number of rows in scores
    if len(alternatives) != scores.shape[0]:
        raise ValueError("Length of 'alternatives' must match number of rows in 'scores'.")
    
    num_of_combinations = scores.shape[1]
    fig = go.Figure(layout_yaxis_title="MCDA score")
    
    # Add a bar trace for each combination (column) in scores
    for i, col_name in enumerate(scores.columns):
        fig.add_trace(go.Bar(
            name=col_name,
            x=alternatives,
            y=scores[col_name],
            showlegend=True
        ))
    
    fig.update_traces(showlegend=True)
    fig.update_layout(
        barmode='group',
        height=600,
        width=1000,
        title='<b>MCDA Analysis: Non-normalized Scores</b>',
        title_font_size=22,
        xaxis=dict(
            tickmode="array",
            tickvals=np.arange(len(alternatives)),
            ticktext=alternatives,
            tickangle=45
        )
    )
    return fig


def plot_grouped_bar_subplots(scores_dict, alternatives):
    num_plots = len(scores_dict)
    
    fig = make_subplots(
        rows=1,
        cols=num_plots,
        subplot_titles=[f"Aggregation: {key.upper()}" for key in scores_dict.keys()],
        shared_yaxes=True
    )
    
    colors = px.colors.qualitative.Set2

def plot_grouped_bar_subplots(scores_dict, alternatives):
    num_plots = len(scores_dict)
    
    normalizations = set()
    
    for scores_subset in scores_dict.values():
        for col in scores_subset.columns:
            if col.startswith("ws-"):
                norm_part = col[len("ws-"):]             
                norm = norm_part.split('_')[0]                          
            normalizations.add(norm)

    normalizations = sorted(normalizations)

    fig = make_subplots(
        rows=1,
        cols=num_plots,
        subplot_titles=[f"{key.upper()}" for key in scores_dict.keys()],
        shared_yaxes=True,
        horizontal_spacing=0.05
    )

    colors = px.colors.qualitative.Set2

    for i, (key, scores_subset) in enumerate(scores_dict.items(), start=1):
        for j, col in enumerate(scores_subset.columns):
            normalization = normalizations[j % len(normalizations)]
            legend_name = f"{normalization}"
            fig.add_trace(
                go.Bar(
                    name=legend_name,
                    x=alternatives,
                    y=scores_subset[col],
                    marker=dict(
                        color=colors[j % len(colors)],
                        line=dict(color='black', width=1)  
                    ),
                    showlegend=(i == 1)
                ),
                row=1,
                col=i
            )

    fig.update_layout(
        height=500,
        width=1150,  
        barmode="group",
        title_text="MCDA scores comparison by aggregation function",
        title_font_size=20,
        legend_title_text="Normalization",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=100, b=100, l=80, r=40),
        font=dict(size=12)
    )

    fig.update_layout(shapes=[
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="lightgrey", width=1),
            layer="below"
        )
    ])

    for i in range(1, num_plots + 1):
        fig.update_xaxes(
            tickangle=45,
            showgrid=False,
            linecolor='black',
            mirror=True,
            row=1,
            col=i
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="lightgrey",
            zeroline=False,
            linecolor='black',
            mirror=True,
            row=1,
            col=i
        )

    fig.show()


def plot_bar_with_std(mean_df, std_df, alternatives):

    title="MCDA mean scores with uncertainty"
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, col in enumerate(mean_df.columns):
        fig.add_trace(
            go.Bar(
                name=col,
                x=alternatives,
                y=mean_df[col],
                error_y=dict(
                    type='data',
                    array=std_df[col],
                    visible=True,
                    thickness=1.5,
                    width=6,
                    color='black'
                ),
                marker=dict(
                    color=colors[i % len(colors)],
                    line=dict(color='black', width=1)
                )
            )
        )

    fig.update_layout(
        title=title,
        title_font_size=20,
        barmode='group',
        xaxis_title='Alternatives',
        yaxis_title='Score (mean ± std)',
        legend_title='Normalization',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=80, b=100, l=80, r=40),
        font=dict(size=12)
    )

    fig.update_xaxes(
        tickangle=45,
        linecolor='black',
        mirror=True
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgrey",
        zeroline=False,
        linecolor='black',
        mirror=True
    )

    fig.show()