import pandas as pd
import argparse
import json



def read_input_matrix(input_matrix_path: str) -> pd.DataFrame():
    try:
        with open(input_matrix_path, 'r') as fp:
            return pd.read_csv(fp, sep=';')
    except Exception as e:
        print(e)

#def save_result() ->:


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