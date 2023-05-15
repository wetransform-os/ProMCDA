import pandas as pd
import argparse
import json
import os


def read_matrix(input_matrix_path: str) -> pd.DataFrame():
    try:
        file_path = os.path.dirname(os.path.realpath(__file__))
        filename = file_path + '/../' + input_matrix_path #TODO: is there any more pythonistic way to extract the path?
        with open(filename, 'r') as fp:
            test = pd.read_csv(fp, sep=';', decimal='.')
            return test
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


def save_df(df: pd.DataFrame, folder_path: str, filename: str):
    result_path = os.path.join(folder_path, filename)
    df.to_csv(path_or_buf=result_path)

def save_config(config: dict, folder_path: str, filename: str):
    result_path = os.path.join(folder_path, filename)
    with open(result_path, 'w') as fp:
        json.dump(config, fp)