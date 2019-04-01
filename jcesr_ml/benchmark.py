import pandas as pd
import os

default_dataset = os.path.join(os.path.dirname(__file__), '..', 'data',
                               'output', 'g4mp2_data.json.gz')


def load_benchmark_data(path=default_dataset):
    """Load the benchmark dataset
    
    Args:
        path (str): Path to the benchmark data
    Returns:
        - (pd.DataFrame): Training data
        - (pd.DataFrame): Hold-out data
    """
    
    data = pd.read_json(path, lines=True)
    train_data = data.query('not in_holdout')
    holdout_data = data.query('in_holdout')
    return train_data, holdout_data
