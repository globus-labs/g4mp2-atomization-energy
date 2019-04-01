"""Utility functions related to QML use"""
from qml.fchl import get_local_kernels, get_local_symmetric_kernels
from qml.data import Compound
from sklearn.base import BaseEstimator
from io import StringIO
import numpy as np


class FCHLKernel(BaseEstimator):
    """Class for computing the kernel matrix using the qml utility functions
    
    The input `X` to all of the function is the FCHL representation vectors
    """
    
    def __init__(self):
        super(FCHLKernel, self).__init__()
        self.train_points = None
    
    def fit(self, X, y=None):
        # Store the training set
        self.train_points = np.array(X)
        return self
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return np.squeeze(get_local_symmetric_kernels(self.train_points))
    
    def transform(self, X, y=None):
        return get_local_kernels(np.array(X), self.train_points)[0]


def run_model(model, data, xyz_col, max_size=30):
    """Run a FCHL model on new molecules

    Args:
        model (Pipeline): FCHL model (first step is kernel, second is model)
        data (DataFrame): Data to evaluate
        xyz_col (str): Column witht eh XYZ-format molecular structure
        max_size (int): Maximum size to use for the representation
    Returns:
        (ndarray): Predictions from the model
    """

    def compute_rep(xyz):
        c = Compound(StringIO(xyz))
        c.generate_fchl_representation(max_size=max_size)
        return c.representation

    X = np.array(data[xyz_col].apply(compute_rep).tolist())
    return model.predict(X)
