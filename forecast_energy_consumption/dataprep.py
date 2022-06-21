"""Prepare Data so as to be used in a Pipelined ML model"""

import numpy as np
from forecast_energy_consumption.params import DATA
from typing import Tuple, List
import numpy as np
import pandas as pd

def load_data(data_path: str):
    """Load data from `data_path` into to memory
    Returns a 2D array with (axis 0) representing timesteps, and (axis 1) columns containing tagets and covariates
    ref: https://github.com/lewagon/data-images/blob/master/DL/time-series-covariates.png?raw=true
    """

    # path '../raw_data/data_preparation.csv'

    df = pd.read_csv(data_path, index_col=[0])

    return df


def clean_data(data: np.ndarray):
    """Clean data without creating data leakage:
        - make sure there is no NaN between any timestep
        - etc...
    """
    # YOUR_CODE_HERE
    pass


def get_X_y(
    data: np.ndarray,
    input_length: int,
    output_length: int,
    horizon: int,
    stride: int,
    shuffle=True,
    **kwargs,
):
    """
    Use `data`, a 2D-array with axis=0 as timesteps, and axis=1 as (tagets+covariates columns)

    Returns a Tuple (X,y) of two ndarrays :
        X.shape = (n_samples, input_length, n_covariates)
        y.shape =
            (n_samples, output_length, n_targets) if all 3-dimensions are of size > 1
            (n_samples, output_length) if n_targets == 1
            (n_samples, n_targets) if output_length == 1
            (n_samples, ) if both n_targets and lenghts == 1

    ❗️ Raise error if data contains NaN
    ❗️ Make sure to shuffle the pairs in unison if `shuffle=True` for idd purpose
    ❗️ Don't ditch past values of your target time-series in your features - they are very useful features!
    👉 illustration: https://raw.githubusercontent.com/lewagon/data-images/master/DL/rnn-1.png

    [💡 Hints ] You can use a sliding method
        - Reading `data` in ascending order
        - `stride` timestamps after another
    Feel free to use another approach, for example random sampling without replacement

    """
    pass  # YOUR CODE HERE





def get_folds(data: np.ndarray,
              fold_length: int,
              fold_stride: int,
              **kwargs):
    """Slide through `data` time-series (2D array) to create folds of equal `fold_length`, using `fold_stride` between each fold
    Returns a list of folds, each as a 2D-array time series
    """
    pass  # YOUR CODE HERE


def train_test_split(data: np.ndarray,
                     train_test_ratio: float,
                     input_length: int,
                     **kwargs):
    """Returns a train and test 2D-arrays, that will not create any data leaks when sampling (X, y) from them
    Inspired from "https://raw.githubusercontent.com/lewagon/data-images/master/DL/rnn-3.png"
    """
    pass  # YOUR CODE HERE
