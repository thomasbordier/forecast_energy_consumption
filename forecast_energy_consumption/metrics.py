'''
Computes usefull Time Series metrics from (y_true, y_test)
'''
from sklearn.metrics import mean_absolute_percentage_error

import numpy as np
#from tensorflow import reduce_mean



def mae(y_true: np.ndarray, y_pred: np.ndarray):
    """Returns Mean Absolute Error"""
    pass  # YOUR CODE HERE

def mape(y_true, y_pred):
    """Returns Mean Absolute Percentage Error"""
    MAPE = mean_absolute_percentage_error(y_true, y_pred)

    return MAPE

def mase(y_true: np.ndarray, y_pred: np.ndarray):
    """Returns Mean Absolute Scaled Error (https://en.wikipedia.org/wiki/Mean_absolute_scaled_error)
    """
    pass


def play_trading_strategy(y_true: np.ndarray, y_pred: np.ndarray):
    """Returns the array of relative portfolio values over the test period"""
    pass


def return_on_investment(played_trading_strategy: np.ndarray):
    """Returns the ROI of an investment strategy"""
    pass


def sharpe_ratio(played_trading_strategy: np.ndarray):
    """Returns the Sharpe Ratio (Return on Investment / Volatility) of an investment strategy"""
    pass
