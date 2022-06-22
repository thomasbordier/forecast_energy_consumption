"""Prepare Data so as to be used in a Pipelined ML model"""

import numpy as np
from forecast_energy_consumption.params import DATA
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import os



def load_data():
    """Load data from `data_path` into to memory
    Returns a 2D array with (axis 0) representing timesteps, and (axis 1) columns containing tagets and covariates
    ref: https://github.com/lewagon/data-images/blob/master/DL/time-series-covariates.png?raw=true
    """

    csv_path = os.path.dirname(os.path.dirname(__file__))

    csv_path = os.path.join(csv_path, 'raw_data','data_preparation.csv')

    df = pd.read_csv(csv_path, index_col=[0])

    return df



def df_train_test(Date_debut_test, Nombre_jours_test):


    ''' création des df_train et df_test'''

    #data_path = '../raw_data/data_preparation.csv'

    df = load_data()

    Date_debut_test_time_ = pd.to_datetime(Date_debut_test)
    Date_fin_test_time_ =  Date_debut_test_time_ + timedelta(Nombre_jours_test)
    Date_fin_test = str(Date_fin_test_time_)[0:10]

    df_train = df[ (df["Date"] < Date_debut_test)]
    df_test = df[ (df["Date"] >= Date_debut_test) & (df["Date"] < Date_fin_test)]

    return df_train, df_test


def X_y_train_test(Date_debut_test, Nombre_jours_test):



    df_train, df_test = df_train_test(Date_debut_test, Nombre_jours_test)

    X_train = df_train.drop(columns=['Date', 'Code INSEE région', 'Consommation (MW)', 'Thermique (MW)',
       'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)',
       'Pompage (MW)', 'Bioénergies (MW)', 'Ech. physiques (MW)',
       'Stockage batterie', 'Déstockage batterie', 'Eolien terrestre',
       'Eolien offshore', 'TCO Thermique (%)', 'TCH Thermique (%)',
       'TCO Nucléaire (%)', 'TCH Nucléaire (%)', 'TCO Eolien (%)',
       'TCH Eolien (%)', 'TCO Solaire (%)', 'TCH Solaire (%)', 'Column 30','YEAR', 'MONTH', 'DAY', 'season', 'num_day'])

    y_train = df_train['Consommation (MW)']

    X_test = df_test.drop(columns=['Date', 'Code INSEE région', 'Consommation (MW)', 'Thermique (MW)',
       'Nucléaire (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)',
       'Pompage (MW)', 'Bioénergies (MW)', 'Ech. physiques (MW)',
       'Stockage batterie', 'Déstockage batterie', 'Eolien terrestre',
       'Eolien offshore', 'TCO Thermique (%)', 'TCH Thermique (%)',
       'TCO Nucléaire (%)', 'TCH Nucléaire (%)', 'TCO Eolien (%)',
       'TCH Eolien (%)', 'TCO Solaire (%)', 'TCH Solaire (%)', 'Column 30','YEAR',
        'MONTH', 'DAY', 'season', 'num_day',
       ])

    y_test = df_test['Consommation (MW)']

    return X_train,y_train,X_test,y_test, df_train






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
