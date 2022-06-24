'''
Top level orchestrator of the project. To be called from the CLI.
It comprises all the "routes" you may want to call
'''
from email import header
from operator import mod
import numpy as np
import pandas as pd
import os

from typing import Tuple, List
import matplotlib.pyplot as plt
from forecast_energy_consumption.dataprep import X_y_train_test
from  forecast_energy_consumption.model import get_model, fit_model, save_model
from forecast_energy_consumption.predict import predict_output
from forecast_energy_consumption.modif_X_train import X_Test_Plus, X_Test_Moins





def main(model_name,date_debut_test, nombre_jours_test):

    X_test, y_test, df_train = train(model_name,date_debut_test, nombre_jours_test)

    predictions, mape = predict_output(X_test,y_test, metric = True)
    df_X_test_moins = X_test.copy()
    df_X_test_plus = X_test.copy()
    
    # Création predictions avec + 5 degrés

    X_test_plus = X_Test_Plus(df_X_test_plus)

    predictions_plus_x_celsius = predict_output(X_test_plus,y_test, metric = False)

    # Création predictions avec + 5 degrés

    X_test_moins = X_Test_Moins(df_X_test_moins)

    predictions_moins_x_celsius = predict_output(X_test_moins,y_test, metric = False)



    # ajout du retour de predictions_plus_x_celsius et

    return  df_train, X_test, y_test, predictions, mape, predictions_plus_x_celsius, predictions_moins_x_celsius





def train(model_name, date_debut_test, nombre_jours_test):
    """
    Train the model in this package on one fold `data` containing the 2D-array of time-series for your problem
    Returns `metrics_test` associated with the training
    """

    X_train,y_train,X_test,y_test, df_train = X_y_train_test(date_debut_test, nombre_jours_test)

    model = get_model(model_name)

    model = fit_model(model, X_train,y_train)

    save_model(model)

    return X_test,y_test, df_train

















def cross_validate(data: np.ndarray, print_metrics: bool = False):
    """
    Cross-Validate the model in this package on`data`
    Returns `metrics_cv`: the list of test metrics at each fold
    """
    pass  # YOUR CODE HERE


def backtest(data: np.ndarray,
             stride: int = 1,
             start_ratio: float = 0.9,
             retrain: bool = True,
             retrain_every: int = 1,
             print_metrics=False,
             plot_metrics=False):

    pass
    """Returns historical forecasts for the entire dataset
    - by training model up to `start_ratio` of the dataset
    - then predicting next values using the model in this package (only predict the last time-steps if `predict_only_last_value` is True)
    - then moving `stride` timesteps ahead
    - then retraining the model if `retrain` is True and if we moved `retrain_every` timesteps since last training
    - then predicting next values again

    Return:
    - all historical predictions as 2D-array time-series of shape ((1-start_ratio)*len(data), n_targets)/stride
    - Compute the 'mean-MAPE' per forecast horizon
    - Print historical predictions if you want a visual check

    see https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html#darts.models.forecasting.rnn_model.RNNModel.historical_forecasts

    pass  # YOUR CODE HERE

if __name__ == '__main__':
    data = pd.read_csv(os.path.join( 'data','raw','data.csv')).to_numpy()
    try:
        train(data=data, print_metrics=True)
        # cross_validate(data=data, print_metrics=True)
        # backtest(data=data,
        #      stride = 1,
        #      start_ratio = 0.9,
        #      retrain = True,
        #      retrain_every=1,
        #      print_metrics=True,
        #      plot_metrics=True)
    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
"""
