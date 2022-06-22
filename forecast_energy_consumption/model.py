from webbrowser import get
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Reshape, Lambda, Input
from tensorflow.keras import Model
from forecast_energy_consumption.pipeline import pipeline_preproc
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor


from joblib import dump

# TODO: Should we add here the preprocessing? into a class called "pipeline"?
# TODO: Should we refacto in a class ? Probably!


def get_model(model_name):
    """Instanciate, compile and and return the model of your choice
    xgb = xgboost
    cat = catboostregressor
    grad = gradientboosting
    """

    preproc = pipeline_preproc()

    if model_name == 'xgb':
        model_select = model_xgboost()

    if model_name == 'cat':
        model_select = model_catboost()

    if model_name == 'grad':
        model_select = model_gradient()

    pipeline = make_pipeline(preproc,model_select)

    return pipeline


def fit_model(model, X_train, y_train, **kwargs):
    """Fit the `model` object, including preprocessing if needs be"""

    model = model.fit(X_train, y_train)

    return model


def save_model(model):

    dump(model, 'forecast_energy_consumption/data/model.joblib')






def model_xgboost():
    model_xgb = XGBRegressor(colsample_bytree = 1,
                         gamma= 0.1,
                         max_depth= 16,
                         min_child_weight= 6,
                         n_estimators= 100,
                         learning_rate=0.1)

    return model_xgb


def model_catboost():
    model_CBR = CatBoostRegressor(depth = 7,
              learning_rate = 0.0999,
              iterations = 130)

    return model_CBR



def model_gradient():
    model_grad = GradientBoostingRegressor(learning_rate= 0.011,
        max_depth = 4,
        n_estimators = 1500,
        subsample= 0.5)

    return model_grad


'''
def pipeline_final(model):

    preproc = pipeline_preproc()
    pipeline = make_pipeline(preproc,model)
    return pipeline
'''
