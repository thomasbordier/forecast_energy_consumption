from joblib import load
import pandas as pd

def predict_output(X_test,y_test, metric = True):

    model = load('forecast_energy_consumption/data/model.joblib')

    predictions = []

    X_test_i = pd.DataFrame([X_test.iloc[0,26:]])

    X_test_0 = pd.DataFrame([X_test.iloc[0,:]])

    y_i = model.predict(X_test_0)

    predictions.append(y_i[0])

    for i in range (1,len(X_test)):

        X_test_features = pd.DataFrame([X_test.iloc[i,:26]])

        X_test_i.iloc[0,1:] = X_test_i.iloc[0,:-1]
        X_test_i.iloc[0,0] = y_i[0]

        X_test_decal = pd.DataFrame(X_test_i)

        X_test_pred = pd.merge(left = X_test_features.reset_index(drop = True), right = X_test_decal.reset_index(drop = True),
                    left_index = True, right_index = True)

        y_i= model.predict(X_test_pred)


        predictions.append(y_i[0])



    return predictions
