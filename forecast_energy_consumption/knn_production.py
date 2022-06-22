from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import date, timedelta, datetime

# date en dur ou date debut - X années

def X_y_train_knn(df_train):
    df_train_knn = df_train[(df_train["Date"] >= '2019-01-01')]

    X_train_knn = df_train_knn[['Consommation (MW)','T2M','T2M_MAX','T2M_MIN','RH2M','PRECTOTCORR']]

    y_train_knn = df_train_knn['PS']

    return df_train_knn, X_train_knn, y_train_knn



def X_y_test_knn(y_pred, X_test):

    df_y_pred = pd.DataFrame(y_pred)

    df_test_knn = pd.merge(left = df_y_pred.reset_index(drop = True)
            , right = X_test.reset_index(drop = True),
            left_index = True, right_index = True) \
            .rename(columns = {0: 'Consommation (MW)'}, inplace = True)

    X_test_knn = df_test_knn[['Consommation (MW)','T2M','T2M_MAX','T2M_MIN','RH2M','PRECTOTCORR']]

    y_test_knn = df_test_knn['PS']

    return X_test_knn, y_test_knn



def knn_production(df_train, y_pred, X_test):

    df_train_knn, X_train_knn, y_train_knn = X_y_train_knn(df_train)

    X_test_knn, y_test_knn = X_y_test_knn(y_pred, X_test)

    min_max = MinMaxScaler()

    X_train_knn_scalle = min_max.fit_transform(X_train_knn)

    knn_model = KNeighborsRegressor().fit(X_train_knn_scalle,y_train_knn)

    X_test_knn_scalle = min_max.transform(X_test_knn)

    result_knn = knn_model.kneighbors(X_test_knn_scalle,n_neighbors=2)

    for index_knn in result_knn[1]:
        df_knn_prediction = df_train_knn.iloc[index_knn].groupby(['Code INSEE région']).mean()

        conso = df_knn_prediction['Consommation (MW)']

        thermique = df_knn_prediction['Thermique (MW)']
        eolien = df_knn_prediction['Eolien (MW)']
        solaire = df_knn_prediction['Solaire (MW)']
        hydraulique = df_knn_prediction['Hydraulique (MW)']
        bioenergies = df_knn_prediction['Bioénergies (MW)']
        ech_physiques = df_knn_prediction['Ech. physiques (MW)']

        thermique_pc = thermique/conso*100
        eolien_pc = eolien/conso*100
        solaire_pc = solaire/conso*100
        hydraulique_pc = hydraulique/conso*100
        bioenergies_pc = bioenergies/conso*100
        ech_physiques_pc = ech_physiques/conso*100



        # Date lendemain

        print(date_knn)


        Date_knn_datetime = pd.to_datetime(date_knn)
        Date_knn_1 =  Date_knn_datetime + timedelta(1)
        date_knn = str(Date_knn_1)[0:10]




    return 'ok'

    # voir le traitement des values
