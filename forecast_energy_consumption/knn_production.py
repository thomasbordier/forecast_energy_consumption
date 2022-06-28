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
    #print('df_y_pred.reset_index(drop = True)=',df_y_pred.reset_index(drop = True))
    #print('X_test.reset_index(drop = True)=',X_test.reset_index(drop = True)) #df OK
    df_test_knn = pd.merge(left = df_y_pred.reset_index(drop = True)
            , right = X_test.reset_index(drop = True),
            left_index = True, right_index = True)

    df_test_knn.rename(columns = {0: 'Consommation (MW)'}, inplace = True)

    X_test_knn = df_test_knn[['Consommation (MW)','T2M','T2M_MAX','T2M_MIN','RH2M','PRECTOTCORR']]
    y_test_knn = df_test_knn['PS']

    return X_test_knn, y_test_knn



def knn_production(df_train, X_test, y_pred, date_knn, nb_neighbors):
    # import plotly.graph_objects as go
    # ajout de , Date_debut_test ???

    df_train_knn, X_train_knn, y_train_knn = X_y_train_knn(df_train)

    X_test_knn, y_test_knn = X_y_test_knn(y_pred, X_test)

    min_max = MinMaxScaler()


    X_train_knn_scalle = min_max.fit_transform(X_train_knn)

    knn_model = KNeighborsRegressor().fit(X_train_knn_scalle,y_train_knn)

    X_test_knn_scalle = min_max.transform(X_test_knn)

    result_knn = knn_model.kneighbors(X_test_knn_scalle,n_neighbors=nb_neighbors)

    date_list = []
    thermique_list = []
    eolien_list = []
    solaire_list = []
    hydraulique_list = []
    bioenergies_list = []
    ech_physiques_list = []

    for index_knn in result_knn[1]:
        list_index= []
        for i in index_knn:
            list_index.append(i)
        #print(list_index)

        df_knn_prediction = df_train_knn.iloc[list_index]

        thermique = df_knn_prediction['Thermique (MW)'].mean()
        eolien = df_knn_prediction['Eolien (MW)'].mean()
        solaire = df_knn_prediction['Solaire (MW)'].mean()
        hydraulique = df_knn_prediction['Hydraulique (MW)'].mean()
        bioenergies = df_knn_prediction['Bioénergies (MW)'].mean()
        ech_physiques = df_knn_prediction['Ech. physiques (MW)'].mean()

        date_list.append(date_knn)
        thermique_list.append(thermique)
        eolien_list.append(eolien)
        solaire_list.append(solaire)
        hydraulique_list.append(hydraulique)
        bioenergies_list.append(bioenergies)
        ech_physiques_list.append(ech_physiques)

        Date_knn_datetime = pd.to_datetime(date_knn)
        Date_knn_1 =  Date_knn_datetime + timedelta(1)
        date_knn = str(Date_knn_1)[0:10]



    return date_list, thermique_list, eolien_list, solaire_list, hydraulique_list, bioenergies_list, ech_physiques_list


    # voir le traitement des values
