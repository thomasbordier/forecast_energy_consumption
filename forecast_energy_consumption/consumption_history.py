from forecast_energy_consumption.dataprep import df_train_test


def consumption_history(Date_debut_test):

    df_train, df_test = df_train_test(Date_debut_test, Nombre_jours_test = 14)

    df_history = df_train.tail(365)

    date_conso = df_history[['Date','Consommation (MW)']]

    productions_history = df_history.groupby(['Code INSEE région']).mean()

    productions_history = productions_history[['Thermique (MW)','Eolien (MW)',
        'Solaire (MW)', 'Hydraulique (MW)',  'Bioénergies (MW)',
       'Ech. physiques (MW)']]

# 'Pompage (MW)',

    return date_conso, productions_history


