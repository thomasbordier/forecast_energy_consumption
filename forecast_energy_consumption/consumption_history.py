


def consumption_history(df_train):
    df_history = df_train.tail(366)

    date_conso = df_history[['Date','Consommation (MW)']]

    productions_history = df_history.groupby(['Code INSEE région']).mean()

    productions_history = productions_history[['Thermique (MW)','Eolien (MW)',
        'Solaire (MW)', 'Hydraulique (MW)',  'Bioénergies (MW)',
       'Ech. physiques (MW)']]

# 'Pompage (MW)',

    return date_conso, productions_history
