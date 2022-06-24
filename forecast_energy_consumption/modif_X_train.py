


def t_plus_x_celsius(value):

    value = value + 2

    return value


def X_Test_Plus(X_test):

    X_test['T2M']=X_test['T2M'].apply(t_plus_x_celsius)
    X_test['T2MDEW']=X_test['T2MDEW'].apply(t_plus_x_celsius)
    X_test['T2MWET']=X_test['T2MWET'].apply(t_plus_x_celsius)

    X_test['T2M_MAX']=X_test['T2M_MAX'].apply(t_plus_x_celsius)
    X_test['T2M_MIN']=X_test['T2M_MIN'].apply(t_plus_x_celsius)
    X_test['QV2M']=X_test['QV2M'].apply(t_plus_x_celsius)
    X_test['RH2M']=X_test['RH2M'].apply(t_plus_x_celsius)
    X_test['PRECTOTCORR']=X_test['PRECTOTCORR'].apply(t_plus_x_celsius)
    X_test['WS10M']=X_test['WS10M'].apply(t_plus_x_celsius)
    X_test['WS50M']=X_test['WS50M'].apply(t_plus_x_celsius)

    return X_test





def t_moins_x_celsius(value):

    value = value - 2

    return value


def X_Test_Moins(X_test):

    X_test['T2M']=X_test['T2M'].apply(t_moins_x_celsius)
    X_test['T2MDEW']=X_test['T2MDEW'].apply(t_moins_x_celsius)
    X_test['T2MWET']=X_test['T2MWET'].apply(t_moins_x_celsius)

    X_test['T2M_MAX']=X_test['T2M_MAX'].apply(t_moins_x_celsius)
    X_test['T2M_MIN']=X_test['T2M_MIN'].apply(t_moins_x_celsius)
    X_test['QV2M']=X_test['QV2M'].apply(t_moins_x_celsius)
    X_test['RH2M']=X_test['RH2M'].apply(t_moins_x_celsius)
    X_test['PRECTOTCORR']=X_test['PRECTOTCORR'].apply(t_moins_x_celsius)
    X_test['WS10M']=X_test['WS10M'].apply(t_moins_x_celsius)
    X_test['WS50M']=X_test['WS50M'].apply(t_moins_x_celsius)

    return X_test
