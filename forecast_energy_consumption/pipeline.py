from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.compose import make_column_transformer




def pipeline_preproc():
    preproc_MinMaxScaler = make_pipeline(MinMaxScaler())
    preproc_StandardScaler = make_pipeline(StandardScaler())
    preproc_RobustScaler = make_pipeline( RobustScaler())

    min_max, stand, robus = list_scaller()

    preproc = make_column_transformer(
        (preproc_MinMaxScaler, min_max),
        (preproc_StandardScaler, stand),
        (preproc_RobustScaler, robus),
        remainder="passthrough")

    return preproc











def list_scaller():

    min_max = ['t - 1', 't - 2', 't - 3', 't - 4', 't - 5', 't - 6', 't - 7', 't - 8', 't - 9',
       't - 10', 't - 11', 't - 12', 't - 13', 't - 14', 't - 15', 't - 16',
       't - 17', 't - 18', 't - 19', 't - 20', 't - 21', 't - 22', 't - 23',
       't - 24', 't - 25', 't - 26', 't - 27', 't - 28', 't - 29', 't - 30']
    stand = ['T2MDEW','T2M_RANGE']
    robus = ['T2M', 'T2MWET', 'TS', 'T2M_MAX', 'T2M_MIN','QV2M', 'RH2M', 'PRECTOTCORR', 'PS',
            'WS10M', 'WS50M']

    return min_max, stand, robus
