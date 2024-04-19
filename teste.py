import pandas as pd
from scipy.io import arff

#dt_train = arff.loadarff('C:/Projetos/codes_tcc/data/arff_files/train_mask.arff')
#dt_test = arff.loadarff('C:/Projetos/codes_tcc/data/arff_files/test_mask.arff')
dt_train = arff.loadarff('C:/Projetos/codes_tcc/data/Datasets/arff/train/ARFF_ORIGIN_TRAIN.arff')
dt_test = arff.loadarff('C:/Projetos/codes_tcc/data/Datasets/arff/test/ARFF_ORIGIN_TEST.arff')

df_train = pd.DataFrame(dt_train[0])
df_test = pd.DataFrame(dt_test[0])

for column in df_train.select_dtypes([object]):
    df_train[column] = df_train[column].str.decode('utf-8')

for column in df_test.select_dtypes([object]):
    df_test[column] = df_test[column].str.decode('utf-8')

'''
df_train['time'] = df_train['time'] + 1711300000000
df_train['GooseTimestamp'] = df_train['GooseTimestamp'] + 1711300000000

df_test['time'] = df_test['time'] + 1711300000000
df_test['GooseTimestamp'] = df_test['GooseTimestamp'] + 1711300000000
'''

for column in df_train.select_dtypes(include='float64'):
    df_train[column] = df_train[column].apply(lambda x: '{:.0f}'.format(x) if isinstance(x, float) and x.is_integer() else x)

for column in df_test.select_dtypes(include='float64'):
    df_test[column] = df_test[column].apply(lambda x: '{:.0f}'.format(x) if isinstance(x, float) and x.is_integer() else x)

df_train.to_csv('C:/Projetos/codes_tcc/data/Datasets/csv/train/train_origin.csv', index=False)#, float_format='%.0f')
df_test.to_csv('C:/Projetos/codes_tcc/data/Datasets/csv/test/test_origin.csv', index=False)#, float_format='%.0f')
