import os
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import datetime
from scipy.stats import skew, kurtosis
import re

def load_data(dt_train, dt_test):
    # Carregamento de dados
    train_df = pd.read_csv('data/' + dt_train, sep=',')
    test_df = pd.read_csv('data/' + dt_test, sep=',')

    # Colunas enriquecidas para remover
    columns_to_remove = ['stDiff', 'sqDiff', 'gooseLenghtDiff', 'cbStatusDiff', 'apduSizeDiff',
                         'frameLengthDiff', 'timestampDiff', 'tDiff', 'timeFromLastChange',
                         'delay', 'isbARms', 'isbBRms', 'isbCRms', 'ismARms', 'ismBRms', 'ismCRms',
                         'ismARmsValue', 'ismBRmsValue', 'ismCRmsValue', 'csbArms', 'csvBRms',
                         'csbCRms', 'vsmARms', 'vsmBRms', 'vsmCRms', 'isbARmsValue', 'isbBRmsValue',
                         'isbCRmsValue', 'vsbARmsValue', 'vsbBRmsValue', 'vsbCRmsValue',
                         'vsmARmsValue', 'vsmBRmsValue', 'vsmCRmsValue', 'isbATrapAreaSum',
                         'isbBTrapAreaSum', 'isbCTrapAreaSum', 'ismATrapAreaSum', 'ismBTrapAreaSum',
                         'ismCTrapAreaSum', 'csvATrapAreaSum', 'csvBTrapAreaSum', 'vsbATrapAreaSum',
                         'vsbBTrapAreaSum', 'vsbCTrapAreaSum', 'vsmATrapAreaSum', 'vsmBTrapAreaSum',
                         'vsmCTrapAreaSum', 'gooseLengthDiff']

    # Remoção de colunas enriquecidas ou com NaN
    train_df = train_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')
    test_df = test_df.dropna(axis=1)  # .drop(columns=columns_to_remove, errors='ignore')

    # Separação de features e labels
    X_train = train_df.drop(columns=['@class@'])
    y_train = train_df['@class@']
    X_test = test_df.drop(columns=['@class@'])
    y_test = test_df['@class@']

    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, y_test):

    try:
        # Colunas de tempo
        #time_columns = X_train_time.select_dtypes(include=['datetime64'])
        time_columns = ['time', 'GooseTimestamp', 'timeFromLastChange'] # setado manualmente por não serem do tipo datetime originalmente
        # Cria dataframe com colunas dos dados de tempo
        X_train_time = pd.DataFrame(X_train[time_columns], columns = time_columns)
        X_test_time = pd.DataFrame(X_test[time_columns], columns = time_columns)
    except Exception as e:
        print(e)
        exit(0)

    #Adicionar ao dataset uma linha temporal com estatísticas de janelas de 2 segundos(2 second non overlapping moving in)
    X_train_time = T_SNOMI_enrich(X_train_time)
    X_test_time = T_SNOMI_enrich(X_test_time)

    X_train = X_train.drop(time_columns, axis=1)
    X_test = X_test.drop(time_columns, axis=1)
    # Concatena os dataset original com as features extraídas do enriquecimento temporal
    X_train = pd.concat([X_train_time, X_train], axis=1)
    X_test = pd.concat([X_test_time, X_test], axis=1)

    # Identificar colunas numéricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Testando Selecionando colunas numéricas, excluindo as colunas de tempo
    #num_cols = [col for col in X_train.select_dtypes(include=[np.number]).columns.tolist() if col not in time_columns]
    # Selecionando colunas categóricas, excluindo as colunas especiais
    #cat_cols = [col for col in X_train.select_dtypes(include=['object']).columns.tolist() if col not in time_columns]

    print("Colunas numéricas: ", num_cols)
    print("Colunas categóricas: ", cat_cols)
    
    # Utilizar StandardScaler para normalizar os dados numéricos
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    #experimento filtro
    #X_train = X_train.drop(['window_mean_time', 'window_skewness','window_kurtosis'], axis=1)
    #X_test = X_test.drop(['window_mean_time', 'window_skewness','window_kurtosis'], axis=1)
    #X_train = pd.concat([X_train_time['window_mean_time'],X_train_time['window_skewness'],X_train_time['window_kurtosis'], X_train], axis=1)
    #X_test = pd.concat([X_test_time['window_mean_time'],X_test_time['window_skewness'],X_test_time['window_kurtosis'], X_test], axis=1)


    # Inicializar listas vazias para armazenar os nomes das colunas categóricas
    cat_column_names = []

    # Utilizar OneHotEncoder para colunas categóricas
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary')
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])
        
        cat_column_names = encoder.get_feature_names_out()

        # Criar DataFrames para os dados categóricos
        X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_column_names)
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_column_names)

        # Transformar em DataFrames do Pandas
        X_train_num_df = pd.DataFrame(X_train[num_cols], columns=num_cols)
        X_test_num_df = pd.DataFrame(X_test[num_cols], columns=num_cols)
        
        # Concatenar dados numéricos e categóricos
        X_train = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
        X_test = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

    # Inicializar o LabelEncoder para os rótulos
    le = LabelEncoder()

    # Transformar y_train e y_test para numérico
    if y_train.dtype == 'object':
        y_train = le.fit_transform(y_train)
    if y_test.dtype == 'object':
        y_test = le.transform(y_test)  # usar o mesmo encoder para garantir uma codificação consistente

    # Concatena os dados pre processados numéricos ou categóricos aos de processamento de tempo
    #X_train = pd.concat([X_train_time, X_train], axis=1)
    #X_test = pd.concat([X_test_time, X_test], axis=1)

    # Salvar os dados processados em CSV, se fornecido o caminho de saída
    processed_data = pd.concat([pd.Series(y_train),X_train], axis=1)
    processed_data.to_csv(os.getcwd() + '\\data_samples\\preprocess_train.csv', index=False)
    processed_data = pd.concat([pd.Series(y_test),X_test], axis=1)
    processed_data.to_csv(os.getcwd() + '\\data_samples\\preprocess_test.csv', index=False)

    # Mapear os valores numéricos para as classes originais
    dic_number_of_class = {num: cls for num, cls in enumerate(le.classes_)}

    # Retornar os dados em objetos pandas e o LabelEncoder
    return pd.Series(y_train), pd.Series(y_test), X_train, X_test, le, dic_number_of_class

def T_SNOMI_enrich(df):

    try:
        # Convertendo os timestamps para o formato de datetime
        df['dt_time'] = df['time'].apply(lambda t: datetime.datetime.fromtimestamp(t/1000) if len(str(t)) >= 12 else datetime.datetime.fromtimestamp(t))
        df['dt_GooseTimestamp'] = df['GooseTimestamp'].apply(lambda t: datetime.datetime.fromtimestamp(t/1000) if len(str(t)) >= 12 else datetime.datetime.fromtimestamp(t))
        df['dt_clock'] = df['dt_GooseTimestamp'].apply(lambda t: float(t.strftime('%M.%S')) + (t.microsecond / 1e6))
    except:
        exit(0)

    # Ordenando o DataFrame pelos datetime
    df = df.sort_values(by='dt_time')

    # Definindo a janela de tempo de 2 segundos
    window_size = pd.Timedelta(seconds=2)

    # Criando uma lista para armazenar os grupos de linhas
    groups = []

    # Iterando sobre as linhas do DataFrame para agrupá-las
    current_group = [df.iloc[0]]
    for i in range(1, len(df)):
        if df['dt_time'].iloc[i] - current_group[-1]['dt_time'] <= window_size:
            #print(df['dt_time'].iloc[i], '-', current_group[-1]['dt_time'], '<=', window_size)
            current_group.append(df.iloc[i])
        else:
            groups.append(current_group)
            current_group = [df.iloc[i]]

    # Adicionando o último grupo
    groups.append(current_group)

    # Criando DataFrame de estatísticas do grupo
    group_stats = pd.DataFrame(columns=['window_index', 'window_mean_time', 'window_variance_time',
                                        'window_std_deviation','window_min_timestamp', 'window_max_timestamp',
                                        'window_size', 'window_skewness', 'window_kurtosis'])

    group_stats_list = []

    # Calcula as estatísticas para cada grupo
    for idx, group in enumerate(groups):
        timestamps_init = [t['GooseTimestamp'] for t in group]
        timestamps_end = [t['time'] for t in group]
        durations = [d['timeFromLastChange'] for d in group]
        group_size = len(group)
        mean_time_diff = sum((x['dt_GooseTimestamp'] - x['dt_time']).total_seconds() for x in group) / group_size
        var_time_diff = sum(((x['dt_GooseTimestamp'] - x['dt_time']).total_seconds() - mean_time_diff) ** 2 for x in group) / group_size
        std_time_diff = var_time_diff ** 0.5  # Convertendo a soma de quadrados para o desvio padrão
        min_timestamp = min(timestamps_init)
        max_timestamp = max(timestamps_end)
        skewness = skew(durations) if group_size > 1 and var_time_diff > 0.0 else skew([min_timestamp, max_timestamp]) 
        kurt = kurtosis(durations) if group_size > 1 and var_time_diff > 0.0 else kurtosis([min_timestamp, max_timestamp]) # grau de achatamento de uma distribuição

        # Cria DataFrame de estatísticas do grupo
        group_stats_df = pd.DataFrame({'window_index': [idx],
                                    'window_mean_time': [mean_time_diff],
                                    'window_variance_time': [var_time_diff],
                                    'window_std_deviation': [std_time_diff],
                                    'window_min_timestamp': [min_timestamp],
                                    'window_max_timestamp': [max_timestamp],
                                    'window_size': [group_size],
                                    'window_skewness': [skewness],
                                    'window_kurtosis': [kurt]})
        
        # Adiciona DataFrame individual à lista
        group_stats_list.append(group_stats_df)


    # Concatena todos os DataFrames individuais de estatísticas do grupo
    group_stats = pd.concat(group_stats_list, ignore_index=True)


    # Realiza o join entre o DataFrame original e as estatísticas do grupo
    df['window_index'] = -1  # Inicializar o grupo_index com -1 para indicar que a linha não pertence a nenhum grupo

    for idx, group in enumerate(groups):
        for row in group:
            df.loc[df['time'] == row['time'], 'window_index'] = idx

    # Merge com o DataFrame de estatísticas do grupo
    df = df.merge(group_stats, on='window_index', how='left')

    # Exibe o DataFrame resultante
    #pd.set_option('display.max_columns', None)
    #print(df)
    #df = df.drop(['window_index', 'dt_time', 'dt_GooseTimestamp', 'time', 'GooseTimestamp', 'window_size'], axis=1)
    df = df.drop(['window_index'], axis=1)
    print("DESCRIBE T_SNOMI:\n",df.describe())
    return df


def calculate_metrics_manual(y_test, y_pred):
    # Calcula matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Extrai TP, FP, TN, FN da matriz de confusão
    TN, FP, FN, TP = conf_matrix.ravel()
    
    # Calcula acurácia
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    # Calcula precisão
    precision = TP / (TP + FP)
    
    # Calcula recall
    recall = TP / (TP + FN)
    
    # Calcula F1-score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1, conf_matrix


def calculate_metrics(y_test, y_pred):
    if y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
        print('new',y_pred)

    num_classes = len(np.unique(y_test))
    if num_classes > 2:
        return calculate_metrics_multiclass(y_test, y_pred)
    else:
        return calculate_metrics_binary(y_test, y_pred)


def calculate_metrics_multiclass(y_test, y_pred):
    # Calcula matriz de confusão, acurácia, precisão, recall e F1-score para várias classes
    try:
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    except:
        label_encoder = LabelEncoder()
        y_test = label_encoder.fit_transform(y_test)  # Codifica rótulos de classe em valores numéricos
        y_pred = label_encoder.transform(y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')  # Precisão média para multiclasse
        recall = recall_score(y_test, y_pred, average='macro')  # Recall médio para multiclasse
        f1 = f1_score(y_test, y_pred, average='macro')
    
    return accuracy, precision, recall, f1, conf_matrix

def calculate_metrics_binary(y_test, y_pred):
    # Calcula as métricas para classes binárias
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix


def plot_confusion_matrix(conf_matrix, dic_number_of_class, y_pred, path_save_metrics=os.getcwd()):
    #target_class = {v: k for k, v in dic_number_of_class.items()}
    #target_class = {key: dic_number_of_class[key] for key in np.unique(y_pred)}
    #target_class = dict(sorted({0 if v == 'normal' else k - min(target_class.keys()) + 1: v for k, v in target_class.items()}.items()))
    target_class = dic_number_of_class
    print(dic_number_of_class)
    print(target_class)
    df_cm = pd.DataFrame(conf_matrix, index=target_class.values(), columns=target_class.values())
    
    print(df_cm)
    # Plota a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.savefig(os.path.join(path_save_metrics, 'confusion_matrix_plot.png'))
    plt.show()
    


def plot_metrics(accuracy, precision, recall, f1, path_save_metrics=os.getcwd()):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [accuracy, precision, recall, f1]
    
    # Converte os valores para porcentagem
    values = [value * 100 for value in values]
    
    plt.figure(figsize=(10, 7))
    sns.barplot(x=metrics, y=values, palette='muted')
    plt.xlabel('Metrics')
    plt.ylabel('Percentage (%)')
    plt.title('Performance Metrics')
    if accuracy >= 0.90:
        plt.ylim(90, 100)
    else:
        plt.ylim(50, 100)
    plt.savefig(os.path.join(path_save_metrics, 'metrics_plot.png'))
    plt.show()

def find_most_frequent_label(y_test):
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    most_frequent_label = unique_labels[np.argmax(label_counts)]
    return most_frequent_label


def find_least_frequent_label(y_test):
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    least_frequent_label = unique_labels[np.argmin(label_counts)]
    return least_frequent_label


def clean_variable_name(nm_var):
    # Substitui caracteres não aceitos por "_"
    cl_var = re.sub(r'[^a-zA-Z0-9_.-]', '_', nm_var)
    return cl_var



def load_small_data(sample_size=0.2, random_seed=42, save_path=None):
    # Carregamento de dados completo
    train_df = pd.read_csv('data/test_gray.csv', sep=',')
    test_df = pd.read_csv('data/test_gray.csv', sep=',')

    # Remoção de colunas enriquecidas ou com NaN
    train_df = train_df.dropna(axis=1)
    test_df = test_df.dropna(axis=1)

    # Realiza amostragem estratificada para manter a distribuição das classes
    X_train, _, y_train, _ = train_test_split(train_df.drop(columns=['@class@']), train_df['@class@'],
                                              test_size=1 - sample_size, stratify=train_df['@class@'],
                                              random_state=random_seed)

    X_test, _, y_test, _ = train_test_split(test_df.drop(columns=['@class@']), test_df['@class@'],
                                            test_size=1 - sample_size, stratify=test_df['@class@'],
                                            random_state=random_seed)

    # Salva partes estratificadas como CSV, se o caminho de salvamento for fornecido
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        X_train.to_csv(os.path.join(save_path, 'X_train_stratified.csv'), index=False)
        pd.DataFrame(y_train, columns=['@class@']).to_csv(os.path.join(save_path, 'y_train_stratified.csv'), index=False)
        X_test.to_csv(os.path.join(save_path, 'X_test_stratified.csv'), index=False)
        pd.DataFrame(y_test, columns=['@class@']).to_csv(os.path.join(save_path, 'y_test_stratified.csv'), index=False)

    # Retorna amostra estratificada
    return X_train, y_train, X_test, y_test


def preprocess_small_data(X_train, y_train, X_test, y_test):
    # Identifica colunas numéricas e categóricas
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Trata valores ausentes (NaN) com SimpleImputer para dados numéricos
    imputer_num = SimpleImputer(strategy='mean')
    X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
    X_test[num_cols] = imputer_num.transform(X_test[num_cols])

    # Trata valores ausentes (NaN) com SimpleImputer para dados categóricos
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = imputer_cat.transform(X_test[cat_cols])

    # Utiliza StandardScaler para normalizar os dados numéricos
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Utiliza OneHotEncoder para colunas categóricas
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])

        # Recuperar os nomes das colunas após a transformação OneHotEncoder
        cat_column_names = [f"{str(col).strip()}_{str(category).strip()}" for col in cat_cols for category in encoder.categories_[0]]

        # Criar DataFrames para os dados categóricos
        X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_column_names)
        X_test_cat_df = pd.DataFrame(X_test_cat, columns=cat_column_names)

        # Transformar em DataFrames do Pandas
        X_train_num_df = pd.DataFrame(X_train[num_cols], columns=num_cols)
        X_test_num_df = pd.DataFrame(X_test[num_cols], columns=num_cols)

        # Concatenar dados numéricos e categóricos
        X_train = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
        X_test = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

    # Inicializa o LabelEncoder para os rótulos
    le = LabelEncoder()

    # Transforma y_train e y_test para numérico
    if y_train.dtype == 'object':
        y_train = le.fit_transform(y_train)
    if y_test.dtype == 'object':
        y_test = le.transform(y_test)  # usar o mesmo encoder para garantir uma codificação consistente


    # Salva os dados processados em CSV, se fornecido o caminho de saída
    processed_data = pd.concat([pd.Series(y_train),X_train], axis=1)
    processed_data.to_csv(os.getcwd() + '\\data_samples\\preprocess_small_train.csv', index=False)
    processed_data = pd.concat([pd.Series(y_test),X_test], axis=1)
    processed_data.to_csv(os.getcwd() + '\\data_samples\\preprocess_small_test.csv', index=False)


    # Retorna os dados em objetos pandas e o LabelEncoder
    return pd.Series(y_train), pd.Series(y_test), X_train, X_test, le