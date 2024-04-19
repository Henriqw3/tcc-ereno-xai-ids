# Author: Henrique Corrêa De Oliveira
# Creation date: since 2023-12-07
# Description: Main code that will perform the classifications, in addition to XAI techniques

#imports
import os
import pandas as pd
import numpy as np
from utils.functions_feature_engineering import preprocess_data, load_data, calculate_metrics_multiclass, calculate_metrics_binary
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import seaborn as sns
import time
import shap
from sklearn.impute import SimpleImputer
import datetime
#import tensorflow as tf

random_seed = 42
timestamp_fig = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # To save figures NOT rewriting them

# Dicionário para armazenar modelos treinados
trained_models = {}

def train_random_forest(X_train, y_train):
    if 'Random Forest' not in trained_models:
        model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
        model.fit(X_train, y_train)
        trained_models['Random Forest'] = model
    return trained_models['Random Forest']

def train_decision_tree(X_train, y_train):
    if 'Decision Tree' not in trained_models:
        model = DecisionTreeClassifier(random_state=random_seed)
        model.fit(X_train, y_train)
        trained_models['Decision Tree'] = model
    return trained_models['Decision Tree']

def train_xgboost(X_train, y_train):
    if 'XGBoost' not in trained_models:
        model = xgb.XGBClassifier(objective="multi:softprob", num_class=7, random_state=random_seed)
        model.fit(X_train, y_train)
        trained_models['XGBoost'] = model
    return trained_models['XGBoost']

def train_catboost(X_train, y_train):
    if 'CatBoost Classifier' not in trained_models:
        catboost_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'random_seed': random_seed,
            'class_weights': [1] * 7
        }
        model = CatBoostClassifier(**catboost_params)
        model.fit(X_train, y_train, verbose=False)
        trained_models['CatBoost Classifier'] = model
    return trained_models['CatBoost Classifier']

def train_svm(X_train, y_train):
    if 'SVM' not in trained_models:
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        trained_models['SVM'] = model
    return trained_models['SVM']

def train_knn(X_train, y_train):
    if 'KNN' not in trained_models:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        trained_models['KNN'] = model
    return trained_models['KNN']

def train_catboost_regressor(X_train, y_train):
    if 'CatBoost Regressor' not in trained_models:
        model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=random_seed)
        model.fit(X_train, y_train, verbose=False)
        trained_models['CatBoost Regressor'] = model
    return trained_models['CatBoost Regressor']

def train_naive_bayes(X_train, y_train):
    if 'Naive Bayes' not in trained_models:
        model = GaussianNB()
        model.fit(X_train, y_train)
        trained_models['Naive Bayes'] = model
    return trained_models['Naive Bayes']


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
    except:
        try:
            imputer = SimpleImputer(strategy='mean')
            X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)
            y_pred = model.predict(X_test)
        except:
            print("\n\n\n\n\nErro Pridição do Modelo\n\n\n\n")
            exit(0)
    finally:
        if len(np.unique(y_test)) > 2:
            accuracy, precision, recall, f1, conf_matrix = calculate_metrics_multiclass(y_test, y_pred)
        else:
            if not np.array_equal(np.unique(y_pred), np.array([0, 1])):
                # Se não for binário, passa por um threshold padrão
                y_pred= (y_pred > 0.5).astype(int)
                accuracy, precision, recall, f1, conf_matrix = calculate_metrics_binary(y_test, y_pred)

    print(f'\nAcurácia: {accuracy}')
    print(f'Precisão: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')

    return accuracy, precision, recall, f1, conf_matrix, y_pred



#Função Principal

if __name__ == '__main__':

    path_save_graphics = os.getcwd() + "\\graphics\\"
    path_save_metrics = os.getcwd() + "\\graphics\\Metricas\\"
    path_save_data = os.getcwd() + "\\data_samples\\"
    dt_train = '/train/train_all.csv'
    dt_test = '/test/test_all.csv'

    tempo_inicial = time.time() # tempo em segundos

    #---------- Carrega e pré-processa os dados --------------#

    # Carrega todo dataset
    X_train, y_train, X_test, y_test = load_data(dt_train, dt_test)

    tempo_final = time.time()
    print(f">> Carregou os dados em: {(tempo_final - tempo_inicial):.2f} segundos")
    
    #debug load
    pd.set_option('display.max_columns', None)
    print("\n\ndebug load:\n",X_train.isnull().sum())
    print("X_train Head: \n", X_train.head())
    print("y_train Head: \n", y_train.head())
    hist_test = X_test.hist(figsize=(15,30),layout=(9,3))
    plt.savefig(f"graphics/histogram/hist_test_{timestamp_fig}.png")
    plt.close()
    hist_train = X_train.hist(figsize=(15,30),layout=(9,3))
    plt.savefig(f"graphics/histogram/hist_train_{timestamp_fig}.png")
    plt.close()
    print("\n")

    y_train, y_test, X_train, X_test, le, dic_number_of_class = preprocess_data(X_train, y_train, X_test, y_test)
    
    #debug preprocess
    pd.set_option('display.max_columns', None)
    print("\n\ndebug preprocess:\n",X_train.isnull().sum())
    print("X_train Head: \n", X_train)
    print("y_train Head: \n", y_train.head())
    print("\n")

    tempo_final = time.time()
    print(f">> Normalizou os dados em: {(tempo_final - tempo_inicial):.2f} segundos")


    #-------------- Treinamento de modelo --------------------#
    
    exclude_columns = ['time',
                       'GooseTimestamp',
                       'window_size',
                       'window_min_timestamp',
                       'window_max_timestamp'
                       ]

    X_train = X_train.drop(columns=exclude_columns, axis=1)
    X_test = X_test.drop(columns=exclude_columns, axis=1)
    print('LAST_DESCRIBE_DF', X_test.describe())

    

    models = {
        'Random Forest': train_random_forest(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train),
        'KNN': train_knn(X_train, y_train),
        'Decision Tree': train_decision_tree(X_train, y_train),
        'CatBoost Classifier': train_catboost(X_train, y_train),
        'SVM': train_svm(X_train, y_train),
        #'CatBoost Regressor': train_catboost_regressor(X_train, y_train),
        #'Naive Bayes': train_naive_bayes(X_train, y_train)
    }

    results = {}

    for name, model in models.items():
        accuracy, precision, recall, f1, conf_matrix, y_pred = evaluate_model(model, X_test, y_test)
        results[name] = (accuracy, precision, recall, f1, conf_matrix)


    # Métricas dos ataques
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    attacks = list(dic_number_of_class.values())

    colors = ['darkcyan', 'turquoise', 'mediumspringgreen', 'aqua']

    # Coleta os resultados para cada métrica e classificador
    results_metrics = {metric: [] for metric in metrics}
    for metric in metrics:
        for model_name in models:
            # valores reais da métrica para o modelo atual
            metric_value = results[model_name][metrics.index(metric)]
            results_metrics[metric].append(metric_value)

    #---------------- Plotagem Gráficos de Métricas e Matriz de Confusão --------------------#

    width = 0.2  # Largura das barras
    x = np.arange(len(models))  # Posições dos grupos

    # Define os limites do subplot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    fig.subplots_adjust(left=0.06, bottom=0.347, right=0.98, top=0.935)

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, [results_metrics[metric][j] * 100 for j in range(len(models))], width, label=metric, color=colors[i])
        # Adiciona rótulos de porcentagem embaixo de cada barra
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Deslocamento vertical
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=6)
    #Plota Métricas Classificadores
    ax.set_xlabel('Classifier')
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models.keys())
    ax.set_ylim(50, 110)  #limite de 50% a 100% para eixo y
    ax.legend()
    plt.savefig(os.path.join(path_save_metrics, f"metrics_classifiers_{timestamp_fig}.png"))
    plt.close()


    # Plota matriz de confusão
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=100)
    fig.subplots_adjust(left=0.07, bottom=0.1, right=0.87, top=0.95, wspace=0.077, hspace=0.4)
    axes = axes.flatten()

    class_labels = list(dic_number_of_class.keys())  # Obtém os nomes das classes do dicionário

    for i, (name, result) in enumerate(results.items()):
        conf_matrix = result[4]  # Obtém a matriz de confusão do resultado
        print(i)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_labels, yticklabels=class_labels)

        axes[i].set_title(f'Confusion Matrix - {name}')
        axes[i].set_xlabel('Predicted Labels')
        axes[i].set_ylabel('True Labels')

    plt.tight_layout()
    plt.savefig(os.path.join(path_save_metrics, f"matrix_confusion_classifiers_{timestamp_fig}.png"))
    plt.close()


    tempo_final = time.time()
    print(f">> Treinou o modelo e gerou as métricas em: {(tempo_final - tempo_inicial):.2f} segundos")

    
    #---------------- Plotagem Gráficos SHAP --------------------#

    #dic_number_of_class = {0: 'high_StNum', 1: 'injection', 2: 'inverse_replay', 
    #                       3: 'masquerade_fake_fault', 4: 'normal', 
    #                       5: 'poisoned_high_rate', 6: 'random_replay'}

    shap.initjs()
    show_plot = False
    for nm_classifier, model in models.items():
        try:
            explainer = shap.TreeExplainer(model)  # Usar TreeExplainer para modelos baseados em árvores
            shap_values = explainer.shap_values(X_test)

            feature_names = X_test.columns.tolist()

            shap_exp = shap.Explanation(values=shap_values, data=X_test.values, feature_names=feature_names)
            '''
            print(feature_names)
            print(f'SHAP VALUES in {nm_classifier}:\n', shap_values, '\n')
            print(f'SHAP EXPLANATION in {nm_classifier}:\n', shap_exp, '\n')
            '''
            shap.summary_plot(shap_values, X_test, class_inds="original", class_names=list(dic_number_of_class.values()), show=show_plot)
            plt.title(f'SHAP Summary Plot - {nm_classifier}')
            plt.savefig(f'graphics/Classifiers/{nm_classifier}/Summary/SHAP_summary_plot_{nm_classifier}_{timestamp_fig}.png')
            plt.close()


            variable_focus = ['dt_clock', 'window_skewness', 'window_kurtosis', 'window_mean_time', 'timestampDiff']
            variable_interaction = ['StNum', 'sqDiff', 'confRev', 'cbStatus', 'cbStatusDiff', 'SqNum', 'stDiff', 'window_kurtosis', 'window_skewness', 'window_mean_time', 'dt_clock', 'timestampDiff']
            idx_atack = 0
            for idx_atack, name_atack in dic_number_of_class.items():
                for var_focus in variable_focus:
                    for var_itc in variable_interaction:
                        if var_focus == var_itc:
                            continue
                        shap.dependence_plot(var_focus, shap_values[idx_atack], X_test, interaction_index=var_itc, show=show_plot)
                        plt.savefig(os.path.join(path_save_graphics, f"Classifiers\\{nm_classifier}\\Dependence\\{name_atack}_dpplot_X-{var_focus}_Y-{var_itc}_{timestamp_fig}.png"))
                        plt.close()


            #plot = shap.force_plot(explainer.expected_value[class_index], shap_values[class_index][0, :], X_test.iloc[0, :], show=True)
            #shap.save_html(f'graphics/Classifiers/{nm_classifier}/Force/force_plot_{nm_classifier}_{timestamp_fig}.html', plot)
            class_index = 0
            for class_index, name_atack in dic_number_of_class.items():
                force_plot = shap.force_plot(explainer.expected_value[class_index], shap_values[class_index][0, :], X_test.iloc[0, :],
                                            matplotlib=True, figsize=(60, 8), show=show_plot)
                plt.subplots_adjust(left=0.7, right=0.95, top=0.9, bottom=0.1, wspace=0.2, hspace=0.1)
                plt.savefig(f'graphics/Classifiers/{nm_classifier}/Force/force_plot_{nm_classifier}_{timestamp_fig}.png')
                plt.close()

        except Exception as e:
            print(f"Erro ao gerar explicações SHAP para o modelo {nm_classifier}: {str(e)}")
            continue


####################

    '''
    #XGBoosting
    dt = xgb.DMatrix(X_train, label=y_train.values)
    dv = xgb.DMatrix(X_test, label=y_test.values)

    num_classes = len(np.unique(y_train))
    if num_classes == 2:
        params = {
            "objective": "binary:logistic",
            "base_score": np.mean(y_train),
            "eval_metric": "logloss",
        }
    else:
        params = {
            "objective": "multi:softmax",
            "num_class": num_classes,
            "base_score": np.mean(y_train),
            "eval_metric": "mlogloss",
        }
    
    model = xgb.train(
        params,
        dt,
        num_boost_round=10,
        evals=[(dt, "train"), (dv, "test")],
        early_stopping_rounds=5,
        verbose_eval=25,
    )
    '''

    '''
    ################ MODELO RNN #######################
    # Hiperparâmetros
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    hidden_dim = 128
    learning_rate = 0.001
    epochs = 10
    batch_size = 64

    # Definição do modelo RNN
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((1, input_dim)),
        tf.keras.layers.LSTM(hidden_dim),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    # Compilação do modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Treinamento do modelo
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    tempo_final = time.time()
    print(f">> Treinou o modelo em: {(tempo_final - tempo_inicial):.2f} segundos")

    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f'Acurácia: {accuracy}')
    print(f'Loss: {loss}')
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plotar a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    tempo_final = time.time()
    print(f">> Avaliou o modelo em: {(tempo_final - tempo_inicial):.2f} segundos")
    #####################################################################################
    '''