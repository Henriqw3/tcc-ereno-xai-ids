import os
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import shutil
from utils.convert_file import convert_arff_to_csv

current_dir = os.getcwd()

pasta_monitorada = os.path.join(current_dir, 'inputs_ereno_ui')

padroes = ['*.csv', '*.arff']

def executar_codigo(evento):
    try:
        print('entrou')
        # Obtem o caminho do arquivo adicionado
        caminho_arquivo = evento.src_path
        
        if caminho_arquivo.endswith('.csv'):
            print('entrou IF1')
            #Cria pasta csv se não existe
            csv_folder= os.path.join(current_dir, 'Data', 'csv')
            if not os.path.exists(csv_folder):
                print(f'Criou a pasta {csv_folder}')
                os.makedirs(csv_folder)

            # Move o arquivo CSV para a pasta Data/csv
            path_to_move_csv = os.path.join('Data', 'csv', os.path.basename(caminho_arquivo))
            shutil.move(caminho_arquivo, path_to_move_csv)
            print(f"Arquivo CSV movido para '{path_to_move_csv}'")

        elif caminho_arquivo.endswith('.arff'):
            print('entrou IF2')
            # Cria pasta csv se não existir
            csv_file_name = os.path.basename(caminho_arquivo).replace('.arff', '.csv')
            csv_folder= os.path.join(current_dir, 'Data', 'csv')
            if not os.path.exists(csv_folder):
                print(f'Criou a pasta {csv_folder}')
                os.makedirs(csv_folder)

            print(f'convert_Arff_To_csv({caminho_arquivo}, {os.path.join(csv_folder, csv_file_name)})')

            sucess = convert_arff_to_csv(caminho_arquivo, os.path.join(csv_folder, csv_file_name))

            if sucess == 0:
                print('sucess')
                #Cria pasta arff se não existir
                path_to_move_arff = os.path.join(current_dir, 'Data', 'arff')
                if not os.path.exists(path_to_move_arff):
                    print(f'Criou a pasta {path_to_move_arff}')
                    os.makedirs(path_to_move_arff)
                # Move o arquivo ARFF para a pasta Data/arff
                path_to_move_arff = os.path.join(path_to_move_arff, os.path.basename(caminho_arquivo))
                shutil.move(caminho_arquivo, path_to_move_arff)
                print(f"Arquivo ARFF movido para '{path_to_move_arff}'")

        print('end function')
    except Exception as e:
        print('exc')
        print(e)


# Manipulador de eventos que executa a função quando um novo arquivo for adicionado
manipulador_eventos = PatternMatchingEventHandler(padroes, None, False, True)
manipulador_eventos.on_created = executar_codigo

# observador que monitore a pasta
observador = Observer()
observador.schedule(manipulador_eventos, pasta_monitorada, recursive=True)

# Inicia observador
observador.start()

try:
    while True:
        time.sleep(1)
        print('esperando')
except KeyboardInterrupt:
    observador.stop()

observador.join()