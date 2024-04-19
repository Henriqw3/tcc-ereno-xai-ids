import os
import pandas as pd
from scipy.io import arff

def convert_arff_to_csv(arff_file_path, csv_file_path):
    print('entrou convert')
    try:
        dt = arff.loadarff(arff_file_path)
        
        df = pd.DataFrame(dt[0])
        
        for column in df.select_dtypes([object]):
            df[column] = df[column].str.decode('utf-8')
        
        # Converte floats inteiros(ex: 1.0 or 42.0, etc) para inteiros
        for column in df.select_dtypes(include='float64'):
            df[column] = df[column].apply(lambda x: '{:.0f}'.format(x) if isinstance(x, float) and x.is_integer() else x)
        
        # Salva o DataFrame como CSV
        df.to_csv(csv_file_path, index=False)
        
        print(f"Arquivo ARFF '{arff_file_path}' convertido para CSV '{csv_file_path}'")
        return 0
    except Exception as e:
        print(e)
        return -1