import pandas as pd
import os

def filter_dfs():
    directory = 'Crypto/data/'
    if not os.path.exists(directory):
        print("Directory does not exist:", directory)
        return
    
    files = os.listdir(directory)
    if not files:
        print("No files found in the directory.")
        return
    
    for file in files:
        filepath = os.path.join(directory, file)
        try:
            df = pd.read_csv(filepath)
            if len(df) < 100:
                os.remove(filepath)
                print(f'{file} is too short and has been deleted')
        except Exception as e:
            print(f'Error processing {file}: {e}')

filter_dfs()
