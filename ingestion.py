# -*- coding: utf-8 -*-

"""
Data ingestion module.
"""

import json
import os

import pandas as pd

with open('config.json','r') as f:
    config = json.load(f)
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def merge_multiple_dataframe(directory):
    """
    Check for datasets, compile them together, and write to an output file.
    """
    csv_files = [file for file in os.listdir(directory) 
                 if file.endswith('.csv')]

    dfs_list = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file))
        dfs_list.append(df)

    df_concatenated = pd.concat(dfs_list, ignore_index=True, axis=0)
    df_concatenated.drop_duplicates(inplace=True, ignore_index=True)
    
    df_dict = {
        'data': df_concatenated, 
        'files': [os.path.join(directory, file) for file in csv_files]
    }

    return df_dict


if __name__ == '__main__':
    data_records_dict = merge_multiple_dataframe(input_folder_path)
    df_final, records = data_records_dict['data'], data_records_dict['files']
    df_final.to_csv(os.path.join(output_folder_path, 'finaldata.csv'),
                    index=False)
    with open(
        os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w'
        ) as record_file:
        for line in records:
            record_file.write(line + '\n')
