# -*- coding: utf-8 -*-

"""
Module for data and model diagnostics.
"""

import json
import os
import pickle
import subprocess
import timeit

import numpy as np
import pandas as pd

from training import get_vars_from_pandas

with open('config.json','r') as f:
    config = json.load(f) 
dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def model_predictions(pandas_df, model_path=None, model_object=None):
    """Read the deployed model and calculate predictions for a dataset."""
    if model_object is not None:
        model_path = None
        model = model_object
    if model_path is not None:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    X = get_vars_from_pandas(pandas_df, target_col=None)
    predictions = model.predict(X).tolist()
    return predictions


def dataframe_summary(pandas_df, exclude=None):
    """calculate summary statistics of data frame."""
    df_num = pandas_df.select_dtypes(include=np.number).copy()
    if exclude is not None:
        df_num.drop(exclude, axis=1, inplace=True)

    summary_stats = []
    for num_col in df_num:
        summary_stats.append([num_col, 'mean', df_num[num_col].mean()])
        summary_stats.append([num_col, 'median', df_num[num_col].median()])
        summary_stats.append([num_col, 'std', df_num[num_col].std()])

    return summary_stats


def missing_data_pct(pandas_df):
    """Return the percentage of missing data for each column."""
    return list(pandas_df.isna().sum()/len(pandas_df))


def execution_time(list_of_scripts):
    """Calculate timings of provided scripts (defined by paths)."""
    timings = []
    for script in list_of_scripts:
        start_time = timeit.default_timer()
        os.system(f'python {script}')
        timing = timeit.default_timer() - start_time
        timings.append(timing)
    return timings


def outdated_packages_list():
    """Get a table with outdated packages."""
    outdated_pckgs = subprocess.run(
        ["pip", "list", "--outdated"], 
        capture_output=True, 
        text=True
    )
    return outdated_pckgs.stdout


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    model_predictions(
        df, os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    )
    dataframe_summary(
        df, exclude=['exited']
    )
    missing_data_pct(
        df
    )
    execution_time(
        ['ingestion.py', 'training.py']
    )
    outdated_packages_list()
