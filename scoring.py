# -*- coding: utf-8 -*-

"""
Module for ML model scoring.
"""

import json
import os
import pickle

import pandas as pd
from sklearn import metrics

from training import get_vars_from_pandas

with open('config.json','r') as f:
    config = json.load(f)
test_data_path = os.path.join(config['test_data_path'])
output_folder_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def score_model(target_col='exited', prod_run=False):
    """
    Function takes a trained model, loads data, calculates 
    a F1 score for the model relative to the data and writes the result 
    to the file.
    """
    if prod_run:
        model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
        data_path = os.path.join(output_folder_path, 'finaldata.csv')
    else:
        model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
        data_path = os.path.join(test_data_path, 'testdata.csv')

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    df = pd.read_csv(data_path)
    X, y = get_vars_from_pandas(df, target_col)
    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    if not prod_run:
        result_path = os.path.join(output_model_path, 'latestscore.txt')
        with open(result_path, 'w') as result_file:
            result_file.write(str(f1_score))

    return f1_score


if __name__ == "__main__":
    score_model()
