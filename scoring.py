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


def score_model(model_path, data_path, result_path=None, target_col='exited'):
    """
    Function takes a trained model, loads test data, calculates 
    an F1 score for the model relative to the test data and writes the result 
    to the file.
    """
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    df = pd.read_csv(data_path)
    X, y = get_vars_from_pandas(df, target_col)
    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)
    
    if result_path is not None:
        with open(result_path, 'w') as result_file:
            result_file.write(str(f1_score))

    return f1_score


if __name__ == "__main__":
    with open('config.json','r') as f:
        config = json.load(f)
    test_data_path = os.path.join(config['test_data_path'])
    output_model_path = os.path.join(config['output_model_path'])
    
    score_model(
        os.path.join(output_model_path, 'trainedmodel.pkl'),
        os.path.join(test_data_path, 'testdata.csv'),
        os.path.join(output_model_path, 'latestscore.txt')
    )
