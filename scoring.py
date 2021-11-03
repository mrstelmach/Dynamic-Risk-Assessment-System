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
output_model_path = os.path.join(config['output_model_path'])


def score_model(target_col='exited'):
    """
    Function takes a trained model, loads test data, calculates 
    an F1 score for the model relative to the test data and writes the result 
    to the file.
    """
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    data_path = os.path.join(test_data_path, 'testdata.csv')
    result_path = os.path.join(output_model_path, 'latestscore.txt')
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    df = pd.read_csv(data_path)
    X, y = get_vars_from_pandas(df, target_col)
    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    with open(result_path, 'w') as result_file:
        result_file.write(str(f1_score))

    return f1_score


if __name__ == "__main__":
    score_model()
