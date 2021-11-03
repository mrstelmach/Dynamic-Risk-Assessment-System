# -*- coding: utf-8 -*-

"""
Module for generating and storing confusion matrix.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

from diagnostics import model_predictions
from training import get_vars_from_pandas

with open('config.json','r') as f:
    config = json.load(f)
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


def score_model(data_path, model_path, save_path, target_col):
    """
    Calculate a confusion matrix using the test data and the deployed model.
    Save confusion matrix to a save_path location.
    """
    df = pd.read_csv(data_path)
    y_pred = model_predictions(df, model_path)
    _, y_true = get_vars_from_pandas(df, target_col)
    conf_mat = metrics.confusion_matrix(y_true.tolist(), y_pred)
    
    plt.figure(figsize=(7, 7))
    plt.title('Confusion matrix', fontsize=20)
    sns.heatmap(pd.DataFrame(conf_mat), cmap=plt.cm.Blues,
                cbar=False, annot=True)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':    
    score_model(
        os.path.join(test_data_path, 'testdata.csv'),
        os.path.join(output_model_path, 'trainedmodel.pkl'),
        os.path.join(output_model_path, 'confusionmatrix.png'),
        'exited'
    )
