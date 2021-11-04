# -*- coding: utf-8 -*-

"""
Training ML model module.
"""

import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

with open('config.json', 'r') as f:
    config = json.load(f) 
dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


def get_vars_from_pandas(pandas_df, target_col=None,
                         predictors_list=['lastmonth_activity',
                                          'lastyear_activity',
                                          'number_of_employees']):
    """Get X and y (if provided) from a given pandas data frame."""
    X = pandas_df[predictors_list].to_numpy()
    if target_col is None:
        return X
    y = pandas_df[target_col].to_numpy()
    return X, y


def train_model(pandas_df, model_dir, target_col='exited', val_size=0.2):
    """
    Script for training logistic regression model and saving to model_dir.
    """
    
    X, y = get_vars_from_pandas(pandas_df, target_col)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=0
    )
    
    # use logistic regression for training
    lr_model = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
    )
    
    # fit the logistic regression to the training data
    lr_model.fit(X_train, y_train)
    
    # get F1 score for the fitted model
    print(f'F1 Score for training: {f1_score(y_train, lr_model.predict(X_train))}')
    print(f'F1 Score for validation: {f1_score(y_val, lr_model.predict(X_val))}')
    
    # write the trained model to a file called trainedmodel.pkl
    with open(os.path.join(model_dir, 'trainedmodel.pkl'), 'wb') as model_file:
        pickle.dump(lr_model, model_file)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    train_model(df, model_path)
