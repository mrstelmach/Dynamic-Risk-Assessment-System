from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


def get_vars_from_pandas(pandas_df, target_col='exited',
                         predictors_list=['lastmonth_activity',
                                          'lastyear_activity',
                                          'number_of_employees']):
    """Get X and y from provided pandas data frame."""
    X, y = (pandas_df[predictors_list].to_numpy(),
            pandas_df[target_col].to_numpy())
    return X, y


def train_model(pandas_df, model_dir):
    """
    Script for training logistic regression model and saving to model_dir.
    """
    
    X, y = get_vars_from_pandas(pandas_df)
    
    # use logistic regression for training
    lr_model = LogisticRegression(
        C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='auto', n_jobs=None, penalty='l2',
        random_state=0, solver='liblinear', tol=0.0001, verbose=0,
        warm_start=False
    )
    
    # fit the logistic regression to data
    lr_model.fit(X, y)
    
    # write the trained model to a file called trainedmodel.pkl
    with open(os.path.join(model_dir, 'trainedmodel.pkl'), 'wb') as model_file:
        pickle.dump(lr_model, model_file)


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f) 
    dataset_csv_path = os.path.join(config['output_folder_path']) 
    model_path = os.path.join(config['output_model_path']) 
    
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    train_model(df, model_path)
