# -*- coding: utf-8 -*-

"""
Flask app with four endpoints for checking model and data quality.
"""

import json
import os
import pickle

import pandas as pd
from flask import Flask, request

from diagnostics import (dataframe_summary, execution_time, missing_data_pct,
                         model_predictions, outdated_packages_list)
from scoring import score_model

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

custom_data_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
custom_data = pd.read_csv(custom_data_path)

model_path = os.path.join(config['prod_deployment path'], 'trainedmodel.pkl')
with open(model_path, 'rb') as model_file:
    prediction_model = pickle.load(model_file)


@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    """Call the model prediction for a given dataset."""
    data_path = request.json.get('data_path')
    df = pd.read_csv(data_path)
    predictions = model_predictions(df, model_object=prediction_model)
    return str(predictions)


@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    """Check the score of the deployed model."""
    f1_score = score_model()
    return str(f1_score)


@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    """Check means, medians, and modes for each column."""
    summary_stats = dataframe_summary(custom_data, exclude='exited')
    return str(summary_stats)


@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():
    """Check timing, percentage of NA values and dependencies."""
    exec_time = execution_time(['ingestion.py', 'training.py'])
    miss_pct = missing_data_pct(custom_data)
    dep_check = outdated_packages_list()
    return str(f'\
execution_time: {exec_time}\n\
missing_percentage: {miss_pct}\n\
dependencies_check: \n{dep_check}'
)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
