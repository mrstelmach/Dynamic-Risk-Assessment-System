from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import model_predictions, dataframe_summary
from scoring import score_model

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
data_path = os.path.join(dataset_csv_path, 'finaldata.csv')

model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
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
    df = pd.read_csv(data_path)
    summary_stats = dataframe_summary(df, exclude='exited')
    return str(summary_stats)


"""
#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    return #add return value for all diagnostics
"""


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
