from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import model_predictions

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 
dataset_csv_path = os.path.join(config['output_folder_path'])

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


"""
#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():        
    #check the score of the deployed model
    return #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    return #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    return #add return value for all diagnostics
"""


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
