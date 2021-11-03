
import json
import os
from scoring import score_model
"""
import training
import scoring
import deployment
import diagnostics
import reporting
"""

with open('config.json', 'r') as f:
    config = json.load(f)
prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])


# Check for new data and if found run ingestion module.
with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'r') as f:
    ingested_files = f.readlines()
ingested_files = [file.rstrip() for file in ingested_files]
current_files = [os.path.join(input_folder_path, file) 
                 for file in os.listdir(input_folder_path)
                 if file.endswith('.csv')]

if (set(current_files) - set(ingested_files)):
    print('Running ingestion module.')
    os.system("python ingestion.py")
else:
    print('Exit.')
    exit(0)


# Check for model drift and if found redeploy a model.
with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
    latest_score = float(f.read())
current_score = score_model(prod_run=True)

if current_score < latest_score:
    print('Found model drift.')
    os.system('python training.py')
    score_model() # get the latest score for the newly trained model
    os.system('python deployment.py')
else:
    print('Exit.')
    exit(0)


# Run diagnostics and reporting for redeployed model.
print('Diagnostics.')
os.system('python apicalls.py')
print('Reporting.')
os.system('python reporting.py')
