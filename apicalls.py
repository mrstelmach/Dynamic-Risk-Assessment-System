import os
import json
import requests

with open('config.json', 'r') as f:
    config = json.load(f)
output_path = os.path.join(config['output_model_path'])

URL = "http://127.0.0.1:8000"

response1 = requests.post(
    f'{URL}/prediction',
    json={'data_path': 'testdata/testdata.csv'}
    ).text
response2 = requests.get(f'{URL}/scoring').text
response3 = requests.get(f'{URL}/summarystats').text
response4 = requests.get(f'{URL}/diagnostics').text

responses = f'{response1}\n{response2}\n{response3}\n{response4}'

with open(os.path.join(output_path, 'apireturns.txt'), 'w') as api_file:
    api_file.write(responses)
