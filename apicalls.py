import requests

URL = "http://127.0.0.1:8000"

response1 = requests.post(
    f'{URL}/prediction',
    json={'data_path': 'testdata/testdata.csv'}
    ).text
response2 = requests.get(f'{URL}/scoring').text
response3 = requests.get(f'{URL}/summarystats').text
response4 = requests.get(f'{URL}/diagnostics').text

#combine all API responses
responses = #combine reponses here

#write the responses to your workspace



