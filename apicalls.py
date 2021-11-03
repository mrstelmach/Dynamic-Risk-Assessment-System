import requests

URL = "http://127.0.0.1:8000"

response1 = requests.post(
    f'{URL}/prediction',
    json={'data_path': 'testdata/testdata.csv'}
    ).text
response2 = #put an API call here
response3 = #put an API call here
response4 = #put an API call here

#combine all API responses
responses = #combine reponses here

#write the responses to your workspace



