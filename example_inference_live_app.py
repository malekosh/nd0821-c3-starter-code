import requests
import json

ata ={'age': 25,
    'workclass': 'State-gov',
    'fnlwgt':  77516,
    'education': 'Bachelors',
    'education_num': 13,
    'marital_status': 'Never-married',
    'occupation': 'Exec-managerial',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'captial_gain': 100000,
    'capital_loss': 0,
    'hours_per_week': 40,
    'native_country': 'United-States'
      }

r = requests.post("https://my-mlops-app.herokuapp.com/inference/", data=json.dumps(data))

print(r.status_code)
print(r.json())