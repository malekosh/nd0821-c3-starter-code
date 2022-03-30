import os
import pandas as pd
import json
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

train_csv =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/train_census.csv')

print(train_csv)

def test_get_home():
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json() == {"greetings from Malek": "This app is to infer salary category from census data"}
    
    
def test_over_50k():
    train_data = pd.read_csv(train_csv)
    train_data.drop(train_data.filter(regex="Unname"), axis=1, inplace=True)
    over = train_data[train_data['salary']=='>50K'].iloc[1]
    sample = over.to_dict()
    sample = { key.replace('_','-') : (int(val) if not isinstance(val,str) else val)  for key, val in sample.items()}
    
    response = client.post("/inference/", json=json.dumps(sample))
    assert response.status_code == 200 
    assert response.json()['pred'] == '>50K'
                    
    
    
def test_under_50k():
    train_data = pd.read_csv(train_csv)
    train_data.drop(train_data.filter(regex="Unname"), axis=1, inplace=True)
    under = train_data[train_data['salary']=='<=50K'].iloc[3]
    sample = under.to_dict()
    sample ={ key.replace('_','-') : (int(val) if not isinstance(val,str) else val)  for key, val in sample.items()}
    
    response = client.post("/inference/", json=json.dumps(sample))
    assert response.status_code == 200 
    assert response.json()['pred'] == '<=50K'