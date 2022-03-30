import os
from typing import Union
from fastapi import FastAPI, Body

from pydantic import BaseModel, Field

import pandas as pd

from starter.ml.model import load_model, inference
from starter.ml.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    print('running dvc')
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull")!=0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/encoder.pkl')
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/model.pkl')

model = load_model(model_path)
encoder = load_model(encoder_path)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class CensusEntry(BaseModel):
    age: int = None
    workclass: str = None
    fnlgt: int = None
    education: str = None
    education_num: int = Field(None, alias='education-num')
    marital_status: str = Field(None, alias='marital-status')
    occupation: str = None
    relationship: str = None
    race: str = None
    sex: str = None
    captial_gain: int = Field(None, alias='capital-gain')
    capital_loss: int = Field(None, alias='capital-loss')
    hours_per_week: int = Field(None, alias='hours-per-week')
    native_country: str = Field(None, alias='native-country')
    
    class Config:
        allow_population_by_field_name = True

@app.get("/")
async def say_hello():
    return {"greetings from Malek": "This app is to infer salary category from census data"}




@app.post("/inference")
async def update_item(
    *,
    item: CensusEntry = Body(
        ...,
        examples={
            'example1': {
                'age': 25,
                'workclass': 'State-gov',
                'fnlgt':  76413,
                'education': 'Bachelors',
                'education_num': 9,
                'marital_status': 'Never-married',
                'occupation': 'Exec-managerial',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'captial_gain': 3000,
                'capital_loss': 0,
                'hours_per_week': 40,
                'native_country': 'United-States',
                },
            'example2': {
                'age': 28,
                'workclass': 'State-gov',
                'fnlgt':  76413,
                'education': 'Bachelors',
                'education_num': 9,
                'marital_status': 'Married',
                'occupation': 'Exec-managerial',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'captial_gain': 0,
                'capital_loss': 1000,
                'hours_per_week': 60,
                'native_country': 'United-States',
                },
        },
    ),
):
    # return item
    dictionary = item.dict(by_alias=True)
    dataframe = pd.DataFrame.from_dict(dictionary, 'index').transpose()
    
    print('model_path', model_path)
    X_test, _, _, lb = process_data(
    dataframe, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=None)
    
    pred = inference(model_path, X_test)
    
    if int(pred):
        pred = '>50K'
    else:
        pred = '<=50K'
    
    print(pred)
    
    return {'pred': str(pred)}



    
    