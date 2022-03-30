# Script to train machine learning model.
import os
import pandas as pd
from ml.model import train_model, compute_model_metrics, inference, save_model
from ml.data import process_data, load_data


top_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
data_path = os.path.join(top_path,'data')
model_path = os.path.join(top_path,'model/model.pkl')
encoder_path = os.path.join(top_path,'model/encoder.pkl')

census_train_path = os.path.join(data_path, 'train_census.csv')
census_test_path = os.path.join(data_path, 'val_census.csv')

train_data = load_data(census_train_path)
test_data = load_data(census_test_path)

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
X_train, y_train, encoder, lb = process_data(
    train_data, categorical_features=cat_features, label="salary", training=True
)

save_model(encoder, encoder_path)

X_test, y_test, encoder, lb = process_data(
    test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

model = train_model(X_train, y_train)
save_model(model, model_path)
