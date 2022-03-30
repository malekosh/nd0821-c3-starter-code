import os
import pandas as pd
from ml.model import train_model, compute_model_metrics, inference, load_model, get_metrics_slices_per_category
from ml.data import process_data, load_data


top_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
data_path = os.path.join(top_path,'data')
model_path = os.path.join(top_path,'model/model.pkl')
encoder_path = os.path.join(top_path,'model/encoder.pkl')

census_val_path = os.path.join(data_path, 'val_census.csv')
census_train_path = os.path.join(data_path, 'train_census.csv')
slice_output = os.path.join(top_path,'slice_output.txt')

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

def get_slice_performance(val_path, y, y_pred, cat_list, output_path):
    X = load_data(val_path)
    
    with open(output_path, 'w') as file:
        for cat in cat_list:
            file.write("**************{}**************\n".format(str(cat)))
            slice_eval  = get_metrics_slices_per_category(X, y, y_pred, cat)
            file.write(slice_eval.to_string())
            file.write("\n")

encoder = load_model(encoder_path)
model = load_model(model_path)

dataframe_train = load_data(census_val_path)
dataframe_test = load_data(census_val_path)


_, _, _, lb = process_data(
    dataframe_train, categorical_features=cat_features, label="salary", training=True
)


X_test, y_test, encoder, lb = process_data(
    dataframe_test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

pred = model.predict(X_test)

precision, recall, fbeta = compute_model_metrics(y_test, pred)


get_slice_performance(census_val_path, y_test, pred, cat_features, slice_output)