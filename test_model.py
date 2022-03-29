import os
from starter.ml.model import train_model, compute_model_metrics, inference, load_model
from starter.ml.data import process_data, load_data

top_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
model_path = os.path.join(top_path,'model/model.pkl')
data_path = os.path.join(top_path,'data/census_clean.csv')
test_path = os.path.join(top_path,'data/val_census.csv')
train_path = os.path.join(top_path,'data/train_census.csv')

def test_load_data():
    assert os.path.isfile(data_path)
    
    dataframe = load_data(data_path)
    assert dataframe.shape[0] > 0
    assert dataframe.shape[1] > 0
    
    

def test_load_model():
    assert os.path.isfile(model_path)
    
    try:
        model = load_model(model_path)
    except Exception:
        raise ValueError('A very bad thing happened')
        

def test_compute_model_metrics():
    
    cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    train = load_data(train_path)
    test = load_data(test_path)
    
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat, label='salary', training=True, encoder=None, lb=None)
    X_test, y_test, encoder, lb = process_data(test, categorical_features=cat, label='salary', training=False, encoder=encoder, lb=lb)
    
    model = load_model(model_path)
    pred = model.predict(X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, pred)
    
    assert precision > 0.50
    assert fbeta > 0.50
    assert recall > 0.50

    

    
        
        

