from sklearn.metrics import fbeta_score, precision_score, recall_score
import pickle
import os

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    random_Forest = RandomForestClassifier(n_estimators=100,  max_depth=15)
    random_Forest.fit(X_train, y_train)
    return random_Forest


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : path to pickle model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    random_forest = load_model(model)
    preds = random_forest.predict(X)
    return preds


def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
        

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def get_TP_TN_FP_FN(X, cat, sub_cat):
    TP = X.loc[(X['y']==1) & (X['pred']==1) & (X[cat]==sub_cat)].shape[0]
    TN = X.loc[(X['y']==0) & (X['pred']==0) & (X[cat]==sub_cat)].shape[0]
    FN = X.loc[(X['y']==1) & (X['pred']==0) & (X[cat]==sub_cat)].shape[0]
    FP = X.loc[(X['y']==0) & (X['pred']==1) & (X[cat]==sub_cat)].shape[0]
    
    return TP, TN, FN, FP

def get_eval_metrics(TP, TN, FN, FP):
    if not TP:
        TP = 1e-7
    recall = TP / (TP+FN)
    prec = TP / (TP+FP)
    fscore = (2*prec*recall) / (prec+recall)
    return recall, prec, fscore

def create_pd_dict(sub_cat, recall, prec, fscore):
    return {'sub-category': sub_cat, 'recall': round(recall,2), 'precision': round(prec,2), 'fscore': round(fscore,2)}
    

def get_metrics_slices_per_category(X, y, y_pred, cat, value=None):
    X = X.drop(['salary'], axis=1)
    X['y'] =  y.tolist()
    X['pred'] = y_pred.tolist()
    
    if value is not None:
        TP, TN, FN, FP = get_TP_TN_FP_FN(X, cat, value)
        recall, prec, fscore = get_eval_metrics(TP, TN, FN, FP)
        row_list = [create_pd_dict(s, recall, prec, fscore)]
    else:
        sub_cat = X[cat].unique().tolist()

        row_list = []
        for s in sub_cat:
            TP, TN, FN, FP = get_TP_TN_FP_FN(X, cat, s)

            recall, prec, fscore = get_eval_metrics(TP, TN, FN, FP)
            row_dict = create_pd_dict(s, recall, prec, fscore)
            row_list.append(row_dict)

    sub_cat_dataframe = pd.DataFrame.from_dict(row_list, orient='columns')
    
    return sub_cat_dataframe