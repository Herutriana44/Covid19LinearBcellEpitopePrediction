import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from hmmlearn import hmm
import pickle
import os

model_dir = 'model/'
os.makedirs(model_dir, exist_ok=True)
os.makedirs('results', exist_ok=True)

def save_pkl(model, name):
    with open(f'{model_dir}{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

def get_data(df, X, y, data_test):
    df = df.dropna(subset=["Position"])
    X_data = df[X]
    y_data = df[y].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=data_test, random_state=42)
    return X_train, X_test, y_train, y_test

def dl(df, X, y, data_test):
    X_train, X_test, y_train, y_test = get_data(df, X, y, data_test)
    model = Sequential([
        Dense(16, activation='relu', input_dim=X_train.shape[1]),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    model.save(model_dir+'deep_learning_model.h5')
    return accuracy, auc

def nn(df, X, y, data_test):
    X_train, X_test, y_train, y_test = get_data(df, X, y, data_test)
    model = MLPClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    save_pkl(model, 'nnMLP')
    accuracy = (y_pred == y_test).mean()
    return accuracy, auc

def rf(df, X, y, data_test):
    X_train, X_test, y_train, y_test = get_data(df, X, y, data_test)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    save_pkl(model, 'RandomForest')
    accuracy = (y_pred == y_test).mean()
    return accuracy, auc

def dt(df, X, y, data_test):
    X_train, X_test, y_train, y_test = get_data(df, X, y, data_test)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    save_pkl(model, 'DecisionTree')
    accuracy = (y_pred == y_test).mean()
    return accuracy, auc

def svm(df, X, y, data_test):
    X_train, X_test, y_train, y_test = get_data(df, X, y, data_test)
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    save_pkl(model, 'SVM')
    accuracy = (y_pred == y_test).mean()
    return accuracy, auc

def hmm_model(df, X, y, data_test):
    X_train, X_test, y_train, y_test = get_data(df, X, y, data_test)
    model = hmm.GaussianHMM(n_components=2)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    save_pkl(model, 'hmmmodel')
    accuracy = (y_pred == y_test).mean()
    return accuracy, auc
